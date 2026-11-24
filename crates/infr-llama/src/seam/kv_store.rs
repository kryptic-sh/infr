//! Cross-conversation KV store, keyed in fixed-size token blocks.
//!
//! A prompt's K/V rows never change once written, so the rows one request materialized can seed
//! any later request that starts with the same tokens — a new conversation, an evicted slot, or a
//! different chat on the same model. The slot pool already reuses KV between its own slots
//! (`ParallelSeam::checkout` seeds from a free slot); this store is the same idea with the rows in
//! host RAM, so the reuse survives eviction and outlives the conversation that produced it.
//!
//! Rows are stored per 128-token block, chained: a block's id hashes its predecessor's id with its
//! own tokens, so a block only matches at the position it was written at (rope makes rows
//! position-bound) and ten conversations sharing a system prompt keep ONE copy of it.
//!
//! The payoff is the prefill it skips: 8k of KV uploads over PCIe in about 0.12 s, while
//! re-prefilling the same tokens costs tens of seconds.

use crate::Config;
use anyhow::Result;
use infr_core::backend::Backend;
use infr_core::DType;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use super::{
    kv_tier::{DeviceKv, KvRows, KvStore},
    weights::{RecurrentSnap, SeamKv},
};

/// Tokens per stored block. Smaller blocks share more (a 600-token system prompt with two
/// different tails stores each half once) at the price of one download per new block.
pub(crate) const BLOCK_TOKENS: usize = 128;

/// Below this many shared tokens the upload costs more than re-prefilling them.
pub(crate) const MIN_PREFIX_TOKENS: usize = 256;

/// Chain id of a block: the previous block's id mixed with this block's tokens.
///
/// FNV-1a, not `DefaultHasher`: the ids are written to a cold tier, so they must not change
/// with the toolchain. Every lookup also compares the token ids, so a collision costs a miss, and
/// a capture under a colliding id replaces the older block.
fn block_id(prev: u64, tokens: &[u32]) -> u64 {
    let mut h = Fnv::new();
    h.bytes(&prev.to_le_bytes());
    for t in tokens {
        h.bytes(&t.to_le_bytes());
    }
    h.0
}

const STORE_MAGIC: &[u8; 8] = b"INFRKV02";
const STORE_VERSION: u32 = 2;

/// Identity of the model a cold tier's blocks belong to: the geometry that shapes every row, plus
/// the context and vocabulary. Blocks under another key are never loaded.
pub(crate) fn store_key(cfg: &Config) -> u64 {
    let mut h = Fnv::new();
    for v in [
        cfg.n_layer,
        cfg.n_embd,
        cfg.n_head,
        cfg.n_kv,
        cfg.head_dim,
        cfg.rope_dim,
        cfg.n_ctx_train,
        cfg.vocab,
        cfg.n_ff,
    ] {
        h.bytes(&(v as u64).to_le_bytes());
    }
    h.0
}

/// FNV-1a over the bytes the store writes, so a truncated or damaged file is discarded at load.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn put_u32(w: &mut impl Write, h: &mut Fnv, v: u32) -> std::io::Result<()> {
    let b = v.to_le_bytes();
    w.write_all(&b)?;
    h.bytes(&b);
    Ok(())
}

fn put_u64(w: &mut impl Write, h: &mut Fnv, v: u64) -> std::io::Result<()> {
    let b = v.to_le_bytes();
    w.write_all(&b)?;
    h.bytes(&b);
    Ok(())
}

fn put_bytes(w: &mut impl Write, h: &mut Fnv, bytes: &[u8]) -> std::io::Result<()> {
    put_u32(w, h, bytes.len() as u32)?;
    w.write_all(bytes)?;
    h.bytes(bytes);
    Ok(())
}

fn get_u32(r: &mut impl Read, h: &mut Fnv) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    h.bytes(&b);
    Ok(u32::from_le_bytes(b))
}

fn get_u64(r: &mut impl Read, h: &mut Fnv) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    h.bytes(&b);
    Ok(u64::from_le_bytes(b))
}

fn get_bytes(r: &mut impl Read, h: &mut Fnv) -> std::io::Result<Vec<u8>> {
    let n = get_u32(r, h)? as usize;
    if n > (1 << 31) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "kv-prefix store claims a huge buffer",
        ));
    }
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    h.bytes(&b);
    Ok(b)
}

fn dtype_name(d: DType) -> Option<&'static str> {
    Some(match d {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::Bf16 => "bf16",
        DType::Q8_0 => "q8_0",
        DType::Q4_0 => "q4_0",
        DType::Q4_1 => "q4_1",
        DType::Q5_0 => "q5_0",
        DType::Q5_1 => "q5_1",
        DType::Iq4Nl => "iq4_nl",
        DType::Turbo2 => "turbo2",
        DType::Turbo3 => "turbo3",
        DType::Turbo4 => "turbo4",
        _ => return None,
    })
}

fn put_name(w: &mut impl Write, h: &mut Fnv, d: DType) -> std::io::Result<()> {
    let name = dtype_name(d).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "kv-prefix store cannot name this KV dtype",
        )
    })?;
    put_bytes(w, h, name.as_bytes())
}

fn get_name(r: &mut impl Read, h: &mut Fnv) -> Result<DType> {
    let bytes = get_bytes(r, h)?;
    let name = std::str::from_utf8(&bytes)?;
    infr_core::budget::parse_kv_dtype(name)
        .ok_or_else(|| anyhow::anyhow!("kv-prefix store names an unknown KV dtype {name:?}"))
}

fn write_header(w: &mut impl Write, key: u64, k_fmt: DType, v_fmt: DType) -> std::io::Result<()> {
    w.write_all(STORE_MAGIC)?;
    let mut h = Fnv::new();
    put_u32(w, &mut h, STORE_VERSION)?;
    put_u64(w, &mut h, key)?;
    put_name(w, &mut h, k_fmt)?;
    put_name(w, &mut h, v_fmt)
}

fn read_header(r: &mut impl Read) -> Result<(u64, DType, DType)> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != STORE_MAGIC {
        anyhow::bail!("not a kv-prefix store file");
    }
    let mut h = Fnv::new();
    if get_u32(r, &mut h)? != STORE_VERSION {
        anyhow::bail!("kv-prefix store version mismatch");
    }
    let key = get_u64(r, &mut h)?;
    let k_fmt = get_name(r, &mut h)?;
    let v_fmt = get_name(r, &mut h)?;
    Ok((key, k_fmt, v_fmt))
}

fn write_block(w: &mut impl Write, h: &mut Fnv, b: &BlockRef<'_>) -> std::io::Result<()> {
    put_u64(w, h, b.id)?;
    put_u32(w, h, b.tokens.len() as u32)?;
    for &t in b.tokens {
        put_u32(w, h, t)?;
    }
    for side in 0..2 {
        put_u32(w, h, b.rows.layers.len() as u32)?;
        for l in 0..b.rows.layers.len() {
            // A layer holding only some of the block's rows is stored as absent.
            let bytes = b
                .rows
                .whole_layer(l)
                .map_or(&[][..], |(k, v)| if side == 0 { k } else { v });
            put_bytes(w, h, bytes)?;
        }
    }
    match b.snap {
        None => w.write_all(&[0])?,
        Some(s) => {
            w.write_all(&[1])?;
            put_u32(w, h, s.pos as u32)?;
            for side in [&s.k, &s.v] {
                put_u32(w, h, side.len() as u32)?;
                for (l, bytes) in side {
                    put_u32(w, h, *l as u32)?;
                    put_bytes(w, h, bytes)?;
                }
            }
        }
    }
    put_u64(w, h, b.used)
}

fn read_block(r: &mut impl Read, h: &mut Fnv) -> Result<Block> {
    let id = get_u64(r, h)?;
    let len = get_u32(r, h)? as usize;
    if len > BLOCK_TOKENS {
        anyhow::bail!("kv-prefix block claims {len} tokens");
    }
    let mut tokens = Vec::with_capacity(len);
    for _ in 0..len {
        tokens.push(get_u32(r, h)?);
    }
    let mut sides = Vec::new();
    for _ in 0..2 {
        let layers = get_u32(r, h)? as usize;
        if layers > 4096 {
            anyhow::bail!("kv-prefix block claims {layers} layers");
        }
        let mut side = Vec::with_capacity(layers);
        for _ in 0..layers {
            side.push(get_bytes(r, h)?);
        }
        sides.push(side);
    }
    let mut present = [0u8; 1];
    r.read_exact(&mut present)?;
    let snap = if present[0] == 1 {
        let pos = get_u32(r, h)? as usize;
        let mut snap_sides = Vec::new();
        let mut bytes = 0usize;
        for _ in 0..2 {
            let entries = get_u32(r, h)? as usize;
            if entries > 4096 {
                anyhow::bail!("kv-prefix snapshot claims {entries} layers");
            }
            let mut side = Vec::with_capacity(entries);
            for _ in 0..entries {
                let l = get_u32(r, h)? as usize;
                let b = get_bytes(r, h)?;
                bytes += b.len();
                side.push((l, b));
            }
            snap_sides.push(side);
        }
        let v = snap_sides.pop().expect("two sides");
        let k = snap_sides.pop().expect("two sides");
        Some(RecurrentSnap { pos, k, v, bytes })
    } else {
        None
    };
    let used = get_u64(r, h)?;
    let v = sides.pop().expect("two sides");
    let k = sides.pop().expect("two sides");
    Ok(Block {
        id,
        rows: KvRows::whole(len, k, v),
        tokens,
        snap,
        used,
    })
}

/// A block as a cold tier hands it back: its token ids, rows and snapshot.
pub(crate) struct Block {
    id: u64,
    tokens: Vec<u32>,
    rows: KvRows,
    snap: Option<RecurrentSnap>,
    used: u64,
}

impl Block {
    #[cfg(test)]
    fn as_ref(&self) -> BlockRef<'_> {
        BlockRef {
            id: self.id,
            tokens: &self.tokens,
            rows: &self.rows,
            snap: self.snap.as_ref(),
            used: self.used,
        }
    }
}

/// A block as a cold tier writes it, borrowed from the cache's index and store.
pub(crate) struct BlockRef<'a> {
    id: u64,
    tokens: &'a [u32],
    rows: &'a KvRows,
    snap: Option<&'a RecurrentSnap>,
    used: u64,
}

/// The index entry of one stored block: `tokens.len()` token ids whose rows are extent `id`.
///
/// The length is `BLOCK_TOKENS` except for the tail of a captured prefix, which may be short; a
/// short block only ever matches at the end of a prompt.
struct Entry {
    tokens: Vec<u32>,
    /// Rows plus snapshot, what the budget counts.
    bytes: usize,
    used: u64,
    /// A recurrent-state snapshot at the block's end sits in extent [`snap_id`].
    snap: bool,
    /// Changed since the last write to the cold tier.
    dirty: bool,
}

/// The extent holding the snapshot at the end of block `id`.
fn snap_id(id: u64) -> u64 {
    !id
}

/// The prefix cache: a token index over content-addressed block extents in `S`.
///
/// Rows are kept per 128-token block, chained: a block's id hashes its predecessor's id with its
/// own tokens, so a block only matches at the position it was written at (rope makes rows
/// position-bound) and ten conversations sharing a system prompt keep one copy of it. A block's
/// rows are extent `id` (rows `0..len`); a recurrent snapshot at its end is extent `snap_id(id)`.
/// The cold tier reads extents through [`KvStore::host`].
pub(crate) struct BlockCache<S: KvStore> {
    store: S,
    index: HashMap<u64, Entry>,
    cap_bytes: usize,
    /// Monotonic access counter (LRU order, not wall time — the cache never sleeps).
    clock: u64,
    /// The KV layouts the blocks were captured with; a dtype change invalidates every block.
    k_fmt: DType,
    v_fmt: DType,
    pub(crate) restores: u64,
    pub(crate) restored_tokens: u64,
    pub(crate) captures: u64,
    /// Where blocks outlive this process, when one is configured.
    tier: Option<Box<dyn KvColdTier>>,
    /// Model identity the tier's blocks are keyed under.
    key: u64,
    /// Blocks changed since the last write to the tier.
    dirty: bool,
    /// The KV formats the tier last saw; a change forces a rewrite.
    tier_fmt: Option<(DType, DType)>,
}

impl<S: KvStore> Drop for BlockCache<S> {
    fn drop(&mut self) {
        if self.dirty {
            if let Err(e) = self.rewrite_tier() {
                tracing::warn!(error = %e, "kv store: final save to the cold tier failed");
            }
        }
    }
}

/// What a cold tier hands back at startup: the formats its blocks were captured with, and the
/// blocks in the order they were written (a later copy of a block replaces an earlier one).
pub(crate) struct TierContents {
    k_fmt: DType,
    v_fmt: DType,
    blocks: Vec<Block>,
}

/// Where the host store's blocks live beyond this process: a local file today, and a shared
/// key-value cache such as Redis or Valkey later. The store keeps its RAM index and LRU in front;
/// a tier only persists blocks and hands them back at startup.
pub(crate) trait KvColdTier: Send {
    /// A name for logs: the path or the endpoint.
    fn describe(&self) -> String;
    /// The blocks stored under `key`, or `None` when the tier holds nothing for it.
    fn load(&mut self, key: u64) -> Result<Option<TierContents>>;
    /// Add `blocks` captured with `k_fmt`/`v_fmt`, which match what the tier already holds.
    fn append(
        &mut self,
        key: u64,
        k_fmt: DType,
        v_fmt: DType,
        blocks: &[BlockRef<'_>],
    ) -> Result<()>;
    /// Replace everything stored under `key` with `blocks`.
    fn rewrite(
        &mut self,
        key: u64,
        k_fmt: DType,
        v_fmt: DType,
        blocks: &[BlockRef<'_>],
    ) -> Result<()>;
}

/// A [`KvColdTier`] in one local file: a header, then checksummed records appended as blocks are
/// captured. A crash leaves the file valid up to its last whole record.
pub(crate) struct FileTier {
    path: PathBuf,
}

impl FileTier {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn write_record(w: &mut impl Write, blocks: &[BlockRef<'_>]) -> std::io::Result<()> {
        let mut h = Fnv::new();
        put_u32(w, &mut h, blocks.len().min(u32::MAX as usize) as u32)?;
        for b in blocks {
            write_block(w, &mut h, b)?;
        }
        w.write_all(&h.0.to_le_bytes())
    }

    /// One appended record, or `None` at a clean end of file or a torn tail.
    fn read_record(r: &mut impl Read) -> Option<Vec<Block>> {
        let mut h = Fnv::new();
        let count = get_u32(r, &mut h).ok()? as usize;
        if count > (1 << 20) {
            return None;
        }
        let mut record = Vec::with_capacity(count);
        for _ in 0..count {
            record.push(read_block(r, &mut h).ok()?);
        }
        let mut stored = [0u8; 8];
        r.read_exact(&mut stored).ok()?;
        (u64::from_le_bytes(stored) == h.0).then_some(record)
    }
}

impl KvColdTier for FileTier {
    fn describe(&self) -> String {
        self.path.display().to_string()
    }

    fn load(&mut self, key: u64) -> Result<Option<TierContents>> {
        let file = match std::fs::File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let len = file.metadata()?.len();
        let mut r = BufReader::new(file);
        let (file_key, k_fmt, v_fmt) = read_header(&mut r)?;
        if file_key != key {
            anyhow::bail!("the file holds another model's prefixes");
        }
        let mut blocks: Vec<Block> = Vec::new();
        let mut read = 0u64;
        loop {
            if std::io::BufRead::fill_buf(&mut r)?.is_empty() {
                break;
            }
            let Some(record) = Self::read_record(&mut r) else {
                tracing::warn!(
                    path = %self.path.display(),
                    file_bytes = len,
                    "kv store: the journal tail is torn; keeping the whole records before it"
                );
                break;
            };
            read += 1;
            for b in record {
                match blocks
                    .iter_mut()
                    .find(|old| old.id == b.id && old.tokens == b.tokens)
                {
                    Some(old) => *old = b,
                    None => blocks.push(b),
                }
            }
        }
        tracing::debug!(
            records = read,
            blocks = blocks.len(),
            "kv store: file tier read"
        );
        Ok(Some(TierContents {
            k_fmt,
            v_fmt,
            blocks,
        }))
    }

    fn append(
        &mut self,
        key: u64,
        k_fmt: DType,
        v_fmt: DType,
        blocks: &[BlockRef<'_>],
    ) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        if f.metadata()?.len() == 0 {
            write_header(&mut f, key, k_fmt, v_fmt)?;
        }
        let mut w = BufWriter::new(&mut f);
        Self::write_record(&mut w, blocks)?;
        w.flush()?;
        drop(w);
        f.sync_data()?;
        Ok(())
    }

    fn rewrite(
        &mut self,
        key: u64,
        k_fmt: DType,
        v_fmt: DType,
        blocks: &[BlockRef<'_>],
    ) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self
            .path
            .with_extension(format!("tmp.{}", std::process::id()));
        let mut w = BufWriter::new(std::fs::File::create(&tmp)?);
        write_header(&mut w, key, k_fmt, v_fmt)?;
        Self::write_record(&mut w, blocks)?;
        w.flush()?;
        let f = w.into_inner().map_err(|e| e.into_error())?;
        f.sync_all()?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

/// A [`KvColdTier`] in Redis or Valkey: one hash-free key per block under a model prefix, an
/// index set of the live block ids, and a meta key naming the KV formats. A shared server then
/// serves every box pointed at it, and its own `maxmemory` policy trims what nobody asked for.
pub(crate) struct RedisTier {
    conn: redis::Connection,
    endpoint: String,
}

impl RedisTier {
    fn connect(url: &str) -> Result<Self> {
        let client = redis::Client::open(url)?;
        let conn = client.get_connection()?;
        conn.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
        conn.set_write_timeout(Some(std::time::Duration::from_secs(30)))?;
        Ok(Self {
            conn,
            endpoint: redact(url),
        })
    }

    fn prefix(key: u64) -> String {
        format!("infr:kv:{key:016x}")
    }

    fn key_block(key: u64, id: u64) -> String {
        format!("{}:b{id:016x}", Self::prefix(key))
    }

    fn key_index(key: u64) -> String {
        format!("{}:idx", Self::prefix(key))
    }

    fn key_meta(key: u64) -> String {
        format!("{}:meta", Self::prefix(key))
    }

    fn encode(b: &BlockRef<'_>) -> Vec<u8> {
        let mut v = Vec::with_capacity(b.rows.bytes() + 16);
        let mut h = Fnv::new();
        write_block(&mut v, &mut h, b).expect("writing to a Vec cannot fail");
        v.extend_from_slice(&h.0.to_le_bytes());
        v
    }

    fn decode(v: &[u8]) -> Result<Block> {
        if v.len() < 8 {
            anyhow::bail!("kv store: a Redis block is too short");
        }
        let (payload, hash) = v.split_at(v.len() - 8);
        let stored = u64::from_le_bytes(hash.try_into().expect("eight bytes"));
        let mut r = payload;
        let mut h = Fnv::new();
        let b = read_block(&mut r, &mut h)?;
        if !r.is_empty() || h.0 != stored {
            anyhow::bail!("kv store: a Redis block failed its checksum");
        }
        Ok(b)
    }

    fn meta(k_fmt: DType, v_fmt: DType) -> String {
        format!(
            "{} {}",
            dtype_name(k_fmt).unwrap_or("f16"),
            dtype_name(v_fmt).unwrap_or("f16")
        )
    }
}

impl KvColdTier for RedisTier {
    fn describe(&self) -> String {
        self.endpoint.clone()
    }

    fn load(&mut self, key: u64) -> Result<Option<TierContents>> {
        let Some(meta): Option<String> = redis::cmd("GET")
            .arg(Self::key_meta(key))
            .query(&mut self.conn)?
        else {
            return Ok(None);
        };
        let mut parts = meta.split_whitespace();
        let k_fmt = parts
            .next()
            .and_then(infr_core::budget::parse_kv_dtype)
            .ok_or_else(|| anyhow::anyhow!("kv store: Redis meta names no K format"))?;
        let v_fmt = parts
            .next()
            .and_then(infr_core::budget::parse_kv_dtype)
            .ok_or_else(|| anyhow::anyhow!("kv store: Redis meta names no V format"))?;
        let ids: Vec<String> = redis::cmd("SMEMBERS")
            .arg(Self::key_index(key))
            .query(&mut self.conn)?;
        let mut blocks: Vec<Block> = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(64) {
            let keys: Vec<String> = chunk
                .iter()
                .filter_map(|id| u64::from_str_radix(id, 16).ok())
                .map(|id| Self::key_block(key, id))
                .collect();
            if keys.is_empty() {
                continue;
            }
            let vals: Vec<Option<Vec<u8>>> = redis::cmd("MGET").arg(&keys).query(&mut self.conn)?;
            for v in vals.into_iter().flatten() {
                match Self::decode(&v) {
                    Ok(b) => match blocks
                        .iter_mut()
                        .find(|old| old.id == b.id && old.tokens == b.tokens)
                    {
                        Some(old) => *old = b,
                        None => blocks.push(b),
                    },
                    Err(e) => {
                        tracing::warn!(error = %e, "kv store: dropped an unreadable Redis block")
                    }
                }
            }
        }
        Ok(Some(TierContents {
            k_fmt,
            v_fmt,
            blocks,
        }))
    }

    fn append(
        &mut self,
        key: u64,
        k_fmt: DType,
        v_fmt: DType,
        blocks: &[BlockRef<'_>],
    ) -> Result<()> {
        if blocks.is_empty() {
            return Ok(());
        }
        let mut pipe = redis::pipe();
        pipe.cmd("SET")
            .arg(Self::key_meta(key))
            .arg(Self::meta(k_fmt, v_fmt))
            .ignore();
        let mut ids = Vec::with_capacity(blocks.len());
        for b in blocks {
            ids.push(format!("{:016x}", b.id));
            pipe.cmd("SET")
                .arg(Self::key_block(key, b.id))
                .arg(Self::encode(b))
                .ignore();
        }
        pipe.cmd("SADD").arg(Self::key_index(key)).arg(ids).ignore();
        let _: () = pipe.query(&mut self.conn)?;
        Ok(())
    }

    fn rewrite(
        &mut self,
        key: u64,
        k_fmt: DType,
        v_fmt: DType,
        blocks: &[BlockRef<'_>],
    ) -> Result<()> {
        let old: Vec<String> = redis::cmd("SMEMBERS")
            .arg(Self::key_index(key))
            .query(&mut self.conn)?;
        if !old.is_empty() {
            let keys: Vec<String> = old
                .iter()
                .filter_map(|id| u64::from_str_radix(id, 16).ok())
                .map(|id| Self::key_block(key, id))
                .collect();
            let mut pipe = redis::pipe();
            pipe.cmd("DEL").arg(keys).ignore();
            pipe.cmd("DEL")
                .arg(Self::key_index(key))
                .arg(Self::key_meta(key))
                .ignore();
            let _: () = pipe.query(&mut self.conn)?;
        } else {
            let mut pipe = redis::pipe();
            pipe.cmd("DEL")
                .arg(Self::key_index(key))
                .arg(Self::key_meta(key))
                .ignore();
            let _: () = pipe.query(&mut self.conn)?;
        }
        if blocks.is_empty() {
            return Ok(());
        }
        self.append(key, k_fmt, v_fmt, blocks)
    }
}

/// Strip credentials from a URL before it reaches a log line.
fn redact(url: &str) -> String {
    match url.split_once("://") {
        Some((scheme, rest)) => match rest.rsplit_once('@') {
            Some((_, host)) => format!("{scheme}://{host}"),
            None => url.to_string(),
        },
        None => url.to_string(),
    }
}

/// The cold tier `kv.store_url` / `kv.store_file` select. A URL wins; an unreachable URL falls
/// back to the file when one is set, and a missing/broken tier only costs persistence.
pub(crate) fn cold_tier_for(kv: &crate::EngineConfig) -> Option<Box<dyn KvColdTier>> {
    if let Some(url) = kv
        .kv
        .store_url
        .as_deref()
        .map(str::trim)
        .filter(|u| !u.is_empty())
    {
        match RedisTier::connect(url) {
            Ok(tier) => {
                tracing::info!(endpoint = %tier.endpoint, "kv store: sharing the cold tier over Redis");
                return Some(Box::new(tier));
            }
            Err(e) => tracing::warn!(
                endpoint = %redact(url),
                error = %e,
                "kv store: cannot reach the Redis cold tier; falling back to the file"
            ),
        }
    }
    kv.kv
        .store_file
        .as_ref()
        .map(|p| Box::new(FileTier::new(p.clone())) as Box<dyn KvColdTier>)
}

/// Borrow block `id` for a tier write; `None` when the store keeps it outside host memory.
fn block_ref<'a, S: KvStore>(store: &'a S, id: u64, e: &'a Entry) -> Option<BlockRef<'a>> {
    let rows = store.host(id)?;
    let snap = if e.snap {
        store.host(snap_id(id)).and_then(|r| r.snap.as_deref())
    } else {
        None
    };
    Some(BlockRef {
        id,
        tokens: &e.tokens,
        rows,
        snap,
        used: e.used,
    })
}

/// The device slot's extent id inside [`DeviceKv`] while the cache reads or writes it.
const SLOT: u64 = 0;

impl<S: KvStore> BlockCache<S> {
    /// Build a cache with `cap_bytes` of budget over `store`. A zero cap is a disabled cache:
    /// every call returns immediately, so the engine pays nothing for a feature nobody asked for.
    #[cfg(test)]
    pub(crate) fn new(store: S, cap_bytes: usize) -> Self {
        Self::with_tier(store, cap_bytes, None, 0)
    }

    /// [`new`](Self::new), backed by a cold `tier`: the cache loads what the tier holds for `key`
    /// now, writes each capture through to it, and compacts it on drop. `key` identifies the
    /// model geometry the blocks belong to.
    pub(crate) fn with_tier(
        store: S,
        cap_bytes: usize,
        tier: Option<Box<dyn KvColdTier>>,
        key: u64,
    ) -> Self {
        let mut cache = Self {
            store,
            index: HashMap::new(),
            cap_bytes,
            clock: 0,
            k_fmt: DType::F16,
            v_fmt: DType::F16,
            restores: 0,
            restored_tokens: 0,
            captures: 0,
            tier: tier.filter(|_| cap_bytes > 0),
            key,
            dirty: false,
            tier_fmt: None,
        };
        cache.load_tier();
        cache
    }

    /// Fill the index and store from the tier. A tier that fails or holds another model's blocks
    /// leaves the cache empty; it never fails startup.
    fn load_tier(&mut self) {
        let Some(tier) = self.tier.as_mut() else {
            return;
        };
        let name = tier.describe();
        let loaded = tier.load(self.key);
        let loaded = loaded.and_then(|c| {
            let Some(c) = c else { return Ok(false) };
            self.k_fmt = c.k_fmt;
            self.v_fmt = c.v_fmt;
            self.tier_fmt = Some((c.k_fmt, c.v_fmt));
            self.clock = c.blocks.iter().map(|b| b.used).max().unwrap_or(0);
            for b in c.blocks {
                self.insert(b.id, b.tokens, b.rows, b.used, false)?;
                if let Some(snap) = b.snap {
                    self.attach(b.id, Arc::new(snap), false)?;
                }
            }
            Ok(true)
        });
        match loaded {
            Ok(true) => {
                self.trim();
                tracing::info!(
                    tier = %name,
                    blocks = self.index.len(),
                    "kv store: loaded the saved prefixes"
                );
            }
            Ok(false) => {}
            Err(e) => {
                self.clear();
                tracing::warn!(
                    tier = %name,
                    error = %e,
                    "kv store: the saved prefixes are unusable; starting empty"
                );
            }
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.cap_bytes > 0
    }

    /// `(blocks, bytes)` currently resident, for the capture log.
    pub(crate) fn stats(&self) -> (usize, usize) {
        (self.index.len(), self.index.values().map(|e| e.bytes).sum())
    }

    /// Store block `id`'s rows and index it, replacing any block under the same id.
    fn insert(
        &mut self,
        id: u64,
        tokens: Vec<u32>,
        rows: KvRows,
        used: u64,
        dirty: bool,
    ) -> Result<()> {
        self.drop_entry(id);
        let bytes = rows.bytes();
        self.store.put(id, 0, rows)?;
        let entry = Entry {
            tokens,
            bytes,
            used,
            snap: false,
            dirty,
        };
        self.index.insert(id, entry);
        Ok(())
    }

    /// Store `snap` as the state at the end of indexed block `id`.
    fn attach(&mut self, id: u64, snap: Arc<RecurrentSnap>, dirty: bool) -> Result<()> {
        let bytes = snap.bytes;
        let rows = KvRows {
            snap: Some(snap),
            ..KvRows::default()
        };
        self.store.put(snap_id(id), 0, rows)?;
        let e = self.index.get_mut(&id).expect("attach to an indexed block");
        e.snap = true;
        e.bytes += bytes;
        e.dirty |= dirty;
        Ok(())
    }

    fn drop_entry(&mut self, id: u64) {
        let Some(e) = self.index.remove(&id) else {
            return;
        };
        let mut removed = self.store.remove(id);
        if e.snap {
            removed = removed.and(self.store.remove(snap_id(id)));
        }
        if let Err(e) = removed {
            tracing::warn!(error = %e, "kv store: dropping a block's extents failed");
        }
    }

    fn clear(&mut self) {
        let ids: Vec<u64> = self.index.keys().copied().collect();
        for id in ids {
            self.drop_entry(id);
        }
    }

    /// Replace the tier's contents with the cache's current blocks, which drops what the LRU
    /// evicted and any superseded copies.
    fn rewrite_tier(&mut self) -> Result<()> {
        let Some(tier) = self.tier.as_mut() else {
            self.dirty = false;
            return Ok(());
        };
        let all: Vec<BlockRef<'_>> = self
            .index
            .iter()
            .filter_map(|(&id, e)| block_ref(&self.store, id, e))
            .collect();
        tier.rewrite(self.key, self.k_fmt, self.v_fmt, &all)?;
        for e in self.index.values_mut() {
            e.dirty = false;
        }
        self.dirty = false;
        self.tier_fmt = Some((self.k_fmt, self.v_fmt));
        Ok(())
    }

    /// Write the blocks changed since the last write through to the tier, so a crash or a signal
    /// exit still leaves the next start with them.
    fn append_dirty(&mut self) -> Result<()> {
        if self.tier.is_none() {
            return Ok(());
        }
        if self.tier_fmt != Some((self.k_fmt, self.v_fmt)) {
            return self.rewrite_tier();
        }
        let batch: Vec<BlockRef<'_>> = self
            .index
            .iter()
            .filter(|(_, e)| e.dirty)
            .filter_map(|(&id, e)| block_ref(&self.store, id, e))
            .collect();
        if batch.is_empty() {
            return Ok(());
        }
        let tier = self.tier.as_mut().expect("checked above");
        tier.append(self.key, self.k_fmt, self.v_fmt, &batch)?;
        for e in self.index.values_mut() {
            e.dirty = false;
        }
        self.dirty = false;
        Ok(())
    }

    fn find(&self, id: u64, tokens: &[u32]) -> bool {
        self.index.get(&id).is_some_and(|e| e.tokens == tokens)
    }

    /// Walk `prompt` block by block; return the tokens covered by the leading stored run and the
    /// ids of the blocks that cover them, with their use stamped.
    fn matched(&mut self, prompt: &[u32]) -> (usize, Vec<u64>) {
        let mut id = 0u64;
        let mut n = 0usize;
        let mut hits = Vec::new();
        while n < prompt.len() {
            let len = BLOCK_TOKENS.min(prompt.len() - n);
            let block = &prompt[n..n + len];
            id = block_id(id, block);
            let Some(e) = self.index.get_mut(&id).filter(|e| e.tokens == block) else {
                break;
            };
            self.clock += 1;
            e.used = self.clock;
            hits.push(id);
            n += len;
            if len < BLOCK_TOKENS {
                break;
            }
        }
        (n, hits)
    }

    /// Copy the longest stored run of blocks into `kv` (a device upload per layer per run of
    /// present rows) and mark those tokens materialized. Returns the number of tokens restored.
    ///
    /// A sliding-window ring is written at each block's own ring position (`start % rows`), so a
    /// prefix longer than the ring still seeds it: the newest blocks overwrite the oldest, which
    /// is exactly the window attention will read. Only a destination whose own cache already
    /// wrapped is refused — its rows no longer describe its `cached`.
    pub(crate) fn restore(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        kv: &mut SeamKv,
        prompt: &[u32],
    ) -> Result<usize> {
        if !self.enabled() {
            return Ok(0);
        }
        let (mut p, hits) = self.matched(prompt);
        if p < MIN_PREFIX_TOKENS || self.k_fmt != kv.k_fmt || self.v_fmt != kv.v_fmt {
            return Ok(0);
        }
        p = p.min(kv.max_ctx);
        if kv
            .window_ring_rows(be, cfg)
            .is_some_and(|rows| kv.cached.len() > rows)
        {
            return Ok(0);
        }
        let len = |id: &u64| self.index[id].tokens.len();
        let mut keep = Vec::new();
        let mut n = 0usize;
        for id in hits {
            if n + len(&id) > p {
                break;
            }
            n += len(&id);
            keep.push(id);
        }
        // A recurrent model's rows are useless without the state they were written beside: only a
        // block carrying a snapshot can seed it, and everything after that block must be prefilled.
        if cfg.has_recurrent_state() {
            let Some(at) = keep.iter().rposition(|id| self.index[id].snap) else {
                return Ok(0);
            };
            keep.truncate(at + 1);
            n = keep.iter().map(len).sum();
        }
        if n < MIN_PREFIX_TOKENS {
            return Ok(0);
        }
        let mut rows = KvRows::default();
        for id in &keep {
            let len = self.index[id].tokens.len();
            anyhow::ensure!(
                self.store.get(*id, 0..len, &mut rows)?,
                "kv store: block {id:x} is indexed but not stored"
            );
        }
        if cfg.has_recurrent_state() {
            let last = snap_id(*keep.last().expect("the snapshot block"));
            anyhow::ensure!(
                self.store.get(last, 0..0, &mut rows)?,
                "kv store: a snapshot is indexed but not stored"
            );
        }
        if let Some(p) = kv.paged.as_mut() {
            p.make_private(be, 0, n)?;
        }
        // Every row attention will read must be there: a window layer keeps only the newest rows
        // it has, and a restore that cannot fill that window must re-prefill instead.
        for l in 0..cfg.n_layer {
            if cfg.is_recurrent_layer(l) {
                continue;
            }
            let need = kv.row_layout(be, cfg, l).oldest_resident(n)..n;
            if !need.is_empty() && !rows.layers.get(l).is_some_and(|r| r.covers(need.clone())) {
                tracing::debug!(layer = l, need = ?need, "kv store: the stored blocks miss the window");
                return Ok(0);
            }
        }
        DeviceKv::new(SLOT, be, cfg, kv).put(SLOT, 0, rows)?;
        kv.written_end = n;
        kv.cached = prompt[..n].to_vec();
        self.restores += 1;
        self.restored_tokens += n as u64;
        Ok(n)
    }

    /// Capture `kv`'s first `upto` rows as reusable blocks. Returns the token count kept.
    ///
    /// Blocks already stored are skipped, so a conversation that extends its own prefix keeps only
    /// the new suffix. A recurrent layer has no per-position rows: its state rides along as a
    /// snapshot on the block that ends where the state was captured, and a restore needs one.
    pub(crate) fn capture(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        kv: &SeamKv,
        upto: usize,
    ) -> Result<usize> {
        if !self.enabled() {
            return Ok(0);
        }
        let p = upto.min(kv.cached.len()).min(kv.max_ctx);
        tracing::debug!(
            upto,
            cached = kv.cached.len(),
            p,
            snaps = kv.recurrent_snaps.len(),
            "kv store: capture plan"
        );
        if p < MIN_PREFIX_TOKENS {
            return Ok(0);
        }
        if self.k_fmt != kv.k_fmt || self.v_fmt != kv.v_fmt {
            self.clear();
            self.k_fmt = kv.k_fmt;
            self.v_fmt = kv.v_fmt;
        }
        let tokens = &kv.cached[..p];
        let mut id = 0u64;
        let mut start = 0usize;
        let mut missing: Vec<(usize, usize, u64)> = Vec::new();
        while start < p {
            let len = BLOCK_TOKENS.min(p - start);
            id = block_id(id, &tokens[start..start + len]);
            // Keep the chain complete from token 0: a block whose ring rows were recycled is
            // still stored (with its tokens and any full-attention rows), so `matched` can walk
            // across it to the newest snapshot.
            if !self.find(id, &tokens[start..start + len]) {
                missing.push((start, len, id));
            }
            start += len;
        }
        if missing.is_empty() {
            self.attach_snapshots(&kv.recurrent_snaps, &kv.cached, p, None)?;
            self.trim();
            self.persist();
            return Ok(0);
        }
        // One download per layer from the first new block; blocks already stored in between are
        // read and dropped, and rows a ring recycled come back absent.
        let from = missing[0].0;
        let mut all = KvRows::default();
        DeviceKv::new(SLOT, be, cfg, kv).get(SLOT, from..p, &mut all)?;
        for (start, len, id) in &missing {
            let mut rows = KvRows::default();
            rows.append(&all, start - from..start - from + len)?;
            rows.snap = None;
            self.clock += 1;
            let block = tokens[*start..*start + *len].to_vec();
            self.insert(*id, block, rows, self.clock, true)?;
        }
        self.attach_snapshots(&kv.recurrent_snaps, &kv.cached, p, all.snap.take())?;
        self.captures += 1;
        self.trim();
        self.persist();
        Ok(p)
    }

    /// Append the changed blocks to the configured tier; a failure keeps them dirty for the drop
    /// compaction and never fails the request.
    fn persist(&mut self) {
        if self.tier.is_none() {
            return;
        }
        if let Err(e) = self.append_dirty() {
            self.dirty = true;
            tracing::warn!(error = %e, "kv store: write-through to the cold tier failed");
        }
    }

    /// Attach the slot's recurrent-state snapshots to the blocks that end at their positions.
    /// Only snapshots whose whole chain is present can ever restore, and a block's state is the
    /// same whichever turn wrote it, so an already-attached snapshot is kept. `at_p` is a copy of
    /// the snapshot at `p` already taken, used instead of cloning it again.
    fn attach_snapshots(
        &mut self,
        snaps: &[RecurrentSnap],
        tokens: &[u32],
        p: usize,
        mut at_p: Option<Arc<RecurrentSnap>>,
    ) -> Result<()> {
        for snap in snaps {
            if snap.pos == 0 || snap.pos > p {
                continue;
            }
            let Some(id) = self.block_at(tokens, snap.pos) else {
                continue;
            };
            if !self.index[&id].snap {
                let shared = match at_p.take() {
                    Some(s) if s.pos == snap.pos => s,
                    other => {
                        at_p = other;
                        Arc::new(snap.clone())
                    }
                };
                self.attach(id, shared, true)?;
                tracing::debug!(pos = snap.pos, "kv store: attached a recurrent snapshot");
            }
        }
        Ok(())
    }

    /// The stored block ending at token `pos`, when the chain covers it exactly.
    fn block_at(&self, tokens: &[u32], pos: usize) -> Option<u64> {
        if pos == 0 || !pos.is_multiple_of(BLOCK_TOKENS) || pos > tokens.len() {
            return None;
        }
        let mut id = 0u64;
        let mut start = 0usize;
        while start < pos {
            let len = BLOCK_TOKENS.min(pos - start);
            id = block_id(id, &tokens[start..start + len]);
            start += len;
        }
        self.find(id, &tokens[pos - BLOCK_TOKENS..pos])
            .then_some(id)
    }

    /// Drop least-recently-used blocks until the cache fits its budget.
    fn trim(&mut self) {
        let mut total: usize = self.index.values().map(|e| e.bytes).sum();
        while total > self.cap_bytes {
            let lru = self.index.iter().min_by_key(|(_, e)| e.used);
            let Some((id, bytes)) = lru.map(|(&id, e)| (id, e.bytes)) else {
                break;
            };
            total -= bytes;
            self.drop_entry(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::kv_tier::RamKv;
    use crate::seam::rows::{read_rows, write_rows};

    type Cache = BlockCache<RamKv>;

    fn tokens(seed: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| seed * 1000 + i).collect()
    }

    /// Index the chain for `tokens` as if it had just been captured, each block counted as
    /// `bytes`. Returns the block ids.
    fn seed(s: &mut Cache, tokens: &[u32], bytes: usize) -> Vec<u64> {
        let mut id = 0u64;
        let mut n = 0usize;
        let mut ids = Vec::new();
        while n < tokens.len() {
            let len = BLOCK_TOKENS.min(tokens.len() - n);
            let block = &tokens[n..n + len];
            id = block_id(id, block);
            s.clock += 1;
            let rows = KvRows {
                rows: len,
                ..KvRows::default()
            };
            s.insert(id, block.to_vec(), rows, s.clock, false).unwrap();
            s.index.get_mut(&id).unwrap().bytes = bytes;
            ids.push(id);
            n += len;
        }
        ids
    }

    fn entry<'a>(s: &'a Cache, tokens: &[u32]) -> &'a Entry {
        s.index
            .values()
            .find(|e| e.tokens == tokens)
            .expect("a block with these tokens")
    }

    #[test]
    fn matched_walks_the_chain_and_stops_at_a_gap() {
        let a = tokens(1, 300);
        let mut s = Cache::new(RamKv::default(), 1 << 20);
        seed(&mut s, &a, 1);
        assert_eq!(s.matched(&a).0, 300);
        let mut extended = a.clone();
        extended.extend([7, 8, 9]);
        // The short tail block only matches a prompt that ends there.
        assert_eq!(s.matched(&extended).0, 256);
        let mut diverged = a.clone();
        diverged[150] = 99;
        assert_eq!(s.matched(&diverged).0, BLOCK_TOKENS);
    }

    #[test]
    fn shared_blocks_are_stored_once() {
        let a = tokens(1, 300);
        let mut b = a[..256].to_vec();
        b.extend(tokens(2, 200));
        let mut s = Cache::new(RamKv::default(), 1 << 20);
        seed(&mut s, &a, 1);
        assert_eq!(s.index.len(), 3);
        // The second prompt shares two blocks and adds two of its own.
        assert_eq!(s.matched(&b).0, 256);
        seed(&mut s, &b[256..], 1);
        assert_eq!(s.index.len(), 5);
        assert_eq!(entry(&s, &b[384..]).tokens.len(), 72);
    }

    #[test]
    fn trim_drops_the_least_recently_used() {
        let mut s = Cache::new(RamKv::default(), 10);
        let first = seed(&mut s, &tokens(1, 300), 6);
        seed(&mut s, &tokens(2, 300), 6);
        assert_eq!(s.index.len(), 6);
        s.trim();
        // One block fits; it is the most recently used, i.e. the second prompt's tail.
        assert_eq!(s.index.len(), 1);
        assert_eq!(entry(&s, &tokens(2, 300)[256..]).bytes, 6);
        assert!(
            !s.store.contains(first[0]),
            "the store dropped the extent too"
        );
    }

    #[test]
    fn block_at_names_only_a_complete_chain_position() {
        let a = tokens(1, 300);
        let mut s = Cache::new(RamKv::default(), 1 << 20);
        seed(&mut s, &a, 1);
        assert!(s.block_at(&a, 0).is_none());
        assert!(s.block_at(&a, 128).is_some());
        assert!(s.block_at(&a, 256).is_some());
        // The short tail block only answers for its own end.
        assert!(s.block_at(&a, 300).is_none());
        // A different token at the position breaks the chain hash.
        let mut b = a.clone();
        b[128] ^= 1;
        assert!(s.block_at(&b, 256).is_none());
    }

    fn snap(pos: usize) -> RecurrentSnap {
        RecurrentSnap {
            pos,
            k: vec![(0, vec![7u8; 64])],
            v: vec![(0, vec![9u8; 32])],
            bytes: 96,
        }
    }

    #[test]
    fn a_snapshot_rides_the_block_that_ends_at_it() {
        let a = tokens(1, 512);
        let mut s = Cache::new(RamKv::default(), 1 << 20);
        seed(&mut s, &a, 1);
        s.attach_snapshots(&[snap(512)], &a, 512, None).unwrap();
        let id = s.block_at(&a, 512).expect("block at 512");
        assert_eq!(s.index[&id].bytes, 1 + 96);
        assert!(s.index[&id].snap);
        let held = s.store.host(snap_id(id)).and_then(|r| r.snap.clone());
        assert_eq!(held.expect("the snapshot extent").pos, 512);

        // A row-only budget fits the blocks; counting the snapshot pushes them over.
        let mut tight = Cache::new(RamKv::default(), 96);
        seed(&mut tight, &a, 1);
        tight.trim();
        assert_eq!(tight.index.len(), 4);
        tight.attach_snapshots(&[snap(512)], &a, 512, None).unwrap();
        tight.trim();
        assert!(tight.index.is_empty());
        assert!(!tight.store.contains(snap_id(id)));
    }

    /// The chain id must not move with the toolchain: it is the key of every block in a cold
    /// tier, so a hash change would silently turn every stored prefix into a miss.
    #[test]
    fn block_ids_are_stable_across_releases() {
        assert_eq!(block_id(0, &[1, 2, 3]), 0x973d_5966_9a25_a835);
    }

    fn mk_block(id: u64, tokens: Vec<u32>) -> Block {
        let len = tokens.len();
        Block {
            id,
            rows: KvRows::whole(len, vec![vec![1, 2, 3]], vec![vec![4, 5]]),
            tokens,
            snap: None,
            used: 1,
        }
    }

    #[test]
    #[ignore = "needs a throwaway Redis/Valkey in INFR_TEST_REDIS_URL"]
    fn a_redis_tier_round_trips_blocks_and_snapshots() {
        let url = std::env::var("INFR_TEST_REDIS_URL").expect("set INFR_TEST_REDIS_URL");
        let key = 0x1234_5678_9abc_def0 ^ u64::from(std::process::id());
        let mut tier = RedisTier::connect(&url).expect("connect");
        tier.rewrite(key, DType::F16, DType::F16, &[])
            .expect("clear");
        let a = mk_block(11, vec![1, 2, 3]);
        let mut b = mk_block(22, vec![4, 5, 6]);
        b.snap = Some(RecurrentSnap {
            pos: 6,
            k: vec![(0, vec![7; 8])],
            v: vec![(1, vec![9; 4])],
            bytes: 12,
        });
        tier.append(key, DType::F16, DType::F16, &[a.as_ref(), b.as_ref()])
            .expect("append");
        let loaded = tier.load(key).expect("load").expect("meta");
        assert_eq!((loaded.k_fmt, loaded.v_fmt), (DType::F16, DType::F16));
        assert_eq!(loaded.blocks.len(), 2);
        let got = loaded
            .blocks
            .iter()
            .find(|x| x.id == 22)
            .expect("the snapshot block");
        assert_eq!(got.tokens, vec![4, 5, 6]);
        assert_eq!(got.snap.as_ref().expect("snapshot").pos, 6);

        // A rewrite drops what the LRU no longer holds.
        tier.rewrite(key, DType::F16, DType::F16, &[a.as_ref()])
            .expect("rewrite");
        let loaded = tier.load(key).expect("load").expect("meta");
        assert_eq!(loaded.blocks.len(), 1);
        assert_eq!(loaded.blocks[0].id, 11);

        tier.rewrite(key, DType::F16, DType::F16, &[])
            .expect("clear");
        assert!(tier.load(key).expect("load").is_none(), "nothing left");
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("infr-kv-{}-{name}.bin", std::process::id()))
    }

    #[test]
    fn a_saved_store_loads_its_blocks_and_snapshots() {
        let path = temp_path("roundtrip");
        std::fs::remove_file(&path).ok();
        let key = 42u64;
        let file = |p: &std::path::PathBuf| -> Option<Box<dyn KvColdTier>> {
            Some(Box::new(FileTier::new(p.clone())))
        };
        let open = |p| Cache::with_tier(RamKv::default(), 1 << 20, file(p), key);
        {
            let mut s = open(&path);
            let ids = seed(&mut s, &tokens(1, 300), 7);
            s.attach(ids[2], Arc::new(snap(300)), true).unwrap();
            for e in s.index.values_mut() {
                e.dirty = true;
            }
            s.tier_fmt = Some((s.k_fmt, s.v_fmt));
            s.persist();
            assert!(!s.dirty, "the append went through");
        }
        let journal_one = std::fs::metadata(&path).expect("journal").len();
        {
            // A second conversation appends a record of its own.
            let mut s = open(&path);
            let ids = seed(&mut s, &tokens(2, 128), 5);
            s.index.get_mut(&ids[0]).expect("seeded").dirty = true;
            s.persist();
            s.dirty = false;
        }
        let loaded = open(&path);
        assert_eq!(loaded.index.len(), 4, "both records came back");
        drop(loaded);

        // A torn second record keeps the first record whole.
        let bytes = std::fs::read(&path).expect("read");
        std::fs::write(&path, &bytes[..bytes.len() - 3]).expect("write torn");
        assert!(bytes.len() as u64 > journal_one);
        let loaded = open(&path);
        assert_eq!(loaded.index.len(), 3);
        let tail = &tokens(1, 300)[256..];
        assert_eq!(
            entry(&loaded, tail).bytes,
            96,
            "the snapshot's bytes, recomputed"
        );
        let tail_id = *loaded
            .index
            .iter()
            .find(|(_, e)| e.tokens == tail)
            .expect("the tail")
            .0;
        let snap = loaded
            .store
            .host(snap_id(tail_id))
            .and_then(|r| r.snap.clone())
            .expect("the snapshot survived");
        assert_eq!((snap.pos, snap.k[0].1.len()), (300, 64));

        drop(loaded);

        // Another model's key discards the file whole.
        let wrong = Cache::with_tier(RamKv::default(), 1 << 20, file(&path), key + 1);
        assert!(wrong.index.is_empty());
        drop(wrong);

        // A first record torn mid-way leaves nothing to load, and never a half-read block.
        let bytes = std::fs::read(&path).expect("read");
        std::fs::write(&path, &bytes[..journal_one as usize - 3]).expect("write torn");
        let broken = open(&path);
        assert!(broken.index.is_empty());
        drop(broken);
        std::fs::remove_file(path).ok();
    }

    const ROW: usize = 32;

    /// Two attention layers of 32-element f16 rows; with `window`, layer 0 is a sliding window.
    fn slot_cfg(window: usize) -> Config {
        Config {
            n_layer: 2,
            n_head: 4,
            n_kv: 2,
            n_embd: 64,
            head_dim: 16,
            rope_dim: 16,
            head_dim_swa: 16,
            n_kv_swa: 2,
            swa_window: window,
            swa_pattern: if window > 0 { 2 } else { 0 },
            ..Default::default()
        }
    }

    /// The bytes of logical row `r` of layer `l`'s side `side` in the test slots.
    fn row_data(l: usize, side: u16, r: usize) -> Vec<u8> {
        let v = (r as u16) ^ ((l as u16) << 12) ^ (side << 14);
        (0..ROW).flat_map(|_| v.to_le_bytes()).collect()
    }

    /// A slot holding `n` tokens, every row written through its own layout.
    fn filled_slot(be: &dyn Backend, cfg: &Config, ring: bool, n: usize) -> SeamKv {
        let mut kv = SeamKv::for_test(be, cfg, 1024, ring, 56, DType::F16).unwrap();
        for l in 0..cfg.n_layer {
            let layout = kv.row_layout(be, cfg, l);
            for (side, buf) in [(0, &kv.kbufs[l]), (1, &kv.vbufs[l])] {
                let data: Vec<u8> = (0..n).flat_map(|r| row_data(l, side, r)).collect();
                write_rows(be, buf.as_ref(), DType::F16, ROW, &*layout, 0..n, &data).unwrap();
            }
        }
        kv.cached = tokens(3, n);
        kv.written_end = n;
        kv
    }

    fn read_side(
        be: &dyn Backend,
        cfg: &Config,
        kv: &SeamKv,
        l: usize,
        side: u16,
        rows: std::ops::Range<usize>,
    ) -> Vec<u8> {
        let layout = kv.row_layout(be, cfg, l);
        let buf = if side == 0 {
            &kv.kbufs[l]
        } else {
            &kv.vbufs[l]
        };
        let mut out = vec![0u8; rows.len() * ROW * 2];
        read_rows(be, buf.as_ref(), DType::F16, ROW, &*layout, rows, &mut out).unwrap();
        out
    }

    #[test]
    fn a_linear_slot_round_trips_through_the_cache() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(0);
        let src = filled_slot(&be, &cfg, false, 300);
        let mut s = Cache::new(RamKv::default(), 1 << 20);
        assert_eq!(s.capture(&be, &cfg, &src, 300).unwrap(), 300);
        assert_eq!(s.index.len(), 3);
        // Capturing the same prefix again stores nothing new.
        assert_eq!(s.capture(&be, &cfg, &src, 300).unwrap(), 0);
        assert_eq!(s.captures, 1);

        let mut dst = SeamKv::for_test(&be, &cfg, 1024, false, 56, DType::F16).unwrap();
        assert_eq!(s.restore(&be, &cfg, &mut dst, &src.cached).unwrap(), 300);
        assert_eq!(dst.cached, src.cached);
        assert_eq!((s.restores, s.restored_tokens), (1, 300));
        for l in 0..2 {
            for side in 0..2 {
                let want: Vec<u8> = (0..300).flat_map(|r| row_data(l, side, r)).collect();
                assert_eq!(
                    read_side(&be, &cfg, &dst, l, side, 0..300),
                    want,
                    "layer {l} side {side}"
                );
            }
        }
    }

    /// A wrapped ring keeps only its window: the block that straddles the window's start stores
    /// the rows of it that are still resident, and the restore writes the window back in place.
    #[test]
    fn a_wrapped_ring_round_trips_its_window() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(200);
        let n = 700;
        let src = filled_slot(&be, &cfg, true, n);
        let ring = src.window_ring_rows(&be, &cfg).expect("a ring");
        assert_eq!(ring, 256);
        let mut s = Cache::new(RamKv::default(), 1 << 20);
        assert_eq!(s.capture(&be, &cfg, &src, n).unwrap(), n);
        let ids = s.matched(&src.cached).1;
        assert_eq!(ids.len(), 6);
        // The window starts at row 444: blocks below it are empty, the one straddling it keeps
        // the rows from 444 on, and the rest are whole.
        for (j, id) in ids.iter().enumerate() {
            let rows = s.store.host(*id).unwrap();
            let runs = &rows.layers[0].runs;
            let run = (runs.len() == 1).then(|| runs[0].clone());
            match j {
                0..=2 => assert!(runs.is_empty(), "block {j}"),
                3 => assert_eq!(run, Some(60..128), "block {j}"),
                4 => assert_eq!(run, Some(0..BLOCK_TOKENS), "block {j}"),
                _ => assert_eq!(run, Some(0..60), "block {j}"),
            }
            assert!(!rows.layers[1].k.is_empty());
        }

        let mut dst = SeamKv::for_test(&be, &cfg, 1024, true, 56, DType::F16).unwrap();
        assert_eq!(s.restore(&be, &cfg, &mut dst, &src.cached).unwrap(), n);
        for side in 0..2 {
            for l in 0..2 {
                let from = dst.row_layout(&be, &cfg, l).oldest_resident(n);
                let want: Vec<u8> = (from..n).flat_map(|r| row_data(l, side, r)).collect();
                assert_eq!(
                    read_side(&be, &cfg, &dst, l, side, from..n),
                    want,
                    "layer {l} side {side} from row {from}"
                );
            }
        }
    }

    /// A restore whose blocks cannot fill the destination's window re-prefills instead of leaving
    /// rows attention will read unwritten.
    #[test]
    fn a_restore_into_a_wider_ring_refuses() {
        let be = infr_cpu::CpuBackend::new();
        let src_cfg = slot_cfg(200);
        let n = 700;
        let src = filled_slot(&be, &src_cfg, true, n);
        let mut s = Cache::new(RamKv::default(), 1 << 20);
        s.capture(&be, &src_cfg, &src, n).unwrap();

        let dst_cfg = slot_cfg(400);
        let mut dst = SeamKv::for_test(&be, &dst_cfg, 1024, true, 56, DType::F16).unwrap();
        assert_eq!(dst.window_ring_rows(&be, &dst_cfg), Some(512));
        assert_eq!(s.restore(&be, &dst_cfg, &mut dst, &src.cached).unwrap(), 0);
        assert!(dst.cached.is_empty());
    }

    /// Decode can write rows past the tokens it accepted, so the ring recycles from the row count
    /// actually written, not from the cached prefix.
    #[test]
    fn a_capture_skips_rows_a_chain_wrote_past_the_prompt() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(200);
        let mut src = filled_slot(&be, &cfg, true, 600);
        src.written_end = 800;
        let mut s = Cache::new(RamKv::default(), 1 << 20);
        assert_eq!(s.capture(&be, &cfg, &src, 600).unwrap(), 600);
        let ids = s.matched(&src.cached).1;
        assert_eq!(ids.len(), 5);
        // Rows below 544 are gone: the block over 512..640 keeps only 544..600.
        let rows = s.store.host(ids[4]).unwrap();
        assert_eq!(rows.layers[0].runs, vec![32..88]);
    }
}
