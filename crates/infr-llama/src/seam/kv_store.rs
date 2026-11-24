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
use crate::EngineConfig;
use anyhow::Result;
use infr_core::backend::{Backend, Buffer};
use infr_core::budget::kv_fmt_bytes;
use infr_core::DType;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use super::{
    kv_row_elems, kv_rows_at, ubatch_rows,
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
/// with the toolchain. A collision only costs a wasted comparison, since every lookup also
/// compares the token ids.
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

fn write_block(w: &mut impl Write, h: &mut Fnv, b: &Block) -> std::io::Result<()> {
    put_u64(w, h, b.id)?;
    put_u32(w, h, b.len as u32)?;
    for &t in &b.tokens {
        put_u32(w, h, t)?;
    }
    for side in [&b.k, &b.v] {
        put_u32(w, h, side.len() as u32)?;
        for bytes in side {
            put_bytes(w, h, bytes)?;
        }
    }
    match &b.snap {
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

/// Write one stored block's rows into a sliding-window ring at the block's own position, wrapping
/// at `rcap` so the newest rows end up where the window reads them.
#[allow(clippy::too_many_arguments)]
fn write_ring_block(
    be: &dyn Backend,
    dst: &dyn Buffer,
    fmt: DType,
    row: usize,
    rcap: usize,
    start: usize,
    len: usize,
    bytes: &[u8],
) -> Result<()> {
    if len == 0 || rcap == 0 {
        return Ok(());
    }
    let first = start % rcap;
    let head = len.min(rcap - first);
    let hb = kv_fmt_bytes(fmt, head * row);
    be.upload_at(dst, kv_fmt_bytes(fmt, first * row), &bytes[..hb])?;
    if head < len {
        let tail = len - head;
        let tb = kv_fmt_bytes(fmt, tail * row);
        be.upload_at(dst, 0, &bytes[hb..hb + tb])?;
    }
    Ok(())
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
        let mut bytes = 0usize;
        for _ in 0..layers {
            let b = get_bytes(r, h)?;
            bytes += b.len();
            side.push(b);
        }
        sides.push((side, bytes));
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
    let (k, kb) = sides.remove(0);
    let (v, vb) = sides.remove(0);
    let snap_bytes = snap.as_ref().map_or(0, |s| s.bytes);
    Ok(Block {
        id,
        len,
        tokens,
        k,
        v,
        snap,
        bytes: kb + vb + snap_bytes,
        used,
        dirty: false,
    })
}

/// One stored block: `len` token ids and the raw K/V rows for that token range.
///
/// `len` is `BLOCK_TOKENS` except for the tail of a captured prefix, which may be short; a short
/// block only ever matches at the end of a prompt.
pub(crate) struct Block {
    id: u64,
    len: usize,
    tokens: Vec<u32>,
    /// Per layer, in `SeamKv::kbufs` order. `v[l]` is empty for an MLA layer (no V side) and both
    /// are empty for a recurrent layer, whose state a `snap` carries instead.
    k: Vec<Vec<u8>>,
    v: Vec<Vec<u8>>,
    /// The recurrent state at this block's end, when one was captured there.
    snap: Option<RecurrentSnap>,
    bytes: usize,
    used: u64,
    /// Changed since the last save/append.
    dirty: bool,
}

/// Host-RAM LRU store of captured prompt blocks.
pub(crate) struct KvPrefixStore {
    blocks: Vec<Block>,
    cap_bytes: usize,
    /// Monotonic access counter (LRU order, not wall time — the store never sleeps).
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

impl Drop for KvPrefixStore {
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
    fn append(&mut self, key: u64, k_fmt: DType, v_fmt: DType, blocks: &[&Block]) -> Result<()>;
    /// Replace everything stored under `key` with `blocks`.
    fn rewrite(&mut self, key: u64, k_fmt: DType, v_fmt: DType, blocks: &[&Block]) -> Result<()>;
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

    fn write_record(w: &mut impl Write, blocks: &[&Block]) -> std::io::Result<()> {
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

    fn append(&mut self, key: u64, k_fmt: DType, v_fmt: DType, blocks: &[&Block]) -> Result<()> {
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

    fn rewrite(&mut self, key: u64, k_fmt: DType, v_fmt: DType, blocks: &[&Block]) -> Result<()> {
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

    fn encode(b: &Block) -> Vec<u8> {
        let mut v = Vec::with_capacity(b.bytes + 16);
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

    fn append(&mut self, key: u64, k_fmt: DType, v_fmt: DType, blocks: &[&Block]) -> Result<()> {
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

    fn rewrite(&mut self, key: u64, k_fmt: DType, v_fmt: DType, blocks: &[&Block]) -> Result<()> {
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

impl KvPrefixStore {
    /// Build a store with `cap_bytes` of host memory. A zero cap is a disabled store: every call
    /// returns immediately, so the engine pays nothing for a feature nobody asked for.
    #[cfg(test)]
    pub(crate) fn new(cap_bytes: usize) -> Self {
        Self::with_tier(cap_bytes, None, 0)
    }

    /// [`new`](Self::new), backed by a cold `tier`: the store loads what the tier holds for `key`
    /// now, writes each capture through to it, and compacts it on drop. `key` identifies the
    /// model geometry the blocks belong to.
    pub(crate) fn with_tier(cap_bytes: usize, tier: Option<Box<dyn KvColdTier>>, key: u64) -> Self {
        let mut store = Self {
            blocks: Vec::new(),
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
        store.load_tier();
        store
    }

    /// Fill the RAM index from the tier. A tier that fails or holds another model's blocks leaves
    /// the store empty; it never fails startup.
    fn load_tier(&mut self) {
        let Some(tier) = self.tier.as_mut() else {
            return;
        };
        let name = tier.describe();
        match tier.load(self.key) {
            Ok(Some(c)) => {
                self.k_fmt = c.k_fmt;
                self.v_fmt = c.v_fmt;
                self.tier_fmt = Some((c.k_fmt, c.v_fmt));
                self.clock = c.blocks.iter().map(|b| b.used).max().unwrap_or(0);
                self.blocks = c.blocks;
                self.trim();
                tracing::info!(
                    tier = %name,
                    blocks = self.blocks.len(),
                    "kv store: loaded the saved prefixes"
                );
            }
            Ok(None) => {}
            Err(e) => tracing::warn!(
                tier = %name,
                error = %e,
                "kv store: the saved prefixes are unusable; starting empty"
            ),
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.cap_bytes > 0
    }

    /// `(blocks, bytes)` currently resident, for the capture log.
    pub(crate) fn stats(&self) -> (usize, usize) {
        (self.blocks.len(), self.blocks.iter().map(|b| b.bytes).sum())
    }

    /// Replace the tier's contents with the store's current blocks, which drops what the LRU
    /// evicted and any superseded copies.
    fn rewrite_tier(&mut self) -> Result<()> {
        let Some(tier) = self.tier.as_mut() else {
            self.dirty = false;
            return Ok(());
        };
        let all: Vec<&Block> = self.blocks.iter().collect();
        tier.rewrite(self.key, self.k_fmt, self.v_fmt, &all)?;
        for b in &mut self.blocks {
            b.dirty = false;
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
        let dirty: Vec<usize> = (0..self.blocks.len())
            .filter(|&i| self.blocks[i].dirty)
            .collect();
        if dirty.is_empty() {
            return Ok(());
        }
        let batch: Vec<&Block> = dirty.iter().map(|&i| &self.blocks[i]).collect();
        let tier = self.tier.as_mut().expect("checked above");
        tier.append(self.key, self.k_fmt, self.v_fmt, &batch)?;
        for i in dirty {
            self.blocks[i].dirty = false;
        }
        self.dirty = false;
        Ok(())
    }

    fn find(&self, id: u64, tokens: &[u32]) -> Option<usize> {
        self.blocks
            .iter()
            .position(|b| b.id == id && b.tokens == tokens)
    }

    /// Walk `prompt` block by block; return the tokens covered by the leading stored run and the
    /// blocks that cover them, with their use stamped.
    fn matched(&mut self, prompt: &[u32]) -> (usize, Vec<usize>) {
        let mut id = 0u64;
        let mut n = 0usize;
        let mut hits = Vec::new();
        while n < prompt.len() {
            let len = BLOCK_TOKENS.min(prompt.len() - n);
            let block = &prompt[n..n + len];
            id = block_id(id, block);
            let Some(i) = self.find(id, block) else { break };
            self.clock += 1;
            self.blocks[i].used = self.clock;
            hits.push(i);
            n += len;
            if len < BLOCK_TOKENS {
                break;
            }
        }
        (n, hits)
    }

    /// Copy the longest stored run of blocks into `kv` (a device upload per layer) and mark those
    /// tokens materialized. Returns the number of tokens restored.
    ///
    /// A sliding-window ring is written at each block's own ring position (`start % rows`), so a
    /// prefix longer than the ring still seeds it: the newest blocks overwrite the oldest, which
    /// is exactly the window attention will read. Only a destination whose own cache already
    /// wrapped is refused — its rows no longer describe its `cached`.
    pub(crate) fn restore(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        ec: &EngineConfig,
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
        if kv.kv_ring {
            for l in 0..cfg.n_layer {
                if cfg.is_swa_layer(l) {
                    let rows =
                        kv_rows_at(cfg, l, kv.max_ctx, true, ubatch_rows(ec), be.kv_row_align());
                    if kv.cached.len() > rows {
                        return Ok(0);
                    }
                }
            }
        }
        let mut keep = Vec::new();
        let mut n = 0usize;
        for i in hits {
            let len = self.blocks[i].len;
            if n + len > p {
                break;
            }
            n += len;
            keep.push(i);
        }
        // A recurrent model's rows are useless without the state they were written beside: only a
        // block carrying a snapshot can seed it, and everything after that block must be prefilled.
        if cfg.has_recurrent_state() {
            let Some(at) = keep.iter().rposition(|&i| self.blocks[i].snap.is_some()) else {
                return Ok(0);
            };
            keep.truncate(at + 1);
            n = keep.iter().map(|&i| self.blocks[i].len).sum();
        }
        if n < MIN_PREFIX_TOKENS {
            return Ok(0);
        }
        if let Some(p) = kv.paged.as_mut() {
            p.make_private(be, 0, n)?;
        }
        for l in 0..cfg.n_layer {
            if cfg.is_recurrent_layer(l) {
                continue;
            }
            let (k_row, v_row) = kv_row_elems(cfg, l);
            if kv.kv_ring && cfg.is_swa_layer(l) {
                let rcap = kv_rows_at(cfg, l, kv.max_ctx, true, ubatch_rows(ec), be.kv_row_align());
                let mut start = 0usize;
                for i in &keep {
                    let b = &self.blocks[*i];
                    // A block captured before the ring's window was recycled has no rows.
                    if !b.k[l].is_empty() {
                        write_ring_block(
                            be,
                            kv.kbufs[l].as_ref(),
                            kv.k_fmt,
                            k_row,
                            rcap,
                            start,
                            b.len,
                            &b.k[l],
                        )?;
                        if v_row > 0 && !b.v[l].is_empty() {
                            write_ring_block(
                                be,
                                kv.vbufs[l].as_ref(),
                                kv.v_fmt,
                                v_row,
                                rcap,
                                start,
                                b.len,
                                &b.v[l],
                            )?;
                        }
                    }
                    start += b.len;
                }
                continue;
            }
            let kb = kv_fmt_bytes(kv.k_fmt, n * k_row);
            let mut buf = Vec::with_capacity(kb);
            for i in &keep {
                buf.extend_from_slice(&self.blocks[*i].k[l]);
            }
            debug_assert_eq!(buf.len(), kb);
            Self::write_rows(
                be,
                cfg,
                kv,
                l,
                kv.k_fmt,
                k_row,
                n,
                kv.kbufs[l].as_ref(),
                &buf,
            )?;
            if v_row > 0 {
                let vb = kv_fmt_bytes(kv.v_fmt, n * v_row);
                let mut buf = Vec::with_capacity(vb);
                for i in &keep {
                    buf.extend_from_slice(&self.blocks[*i].v[l]);
                }
                debug_assert_eq!(buf.len(), vb);
                Self::write_rows(
                    be,
                    cfg,
                    kv,
                    l,
                    kv.v_fmt,
                    v_row,
                    n,
                    kv.vbufs[l].as_ref(),
                    &buf,
                )?;
            }
        }
        if cfg.has_recurrent_state() {
            let snap = self.blocks[*keep.last().expect("the snapshot block")]
                .snap
                .as_ref()
                .expect("the truncation kept the snapshot block");
            for (l, bytes) in &snap.k {
                be.upload(kv.kbufs[*l].as_ref(), bytes)?;
            }
            for (l, bytes) in &snap.v {
                be.upload(kv.vbufs[*l].as_ref(), bytes)?;
            }
            kv.written_end = n;
        }
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
        ec: &EngineConfig,
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
            self.blocks.clear();
            self.k_fmt = kv.k_fmt;
            self.v_fmt = kv.v_fmt;
        }
        // A wrapped SWA ring has recycled every row below the window it still holds.
        let mut safe_start = 0usize;
        if kv.kv_ring {
            let mut min_rows = usize::MAX;
            for l in 0..cfg.n_layer {
                if cfg.is_swa_layer(l) {
                    min_rows = min_rows.min(kv_rows_at(
                        cfg,
                        l,
                        kv.max_ctx,
                        true,
                        ubatch_rows(ec),
                        be.kv_row_align(),
                    ));
                }
            }
            if min_rows != usize::MAX {
                safe_start = kv.cached.len().saturating_sub(min_rows);
            }
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
            if self.find(id, &tokens[start..start + len]).is_none() {
                missing.push((start, len, id));
            }
            start += len;
        }
        if missing.is_empty() {
            self.attach_snapshots(&kv.recurrent_snaps, &kv.cached, p);
            self.trim();
            self.persist();
            return Ok(0);
        }
        let base = self.blocks.len();
        for (start, len, id) in &missing {
            self.clock += 1;
            self.blocks.push(Block {
                id: *id,
                len: *len,
                tokens: tokens[*start..*start + *len].to_vec(),
                k: vec![Vec::new(); cfg.n_layer],
                v: vec![Vec::new(); cfg.n_layer],
                snap: None,
                bytes: 0,
                used: self.clock,
                dirty: true,
            });
        }
        let filled = self.fill_from_device(be, cfg, kv, p, safe_start, &missing, base);
        if let Err(e) = filled {
            self.blocks.truncate(base);
            return Err(e);
        }
        self.attach_snapshots(&kv.recurrent_snaps, &kv.cached, p);
        self.captures += 1;
        self.trim();
        self.persist();
        Ok(p)
    }

    /// Append the changed blocks to the configured file; a failure keeps them dirty for the drop
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
    /// same whichever turn wrote it, so an already-attached snapshot is kept.
    fn attach_snapshots(&mut self, snaps: &[RecurrentSnap], tokens: &[u32], p: usize) {
        for snap in snaps {
            if snap.pos == 0 || snap.pos > p {
                continue;
            }
            let Some(i) = self.block_at(tokens, snap.pos) else {
                continue;
            };
            if self.blocks[i].snap.is_none() {
                self.blocks[i].bytes += snap.bytes;
                self.blocks[i].snap = Some(snap.clone());
                self.blocks[i].dirty = true;
                tracing::debug!(
                    pos = snap.pos,
                    block = i,
                    "kv store: attached a recurrent snapshot"
                );
            }
        }
    }

    /// The stored block ending at token `pos`, when the chain covers it exactly.
    fn block_at(&self, tokens: &[u32], pos: usize) -> Option<usize> {
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
    }

    /// Download each new block's rows into the blocks pushed at `base .. base + missing.len()`.
    fn fill_from_device(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        kv: &SeamKv,
        p: usize,
        safe_start: usize,
        missing: &[(usize, usize, u64)],
        base: usize,
    ) -> Result<()> {
        for l in 0..cfg.n_layer {
            if cfg.is_recurrent_layer(l) {
                continue;
            }
            let (k_row, v_row) = kv_row_elems(cfg, l);
            // A sliding-window ring holds only its last `rows` positions, at `pos % rows`; the
            // bulk read below would run past its buffer once the conversation is longer than the
            // window.
            if kv.kv_ring && cfg.is_swa_layer(l) {
                let rcap = kv_rows_at(cfg, l, kv.max_ctx, true, kv.ubatch, be.kv_row_align());
                for (j, (start, len, _)) in missing.iter().enumerate() {
                    if *start < safe_start {
                        continue;
                    }
                    let first = start % rcap;
                    let n1 = (*len).min(rcap - first);
                    let mut kslice = vec![0u8; kv_fmt_bytes(kv.k_fmt, *len * k_row)];
                    be.download_at(
                        kv.kbufs[l].as_ref(),
                        kv_fmt_bytes(kv.k_fmt, first * k_row),
                        &mut kslice[..kv_fmt_bytes(kv.k_fmt, n1 * k_row)],
                    )?;
                    if n1 < *len {
                        be.download_at(
                            kv.kbufs[l].as_ref(),
                            0,
                            &mut kslice[kv_fmt_bytes(kv.k_fmt, n1 * k_row)..],
                        )?;
                    }
                    let vslice = if v_row > 0 {
                        let mut buf = vec![0u8; kv_fmt_bytes(kv.v_fmt, *len * v_row)];
                        be.download_at(
                            kv.vbufs[l].as_ref(),
                            kv_fmt_bytes(kv.v_fmt, first * v_row),
                            &mut buf[..kv_fmt_bytes(kv.v_fmt, n1 * v_row)],
                        )?;
                        if n1 < *len {
                            be.download_at(
                                kv.vbufs[l].as_ref(),
                                0,
                                &mut buf[kv_fmt_bytes(kv.v_fmt, n1 * v_row)..],
                            )?;
                        }
                        buf
                    } else {
                        Vec::new()
                    };
                    let b = &mut self.blocks[base + j];
                    b.bytes += kslice.len() + vslice.len();
                    b.k[l] = kslice;
                    b.v[l] = vslice;
                }
                continue;
            }
            let mut kbuf = vec![0u8; kv_fmt_bytes(kv.k_fmt, p * k_row)];
            Self::read_rows(
                be,
                cfg,
                kv,
                l,
                kv.k_fmt,
                k_row,
                p,
                kv.kbufs[l].as_ref(),
                &mut kbuf,
            )?;
            let vbuf = if v_row > 0 {
                let mut buf = vec![0u8; kv_fmt_bytes(kv.v_fmt, p * v_row)];
                Self::read_rows(
                    be,
                    cfg,
                    kv,
                    l,
                    kv.v_fmt,
                    v_row,
                    p,
                    kv.vbufs[l].as_ref(),
                    &mut buf,
                )?;
                buf
            } else {
                Vec::new()
            };
            for (j, (start, len, _)) in missing.iter().enumerate() {
                let ks = kv_fmt_bytes(kv.k_fmt, *start * k_row);
                let ke = ks + kv_fmt_bytes(kv.k_fmt, *len * k_row);
                let kslice = kbuf[ks..ke].to_vec();
                let vslice = if v_row > 0 {
                    let vs = kv_fmt_bytes(kv.v_fmt, *start * v_row);
                    let ve = vs + kv_fmt_bytes(kv.v_fmt, *len * v_row);
                    vbuf[vs..ve].to_vec()
                } else {
                    Vec::new()
                };
                let b = &mut self.blocks[base + j];
                b.bytes += kslice.len() + vslice.len();
                b.k[l] = kslice;
                b.v[l] = vslice;
            }
        }
        Ok(())
    }

    /// Download `rows` rows of one side into `buf`, walking the slot's page table when the layer
    /// is pooled (a paged layer has no single row base to read from).
    #[allow(clippy::too_many_arguments)]
    fn read_rows(
        be: &dyn Backend,
        cfg: &Config,
        kv: &SeamKv,
        l: usize,
        fmt: DType,
        row: usize,
        rows: usize,
        src: &dyn Buffer,
        buf: &mut [u8],
    ) -> Result<()> {
        let Some(paged) = kv
            .paged
            .as_ref()
            .filter(|_| crate::seam::paged_layer(cfg, l))
        else {
            be.download_at(src, 0, buf)?;
            return Ok(());
        };
        for page in 0..rows.div_ceil(infr_vulkan::paged::PAGE_ROWS) {
            let first = page * infr_vulkan::paged::PAGE_ROWS;
            let n = infr_vulkan::paged::PAGE_ROWS.min(rows - first);
            let phys = paged.map.page(page) as usize;
            let at = kv_fmt_bytes(fmt, first * row);
            let bytes = kv_fmt_bytes(fmt, n * row);
            be.download_at(
                src,
                kv_fmt_bytes(fmt, phys * infr_vulkan::paged::PAGE_ROWS * row),
                &mut buf[at..at + bytes],
            )?;
        }
        Ok(())
    }

    /// Upload `rows` rows of one side from `buf`, page by page when the layer is pooled.
    #[allow(clippy::too_many_arguments)]
    fn write_rows(
        be: &dyn Backend,
        cfg: &Config,
        kv: &SeamKv,
        l: usize,
        fmt: DType,
        row: usize,
        rows: usize,
        dst: &dyn Buffer,
        buf: &[u8],
    ) -> Result<()> {
        let Some(paged) = kv
            .paged
            .as_ref()
            .filter(|_| crate::seam::paged_layer(cfg, l))
        else {
            be.upload_at(dst, 0, buf)?;
            return Ok(());
        };
        for page in 0..rows.div_ceil(infr_vulkan::paged::PAGE_ROWS) {
            let first = page * infr_vulkan::paged::PAGE_ROWS;
            let n = infr_vulkan::paged::PAGE_ROWS.min(rows - first);
            let phys = paged.map.page(page) as usize;
            let at = kv_fmt_bytes(fmt, first * row);
            let bytes = kv_fmt_bytes(fmt, n * row);
            be.upload_at(
                dst,
                kv_fmt_bytes(fmt, phys * infr_vulkan::paged::PAGE_ROWS * row),
                &buf[at..at + bytes],
            )?;
        }
        Ok(())
    }

    /// Drop least-recently-used blocks until the store fits its budget.
    fn trim(&mut self) {
        let mut total: usize = self.blocks.iter().map(|b| b.bytes).sum();
        while total > self.cap_bytes && !self.blocks.is_empty() {
            let (i, _) = self
                .blocks
                .iter()
                .enumerate()
                .min_by_key(|(_, b)| b.used)
                .expect("non-empty");
            total -= self.blocks.remove(i).bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(seed: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| seed * 1000 + i).collect()
    }

    /// Push the chain for `tokens` as if it had just been captured.
    fn seed(s: &mut KvPrefixStore, tokens: &[u32], bytes: usize) {
        let mut id = 0u64;
        let mut n = 0usize;
        while n < tokens.len() {
            let len = BLOCK_TOKENS.min(tokens.len() - n);
            let block = &tokens[n..n + len];
            id = block_id(id, block);
            s.clock += 1;
            s.blocks.push(Block {
                id,
                len,
                tokens: block.to_vec(),
                k: Vec::new(),
                v: Vec::new(),
                snap: None,
                bytes,
                used: s.clock,
                dirty: false,
            });
            n += len;
        }
    }

    #[test]
    fn matched_walks_the_chain_and_stops_at_a_gap() {
        let a = tokens(1, 300);
        let mut s = KvPrefixStore::new(1 << 20);
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
        let mut s = KvPrefixStore::new(1 << 20);
        seed(&mut s, &a, 1);
        assert_eq!(s.blocks.len(), 3);
        // The second prompt shares two blocks and adds two of its own.
        assert_eq!(s.matched(&b).0, 256);
        seed(&mut s, &b[256..], 1);
        assert_eq!(s.blocks.len(), 5);
        assert_eq!(s.blocks[4].len, 72);
    }

    #[test]
    fn trim_drops_the_least_recently_used() {
        let mut s = KvPrefixStore::new(10);
        seed(&mut s, &tokens(1, 300), 6);
        seed(&mut s, &tokens(2, 300), 6);
        assert_eq!(s.blocks.len(), 6);
        s.trim();
        // One block fits; it is the most recently used, i.e. the second prompt's tail.
        assert_eq!(s.blocks.len(), 1);
        assert_eq!(s.blocks[0].tokens, tokens(2, 300)[256..]);
    }

    #[test]
    fn block_at_names_only_a_complete_chain_position() {
        let a = tokens(1, 300);
        let mut s = KvPrefixStore::new(1 << 20);
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

    #[test]
    fn a_snapshot_rides_the_block_that_ends_at_it() {
        let a = tokens(1, 512);
        let mut s = KvPrefixStore::new(1 << 20);
        seed(&mut s, &a, 1);
        let snap = RecurrentSnap {
            pos: 512,
            k: vec![(0, vec![7u8; 64])],
            v: vec![(0, vec![9u8; 32])],
            bytes: 96,
        };
        let kv_snaps = vec![snap];
        s.attach_snapshots(&kv_snaps, &a, 512);
        let i = s.block_at(&a, 512).expect("block at 512");
        assert_eq!(s.blocks[i].bytes, 1 + 96);
        assert!(s.blocks[i].snap.is_some());
        assert_eq!(s.block_at(&a, 512), Some(i));
        // A row-only budget fits the block; counting the snapshot pushes it over and drops it.
        let snap_block = s.blocks.pop().expect("the block at 512");
        assert_eq!(snap_block.bytes, 97);
        let mut tight = KvPrefixStore::new(96);
        tight.blocks.push(snap_block);
        tight.trim();
        assert!(tight.blocks.is_empty());
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
            len,
            tokens,
            k: vec![vec![1, 2, 3]],
            v: vec![vec![4, 5]],
            snap: None,
            bytes: 5,
            used: 1,
            dirty: false,
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
        tier.append(key, DType::F16, DType::F16, &[&a, &b])
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
        tier.rewrite(key, DType::F16, DType::F16, &[&a])
            .expect("rewrite");
        let loaded = tier.load(key).expect("load").expect("meta");
        assert_eq!(loaded.blocks.len(), 1);
        assert_eq!(loaded.blocks[0].id, 11);

        tier.rewrite(key, DType::F16, DType::F16, &[])
            .expect("clear");
        assert!(tier.load(key).expect("load").is_none(), "nothing left");
    }

    /// A stored block lands at its own ring position and wraps at the capacity, so a prefix
    /// longer than the ring still leaves the newest rows where attention reads them.
    #[test]
    fn a_ring_block_wraps_at_its_capacity() {
        use infr_core::backend::BufferUsage;
        let be = infr_cpu::CpuBackend::new();
        let buf = be.alloc(4 * 512, BufferUsage::KvCache).expect("buffer");
        let mut rows = Vec::with_capacity(300 * 4);
        for i in 0..300u32 {
            rows.extend_from_slice(&i.to_le_bytes());
        }
        write_ring_block(&be, buf.as_ref(), DType::F32, 1, 512, 400, 300, &rows).expect("write");
        let mut out = vec![0u8; 4 * 512];
        be.download(buf.as_ref(), &mut out).expect("read");
        let at = |row: usize| u32::from_le_bytes(out[row * 4..row * 4 + 4].try_into().unwrap());
        assert_eq!(at(400), 0);
        assert_eq!(at(511), 111);
        assert_eq!(at(0), 112);
        assert_eq!(at(187), 299);
        assert_eq!(at(188), 0, "nothing past the block");
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
        {
            let mut s = KvPrefixStore::with_tier(1 << 20, file(&path), key);
            seed(&mut s, &tokens(1, 300), 7);
            s.blocks[2].snap = Some(RecurrentSnap {
                pos: 300,
                k: vec![(0, vec![7u8; 64])],
                v: vec![(0, vec![9u8; 32])],
                bytes: 96,
            });
            for b in &mut s.blocks {
                b.dirty = true;
            }
            s.tier_fmt = Some((s.k_fmt, s.v_fmt));
            s.persist();
            assert!(!s.dirty, "the append went through");
        }
        let journal_one = std::fs::metadata(&path).expect("journal").len();
        {
            // A second conversation appends a record of its own.
            let mut s = KvPrefixStore::with_tier(1 << 20, file(&path), key);
            seed(&mut s, &tokens(2, 128), 5);
            s.blocks.last_mut().expect("seeded").dirty = true;
            s.persist();
            s.dirty = false;
        }
        let loaded = KvPrefixStore::with_tier(1 << 20, file(&path), key);
        assert_eq!(loaded.blocks.len(), 4, "both records came back");
        drop(loaded);

        // A torn second record keeps the first record whole.
        let bytes = std::fs::read(&path).expect("read");
        std::fs::write(&path, &bytes[..bytes.len() - 3]).expect("write torn");
        assert!(bytes.len() as u64 > journal_one);
        let loaded = KvPrefixStore::with_tier(1 << 20, file(&path), key);
        assert_eq!(loaded.blocks.len(), 3);
        assert_eq!(loaded.blocks[2].tokens, tokens(1, 300)[256..]);
        assert_eq!(
            loaded.blocks[2].bytes, 96,
            "the snapshot's bytes, recomputed"
        );
        let snap = loaded.blocks[2]
            .snap
            .as_ref()
            .expect("the snapshot survived");
        assert_eq!((snap.pos, snap.k[0].1.len()), (300, 64));

        drop(loaded);

        // Another model's key discards the file whole.
        let wrong = KvPrefixStore::with_tier(1 << 20, file(&path), key + 1);
        assert!(wrong.blocks.is_empty());
        drop(wrong);

        // A first record torn mid-way leaves nothing to load, and never a half-read block.
        let bytes = std::fs::read(&path).expect("read");
        std::fs::write(&path, &bytes[..journal_one as usize - 3]).expect("write torn");
        let broken = KvPrefixStore::with_tier(1 << 20, file(&path), key);
        assert!(broken.blocks.is_empty());
        drop(broken);
        std::fs::remove_file(path).ok();
    }
}
