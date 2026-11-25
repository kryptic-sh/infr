//! Reading one pageable block's bytes off the model file — the bottom tier of the weight pager
//! (`docs/disk-streaming-plan.md` §3.2, §3.5).
//!
//! What a block IS lives in [`crate::pager`] (an opaque `BlockId` plus residency bookkeeping);
//! what a block's bytes ARE lives here: a [`BlockDesc`] naming one or more byte ranges of the
//! model file, in upload order, and a [`BlockIo`] that fills a caller's slot from them.
//!
//! Positioned reads, never a shared cursor: [`FileBlockIo`] reads at an explicit offset
//! (`pread`/`seek_read`), so any number of reader threads share the open files with no seek race
//! and no lock (on Windows each file is open several times over — see [`HANDLES_PER_FILE`]). This
//! is the deliberate alternative to reaching through the GGUF mmap — the page cache evicts by
//! recency, which is the pathological policy for the cyclic sweep a forward pass
//! performs (measured: `docs/perf/results.md`, "Weights that do not fit memory").
//!
//! A `gguf-split` model is several files rather than one, and they are addressed end to end in
//! shard order ([`FileBlockIo::from_handles`]) so a [`BlockExtent`] stays one number. A single-file
//! model is the one-shard case of that, at base 0, and reads exactly as it did before.
//!
//! That freedom is also what makes a block's read CONCURRENT rather than one syscall: an NVMe
//! reaches its bandwidth only with several requests in flight, so one block is split across
//! `IO_FANOUT` positioned reads (see `read_pieces`). This is what puts the tier's reader on
//! even footing with the mapping it replaces, whose faults the kernel already issues in parallel.
//!
//! The file can also change under a live run. A mapping makes that a `SIGBUS` or silently
//! different bytes; explicit reads make it detectable. So the reader is built from the very
//! descriptors the mapping was made from ([`ShardHandle`]), never by re-opening the path, and every
//! [`BlockIo::read_block`] re-`fstat`s each shard it touched against the [`FileStamp`] taken when
//! that file was mapped. A rename into place is not a change — this descriptor stays on the intact
//! old inode, which is exactly what `infr pull` relies on (backlog B30); an in-place rewrite or a
//! truncation is, and it fails the read by name.

use crate::error::{Error, Result};
use crate::pager::BlockId;
use infr_plat::fileio::{read_exact_at, reopen_for_reads};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::SystemTime;

/// One contiguous byte range of the model file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockExtent {
    /// Absolute offset in the model's file address space (the tensor-data region's start already
    /// added in). A `gguf-split` model is several files addressed end to end in shard order — see
    /// [`FileBlockIo::from_handles`] — and a single file is the one-shard case of that, at base 0.
    pub offset: u64,
    pub len: usize,
}

/// A block's identity plus where its bytes live, in the order they must be laid down.
///
/// A fused weight group (qkv, gate+up) lists one extent per component tensor, so the concatenation
/// happens directly into the destination slot and is never materialized on the side — the same
/// property `infr_vulkan::pager::DenseSource`'s segment list has for the mmap path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockDesc {
    pub id: BlockId,
    pub extents: Vec<BlockExtent>,
}

impl BlockDesc {
    /// Total bytes this block occupies in a slot — the sum of its extents. Not stored alongside
    /// them: a stored total is a second source of truth that can disagree with the extents it
    /// claims to describe, and every caller that needs it has the extents in hand.
    pub fn nbytes(&self) -> usize {
        self.extents.iter().map(|e| e.len).sum()
    }
}

/// Fills a slot with one block's bytes. The tier's only I/O surface, so a test can drive the whole
/// pager off an in-memory implementation (see `infr-testkit`) and inject short reads and errors
/// that a real file will not produce on demand.
pub trait BlockIo: Send + Sync {
    /// Write `desc`'s extents, in order, into the front of `dst`.
    ///
    /// `dst` may be longer than the block (slots are a padded stride); bytes past `desc.nbytes()`
    /// are left alone. Fails if `dst` is SHORTER — a truncated block is silent wrong output, so it
    /// is never a partial success.
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()>;
}

/// A weight file's identity at a point in time: the two fields an accidental overwrite moves.
///
/// Read from an open descriptor ([`Self::of`]), never from a path, so it follows the inode that
/// descriptor reads even after the path names another file. Shared by this reader and
/// `infr_gguf::watch::WeightWatch`, which compare the same stamp — the one the model's mapping took
/// when it opened the file.
///
/// What it cannot see: a same-length in-place write whose mtime is then restored. The only way to
/// see that is hashing the whole file per check (backlog B30).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileStamp {
    pub len: u64,
    /// `None` where the platform or filesystem reports no modification time; then only the length
    /// is compared, which is still the half that catches truncation.
    pub mtime: Option<SystemTime>,
}

impl FileStamp {
    /// Stamp the open descriptor `file` — one `fstat` (`GetFileInformationByHandle` on Windows).
    pub fn of(file: &File) -> Result<Self> {
        let md = file.metadata()?;
        Ok(Self {
            len: md.len(),
            mtime: md.modified().ok(),
        })
    }

    /// What moved between `self` and the later stamp `now`, in words, for an error message.
    pub fn describe_change(&self, now: &Self) -> String {
        if now.len != self.len {
            format!("size changed {} -> {} bytes", self.len, now.len)
        } else {
            "modification time moved, size unchanged".to_owned()
        }
    }
}

/// One open file of a model, as the mapping that loaded it holds it — what
/// [`FileBlockIo::from_handles`] reads through instead of re-opening `path`, which may by then name
/// a different file. `infr_gguf::Gguf::shard_handles` hands these out.
#[derive(Clone, Copy, Debug)]
pub struct ShardHandle<'a> {
    /// The descriptor the model was mapped from.
    pub file: &'a File,
    /// The path it was opened by, for error messages only: nothing is opened by it.
    pub path: &'a Path,
    /// The file as it was when the model was mapped. Its length is the one every tensor offset was
    /// computed against.
    pub stamp: FileStamp,
}

/// One file of the model, and where its bytes sit in the address space the extents are written in.
struct ShardIo {
    /// [`HANDLES_PER_FILE`] handles to the same file, handed out in turn by [`Self::file`]. Each
    /// is reopened from the mapping's own descriptor ([`infr_plat::fileio::reopen_for_reads`]),
    /// never from the path, so all of them are the mapped file; on Windows each is a file object
    /// of its own, since one file object's reads run one at a time.
    files: Vec<File>,
    /// Which handle the next read takes.
    next: AtomicUsize,
    /// The file as the mapping found it ([`ShardHandle::stamp`]); every read re-checks it.
    stamp: FileStamp,
    /// Path kept for error messages only — every read and every re-stat goes through `files`.
    path: String,
    /// First offset of the concatenated address space this file serves. Zero for the first shard,
    /// hence for every single-file model.
    base: u64,
}

/// Reads blocks from the open model file (or shard set) with positioned reads.
pub struct FileBlockIo {
    /// The model's files in shard order, each covering `[base, base + stamp.len)` of the address
    /// space [`BlockExtent::offset`] is written in. A single-file model has exactly one.
    shards: Vec<ShardIo>,
}

impl FileBlockIo {
    /// Open a single-file model by path and stamp it now. For a reader beside a model that is
    /// already mapped, [`Self::from_handles`] is the one to use: this cannot know whether `path`
    /// still names the file the mapping holds.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let stamp = FileStamp::of(&file)?;
        Self::from_handles(&[ShardHandle {
            file: &file,
            path,
            stamp,
        }])
    }

    /// Read a model through the descriptors it was mapped from, addressing a `gguf-split` set's
    /// files end to end in the given order.
    ///
    /// Each handle's current stamp must still be the one taken at mapping time. Checked, not
    /// assumed: a shard that grew or shrank since shifts every offset in every LATER shard, and one
    /// rewritten in place holds other weights at the same offsets — neither is a read that fails,
    /// both are reads that silently return the wrong bytes. [`BlockIo::read_block`] keeps checking
    /// the same stamp afterwards.
    pub fn from_handles(handles: &[ShardHandle<'_>]) -> Result<Self> {
        let mut out = Vec::with_capacity(handles.len());
        let mut base = 0u64;
        for h in handles {
            let s = Self::open_one(h, base)?;
            base = base.checked_add(s.stamp.len).ok_or_else(|| {
                Error::Loader("model shard set is larger than u64 bytes".to_string())
            })?;
            out.push(s);
        }
        Ok(Self { shards: out })
    }

    fn open_one(h: &ShardHandle<'_>, base: u64) -> Result<ShardIo> {
        let files = (0..HANDLES_PER_FILE)
            .map(|_| reopen_for_reads(h.file))
            .collect::<std::io::Result<Vec<_>>>()?;
        let now = FileStamp::of(&files[0])?;
        if now != h.stamp {
            return Err(Error::Loader(format!(
                "model file changed after it was loaded, before it was streamed: {} ({}) — \
                 reading it now would return bytes other than the weights the model was loaded \
                 against. Restart against the current file",
                h.path.display(),
                h.stamp.describe_change(&now)
            )));
        }
        Ok(ShardIo {
            files,
            next: AtomicUsize::new(0),
            stamp: h.stamp,
            path: h.path.display().to_string(),
            base,
        })
    }

    /// The model's total length at open — what a caller bounds-checks its extents against. The sum
    /// over a shard set, which is the space those extents are written in.
    pub fn len(&self) -> u64 {
        self.shards.last().map_or(0, |s| s.base + s.stamp.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The shard holding `[offset, offset + len)`, and that range's offset within it.
    ///
    /// A range that runs off the end of its shard is refused rather than continued into the next
    /// one: the shards are separate files laid end to end for addressing only, so bytes never
    /// actually span the join, and a request that claims they do is a corrupt descriptor.
    fn locate(&self, offset: u64, len: usize) -> Result<(&ShardIo, u64)> {
        // Ordered by `base`, each covering its whole file, so the owner is the last shard starting
        // at or before `offset`. One shard at base 0 → always index 0.
        let idx = self
            .shards
            .partition_point(|s| s.base <= offset)
            .saturating_sub(1);
        let shard = self.shards.get(idx).ok_or_else(|| {
            Error::Loader(format!("read at {offset}+{len} of a model with no files"))
        })?;
        let local = offset - shard.base;
        // `checked_add`: a descriptor carrying a near-`u64::MAX` length must be refused, never wrap
        // the sum into a range that looks in-bounds.
        if local
            .checked_add(len as u64)
            .is_none_or(|e| e > shard.stamp.len)
        {
            return Err(Error::Loader(format!(
                "read at {offset}+{len} runs past the end of {} ({} bytes at offset {})",
                shard.path, shard.stamp.len, shard.base
            )));
        }
        Ok((shard, local))
    }
}

impl ShardIo {
    /// The handle the next read goes through: each in turn, so that concurrent reads land on
    /// different handles (see `files`).
    fn file(&self) -> &File {
        &self.files[self.next.fetch_add(1, Ordering::Relaxed) % self.files.len()]
    }

    /// What changed in this file since it was mapped, or `None` while it still stamps the same.
    /// One `fstat`: every handle was reopened from the mapping's descriptor, so all of them are
    /// the same file and one speaks for all.
    fn changed(&self) -> Result<Option<String>> {
        let now = FileStamp::of(&self.files[0])?;
        Ok((now != self.stamp).then(|| self.stamp.describe_change(&now)))
    }
}

impl BlockIo for FileBlockIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        let need = desc.nbytes();
        if dst.len() < need {
            return Err(Error::backend(format!(
                "block {} needs {need} bytes, slot holds {}",
                desc.id,
                dst.len()
            )));
        }
        // Resolve every piece to the file that holds it BEFORE any read runs, so a descriptor
        // naming bytes outside the model fails by name instead of being read from a neighbour.
        let pieces = plan_pieces(&desc.extents)
            .into_iter()
            .map(|p| {
                let (shard, local) = self.locate(p.offset, p.len)?;
                Ok((
                    shard,
                    BlockExtent {
                        offset: local,
                        len: p.len,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let read = read_pieces(&pieces, &mut dst[..need]).map_err(|(shard, e, err)| {
            format!(
                "reading block {} at {}+{} of {}: {err}",
                desc.id, e.offset, e.len, shard.path
            )
        });
        // Stamped AFTER the read, so a change that landed before it or while it ran is seen: a
        // file rewritten under this read may have handed back a mix of old and new bytes that
        // reads perfectly well. A failed read is reported alongside, since a truncation shows up
        // first as a short read and the change is what explains it. The cost is one `fstat` per
        // shard touched (pieces come in extent order, so skipping a repeat of the previous shard
        // checks each once in the usual case): noise beside a weight block of a megabyte or
        // more, but not beside a block of a few KiB (backlog B35 has the measurement).
        let also = || {
            read.as_ref()
                .err()
                .map_or(String::new(), |e| format!("; {e}"))
        };
        let mut prev: Option<&ShardIo> = None;
        for &(shard, _) in &pieces {
            if prev.is_some_and(|p| std::ptr::eq(p, shard)) {
                continue;
            }
            prev = Some(shard);
            let what = shard.changed().map_err(|e| {
                Error::Loader(format!(
                    "could not re-check {} after reading block {}: {e}{}",
                    shard.path,
                    desc.id,
                    also()
                ))
            })?;
            if let Some(what) = what {
                return Err(Error::Loader(format!(
                    "model file changed while it was being streamed: {} ({what}) — block {} and \
                     every weight read after it would not match the ones already loaded. Restart \
                     against the current file{}",
                    shard.path,
                    desc.id,
                    also()
                )));
            }
        }
        read.map_err(Error::Loader)
    }
}

/// How many positioned reads one block is split across.
///
/// A drive reaches its bandwidth on queue depth, not on request size: measured on this workspace's
/// NVMe over 16-128 MB blocks, one read sustains 1.2-1.5 GB/s and the device tops out at
/// 2.2 GB/s, which two to four concurrent reads already reach — eight and sixteen buy nothing and
/// cost threads. The gap between those two figures is the whole reason this exists; a serial reader
/// loses to the mapping it replaces, whose faults the kernel issues in parallel for free.
///
/// Those figures are from Linux on NVMe, which is also where the end-to-end gain was measured
/// (`docs/perf/results.md`). On Windows the same holds only with a handle per concurrent read:
/// `seek_read` issues `ReadFile` on a handle opened for synchronous I/O, and Windows runs such a
/// handle's requests one at a time. Measured on the Windows box's NVMe with random 2.75 MB reads
/// of a 98 GB file: sixteen threads sharing one handle read at the speed of one (0.65 GB/s), while
/// four threads with a handle each reached 1.05 GB/s and eight or sixteen no more. So on Windows
/// every shard is opened this many times ([`HANDLES_PER_FILE`]). Unverified on macOS; see backlog
/// B35.
///
/// Public because a caller that issues several block reads at once — the CPU backend pinning an
/// op's routed experts — sizes its concurrency to it: past this many reads in flight there is no
/// handle left to run them on.
pub const IO_FANOUT: usize = 4;

/// How many times each model file is opened, so that this many positioned reads can run on it at
/// once.
///
/// - **Windows: [`IO_FANOUT`].** `seek_read` is `ReadFile` on a handle opened for synchronous
///   I/O, and Windows runs one such handle's requests one at a time (measured: [`IO_FANOUT`]'s
///   doc), so concurrent reads need a handle each.
/// - **Linux and the other Unixes, macOS included: one.** `pread` carries its own offset and the
///   kernel runs concurrent `pread`s on one descriptor in parallel; on Linux that is what the
///   [`IO_FANOUT`] figures were measured with. On macOS the same is assumed and not measured
///   (backlog B35). More descriptors there would buy nothing.
#[cfg(windows)]
pub const HANDLES_PER_FILE: usize = IO_FANOUT;
// Windows serializes one handle's reads, so one handle there would make every read serial.
#[cfg(windows)]
const _: () = assert!(HANDLES_PER_FILE > 1);
/// How many times each model file is opened — see the Windows definition for why it differs.
#[cfg(not(windows))]
pub const HANDLES_PER_FILE: usize = 1;

/// A piece below this is not worth its own thread — the fanout is for keeping the drive busy, and
/// small blocks (the norm/bias tensors of a fused group) are latency-bound rather than
/// bandwidth-bound, where an extra thread is pure overhead.
const MIN_CHUNK: usize = 4 << 20;

/// Cut `extents` into the file ranges one block's concurrent reads will each cover, in the order
/// their bytes appear in the destination slot.
///
/// Aims for [`IO_FANOUT`] pieces overall, never smaller than [`MIN_CHUNK`]. Extent boundaries can
/// push the count slightly above the fanout on a fused group; that is harmless — the measured
/// bandwidth curve is flat from 2 to 16 concurrent reads — and keeping every piece inside one
/// extent is what lets each be a single contiguous range of both the file and the slot.
///
/// Separate from the reading so the split can be checked on its own: a piece list that silently
/// stayed one element would leave every read serial and every byte still correct.
fn plan_pieces(extents: &[BlockExtent]) -> Vec<BlockExtent> {
    let total: usize = extents.iter().map(|e| e.len).sum();
    let target = total.div_ceil(IO_FANOUT).max(MIN_CHUNK);
    let mut out = Vec::new();
    for e in extents {
        let mut at = 0usize;
        while at < e.len {
            let n = target.min(e.len - at);
            out.push(BlockExtent {
                offset: e.offset + at as u64,
                len: n,
            });
            at += n;
        }
    }
    out
}

/// Fill `dst` with `pieces`, in order, using up to [`IO_FANOUT`] concurrent positioned reads.
///
/// Each piece is already resolved to the shard that holds it and carries that shard's LOCAL offset
/// (see [`FileBlockIo::locate`]); their total must be exactly `dst.len()`, which the caller has
/// already length-checked. On failure returns the piece that failed and its shard alongside the
/// error — a sub-range of one extent, which names the failing bytes more precisely than the extent
/// enclosing them would.
///
/// Ordering is preserved by construction rather than by sequencing the reads: each piece is handed
/// a disjoint `&mut` sub-slice of `dst` carved at the position that piece belongs at, so the reads
/// may complete in any order and still land where they must. Nothing is shared between them — the
/// files are read at explicit offsets, so there is no cursor to race on.
fn read_pieces<'a>(
    pieces: &[(&'a ShardIo, BlockExtent)],
    dst: &mut [u8],
) -> std::result::Result<(), (&'a ShardIo, BlockExtent, std::io::Error)> {
    debug_assert_eq!(
        pieces.iter().map(|(_, e)| e.len).sum::<usize>(),
        dst.len(),
        "dst must be exactly the pieces' total"
    );
    let mut rest = dst;
    let mut placed: Vec<(&ShardIo, BlockExtent, &mut [u8])> = Vec::with_capacity(pieces.len());
    for &(shard, e) in pieces {
        let (mine, tail) = rest.split_at_mut(e.len);
        rest = tail;
        placed.push((shard, e, mine));
    }
    debug_assert!(rest.is_empty(), "the pieces must cover dst exactly");

    // The calling thread takes the last piece rather than parking on a join, so a single-piece
    // block spawns nothing at all and the common case costs no threads.
    let Some((last_shard, last, last_dst)) = placed.pop() else {
        return Ok(());
    };
    let mut first_err = None;
    std::thread::scope(|s| {
        let handles: Vec<_> = placed
            .into_iter()
            .map(|(shard, e, buf)| {
                s.spawn(move || {
                    read_exact_at(shard.file(), buf, e.offset).map_err(|err| (shard, e, err))
                })
            })
            .collect();
        first_err = read_exact_at(last_shard.file(), last_dst, last.offset)
            .map_err(|err| (last_shard, last, err))
            .err();
        for h in handles {
            // A panicking reader is a bug in this function, not an I/O condition: propagate it
            // rather than reporting a block that was never filled as read, with its own message.
            if let Err(e) = h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)) {
                first_err.get_or_insert(e);
            }
        }
    });
    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A file of `n` bytes whose byte at index `i` is `i as u8` — so any wrong offset, any
    /// mis-ordered extent and any short read shows up as a value mismatch rather than as a length
    /// that happens to be right. The pattern repeats every 256 bytes, so a test that must
    /// distinguish two ranges has to pick offsets that differ modulo 256.
    fn ramp_file(n: usize) -> (tempfile::NamedTempFile, Vec<u8>) {
        let bytes: Vec<u8> = (0..n).map(|i| i as u8).collect();
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(&bytes).expect("write");
        f.flush().expect("flush");
        (f, bytes)
    }

    #[test]
    fn reads_one_extent_at_its_offset() {
        let (f, bytes) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        let desc = BlockDesc {
            id: 7,
            extents: vec![BlockExtent {
                offset: 1000,
                len: 256,
            }],
        };
        let mut dst = vec![0u8; 256];
        io.read_block(&desc, &mut dst).expect("read");
        assert_eq!(dst, bytes[1000..1256]);
    }

    /// A fused group: the extents must land back to back in the order listed, which is what makes
    /// the concatenation free. Reversing the two extents here must NOT produce the same bytes.
    #[test]
    fn concatenates_extents_in_order() {
        let (f, bytes) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        // Offsets differ modulo 256, so the two ranges hold different bytes and the reversed
        // order below cannot coincidentally match.
        let fwd = BlockDesc {
            id: 1,
            extents: vec![
                BlockExtent {
                    offset: 2048,
                    len: 64,
                },
                BlockExtent {
                    offset: 100,
                    len: 64,
                },
            ],
        };
        let mut dst = vec![0u8; 128];
        io.read_block(&fwd, &mut dst).expect("read");
        assert_eq!(&dst[..64], &bytes[2048..2112]);
        assert_eq!(&dst[64..], &bytes[100..164]);

        let rev = BlockDesc {
            id: 1,
            extents: fwd.extents.iter().rev().copied().collect(),
        };
        let mut other = vec![0u8; 128];
        io.read_block(&rev, &mut other).expect("read");
        assert_ne!(dst, other, "extent order must decide the layout");
    }

    /// A slot is a PADDED stride, so a longer destination is normal and the tail must be left
    /// alone — the pager reuses slots, and clobbering past the block would corrupt nothing today
    /// but would hide a sizing bug tomorrow.
    #[test]
    fn a_longer_slot_keeps_its_tail() {
        let (f, _) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        let desc = BlockDesc {
            id: 0,
            extents: vec![BlockExtent { offset: 0, len: 32 }],
        };
        let mut dst = vec![0xAAu8; 64];
        io.read_block(&desc, &mut dst).expect("read");
        assert_eq!(&dst[32..], &[0xAAu8; 32], "padding was overwritten");
    }

    /// A slot too small is a caller bug that must fail, not truncate: a short block is wrong
    /// output with no error attached.
    #[test]
    fn a_short_slot_is_rejected() {
        let (f, _) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        let desc = BlockDesc {
            id: 3,
            extents: vec![BlockExtent {
                offset: 0,
                len: 100,
            }],
        };
        let mut dst = vec![0u8; 99];
        let err = io.read_block(&desc, &mut dst).expect_err("must reject");
        assert!(
            err.to_string().contains("slot holds 99"),
            "unexpected error: {err}"
        );
    }

    /// An extent naming bytes the file does not have must error rather than silently leaving the
    /// slot half-filled. It is refused by `locate` before any read is issued: with a shard set the
    /// next file's bytes sit immediately after this one in the address space, so "runs off the end
    /// of shard k" has to be a refusal and not something the reader could paper over.
    #[test]
    fn reading_past_the_end_errors() {
        let (f, _) = ramp_file(512);
        let io = FileBlockIo::open(f.path()).expect("open");
        let desc = BlockDesc {
            id: 4,
            extents: vec![BlockExtent {
                offset: 256,
                len: 512,
            }],
        };
        let mut dst = vec![0u8; 512];
        let err = io.read_block(&desc, &mut dst).expect_err("must reject");
        assert!(
            err.to_string().contains("runs past the end of"),
            "unexpected: {err}"
        );
    }

    /// A file opened and stamped the way `infr_gguf::Gguf::open` opens each file it maps, so a
    /// test can hand [`FileBlockIo::from_handles`] the same thing the loader does.
    struct Mapped {
        file: File,
        path: std::path::PathBuf,
        stamp: FileStamp,
    }

    impl Mapped {
        fn open(path: &Path) -> Self {
            let file = File::open(path).expect("open");
            let stamp = FileStamp::of(&file).expect("stamp");
            Self {
                file,
                path: path.to_path_buf(),
                stamp,
            }
        }

        fn handle(&self) -> ShardHandle<'_> {
            ShardHandle {
                file: &self.file,
                path: &self.path,
                stamp: self.stamp,
            }
        }
    }

    /// Overwrite `path`'s bytes in place, at the same length, and move its mtime well clear of the
    /// original: a test writes fast enough to land in the same mtime tick on a coarse filesystem,
    /// which would make a same-size rewrite invisible for a reason that has nothing to do with the
    /// code under test.
    fn rewrite_in_place(path: &Path, bytes: &[u8]) {
        let f = File::options().write(true).open(path).expect("reopen");
        (&f).write_all(bytes).expect("rewrite");
        let later = SystemTime::now() + std::time::Duration::from_secs(60);
        f.set_times(std::fs::FileTimes::new().set_modified(later))
            .expect("bump mtime");
    }

    /// Put a file of `bytes` at `path` by renaming it over whatever is there — the way `infr pull`
    /// installs a download. The old inode lives on for anyone holding it open.
    ///
    /// Returns whether the rename happened. Unix always allows it. Windows may refuse to replace a
    /// file that is open or mapped, and that refusal is a pass in its own right — the OS is
    /// protecting the file the model reads — so it comes back as `false` (after checking the path
    /// still holds the original), never as a failure. Any other error fails the test.
    fn rename_into_place(path: &Path, bytes: &[u8]) -> bool {
        let before = std::fs::read(path).expect("read the original");
        let next = path.with_extension("next");
        std::fs::write(&next, bytes).expect("write");
        match std::fs::rename(&next, path) {
            Ok(()) => true,
            Err(e) if cfg!(windows) && is_windows_refusal(&e) => {
                assert_eq!(
                    std::fs::read(path).expect("read the path"),
                    before,
                    "a refused rename must leave the original in place"
                );
                false
            }
            Err(e) => panic!("rename into place: {e}"),
        }
    }

    /// The errors Windows answers a rename over an open or mapped file with: access denied (which
    /// std reports as `PermissionDenied`), a sharing violation, or a user-mapped section.
    fn is_windows_refusal(e: &std::io::Error) -> bool {
        const ERROR_SHARING_VIOLATION: i32 = 32;
        const ERROR_USER_MAPPED_FILE: i32 = 1224;
        e.kind() == std::io::ErrorKind::PermissionDenied
            || matches!(
                e.raw_os_error(),
                Some(ERROR_SHARING_VIOLATION | ERROR_USER_MAPPED_FILE)
            )
    }

    /// Read `desc` once per handle, so the round-robin hands every handle one read and a wrong
    /// one cannot hide behind the first; each read must be `want`.
    fn read_on_every_handle(io: &FileBlockIo, desc: &BlockDesc, want: &[u8]) {
        for i in 0..HANDLES_PER_FILE {
            let mut dst = vec![0u8; want.len()];
            io.read_block(desc, &mut dst)
                .unwrap_or_else(|e| panic!("read {i}: {e}"));
            assert_eq!(
                dst, want,
                "read {i} returned other bytes than the loaded file"
            );
        }
    }

    fn one_extent(id: BlockId, offset: u64, len: usize) -> BlockDesc {
        BlockDesc {
            id,
            extents: vec![BlockExtent { offset, len }],
        }
    }

    /// The baseline every other file-change test departs from: a file nobody touches reads the
    /// same bytes, read after read, with no error.
    #[test]
    fn an_unchanged_file_reads_the_same_bytes_every_time() {
        let (f, bytes) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        let desc = one_extent(1, 1000, 256);
        for _ in 0..3 {
            let mut dst = vec![0u8; 256];
            io.read_block(&desc, &mut dst)
                .expect("an untouched file must read");
            assert_eq!(dst, bytes[1000..1256]);
        }
    }

    /// An extent that WAS in range when the model was loaded, on a file since truncated below it.
    /// `locate` passes (it checks the length stamped at open), the read hits EOF, and the error
    /// must say both things: that the file changed, and the short read it caused.
    #[test]
    fn a_file_truncated_under_the_block_fails_it_by_name() {
        let (f, _) = ramp_file(512);
        let io = FileBlockIo::open(f.path()).expect("open");
        File::create(f.path())
            .expect("truncate")
            .set_len(64)
            .expect("set_len");
        let mut dst = vec![0u8; 128];
        let err = io
            .read_block(&one_extent(4, 256, 128), &mut dst)
            .expect_err("must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("changed while it was being streamed") && msg.contains("512 -> 64"),
            "unexpected: {msg}"
        );
        assert!(
            msg.contains("short read"),
            "the read's own error is lost: {msg}"
        );
    }

    /// A truncation that leaves this block's bytes in place still fails it: the bytes it would
    /// return are the old ones, but the model they belong to is gone, and the next block may not
    /// be there at all. Silently serving this read is how a half-replaced model gets used.
    #[test]
    fn a_truncation_that_spares_the_block_still_fails_it() {
        let (f, _) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        File::options()
            .write(true)
            .open(f.path())
            .expect("reopen")
            .set_len(2048)
            .expect("truncate");
        let mut dst = vec![0u8; 64];
        let err = io
            .read_block(&one_extent(5, 100, 64), &mut dst)
            .expect_err("a truncated file must fail the read");
        assert!(
            err.to_string().contains("size changed 4096 -> 2048"),
            "unexpected: {err}"
        );
    }

    /// The `cp new.gguf live.gguf` of a same-size file: every offset still in range, every byte
    /// different. Only the mtime moved, so that half of the stamp is load-bearing on its own.
    #[test]
    fn a_same_size_rewrite_fails_the_read() {
        let (f, _) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        rewrite_in_place(f.path(), &[0xEEu8; 4096]);
        let mut dst = vec![0u8; 64];
        let err = io
            .read_block(&one_extent(6, 100, 64), &mut dst)
            .expect_err("a rewritten file must fail the read");
        // The whole message, as for the construction-time refusal below.
        assert_eq!(
            err.to_string(),
            format!(
                "loader: model file changed while it was being streamed: {} (modification time \
                 moved, size unchanged) — block 6 and every weight read after it would not match \
                 the ones already loaded. Restart against the current file",
                f.path().display()
            )
        );
    }

    /// The design decision, pinned: a rename into place is not a change. This reader holds the
    /// old inode, which still has exactly the bytes the model was loaded from — the same bytes the
    /// mapping serves — so it keeps reading them, with no error. Failing here would be a false
    /// alarm on the one replacement `infr pull` itself performs.
    #[test]
    fn a_rename_into_place_keeps_reading_the_loaded_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("m.gguf");
        let bytes: Vec<u8> = (0..4096usize).map(|i| i as u8).collect();
        std::fs::write(&path, &bytes).expect("write");
        let io = FileBlockIo::open(&path).expect("open");
        // Either way the reader must keep serving the loaded bytes: replaced, it holds the old
        // inode; refused (Windows), the path still names them.
        rename_into_place(&path, &[0xEEu8; 4096]);
        read_on_every_handle(&io, &one_extent(7, 100, 64), &bytes[100..164]);
    }

    /// The reader is built from the descriptor the model was mapped from, not from its path. A
    /// same-size replacement renamed in between leaves the path naming other bytes; a reader that
    /// re-opened it would serve those, at offsets computed against the old file, as weights.
    #[test]
    fn from_handles_reads_the_mapped_file_not_its_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("m.gguf");
        let bytes: Vec<u8> = (0..4096usize).map(|i| i as u8).collect();
        std::fs::write(&path, &bytes).expect("write");
        let mapped = Mapped::open(&path);
        let replaced = rename_into_place(&path, &[0xEEu8; 4096]);
        if replaced {
            assert_eq!(
                std::fs::read(&path).expect("read the path"),
                vec![0xEEu8; 4096],
                "the path must name the replacement, or this test proves nothing"
            );
        }

        let io = FileBlockIo::from_handles(&[mapped.handle()]).expect("the mapped file is intact");
        read_on_every_handle(&io, &one_extent(8, 100, 64), &bytes[100..164]);
    }

    /// A file rewritten in place between the mapping and the reader: the descriptor is the right
    /// one, but its bytes are not the ones the model was loaded against. Refused at construction.
    #[test]
    fn from_handles_refuses_a_file_changed_since_it_was_mapped() {
        let (f, _) = ramp_file(4096);
        let mapped = Mapped::open(f.path());
        rewrite_in_place(f.path(), &[0xEEu8; 4096]);
        let msg = match FileBlockIo::from_handles(&[mapped.handle()]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a file rewritten since it was mapped must be refused"),
        };
        // The whole message, so a literal that loses a line continuation (and with it gains a run
        // of spaces) fails here rather than reaching a user.
        assert_eq!(
            msg,
            format!(
                "loader: model file changed after it was loaded, before it was streamed: {} \
                 (modification time moved, size unchanged) — reading it now would return bytes \
                 other than the weights the model was loaded against. Restart against the current \
                 file",
                f.path().display()
            )
        );
    }

    /// Two shards addressed end to end: an extent past the first file's length must be read from
    /// the SECOND file, at its own offset. Both files hold the same ramp pattern, so an extent
    /// resolved against the wrong file would still return readable bytes of the right length —
    /// what tells them apart here is that shard 2's ramp is offset by one.
    #[test]
    fn from_handles_reads_across_the_shard_join() {
        let (a, a_bytes) = ramp_file(4096);
        let b_bytes: Vec<u8> = (0..4096usize).map(|i| (i + 1) as u8).collect();
        let mut b = tempfile::NamedTempFile::new().expect("temp file");
        b.write_all(&b_bytes).expect("write");
        b.flush().expect("flush");

        let (ma, mb) = (Mapped::open(a.path()), Mapped::open(b.path()));
        let io = FileBlockIo::from_handles(&[ma.handle(), mb.handle()]).expect("open set");
        assert_eq!(
            io.len(),
            8192,
            "the set is addressed as one 8192-byte space"
        );

        // One extent inside shard 1, one inside shard 2, in one block — the fused-group shape.
        let desc = BlockDesc {
            id: 9,
            extents: vec![
                BlockExtent {
                    offset: 100,
                    len: 64,
                },
                BlockExtent {
                    offset: 4096 + 100,
                    len: 64,
                },
            ],
        };
        let mut dst = vec![0u8; 128];
        io.read_block(&desc, &mut dst).expect("read");
        assert_eq!(&dst[..64], &a_bytes[100..164]);
        assert_eq!(&dst[64..], &b_bytes[100..164]);
    }

    /// A read that would run off the end of one shard is refused, never continued into the next —
    /// the shards are separate files laid end to end for ADDRESSING, and no tensor spans the join.
    #[test]
    fn a_read_never_spans_the_shard_join() {
        let (a, _) = ramp_file(4096);
        let (b, _) = ramp_file(4096);
        let (ma, mb) = (Mapped::open(a.path()), Mapped::open(b.path()));
        let io = FileBlockIo::from_handles(&[ma.handle(), mb.handle()]).expect("open set");
        let mut dst = vec![0u8; 128];
        let err = io
            .read_block(&one_extent(3, 4032, 128), &mut dst)
            .expect_err("must reject");
        assert!(
            err.to_string().contains("runs past the end of"),
            "unexpected: {err}"
        );
    }

    /// A shard whose length is no longer the one the weights were loaded against shifts every
    /// offset in every LATER shard — not a read that fails, a read that returns other weights. It
    /// must be refused at open, naming the file and both lengths.
    #[test]
    fn from_handles_refuses_a_shard_of_the_wrong_length() {
        let (a, _) = ramp_file(4096);
        let (b, _) = ramp_file(4096);
        let (mut ma, mb) = (Mapped::open(a.path()), Mapped::open(b.path()));
        ma.stamp.len = 2048;
        let msg = match FileBlockIo::from_handles(&[ma.handle(), mb.handle()]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a resized shard must be refused"),
        };
        assert!(
            msg.contains("size changed 2048 -> 4096")
                && msg.contains(&ma.path.display().to_string()),
            "unexpected: {msg}"
        );
    }

    /// The split itself, checked without touching a disk: a block big enough to be worth
    /// parallelising must come back as several pieces that tile its extent exactly and in order.
    /// Asserting only on the bytes a read produced would pass just as well if the fanout silently
    /// collapsed to one piece and every read stayed serial.
    #[test]
    fn a_big_block_is_cut_into_concurrent_pieces() {
        // Sized against MIN_CHUNK alone. Deriving it from IO_FANOUT and then asserting the count
        // against IO_FANOUT would move both sides of the comparison together, so it would hold for
        // any fanout — including 1, which is the case this exists to rule out.
        let len = 16 * MIN_CHUNK;
        let pieces = plan_pieces(&[BlockExtent { offset: 100, len }]);
        assert!(
            pieces.len() > 1,
            "a {len}-byte block must be split to be read concurrently, got {}",
            pieces.len()
        );
        let mut at = 100u64;
        for p in &pieces {
            assert_eq!(p.offset, at, "pieces must tile the extent in order");
            at += p.len as u64;
        }
        assert_eq!(at, 100 + len as u64, "pieces must cover the whole extent");
        assert!(
            pieces.iter().all(|p| p.len >= MIN_CHUNK),
            "no piece may fall below the minimum: {pieces:?}"
        );
    }

    /// The other half of the same rule: a block that is only worth one read must not be split, or
    /// every small tensor pays thread-spawn overhead to no purpose.
    ///
    /// Sized as a concrete small tensor — 64 KiB, the order of a norm vector — rather than as
    /// `MIN_CHUNK - 1`, which sits below the threshold by construction and so would hold no matter
    /// how small the minimum became.
    #[test]
    fn a_small_block_stays_one_read() {
        let pieces = plan_pieces(&[BlockExtent {
            offset: 0,
            len: 64 << 10,
        }]);
        assert_eq!(pieces.len(), 1, "small block must not be split: {pieces:?}");
    }

    /// A piece never straddles an extent boundary — it could not, since the two sides come from
    /// different offsets in the file. A fused group of large components splits within each.
    #[test]
    fn pieces_never_straddle_an_extent_boundary() {
        // Lengths deliberately NOT a whole multiple of the piece size: if they divided evenly, the
        // last piece of each extent would land exactly on the boundary and a splitter that ignored
        // the extent's end would produce the same list as one that respected it.
        let a = BlockExtent {
            offset: 0,
            len: 5 * MIN_CHUNK + 12345,
        };
        let b = BlockExtent {
            offset: 1 << 30,
            len: 5 * MIN_CHUNK + 999,
        };
        let pieces = plan_pieces(&[a, b]);
        for p in &pieces {
            let end = p.offset + p.len as u64;
            let within_a = p.offset >= a.offset && end <= a.offset + a.len as u64;
            let within_b = p.offset >= b.offset && end <= b.offset + b.len as u64;
            assert!(within_a || within_b, "piece {p:?} straddles a boundary");
        }
        assert_eq!(
            pieces.iter().map(|p| p.len).sum::<usize>(),
            a.len + b.len,
            "pieces must cover both extents"
        );
    }

    /// Concurrent reads of one shard go through different handles. On Windows a single handle runs
    /// its reads one at a time, so a reader that kept one would be serial however many threads
    /// called it — and still return the right bytes, which is why this checks the handles. Where
    /// one descriptor serves concurrent reads (`HANDLES_PER_FILE` is 1) the check is that it is
    /// the only one.
    #[test]
    fn each_shard_hands_out_its_handles_in_turn() {
        let (f, _) = ramp_file(4096);
        let io = FileBlockIo::open(f.path()).expect("open");
        let shard = &io.shards[0];
        assert_eq!(shard.files.len(), HANDLES_PER_FILE);
        let picked: std::collections::HashSet<*const File> = (0..HANDLES_PER_FILE)
            .map(|_| shard.file() as *const File)
            .collect();
        assert_eq!(
            picked.len(),
            HANDLES_PER_FILE,
            "a read reused a handle another read held"
        );
    }

    /// End to end over a real file, at a size that actually engages the concurrent path: the bytes
    /// must be identical to what a serial read would have produced, and the two extents must still
    /// land in the order they were listed even though their pieces complete in any order.
    #[test]
    fn a_concurrently_read_block_matches_its_file() {
        // Two extents, each several pieces, at offsets that differ modulo 256 so a swapped piece
        // shows up as a value mismatch rather than a coincidentally equal ramp.
        let each = 3 * MIN_CHUNK;
        let (f, bytes) = ramp_file(2 * each + 1000);
        let io = FileBlockIo::open(f.path()).expect("open");
        let desc = BlockDesc {
            id: 9,
            extents: vec![
                BlockExtent {
                    offset: each as u64 + 1000,
                    len: each,
                },
                BlockExtent {
                    offset: 0,
                    len: each,
                },
            ],
        };
        assert!(
            plan_pieces(&desc.extents).len() > 1,
            "this test must engage the concurrent path"
        );
        let mut dst = vec![0u8; 2 * each];
        io.read_block(&desc, &mut dst).expect("read");
        assert_eq!(&dst[..each], &bytes[each + 1000..2 * each + 1000]);
        assert_eq!(&dst[each..], &bytes[..each]);
    }

    /// A failure in ANY piece must surface, including one the calling thread did not read itself —
    /// a spawned reader whose error was dropped would leave a partly-filled slot reported as read.
    ///
    /// The FIRST extent is the one that runs off the end and the last is in range, because the
    /// calling thread keeps the last piece: a failure placed at the end would be caught by the
    /// caller's own read and would say nothing about whether a spawned reader's error is joined.
    ///
    /// The file is TRUNCATED after opening so the descriptor stays inside the length `locate`
    /// checks (an extent naming bytes past the stamped length never reaches a reader at all) while
    /// the read itself still hits EOF — which is the condition a spawned reader has to report.
    #[test]
    fn a_failing_piece_fails_the_block() {
        let each = 2 * MIN_CHUNK;
        let (f, _) = ramp_file(2 * each);
        let io = FileBlockIo::open(f.path()).expect("open");
        std::fs::File::options()
            .write(true)
            .open(f.path())
            .expect("reopen")
            .set_len(each as u64)
            .expect("truncate");
        let desc = BlockDesc {
            id: 11,
            extents: vec![
                BlockExtent {
                    offset: each as u64, // starts exactly at EOF
                    len: each,
                },
                BlockExtent {
                    offset: 0,
                    len: each,
                },
            ],
        };
        let pieces = plan_pieces(&desc.extents);
        let last = pieces.last().expect("pieces");
        assert!(
            last.offset + last.len as u64 <= each as u64,
            "the caller's own piece must be the readable one, else this proves nothing"
        );
        let mut dst = vec![0u8; 2 * each];
        let err = io.read_block(&desc, &mut dst).expect_err("must reject");
        assert!(err.to_string().contains("short read"), "unexpected: {err}");
    }

    #[test]
    fn nbytes_is_the_extent_sum() {
        let desc = BlockDesc {
            id: 0,
            extents: vec![
                BlockExtent { offset: 0, len: 10 },
                BlockExtent {
                    offset: 100,
                    len: 5,
                },
            ],
        };
        assert_eq!(desc.nbytes(), 15);
    }
}
