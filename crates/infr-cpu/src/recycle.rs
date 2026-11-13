//! Reuse of the op interpreter's large f32 buffers within one `execute`.
//!
//! Every op used to build its output in a fresh `vec![0f32; n]` and drop the tensor it replaced.
//! A buffer that large is mapped fresh by the system allocator and handed back when dropped, so
//! each op paid a page fault per 4 KiB page on first touch: on Windows, a warm pp512 forward of a
//! dense 3B model spent close to a third of its time faulting (measured with the per-op profile:
//! `GatedAct` 5.6 ms → 1.3 ms, `Add` 1.0 → 0.4 ms once buffers were reused). The graph rewrites
//! the same tensors every layer, so the buffer a replaced tensor frees is the right size for the
//! next layer's: recycling them turns those faults into a reuse.
//!
//! Scheduling only: what an op computes is unchanged, so results are bit-identical. Buffers
//! below [`MIN_RECYCLE_BYTES`] are left to the allocator (its small-block caches already reuse
//! them). What this can hold beyond the old code's memory, and no more: up to
//! [`MAX_IDLE_BYTES`] of idle buffers, and spare capacity inside live tensors of less than
//! `MAX_FIT_RATIO - 1` times each one's size (a buffer is handed out only to a request at least
//! `1 / MAX_FIT_RATIO` of its capacity, and a tensor never shrinks it). Everything is freed when
//! `execute` returns.

/// Smallest buffer worth recycling, in bytes. Below this the system heap serves blocks from its
/// own caches without a fresh mapping.
const MIN_RECYCLE_BYTES: usize = 64 * 1024;

/// Largest capacity, as a multiple of the request, a pooled buffer may have and still be handed
/// out: a tensor never shrinks its buffer, so this bounds the spare capacity a live tensor can
/// hold to `MAX_FIT_RATIO - 1` times its own size.
const MAX_FIT_RATIO: usize = 2;

/// Cap on the bytes held idle in the pool; a buffer given back beyond it is dropped. The peak a
/// warm pp512 forward reached, measured: SmolLM3-3B 26 MiB, Qwen3-30B-A3B 85 MiB,
/// ERNIE-4.5-21B-A3B 166 MiB.
const MAX_IDLE_BYTES: usize = 256 * 1024 * 1024;

/// Free list of f32 buffers, owned by one `execute` call.
#[derive(Default)]
pub(crate) struct Recycler {
    free: Vec<Vec<f32>>,
    idle_bytes: usize,
}

fn bytes_of(v: &Vec<f32>) -> usize {
    v.capacity() * std::mem::size_of::<f32>()
}

impl Recycler {
    /// The smallest idle buffer that holds `n` floats, removed from the pool.
    fn best_fit(&mut self, n: usize) -> Option<Vec<f32>> {
        if n * std::mem::size_of::<f32>() < MIN_RECYCLE_BYTES {
            return None;
        }
        let i = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, v)| v.capacity() >= n && v.capacity() <= MAX_FIT_RATIO * n)
            .min_by_key(|(_, v)| v.capacity())
            .map(|(i, _)| i)?;
        let v = self.free.swap_remove(i);
        self.idle_bytes -= bytes_of(&v);
        Some(v)
    }

    /// `vec![0f32; n]`, from the pool when an idle buffer fits.
    pub(crate) fn zeroed(&mut self, n: usize) -> Vec<f32> {
        match self.best_fit(n) {
            Some(mut v) => {
                v.clear();
                v.resize(n, 0.0);
                v
            }
            None => vec![0f32; n],
        }
    }

    /// A buffer of `n` floats whose contents are unspecified: stale values when it comes from the
    /// pool, zeros when fresh. For an output its op writes in full, which skips `zeroed`'s fill —
    /// a serial pass over the whole buffer on the op loop's thread. Debug builds fill it with NaN
    /// so an element the op forgot to write shows up in the results.
    pub(crate) fn for_overwrite(&mut self, n: usize) -> Vec<f32> {
        match self.best_fit(n) {
            Some(mut v) => {
                // Shrinking never writes; growing within the capacity writes only the new tail.
                v.resize(n, 0.0);
                if cfg!(debug_assertions) {
                    v.fill(f32::NAN);
                }
                v
            }
            // A fresh buffer is poisoned the same way in a debug build.
            None if cfg!(debug_assertions) => vec![f32::NAN; n],
            None => vec![0f32; n],
        }
    }

    /// A copy of `src`, from the pool when an idle buffer fits.
    pub(crate) fn copy_of(&mut self, src: &[f32]) -> Vec<f32> {
        match self.best_fit(src.len()) {
            Some(mut v) => {
                v.clear();
                v.extend_from_slice(src);
                v
            }
            None => src.to_vec(),
        }
    }

    /// Hand a buffer back for reuse (dropped if it is small or the pool is full).
    pub(crate) fn give(&mut self, v: Vec<f32>) {
        let b = bytes_of(&v);
        if b >= MIN_RECYCLE_BYTES && self.idle_bytes + b <= MAX_IDLE_BYTES {
            self.idle_bytes += b;
            self.free.push(v);
        }
    }

    /// `vals[i] = v`, recycling the tensor it replaces.
    pub(crate) fn set(&mut self, vals: &mut [Vec<f32>], i: usize, v: Vec<f32>) {
        let old = std::mem::replace(&mut vals[i], v);
        self.give(old);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BIG: usize = MIN_RECYCLE_BYTES / 4;

    #[test]
    fn a_replaced_tensor_backs_the_next_allocation_zeroed() {
        let mut rc = Recycler::default();
        let mut vals = vec![vec![7.0f32; 2 * BIG]];
        let ptr = vals[0].as_ptr();
        rc.set(&mut vals, 0, Vec::new());
        let v = rc.zeroed(BIG + 3);
        assert_eq!(v.as_ptr(), ptr, "the freed buffer was not reused");
        assert_eq!(v.len(), BIG + 3);
        assert!(
            v.iter().all(|&x| x == 0.0),
            "a reused buffer kept stale values"
        );
        assert_eq!(rc.idle_bytes, 0);
    }

    /// `for_overwrite` hands back the pooled buffer at the requested length, growing or
    /// shrinking it in place; debug builds poison it so an unwritten element is visible.
    #[test]
    fn for_overwrite_reuses_the_buffer_at_the_requested_length() {
        let mut rc = Recycler::default();
        let mut big = vec![7.0f32; 4 * BIG];
        big.truncate(BIG);
        let ptr = big.as_ptr();
        rc.give(big);
        let v = rc.for_overwrite(3 * BIG);
        assert_eq!(v.as_ptr(), ptr, "the pooled buffer was not reused");
        assert_eq!(v.len(), 3 * BIG);
        if cfg!(debug_assertions) {
            assert!(
                v.iter().all(|x| x.is_nan()),
                "a debug build did not poison it"
            );
        }
        // Nothing pooled fits: a fresh buffer, poisoned in a debug build and zeroed otherwise.
        let fresh = rc.for_overwrite(8 * BIG);
        assert_eq!(fresh.len(), 8 * BIG);
        if cfg!(debug_assertions) {
            assert!(
                fresh.iter().all(|x| x.is_nan()),
                "a fresh debug buffer is not poisoned"
            );
        } else {
            assert!(fresh.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn best_fit_picks_the_smallest_buffer_that_holds_the_request() {
        let mut rc = Recycler::default();
        rc.give(vec![0f32; 4 * BIG]);
        rc.give(vec![0f32; 2 * BIG]);
        rc.give(vec![0f32; BIG]);
        let v = rc.copy_of(&vec![3.0f32; BIG + 1]);
        assert_eq!(v.capacity(), 2 * BIG);
        assert!(v.iter().all(|&x| x == 3.0));
        assert_eq!(rc.free.len(), 2);
        // The 4 × BIG buffer is more than `MAX_FIT_RATIO` times a BIG + 1 request: not handed out.
        assert!(
            rc.best_fit(BIG + 1).is_none(),
            "an oversized buffer was handed out"
        );
        assert!(
            rc.best_fit(2 * BIG + 1).is_some(),
            "a buffer within the ratio was refused"
        );
    }

    #[test]
    fn small_buffers_and_overflow_are_dropped() {
        let mut rc = Recycler::default();
        rc.give(vec![0f32; BIG / 2]);
        assert!(rc.free.is_empty(), "a small buffer was pooled");
        assert!(rc.best_fit(BIG / 2).is_none());
        let cap = MAX_IDLE_BYTES / 4;
        // Capacity alone counts, so the cap-sized buffer never touches its pages.
        rc.give(Vec::with_capacity(cap));
        rc.give(vec![0f32; BIG]);
        assert_eq!(rc.free.len(), 1, "the pool grew past its cap");
        assert_eq!(rc.idle_bytes, MAX_IDLE_BYTES);
    }
}
