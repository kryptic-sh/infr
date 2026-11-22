//! The split-K chunk plan of decode attention: how many keys one `(head, chunk)` workgroup scans.
//!
//! Two call paths record decode attention, and they must cut a span the same way, because a
//! chunk boundary is a point where the online-softmax partials are merged and moving one changes
//! the result's last bits:
//!
//! - the STATIC path (`INFR_SEAM_NO_REPLAY`, and every batched forward) pushes the chunk as a
//!   constant: [`static_chunk`];
//! - the record-once REPLAY path bakes a floor from the cache's capacity ([`baked_min_chunk`])
//!   and lets `attn_live.comp`, `attn_decode.comp` and `attn_partial.comp` (SELF_CHUNK) derive
//!   the chunk from the live key count each token. The tests here hold a host copy of that
//!   expression against the shaders' text and against [`static_chunk`].
//!
//! The floor is the tuning parameter `attn_split.decode_min_chunk`. An MTP verify batch scores
//! the positions a decode step would, so it takes the decode floor too (`uses_decode_floor`).

use infr_core::tier::{self, AttnSplitCfg};

use crate::adapter::{ATTN_MAX_CHUNKS, ATTN_SPLIT};

/// Decode's split policy at the floor `min_chunk`: [`ATTN_SPLIT`]'s shape with that floor.
fn decode_split(min_chunk: usize) -> AttnSplitCfg {
    AttnSplitCfg {
        min_chunk,
        ..ATTN_SPLIT
    }
}

/// Whether a static attention of `rows` query rows is cut with the decode floor: a decode row,
/// and an MTP verify batch, whose rows must match the decode rows at the same positions.
pub(crate) fn uses_decode_floor(rows: usize, mtp_verify: bool) -> bool {
    rows == 1 || mtp_verify
}

/// The chunk the static path records for `span` keys at the decode floor `min_chunk`.
pub(crate) fn static_chunk(span: usize, min_chunk: usize) -> usize {
    tier::adaptive_chunk(span, &decode_split(min_chunk))
}

/// The floor a record-once replay bakes for a cache of `cap_rows` rows: `min_chunk`, raised only
/// when the capacity would otherwise need more chunks than the combine can index.
pub(crate) fn baked_min_chunk(cap_rows: usize, min_chunk: usize) -> usize {
    tier::baked_chunk(cap_rows, ATTN_MAX_CHUNKS, min_chunk)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contributor's floor (PR 105) and the one every device ran before it.
    const FLOORS: [usize; 2] = [16, 64];

    /// What the replay shaders compute per token from the baked floor and the live span.
    fn live_chunk(baked: usize, span: usize) -> usize {
        baked.max((span / 32).clamp(1, 512))
    }

    /// The key ranges a plan of `chunk` keys cuts `[lo, kv_len)` into, as the kernels index them
    /// (`j0 = lo + c * chunk`, clipped to the live keys).
    fn ranges(lo: usize, kv_len: usize, chunk: usize) -> Vec<std::ops::Range<usize>> {
        (0..tier::n_chunks(kv_len - lo, chunk))
            .map(|c| lo + c * chunk..(lo + (c + 1) * chunk).min(kv_len))
            .collect()
    }

    /// The plan, as literals. A change of the floor, of the target chunk count or of the ceiling
    /// has to face these.
    #[test]
    fn the_plan_is_pinned_at_both_floors() {
        // (span, chunk at floor 16, chunk at floor 64)
        for (span, at16, at64) in [
            (1usize, 16usize, 64usize),
            (17, 16, 64),
            (100, 16, 64),
            (511, 16, 64),
            (512, 16, 64),
            (544, 17, 64),
            (1024, 32, 64),
            (2047, 63, 64),
            (2048, 64, 64),
            (2080, 65, 65),
            (4096, 128, 128),
            (16384, 512, 512),
            (1 << 20, 512, 512),
        ] {
            assert_eq!(static_chunk(span, 16), at16, "span {span}, floor 16");
            assert_eq!(static_chunk(span, 64), at64, "span {span}, floor 64");
        }
        // The two floors differ exactly below 32 chunks of 64 keys: PR 105's claim that only
        // spans under 512 changed was off by this much.
        for span in 1..=4096 {
            assert_eq!(
                static_chunk(span, 16) != static_chunk(span, 64),
                span < 2048,
                "span {span}"
            );
        }
        // At the floor every device ran before PR 105, the plan is the prefill rows' policy.
        for span in (1..=1 << 20).step_by(61) {
            assert_eq!(
                static_chunk(span, ATTN_SPLIT.min_chunk),
                tier::adaptive_chunk(span, &ATTN_SPLIT),
                "span {span}"
            );
        }
    }

    /// Every key of the span lands in exactly one chunk, no chunk is empty, and the count stays
    /// inside what the combine's shared array holds, for full-context and windowed spans alike.
    #[test]
    fn the_chunks_cover_each_key_once() {
        for floor in FLOORS {
            for (lo, kv_len) in [
                (0usize, 1usize),
                (0, 15),
                (0, 16),
                (0, 17),
                (0, 81),
                (0, 513),
                (0, 2049),
                (0, 40_000),
                (488, 1000),
                (2700, 3213),
            ] {
                let chunk = static_chunk(kv_len - lo, floor);
                let cut = ranges(lo, kv_len, chunk);
                assert!(
                    cut.len() <= ATTN_MAX_CHUNKS,
                    "floor {floor} [{lo}, {kv_len})"
                );
                let mut next = lo;
                for r in &cut {
                    assert_eq!(
                        r.start, next,
                        "floor {floor} [{lo}, {kv_len}): a gap or overlap"
                    );
                    assert!(
                        r.end > r.start,
                        "floor {floor} [{lo}, {kv_len}): an empty chunk"
                    );
                    next = r.end;
                }
                assert_eq!(
                    next, kv_len,
                    "floor {floor} [{lo}, {kv_len}): keys left out"
                );
            }
        }
    }

    /// The replay path's per-token chunk equals the static path's wherever the cache's capacity
    /// leaves the baked floor at the tuned one, so the two paths cut a span identically. Past
    /// that capacity the baked floor rises and the replay chunk is the larger of the two.
    #[test]
    fn replay_cuts_a_span_like_the_static_path() {
        for floor in FLOORS {
            let largest_unraised = floor * ATTN_MAX_CHUNKS;
            for cap_rows in [1, 4096, largest_unraised] {
                let baked = baked_min_chunk(cap_rows, floor);
                assert_eq!(baked, floor, "floor {floor}, capacity {cap_rows}");
                for span in (1..=cap_rows).step_by(7) {
                    assert_eq!(
                        live_chunk(baked, span),
                        static_chunk(span, floor),
                        "floor {floor}, capacity {cap_rows}, span {span}"
                    );
                }
            }
            let raised = baked_min_chunk(largest_unraised + 1, floor);
            assert_eq!(raised, floor + 1, "floor {floor}");
            assert_eq!(live_chunk(raised, 1), floor + 1);
            assert!(tier::n_chunks(largest_unraised + 1, raised) <= ATTN_MAX_CHUNKS);
        }
    }

    /// `live_chunk` above is a copy of what the three replay kernels evaluate; this holds the copy
    /// to their text, and the host policy's shape to the literals in it.
    #[test]
    fn the_replay_kernels_spell_the_same_policy() {
        assert_eq!((ATTN_SPLIT.target_chunks, ATTN_SPLIT.max_chunk), (32, 512));
        for (name, src, floor) in [
            (
                "attn_live.comp",
                include_str!("../shaders/attn_live.comp"),
                "pc.min_chunk",
            ),
            (
                "attn_decode.comp",
                include_str!("../shaders/attn_decode.comp"),
                "pc.chunk",
            ),
            (
                "attn_partial.comp",
                include_str!("../shaders/attn_partial.comp"),
                "pc.chunk",
            ),
        ] {
            let want = format!("uint chunkv = max({floor}, min(max(span / 32u, 1u), 512u));");
            assert!(src.contains(&want), "{name} no longer computes `{want}`");
        }
    }

    #[test]
    fn an_mtp_verify_batch_takes_the_decode_floor() {
        assert!(uses_decode_floor(1, false));
        assert!(uses_decode_floor(1, true));
        assert!(uses_decode_floor(7, true));
        assert!(!uses_decode_floor(7, false));
    }
}
