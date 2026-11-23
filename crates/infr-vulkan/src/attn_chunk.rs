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
//! How a chunk's key range is indexed is the kernels' own arithmetic, with no host copy to test
//! here. That every live key is scanned exactly once is held on a device, against a host
//! attention, by `tests/attn_split_host.rs`, an ignored test, and only for full-causal spans (no
//! window, no canvas) up to its largest case. `tests/attn_decode_parity.rs` has windowed and ring
//! cases, but compares two builds of the partial pass with each other, not with a host
//! reference, so nothing host-referenced covers a windowed or a canvas span.
//!
//! The floor is the tuning parameter `attn_split.decode_min_chunk`. An MTP verify batch scores
//! the positions a decode step would, so it takes the decode floor too (`uses_decode_floor`).

use infr_core::tier::{self, AttnSplitCfg};

use crate::adapter::{ATTN_MAX_CHUNKS, ATTN_SPLIT};

/// The default of `attn_split.decode_min_chunk` on a discrete device in the slow class, the class
/// PR 105 measured it on (`docs/perf/vulkan-decode-attn-chunk.md`).
pub(crate) const SLOW_CLASS_DECODE_MIN_CHUNK: usize = 16;

/// The default of `attn_split.decode_min_chunk` everywhere else: the floor of the prefill rows'
/// policy, which every device's decode ran before PR 105.
pub(crate) const DECODE_MIN_CHUNK: usize = ATTN_SPLIT.min_chunk;

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

    /// Both defaults of the floor.
    const FLOORS: [usize; 2] = [SLOW_CLASS_DECODE_MIN_CHUNK, DECODE_MIN_CHUNK];

    /// What the replay shaders compute per token from the baked floor and the live span.
    fn live_chunk(baked: usize, span: usize) -> usize {
        baked.max((span / 32).clamp(1, 512))
    }

    /// The plan, as literals. A change of the floor, of the target chunk count or of the ceiling
    /// has to face these.
    #[test]
    fn the_plan_is_pinned_at_both_floors() {
        assert_eq!(FLOORS, [16, 64]);
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
