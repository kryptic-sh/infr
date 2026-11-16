//! DeepSeek V4's per-ubatch compressor plan — the state machine that decides which
//! compressor-state rows a ubatch reads, which cache rows it commits, and which persistent-state
//! ring rows it overwrites. Ported from `dsv4_build_comp_plan` in llama.cpp's
//! `src/llama-kv-cache-dsv4.cpp` (the pin `docs/plan/09-deepseek-v4-compressed.md` names), and
//! specified in prose in `docs/deepseek.md`'s "The compressed-KV state machine" section — read
//! that first, it names the traps this file guards against.
//!
//! This is a PURE function: no graph emission, no backend calls, no I/O. It only computes index
//! and position vectors; nothing here consumes them yet (that is the wiring of
//! `Op::CompressPool` into the V4 graph), so this module has no caller outside its own tests.
//!
//! Scope, deliberately narrower than the reference:
//! - **Single stream only.** infr's V4 is single-stream today, so `dsv4_stream_offset` (which is
//!   `0` whenever `n_stream <= 1`) is dropped entirely rather than carried as dead arithmetic.
//!   [`build_dsv4_comp_plan`] instead takes an explicit `n_seqs` and refuses (mirroring the
//!   reference's own throw) when asked to serve more than one sequence.
//! - **No rollback/`n_rs_seq` planes.** The reference's `state_restore_*`/`state_snapshot_*`
//!   fields exist for speculative-decode rollback; infr's V4 has no consumer for them, so they are
//!   left off [`Dsv4CompPlan`] rather than carried as empty vectors nobody fills.
//! - **No coupled-ubatch branch.** The reference's `dsv4_ubatch_has_coupled` path pads a ubatch
//!   whose tokens belong to several sequences at once; a single sequence is never coupled.
//! - **No graph-width padding.** The reference pads its `n_kv` to a 256-row multiple with a floor
//!   of 256; this port reports the unpadded width instead — see
//!   [`Dsv4CompPlan::n_visible_max`] for why that loses nothing.
//!
//! Landed but unwired, same as `Op::CompressPool` itself was before this slice: the public items
//! below have no caller outside `#[cfg(test)]`, so `dead_code` is silenced deliberately rather
//! than by exporting them somewhere that isn't ready to call them.
#![allow(dead_code)]

use anyhow::{anyhow, bail, Result as AResult};
use std::collections::HashMap;

/// Ratio at which the CSA compressor (and the LID compressor beside it) pools rows. Mirrors
/// `DSV4_CSA_RATIO` in `llama-kv-cache-dsv4.cpp`.
pub const DSV4_CSA_RATIO: u32 = 4;

/// Ratio at which the HCA compressor pools rows. Mirrors `DSV4_HCA_RATIO`.
pub const DSV4_HCA_RATIO: u32 = 128;

/// The index `Op::GatherRows` fills from its `fill` value: the sentinel row's entry in
/// [`Dsv4CompPlan::gather_read_idxs`]. The reference names that row by its position past the
/// source ([`Dsv4CompPlan::sentinel_row`]), which the gather treats as out of range, so the emit
/// takes this form rather than translating.
pub const DSV4_GATHER_SENTINEL: i32 = -1;

/// The per-ubatch compressor-state recipe: which compressor-state rows this ubatch reads to
/// build compressed rows, which cache rows it commits, and which persistent-state ring rows it
/// overwrites. See `docs/deepseek.md`'s "The per-ubatch plan" for the derivation this mirrors.
///
/// **Real commits come first.** Every dummy commit (see [`build_dsv4_comp_plan`]) is appended
/// after the loop over tokens has pushed every real one, in the write vectors and in each of the
/// two read halves alike, so the first [`Self::n_real_commits`] entries of
/// [`Self::state_write_idxs`] / [`Self::state_write_pos`] — and the first
/// `n_real_commits * ratio` entries of each read half — are the real blocks.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Dsv4CompPlan {
    /// Number of completed compressed rows visible to each query token, in ubatch token order
    /// (dense: one entry per token, including padding tokens with `pos < 0`, which get `0`).
    /// `n_visible[i] = (pos + 1) / ratio`, an integer FLOOR — a trailing partial block is
    /// invisible and never committed (see `n_visible`'s derivation in [`build_dsv4_comp_plan`]).
    pub n_visible: Vec<i32>,

    /// `max_i n_visible[i]` — how many compressed-cache rows any token of this ubatch can see.
    ///
    /// The reference's `plan.n_kv` is this value passed through `max(GGML_PAD(n_kv, 256), 256)`,
    /// which only sets the width of the compressed mask and K view so the ggml graph keeps one
    /// topology: every padded column is masked `-inf` and so weighs exactly zero, and the LID
    /// top-k count `min(width, indexer_top_k)` keeps every visible row whether or not the width
    /// was padded. infr's compressed attention takes the visible length itself, so this port
    /// carries only the unpadded value; `0` means no token sees a compressed row, and a consumer
    /// may skip the compressed segment then (it would add nothing).
    pub n_visible_max: usize,

    /// APE row id — `pos % ratio` — for each token with `pos >= 0`, in ubatch token order
    /// (compact: padding tokens are skipped, so this is NOT the same length as `n_visible`).
    pub state_pos: Vec<i32>,

    /// Flattened source row ids for state-backed commits, addressing a graph-local tensor laid
    /// out `[persistent_state | current_ubatch_scratch | sentinel]` (see `state_source_idx` in
    /// [`build_dsv4_comp_plan`] for how a row id is chosen). For an overlapping compressor
    /// (`overlap == true`) this is laid out as **two contiguous halves** — every committed
    /// block's previous-window indices, `ratio` each, followed by every block's current-window
    /// indices, `ratio` each — NOT interleaved per block. For a non-overlapping compressor it is
    /// one `ratio`-sized run per committed block, current-window only.
    pub state_read_idxs: Vec<i32>,

    /// The sentinel row's id in [`Self::state_read_idxs`]: `state rows + tokens`, one past the
    /// `[persistent_state | current_ubatch_scratch]` rows, where the reference appends its
    /// zero/`-inf` row (`dsv4_append_zero_row`). It is also that source's row count.
    pub sentinel_row: i32,

    /// Compressed-cache row ids written by state-backed commits, one per committed block: the
    /// real commits, then any dummy commit, which targets the cache's last row.
    pub state_write_idxs: Vec<i32>,

    /// RoPE positions for state-backed commits — the committed block's FIRST position
    /// (`pos + 1 - ratio`), which is what the compressed row then ropes at. A dummy commit's
    /// entry is always `0`.
    pub state_write_pos: Vec<i32>,

    /// How many entries of [`Self::state_write_idxs`] are real block commits; the rest are
    /// dummies. A consumer writes only these rows into the compressed cache.
    pub n_real_commits: usize,

    /// Current-ubatch token indices to read new compressor state from, one per distinct
    /// persistent-state ring row this ubatch touches. Sorted by, and index-paired with,
    /// [`Self::state_persist_dst_idxs`].
    pub state_persist_src_idxs: Vec<i32>,

    /// Persistent-state ring row ids (`pos % state_size`) this ubatch overwrites, deduplicated —
    /// when several tokens in the ubatch land on the same ring row, only the entry for the
    /// HIGHEST `pos` survives — and sorted ascending so the write order is deterministic.
    pub state_persist_dst_idxs: Vec<i32>,
}

impl Dsv4CompPlan {
    /// [`Self::state_read_idxs`] in the form `Op::GatherRows` takes: every sentinel entry
    /// ([`Self::sentinel_row`]) becomes [`DSV4_GATHER_SENTINEL`], so the gather writes its fill
    /// there, and every other entry is a row of the `[persistent_state | current_ubatch_scratch]`
    /// source, below its row count.
    pub fn gather_read_idxs(&self) -> Vec<i32> {
        self.state_read_idxs
            .iter()
            .map(|&i| {
                if i == self.sentinel_row {
                    DSV4_GATHER_SENTINEL
                } else {
                    i
                }
            })
            .collect()
    }
}

struct PersistRow {
    dst: i32,
    src: i32,
    pos: i32,
}

/// Builds the per-ubatch compressor plan for a single-stream, single-sequence DeepSeek V4 ubatch.
///
/// - `positions`: this ubatch's token positions, in ubatch order; a negative entry marks an
///   unused/padding slot.
/// - `n_seqs`: number of distinct sequences represented in this ubatch. Refused above `1` —
///   infr's V4 is single-stream and has no per-sequence ring offset to serve more (mirrors the
///   reference's `n_stream <= 1 && ubatch.n_seqs_unq > 1` throw, which always applies here since
///   `n_stream` is always `1`).
/// - `ratio`: the compressor's pooling ratio ([`DSV4_CSA_RATIO`] or [`DSV4_HCA_RATIO`]).
/// - `overlap`: whether the compressor pools two `ratio`-sized windows (previous + current) per
///   block, vs. one.
/// - `pad_blocks`: which dummy rule applies — the reference's argument of the same name, `true`
///   for the CSA and LID plans and `false` for the HCA plan (`llama_kv_cache_dsv4_context`'s
///   `dsv4_build_comp_plans` calls).
/// - `state_size`: row count of the persistent compressor-state ring.
/// - `kv_size`: row count of the compressed-cache; only its last row (`kv_size - 1`) is
///   addressed here, as the dummy commits' target.
pub fn build_dsv4_comp_plan(
    positions: &[i32],
    n_seqs: u32,
    ratio: u32,
    overlap: bool,
    pad_blocks: bool,
    state_size: u32,
    kv_size: u32,
) -> AResult<Dsv4CompPlan> {
    if n_seqs > 1 {
        return Err(anyhow!(
            "DSV4 single compressed stream cannot serve multiple sequences"
        ));
    }

    let n_tokens = positions.len();
    let ratio_i = ratio as i32;
    let state_size_i = state_size as i32;
    // Single stream: the reference's `state_rows = state_size*n_stream` degenerates to
    // `state_size` exactly, and `dsv4_stream_offset` degenerates to `0` throughout.
    let state_rows = state_size as i64;

    // Last-token-wins position -> ubatch-index map, mirroring the reference's
    // `curr_token_idx_map` build (a later token's assignment overwrites an earlier one at the
    // same position).
    let mut curr_token_idx: HashMap<i32, usize> = HashMap::new();
    for (i, &pos) in positions.iter().enumerate() {
        if pos >= 0 {
            curr_token_idx.insert(pos, i);
        }
    }

    // Addresses the graph-local `[persistent_state | current_ubatch_scratch | sentinel]` tensor:
    // a negative pos is the appended zero/-inf sentinel row; a pos this ubatch itself produced
    // reads that token's own scratch row; anything else reads the persistent ring.
    let state_source_idx = |pos: i32| -> i32 {
        if pos < 0 {
            return (state_rows + n_tokens as i64) as i32;
        }
        if let Some(&i) = curr_token_idx.get(&pos) {
            return (state_rows + i as i64) as i32;
        }
        pos % state_size_i
    };

    let mut n_visible = vec![0i32; n_tokens];
    let mut n_visible_max: i64 = 0;
    let mut state_pos = Vec::new();
    let mut state_write_idxs = Vec::new();
    let mut state_write_pos = Vec::new();
    let mut state_read_idxs = Vec::new();
    let mut persist_rows: Vec<PersistRow> = Vec::new();

    // The overlap compressor needs its reads as two contiguous halves (see `state_read_idxs`'s
    // doc): collect prev/cur separately per block and concatenate once every block has been
    // visited, rather than interleaving them as each block is processed.
    let mut overlap_prev_reads = Vec::new();
    let mut overlap_cur_reads = Vec::new();

    let mut first_valid_token: Option<usize> = None;

    for (i, &pos) in positions.iter().enumerate() {
        if pos < 0 {
            continue;
        }
        if first_valid_token.is_none() {
            first_valid_token = Some(i);
        }

        state_pos.push(pos % ratio_i);

        // FLOOR, not ceiling: a trailing partial block is invisible until a later ubatch
        // completes it. Ceiling this exposes a row built from a half-filled window.
        let visible = (pos as i64 + 1) / ratio as i64;
        n_visible[i] = visible as i32;
        n_visible_max = n_visible_max.max(visible);

        // Persist-row dedup: single stream, so the ring destination is just `pos % state_size`.
        // Keep the entry for the highest `pos` when several tokens collide on one ring row.
        let dst = pos % state_size_i;
        match persist_rows.iter_mut().find(|row| row.dst == dst) {
            Some(row) if pos > row.pos => {
                row.src = i as i32;
                row.pos = pos;
            }
            Some(_) => {}
            None => persist_rows.push(PersistRow {
                dst,
                src: i as i32,
                pos,
            }),
        }

        if (pos + 1) % ratio_i != 0 {
            continue;
        }

        // Block boundary: commit cache row `pos / ratio`, roped at the block's FIRST position.
        let source_start = pos + 1 - ratio_i;
        state_write_idxs.push(pos / ratio_i);
        state_write_pos.push(source_start);

        if overlap {
            let prev_start = source_start - ratio_i;
            for j in 0..ratio_i {
                overlap_prev_reads.push(state_source_idx(prev_start + j));
            }
            for j in 0..ratio_i {
                overlap_cur_reads.push(state_source_idx(source_start + j));
            }
        } else {
            for j in 0..ratio_i {
                state_read_idxs.push(state_source_idx(source_start + j));
            }
        }
    }

    let n_real_commits = state_write_idxs.len();
    // Both dummy rules repeat the first valid token's source row and write the cache's last
    // row, at rope position 0. That row is garbage, kept harmless because no token's
    // `n_visible` reaches `kv_size`, so it stays masked until a real commit overwrites it.
    let dummy_source = first_valid_token.map(|i| state_source_idx(positions[i]));

    if pad_blocks {
        if let Some(source_idx) = dummy_source {
            debug_assert!(kv_size > 0, "DSV4 dummy block needs a non-empty cache");

            // The reference pads each sequence's commits up to the ubatch's block count,
            // `ceil(max(1, n_seq_tokens)/ratio)`, so a ubatch of a given size always emits the
            // same number of blocks. `llama_kv_cache_dsv4::init_batch` splits with `split_equal`,
            // whose `ubatch_add(idxs, n_seqs, true)` makes `n_seq_tokens = n_tokens/n_seqs`: for
            // one sequence, the whole ubatch. Contiguous positions complete either that many
            // blocks or one fewer, so at most ONE dummy is ever added; a sequence two or more
            // blocks short has positions with a gap, which the reference refuses rather than pad.
            //
            // `n_seq_tokens` counts the tokens with a position: a reference ubatch carries no
            // negative positions, so the two counts agree on every ubatch it builds, and a
            // padding slot here completes no block and so must not demand one.
            let n_seq_tokens = state_pos.len() as u32;
            let n_blocks = n_seq_tokens.max(1).div_ceil(ratio) as usize;
            if n_real_commits < n_blocks {
                if n_real_commits + 1 != n_blocks {
                    bail!("DSV4 CSA sequence positions are not contiguous");
                }
                state_write_idxs.push(kv_size as i32 - 1);
                state_write_pos.push(0);

                if overlap {
                    for _ in 0..ratio {
                        overlap_prev_reads.push(source_idx);
                        overlap_cur_reads.push(source_idx);
                    }
                } else {
                    for _ in 0..ratio {
                        state_read_idxs.push(source_idx);
                    }
                }
            }
        }
    } else if n_real_commits == 0 {
        // HCA: a ubatch that completes no block still commits one dummy, so the compress and
        // write ops stay in the graph. Reads go straight into `state_read_idxs`, as in the
        // reference — the HCA compressor does not overlap, so there are no halves to keep.
        if let Some(source_idx) = dummy_source {
            debug_assert!(kv_size > 0, "DSV4 dummy block needs a non-empty cache");
            // A full cache implies a completed block, which implies a real commit, so the last
            // row cannot be live here.
            debug_assert!(
                n_visible_max < kv_size as i64,
                "DSV4 HCA dummy would overwrite a visible row"
            );
            state_write_idxs.push(kv_size as i32 - 1);
            state_write_pos.push(0);
            for _ in 0..ratio {
                state_read_idxs.push(source_idx);
            }
        }
    }

    if overlap {
        // [ all blocks' prev-window indices | all blocks' cur-window indices ]
        state_read_idxs.reserve(overlap_prev_reads.len() + overlap_cur_reads.len());
        state_read_idxs.extend(overlap_prev_reads);
        state_read_idxs.extend(overlap_cur_reads);
    }

    persist_rows.sort_by_key(|row| row.dst);
    let state_persist_src_idxs = persist_rows.iter().map(|row| row.src).collect();
    let state_persist_dst_idxs = persist_rows.iter().map(|row| row.dst).collect();

    Ok(Dsv4CompPlan {
        n_visible,
        n_visible_max: n_visible_max as usize,
        state_pos,
        state_read_idxs,
        sentinel_row: (state_rows + n_tokens as i64) as i32,
        state_write_idxs,
        state_write_pos,
        n_real_commits,
        state_persist_src_idxs,
        state_persist_dst_idxs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CSA compressor state ring: `2 * ratio` rows, because it overlaps.
    const CSA_STATE: u32 = 2 * DSV4_CSA_RATIO;
    /// The HCA compressor state ring: `ratio` rows, because it does not.
    const HCA_STATE: u32 = DSV4_HCA_RATIO;

    fn csa(positions: &[i32], kv_size: u32) -> AResult<Dsv4CompPlan> {
        build_dsv4_comp_plan(positions, 1, DSV4_CSA_RATIO, true, true, CSA_STATE, kv_size)
    }

    fn hca(positions: &[i32], kv_size: u32) -> AResult<Dsv4CompPlan> {
        build_dsv4_comp_plan(
            positions,
            1,
            DSV4_HCA_RATIO,
            false,
            false,
            HCA_STATE,
            kv_size,
        )
    }

    /// 12-token CSA prefill, ratio 4, overlap, `state_size = 8`, positions `0..=11`. Commits
    /// happen at `pos` 3, 7, 11 (block boundaries), giving three full blocks — enough to exercise
    /// the sentinel path (block 0's previous window), the ring-wrap persist dedup (12 tokens >
    /// `state_size` 8), and the two-halves overlap read order, all with vectors derived by hand
    /// below from the reference's rules rather than asserted only on length.
    ///
    /// Derivation (ratio=4, state_size=8, kv_size=16, n_tokens=12, positions[i] == i):
    /// - `state_pos[i] = pos % 4` for all 12 tokens: `[0,1,2,3, 0,1,2,3, 0,1,2,3]`.
    /// - `n_visible[i] = (pos+1)/4` floor: `[0,0,0,1, 1,1,1,2, 2,2,2,3]`; max is 3.
    /// - Commits at pos 3 (source_start=0, write_idx=0, write_pos=0), pos 7 (source_start=4,
    ///   write_idx=1, write_pos=4), pos 11 (source_start=8, write_idx=2, write_pos=8).
    /// - `state_source_idx`: since `positions[i] == i`, any pos in `0..=11` is this ubatch's own
    ///   token i, giving `state_rows(8) + i`. A pos `< 0` gives the sentinel `8 + 12 = 20`.
    ///   - block pos=3: prev_start=-4 -> reads -4,-3,-2,-1, all `< 0` -> sentinel `[20,20,20,20]`.
    ///     cur reads 0,1,2,3 -> `[8,9,10,11]`.
    ///   - block pos=7: prev_start=0 -> reads 0,1,2,3 -> `[8,9,10,11]` (this ubatch's own rows).
    ///     cur reads 4,5,6,7 -> `[12,13,14,15]`.
    ///   - block pos=11: prev_start=4 -> reads 4,5,6,7 -> `[12,13,14,15]`.
    ///     cur reads 8,9,10,11 -> `[16,17,18,19]`.
    ///   - `n_blocks = ceil(12/4) = 3 == n_writes`, so no dummy block.
    ///   - two-halves concat: prev (12) then cur (12), 24 entries total.
    /// - Persist ring dedup (`dst = pos % 8`): pos 8..11 (indices 8..11) overwrite pos 0..3
    ///   (indices 0..3) at the same ring rows since 8 > 0 etc; pos 4..7 (indices 4..7) are
    ///   untouched. Sorted by dst 0..7: `src = [8,9,10,11, 4,5,6,7]`.
    #[test]
    fn prefill_commits_several_csa_blocks_overlap() {
        let positions: Vec<i32> = (0..12).collect();
        let plan = csa(&positions, 16).unwrap();

        assert_eq!(plan.state_pos, vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(plan.n_visible, vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3]);
        assert_eq!(plan.n_visible_max, 3);

        assert_eq!(plan.state_write_idxs, vec![0, 1, 2]);
        assert_eq!(plan.state_write_pos, vec![0, 4, 8]);
        assert_eq!(plan.n_real_commits, 3);

        let expected_reads = vec![
            // prev half: block0 (sentinel), block1, block2
            20, 20, 20, 20, 8, 9, 10, 11, 12, 13, 14, 15,
            // cur half: block0, block1, block2
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ];
        assert_eq!(plan.state_read_idxs, expected_reads);

        assert_eq!(plan.state_persist_src_idxs, vec![8, 9, 10, 11, 4, 5, 6, 7]);
        assert_eq!(plan.state_persist_dst_idxs, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    /// A CSA prefill that starts MID-block, so its first block's windows come partly from the
    /// persistent ring, and spans three blocks without a dummy. Every vector is hand-derived from
    /// the reference's rules (ratio=4, overlap, state_size=8, kv_size=16, positions `6..=17`, so
    /// ubatch index i has pos = 6+i; scratch row of pos p is `8 + (p-6)`; sentinel `8+12 = 20`,
    /// unused here because no window reaches below pos 0):
    /// - Commits at pos 7 (i=1), 11 (i=5), 15 (i=9): write_idx `[1,2,3]`, write_pos `[4,8,12]`.
    ///   `n_blocks = ceil(12/4) = 3 == n_writes`, so no dummy.
    /// - block pos=7 (source_start 4): prev reads pos 0..=3, none in this ubatch -> ring
    ///   `pos % 8` -> `[0,1,2,3]`; cur reads pos 4,5 (ring `4,5`) and 6,7 (scratch `8,9`).
    /// - block pos=11 (source_start 8): prev reads 4..=7 -> `[4,5,8,9]`; cur reads 8..=11 ->
    ///   `[10,11,12,13]`.
    /// - block pos=15 (source_start 12): prev reads 8..=11 -> `[10,11,12,13]`; cur reads 12..=15
    ///   -> `[14,15,16,17]`.
    /// - Reads are every block's prev window, then every block's cur window.
    /// - `n_visible = (pos+1)/4` over pos 6..=17: `[1,2,2,2,2,3,3,3,3,4,4,4]`.
    /// - `state_pos = pos % 4`: `[2,3,0,1,2,3,0,1,2,3,0,1]`.
    /// - Persist (`dst = pos % 8`, highest pos wins): dst 0 <- pos 16 (i10), 1 <- 17 (i11),
    ///   2 <- 10 (i4), 3 <- 11 (i5), 4 <- 12 (i6), 5 <- 13 (i7), 6 <- 14 (i8, beats pos 6),
    ///   7 <- 15 (i9, beats pos 7).
    #[test]
    fn csa_prefill_from_mid_block_reads_the_ring_in_two_halves() {
        let positions: Vec<i32> = (6..18).collect();
        let plan = csa(&positions, 16).unwrap();

        assert_eq!(plan.state_write_idxs, vec![1, 2, 3]);
        assert_eq!(plan.state_write_pos, vec![4, 8, 12]);
        assert_eq!(plan.n_real_commits, 3);
        assert_eq!(
            plan.state_read_idxs,
            vec![
                // prev half
                0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, //
                // cur half
                4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            ]
        );
        assert_eq!(plan.n_visible, vec![1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]);
        assert_eq!(plan.n_visible_max, 4);
        assert_eq!(plan.state_pos, vec![2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1]);
        assert_eq!(plan.state_persist_dst_idxs, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(plan.state_persist_src_idxs, vec![10, 11, 4, 5, 6, 7, 8, 9]);
    }

    /// The HCA counterpart: a prefill from pos 100 to 300 completes the blocks ending at 127 and
    /// 255, the first of which is mostly ring. Hand-derived (ratio=128, no overlap,
    /// state_size=128, kv_size=32, ubatch index i has pos 100+i, scratch row of pos p is
    /// `128 + (p-100)`):
    /// - Commits at pos 127 (source_start 0) and 255 (source_start 128): write_idx `[0,1]`,
    ///   write_pos `[0,128]`. HCA has no padding rule, and a real commit exists, so no dummy.
    /// - block 0 reads pos 0..=127: pos 0..=99 from the ring (`0..100`), 100..=127 from scratch
    ///   (`128..156`). block 1 reads pos 128..=255, all scratch (`156..284`).
    /// - `n_visible_max = (300+1)/128 = 2`.
    /// - Persist: ring row d is won by the highest of d, d+128, d+256 inside 100..=300 — pos
    ///   d+256 (i = d+156) for d <= 44, else pos d+128 (i = d+28).
    #[test]
    fn hca_prefill_from_mid_block_reads_the_ring() {
        let positions: Vec<i32> = (100..301).collect();
        let plan = hca(&positions, 32).unwrap();

        assert_eq!(plan.state_write_idxs, vec![0, 1]);
        assert_eq!(plan.state_write_pos, vec![0, 128]);
        assert_eq!(plan.n_real_commits, 2);
        let reads: Vec<i32> = (0..100).chain(128..156).chain(156..284).collect();
        assert_eq!(plan.state_read_idxs, reads);
        assert_eq!(plan.n_visible_max, 2);
        assert_eq!(plan.state_persist_dst_idxs, (0..128).collect::<Vec<i32>>());
        let src: Vec<i32> = (0..128)
            .map(|d| if d <= 44 { d + 156 } else { d + 28 })
            .collect();
        assert_eq!(plan.state_persist_src_idxs, src);
    }

    /// Isolates the overlap two-halves order on its own (smaller than the big prefill test, and
    /// starting mid-stream so the previous window is read from the persistent RING rather than
    /// the sentinel, exercising the other branch of `state_source_idx`).
    ///
    /// Derivation (ratio=4, overlap, state_size=8, positions `4..=11`, so ubatch index i has
    /// pos = 4+i):
    /// - Commits at pos 7 (i=3: source_start=4, write_idx=1, write_pos=4) and pos 11 (i=7:
    ///   source_start=8, write_idx=2, write_pos=8).
    /// - block pos=7: prev_start=0 -> reads pos 0,1,2,3, none in this ubatch -> ring `pos % 8` ->
    ///   `[0,1,2,3]`. cur reads pos 4,5,6,7 -> this ubatch's i=0..3 -> `state_rows(8)+i` ->
    ///   `[8,9,10,11]`.
    /// - block pos=11: prev_start=4 -> reads pos 4,5,6,7 -> this ubatch's i=0..3 -> `[8,9,10,11]`
    ///   (same values as block0's cur half — expected, both address the same rows). cur reads pos
    ///   8,9,10,11 -> this ubatch's i=4..7 -> `[12,13,14,15]`.
    /// - Two-halves concat: `[0,1,2,3, 8,9,10,11]` (prev) then `[8,9,10,11, 12,13,14,15]` (cur).
    ///   An interleaved implementation would instead produce
    ///   `[0,1,2,3, 8,9,10,11, 8,9,10,11, 12,13,14,15]` in per-block prev/cur/prev/cur order,
    ///   which happens to coincide here in the middle four entries but diverges at the front and
    ///   back — this is why the big prefill test's sentinel-filled block0 is the sharper check.
    #[test]
    fn overlap_two_halves_are_not_interleaved() {
        let positions: Vec<i32> = (4..12).collect();
        let plan = csa(&positions, 16).unwrap();

        assert_eq!(plan.state_write_idxs, vec![1, 2]);
        assert_eq!(
            plan.state_read_idxs,
            vec![0, 1, 2, 3, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15]
        );
    }

    /// Block 0's previous window is entirely before the sequence start (`pos < 0`), so all
    /// `ratio` of its prev-half reads must be the sentinel row `state_rows + n_tokens`.
    #[test]
    fn block_zero_prev_window_is_all_sentinel() {
        let positions: Vec<i32> = vec![0, 1, 2, 3];
        let plan = csa(&positions, 16).unwrap();

        let sentinel = CSA_STATE as i32 + positions.len() as i32;
        assert_eq!(plan.state_read_idxs[0..4], vec![sentinel; 4]);
        // cur half: pos 0..3 are this ubatch's own tokens i=0..3 -> state_size + i.
        let s = CSA_STATE as i32;
        assert_eq!(plan.state_read_idxs[4..8], vec![s, s + 1, s + 2, s + 3]);
    }

    /// What the emit hands `Op::GatherRows`: the sentinel entries as [`DSV4_GATHER_SENTINEL`],
    /// everything else unchanged and inside the `[state | scratch]` source, whose row count is
    /// `sentinel_row`. Checked on a prefill whose first block reads the sentinel and whose later
    /// blocks read the ring and this ubatch's own rows.
    #[test]
    fn gather_read_idxs_names_the_sentinel_as_minus_one() {
        let positions: Vec<i32> = (0..12).collect();
        let plan = csa(&positions, 16).unwrap();
        let sentinel = CSA_STATE as i32 + positions.len() as i32;
        assert_eq!(plan.sentinel_row, sentinel);
        let g = plan.gather_read_idxs();
        assert_eq!(g.len(), plan.state_read_idxs.len());
        assert_eq!(g[0..4], [DSV4_GATHER_SENTINEL; 4]);
        for (k, (&got, &raw)) in g.iter().zip(&plan.state_read_idxs).enumerate() {
            if raw == sentinel {
                assert_eq!(got, DSV4_GATHER_SENTINEL, "entry {k}");
            } else {
                assert_eq!(got, raw, "entry {k}");
                assert!(
                    (0..sentinel).contains(&got),
                    "entry {k} = {got} leaves the source"
                );
            }
        }
    }

    /// A single-token CSA decode step off a block boundary: no real commit (pos=13,
    /// `(13+1) % 4 == 2 != 0`), but `n_blocks = ceil(1/4) = 1`, so one dummy block is appended to
    /// the cache's last row. Uses `overlap = false` to exercise that branch's inline
    /// (non-two-halves) read layout.
    #[test]
    fn decode_step_csa_appends_dummy_block() {
        let kv_size = 16u32;
        let plan = build_dsv4_comp_plan(&[13], 1, DSV4_CSA_RATIO, false, true, CSA_STATE, kv_size)
            .unwrap();

        assert_eq!(plan.state_write_idxs, vec![kv_size as i32 - 1]);
        assert_eq!(plan.state_write_pos, vec![0]);
        assert_eq!(plan.n_real_commits, 0);
        // pos 13 is this ubatch's own token i=0 -> state_size + 0.
        assert_eq!(plan.state_read_idxs, vec![CSA_STATE as i32; 4]);
        assert_eq!(plan.state_pos, vec![13 % 4]);
    }

    /// An HCA decode step that completes no block gets exactly ONE dummy, at the cache's last row:
    /// the reference's `!pad_blocks && state_write_idxs.empty()` branch. Hand-derived: pos 50 is
    /// this ubatch's own token i=0, so the dummy reads scratch row `128 + 0`, `ratio` times;
    /// `(50+1)/128` floors to 0 visible rows; the ring row is `50 % 128`.
    #[test]
    fn decode_step_hca_appends_one_dummy_at_the_last_row() {
        let plan = hca(&[50], 32).unwrap();

        assert_eq!(plan.state_write_idxs, vec![31]);
        assert_eq!(plan.state_write_pos, vec![0]);
        assert_eq!(plan.n_real_commits, 0);
        assert_eq!(plan.state_read_idxs, vec![128; DSV4_HCA_RATIO as usize]);
        assert_eq!(plan.n_visible, vec![0]);
        assert_eq!(plan.n_visible_max, 0);
        assert_eq!(plan.state_pos, vec![50]);
        assert_eq!(plan.state_persist_src_idxs, vec![0]);
        assert_eq!(plan.state_persist_dst_idxs, vec![50]);
    }

    /// A trailing partial block at the end of a longer CSA prefill: positions 0..=5 complete one
    /// block (pos 3) but leave pos 4,5 as an incomplete second block. `n_visible` must floor —
    /// tokens 4 and 5 stay at the SAME `n_visible` as token 3 (`1`), not `2` — and that partial
    /// block is never committed: `n_blocks = ceil(6/4) = 2` against one real commit pads ONE
    /// dummy instead, after the real commit.
    #[test]
    fn n_visible_floors_at_a_trailing_partial_block() {
        let positions: Vec<i32> = vec![0, 1, 2, 3, 4, 5];
        let plan = build_dsv4_comp_plan(&positions, 1, DSV4_CSA_RATIO, false, true, CSA_STATE, 16)
            .unwrap();

        assert_eq!(plan.n_visible, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(plan.state_write_idxs, vec![0, 15]); // pos/ratio = 3/4 = 0, then the dummy
        assert_eq!(plan.state_write_pos, vec![0, 0]);
        assert_eq!(plan.n_real_commits, 1);
    }

    /// A 9-token contiguous CSA ubatch that completes two blocks is padded to the reference's
    /// `n_blocks = ceil(9/4) = 3` with ONE dummy, appended after the real commits in both read
    /// halves. Hand-derived (ratio=4, overlap, state_size=8, kv_size=16, positions `1..=9`, so
    /// the scratch row of pos p is `8 + (p-1)` and the sentinel is `8+9 = 17`):
    /// - Commits at pos 3 (source_start 0) and 7 (source_start 4): write_idx `[0,1]`, pos `[0,4]`.
    /// - block pos=3: prev reads -4..=-1 -> sentinel `[17;4]`; cur reads pos 0 (ring `0`) and
    ///   1..=3 (scratch `8,9,10`).
    /// - block pos=7: prev reads 0..=3 -> `[0,8,9,10]`; cur reads 4..=7 -> `[11,12,13,14]`.
    /// - dummy: the first valid token is pos 1 -> scratch `8`, `ratio` times in each half; it
    ///   writes row `kv_size-1 = 15` at pos 0.
    #[test]
    fn csa_ubatch_one_block_short_pads_one_dummy_after_the_real_commits() {
        let positions: Vec<i32> = (1..10).collect();
        let plan = csa(&positions, 16).unwrap();

        assert_eq!(plan.state_write_idxs, vec![0, 1, 15]);
        assert_eq!(plan.state_write_pos, vec![0, 4, 0]);
        assert_eq!(plan.n_real_commits, 2);
        assert_eq!(
            plan.state_read_idxs,
            vec![
                // prev half: block0, block1, dummy
                17, 17, 17, 17, 0, 8, 9, 10, 8, 8, 8, 8, //
                // cur half: block0, block1, dummy
                0, 8, 9, 10, 11, 12, 13, 14, 8, 8, 8, 8,
            ]
        );
        assert_eq!(plan.n_visible, vec![0, 0, 1, 1, 1, 1, 2, 2, 2]);
        assert_eq!(plan.n_visible_max, 2);
        assert_eq!(plan.state_pos, vec![1, 2, 3, 0, 1, 2, 3, 0, 1]);
        // dst = pos % 8; pos 9 (i8) beats pos 1 on ring row 1, pos 8 (i7) alone takes row 0.
        assert_eq!(plan.state_persist_dst_idxs, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(plan.state_persist_src_idxs, vec![7, 8, 1, 2, 3, 4, 5, 6]);
    }

    /// A 9-token CSA ubatch that commits NOTHING is three blocks short of `ceil(9/4) = 3` — which
    /// contiguous positions cannot be — so the reference refuses it ("DSV4 CSA sequence positions
    /// are not contiguous") rather than padding. The positions skip every `%4 == 3` boundary
    /// (0-2, 100-102, 200-202). The same positions on the HCA plan, which has no block count to
    /// meet, get the one dummy that rule always adds.
    #[test]
    fn csa_ubatch_more_than_one_block_short_is_refused() {
        let positions: Vec<i32> = vec![0, 1, 2, 100, 101, 102, 200, 201, 202];
        let err = csa(&positions, 16).unwrap_err();
        assert_eq!(
            err.to_string(),
            "DSV4 CSA sequence positions are not contiguous"
        );

        let plan = hca(&positions, 32).unwrap();
        assert_eq!(plan.state_write_idxs, vec![31]);
        assert_eq!(plan.n_real_commits, 0);
    }

    /// Persist-row dedup under ring wrap: more tokens (12) than `state_size` (8), so positions
    /// 8..11 land on the same ring rows as 0..3. The surviving `src` must be the HIGHER-pos
    /// token's index, and destinations must come out sorted. (This is the same ubatch as
    /// `prefill_commits_several_csa_blocks_overlap`; isolated here so the persist assertion
    /// stands on its own.)
    #[test]
    fn persist_dedup_keeps_highest_pos_and_sorts_by_dst() {
        let positions: Vec<i32> = (0..12).collect();
        let plan = csa(&positions, 16).unwrap();

        assert_eq!(plan.state_persist_dst_idxs, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // dst 0..3 are won by tokens 8..11 (higher pos); dst 4..7 only ever touched by 4..7.
        assert_eq!(plan.state_persist_src_idxs, vec![8, 9, 10, 11, 4, 5, 6, 7]);
    }

    /// Mirrors the reference's `"DSV4 single compressed stream cannot serve multiple sequences"`
    /// throw: infr's V4 is single-stream, so more than one sequence in a ubatch must be refused
    /// rather than silently addressing the wrong ring rows.
    #[test]
    fn refuses_more_than_one_sequence() {
        let err =
            build_dsv4_comp_plan(&[0, 1, 2, 3], 2, DSV4_CSA_RATIO, true, true, 8, 16).unwrap_err();
        assert!(
            err.to_string().contains("single compressed stream"),
            "unexpected error: {err}"
        );
    }
}
