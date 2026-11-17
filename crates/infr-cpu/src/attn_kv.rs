//! The f32 key/value rows `Op::Attention` reads, materialized from the caches it is bound to: the
//! raw cache's valid prefix, and of DeepSeek V4's compressed segment only the rows some query row
//! can give weight to.

use crate::pool::SpinPool;
use crate::recycle::Recycler;
use infr_core::graph::{comp_visible, CompSegment};
use infr_core::DType;

/// Elements of the f16 KV prefix one pool task widens to f32 in `Op::Attention`: large enough
/// that a task outweighs its claim, small enough to spread a few-thousand-row prefix over
/// every thread.
const KV_WIDEN_CHUNK: usize = 1 << 14;

/// The first `need` elements of a KV cache at its `dt`, as f32.
pub(crate) fn kv_prefix_f32(
    pool: &SpinPool,
    rc: &mut Recycler,
    b: &[u8],
    dt: DType,
    need: usize,
) -> Vec<f32> {
    match dt {
        // The f16 cache (the default) widens across the pool with the hardware conversion rather
        // than serially per element on this thread. Widening is exact, so the values are the
        // same either way.
        DType::F16 => {
            use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
            let src: &[half::f16] = bytemuck::cast_slice::<u8, u16>(b)[..need].reinterpret_cast();
            let mut dst = rc.for_overwrite(need);
            pool.for_chunks_mut(&mut dst, KV_WIDEN_CHUNK, 1, &|c, piece| {
                let s0 = c * KV_WIDEN_CHUNK;
                src[s0..s0 + piece.len()].convert_to_f32_slice(piece);
            });
            dst
        }
        DType::Q8_0 => crate::dequant_prefix_q8_0(b, need),
        // TurboQuant blocks store the WHT-rotated values; dequant + inverse WHT recovers the
        // original domain so the f32 SDPA runs unchanged.
        dt @ (DType::Turbo2 | DType::Turbo3 | DType::Turbo4) => {
            crate::turbo::dequant_prefix_orig(dt, b, need)
        }
        // bf16 + mainline low-bit quants: dequant the block-aligned prefix via the shared GGUF
        // dequant (only the valid rows, not the whole cache).
        DType::Bf16 | DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Iq4Nl => {
            let pb = infr_gguf::nbytes(dt, need);
            infr_gguf::dequant::dequant_block(dt, &b[..pb]).expect("cpu backend: KV dequant")
        }
        _ => bytemuck::cast_slice::<u8, f32>(b)[..need].to_vec(),
    }
}

/// What one `Op::Attention` reads of its compressed segment.
pub(crate) struct CompRows {
    /// Per query row: the compressed keys it attends, ascending — every visible key, less those
    /// the segment's bias sets to `-inf`.
    pub(crate) keys: Vec<Vec<u32>>,
    /// Per compressed row `j` up to the last one any query keeps: the row of [`CompRows::rows`]
    /// that holds it (meaningless for a row no query keeps).
    pub(crate) slot: Vec<u32>,
    /// The converted rows, each `n_kv * head_dim` wide.
    pub(crate) rows: Vec<f32>,
}

impl CompRows {
    /// Compressed row `j`'s values for kv head `kvh`.
    #[inline]
    pub(crate) fn row(&self, j: u32, kvh: usize, head_dim: usize, n_kv: usize) -> &[f32] {
        let at = (self.slot[j as usize] as usize * n_kv + kvh) * head_dim;
        &self.rows[at..at + head_dim]
    }
}

/// Collect the compressed keys each of the `rows` query rows from position `pos` can give weight
/// to, and convert to f32 only the cache rows among them — at long context a top-k bias keeps a
/// small share of the `len` committed rows. `b` is the bound cache at `dt`, rows `row_w` wide.
///
/// An f16 or f32 cache converts row by row; any other dtype converts its prefix up to the last
/// kept row, as [`kv_prefix_f32`] does for the raw cache, the slots then naming every row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn comp_rows(
    pool: &SpinPool,
    rc: &mut Recycler,
    b: &[u8],
    dt: DType,
    row_w: usize,
    bias: Option<&[f32]>,
    rows: usize,
    pos: u32,
    c: CompSegment,
) -> CompRows {
    let len = c.len as usize;
    let keys = pool.collect(rows, &|ti| {
        let n = comp_visible(pos + ti as u32, c.ratio, c.len);
        match bias {
            None => (0..n).collect(),
            Some(b) => {
                let b = &b[ti * len..ti * len + n as usize];
                (0..n)
                    .filter(|&j| b[j as usize] != f32::NEG_INFINITY)
                    .collect()
            }
        }
    });
    let n_rows = keys
        .iter()
        .filter_map(|k: &Vec<u32>| k.last())
        .max()
        .map_or(0, |&j| j as usize + 1);
    let gather = |src_row: &(dyn Fn(usize, &mut [f32]) + Sync), rc: &mut Recycler| {
        let mut used = vec![false; n_rows];
        for &j in keys.iter().flatten() {
            used[j as usize] = true;
        }
        let mut slot = vec![u32::MAX; n_rows];
        let mut order = Vec::new();
        for (j, (s, _)) in slot
            .iter_mut()
            .zip(&used)
            .enumerate()
            .filter(|(_, (_, u))| **u)
        {
            *s = order.len() as u32;
            order.push(j);
        }
        let mut out = rc.for_overwrite(order.len() * row_w);
        let per = (KV_WIDEN_CHUNK / row_w.max(1)).max(1);
        pool.for_chunks_mut(&mut out, per * row_w, 1, &|c, piece| {
            for (dst, &j) in piece.chunks_exact_mut(row_w).zip(&order[c * per..]) {
                src_row(j, dst);
            }
        });
        (slot, out)
    };
    let (slot, rows) = match dt {
        DType::F16 => {
            use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
            let src: &[half::f16] =
                bytemuck::cast_slice::<u8, u16>(b)[..n_rows * row_w].reinterpret_cast();
            gather(
                &|j, dst| src[j * row_w..(j + 1) * row_w].convert_to_f32_slice(dst),
                rc,
            )
        }
        DType::F32 => {
            let src = &bytemuck::cast_slice::<u8, f32>(b)[..n_rows * row_w];
            gather(
                &|j, dst| dst.copy_from_slice(&src[j * row_w..(j + 1) * row_w]),
                rc,
            )
        }
        _ => (
            (0..n_rows as u32).collect(),
            kv_prefix_f32(pool, rc, b, dt, n_rows * row_w),
        ),
    };
    CompRows { keys, slot, rows }
}
