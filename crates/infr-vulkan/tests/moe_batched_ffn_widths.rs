//! `Op::MoeFfn` against an f64 host reference at expert widths the batched expert GEMMs' column
//! tile does and does not divide, at row counts on both sides of the small-m threshold and of the
//! GEMMs' row tiles.
//!
//! The batched arm's expert GEMMs (`Recorder::matmul_mmq_experts`) cover their output width in
//! whole `infr_core::tensor::MOE_EXPERT_GEMM_TILE_N`-column tiles and write nothing past the last one. They used to take any
//! width: a gate/up width of 32 got no workgroups at all and one of 96 lost its last 32 columns, so
//! every prefill chunk of more than `moe.small_m` rows read uninitialised scratch where those
//! columns belonged — NaN logits on the synthetic DeepSeek-V2 file (`n_ff_exp` 32). Below the
//! threshold the small-m id-GEMV arm ran and was right, which is why only chunks past it failed.
//! Such a width now takes the small-m arm at every row count.
//!
//! `moe_ffn_matches_host_at_every_width` holds the op to the reference at gate/up widths of 32 and
//! 96 and a model width of 96 (partial tiles, held to the small-m arm's float-noise bound), at 64
//! and 128 (whole tiles, the batched GEMMs' own shapes), and in the fused gate|up layout, whose
//! GEMM is twice `n_ff_exp` wide. Each run also checks, by the kernels its backend built, that the
//! batched expert GEMMs ran exactly where they should: a whole-tile shape that silently took the
//! small-m arm, or a partial one that reached the GEMMs, fails.
//! `expert_gemm_refuses_a_partial_column_tile` holds the recorder to refusing a partial width by
//! name instead of dispatching a short grid.
//!
//! Run: `cargo test -p infr-vulkan --test moe_batched_ffn_widths -- --ignored --nocapture`
use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{Activation, Graph, MoeGating, Op};
use infr_core::tensor::TensorDesc;
use infr_core::DType;
use infr_vulkan::linear::pad_to_u32_align;
use infr_vulkan::VulkanBackend;

/// The expert banks' format: the one the synthetic DeepSeek-V2 file failed with, and a member of
/// `infr_core::tensor::MOE_MMQ_DTYPES`, so a chunk past the small-m threshold takes the batched arm
/// whenever its widths allow.
const BANK: DType = DType::Q8_0;

const N_EXPERT: usize = 8;
const N_USED: usize = 2;

/// The small-m threshold every backend here is opened with (`moe.small_m`'s shipped value), so
/// which arm a row count takes does not depend on the environment.
const SMALL_M: usize = 8;

/// Row counts: [`SMALL_M`] and the rows just past it, the BM=32 and BM=64 row tiles and one row
/// past each.
const ROWS: &[usize] = &[SMALL_M, SMALL_M + 1, SMALL_M + 2, 31, 32, 33, 64, 65];

/// The bound on `max |got - want|` as a fraction of the reference's RMS. The batched arm rounds
/// both GEMMs' activations to int8 per 32-element block (the reference does not), which costs a
/// few percent of the RMS here; the small-m arm keeps them f32 and lands within float noise. A
/// column left unwritten is NaN or garbage the size of the RMS itself.
const MAX_REL: f64 = 0.15;

/// The bound for a shape the expert GEMMs' column tile does not divide: the small-m arm must have
/// run it, and that arm reads f32 activations, so it lands within float noise of the reference
/// where the batched arm's int8 activations would not.
const SMALL_M_MAX_REL: f64 = 1e-4;

/// Whether a built kernel is one of the batched arm's expert GEMMs (`native_gemm_mmq_*_xp*`, the
/// resident and paged EXPERT_GRID builds, every tile).
fn is_expert_gemm(kernel: &str) -> bool {
    kernel.starts_with("native_gemm_mmq_") && kernel.contains("_xp")
}

/// A fresh backend on the environment's device (`INFR_DEV`) with the small-m threshold at
/// [`SMALL_M`]. Fresh per run, so its built-kernel list is that run's alone.
fn backend() -> Option<VulkanBackend> {
    let env = infr_core::config::ConfigLayer::env().expect("test environment config");
    let mut cfg = infr_core::config::Config::load_from_layers(&[env]);
    cfg.kernels.vulkan.moe_small_m = SMALL_M;
    infr_testkit::gpu_or_skip_noted(VulkanBackend::new_with(std::sync::Arc::new(cfg)))
}

/// The least gap between the last chosen expert's router logit and the next one's a token is
/// drawn with: far past the f32 router GEMV's rounding.
const ROUTING_MARGIN: f64 = 1e-2;

/// A deterministic value in `[-0.5, 0.5)` for element `i` of stream `salt`.
fn hashed(i: usize, salt: u64) -> f32 {
    let h = infr_testkit::gguf::splitmix64((i as u64) ^ (salt << 40));
    (h >> 40) as f32 / (1u64 << 24) as f32 - 0.5
}

/// `n_elem` elements of synthetic [`BANK`] blocks and their host dequantization.
fn bank(n_elem: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let bytes = infr_testkit::synth_weight(BANK, n_elem, seed);
    let deq = infr_gguf::dequant::dequant_block(BANK, &bytes).unwrap();
    (bytes, deq)
}

fn dot(w: &[f32], x: &[f32]) -> f64 {
    w.iter()
        .zip(x)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum()
}

fn silu(z: f64) -> f64 {
    z / (1.0 + (-z).exp())
}

/// One `MoeFfn` shape: `ne` the model width, `nff` the expert FFN width, `fused` one
/// `[2·nff, ne]` gate|up slice per expert (gate rows first) instead of two `[nff, ne]` banks.
/// Each row count runs on its own [`backend`], which must have built the batched arm's expert
/// GEMMs exactly when the arm applies: more than [`SMALL_M`] rows, whole column tiles, and a
/// device with packed int8 dot products.
fn run(ne: usize, nff: usize, fused: bool) {
    let gu_rows = if fused { 2 * nff } else { nff };
    let router: Vec<f32> = (0..N_EXPERT * ne).map(|i| hashed(i, 1)).collect();
    let (gq, gd) = bank(N_EXPERT * gu_rows * ne, 2);
    let (uq, ud) = bank(N_EXPERT * nff * ne, 3);
    let (dq, dd) = bank(N_EXPERT * ne * nff, 4);
    // The gate and up rows of expert `e`'s FFN row `j`.
    let gate_row = |e: usize, j: usize| &gd[(e * gu_rows + j) * ne..][..ne];
    let up_row = |e: usize, j: usize| {
        if fused {
            &gd[(e * gu_rows + nff + j) * ne..][..ne]
        } else {
            &ud[(e * nff + j) * ne..][..ne]
        }
    };
    for &rows in ROWS {
        let label = format!(
            "ne={ne} n_ff_exp={nff} {} rows={rows}",
            if fused { "fused" } else { "split" }
        );
        let Some(be) = backend() else {
            return;
        };
        let whole_tiles = infr_core::tensor::moe_expert_gemm_widths_ok(gu_rows, ne);
        let batched = rows > SMALL_M && whole_tiles && be.capabilities().i8_dot;
        let logits_of = |xr: &[f32]| -> Vec<f64> {
            (0..N_EXPERT)
                .map(|e| dot(&router[e * ne..][..ne], xr))
                .collect()
        };
        let ranked = |logits: &[f64]| -> Vec<usize> {
            let mut order: Vec<usize> = (0..N_EXPERT).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            order
        };
        // Each token's row is redrawn until its routing has no near-tie, so the GPU's f32 router
        // picks the experts the reference does: what is compared is the expert FFN, not a flip.
        let mut x = Vec::with_capacity(rows * ne);
        for t in 0..rows {
            let row = (0u64..)
                .map(|draw| -> Vec<f32> {
                    (0..ne)
                        .map(|i| hashed(t * ne + i, 5 + (draw << 8)))
                        .collect()
                })
                .find(|row| {
                    let logits = logits_of(row);
                    let order = ranked(&logits);
                    logits[order[N_USED - 1]] - logits[order[N_USED]] > ROUTING_MARGIN
                })
                .expect("a row without a near-tie");
            x.extend(row);
        }
        let mut want = vec![0f64; rows * ne];
        for t in 0..rows {
            let xr = &x[t * ne..][..ne];
            let logits = logits_of(xr);
            let order = ranked(&logits);
            let max = logits[order[0]];
            let probs: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
            let chosen = &order[..N_USED];
            let wsum: f64 = chosen.iter().map(|&e| probs[e]).sum();
            for &e in chosen {
                let act: Vec<f32> = (0..nff)
                    .map(|j| (silu(dot(gate_row(e, j), xr)) * dot(up_row(e, j), xr)) as f32)
                    .collect();
                for i in 0..ne {
                    want[t * ne + i] +=
                        probs[e] / wsum * dot(&dd[(e * ne + i) * nff..][..nff], &act);
                }
            }
        }

        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
        let ri = g.weight(TensorDesc::new(vec![N_EXPERT, ne], DType::F32));
        let gi = g.weight(TensorDesc::new(vec![N_EXPERT, gu_rows, ne], BANK));
        let ui = if fused {
            gi
        } else {
            g.weight(TensorDesc::new(vec![N_EXPERT, nff, ne], BANK))
        };
        let di = g.weight(TensorDesc::new(vec![N_EXPERT, ne, nff], BANK));
        let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
        g.push(Op::MoeFfn {
            x: xi,
            router_x: xi,
            router: ri,
            gate_exps: gi,
            up_exps: ui,
            down_exps: di,
            down_scale: None,
            fused_gate_up: fused,
            dst: yi,
            ne: ne as u32,
            n_expert: N_EXPERT as u32,
            n_used: N_USED as u32,
            n_ff_exp: nff as u32,
            scale: 1.0,
            act: Activation::Silu,
            gating: MoeGating::Softmax,
            norm_w: true,
            weight_before: false,
            ep_band: None,
            exp_probs_b: None,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
            swiglu_clamp: None,
            expert_ids: None,
        });
        let mk = |bytes: &[u8], usage| {
            let padded = pad_to_u32_align(bytes);
            let b = be.alloc(padded.len(), usage).unwrap();
            be.upload(b.as_ref(), &padded).unwrap();
            b
        };
        let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
        let rb = mk(bytemuck::cast_slice(&router), BufferUsage::Weights);
        let gb = mk(&gq, BufferUsage::Weights);
        let ub = mk(&uq, BufferUsage::Weights);
        let db = mk(&dq, BufferUsage::Weights);
        let yb = be.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
        let plan = be.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(ri, rb.as_ref());
        bind.bind(gi, gb.as_ref());
        if !fused {
            bind.bind(ui, ub.as_ref());
        }
        bind.bind(di, db.as_ref());
        bind.bind(yi, yb.as_ref());
        be.execute(plan.as_ref(), &bind)
            .unwrap_or_else(|e| panic!("{label}: execute: {e}"));
        let mut got = vec![0f32; rows * ne];
        be.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();

        let rms = (want.iter().map(|w| w * w).sum::<f64>() / want.len() as f64).sqrt();
        let bound = if whole_tiles {
            MAX_REL
        } else {
            SMALL_M_MAX_REL
        };
        let mut worst = 0f64;
        for (i, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
            let d = (f64::from(gv) - wv).abs();
            assert!(
                d.is_finite() && d <= bound * rms,
                "{label}: element {i} (row {}, col {}): got {gv}, want {wv} (rms {rms:e}, \
                 bound {bound:e} of it)",
                i / ne,
                i % ne
            );
            worst = worst.max(d);
        }
        let gemms: Vec<_> = be
            .built_kernel_names()
            .into_iter()
            .filter(|k| is_expert_gemm(k))
            .collect();
        assert_eq!(
            !gemms.is_empty(),
            batched,
            "{label}: the batched expert GEMMs {} (built: {gemms:?})",
            if batched { "did not run" } else { "ran" }
        );
        println!(
            "{label}: max |got - want| = {:.2e} of the RMS, {}",
            worst / rms,
            if batched { "batched" } else { "small-m" }
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn moe_ffn_matches_host_at_every_width() {
    if backend().is_none() {
        return;
    }
    for nff in [32, 64, 96, 128] {
        run(64, nff, false);
    }
    // A model width the down GEMM's column tile does not divide.
    run(96, 64, false);
    // Fused: the gate|up GEMM is 2·n_ff_exp wide, so the n_ff_exp widths that are partial tiles
    // split are whole tiles here and take the batched GEMMs.
    run(64, 32, true);
    run(64, 96, true);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn expert_gemm_refuses_a_partial_column_tile() {
    let Some(be) = infr_testkit::gpu_or_skip_noted(VulkanBackend::new()) else {
        return;
    };
    let (k, rows) = (64usize, 16usize);
    for n in [32usize, 64, 96] {
        let (wq, _) = bank(N_EXPERT * n * k, 6);
        let padded = pad_to_u32_align(&wq);
        let w = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
        be.upload(w.as_ref(), &padded).unwrap();
        let npad = rows.div_ceil(64) * 64 + 64;
        let qa = be.alloc(npad * k, BufferUsage::Activations).unwrap();
        let qda = be
            .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
            .unwrap();
        let counts: Vec<u32> = (0..N_EXPERT as u32).map(|e| u32::from(e < 2) * 8).collect();
        let offsets: Vec<u32> = (0..N_EXPERT as u32).map(|e| e.min(2) * 8).collect();
        let mk_u32 = |v: &[u32]| {
            let b = be.alloc(v.len() * 4, BufferUsage::Activations).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let (cb, ob) = (mk_u32(&counts), mk_u32(&offsets));
        let out = be.alloc(npad * n * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.matmul_mmq_experts(
            BANK,
            "expert_gateup",
            qa.as_ref(),
            qda.as_ref(),
            None,
            w.as_ref(),
            0,
            n * k,
            cb.as_ref(),
            ob.as_ref(),
            out.as_ref(),
            rows,
            k,
            n,
            N_EXPERT,
            1,
        );
        let res = rec.finish();
        if n.is_multiple_of(infr_core::tensor::MOE_EXPERT_GEMM_TILE_N) {
            res.unwrap_or_else(|e| panic!("n={n}: a whole-tile width was refused: {e}"));
        } else {
            let e = res.expect_err("a partial column tile was dispatched");
            assert!(
                e.to_string().contains(&format!("n={n} is not a multiple")),
                "n={n}: refused for another reason: {e}"
            );
        }
    }
}
