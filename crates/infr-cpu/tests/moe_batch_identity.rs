//! Prefill `Op::MoeFfn` against decode `Op::MoeFfn`, bit for bit, on every expert bank format.
//!
//! The CPU backend feeds a quantized expert bank (Q4_K, Q5_K, Q6_K and Q8_0 at a 256-multiple
//! width; Q8_0 and Q5_0 at a 32-multiple one) int8 activations at every row count: each row
//! quantized on its own, then the bank's expert kernels over the bucket of rows routed to each
//! expert: per-row kernels, the Q4_K and 32-block 8-row tiles, on AVX-512 VNNI hosts the packed
//! Q5_K/Q6_K GEMM above its token threshold and, for a fused Q4_K gate/up bank, the cached packed
//! groups. A chunked prefill is the same computation as one position per graph (and as the decode
//! steps after it) only if every one of those routes gives a row the bits it gets alone. This
//! file pins that through the public op: one `execute` over `R` rows against `R` executes of one
//! row each, on the same bytes, at every SIMD tier the host has (`kernels.cpu.max_tier`) and
//! uncapped, and in reference mode (f32 activations at every row count).
//!
//! It also checks that one row really takes the int8 path: on a quantized bank the one-row
//! production output differs from the one-row reference output, and on a bank with no int8
//! kernel (Q2_K, F16) it equals it.
//!
//! The ceiling is process-wide, so every case runs inside one test.
//!
//! What it cannot see: a wrong sparse-part add (`Q8::hi`, `kernels::vec_dot_hi`) in the expert
//! GEMM. Both sides of each comparison take the same `moe::expert_gemm_range` route, so a fault
//! there moves both alike. `moe::tests::expert_gemm_range_matches_the_single_token_kernels_bitwise`
//! catches it (its rows split, and it holds the GEMM to the one-row kernel plus `vec_dot_hi`), and
//! `op_hashes`' spiky `MoeFfn` cases pin the value across tiers, thread counts and systems.
//!
//! Run: `cargo nextest run -p infr-cpu --test moe_batch_identity`

use std::sync::Arc;

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::config::Config;
use infr_core::graph::{Activation, Graph, MoeGating, Op};
use infr_core::simd::{self, SimdTier};
use infr_core::tensor::{DType, TensorDesc};
use infr_cpu::CpuBackend;

/// Experts, experts per row, and the row counts a batch runs at: a pair, a few rows (every
/// bucket under the packed Q6_K threshold or near it), and enough that buckets pass each packed
/// threshold and the 4-token GEMM tile with tails.
const N_EXPERT: usize = 8;
const N_USED: usize = 3;
const ROWS: [usize; 3] = [2, 5, 29];

/// One bank configuration.
struct Bank {
    name: &'static str,
    dt: DType,
    router: DType,
    fused: bool,
    ne: usize,
    nff: usize,
    /// Whether production quantizes this bank's activations to int8.
    int8: bool,
}

fn banks() -> Vec<Bank> {
    let b = |name, dt, router, fused, ne, nff, int8| Bank {
        name,
        dt,
        router,
        fused,
        ne,
        nff,
        int8,
    };
    vec![
        b("Q4_K", DType::Q4K, DType::F32, false, 256, 256, true),
        b(
            "Q4_K fused gate/up",
            DType::Q4K,
            DType::F32,
            true,
            256,
            256,
            true,
        ),
        b("Q5_K", DType::Q5K, DType::F32, false, 256, 512, true),
        b("Q6_K", DType::Q6K, DType::F32, false, 512, 256, true),
        b("Q8_0", DType::Q8_0, DType::F32, false, 256, 256, true),
        b(
            "Q8_0 32-block",
            DType::Q8_0,
            DType::F32,
            false,
            288,
            96,
            true,
        ),
        b("Q5_0", DType::Q5_0, DType::F32, false, 256, 288, true),
        b(
            "Q4_K, Q8_0 router",
            DType::Q4K,
            DType::Q8_0,
            false,
            256,
            256,
            true,
        ),
        b("Q2_K", DType::Q2K, DType::F32, false, 256, 256, false),
        b("F16", DType::F16, DType::F32, false, 64, 96, false),
    ]
}

/// Element `i` of stream `seed`: a multiple of 2^-9 in about [-2, 2), from integers only.
fn val(seed: u64, i: usize) -> f32 {
    let h = (i as u64 + 1)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(seed.wrapping_mul(0xbf58_476d_1ce4_e5b9));
    let h = (h ^ (h >> 29)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((h >> 40) % 2048) as f32 / 512.0 - 2.0
}

fn weight_bytes(dt: DType, n: usize, seed: u32) -> Vec<u8> {
    match dt {
        DType::F32 => (0..n)
            .flat_map(|i| (val(seed as u64, i) / 16.0).to_le_bytes())
            .collect(),
        DType::F16 => (0..n)
            .flat_map(|i| half::f16::from_f32(val(seed as u64, i) / 16.0).to_le_bytes())
            .collect(),
        _ => infr_testkit::synth_weight(dt, n, seed),
    }
}

/// The op over `rows` rows of `x` (row-major, `ne` wide), on `be`.
fn run(be: &CpuBackend, bank: &Bank, x: &[f32], rows: usize) -> Vec<f32> {
    let (ne, nff) = (bank.ne, bank.nff);
    let mut g = Graph::new();
    let xin = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
    let router = g.weight(TensorDesc::new(vec![N_EXPERT, ne], bank.router));
    let gu_rows = if bank.fused { 2 * nff } else { nff };
    let gate = g.weight(TensorDesc::new(vec![N_EXPERT, gu_rows, ne], bank.dt));
    let up = g.weight(TensorDesc::new(vec![N_EXPERT, nff, ne], bank.dt));
    let down = g.weight(TensorDesc::new(vec![N_EXPERT, ne, nff], bank.dt));
    let dst = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
    g.push(Op::MoeFfn {
        x: xin,
        router_x: xin,
        router,
        gate_exps: gate,
        up_exps: up,
        down_exps: down,
        down_scale: None,
        fused_gate_up: bank.fused,
        dst,
        ne: ne as u32,
        n_expert: N_EXPERT as u32,
        n_used: N_USED as u32,
        n_ff_exp: nff as u32,
        scale: 1.0,
        act: Activation::Silu,
        swiglu_clamp: None,
        gating: MoeGating::Softmax,
        norm_w: true,
        weight_before: false,
        ep_band: None,
        exp_probs_b: None,
        n_expert_groups: 0,
        n_expert_groups_used: 0,
        expert_ids: None,
    });
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bound = [
        (xin, x_bytes, BufferUsage::Activations),
        (
            router,
            weight_bytes(bank.router, N_EXPERT * ne, 31),
            BufferUsage::Weights,
        ),
        (
            gate,
            weight_bytes(bank.dt, N_EXPERT * gu_rows * ne, 32),
            BufferUsage::Weights,
        ),
        (
            up,
            weight_bytes(bank.dt, N_EXPERT * nff * ne, 33),
            BufferUsage::Weights,
        ),
        (
            down,
            weight_bytes(bank.dt, N_EXPERT * ne * nff, 34),
            BufferUsage::Weights,
        ),
    ];
    let plan = be.compile(&g).expect("compile");
    let mut bufs = Vec::new();
    for (id, bytes, usage) in bound {
        let b = be.alloc(bytes.len(), usage).expect("alloc");
        be.upload(b.as_ref(), &bytes).expect("upload");
        bufs.push((id, b));
    }
    let out = be
        .alloc(rows * ne * 4, BufferUsage::Readback)
        .expect("alloc out");
    let mut binds = Bindings::new();
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    binds.bind(dst, out.as_ref());
    be.execute(plan.as_ref(), &binds).expect("execute");
    let mut o = vec![0u8; rows * ne * 4];
    be.download(out.as_ref(), &mut o).expect("download");
    o.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// A backend at `tier` (`None`: no ceiling, the uncapped interpreter), in reference mode or not.
fn backend(tier: Option<SimdTier>, reference: bool) -> CpuBackend {
    let mut cfg = Config::default();
    cfg.kernels.cpu.max_tier = tier;
    cfg.kernels.cpu.reference = reference;
    let be = CpuBackend::new_with(Arc::new(cfg));
    assert_eq!(simd::active(), tier.unwrap_or_else(SimdTier::best));
    be
}

/// Each row of an `R`-row `Op::MoeFfn` equals that row run alone, bit for bit, for every bank
/// format, at every tier the host has and uncapped, in production and in reference mode; and one
/// row takes int8 activations exactly where a batch does. Turns red if a row's activations are
/// quantized with anything but their own scale, if an expert kernel's per-row result depends on
/// its bucket's size or tile, or if one row and several take different activation paths.
#[test]
fn prefill_moe_matches_one_row_moe_bitwise_for_every_bank_format() {
    let mut tiers: Vec<Option<SimdTier>> = SimdTier::ALL
        .into_iter()
        .filter(|t| t.supported())
        .map(Some)
        .collect();
    tiers.push(None);
    let mut checked = 0usize;
    let mut failures = Vec::new();
    for bank in banks() {
        let rmax = *ROWS.iter().max().unwrap();
        let x: Vec<f32> = (0..rmax * bank.ne).map(|i| val(7, i)).collect();
        for &tier in &tiers {
            let tname = tier.map_or("uncapped", |t| t.name());
            let mut alone = [Vec::new(), Vec::new()];
            for (k, reference) in [false, true].into_iter().enumerate() {
                let be = backend(tier, reference);
                alone[k] = (0..rmax)
                    .map(|r| run(&be, &bank, &x[r * bank.ne..(r + 1) * bank.ne], 1))
                    .collect::<Vec<_>>();
                for &rows in &ROWS {
                    let batch = run(&be, &bank, &x[..rows * bank.ne], rows);
                    for (r, one) in alone[k][..rows].iter().enumerate() {
                        let got = &batch[r * bank.ne..(r + 1) * bank.ne];
                        let diff = got
                            .iter()
                            .zip(one)
                            .position(|(a, b)| a.to_bits() != b.to_bits());
                        if let Some(i) = diff {
                            failures.push(format!(
                                "{} at {tname}{}: row {r} of {rows}, element {i}: {:e} in the \
                                 batch, {:e} alone",
                                bank.name,
                                if reference { " (reference)" } else { "" },
                                got[i],
                                one[i]
                            ));
                        }
                        checked += 1;
                    }
                }
            }
            // One row is on the int8 path exactly where the format has one: production and
            // reference then differ, and elsewhere they are the same computation.
            let same = alone[0] == alone[1];
            assert!(
                alone[0].iter().flatten().any(|v| *v != 0.0),
                "{}: the op wrote only zeros",
                bank.name
            );
            assert_eq!(
                !same,
                bank.int8,
                "{} at {tname}: one-row production {} one-row reference",
                bank.name,
                if same { "equals" } else { "differs from" }
            );
        }
    }
    assert!(
        failures.is_empty(),
        "{} rows differ from their one-row run:\n{}",
        failures.len(),
        failures.join("\n")
    );
    let per: usize = ROWS.iter().sum();
    assert_eq!(checked, banks().len() * tiers.len() * 2 * per);
    simd::set_max(None);
}
