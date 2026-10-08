//! Synthetic grid id-GEMV regression: discrete real dimensions, integrated scaled parity only.
//! Submissions are bounded by bench_support, not a replay of a single decode-step submit.
//! Run serially: `cargo test --release --locked --offline -p infr-vulkan --test moe_id_gemv_real_dims -- --include-ignored --nocapture --test-threads=1`
use std::cell::Cell;
use std::time::Duration;

mod bench_support;

use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::linear::pad_to_u32_align;
use infr_vulkan::{Recorder, VulkanBackend};

/// Deterministic byte stream (SplitMix64-ish) so failures reproduce.
struct Rng(u64);
impl Rng {
    fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

/// (elements per block, bytes per block) — grid i-quant subset under test.
fn block_geom(dt: DType) -> (usize, usize) {
    match dt {
        DType::Iq2Xxs => (256, 66),
        DType::Iq2Xs => (256, 74),
        DType::Iq2S => (256, 82),
        DType::Iq3Xxs => (256, 98),
        DType::Iq3S => (256, 110),
        DType::Iq1S => (256, 50),
        DType::Iq1M => (256, 56),
        other => panic!("no geometry for {other:?}"),
    }
}

/// One synthetic VALID block: random payload, scale patched small (same recipe as the small-bank
/// parity suite — see moe_id_gemv_new_formats_parity.rs's module doc for why this is valid).
fn synth_bank(dt: DType, n_elems: usize, seed: u64) -> Vec<u8> {
    let (epb, bpb) = block_geom(dt);
    assert_eq!(n_elems % epb, 0);
    let mut rng = Rng(seed);
    let d16 = half::f16::from_f32(0.02).to_le_bytes();
    let mut out = vec![0u8; n_elems / epb * bpb];
    for blk in out.chunks_exact_mut(bpb) {
        for b in blk.iter_mut() {
            *b = rng.byte();
        }
        match dt {
            DType::Iq1M => {
                let d_bits = half::f16::from_f32(0.02).to_bits();
                for i in 0..4usize {
                    let lo = u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]]) & 0x0FFF;
                    let w = lo | (((d_bits >> (4 * i)) & 0xF) << 12);
                    blk[48 + 2 * i..50 + 2 * i].copy_from_slice(&w.to_le_bytes());
                }
            }
            _ => blk[0..2].copy_from_slice(&d16),
        }
    }
    out
}

fn host_gemv(w: &[f32], x: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
    (0..out_f)
        .map(|o| (0..in_f).map(|i| w[o * in_f + i] * x[i]).sum())
        .collect()
}

/// One real-shape role bank on the GPU plus everything needed to dispatch/check it.
struct Role {
    dt: DType,
    used: usize,
    in_f: usize,
    out_f: usize,
    stride: usize,
    bank: Vec<u8>,
    wbuf: Box<dyn infr_core::backend::Buffer>,
    x_buf: Box<dyn infr_core::backend::Buffer>,
    y_buf: Box<dyn infr_core::backend::Buffer>,
    x: Vec<f32>,
}

const N_LAYER: usize = 40;

#[derive(Debug)]
struct Case {
    experts: usize,
    ids: &'static [u32],
    gate: (usize, usize),
    banks: usize,
    integrated: bool,
}

impl Case {
    fn selected(integrated: bool) -> Self {
        if integrated {
            Self {
                experts: 8,
                ids: &[7, 0, 4, 2],
                gate: (512, 256),
                banks: 2,
                integrated,
            }
        } else {
            Self {
                experts: 256,
                ids: &[255, 0, 128, 37, 200, 91, 250, 3],
                gate: (2048, 512),
                banks: 8,
                integrated,
            }
        }
    }
}

fn selected_backend() -> Option<(VulkanBackend, Case)> {
    let be = bench_support::optional_backend()?;
    let case = Case::selected(be.capabilities().integrated);
    // The grid mirrors plus the tree kernel's reduction scratch must fit before any allocation.
    use infr_core::iquant_grids::*;
    let grid_bytes = [
        IQ2XXS_GRID.len() * 8 + KSIGNS_IQ2XS.len() * 4,
        IQ2XS_GRID.len() * 8 + KSIGNS_IQ2XS.len() * 4,
        IQ2S_GRID.len() * 8,
        IQ3XXS_GRID.len() * 4 + KSIGNS_IQ2XS.len() * 4,
        IQ3S_GRID.len() * 4,
        IQ1S_GRID.len() * 8,
    ]
    .into_iter()
    .max()
    .unwrap();
    if !be.capabilities().buffer_device_address
        || (be.max_shared_memory_bytes() as usize) < grid_bytes + 8 * 4
    {
        println!("excluded: native id-GEMV requires BDA and sufficient grid shared memory");
        return None;
    }
    println!(
        "coverage={} case={case:?}; excludes single-submit decode replay{}",
        if case.integrated {
            "integrated scaled parity, NOT real dimensions"
        } else {
            "discrete real dimensions"
        },
        if case.integrated {
            " and discrete timing budgets"
        } else {
            ""
        }
    );
    Some((be, case))
}

fn make_role(
    be: &VulkanBackend,
    case: &Case,
    dt: DType,
    in_f: usize,
    out_f: usize,
    seed: u64,
) -> Role {
    let stride = in_f * out_f;
    let bank = synth_bank(dt, case.experts * stride, 0x5eed ^ dt as u64 ^ (seed << 32));
    let padded = pad_to_u32_align(&bank);
    let wbuf = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
    be.upload(wbuf.as_ref(), &padded).unwrap();
    let x: Vec<f32> = (0..in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let y_buf = be
        .alloc(case.ids.len() * out_f * 4, BufferUsage::Activations)
        .unwrap();
    Role {
        dt,
        used: case.ids.len(),
        in_f,
        out_f,
        stride,
        bank,
        wbuf,
        x_buf,
        y_buf,
        x,
    }
}

fn record_role(rec: &Recorder, role: &Role, ids_buf: &dyn infr_core::backend::Buffer) {
    rec.linear_native_id_multi(
        role.dt,
        role.wbuf.as_ref(),
        ids_buf,
        role.used,
        role.stride,
        role.x_buf.as_ref(),
        false,
        role.y_buf.as_ref(),
        role.in_f,
        role.out_f,
        1,
    );
}

fn check_role(role: &Role, be: &VulkanBackend, ids: &[u32]) {
    let (epb, bpb) = block_geom(role.dt);
    let stride_bytes = role.stride / epb * bpb;
    let mut out = vec![0u8; role.used * role.out_f * 4];
    be.download(role.y_buf.as_ref(), &mut out).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();
    for (slot, &eid) in ids.iter().enumerate() {
        let eb = &role.bank[eid as usize * stride_bytes..(eid as usize + 1) * stride_bytes];
        let w = infr_gguf::dequant::dequant_block(role.dt, eb).unwrap();
        let want = host_gemv(&w, &role.x, role.in_f, role.out_f);
        let g = &got[slot * role.out_f..(slot + 1) * role.out_f];
        assert!(g.iter().all(|v| v.is_finite()));
        assert!(want.iter().all(|v| v.is_finite()));
        assert!(g.iter().any(|v| *v != 0.0));
        assert!(want.iter().any(|v| *v != 0.0));
        for (i, (gv, wv)) in g.iter().zip(want.iter()).enumerate() {
            assert!(
                (gv - wv).abs() < 1e-3 + 1e-3 * wv.abs(),
                "{:?} slot {slot} (expert {eid}) mismatch at {i}: got {gv} want {wv}",
                role.dt
            );
        }
    }
}

fn measured_timing(timing: &bench_support::Timing, required: bool) -> Option<f64> {
    let mean = timing.mean_us();
    if required {
        assert_eq!(
            timing.outcome,
            bench_support::Outcome::Complete,
            "required timing is incomplete"
        );
        assert!(mean.is_some(), "required timing has no measured samples");
    }
    mean
}

#[test]
fn required_timing_rejects_incomplete_samples() {
    use bench_support::{Batch, Outcome, Timing};
    for outcome in [
        Outcome::Stopped { remaining: 120 },
        Outcome::Stopped { remaining: 0 },
        Outcome::InsufficientSamples,
        Outcome::Complete,
    ] {
        let mut timing = Timing {
            cold: Batch {
                operations: 1,
                dispatches: 1,
                elapsed: Duration::from_millis(51),
            },
            batches: Vec::new(),
            outcome,
        };
        assert_eq!(measured_timing(&timing, false), None);
        assert!(
            std::panic::catch_unwind(|| measured_timing(&timing, true)).is_err(),
            "required timing accepted {:?} without measurements",
            timing.outcome
        );
        timing.batches.push(Batch {
            operations: 1,
            dispatches: 1,
            elapsed: Duration::from_millis(1),
        });
        if timing.outcome == Outcome::Complete {
            assert_eq!(measured_timing(&timing, true), Some(1000.0));
        } else {
            assert_eq!(measured_timing(&timing, false), None);
            assert!(std::panic::catch_unwind(|| measured_timing(&timing, true)).is_err());
        }
    }
}

#[test]
fn selected_profile_invariants() {
    for integrated in [true, false] {
        let case = Case::selected(integrated);
        assert_eq!(case.experts, if integrated { 8 } else { 256 });
        assert_eq!(case.ids.len(), if integrated { 4 } else { 8 });
        assert_eq!(case.banks, if integrated { 2 } else { 8 });
        assert_eq!(case.gate, if integrated { (512, 256) } else { (2048, 512) });
        assert_eq!(case.ids[0] as usize, case.experts - 1);
        assert_eq!(case.ids[1], 0);
        for (i, id) in case.ids.iter().enumerate() {
            assert!((*id as usize) < case.experts);
            assert!(!case.ids[..i].contains(id));
        }
        assert!(case.ids.windows(2).any(|w| w[0] > w[1]));
        assert_eq!(N_LAYER % case.banks, 0);
        assert_eq!(case.gate.0 % 256, 0);
        assert_eq!(case.gate.1 % 256, 0);
        let a = synth_bank(DType::Iq2S, 256, 0);
        let b = synth_bank(DType::Iq2S, 256, 1 << 32);
        assert_ne!(a, b);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn real_dims_decode_step_iq2s_iq3s() {
    let Some((be, case)) = selected_backend() else {
        return;
    };
    let ids_buf = be
        .alloc(case.ids.len() * 4, BufferUsage::Activations)
        .unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(case.ids))
        .unwrap();
    let gates: Vec<_> = (0..case.banks)
        .map(|i| make_role(&be, &case, DType::Iq2S, case.gate.0, case.gate.1, i as u64))
        .collect();
    let downs: Vec<_> = (0..case.banks)
        .map(|i| make_role(&be, &case, DType::Iq3S, case.gate.1, case.gate.0, i as u64))
        .collect();
    for (orientation, roles) in [("gate", &gates), ("down", &downs)] {
        for (bank, role) in roles.iter().enumerate() {
            bench_support::time(
                &be,
                &format!("warm/check {orientation} bank={bank}"),
                0,
                |rec| record_role(rec, role, ids_buf.as_ref()),
            )
            .unwrap();
            check_role(role, &be, case.ids);
            println!("checked {:?} {orientation} bank={bank}", role.dt);
        }
    }
    // Cold operation uses the last layer; measured operations start at layer zero.
    let index = Cell::new(N_LAYER * 3 - 1);
    let timing = bench_support::time(&be, "bounded gate/up/down sequence", N_LAYER * 3, |rec| {
        let i = index.get();
        let bank = (i / 3) % case.banks;
        let role = if i % 3 == 2 {
            &downs[bank]
        } else {
            &gates[bank]
        };
        record_role(rec, role, ids_buf.as_ref());
        index.set((i + 1) % (N_LAYER * 3));
    })
    .unwrap();
    for (orientation, roles) in [("gate", &gates), ("down", &downs)] {
        for (bank, role) in roles.iter().enumerate() {
            check_role(role, &be, case.ids);
            println!(
                "checked after sequence {:?} {orientation} bank={bank}",
                role.dt
            );
        }
    }
    if measured_timing(&timing, !case.integrated).is_some() {
        let elapsed: Duration = timing.batches.iter().map(|b| b.elapsed).sum();
        println!(
            "complete bounded sequence: {elapsed:?}; includes submission seams, excludes cold"
        );
        if !case.integrated {
            assert!(
                elapsed.as_secs_f64() < 0.2,
                "decode-step bounded sequence took {elapsed:?}"
            );
        }
    } else {
        println!(
            "sequence incomplete: {:?}; no full-sequence timing coverage",
            timing.outcome
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn real_dims_all_grid_formats() {
    let Some((be, case)) = selected_backend() else {
        return;
    };
    let ids_buf = be
        .alloc(case.ids.len() * 4, BufferUsage::Activations)
        .unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(case.ids))
        .unwrap();
    let mut checked = 0;
    for dt in [
        DType::Iq2S,
        DType::Iq3S,
        DType::Iq2Xxs,
        DType::Iq2Xs,
        DType::Iq3Xxs,
        DType::Iq1S,
        DType::Iq1M,
    ] {
        for (orientation, (in_f, out_f)) in
            [("gate", case.gate), ("down", (case.gate.1, case.gate.0))]
        {
            for bank in 0..case.banks {
                let role = make_role(&be, &case, dt, in_f, out_f, bank as u64);
                let label = format!("{dt:?} {orientation} bank={bank} {in_f}x{out_f}");
                bench_support::time(&be, &format!("warm/check {label}"), 0, |rec| {
                    record_role(rec, &role, ids_buf.as_ref())
                })
                .unwrap();
                check_role(&role, &be, case.ids);
                let timing = bench_support::time(&be, &label, 1, |rec| {
                    record_role(rec, &role, ids_buf.as_ref())
                })
                .unwrap();
                check_role(&role, &be, case.ids);
                checked += 1;
                println!(
                    "checked after timing {label}; cumulative checked banks/orientations={checked}"
                );
                let required = !case.integrated
                    && orientation == "gate"
                    && !matches!(dt, DType::Iq2S | DType::Iq3S);
                if let Some(us) = measured_timing(&timing, required) {
                    if required {
                        assert!(us < 3000.0, "{label} single dispatch took {us} us");
                    }
                } else {
                    println!("{label}: no measured timing coverage: {:?}", timing.outcome);
                }
            }
        }
    }
    println!("parity complete: checked banks/orientations={checked}");
}
