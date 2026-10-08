//! PROBE (docs/backlog.md B7, slice 1) — LDS-staged **K-tile** decode attention pass 1.
//!
//! `attn_partial.comp` gives each 32-lane wave ONE key and reduces that key's 128-dim dot with a
//! cross-lane `subgroupAdd`; that reduction ALU scales with keys x heads and is 59% of decode GPU
//! time at d32768 (177 us per layer-token on a 7900 XTX). `attn_ktile.comp` instead stages a tile
//! of K in shared memory with coalesced global reads and gives each THREAD a whole 128-dim dot, so
//! the per-key cross-lane reduction disappears. This file is the ONLY caller of that kernel —
//! nothing in production dispatches it.
//!
//! Two tests:
//!  * `ktile_matches_split_reference` — combined output vs the shipped `attention_kv_split_at` at
//!    several shapes. Bitwise equality is NOT expected (the key→thread mapping, and therefore the
//!    dot summation order, differs by design), so this is a tight RELATIVE tolerance; the reference
//!    is first asserted non-zero and all-finite so the compare cannot pass vacuously.
//!  * Ignored timing uses the same shape roster as parity. Integrated devices select small
//!    MHA/GQA cases before allocation; discrete devices retain the long-context cases.
//!    Timings include record/submit/wait and stop when the bounded helper declines more work.
//!
//! Run: `cargo test --release -p infr-vulkan --test attn_ktile_probe -- --include-ignored --nocapture --test-threads=1`
//! (the cargo wrapper swallows test stdout — run `target/release/deps/attn_ktile_probe-*` directly).
mod bench_support;

use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_vulkan::{Recorder, VulkanBackend};

/// The four `attn_ktile` build configurations, as `Recorder::attention_kv_split_ktile_at`'s `cfg`.
/// LDS figures are the K tile only (`keys * row_stride_words * 4`); each adds ~3.8 KB of
/// `sc`/`qf4`/`red`/`vsh` on top.
const CFGS: &[(u32, &str)] = &[
    (0, "w64      (64-key tile, 68-word rows, 17.0 KB K-LDS)"),
    (1, "w64_nopad(64-key tile, 64-word rows, 16.0 KB K-LDS)"),
    (2, "w128     (128-key tile, 68-word rows, 34.0 KB K-LDS)"),
    (3, "w64_dw32 (64-key tile, half-depth stage, 9.0 KB K-LDS)"),
];

// Mirrors build.rs defines and every shared array in attn_ktile.comp.
fn shared_bytes(cfg: u32) -> u32 {
    let (kwg, kdw, kpad) = match cfg {
        0 => (64, 64, 4),
        1 => (64, 64, 0),
        2 => (128, 64, 4),
        3 => (64, 32, 4),
        _ => panic!("unknown ktile config {cfg}"),
    };
    kwg * (kdw + kpad) * 4 + 32 * 16 + 512 * 4 + kwg * 4 + kwg * 16
}

fn eligible_configs(capacity: u32) -> Vec<(u32, &'static str)> {
    CFGS.iter()
        .copied()
        .filter(|&(cfg, name)| {
            let required = shared_bytes(cfg);
            if required > capacity {
                println!(
                    "skip {name}: shared memory required={required} capacity={capacity} bytes"
                );
                false
            } else {
                true
            }
        })
        .collect()
}

#[test]
fn shared_memory_eligibility_boundaries() {
    assert!(eligible_configs(0).is_empty());
    for &(cfg, _) in CFGS {
        let required = shared_bytes(cfg);
        assert!(!eligible_configs(required - 1)
            .iter()
            .any(|&(id, _)| id == cfg));
        assert!(eligible_configs(required).iter().any(|&(id, _)| id == cfg));
    }
    assert_eq!(shared_bytes(2), 39936);
    assert_eq!(
        eligible_configs(32768)
            .iter()
            .map(|&(id, _)| id)
            .collect::<Vec<_>>(),
        [0, 1, 3]
    );
}

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    }
}

/// `n` f16 elements drawn from [-1, 1). SIGNED (unlike kv_addr_parity's masked-bits helper): a
/// non-negative K/Q makes every score large and positive, which lets one key dominate the softmax
/// and hides disagreement in the rest — the sign is what keeps this comparison discriminating.
fn f16_data(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        out.extend_from_slice(&half::f16::from_f32(r.next_f32()).to_bits().to_le_bytes());
    }
    out
}

struct Shape {
    kv_len: usize,
    nh: usize,
    nkv: usize,
    chunk: usize,
}

const HD: usize = 128;

fn shapes(integrated: bool) -> Vec<Shape> {
    if integrated {
        return vec![
            Shape {
                kv_len: 40,
                nh: 4,
                nkv: 1,
                chunk: 32,
            },
            Shape {
                kv_len: 129,
                nh: 4,
                nkv: 1,
                chunk: 128,
            },
            Shape {
                kv_len: 257,
                nh: 4,
                nkv: 1,
                chunk: 256,
            },
            Shape {
                kv_len: 512,
                nh: 2,
                nkv: 2,
                chunk: 128,
            },
        ];
    }
    vec![
        // The B7 decode shape, scaled down: GQA g=8, several full chunks.
        Shape {
            kv_len: 2048,
            nh: 32,
            nkv: 4,
            chunk: 512,
        },
        // Ragged last chunk (1000 = 3*256 + 232) — exercises the partial-tile stage guard.
        Shape {
            kv_len: 1000,
            nh: 32,
            nkv: 4,
            chunk: 256,
        },
        // A chunk holding a SINGLE key (513 = 512 + 1): 63 of 64 threads idle in that workgroup.
        Shape {
            kv_len: 513,
            nh: 16,
            nkv: 2,
            chunk: 512,
        },
        // MHA (g == 1, nh == nkv) — the other end of the workgroup→(head, chunk) decomposition.
        Shape {
            kv_len: 1024,
            nh: 8,
            nkv: 8,
            chunk: 512,
        },
        // kv_len below one tile (64 keys) so the tile loop runs exactly once, mostly masked.
        Shape {
            kv_len: 40,
            nh: 8,
            nkv: 2,
            chunk: 32,
        },
        Shape {
            kv_len: 8192,
            nh: 32,
            nkv: 4,
            chunk: 512,
        },
        Shape {
            kv_len: 32768,
            nh: 32,
            nkv: 4,
            chunk: 512,
        },
    ]
}

/// Allocates one case's buffers and returns `(reference_o, ktile_o[cfg])`.
fn run_case(be: &VulkanBackend, s: &Shape, cfgs: &[(u32, &str)]) -> (Vec<f32>, Vec<Vec<f32>>) {
    let Shape {
        kv_len,
        nh,
        nkv,
        chunk,
    } = *s;
    let n_chunks = kv_len.div_ceil(chunk);
    let pos = kv_len - 1; // rows == 1: the decode query is position kv_len-1
    let cache_elems = kv_len * nkv * HD;

    let qb = be.alloc(nh * HD * 2, BufferUsage::Activations).unwrap();
    be.upload(qb.as_ref(), &f16_data(nh * HD, 101)).unwrap();
    let kb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
    let vb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
    be.upload(kb.as_ref(), &f16_data(cache_elems, 11)).unwrap();
    be.upload(vb.as_ref(), &f16_data(cache_elems, 23)).unwrap();
    let ka = kb.device_addr().expect("KvCache K device address");
    let va = vb.device_addr().expect("KvCache V device address");

    let pm = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(nh * n_chunks * HD * 4, BufferUsage::Activations)
        .unwrap();
    let o_bytes = nh * HD * 4;

    let read = |b: &dyn Buffer| -> Vec<f32> {
        let mut out = vec![0u8; o_bytes];
        be.download(b, &mut out).unwrap();
        bytemuck::cast_slice::<u8, f32>(&out).to_vec()
    };

    let o_ref = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    reference(
        &rec,
        qb.as_ref(),
        kb.as_ref(),
        vb.as_ref(),
        ka,
        va,
        o_ref.as_ref(),
        pm.as_ref(),
        pl.as_ref(),
        pacc.as_ref(),
        pos,
        kv_len,
        nh,
        nkv,
        chunk,
        n_chunks,
    );
    rec.finish().unwrap();
    let want = read(o_ref.as_ref());

    let mut got = Vec::new();
    for &(cfg, _) in cfgs {
        let o = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_kv_split_ktile_at(
            qb.as_ref(),
            kb.as_ref(),
            vb.as_ref(),
            ka,
            va,
            o.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            pos,
            kv_len,
            nh,
            nkv,
            HD,
            chunk,
            n_chunks,
            0.0,
            cfg,
        );
        rec.finish().unwrap();
        got.push(read(o.as_ref()));
    }
    (want, got)
}

/// The shipped split-K decode path (`attn_partial_bda` + `attn_combine`), f16 K/V by device
/// address, full causal, no window/canvas/Q8/ring — the exact configuration `attn_ktile` targets.
#[allow(clippy::too_many_arguments)]
fn reference(
    rec: &Recorder,
    q: &dyn Buffer,
    kc: &dyn Buffer,
    vc: &dyn Buffer,
    ka: u64,
    va: u64,
    o: &dyn Buffer,
    pm: &dyn Buffer,
    pl: &dyn Buffer,
    pacc: &dyn Buffer,
    pos: usize,
    kv_len: usize,
    nh: usize,
    nkv: usize,
    chunk: usize,
    n_chunks: usize,
) {
    rec.attention_kv_split_at(
        q, kc, vc, ka, va, o, pm, pl, pacc, 1, pos, kv_len, nh, nkv, HD, chunk, n_chunks, 0.0, 0,
        None, false, false, 0, false, None,
    );
}

#[test]
fn ktile_matches_split_reference() {
    let Some(be) = bench_support::optional_backend() else {
        return;
    };
    let cfgs = eligible_configs(be.max_shared_memory_bytes());
    if cfgs.is_empty() {
        println!("no supported configurations");
        return;
    }
    let cases = shapes(be.capabilities().integrated);
    let mut worst = 0f32;
    for s in &cases {
        let (want, got) = run_case(&be, s, &cfgs);
        // Non-vacuity: the reference itself must be finite and carry real signal.
        assert!(
            want.iter().all(|v| v.is_finite()),
            "reference output has non-finite values (kv_len={})",
            s.kv_len
        );
        let nz = want.iter().filter(|v| v.abs() > 1e-6).count();
        assert!(
            nz * 4 > want.len() * 3,
            "reference output is mostly zero ({nz}/{}) — the compare would be vacuous",
            want.len()
        );
        for (gi, g) in got.iter().enumerate() {
            let (_, name) = cfgs[gi];
            for i in 0..want.len() {
                assert!(
                    g[i].is_finite(),
                    "{name} kv_len={} idx {i} not finite",
                    s.kv_len
                );
                let denom = want[i].abs().max(0.05); // outputs are O(0.1); floor keeps near-zeros sane
                let rel = (want[i] - g[i]).abs() / denom;
                worst = worst.max(rel);
                assert!(
                    rel <= 1e-3,
                    "{name} kv_len={} nh={} chunk={}: head {} dim {} reference {} vs ktile {} (rel {rel:.3e})",
                    s.kv_len,
                    s.nh,
                    s.chunk,
                    i / HD,
                    i % HD,
                    want[i],
                    g[i]
                );
            }
        }
    }
    eprintln!(
        "attn_ktile == attention_kv_split_at across {} shapes x {} configs; worst rel {worst:.3e}",
        cases.len(),
        cfgs.len()
    );
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench); run alone, nothing else on the GPU"]
fn ktile_bench() {
    let be = bench_support::backend();
    let integrated = be.capabilities().integrated;
    let reps = if integrated { 3 } else { 200 };
    let cfgs = eligible_configs(be.max_shared_memory_bytes());
    if cfgs.is_empty() {
        println!("no supported configurations");
        return;
    }
    for Shape {
        kv_len,
        nh,
        nkv,
        chunk,
    } in shapes(integrated)
    {
        println!("kv_len={kv_len} nh={nh} nkv={nkv} hd={HD} chunk={chunk}");
        let n_chunks = kv_len.div_ceil(chunk);
        let pos = kv_len - 1;
        let cache_elems = kv_len * nkv * HD;
        let qb = be.alloc(nh * HD * 2, BufferUsage::Activations).unwrap();
        be.upload(qb.as_ref(), &f16_data(nh * HD, 101)).unwrap();
        let kb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
        let vb = be.alloc(cache_elems * 2, BufferUsage::KvCache).unwrap();
        be.upload(kb.as_ref(), &f16_data(cache_elems, 11)).unwrap();
        be.upload(vb.as_ref(), &f16_data(cache_elems, 23)).unwrap();
        let ka = kb.device_addr().unwrap();
        let va = vb.device_addr().unwrap();
        let pm = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pl = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pacc = be
            .alloc(nh * n_chunks * HD * 4, BufferUsage::Activations)
            .unwrap();
        let o = be.alloc(nh * HD * 4, BufferUsage::Activations).unwrap();

        let run_ref = |rec: &Recorder| {
            reference(
                rec,
                qb.as_ref(),
                kb.as_ref(),
                vb.as_ref(),
                ka,
                va,
                o.as_ref(),
                pm.as_ref(),
                pl.as_ref(),
                pacc.as_ref(),
                pos,
                kv_len,
                nh,
                nkv,
                chunk,
                n_chunks,
            );
        };

        let rounds = if integrated { 1 } else { 3 };
        let mut samples = vec![Vec::<f64>::new(); cfgs.len()];
        let mut ratios = samples.clone();
        for _ in 0..rounds {
            let reference_timing =
                bench_support::time(&be, "reference before", reps, run_ref).unwrap();
            let Some(mut r) = reference_timing.mean_us() else {
                println!("timing unavailable: {:?}", reference_timing.outcome);
                return;
            };
            for (i, &(cfg, name)) in cfgs.iter().enumerate() {
                let timing = bench_support::time(&be, name, reps, |rec| {
                    rec.attention_kv_split_ktile_at(
                        qb.as_ref(),
                        kb.as_ref(),
                        vb.as_ref(),
                        ka,
                        va,
                        o.as_ref(),
                        pm.as_ref(),
                        pl.as_ref(),
                        pacc.as_ref(),
                        pos,
                        kv_len,
                        nh,
                        nkv,
                        HD,
                        chunk,
                        n_chunks,
                        0.0,
                        cfg,
                    );
                })
                .unwrap();
                let Some(m) = timing.mean_us() else {
                    println!("{name}: timing unavailable: {:?}", timing.outcome);
                    return;
                };
                let after = bench_support::time(&be, "reference after", reps, run_ref).unwrap();
                let Some(next_r) = after.mean_us() else {
                    println!("timing unavailable: {:?}", after.outcome);
                    return;
                };
                samples[i].push(m);
                ratios[i].push((r + next_r) / (2.0 * m));
                r = next_r;
            }
        }
        for (i, &(_, name)) in cfgs.iter().enumerate() {
            samples[i].sort_by(f64::total_cmp);
            ratios[i].sort_by(f64::total_cmp);
            println!(
                "{name}: median {:.1} us/op (spread {:.1}..{:.1}), paired vs reference median {:.2}x (spread {:.2}..{:.2})",
                samples[i][rounds / 2],
                samples[i][0],
                samples[i][rounds - 1],
                ratios[i][rounds / 2],
                ratios[i][0],
                ratios[i][rounds - 1]
            );
        }
    }
}
