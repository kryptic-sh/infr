//! The production CPU's int8 activations against the CPU's own float reference
//! (`kernels.cpu.reference`), position by position, on the reference's own greedy tokens.
//!
//! Measurements, run by hand where the models are cached (one model per `INFR_AQ_MODELS` key,
//! comma-separated):
//!
//! ```text
//! INFR_AQ_MODELS=smollm3-3b,gemma3-1b cargo test --release -p infr-llama \
//!   --test cpu_act_quant -- --ignored --nocapture --test-threads 1
//! ```

use infr_cpu::linear_tap::{self, ActSim};
use infr_llama::seam::{ForcedLogits, ForcedOn, PrefillHead};
use infr_llama::tolerance::{score_row, RowScore};
use infr_llama::{EngineConfig, SeamModel};
use std::path::PathBuf;

/// The cached models, by key: `(key, repo, file)`.
const MODELS: &[(&str, &str, &str)] = &[
    (
        "smollm3-3b",
        "unsloth--SmolLM3-3B-GGUF",
        "SmolLM3-3B-Q4_K_M.gguf",
    ),
    (
        "gemma3-1b",
        "unsloth--gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-Q4_K_M.gguf",
    ),
    (
        "ernie-0.3b",
        "unsloth--ERNIE-4.5-0.3B-PT-GGUF",
        "ERNIE-4.5-0.3B-PT-Q4_K_M.gguf",
    ),
    (
        "qwen3-0.6b",
        "unsloth--Qwen3-0.6B-GGUF",
        "Qwen3-0.6B-Q4_K_M.gguf",
    ),
    (
        "qwen3-1.7b",
        "unsloth--Qwen3-1.7B-GGUF",
        "Qwen3-1.7B-Q4_K_M.gguf",
    ),
    (
        "qwen3-30b-a3b",
        "unsloth--Qwen3-30B-A3B-GGUF",
        "Qwen3-30B-A3B-Q4_K_M.gguf",
    ),
    (
        "ernie-21b-a3b",
        "unsloth--ERNIE-4.5-21B-A3B-PT-GGUF",
        "ERNIE-4.5-21B-A3B-PT-Q4_K_M.gguf",
    ),
];

/// The user messages each model is scored on, rendered through its chat template.
const PROMPTS: [&str; 3] = [
    "The capital of France is Paris. The capital of Japan is",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a CPU instruction pipeline works, and list the hazards that break it.",
];

/// Greedy tokens the reference generates after each prompt; both sides are then scored on the
/// prompt plus these.
const N_DECODE: usize = 8;

fn find_model(key: &str) -> PathBuf {
    let &(_, repo, file) = MODELS
        .iter()
        .find(|m| m.0 == key)
        .unwrap_or_else(|| panic!("no model {key:?} in the catalogue"));
    infr_hub::Store::discover()
        .ok()
        .and_then(|s| s.snapshot_file(repo, file))
        .unwrap_or_else(|| panic!("{file} is not in the HF cache"))
}

fn load(path: &std::path::Path, reference: bool) -> SeamModel {
    let mut cfg = EngineConfig::default();
    infr_testkit::honour_env(&mut cfg);
    cfg.kernels.cpu.reference = reference;
    SeamModel::load_with(path, None, std::sync::Arc::new(cfg)).expect("model load")
}

fn models_from_env() -> Vec<String> {
    std::env::var("INFR_AQ_MODELS")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default()
}

/// Every prompt's token ids: the three rendered through the chat template, then the first as
/// raw text, so its position 0 is the tokenizer's own first token.
fn prompts(model: &SeamModel) -> Vec<Vec<u32>> {
    let mut texts: Vec<String> = PROMPTS
        .iter()
        .map(|p| model.render_chat(p).expect("chat template"))
        .collect();
    texts.push(PROMPTS[0].to_string());
    // `INFR_AQ_ONE_TOKEN`: each prompt cut to its first token, so the driver runs every position,
    // position 0 included, as a one-row decode step.
    let one = std::env::var_os("INFR_AQ_ONE_TOKEN").is_some();
    texts
        .iter()
        .map(|t| {
            let mut ids = model.encode(t).expect("encode");
            if one {
                ids.truncate(1);
            }
            ids
        })
        .collect()
}

/// The argmax of a logit row, the lowest index on an exact tie.
fn argmax(row: &[f32]) -> u32 {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b }) as u32
}

/// The CPU reference's greedy run on `prompt` through the teacher-forced driver
/// (`SeamModel::forced_logits`): `n` decoded tokens, the logits at every position.
fn reference_run(model: &SeamModel, prompt: &[u32], n: usize) -> ForcedLogits {
    model
        .forced_logits(
            ForcedOn::CpuReference,
            |_| {},
            prompt,
            n,
            PrefillHead::All,
            |_, row| argmax(row),
        )
        .expect("cpu reference run")
}

/// `reference`'s tokens forced through `on`: the same prefill, then the same decode steps.
fn forced(model: &SeamModel, on: ForcedOn, reference: &ForcedLogits) -> ForcedLogits {
    let steps = reference.positions.len() - reference.prefill_rows;
    let p = reference.seq.len() - steps;
    let got = model
        .forced_logits(
            on,
            |_| {},
            &reference.seq[..p],
            steps,
            PrefillHead::All,
            |s, _| reference.seq[p + s],
        )
        .expect("forced run");
    assert_eq!(
        got.positions, reference.positions,
        "{on:?} scored other positions"
    );
    got
}

/// Every position of `test` scored against `reference` (`infr_llama::tolerance::score_row`);
/// entry `k` is position `k`.
fn score_rows(reference: &ForcedLogits, test: &ForcedLogits) -> Vec<RowScore> {
    assert_eq!(reference.positions[0], 0, "position 0 was not scored");
    let v = reference.vocab;
    reference
        .logits
        .chunks(v)
        .zip(test.logits.chunks(v))
        .map(|(r, t)| score_row(r, t).expect("scorable rows"))
        .collect()
}

/// One forward of `toks` as a single prefill, every row's logits (the shape the op tap needs: all
/// rows in one graph).
fn logits(model: &SeamModel, toks: &[u32]) -> Vec<f32> {
    model.verify_logits_and_h_cpu(toks).expect("cpu forward").0
}

/// Position 0, position 1, then the worst, median and mean KL over positions 2.. of `s`; `dev`
/// is `RowScore::logit_dev` (the largest centred logit difference over the reference's range).
fn summary(s: &[RowScore]) -> String {
    let rest = &s[2..];
    let mut kls: Vec<f64> = rest.iter().map(|p| p.kl).collect();
    kls.sort_by(f64::total_cmp);
    let worst = kls[kls.len() - 1];
    let at = 2 + (0..rest.len()).fold(0, |b, i| if rest[i].kl > rest[b].kl { i } else { b });
    let median = kls[kls.len() / 2];
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    let dev = rest.iter().fold(0f64, |a, p| a.max(p.logit_dev));
    let top1 = rest.iter().filter(|p| p.top1_agrees).count();
    format!(
        "p0 KL {:>8.5} dev {:>6.4} {}  p1 KL {:>8.5}  p2.. KL max {:>8.5} @{:<3} med {:>8.6} mean {:>8.6} dev {:>6.4} top1 {}/{}",
        s[0].kl,
        s[0].logit_dev,
        if s[0].top1_agrees { "=" } else { "x" },
        s[1].kl,
        worst,
        at,
        median,
        mean,
        dev,
        top1,
        rest.len()
    )
}

/// The activation schemes the simulation scores: `INFR_AQ_SCHEMES` as comma-separated
/// `block:keep:ratio:qmax[:split[:sub:emax]]` (`ActSim`'s fields), else a default set.
fn schemes() -> Vec<(String, ActSim)> {
    let spec = std::env::var("INFR_AQ_SCHEMES").unwrap_or_else(|_| {
        "256:0:0:127,32:0:0:127,16:0:0:127,256:1:0:127,256:2:0:127,256:1:2:127,\
         256:2:2:127,256:2:4:127,256:4:4:127,256:0:0:32767"
            .into()
    });
    spec.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| (format!("sim {}", s.trim()), parse_scheme(s)))
        .collect()
}

/// One `block:keep:ratio:qmax[:split[:sub:emax]]` scheme.
fn parse_scheme(s: &str) -> ActSim {
    let v: Vec<u32> = s
        .split(':')
        .map(|n| {
            n.trim()
                .parse()
                .expect("block:keep:ratio:qmax[:split[:sub:emax]]")
        })
        .collect();
    ActSim {
        block: v[0] as usize,
        keep: v[1] as usize,
        ratio: v[2],
        qmax: v[3],
        split: v.get(4) == Some(&1),
        sub: v.get(5).map_or(0, |&s| s as usize),
        emax: v.get(6).copied().unwrap_or(0),
    }
}

/// The simulated quantizer for weights production feeds 32-element activation blocks:
/// `INFR_AQ_SIM32` in the `INFR_AQ_SCHEMES` grammar, else production's own (`32:0:0:127`).
fn sim32() -> ActSim {
    let spec = std::env::var("INFR_AQ_SIM32").unwrap_or_else(|_| "32:0:0:127".into());
    parse_scheme(&spec)
}

/// Per-position KL of the production CPU and of each simulated activation scheme against the
/// reference, every prompt of every `INFR_AQ_MODELS` model. Asserts only that it scored.
#[test]
#[ignore = "measurement: needs INFR_AQ_MODELS and the cached models"]
fn per_position_against_the_reference() {
    let keys = models_from_env();
    assert!(!keys.is_empty(), "set INFR_AQ_MODELS");
    let detail = std::env::var_os("INFR_AQ_DETAIL").is_some();
    let mut tapped = false;
    for key in &keys {
        let model = load(&find_model(key), false);
        for (i, prompt) in prompts(&model).iter().enumerate() {
            if !wanted_seq(i) {
                continue;
            }
            let r = reference_run(&model, prompt, N_DECODE);
            let sp = score_rows(&r, &forced(&model, ForcedOn::Cpu, &r));
            println!(
                "{key} prompt {i} ({} tokens, {} from the prefill)",
                r.positions.len(),
                r.prefill_rows
            );
            println!("  {:<22} {}", "production", summary(&sp));
            if detail {
                for (k, s) in sp.iter().enumerate() {
                    println!(
                        "    pos {k:>3} tok {:>6} KL {:.6} dev {:.4} {}",
                        r.seq[k],
                        s.kl,
                        s.logit_dev,
                        if s.top1_agrees { "=" } else { "x" }
                    );
                }
            }
            for (name, sim) in schemes() {
                // The first simulated run also taps its ops, to show the simulation reached the
                // reference's `Op::Linear` (it acts on the thread that executes the graphs).
                if !tapped {
                    linear_tap::start(&[0]);
                }
                linear_tap::simulate(Some((sim, sim32())));
                let q = forced(&model, ForcedOn::CpuReference, &r);
                linear_tap::simulate(None);
                if !tapped {
                    assert!(
                        !linear_tap::take().is_empty(),
                        "the forced run bypassed the tap"
                    );
                    tapped = true;
                }
                println!("  {name:<22} {}", summary(&score_rows(&r, &q)));
            }
        }
    }
}

/// Statistics of one activation row as the production quantizer sees it, per 256-element block.
struct RowStats {
    /// Largest `max|x| / rms(x)` over the row's blocks.
    crest: f32,
    /// Fraction of the row's elements that quantize to 0, and to ±1.
    zero: f32,
    one: f32,
    /// Lowest per-block quantization SNR (dB) of the 256-block quantizer, and of the 32-block one.
    snr256: f32,
    snr32: f32,
    /// Elements above 20 × the row's median magnitude: how many, and the first few indices.
    outliers: Vec<usize>,
}

fn row_stats(x: &[f32]) -> RowStats {
    let rt = |block| {
        ActSim {
            block,
            keep: 0,
            ratio: 0,
            qmax: 127,
            split: false,
            sub: 0,
            emax: 0,
        }
        .round_trip(x)
    };
    let (q256, q32) = (rt(256), rt(32));
    let snr = |q: &[f32], block: usize| {
        x.chunks(block)
            .zip(q.chunks(block))
            .map(|(a, b)| {
                let s: f64 = a.iter().map(|&v| (v as f64).powi(2)).sum();
                let n: f64 = a
                    .iter()
                    .zip(b)
                    .map(|(&v, &w)| (v as f64 - w as f64).powi(2))
                    .sum();
                if n == 0.0 {
                    f64::INFINITY
                } else {
                    10.0 * (s / n).log10()
                }
            })
            .fold(f64::INFINITY, f64::min) as f32
    };
    let crest = x
        .chunks(256)
        .map(|b| {
            let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
            let rms = (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
            if rms > 0.0 {
                amax / rms
            } else {
                0.0
            }
        })
        .fold(0f32, f32::max);
    let qs = |v: f32, i: usize| {
        let d = x[i / 256 * 256..(i / 256 * 256 + 256).min(x.len())]
            .iter()
            .fold(0f32, |m, w| m.max(w.abs()))
            / 127.0;
        if d > 0.0 {
            (v / d).round().abs()
        } else {
            0.0
        }
    };
    let n = x.len() as f32;
    let zero = x
        .iter()
        .enumerate()
        .filter(|&(i, &v)| qs(v, i) == 0.0)
        .count() as f32
        / n;
    let one = x
        .iter()
        .enumerate()
        .filter(|&(i, &v)| qs(v, i) == 1.0)
        .count() as f32
        / n;
    let mut mags: Vec<f32> = x.iter().map(|v| v.abs()).collect();
    mags.sort_by(f32::total_cmp);
    let median = mags[mags.len() / 2];
    let outliers = (0..x.len())
        .filter(|&i| x[i].abs() > 20.0 * median)
        .collect();
    RowStats {
        crest,
        zero,
        one,
        snr256: snr(&q256, 256),
        snr32: snr(&q32, 32),
        outliers,
    }
}

fn rel_err(a: &[f32], b: &[f64]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y).powi(2)).sum();
    let d: f64 = b.iter().map(|y| y * y).sum();
    (n / d).sqrt()
}

fn rel_err64(a: &[f64], b: &[f64]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y).powi(2)).sum();
    let d: f64 = b.iter().map(|y| y * y).sum();
    (n / d).sqrt()
}

/// Where the production CPU's position-0 error comes from: every `Op::Linear` of one forward of
/// the first model's first prompt, rows 0, 1 and a mid-sequence row, with
/// - `local`: the op's own error on that row (production output against the exact f64 dot of the
///   same row), `kern`: the production output against the exact dot of the row after the
///   production quantizer's round trip (the kernel's arithmetic alone: an overflow or a wrong
///   kernel shows here), `q32`: what the 32-block quantizer's local error would be;
/// - the activation statistics of the row entering the op;
/// - `drift`: how far the row entering the op has moved from the reference run's (propagated
///   error).
#[test]
#[ignore = "measurement: needs INFR_AQ_MODELS and the cached models"]
fn localise_the_position_0_error() {
    let keys = models_from_env();
    let key = keys.first().expect("set INFR_AQ_MODELS");
    let path = find_model(key);
    let reference = load(&path, true);
    let production = load(&path, false);
    // `INFR_AQ_SEQ`: a prompt's index (with the reference's greedy continuation), or `bench` for
    // `infr bench`'s prompt (ids `i % 100`).
    let toks = &match std::env::var("INFR_AQ_SEQ").as_deref() {
        Ok("bench") => (0..128u32).map(|i| i % 100).collect(),
        v => {
            let i = v.map_or(0, |v| v.parse::<usize>().expect("INFR_AQ_SEQ"));
            reference_run(&production, &prompts(&production)[i], N_DECODE).seq
        }
    };
    let rows: Vec<usize> = std::env::var("INFR_AQ_ROWS").map_or_else(
        |_| vec![0, 1, toks.len() / 2],
        |v| {
            v.split(',')
                .map(|r| r.trim().parse().expect("INFR_AQ_ROWS"))
                .collect()
        },
    );
    linear_tap::start(&rows);
    logits(&reference, toks);
    let ref_ops = linear_tap::take();
    linear_tap::start(&rows);
    logits(&production, toks);
    let prod_ops = linear_tap::take();
    assert!(!prod_ops.is_empty(), "the tap saw no Op::Linear");
    assert_eq!(
        ref_ops.len(),
        prod_ops.len(),
        "the two runs' op lists differ"
    );
    println!("{key}: {} tokens, rows {rows:?}", toks.len());
    println!(
        "{:>3} {:>5} {:>11} {:>6} | row: {:>8} {:>8} {:>8} {:>8} {:>6} {:>5} {:>5} {:>6} {:>6} {:>4} outliers",
        "#", "op", "shape", "dtype", "local", "kern", "q32", "drift", "crest", "zero", "±1", "snr256", "snr32", "n"
    );
    for (k, (p, r)) in prod_ops.iter().zip(&ref_ops).enumerate() {
        for (pr, rr) in p.rows.iter().zip(&r.rows) {
            let st = row_stats(&pr.x);
            let kern = if linear_tap::uses_q8_32(p.dtype, p.in_f) {
                rel_err(&pr.out, &pr.exact_q32)
            } else if matches!(
                p.dtype,
                infr_core::DType::F32 | infr_core::DType::F16 | infr_core::DType::Bf16
            ) {
                rel_err(&pr.out, &pr.exact)
            } else {
                rel_err(&pr.out, &pr.exact_q256)
            };
            let drift = {
                let n: f64 =
                    pr.x.iter()
                        .zip(&rr.x)
                        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                        .sum();
                let d: f64 = rr.x.iter().map(|&b| (b as f64).powi(2)).sum();
                (n / d).sqrt()
            };
            println!(
                "{k:>3} {:>5} {:>11} {:>6} | r{:<4} {:>8.2e} {:>8.2e} {:>8.2e} {:>8.2e} {:>6.1} {:>5.3} {:>5.3} {:>6.1} {:>6.1} {:>4} {:?}",
                p.op_idx,
                format!("{}x{}", p.in_f, p.out_f),
                format!("{:?}", p.dtype),
                pr.row,
                rel_err(&pr.out, &pr.exact),
                kern,
                rel_err64(&pr.exact_q32, &pr.exact),
                drift,
                st.crest,
                st.zero,
                st.one,
                st.snr256,
                st.snr32,
                st.outliers.len(),
                &st.outliers[..st.outliers.len().min(6)]
            );
        }
    }
    // How often each simulated scheme passes an element through, per 256 elements, over every
    // op's input rows: the side path's work in a kernel that computed it.
    for (name, sim) in schemes() {
        let mut per_row = vec![(0usize, 0usize, 0usize); rows.len()];
        for op in &prod_ops {
            for (k, r) in op.rows.iter().enumerate() {
                for blk in r.x.chunks(sim.block) {
                    let n = sim.kept(blk).len();
                    per_row[k].0 += n;
                    per_row[k].1 += usize::from(n > 0);
                    per_row[k].2 += 1;
                }
            }
        }
        let cells: Vec<String> = per_row
            .iter()
            .zip(&rows)
            .map(|(&(n, hit, blocks), r)| {
                format!(
                    "r{r}: {:.3}/blk, {:.1}% blocks",
                    n as f64 / blocks as f64 * (256 / sim.block) as f64,
                    100.0 * hit as f64 / blocks as f64
                )
            })
            .collect();
        println!("kept {name:<22} {}", cells.join("  "));
    }
}

/// Whether sequence `i` is scored: all of them, or those `INFR_AQ_SEQS` lists (comma-separated).
fn wanted_seq(i: usize) -> bool {
    std::env::var("INFR_AQ_SEQS").map_or(true, |v| v.split(',').any(|s| s.trim().parse() == Ok(i)))
}

/// Position-0 KL bounds for [`position_0_matches_the_reference`], per model, on the first prompt
/// as raw text. Measured (docs/cpu-activations.md): SmolLM3-3B 1.92 before the split
/// and 4e-5 after, Qwen3-0.6B 7.5e-2 and 2.0e-3; each model's later positions run to a few 1e-3
/// on the same prompt. A bound sits between, at the steady state's order.
const POSITION_0_BOUNDS: &[(&str, f64)] = &[("smollm3-3b", 1e-2), ("qwen3-0.6b", 2e-2)];

/// The production CPU's first-token row against the CPU reference on real models whose first
/// token carries massive-activation dimensions: the regression the int8 activation split fixes.
#[test]
#[ignore = "needs SmolLM3-3B and Qwen3-0.6B Q4_K_M in the HF cache"]
fn position_0_matches_the_reference() {
    let mut ran = 0;
    for &(key, bound) in POSITION_0_BOUNDS {
        let model = load(&find_model(key), false);
        let toks = model.encode(PROMPTS[0]).expect("encode");
        let r = reference_run(&model, &toks, 1);
        let s = score_rows(&r, &forced(&model, ForcedOn::Cpu, &r));
        println!(
            "{key}: position 0 KL {:.3e}, dev {:.4}",
            s[0].kl, s[0].logit_dev
        );
        assert!(
            s[0].kl < bound,
            "{key}: position 0 KL {} over {bound}",
            s[0].kl
        );
        ran += 1;
    }
    assert_eq!(ran, POSITION_0_BOUNDS.len());
}

/// llama.cpp's CPU, infr's production CPU and infr's reference against each other on the last
/// row of a `llama-debug --save-logits` dump (a one-token prompt makes that row position 0), on
/// the token ids llama.cpp itself used:
///
/// ```text
/// llama-debug -m <gguf> -p "The" --save-logits --logits-output-dir <dir>
/// INFR_LLAMA_DUMP=<dir>/llamacpp-<model>.bin INFR_LLAMA_DUMP_MODEL=<gguf> cargo test --release \
///   -p infr-llama --test cpu_act_quant -- --ignored --nocapture llama_cpp_last_row
/// ```
#[test]
#[ignore = "measurement: needs INFR_LLAMA_DUMP and INFR_LLAMA_DUMP_MODEL"]
fn llama_cpp_last_row_against_the_reference() {
    let (Ok(dump), Ok(gguf)) = (
        std::env::var("INFR_LLAMA_DUMP"),
        std::env::var("INFR_LLAMA_DUMP_MODEL"),
    ) else {
        panic!("set INFR_LLAMA_DUMP and INFR_LLAMA_DUMP_MODEL");
    };
    let read = |p: &str| std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    let llama: Vec<f32> = read(&dump)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();
    let toks: Vec<u32> = read(&format!("{}-tokens.bin", dump.trim_end_matches(".bin")))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| u32::from_le_bytes(c))
        .collect();
    let path = PathBuf::from(gguf);
    let last = |m: &SeamModel| {
        let l = logits(m, &toks);
        let v = l.len() / toks.len();
        l[l.len() - v..].to_vec()
    };
    let r = last(&load(&path, true));
    let p = last(&load(&path, false));
    assert_eq!(llama.len(), r.len(), "the dump's vocabulary differs");
    let show = |name: &str, reference: &[f32], test: &[f32]| {
        let s = score_row(reference, test).expect("scorable rows");
        println!(
            "  {name:<26} KL {:.3e}  dev {:.4}  top-1 {}",
            s.kl,
            s.logit_dev,
            if s.top1_agrees { "=" } else { "x" }
        );
    };
    println!("{} tokens {toks:?}, last row:", toks.len());
    show("llama.cpp vs reference", &r, &llama);
    show("production vs reference", &r, &p);
    show("production vs llama.cpp", &llama, &p);
}
