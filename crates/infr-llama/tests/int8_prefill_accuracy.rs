//! What the int8 cooperative-matrix prefill GEMM costs in next-token accuracy on real text, next
//! to llama.cpp's Vulkan backend, which quantizes activations the same way on the AMD Windows
//! driver. A measurement, not a gate: it prints a table and asserts only that it scored something.
//!
//! Every backend is scored against infr's CPU reference backend (`kernels.cpu.reference`: f32
//! activations, f32 dot), position by position over the second half of each `n_ctx` window —
//! llama.cpp's `llama-perplexity` protocol, on the very tokens it used. The llama.cpp side comes
//! from its `--kl-divergence-base` dumps, which store each scored row's log-probabilities
//! (16-bit, clamped 16 nats below the top), so llama.cpp rows get KL divergence and top-1
//! agreement but no logit cosine, and no perplexity: the clamp raises the probability of every
//! next token below it, so a perplexity read back from the dump is biased low. Take llama.cpp's
//! perplexity from its own `Final estimate` line instead.
//!
//! Scoring a dump as the candidate against the reference, as this test does, the clamp also biases
//! KL divergence low, by the reference's own probability mass more than 16 nats below its top
//! token, which differs by model. So the infr rows are scored twice: from their
//! raw logits, and from those logits put through the dump's encoding (the `, dumped` rows). Only
//! the dumped rows compare with llama.cpp's; the reference's own dumped row is the size of the bias.
//!
//! Run (one model per run; dumps from `llama-perplexity -f <text> -c 512 --chunks 6
//! --kl-divergence-base <file>`, the first of them supplies the tokens):
//!
//! ```text
//! INFR_DEV=Vulkan0 INFR_ACC_MODEL=<gguf> \
//!   INFR_ACC_LLAMA="vulkan=<dump>,cpu=<dump>" \
//!   cargo test -p infr-llama --release --test int8_prefill_accuracy -- --ignored --nocapture
//! ```
use std::path::Path;

/// A `llama-perplexity --kl-divergence-base` file: the header, every window's tokens, and the
/// scored rows' encoded log-probabilities.
struct LlamaDump {
    n_ctx: usize,
    n_vocab: usize,
    n_chunk: usize,
    tokens: Vec<u32>,
    /// Per scored row: `[scale f32, min_log_prob f32, n_vocab u16 codes]`, padded as written.
    rows: Vec<u8>,
}

impl LlamaDump {
    fn read(path: &Path) -> LlamaDump {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            &b[..8],
            b"_logits_",
            "{}: not a logits dump",
            path.display()
        );
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
        let (n_ctx, n_vocab, n_chunk) = (u32_at(8), u32_at(12), u32_at(16));
        let tok_end = 20 + n_chunk * n_ctx * 4;
        let tokens = b[20..tok_end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        LlamaDump {
            n_ctx,
            n_vocab,
            n_chunk,
            tokens,
            rows: b[tok_end..].to_vec(),
        }
    }

    /// `u16` words per stored row (`perplexity.cpp`'s `nv`).
    fn row_words(&self) -> usize {
        2 * self.n_vocab.div_ceil(2) + 4
    }

    fn scored_per_chunk(&self) -> usize {
        self.n_ctx - 1 - self.n_ctx / 2
    }

    /// Log-probabilities of scored row `r` of chunk `c`; codes of 0 decode to the floor, as
    /// llama.cpp's own KL divergence reads them.
    fn log_probs(&self, c: usize, r: usize) -> Vec<f32> {
        let words = self.row_words();
        let at = ((c * self.scored_per_chunk() + r) * words) * 2;
        let row = &self.rows[at..at + words * 2];
        let f = |o: usize| f32::from_le_bytes(row[o..o + 4].try_into().unwrap());
        let (scale, min_lp) = (f(0), f(4));
        (0..self.n_vocab)
            .map(|i| {
                let q = u16::from_le_bytes([row[8 + 2 * i], row[9 + 2 * i]]);
                min_lp + scale * q as f32
            })
            .collect()
    }
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v)) as f64;
    let lse = logits
        .iter()
        .map(|&v| (v as f64 - max).exp())
        .sum::<f64>()
        .ln()
        + max;
    logits.iter().map(|&v| v as f64 - lse).collect()
}

/// `logits` as log-probabilities the way a `llama-perplexity --kl-divergence-base` dump stores and
/// reads them back (the encoding `perplexity.cpp`'s `log_softmax` writes, in its f32 arithmetic): every logit more
/// than 16 below the top raised to that floor, the rest rounded to 65535 steps above it.
fn as_dumped(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
    let min = logits
        .iter()
        .fold(f32::INFINITY, |a, &v| a.min(v))
        .max(max - 16.0);
    let sum_exp: f64 = logits.iter().map(|&v| (v - max).exp() as f64).sum();
    let min_lp = min - max - sum_exp.ln() as f32;
    let scale = (max - min) / 65535.0;
    let inv = 1.0 / scale;
    logits
        .iter()
        .map(|&v| {
            let code = if v > min {
                (inv * (v - min)).round_ties_even()
            } else {
                0.0
            };
            (min_lp + scale * code) as f64
        })
        .collect()
}

fn argmax<T: PartialOrd + Copy>(v: &[T]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// Running totals for one backend against the reference.
#[derive(Default)]
struct Score {
    rows: usize,
    kld_sum: f64,
    kld_max: f64,
    top1: usize,
    nll_sum: f64,
    cos_sum: f64,
    cos_min: f64,
    dmax: f64,
    has_logits: bool,
}

impl Score {
    /// Add one row: `lp` the backend's log-probabilities, `logits` its raw logits when it has
    /// them, against the reference's `ref_lp` / `ref_logits`; `next` the token that follows.
    fn add(
        &mut self,
        lp: &[f64],
        logits: Option<&[f32]>,
        ref_lp: &[f64],
        ref_logits: &[f32],
        next: usize,
    ) {
        let n = lp.len().min(ref_lp.len());
        let kld: f64 = (0..n).map(|i| ref_lp[i].exp() * (ref_lp[i] - lp[i])).sum();
        if self.rows == 0 {
            self.cos_min = f64::INFINITY;
        }
        self.rows += 1;
        self.kld_sum += kld;
        self.kld_max = self.kld_max.max(kld);
        self.top1 += usize::from(argmax(&lp[..n]) == argmax(&ref_lp[..n]));
        self.nll_sum -= lp[next];
        if let Some(l) = logits {
            self.has_logits = true;
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (a, b) in l.iter().zip(ref_logits) {
                let (a, b) = (*a as f64, *b as f64);
                dot += a * b;
                na += a * a;
                nb += b * b;
                self.dmax = self.dmax.max((a - b).abs());
            }
            let cos = dot / (na.sqrt() * nb.sqrt());
            self.cos_sum += cos;
            self.cos_min = self.cos_min.min(cos);
        }
    }

    fn line(&self, name: &str) -> String {
        let r = self.rows as f64;
        let ppl = if self.has_logits {
            format!("{:.4}", (self.nll_sum / r).exp())
        } else {
            "      -".to_string()
        };
        let logit_cols = if self.has_logits {
            format!(
                "{:.6} {:.6} {:.4}",
                self.cos_sum / r,
                self.cos_min,
                self.dmax
            )
        } else {
            "       -        -      -".to_string()
        };
        format!(
            "{name:<22} {:>5} {:.6} {:.5} {:.4} {ppl} {logit_cols}",
            self.rows,
            self.kld_sum / r,
            self.kld_max,
            self.top1 as f64 / r,
        )
    }
}

fn model(path: &Path, f: impl FnOnce(&mut infr_llama::EngineConfig)) -> infr_llama::SeamModel {
    let mut cfg = infr_llama::EngineConfig::default();
    cfg.device.dev = std::env::var("INFR_DEV").ok();
    infr_testkit::honour_env(&mut cfg);
    f(&mut cfg);
    infr_llama::SeamModel::load_with(path, None, std::sync::Arc::new(cfg)).expect("model load")
}

#[test]
#[ignore = "measurement: needs INFR_ACC_MODEL, llama-perplexity dumps and a Vulkan GPU"]
fn int8_prefill_accuracy_against_the_cpu_reference() {
    let (Ok(gguf), Ok(dumps)) = (
        std::env::var("INFR_ACC_MODEL"),
        std::env::var("INFR_ACC_LLAMA"),
    ) else {
        eprintln!("int8_prefill_accuracy: INFR_ACC_MODEL / INFR_ACC_LLAMA unset — skipped");
        return;
    };
    let gguf = Path::new(&gguf);
    let llama: Vec<(String, LlamaDump)> = dumps
        .split(',')
        .map(|kv| {
            let (k, v) = kv.split_once('=').expect("name=path");
            (format!("llama.cpp {k}"), LlamaDump::read(Path::new(v)))
        })
        .collect();
    let base = &llama[0].1;
    for (name, d) in &llama {
        assert_eq!(
            (d.n_ctx, d.n_chunk, &d.tokens),
            (base.n_ctx, base.n_chunk, &base.tokens),
            "{name}: dumps disagree on the windows"
        );
    }
    let reference = model(gguf, |c| c.kernels.cpu.reference = true);
    let int8_off = model(gguf, |c| c.kernels.vulkan.i8cm_kq = false);
    let int8_on = model(gguf, |_| {});

    let first = base.n_ctx / 2;
    let mut scores: Vec<(String, Score)> = vec![
        ("infr vulkan int8 off".into(), Score::default()),
        ("infr vulkan int8 on".into(), Score::default()),
    ];
    scores.extend(llama.iter().map(|(n, _)| (n.clone(), Score::default())));
    let dumped = scores.len();
    for name in [
        "infr cpu ref, dumped",
        "infr int8 off, dumped",
        "infr int8 on, dumped",
    ] {
        scores.push((name.into(), Score::default()));
    }
    // The reference against itself: its own perplexity, and zeros elsewhere.
    let mut ref_score = Score::default();
    for c in 0..base.n_chunk {
        let toks = &base.tokens[c * base.n_ctx..][..base.n_ctx];
        let ref_logits = reference.verify_logits_and_h_cpu(toks).expect("cpu").0;
        let vocab = ref_logits.len() / toks.len();
        let vk = [
            int8_off.verify_logits_vulkan(toks).expect("vulkan off"),
            int8_on.verify_logits_vulkan(toks).expect("vulkan on"),
        ];
        for v in &vk {
            assert_eq!(v.len(), ref_logits.len(), "vulkan logits shape");
        }
        for r in 0..base.scored_per_chunk() {
            let p = first + r;
            let next = toks[p + 1] as usize;
            let rl = &ref_logits[p * vocab..][..vocab];
            let rlp = log_softmax(rl);
            ref_score.add(&rlp, Some(rl), &rlp, rl, next);
            scores[dumped].1.add(&as_dumped(rl), None, &rlp, rl, next);
            for (i, v) in vk.iter().enumerate() {
                let l = &v[p * vocab..][..vocab];
                scores[i].1.add(&log_softmax(l), Some(l), &rlp, rl, next);
                scores[dumped + 1 + i]
                    .1
                    .add(&as_dumped(l), None, &rlp, rl, next);
            }
            for (j, (_, d)) in llama.iter().enumerate() {
                let lp: Vec<f64> = d.log_probs(c, r).iter().map(|&v| v as f64).collect();
                scores[2 + j].1.add(&lp, None, &rlp, rl, next);
            }
        }
        eprintln!("chunk {c} of {} scored", base.n_chunk);
    }
    println!(
        "{}: {} windows of {} tokens, second half scored, against infr's CPU reference",
        gguf.file_name().unwrap().to_string_lossy(),
        base.n_chunk,
        base.n_ctx
    );
    println!(
        "{:<22} {:>5} {:>8} {:>7} {:>6} {:>6} {:>8} {:>8} {:>6}",
        "backend", "rows", "KLD mean", "KLD max", "top-1", "PPL", "cos mean", "cos min", "max|Δ|"
    );
    println!("{}", ref_score.line("infr cpu reference"));
    for (name, s) in &scores {
        assert!(s.rows > 0, "{name}: nothing scored");
        println!("{}", s.line(name));
    }

    // Greedy continuations: the prompt prefilled with int8 activations (on) or f16 (off), then
    // decoded by the same GEMV either way, each set against the CPU reference's own greedy text.
    // Prompts are the opening of each window. Agreement is the leading tokens two continuations
    // share, re-tokenized from their text.
    let n_prompts = base.n_chunk;
    let lead = |a: &str, b: &str| {
        let (ta, tb) = (int8_on.encode(a).unwrap(), int8_on.encode(b).unwrap());
        ta.iter().zip(&tb).take_while(|(x, y)| x == y).count()
    };
    for c in 0..n_prompts {
        let toks = &base.tokens[c * base.n_ctx..][..first];
        let prompt = int8_on.decode(toks).expect("detokenize");
        let off = int8_off.generate_dense_vulkan(&prompt, 64).expect("off");
        let on = int8_on.generate_dense_vulkan(&prompt, 64).expect("on");
        let mut cpu = String::new();
        reference
            .generate_cpu_reference(&prompt, 64, None, |p| cpu.push_str(p))
            .expect("cpu reference");
        println!(
            "greedy 64 after window {c}'s first {first} tokens: leading tokens shared — on/off {}, off/cpu {}, on/cpu {}",
            lead(&on, &off),
            lead(&off, &cpu),
            lead(&on, &cpu)
        );
    }
}
