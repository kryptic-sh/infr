//! The teacher-forced comparison: the CPU reference's greedy run, the same tokens forced through
//! another backend, and the two scored with `infr_llama::tolerance`.

use infr_llama::seam::{ForcedLogits, ForcedOn};
use infr_llama::tolerance::{tolerance, PathClass, RunScore, MIN_TOP1_SCORED};
use infr_llama::{EngineConfig, SeamModel};
use infr_testkit::gguf::{Fill, Meta, SyntheticModel, TempGguf, TensorSpec};

/// Decode steps forced after each prompt.
pub const N_DECODE: usize = 32;

/// The prefill chunk height of the second prefill run: small enough that the longest prompt
/// spans several chunks and that every prompt's chunks take the small-`m` GEMM tiers.
pub const SMALL_UBATCH: usize = 32;

/// The three prompts every model is scored on: short, medium, and long enough to span several
/// [`SMALL_UBATCH`] chunks.
pub const PROMPTS: [&str; 3] = [
    "The capital of France is Paris. The capital of Japan is",
    "Write a short story about a lighthouse keeper who finds a message in a bottle. \
     The keeper had lived alone on the island for eleven years, and in all that time",
    "Explain how a CPU instruction pipeline works. A classic five-stage pipeline \
     divides the execution of each instruction into fetch, decode, execute, memory access \
     and write-back. While one instruction is being decoded, the next one is already being \
     fetched, so in the ideal case one instruction completes every cycle. Hazards break \
     this ideal. A structural hazard occurs when two instructions need the same hardware \
     resource in the same cycle. A data hazard occurs when an instruction depends on the \
     result of an earlier instruction that has not yet written it back; forwarding paths \
     and pipeline stalls are the usual remedies. A control hazard occurs at a branch, \
     because the processor does not know which instruction to fetch next until the branch \
     is resolved. Branch prediction guesses the outcome and, if the guess was wrong, the \
     speculatively fetched instructions are flushed. List the hazards again, and for each \
     one give a concrete example from a real processor:",
];

/// A model the tests can find in the HF cache: `(key, repo, file)`.
pub const MODELS: &[(&str, &str, &str)] = &[
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
        "gemma3-1b",
        "unsloth--gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-Q4_K_M.gguf",
    ),
    (
        "qwen35-0.8b",
        "unsloth--Qwen3.5-0.8B-GGUF",
        "Qwen3.5-0.8B-Q4_K_M.gguf",
    ),
    (
        "lfm3-q4",
        "LiquidAI--LFM3-1.7B-GGUF",
        "LFM3-1.7B-Q4_K_M.gguf",
    ),
    ("lfm3-f16", "LiquidAI--LFM3-1.7B-GGUF", "LFM3-1.7B-F16.gguf"),
    (
        "ernie-0.3b-q4",
        "unsloth--ERNIE-4.5-0.3B-PT-GGUF",
        "ERNIE-4.5-0.3B-PT-Q4_K_M.gguf",
    ),
    (
        "ernie-0.3b-f16",
        "unsloth--ERNIE-4.5-0.3B-PT-GGUF",
        "ERNIE-4.5-0.3B-PT-F16.gguf",
    ),
    (
        "smollm3-3b",
        "unsloth--SmolLM3-3B-GGUF",
        "SmolLM3-3B-Q4_K_M.gguf",
    ),
    (
        "llama32-1b",
        "unsloth--Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    ),
    (
        "qwen25-0.5b",
        "Qwen--Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ),
    (
        "ernie-21b-moe",
        "unsloth--ERNIE-4.5-21B-A3B-PT-GGUF",
        "ERNIE-4.5-21B-A3B-PT-Q4_K_M.gguf",
    ),
];

/// A model to score and the token ids of its prompts.
pub struct Case {
    pub model: SeamModel,
    pub prompts: Vec<Vec<u32>>,
    /// The synthetic model's file, kept for as long as the model maps it.
    _file: Option<TempGguf>,
}

/// The [`Case`] for a [`MODELS`] key — [`PROMPTS`] tokenized — or for `synthetic-f32` /
/// `synthetic-f16` ([`synthetic_llama`] with f32 or f16 projections; prompts of the same three
/// lengths, as ids); `None` when a cached model is not downloaded.
pub fn open_case(key: &str) -> Option<Case> {
    let weights = match key {
        "synthetic-f32" => Some(Fill::Rand(0.0)),
        "synthetic-f16" => Some(Fill::ExactF16(Vec::new())),
        _ => None,
    };
    if let Some(w) = weights {
        let file = TempGguf::write(&format!("tolerance-{key}"), &synthetic_llama(w));
        let model = SeamModel::load_with(file.path(), None, std::sync::Arc::new(env_cfg()))
            .expect("synthetic model load");
        let prompts = [12usize, 34, 180]
            .iter()
            .enumerate()
            .map(|(salt, &n)| {
                (0..n)
                    .map(|i| ((i * 7 + 3 + salt * 11) % 64) as u32)
                    .collect()
            })
            .collect();
        return Some(Case {
            model,
            prompts,
            _file: Some(file),
        });
    }
    let model = load(&find_model(key)?);
    let prompts = PROMPTS
        .iter()
        .map(|t| model.encode(t).expect("encode"))
        .collect();
    Some(Case {
        model,
        prompts,
        _file: None,
    })
}

/// The name the tolerance table knows a [`MODELS`] key (or a synthetic model) by: the GGUF file
/// name, which is what a command scoring a user's model has in hand.
pub fn table_name(key: &str) -> &'static str {
    match key {
        "synthetic-f32" => "synthetic-f32",
        "synthetic-f16" => "synthetic-f16",
        _ => {
            MODELS
                .iter()
                .find(|m| m.0 == key)
                .unwrap_or_else(|| panic!("no model {key:?} in the catalogue"))
                .2
        }
    }
}

/// The cached GGUF for a [`MODELS`] key, or `None` when it is not downloaded.
pub fn find_model(key: &str) -> Option<std::path::PathBuf> {
    let &(_, repo, file) = MODELS
        .iter()
        .find(|m| m.0 == key)
        .unwrap_or_else(|| panic!("no model {key:?} in the catalogue"));
    infr_hub::Store::discover().ok()?.snapshot_file(repo, file)
}

/// The shipped defaults plus the test environment's GPU knobs (`INFR_DEV`, and through
/// [`infr_testkit::honour_env`] the forced wave64 path and the tuning table's rows).
pub fn env_cfg() -> EngineConfig {
    let mut cfg = EngineConfig::default();
    cfg.device.dev = std::env::var("INFR_DEV").ok();
    infr_testkit::honour_env(&mut cfg);
    cfg
}

pub fn load(path: &std::path::Path) -> SeamModel {
    SeamModel::load_with(path, None, std::sync::Arc::new(env_cfg())).expect("model load")
}

/// The argmax of a logit row, the lowest index on an exact tie.
pub fn argmax(row: &[f32]) -> u32 {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b }) as u32
}

/// The CPU reference's greedy run on `prompt`: [`N_DECODE`] tokens, the logits at every
/// position.
pub fn reference(model: &SeamModel, prompt: &[u32], n: usize) -> ForcedLogits {
    model
        .forced_logits(
            ForcedOn::CpuReference,
            |_| {},
            prompt,
            n,
            |_, row| argmax(row),
        )
        .expect("cpu reference run")
}

/// `reference`'s tokens forced through `on`, its engine configuration adjusted by `edit`.
pub fn forced(
    model: &SeamModel,
    on: ForcedOn,
    edit: impl FnOnce(&mut EngineConfig),
    reference: &ForcedLogits,
) -> ForcedLogits {
    let p = reference.seq.len() - (reference.positions.len() - reference.prefill_rows);
    let steps = reference.positions.len() - reference.prefill_rows;
    let got = model
        .forced_logits(on, edit, &reference.seq[..p], steps, |s, _| {
            reference.seq[p + s]
        })
        .unwrap_or_else(|e| panic!("forced run on {on:?}: {e:#}"));
    assert_eq!(
        got.seq, reference.seq,
        "{on:?} was not fed the reference's tokens"
    );
    assert_eq!(
        got.positions, reference.positions,
        "{on:?} scored other positions"
    );
    got
}

/// One comparison, split by the path the rows came from.
pub struct Scored {
    pub prefill: RunScore,
    pub decode: RunScore,
}

impl Scored {
    pub fn all(&self) -> RunScore {
        let mut a = self.prefill.clone();
        a.merge(&self.decode);
        a
    }
}

/// Score `test` against `reference`, prefill rows and decode rows separately.
pub fn score(reference: &ForcedLogits, test: &ForcedLogits, near_tie: f64) -> Scored {
    assert_eq!(reference.positions, test.positions);
    assert_eq!(reference.prefill_rows, test.prefill_rows);
    let (v, pr) = (reference.vocab, reference.prefill_rows);
    let mut prefill = RunScore::new(near_tie);
    prefill
        .add_rows(
            &reference.logits[..pr * v],
            &test.logits[..pr * v],
            v,
            &reference.positions[..pr],
        )
        .unwrap_or_else(|e| panic!("prefill: {e}"));
    let mut decode = RunScore::new(near_tie);
    decode
        .add_rows(
            &reference.logits[pr * v..],
            &test.logits[pr * v..],
            v,
            &reference.positions[pr..],
        )
        .unwrap_or_else(|e| panic!("decode: {e}"));
    Scored { prefill, decode }
}

/// One machine-readable line per phase, for the measurement log.
pub fn line(tag: &str, s: &Scored) -> String {
    let one = |phase: &str, r: &RunScore| {
        format!(
            "TOL {tag} phase={phase} n={} kl_max={:.4e} kl_mean={:.4e} cos_min={:.6} \
             cos_mean={:.6} dev_max={:.4e} dev_mean={:.4e} rank_max={} top1={}/{} near={} \
             widest_miss={:.3} worst_kl_pos={}",
            r.positions,
            r.kl.worst.value,
            r.kl.mean(),
            r.topk_cos.worst.value,
            r.topk_cos.mean(),
            r.logit_dev.worst.value,
            r.logit_dev.mean(),
            r.ref_top1_rank.worst.value,
            r.top1_scored - r.top1_misses.len(),
            r.top1_scored,
            r.near_ties,
            r.widest_miss,
            r.kl.worst.pos,
        )
    };
    format!(
        "{}\n{}",
        one("prefill", &s.prefill),
        one("decode", &s.decode)
    )
}

/// Turn every int8-activation Vulkan kernel off: the int8 coopmat and dp4a K-quant prefill GEMMs,
/// the dp4a `mmq` GEMMs (the coopmat tier's and the non-coopmat tier's, which wave64 and devices
/// without coopmat take), the int8 multi-row and decode GEMVs: `i8_dot` gates every packed-int8
/// dot kernel; the rest name the tiers it does not.
pub fn vulkan_float(c: &mut EngineConfig) {
    let v = &mut c.kernels.vulkan;
    v.i8cm_kq = false;
    v.dp4a_kq = false;
    v.mmq = false;
    v.mmv_mw = Some(false);
    v.mrow = false;
    v.i8_dot = false;
}

/// A KV-format pair as a config edit — the `INFR_KV_TYPE_K`/`_V` pair, typed.
pub fn kv_types(c: &mut EngineConfig, k: &str, v: &str) {
    c.kv.type_k = infr_core::budget::parse_kv_dtype(k);
    c.kv.type_k_specified = true;
    c.kv.type_v = infr_core::budget::parse_kv_dtype(v);
    c.kv.type_v_specified = true;
}

/// A named Vulkan configuration of the measurement: its edit, and the KV format the CPU reference
/// it is scored against runs (`None`: the reference's own default).
type GpuConfig = (
    &'static str,
    fn(&mut EngineConfig),
    Option<(&'static str, &'static str)>,
);

/// Every configuration [`measure`] can run; `INFR_TOL_ONLY` (comma-separated names) picks some.
pub const GPU_CONFIGS: &[GpuConfig] = &[
    ("gpu", |_| {}, None),
    ("gpu-ub32", |c| c.device.ubatch = Some(SMALL_UBATCH), None),
    ("gpu-float", vulkan_float, None),
    (
        "gpu-float-ub32",
        |c| {
            vulkan_float(c);
            c.device.ubatch = Some(SMALL_UBATCH);
        },
        None,
    ),
    (
        "gpu-kv-q8",
        |c| kv_types(c, "q8_0", "q8_0"),
        Some(("q8_0", "q8_0")),
    ),
    (
        "gpu-kv-q4v",
        |c| kv_types(c, "f16", "q4_0"),
        Some(("f16", "q4_0")),
    ),
    (
        "gpu-kv-turbo3v",
        |c| kv_types(c, "f16", "turbo3"),
        Some(("f16", "turbo3")),
    ),
    (
        "gpu-kv-q8-float",
        |c| {
            kv_types(c, "q8_0", "q8_0");
            vulkan_float(c);
        },
        Some(("q8_0", "q8_0")),
    ),
];

/// The three-way measurement on one model: CPU production against the CPU reference, then each
/// [`GPU_CONFIGS`] entry against the CPU reference (run with the configuration's KV format) and
/// against CPU production. `INFR_TOL_NO_REPLAY` runs the Vulkan decode on the per-token static
/// path (what the scratch mutations need).
pub fn measure(key: &str) {
    let Some(case) = open_case(key) else {
        eprintln!("skip: {key} not in the HF cache");
        return;
    };
    let only: Option<Vec<String>> = std::env::var("INFR_TOL_ONLY")
        .ok()
        .map(|v| v.split(',').map(str::to_string).collect());
    let no_replay = std::env::var_os("INFR_TOL_NO_REPLAY").is_some();
    let model = &case.model;
    for (pi, ids) in case.prompts.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let r = reference(model, ids, N_DECODE);
        let t_ref = t0.elapsed();
        let t0 = std::time::Instant::now();
        let prod = forced(model, ForcedOn::Cpu, |_| {}, &r);
        let t_prod = t0.elapsed();
        let tag = |what: &str| format!("model={key} prompt={pi} len={} {what}", ids.len());
        println!("{}", line(&tag("pair=cpuprod-ref"), &score(&r, &prod, 0.0)));
        for &(name, edit, kv) in GPU_CONFIGS {
            if only.as_ref().is_some_and(|o| !o.iter().any(|n| n == name)) {
                continue;
            }
            let reference = match kv {
                None => r.clone(),
                Some((k, v)) => forced(model, ForcedOn::CpuReference, |c| kv_types(c, k, v), &r),
            };
            let t0 = std::time::Instant::now();
            let g = forced(
                model,
                ForcedOn::Vulkan,
                |c| {
                    edit(c);
                    c.kernels.vulkan.no_replay |= no_replay;
                },
                &r,
            );
            let t_gpu = t0.elapsed();
            println!(
                "{}",
                line(
                    &tag(&format!("pair={name}-ref")),
                    &score(&reference, &g, 0.0)
                )
            );
            println!(
                "{}",
                line(
                    &tag(&format!("pair={name}-cpuprod")),
                    &score(&prod, &g, 0.0)
                )
            );
            eprintln!("time {key} prompt {pi} {name}: {:.1}s", t_gpu.as_secs_f64());
        }
        eprintln!(
            "time {key} prompt {pi}: reference {:.1}s, cpu production {:.1}s",
            t_ref.as_secs_f64(),
            t_prod.as_secs_f64()
        );
    }
}

/// The gate: on every prompt of the `key` model, each named [`GPU_CONFIGS`] entry must be within
/// the tolerance table's entry for (`key`, its class) of the CPU reference, on the prefill rows
/// and on the decode rows alike; over the whole case, at least
/// [`infr_llama::tolerance::MIN_TOP1_SCORED`] positions must clear the near-tie margin. Skips
/// without the model (failing instead under `INFR_TEST_REQUIRE_MODELS`) or without a Vulkan
/// device. Prints every measured value, so a new device's numbers can be read off the log.
pub fn gate(key: &str, runs: &[(&str, PathClass)]) {
    let Some(case) = open_case(key) else {
        assert!(
            std::env::var_os("INFR_TEST_REQUIRE_MODELS").is_none(),
            "{key} is not in the HF cache, and INFR_TEST_REQUIRE_MODELS requires it"
        );
        eprintln!("skip: {key} not in the HF cache");
        return;
    };
    if infr_testkit::gpu_or_skip(infr_vulkan::VulkanBackend::new_with(std::sync::Arc::new(
        env_cfg(),
    )))
    .is_none()
    {
        eprintln!("skip: no Vulkan device");
        return;
    }
    let t_case = std::time::Instant::now();
    let mut failures = Vec::new();
    let mut merged: Vec<Option<RunScore>> = vec![None; runs.len()];
    for (pi, ids) in case.prompts.iter().enumerate() {
        let r = reference(&case.model, ids, N_DECODE);
        for (ri, &(name, class)) in runs.iter().enumerate() {
            let &(_, edit, kv) = GPU_CONFIGS
                .iter()
                .find(|c| c.0 == name)
                .unwrap_or_else(|| panic!("no configuration {name:?}"));
            let tol = tolerance(table_name(key), class);
            let reference = match kv {
                None => r.clone(),
                Some((k, v)) => forced(
                    &case.model,
                    ForcedOn::CpuReference,
                    |c| kv_types(c, k, v),
                    &r,
                ),
            };
            let g = forced(&case.model, ForcedOn::Vulkan, edit, &r);
            let s = score(&reference, &g, tol.near_tie);
            for (phase, ps) in [("prefill", &s.prefill), ("decode", &s.decode)] {
                println!(
                    "{key} {name} ({class:?}) prompt {pi} ({} tokens), {phase}:\n{}",
                    ids.len(),
                    ps.report()
                );
                for v in tol.violations(ps) {
                    failures.push(format!("{name} prompt {pi} {phase}: {v}"));
                }
            }
            let all = s.all();
            match &mut merged[ri] {
                Some(m) => m.merge(&all),
                slot => *slot = Some(all),
            }
        }
    }
    for (&(name, _), m) in runs.iter().zip(&merged) {
        let m = m.as_ref().expect("every run scored");
        if m.top1_scored < MIN_TOP1_SCORED {
            failures.push(format!(
                "{name}: only {} of {} positions clear the {}-nat near-tie margin (at least \
                 {MIN_TOP1_SCORED} needed, or the top-1 check is vacuous)",
                m.top1_scored, m.positions, m.near_tie
            ));
        }
    }
    eprintln!(
        "time {key}: {:.1}s on {}",
        t_case.elapsed().as_secs_f64(),
        std::env::var("INFR_DEV").unwrap_or_else(|_| "the default device".into())
    );
    assert!(
        failures.is_empty(),
        "{key} is outside the GPU tolerance of the CPU reference:\n  {}",
        failures.join("\n  ")
    );
}

// ─── the driver's own check ──────────────────────────────────────────────────────

/// A small llama with every projection in `dt` (an f32 embedding and norms), on a 64-token
/// vocabulary written as single printable characters.
pub fn synthetic_llama(weights: Fill) -> SyntheticModel {
    const NE: usize = 256;
    const NFF: usize = 512;
    const N_HEAD: usize = 4;
    const N_KV: usize = 2;
    const HD: usize = NE / N_HEAD;
    const VOCAB: usize = 64;
    let k = |key: &str| format!("llama.{key}");
    let u = |key: &str, v: usize| (k(key), Meta::U32(v as u32));
    let meta = vec![
        (
            "general.architecture".to_string(),
            Meta::Str("llama".to_string()),
        ),
        u("block_count", 2),
        u("context_length", 512),
        u("embedding_length", NE),
        u("feed_forward_length", NFF),
        u("attention.head_count", N_HEAD),
        u("attention.head_count_kv", N_KV),
        u("rope.dimension_count", HD),
        (k("rope.freq_base"), Meta::F32(1.0e4)),
        (k("attention.layer_norm_rms_epsilon"), Meta::F32(1.0e-5)),
        (
            "tokenizer.ggml.model".to_string(),
            Meta::Str("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Meta::StrArr(
                (0..VOCAB)
                    .map(|i| char::from(b'!' + i as u8).to_string())
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            Meta::StrArr(Vec::new()),
        ),
        ("tokenizer.ggml.eos_token_id".to_string(), Meta::U32(0)),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            Meta::Bool(false),
        ),
    ];
    let w = |name: String, shape: Vec<usize>, amp: f32| match &weights {
        Fill::Quant(dt) => TensorSpec::new(name, shape, Fill::Quant(*dt)),
        Fill::ExactF16(_) => {
            let t = TensorSpec::new(name, shape, Fill::Rand(amp));
            let v = t.values();
            TensorSpec::new(t.name, t.shape, Fill::ExactF16(v))
        }
        _ => TensorSpec::new(name, shape, Fill::Rand(amp)),
    };
    let mut tensors = vec![
        TensorSpec::new("token_embd.weight", vec![NE, VOCAB], Fill::Rand(1.0)),
        TensorSpec::new("output_norm.weight", vec![NE], Fill::Gain),
        w("output.weight".into(), vec![NE, VOCAB], 0.5),
    ];
    for l in 0..2 {
        let n = |s: &str| format!("blk.{l}.{s}.weight");
        tensors.push(TensorSpec::new(n("attn_norm"), vec![NE], Fill::Gain));
        tensors.push(TensorSpec::new(n("ffn_norm"), vec![NE], Fill::Gain));
        tensors.push(w(n("attn_q"), vec![NE, N_HEAD * HD], 0.3));
        tensors.push(w(n("attn_k"), vec![NE, N_KV * HD], 0.3));
        tensors.push(w(n("attn_v"), vec![NE, N_KV * HD], 0.3));
        tensors.push(w(n("attn_output"), vec![N_HEAD * HD, NE], 0.2));
        tensors.push(w(n("ffn_gate"), vec![NE, NFF], 0.2));
        tensors.push(w(n("ffn_up"), vec![NE, NFF], 0.2));
        tensors.push(w(n("ffn_down"), vec![NFF, NE], 0.1));
    }
    SyntheticModel { meta, tensors }
}

/// Panic unless `got` and `want` are the same logits bit for bit, naming the first row (of
/// `vocab`) that differs.
fn assert_same_bits(got: &[f32], want: &[f32], vocab: usize, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        panic!(
            "{what}: row {} differs first, at vocab index {} ({} against {})",
            i / vocab,
            i % vocab,
            got[i],
            want[i]
        );
    }
}

/// The driver feeds the tokens it says it fed, at the positions it says it scored: on the CPU
/// reference backend, whose output does not depend on how rows are batched, a forced run must
/// give the greedy run's logits bit for bit at another prefill chunk height, and with more of the
/// sequence taken as prompt (rows that came from decode steps then come from the prefill).
pub fn self_check_on_a_synthetic_llama() {
    let tmp = TempGguf::write(
        "forced-run-q8",
        &synthetic_llama(Fill::Quant(infr_core::DType::Q8_0)),
    );
    let model = SeamModel::load_with(
        tmp.path(),
        None,
        std::sync::Arc::new(EngineConfig::default()),
    )
    .expect("synthetic model load");
    let prompt: Vec<u32> = (0..24).map(|i| (i * 7 + 3) % 64).collect();
    let n = 10;
    let r = reference(&model, &prompt, n);
    assert_eq!(r.prefill_rows, prompt.len() - 1);
    assert_eq!(r.positions, (0..prompt.len() - 1 + n).collect::<Vec<_>>());
    assert_eq!(r.seq.len(), prompt.len() + n);
    assert_eq!(r.logits.len(), r.positions.len() * r.vocab);

    // Another chunk height: the prefill now runs in five chunks.
    let small = forced(
        &model,
        ForcedOn::CpuReference,
        |c| c.device.ubatch = Some(5),
        &r,
    );
    assert_same_bits(&small.logits, &r.logits, r.vocab, "chunked prefill");

    // Four decoded tokens taken as prompt: positions 23..=26 now come from the prefill.
    let longer = model
        .forced_logits(
            ForcedOn::CpuReference,
            |_| {},
            &r.seq[..prompt.len() + 4],
            n - 4,
            |s, _| r.seq[prompt.len() + 4 + s],
        )
        .expect("longer prompt");
    assert_eq!(longer.positions, r.positions);
    assert_eq!(longer.prefill_rows, prompt.len() + 3);
    assert_same_bits(
        &longer.logits,
        &r.logits,
        r.vocab,
        "prefill rows against decode rows at the same positions",
    );

    // The production CPU, forced along the same tokens, quantizes its activations to int8: the
    // metrics must see a distance, or the run never reached the backend it names.
    let prod = forced(&model, ForcedOn::Cpu, |_| {}, &r);
    let s = score(&r, &prod, 0.0).all();
    assert!(
        s.kl.worst.value > 0.0,
        "production CPU matched the reference bit for bit"
    );
}
