//! The teacher-forced comparison: the CPU reference's greedy run, the same tokens forced through
//! another backend, and the two scored with `infr_llama::tolerance`.

use infr_llama::seam::{ForcedLogits, ForcedOn, PrefillHead, Routing};
use infr_llama::tolerance::{
    tolerance, PathClass, RouteScore, RunScore, FIRST_ROWS, MIN_TOP1_SCORED,
};

/// The routed-expert (MoE) half: the synthetic model, the routing comparison, its self-check.
#[path = "routed.rs"]
mod routed;
use infr_llama::{EngineConfig, SeamModel};
use infr_testkit::gguf::{Fill, Meta, SyntheticModel, TempGguf, TensorSpec};
pub use routed::{
    route_line, score_routing, self_check_on_a_synthetic_moe, synthetic_moe, SYNTHETIC_MOE,
};

/// The production greedy decode held to the per-token path, and its tokens to the reference.
#[path = "greedy.rs"]
pub mod greedy;
pub use greedy::greedy;

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
    /// The model file's size, for the host-memory check on unified-memory devices.
    pub bytes: u64,
    /// Whether the model's prompt goes through the batched prefill. `false` only for a model
    /// known to have none (the synthetic MoE: float expert banks have no batched kernel), whose
    /// prompt positions are then not scored on logits ([`Reference::check_prefill`]).
    pub batched_prefill: bool,
    /// Whether the model's greedy decode chains several steps into one submit on a device that
    /// allows it. `false` for the synthetic models, whose token embedding is gathered on the
    /// host, so every step has host work between it and the next.
    pub chains: bool,
    /// The prompt the greedy check also generates from: a chat-templated one
    /// ([`greedy::chat_prompt`]), for a model with a chat template. `None` for the synthetic
    /// models, whose vocabulary spells nothing.
    pub chat_prompt: Option<Vec<u32>>,
    /// The synthetic model's file, kept for as long as the model maps it.
    _file: Option<TempGguf>,
}

/// The [`Case`] for a [`MODELS`] key — [`PROMPTS`] tokenized — or for `synthetic-f32` /
/// `synthetic-f16` ([`synthetic_llama`] with f32 or f16 projections) or [`SYNTHETIC_MOE`]
/// ([`synthetic_moe`]), each prompted with ids at the same three lengths; `None` when a cached
/// model is not downloaded.
pub fn open_case(key: &str) -> Option<Case> {
    open_case_scaled(key, None)
}

/// [`open_case`], for a synthetic model with one tensor's values multiplied by a factor
/// (`(tensor name, factor)`): the planted bug of the red-path tests, made in the model file rather
/// than in any backend, so the backend under test runs its production code on wrong weights.
pub fn open_case_scaled(key: &str, scaled: Option<(&str, f32)>) -> Option<Case> {
    let weights = match key {
        "synthetic-f32" => Some(Fill::Rand(0.0)),
        "synthetic-f16" => Some(Fill::ExactF16(Vec::new())),
        _ => None,
    };
    if let Some(w) = weights {
        let mut m = synthetic_llama(w);
        let mut tag = format!("tolerance-{key}");
        if let Some((name, factor)) = scaled {
            let t = m
                .tensors
                .iter_mut()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("no tensor {name}"));
            let v: Vec<f32> = t.values().iter().map(|x| x * factor).collect();
            t.fill = match t.fill {
                Fill::ExactF16(_) => Fill::ExactF16(v),
                _ => Fill::Exact(v),
            };
            tag = format!("{tag}-{name}-x{factor}");
        }
        return Some(open_file(TempGguf::write(&tag, &m), true, true));
    }
    assert!(scaled.is_none(), "only a synthetic llama can be scaled");
    if key == SYNTHETIC_MOE {
        let file = TempGguf::write(&format!("tolerance-{key}"), &synthetic_moe());
        return Some(open_file(file, true, false));
    }
    let path = find_model(key)?;
    let bytes = std::fs::metadata(&path).map_or(0, |md| md.len());
    let model = load(&path);
    Some(Case {
        prompts: real_prompts(&model),
        chat_prompt: Some(greedy::chat_prompt(&model)),
        model,
        bytes,
        batched_prefill: true,
        chains: true,
        _file: None,
    })
}

/// [`PROMPTS`], tokenized by `model`.
pub fn real_prompts(model: &SeamModel) -> Vec<Vec<u32>> {
    PROMPTS
        .iter()
        .map(|t| model.encode(t).expect("encode"))
        .collect()
}

/// The prompts of a synthetic model, one per role of [`PROMPTS`] (short, medium, long enough to
/// span several [`SMALL_UBATCH`] chunks), as ids of its vocabulary.
pub fn synthetic_prompts() -> Vec<Vec<u32>> {
    [12usize, 34, 180]
        .iter()
        .enumerate()
        .map(|(salt, &n)| {
            (0..n)
                .map(|i| ((i * 7 + 3 + salt * 11) % 64) as u32)
                .collect()
        })
        .collect()
}

/// The [`Case`] of a model written to `file`, which it keeps for as long as the model maps it:
/// prompted with the [`synthetic_prompts`] when `synthetic` (its vocabulary spells nothing), else
/// with [`PROMPTS`]. `batched_prefill` is [`Case::batched_prefill`].
pub fn open_file(file: TempGguf, synthetic: bool, batched_prefill: bool) -> Case {
    let bytes = std::fs::metadata(file.path()).map_or(0, |md| md.len());
    let model = load(file.path());
    Case {
        prompts: if synthetic {
            synthetic_prompts()
        } else {
            real_prompts(&model)
        },
        chat_prompt: (!synthetic).then(|| greedy::chat_prompt(&model)),
        model,
        bytes,
        batched_prefill,
        chains: !synthetic,
        _file: Some(file),
    }
}

/// The name the tolerance table knows a [`MODELS`] key (or a synthetic model) by: the GGUF file
/// name, which is what a command scoring a user's model has in hand.
pub fn table_name(key: &str) -> &'static str {
    match key {
        "synthetic-f32" => "synthetic-f32",
        "synthetic-f16" => "synthetic-f16",
        SYNTHETIC_MOE => SYNTHETIC_MOE,
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
pub fn forced_reference(model: &SeamModel, prompt: &[u32], n: usize) -> ForcedLogits {
    model
        .forced_logits(
            ForcedOn::CpuReference,
            |_| {},
            prompt,
            n,
            PrefillHead::All,
            None,
            |_, row| argmax(row),
        )
        .expect("cpu reference run")
}

/// `reference`'s tokens, and on a routed-expert model its expert selections, forced through
/// `on`, its engine configuration adjusted by `edit`.
pub fn forced(
    model: &SeamModel,
    on: ForcedOn,
    edit: impl FnOnce(&mut EngineConfig),
    reference: &ForcedLogits,
) -> ForcedLogits {
    forced_with(model, on, edit, PrefillHead::All, reference)
}

/// [`forced`] scoring the prefill rows `head` names.
pub fn forced_with(
    model: &SeamModel,
    on: ForcedOn,
    edit: impl FnOnce(&mut EngineConfig),
    head: PrefillHead,
    reference: &ForcedLogits,
) -> ForcedLogits {
    forced_along(model, on, edit, head, reference, reference.routing.as_ref())
}

/// [`forced_with`] along `reference`'s tokens and the expert selections `route` holds (`None`:
/// every router's own).
pub fn forced_along(
    model: &SeamModel,
    on: ForcedOn,
    edit: impl FnOnce(&mut EngineConfig),
    head: PrefillHead,
    reference: &ForcedLogits,
    route: Option<&Routing>,
) -> ForcedLogits {
    let steps = reference.positions.len() - reference.prefill_rows;
    let p = reference.seq.len() - steps;
    let got = model
        .forced_logits(on, edit, &reference.seq[..p], steps, head, route, |s, _| {
            reference.seq[p + s]
        })
        .unwrap_or_else(|e| panic!("forced run on {on:?}: {e:#}"));
    if head != PrefillHead::All {
        assert!(
            got.positions
                .iter()
                .all(|q| reference.positions.contains(q)),
            "{on:?} scored positions the reference has no row for"
        );
        return got;
    }
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

/// One comparison, split by where the rows came from: the first [`FIRST_ROWS`] positions (whose
/// hidden states carry a model's largest activations, and which every quantized path misses by
/// the most), the rest of the batched prefill's rows, and the decode steps' rows.
pub struct Scored {
    pub first: RunScore,
    pub prefill: RunScore,
    pub decode: RunScore,
}

impl Scored {
    /// The three phases with their names, first rows first.
    pub fn phases(&self) -> [(&'static str, &RunScore); 3] {
        [
            ("first", &self.first),
            ("prefill", &self.prefill),
            ("decode", &self.decode),
        ]
    }
}

/// Score every row of `test` against the reference's row at the same position: the first rows at
/// the `near_first` near-tie margin, the rest at `near_steady`. `test` may score a subset of the
/// reference's prefill positions ([`PrefillHead::LastOfChunk`]); its decode rows are the
/// reference's.
pub fn score(
    reference: &ForcedLogits,
    test: &ForcedLogits,
    near_steady: f64,
    near_first: f64,
) -> Scored {
    try_score(reference, test, near_steady, near_first).unwrap_or_else(|e| panic!("{e}"))
}

/// [`score`], returning a row that cannot be scored (a NaN or infinite logit, named with its
/// position) as an error instead of panicking.
pub fn try_score(
    reference: &ForcedLogits,
    test: &ForcedLogits,
    near_steady: f64,
    near_first: f64,
) -> Result<Scored, String> {
    let v = reference.vocab;
    let ref_decode = &reference.positions[reference.prefill_rows..];
    assert_eq!(
        &test.positions[test.prefill_rows..],
        ref_decode,
        "decode positions differ"
    );
    let mut s = Scored {
        first: RunScore::new(near_first),
        prefill: RunScore::new(near_steady),
        decode: RunScore::new(near_steady),
    };
    for (r, &pos) in test.positions.iter().enumerate() {
        let k = reference
            .positions
            .iter()
            .position(|&q| q == pos)
            .unwrap_or_else(|| panic!("the reference has no row at position {pos}"));
        let phase = if pos < FIRST_ROWS {
            &mut s.first
        } else if r < test.prefill_rows {
            &mut s.prefill
        } else {
            &mut s.decode
        };
        phase.add_rows(
            &reference.logits[k * v..(k + 1) * v],
            &test.logits[r * v..(r + 1) * v],
            v,
            &[pos],
        )?;
    }
    Ok(s)
}

/// One line for the measurement log with every scored position's KL, in position order, as
/// `position:KL`; a position whose top-1 differs from the reference's carries `!` and the
/// reference's margin between the two tokens in nats. For asking whether one row stands out, and
/// whether it moves with the prompt's end (`INFR_TOL_SPLIT`).
pub fn rows_line(tag: &str, reference: &ForcedLogits, test: &ForcedLogits) -> String {
    let v = reference.vocab;
    let mut line = format!("ROWS {tag} prefill_rows={}", test.prefill_rows);
    for (r, &pos) in test.positions.iter().enumerate() {
        let k = reference
            .positions
            .iter()
            .position(|&q| q == pos)
            .unwrap_or_else(|| panic!("the reference has no row at position {pos}"));
        let (want, got) = (
            &reference.logits[k * v..(k + 1) * v],
            &test.logits[r * v..(r + 1) * v],
        );
        match infr_llama::tolerance::score_row(want, got) {
            Ok(s) => line.push_str(&format!(" {pos}:{:.3e}", s.kl)),
            Err(e) => line.push_str(&format!(" {pos}:unscored({e})")),
        }
        let (a, b) = (argmax(want) as usize, argmax(got) as usize);
        if a != b {
            line.push_str(&format!("!{:.3}", want[a] - want[b]));
        }
    }
    line
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
    s.phases()
        .iter()
        .filter(|(_, r)| r.positions > 0)
        .map(|(p, r)| one(p, r))
        .collect::<Vec<_>>()
        .join("\n")
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

/// [`vulkan_float`] with the whole cooperative-matrix tier off as well (`kernels.vulkan.coopmat`,
/// what `INFR_NO_COOPMAT` sets): the prefill GEMMs then take f32 operands, on every device.
pub fn vulkan_float_nocoopmat(c: &mut EngineConfig) {
    vulkan_float(c);
    c.kernels.vulkan.coopmat = false;
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
        "gpu-float-static",
        |c| {
            vulkan_float(c);
            c.kernels.vulkan.no_replay = true;
        },
        None,
    ),
    ("gpu-float-nocoopmat", vulkan_float_nocoopmat, None),
    (
        "gpu-float-nocoopmat-ub32",
        |c| {
            vulkan_float_nocoopmat(c);
            c.device.ubatch = Some(SMALL_UBATCH);
        },
        None,
    ),
    (
        "gpu-float-nocoopmat-static",
        |c| {
            vulkan_float_nocoopmat(c);
            c.kernels.vulkan.no_replay = true;
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

/// Configurations for diagnosis, run by [`measure`] only when `INFR_TOL_ONLY` names them and
/// scored by no gate.
pub const DIAGNOSIS_CONFIGS: &[GpuConfig] = &[
    // The shipped configuration without the dp4a K-quant prefill GEMM (`kernels.vulkan.dp4a_kq`):
    // its weights then take the `mmq` GEMM, still on int8 activations.
    ("gpu-no-dp4a-kq", |c| c.kernels.vulkan.dp4a_kq = false, None),
];

/// The three-way measurement on one model: CPU production against the CPU reference, then each
/// [`GPU_CONFIGS`] entry against the CPU reference (run with the configuration's KV format) and
/// against CPU production. `INFR_TOL_NO_REPLAY` runs the Vulkan decode on the per-token static
/// path (what the scratch mutations need). `INFR_TOL_SPLIT` (a token count) moves the end of the
/// prompt: the reference's greedy sequence is kept, its first that-many tokens are fed as the
/// prompt and the rest forced, so the same positions are scored with the prefill-to-decode
/// boundary elsewhere; the forced run refuses a prompt shorter than its batched prefill needs (the
/// check at the top of `seam::forced_run`). `INFR_TOL_ROWS` adds a [`rows_line`] per comparison and
/// the kernels the run built. On a routed-expert model every run is forced along the reference's
/// expert selections and a [`route_line`] per comparison gives its own; `INFR_TOL_OWN_ROUTING`
/// leaves the Vulkan runs on their own routers' selections instead, the comparison a router near
/// tie throws off (`docs/gpu-tolerance.md`), for telling a routing difference from an arithmetic
/// one.
pub fn measure(key: &str) {
    let Some(case) = open_case(key) else {
        eprintln!("skip: {key} not in the HF cache");
        return;
    };
    let only: Option<Vec<String>> = std::env::var("INFR_TOL_ONLY")
        .ok()
        .map(|v| v.split(',').map(str::to_string).collect());
    let no_replay = std::env::var_os("INFR_TOL_NO_REPLAY").is_some();
    // Diagnosis knobs: the submit-splitter cap, the prefill chunk height, which prefill rows get
    // the harness's head, and which prompts run.
    let env_usize = |k: &str| std::env::var(k).ok().map(|v| v.parse::<usize>().expect(k));
    let submit = env_usize("INFR_TOL_SUBMIT");
    let ubatch = env_usize("INFR_TOL_UBATCH");
    let head = match std::env::var("INFR_TOL_HEAD").as_deref() {
        Ok("last") => PrefillHead::LastOfChunk,
        Ok("off") => PrefillHead::Off,
        _ => PrefillHead::All,
    };
    let prompts: Option<Vec<usize>> = std::env::var("INFR_TOL_PROMPTS").ok().map(|v| {
        v.split(',')
            .map(|i| i.parse().expect("prompt index"))
            .collect()
    });
    let split = env_usize("INFR_TOL_SPLIT");
    let rows = std::env::var_os("INFR_TOL_ROWS").is_some();
    let own_routing = std::env::var_os("INFR_TOL_OWN_ROUTING").is_some();
    let model = &case.model;
    for (pi, ids) in case.prompts.iter().enumerate() {
        if prompts.as_ref().is_some_and(|p| !p.contains(&pi)) {
            continue;
        }
        let t0 = std::time::Instant::now();
        let mut r = forced_reference(model, ids, N_DECODE);
        if let Some(p) = split {
            // The same sequence with the prompt ending at `p`: on the CPU reference the rows are
            // the greedy run's (`forced_run_reproduces_the_cpu_reference_on_every_path`); only
            // which of them the batched prefill produced changes.
            let seq = r.seq;
            assert!(p < seq.len(), "INFR_TOL_SPLIT is past the sequence's end");
            r = model
                .forced_logits(
                    ForcedOn::CpuReference,
                    |_| {},
                    &seq[..p],
                    seq.len() - p,
                    PrefillHead::All,
                    None,
                    |s, _| seq[p + s],
                )
                .expect("cpu reference run at the moved prompt end");
        }
        let t_ref = t0.elapsed();
        let t0 = std::time::Instant::now();
        let prod = forced(model, ForcedOn::Cpu, |_| {}, &r);
        let t_prod = t0.elapsed();
        let tag = |what: &str| format!("model={key} prompt={pi} len={} {what}", ids.len());
        println!(
            "{}",
            line(&tag("pair=cpuprod-ref"), &score(&r, &prod, 0.0, 0.0))
        );
        if let Some(l) = route_line(&tag("pair=cpuprod-ref"), &r, &prod) {
            println!("{l}");
        }
        let named = |name: &str| only.as_ref().is_some_and(|o| o.iter().any(|n| n == name));
        let diagnosis = DIAGNOSIS_CONFIGS.iter().filter(|c| named(c.0));
        let gated = GPU_CONFIGS.iter().filter(|c| only.is_none() || named(c.0));
        for &(name, edit, kv) in gated.chain(diagnosis) {
            let reference = match kv {
                None => r.clone(),
                Some((k, v)) => forced(model, ForcedOn::CpuReference, |c| kv_types(c, k, v), &r),
            };
            let t0 = std::time::Instant::now();
            let g = forced_along(
                model,
                ForcedOn::Vulkan,
                |c| {
                    edit(c);
                    c.kernels.vulkan.no_replay |= no_replay;
                    if submit.is_some() {
                        c.device.submit_dispatches = submit;
                    }
                    if ubatch.is_some() {
                        c.device.ubatch = ubatch;
                    }
                },
                head,
                &r,
                r.routing.as_ref().filter(|_| !own_routing),
            );
            let t_gpu = t0.elapsed();
            println!(
                "{}",
                line(
                    &tag(&format!("pair={name}-ref")),
                    &score(&reference, &g, 0.0, 0.0)
                )
            );
            println!(
                "{}",
                line(
                    &tag(&format!("pair={name}-cpuprod")),
                    &score(&prod, &g, 0.0, 0.0)
                )
            );
            if let Some(l) = route_line(&tag(&format!("pair={name}-ref")), &r, &g) {
                println!("{l}");
            }
            if rows {
                println!(
                    "{}",
                    rows_line(&tag(&format!("pair={name}-ref")), &reference, &g)
                );
                println!("KERNELS {} {:?}", tag(&format!("pair={name}")), g.kernels);
            }
            eprintln!("time {key} prompt {pi} {name}: {:.1}s", t_gpu.as_secs_f64());
        }
        eprintln!(
            "time {key} prompt {pi}: reference {:.1}s, cpu production {:.1}s",
            t_ref.as_secs_f64(),
            t_prod.as_secs_f64()
        );
    }
}

/// The longest one [`gate`] call may run. A forced run that stops making progress (the integrated
/// GPU's long Float runs did, B207) fails the test with a message saying so, rather than holding
/// the gate until someone notices. About 4.5 times the slowest test measured (LFM3, both files, on
/// the integrated GPU), so a slower device has room. The bound is per call: a test that calls
/// `gate` twice can run twice as long, and with the test process ended the tests after it in the
/// same process do not run (each nextest test is its own process; plain `cargo test` is one).
pub const GATE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20 * 60);

/// Fails the whole test process with a message if it is still alive when the deadline passes;
/// dropping it disarms it. Each nextest test is its own process, so this ends only the test it
/// guards.
pub struct Watchdog(Option<std::sync::mpsc::Sender<()>>);

impl Drop for Watchdog {
    fn drop(&mut self) {
        // Closing the channel wakes the watcher, which then exits quietly.
        self.0.take();
    }
}

/// Arm a [`Watchdog`] for `what` with the deadline `limit`.
pub fn watchdog(what: &str, limit: std::time::Duration) -> Watchdog {
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let what = what.to_string();
    std::thread::spawn(move || {
        if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = rx.recv_timeout(limit) {
            // Straight to the process's stderr: this thread inherits libtest's output capture,
            // which `eprintln!` would write into and `exit` would then discard.
            use std::io::Write;
            let msg = format!(
                "FAIL: {what} still running after {} s: a hang, not a tolerance result; \
                 stopping the test process (docs/backlog.md B207)\n",
                limit.as_secs()
            );
            // A failed write leaves the exit status as the only report; nothing else can carry it.
            let _ = std::io::stderr().write_all(msg.as_bytes());
            std::process::exit(1);
        }
    });
    Watchdog(Some(tx))
}

/// The config edit of the [`GPU_CONFIGS`] entry named `name`.
pub fn gpu_config_edit(name: &str) -> fn(&mut EngineConfig) {
    gpu_config(name).1
}

/// The [`GPU_CONFIGS`] entry named `name`.
fn gpu_config(name: &str) -> &'static GpuConfig {
    GPU_CONFIGS
        .iter()
        .find(|c| c.0 == name)
        .unwrap_or_else(|| panic!("no configuration {name:?}"))
}

/// Host bytes a model resident on a unified-memory device takes per byte of its file, and the
/// headroom left for everything else on the machine: `cpu_backend`'s `UNIFIED_RESIDENT_FACTOR` and
/// `UNIFIED_HOST_MARGIN`, the measured figures for a resident load. A run that cannot have both
/// is skipped rather than left to starve the host.
const UNIFIED_RESIDENT_FACTOR: u64 = 2;
const UNIFIED_HOST_MARGIN: u64 = 12 << 30;

/// The device the tests load on, if a Vulkan backend opens: `(integrated, name)`.
fn test_device() -> Option<(bool, String)> {
    infr_testkit::gpu_or_skip(infr_vulkan::VulkanBackend::new_with(std::sync::Arc::new(
        env_cfg(),
    )))?;
    let dev = infr_vulkan::VulkanBackend::enumerate_devices(&env_cfg())
        .ok()?
        .into_iter()
        .find(|d| d.is_default_pick)?;
    Some((
        dev.integrated,
        format!("Vulkan{} ({})", dev.index, dev.name),
    ))
}

/// [`open_case`], or `None` with a printed `skip:` line when the model is not downloaded (a
/// failure instead under `INFR_TEST_REQUIRE_MODELS`).
pub fn open_or_skip(key: &str) -> Option<Case> {
    let case = open_case(key);
    if case.is_none() {
        assert!(
            std::env::var_os("INFR_TEST_REQUIRE_MODELS").is_none(),
            "{key} is not in the HF cache, and INFR_TEST_REQUIRE_MODELS requires it"
        );
        eprintln!("skip: {key} not in the HF cache");
    }
    case
}

/// The device the `key` model's `bytes` of model files will run on, named; `None`, with a printed
/// `skip:` line, without a Vulkan device or on an integrated device without the host memory for
/// them.
pub fn device_for(key: &str, bytes: u64) -> Option<String> {
    let Some((integrated, device)) = test_device() else {
        eprintln!("skip: no Vulkan device");
        return None;
    };
    if integrated {
        let need = bytes * UNIFIED_RESIDENT_FACTOR + UNIFIED_HOST_MARGIN;
        match infr_core::hostmem::available_bytes() {
            Some(avail) if avail >= need => {}
            avail => {
                eprintln!(
                    "skip: {key} on {device}, a unified-memory device, needs {:.1} GB of free host \
                     memory ({UNIFIED_RESIDENT_FACTOR} x the model + a {:.1} GB margin); {} free",
                    need as f64 / 1e9,
                    UNIFIED_HOST_MARGIN as f64 / 1e9,
                    avail.map_or("unknown".to_string(), |a| format!("{:.1} GB", a as f64 / 1e9)),
                );
                return None;
            }
        }
    }
    Some(device)
}

/// The gate: on every prompt of the `key` model, each named [`GPU_CONFIGS`] entry must be within
/// the tolerance table's entry for (`key`, its class) of the CPU reference: the first
/// [`FIRST_ROWS`] positions within its `first` bound, the batched prefill's other rows and the
/// decode rows each within its `steady` bound, and each of those two phases must score at least
/// [`MIN_TOP1_SCORED`] positions above the near-tie margin. Then the greedy check ([`greedy`])
/// holds the chained and the replayed decode, which no forced run exercises, to the per-token
/// path, and the greedy tokens to the reference. Skips, with a printed line, without
/// the model (failing instead under `INFR_TEST_REQUIRE_MODELS`), without a Vulkan device, and on
/// an integrated device without the host memory for the model. Fails a run that outlives
/// [`GATE_TIMEOUT`]. Prints every measured value, so a new device's numbers can be read off the
/// log.
pub fn gate(key: &str, runs: &[(&str, PathClass)]) {
    let _guard = watchdog(&format!("the {key} tolerance case"), GATE_TIMEOUT);
    let Some(case) = open_or_skip(key) else {
        return;
    };
    let Some(device) = device_for(key, case.bytes) else {
        return;
    };
    let t_case = std::time::Instant::now();
    let reference = Reference::new(&case);
    let mut failures = evaluate_with(
        key,
        table_name(key),
        &reference,
        &case,
        ForcedOn::Vulkan,
        runs,
        |_, _, _, _| {},
    );
    reference.check_prefill(key, &mut failures);
    let t_greedy = std::time::Instant::now();
    failures.extend(greedy(key, table_name(key), &reference, &case, runs));
    eprintln!(
        "time {key}: the greedy check {:.1}s",
        t_greedy.elapsed().as_secs_f64()
    );
    eprintln!(
        "time {key}: {:.1}s on {device}",
        t_case.elapsed().as_secs_f64()
    );
    assert!(
        failures.is_empty(),
        "{key} is outside the GPU tolerance of the CPU reference:\n  {}",
        failures.join("\n  ")
    );
}

/// Score each run of `test` (driven on `on`) against the CPU reference's run of `reference` along
/// the reference's own greedy tokens, against the table's entry for `key`; return every limit
/// that was crossed, described. The two cases differ only in the red-path tests, where `test`
/// carries a planted bug.
pub fn evaluate(
    key: &str,
    reference: &Case,
    test: &Case,
    on: ForcedOn,
    runs: &[(&str, PathClass)],
) -> Vec<String> {
    let reference = Reference::new(reference);
    evaluate_with(
        key,
        table_name(key),
        &reference,
        test,
        on,
        runs,
        |_, _, _, _| {},
    )
}

/// A case with its CPU reference's greedy run on each prompt, computed once for every
/// [`evaluate_with`] that scores against it.
pub struct Reference<'a> {
    pub case: &'a Case,
    pub runs: Vec<ForcedLogits>,
}

impl<'a> Reference<'a> {
    pub fn new(case: &'a Case) -> Self {
        let runs = case
            .prompts
            .iter()
            .map(|ids| forced_reference(&case.model, ids, N_DECODE))
            .collect();
        Self { case, runs }
    }

    /// Hold the reference's prefill rows to what the case says of its model, adding to
    /// `failures`. A case marked as having no batched prefill ([`Case::batched_prefill`]) must
    /// have produced no prefill row on any prompt; [`evaluate_with`]'s count of scored prefill
    /// rows then reports zero of zero, which says nothing about the run, and that one failure is
    /// dropped, with a printed line (the decode rows are scored, and a routed model's routing at
    /// every position). Any other case must have prefill rows on every prompt: a model that
    /// lost its batched prefill fails here by name.
    pub fn check_prefill(&self, label: &str, failures: &mut Vec<String>) {
        let without: Vec<usize> = (0..self.runs.len())
            .filter(|&pi| self.runs[pi].prefill_rows == 0)
            .collect();
        if self.case.batched_prefill {
            if !without.is_empty() {
                failures.push(format!(
                    "{label}: the reference took no batched prefill on prompts {without:?}, and \
                     the case is not one marked as having none"
                ));
            }
            return;
        }
        if without.len() != self.runs.len() {
            failures.push(format!(
                "{label}: the case is marked as having no batched prefill, and the reference \
                 took one on {} of its {} prompts",
                self.runs.len() - without.len(),
                self.runs.len()
            ));
            return;
        }
        eprintln!(
            "{label}: the model takes no batched prefill; its decode rows are scored, and a \
             routed model's routing at every position"
        );
        failures.retain(|f| !f.contains(" prefill: only 0 of 0 positions"));
    }
}

/// [`evaluate`] against a [`Reference`] computed once, scored against the table's entry for
/// `table` (a model's [`table_name`]), with `seen(prompt, run name, run, score)` called after each
/// run: the score is `None` when a row could not be scored, which is then a failure naming it.
///
/// A routed-expert model is forced along the reference's expert selections too, and its own
/// selections are held to the entry's `route_near_tie` at every position and layer; over a case,
/// at least [`MIN_TOP1_SCORED`] of the reference's selections must clear that margin. A row
/// with a routing near tie and a run without a routing, or the reverse, is a failure.
pub fn evaluate_with(
    key: &str,
    table: &str,
    reference: &Reference,
    test: &Case,
    on: ForcedOn,
    runs: &[(&str, PathClass)],
    mut seen: impl FnMut(usize, &str, &ForcedLogits, Option<&Scored>),
) -> Vec<String> {
    let mut failures = Vec::new();
    // Per run, the steady phases merged over every prompt, for the top-1 count.
    let mut merged: Vec<[Option<RunScore>; 2]> = vec![[None, None]; runs.len()];
    // Per run, the routing comparison merged over every prompt.
    let mut routed: Vec<Option<RouteScore>> = vec![None; runs.len()];
    for (pi, (ids, r)) in reference
        .case
        .prompts
        .iter()
        .zip(&reference.runs)
        .enumerate()
    {
        for (ri, &(name, class)) in runs.iter().enumerate() {
            let &(_, edit, kv) = gpu_config(name);
            let entry = tolerance(table, class);
            let reference_run = match kv {
                None => r.clone(),
                Some((k, v)) => forced(
                    &reference.case.model,
                    ForcedOn::CpuReference,
                    |c| kv_types(c, k, v),
                    r,
                ),
            };
            let g = forced_with(&test.model, on, edit, PrefillHead::All, r);
            match (entry.route_near_tie, g.routing.is_some()) {
                (Some(near), true) => {
                    let rs = score_routing(r, &g, near).expect("both runs carry a routing");
                    println!(
                        "{key} {name} ({class:?}) prompt {pi} ({} tokens):\n{}",
                        ids.len(),
                        rs.report()
                    );
                    for v in rs.violations() {
                        failures.push(format!("{name} prompt {pi} routing: {v}"));
                    }
                    match &mut routed[ri] {
                        Some(m) => m.merge(&rs),
                        slot => *slot = Some(rs),
                    }
                }
                (None, false) => {}
                (near, _) => failures.push(format!(
                    "{name} prompt {pi} routing: the {table} {class:?} row has {} routing near \
                     tie and the run has {} routing",
                    if near.is_some() { "a" } else { "no" },
                    if near.is_some() { "no" } else { "a" },
                )),
            }
            let s = match try_score(
                &reference_run,
                &g,
                entry.steady.near_tie,
                entry.first.near_tie,
            ) {
                Ok(s) => s,
                Err(e) => {
                    seen(pi, name, &g, None);
                    failures.push(format!("{name} prompt {pi}: {e}"));
                    continue;
                }
            };
            seen(pi, name, &g, Some(&s));
            for (phase, ps) in s.phases() {
                if ps.positions == 0 {
                    continue;
                }
                println!(
                    "{key} {name} ({class:?}) prompt {pi} ({} tokens), {phase}:\n{}",
                    ids.len(),
                    ps.report()
                );
                let tol = if phase == "first" {
                    &entry.first
                } else {
                    &entry.steady
                };
                for v in tol.violations(ps) {
                    failures.push(format!("{name} prompt {pi} {phase}: {v}"));
                }
            }
            for (slot, ps) in merged[ri].iter_mut().zip([&s.prefill, &s.decode]) {
                match slot {
                    Some(m) => m.merge(ps),
                    None => *slot = Some(ps.clone()),
                }
            }
        }
    }
    for (&(name, _), rs) in runs.iter().zip(&routed) {
        if let Some(rs) = rs.as_ref().filter(|rs| rs.scored < MIN_TOP1_SCORED) {
            failures.push(format!(
                "{name} routing: only {} of {} selections clear the {:.3e} near tie (at least \
                 {MIN_TOP1_SCORED} needed, or the routing check is vacuous)",
                rs.scored, rs.calls, rs.near_tie
            ));
        }
    }
    for (&(name, _), m) in runs.iter().zip(&merged) {
        for (phase, ps) in ["prefill", "decode"].iter().zip(m) {
            // No score: every prompt's row of this run failed to score, each a failure above.
            let Some(ps) = ps else { continue };
            if ps.top1_scored < MIN_TOP1_SCORED {
                failures.push(format!(
                    "{name} {phase}: only {} of {} positions clear the {}-nat near-tie margin \
                     (at least {MIN_TOP1_SCORED} needed, or the top-1 check is vacuous)",
                    ps.top1_scored, ps.positions, ps.near_tie
                ));
            }
        }
    }
    failures
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
    let r = forced_reference(&model, &prompt, n);
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
            PrefillHead::All,
            None,
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
    let s = score(&r, &prod, 0.0, 0.0);
    assert!(
        s.phases().iter().any(|(_, p)| p.kl.worst.value > 0.0),
        "production CPU matched the reference bit for bit"
    );
}
