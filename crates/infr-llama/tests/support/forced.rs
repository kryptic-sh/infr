//! The teacher-forced comparison as the tests drive it: the cases they score (cached models by
//! key, synthetic models, planted copies), the measurement, and the gate with its watchdog and
//! skips. The evaluation itself is the library's (`infr_llama::tolerance::gate` and `greedy`),
//! the one `infr tune check` runs too; the functions of the same names here call it, print what
//! it logs, and panic where it returns an error.

use infr_llama::seam::{ForcedLogits, ForcedOn, PrefillHead, Routing};
use infr_llama::tolerance::gate;
pub use infr_llama::tolerance::gate::{
    argmax, kv_types, try_score, vulkan_float, vulkan_float_nocoopmat, GpuConfig, Reference,
    Scored, GPU_CONFIGS, N_DECODE, SMALL_UBATCH,
};
use infr_llama::tolerance::oracle::{Mode, RefCache};
use infr_llama::tolerance::{PathClass, RunScore};

/// The residency rule that keeps a model off an integrated device without the host memory for it.
#[path = "hostroom.rs"]
mod hostroom;
/// The routed-expert (MoE) half: the synthetic model, the routing comparison, its self-check.
#[path = "routed.rs"]
mod routed;
use infr_llama::{EngineConfig, SeamModel};
use infr_testkit::gguf::{Fill, Meta, SyntheticModel, TempGguf, TensorSpec};
pub use routed::{route_line, self_check_on_a_synthetic_moe, synthetic_moe, SYNTHETIC_MOE};

/// The greedy check's test-side entry points, for the planted-bug demonstrations (the gate runs
/// the check inside [`gate::check`]).
#[cfg(feature = "mutations")]
pub mod greedy {
    use infr_llama::tolerance::greedy as check;
    use infr_llama::tolerance::PathClass;

    /// Whether the test device packs several decode steps into one submit
    /// ([`check::device_chains`] on the test environment's configuration).
    pub fn device_chains() -> bool {
        check::device_chains(&std::sync::Arc::new(super::env_cfg()))
            .unwrap_or_else(|e| panic!("{e:#}"))
    }

    /// The library's greedy check ([`check::greedy`]) on a test case: its report lines printed,
    /// its failures returned, a run that could not be made a panic.
    pub fn greedy(
        key: &str,
        table: &str,
        reference: &super::Reference,
        test: &super::Case,
        runs: &[(&str, PathClass)],
    ) -> Vec<String> {
        let found = check::greedy(key, table, reference, test, runs, &mut |e| println!("{e}"))
            .unwrap_or_else(|e| panic!("{e:#}"));
        for n in &found.not_exercised {
            eprintln!("{n}");
        }
        found.failures
    }
}
#[cfg(feature = "mutations")]
pub use greedy::greedy;

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
    (
        "qwen3-30b-moe",
        "unsloth--Qwen3-30B-A3B-GGUF",
        "Qwen3-30B-A3B-Q4_K_M.gguf",
    ),
];

/// A [`gate::Case`] the tests opened, with what only they need of it.
pub struct Case {
    case: gate::Case,
    /// The model file's size, for the host-memory check on unified-memory devices.
    pub bytes: u64,
    /// The synthetic model's file, kept for as long as the model maps it.
    _file: Option<TempGguf>,
}

impl std::ops::Deref for Case {
    type Target = gate::Case;
    fn deref(&self) -> &gate::Case {
        &self.case
    }
}

impl std::ops::DerefMut for Case {
    fn deref_mut(&mut self) -> &mut gate::Case {
        &mut self.case
    }
}

/// The [`Case`] for a [`MODELS`] key — [`gate::PROMPTS`] tokenized — or for `synthetic-f32` /
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
        case: gate::Case::of_model(model).expect("the model's prompts"),
        bytes,
        _file: None,
    })
}

/// The prompts of a synthetic model, one per role of [`gate::PROMPTS`] (short, medium, long
/// enough to span several [`SMALL_UBATCH`] chunks), as ids of its vocabulary.
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
/// with [`gate::PROMPTS`]. `batched_prefill` is [`Case::batched_prefill`].
pub fn open_file(file: TempGguf, synthetic: bool, batched_prefill: bool) -> Case {
    let bytes = std::fs::metadata(file.path()).map_or(0, |md| md.len());
    let model = load(file.path());
    let case = if synthetic {
        gate::Case {
            prompts: synthetic_prompts(),
            chat_prompt: None,
            model,
            batched_prefill,
            chains: false,
        }
    } else {
        gate::Case {
            batched_prefill,
            ..gate::Case::of_model(model).expect("the model's prompts")
        }
    };
    Case {
        case,
        bytes,
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

/// The CPU reference's greedy run on `prompt` ([`gate::forced_reference`]).
pub fn forced_reference(model: &SeamModel, prompt: &[u32], n: usize) -> ForcedLogits {
    gate::forced_reference(model, prompt, n).unwrap_or_else(|e| panic!("{e:#}"))
}

/// `reference`'s tokens, and on a routed-expert model its expert selections, forced through
/// `on`, its engine configuration adjusted by `edit` ([`gate::forced`]).
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
/// every router's own) ([`gate::forced_along`]).
pub fn forced_along(
    model: &SeamModel,
    on: ForcedOn,
    edit: impl FnOnce(&mut EngineConfig),
    head: PrefillHead,
    reference: &ForcedLogits,
    route: Option<&Routing>,
) -> ForcedLogits {
    gate::forced_along(model, on, edit, head, reference, route).unwrap_or_else(|e| panic!("{e:#}"))
}

/// [`try_score`], panicking on a row that cannot be scored.
pub fn score(
    reference: &ForcedLogits,
    test: &ForcedLogits,
    near_steady: f64,
    near_first: f64,
) -> Scored {
    try_score(reference, test, near_steady, near_first).unwrap_or_else(|e| panic!("{e}"))
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

/// Configurations for diagnosis, run by [`measure`] only when `INFR_TOL_ONLY` names them and
/// scored by no gate. Each splits the int8 kernels, so each sets the packed int8 dot on
/// ([`int8_on`]) over a table row that turns it off on the device, which would otherwise leave
/// nothing int8 to split.
pub const DIAGNOSIS_CONFIGS: &[GpuConfig] = &[
    // The shipped configuration without the dp4a K-quant prefill GEMM (`kernels.vulkan.dp4a_kq`):
    // its weights then take the `mmq` GEMM, still on int8 activations.
    GpuConfig {
        name: "gpu-no-dp4a-kq",
        edit: |c| {
            int8_on(c);
            c.kernels.vulkan.dp4a_kq = false;
        },
        reference_kv: None,
    },
    // Where the int8 error a routed-expert model's routers see comes from
    // (`docs/gpu-tolerance.md`, "MoE routing, int8 against float"). The router's own GEMV is
    // float in every configuration (no router dispatch of the Vulkan adapter reaches an int8
    // tier, whatever the router's dtype), so these split the int8 kernels upstream of it, at `SMALL_UBATCH` so every prefill chunk fits the small-batch MoE path. With
    // `moe.small_m` at the chunk height the routed experts take the per-row float GEMVs at every
    // row (decode already does) and every dense `Linear` keeps its int8 kernels, the decode rows'
    // too (`mmv_mw.moe_float` off).
    GpuConfig {
        name: "gpu-ub32-moe-float",
        edit: |c| {
            int8_on(c);
            c.device.ubatch = Some(SMALL_UBATCH);
            c.kernels.vulkan.moe_small_m = SMALL_UBATCH;
            tune_off(c, "mmv_mw.moe_float");
        },
        reference_kv: None,
    },
    // The other half: the dense `Linear`s (attention, shared expert, dense FFN) off the int8
    // kernels their own switches name, the routed experts' batched prefill GEMM on int8 (it is
    // gated on `i8_dot` alone) with the plain quantizer (`quant_q8.experts_hi` off). A device
    // without cooperative matrices keeps its dense prefill GEMM on int8 here too:
    // `nc_gemm_choice` picks the int8 `mmq` tile on `i8_dot` and ignores `kernels.vulkan.mmq`;
    // only `mmq_fallback` off takes that GEMM away (the last entry).
    GpuConfig {
        name: "gpu-ub32-dense-float",
        edit: dense_float_ub32,
        reference_kv: None,
    },
    // Both halves: on a device without cooperative matrices, the dense prefill GEMM is the one
    // int8 kernel left; on one with them, nothing int8 runs.
    GpuConfig {
        name: "gpu-ub32-both-float",
        edit: both_float_ub32,
        reference_kv: None,
    },
    // Both halves with the non-coopmat tiled GEMM off (`mmq_fallback`), so a device without
    // cooperative matrices runs its dense prefill on the float GEMVs: `i8_dot` on, no int8 kernel
    // dispatched on the routed path.
    GpuConfig {
        name: "gpu-ub32-both-float-nofallback",
        edit: |c| {
            both_float_ub32(c);
            c.kernels.vulkan.mmq_fallback = false;
        },
        reference_kv: None,
    },
    // The routed experts' batched GEMMs back on the plain int8 quantizer (the tuning parameter
    // `quant_q8.experts_hi` off, `docs/gpu-tolerance.md`), everything else as shipped: at the
    // shipped chunk height and at `SMALL_UBATCH`, beside `gpu` and `gpu-ub32`.
    GpuConfig {
        name: "gpu-no-experts-hi",
        edit: no_experts_hi,
        reference_kv: None,
    },
    GpuConfig {
        name: "gpu-ub32-no-experts-hi",
        edit: |c| {
            no_experts_hi(c);
            c.device.ubatch = Some(SMALL_UBATCH);
        },
        reference_kv: None,
    },
    // The dense decode GEMVs on float activations, everything else as shipped:
    // `kernels.vulkan.mmv_mw = false` empties the int8 decode set (`mmv_int8_decode_dtypes`), so
    // a decode row's dense `Linear`s take the dequantizing GEMVs; a prefill chunk keeps its int8
    // multi-row GEMVs and GEMMs, and the routed experts their int8 batched GEMM. A model with
    // routed experts already runs this way as shipped (`mmv_mw.moe_float`).
    GpuConfig {
        name: "gpu-dense-decode-float",
        edit: |c| {
            int8_on(c);
            c.kernels.vulkan.mmv_mw = Some(false);
        },
        reference_kv: None,
    },
    // A routed-expert model's one-row `Linear`s back on the int8 decode GEMVs (the tuning
    // parameter `mmv_mw.moe_float` off), everything else as shipped; a dense model runs exactly
    // what `gpu` runs (`gpu_seam_tolerance_moe_decode_float_reaches_moe_models_only`).
    GpuConfig {
        name: "gpu-moe-decode-int8",
        edit: |c| {
            int8_on(c);
            tune_off(c, "mmv_mw.moe_float");
        },
        reference_kv: None,
    },
    // Both MoE switches off: the routed-expert configuration shipped before they went on.
    GpuConfig {
        name: "gpu-moe-switches-off",
        edit: |c| {
            no_experts_hi(c);
            tune_off(c, "mmv_mw.moe_float");
        },
        reference_kv: None,
    },
];

/// `gpu-no-experts-hi`'s edit: the tuning parameter `quant_q8.experts_hi` off, the packed int8
/// dot on.
fn no_experts_hi(c: &mut EngineConfig) {
    int8_on(c);
    tune_off(c, "quant_q8.experts_hi");
}

/// The packed int8 dot on (`device.i8_dot`), over a table row that turns it off on the device.
/// `kernels.vulkan.i8_dot` keeps its own setting: either one off still turns the dot off.
pub fn int8_on(c: &mut EngineConfig) {
    tune(c, "device.i8_dot", "true");
}

/// The boolean tuning parameter `key` off.
fn tune_off(c: &mut EngineConfig, key: &str) {
    tune(c, key, "false");
}

/// The tuning parameter `key` set to `value` at the command line's level, the highest, so neither
/// a table row nor another layer of the test environment overrides it.
fn tune(c: &mut EngineConfig, key: &str, value: &str) {
    c.kernels
        .vulkan
        .tune
        .entries
        .push(infr_core::config::TuneEntry {
            key: key.into(),
            value: value.into(),
            level: infr_core::config::TuneLevel::Cli,
        });
}

/// `gpu-ub32-both-float`'s edit: [`dense_float_ub32`] with every row on the small-batch MoE path.
fn both_float_ub32(c: &mut EngineConfig) {
    dense_float_ub32(c);
    c.kernels.vulkan.moe_small_m = SMALL_UBATCH;
}

/// `gpu-ub32-dense-float`'s edit: [`vulkan_float`] with `i8_dot` left on, at [`SMALL_UBATCH`], the
/// routed experts' batched GEMM on the plain quantizer.
fn dense_float_ub32(c: &mut EngineConfig) {
    vulkan_float(c);
    c.kernels.vulkan.i8_dot = true;
    c.device.ubatch = Some(SMALL_UBATCH);
    no_experts_hi(c);
}

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
/// leaves the CPU production and Vulkan runs on their own routers' selections instead (what a user
/// gets), the comparison a router near tie throws off (`docs/gpu-tolerance.md`), for telling a
/// routing difference from an arithmetic one. The `{name}-cpuprod` lines then compare two runs that
/// each followed their own routers, not one routing.
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
        let prod = forced_along(
            model,
            ForcedOn::Cpu,
            |_| {},
            PrefillHead::All,
            &r,
            r.routing.as_ref().filter(|_| !own_routing),
        );
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
        let diagnosis = DIAGNOSIS_CONFIGS.iter().filter(|c| named(c.name));
        let gated = GPU_CONFIGS
            .iter()
            .filter(|c| only.is_none() || named(c.name));
        for &GpuConfig {
            name,
            edit,
            reference_kv: kv,
        } in gated.chain(diagnosis)
        {
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
    gate::gpu_config(name)
        .unwrap_or_else(|e| panic!("{e:#}"))
        .edit
}

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
    if let Err(why) = hostroom::host_room(integrated, bytes) {
        eprintln!("skip: {key} {why}, on {device}");
        return None;
    }
    Some(device)
}

/// The gate ([`gate::check`], with the runs the table's rows give the `key` model,
/// [`gate::runs_for`]): each run within the model's rows of the CPU reference on every prompt,
/// then the greedy check. Skips, with a printed line, without the model (failing instead under
/// `INFR_TEST_REQUIRE_MODELS`), without a Vulkan device, and on an integrated device without the
/// host memory for the model. Fails a run that outlives [`GATE_TIMEOUT`]. Prints every measured
/// value, so a new device's numbers can be read off the log.
pub fn gate(key: &str) {
    let _guard = watchdog(&format!("the {key} tolerance case"), GATE_TIMEOUT);
    let Some(case) = open_or_skip(key) else {
        return;
    };
    let Some(device) = device_for(key, case.bytes) else {
        return;
    };
    let reference = reference_case(&case);
    let cache = ref_cache();
    let t_case = std::time::Instant::now();
    let table = table_name(key);
    let runs = gate::runs_for(table).unwrap_or_else(|| panic!("no tolerance row for {table}"));
    let verdict = gate::check_cached(key, table, &reference, &case, &runs, cache, &mut |e| {
        println!("{e}")
    })
    .unwrap_or_else(|e| panic!("{e:#}"));
    for n in verdict.not_exercised() {
        eprintln!("{n}");
    }
    let failures = verdict.failures();
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

/// The case the gate's CPU reference runs on: `case`'s model file loaded again under the shipped
/// defaults, with `case`'s prompts. Nothing of the test environment's GPU configuration reaches
/// it, so one reference serves every device and mode, and its cache key does not split by them;
/// `infr tune check` loads its reference the same way.
pub fn reference_case(case: &Case) -> gate::Case {
    let path = case.model.gguf().shard_handles()[0].path.to_path_buf();
    let model = SeamModel::load_with(&path, None, std::sync::Arc::new(EngineConfig::default()))
        .expect("reference model load");
    gate::Case {
        model,
        prompts: case.prompts.clone(),
        chat_prompt: case.chat_prompt.clone(),
        batched_prefill: case.batched_prefill,
        chains: case.chains,
    }
}

/// What the gate does with the reference cache, decided from `CI` and `INFR_TOL_REFCACHE` alone
/// ([`cache_choice`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheChoice {
    /// Run without it, for the reason given.
    Off(&'static str),
    /// Remove everything it owns, then run without it.
    Clear,
    /// Open it in `mode`; `asked` when `INFR_TOL_REFCACHE` named the mode rather than defaulting.
    Open { mode: Mode, asked: bool },
}

/// [`CacheChoice`] for `CI` being set (`ci`) and `INFR_TOL_REFCACHE`'s value (`setting`): off
/// whenever `CI` is set, `on` when unset, else the named mode; an error naming the value for one
/// it does not know.
pub fn cache_choice(ci: bool, setting: Option<&str>) -> Result<CacheChoice, String> {
    if ci {
        return Ok(CacheChoice::Off("CI is set"));
    }
    let open = |mode| CacheChoice::Open { mode, asked: true };
    Ok(match setting {
        None => CacheChoice::Open {
            mode: Mode::On,
            asked: false,
        },
        Some("on") => open(Mode::On),
        Some("verify") => open(Mode::Verify),
        Some("refresh") => open(Mode::Refresh),
        Some("off") => CacheChoice::Off("INFR_TOL_REFCACHE=off"),
        Some("clear") => CacheChoice::Clear,
        Some(other) => {
            return Err(format!(
                "INFR_TOL_REFCACHE={other:?}: expected off, on, verify, refresh or clear"
            ))
        }
    })
}

/// Whether a cache that did not open fails the run (`true`) or lets it run without the cache: it
/// fails only for a build with no code fingerprint when the mode was asked for by name. Anything
/// else (a read-only or full cache directory, a lock that cannot be made, no fingerprint under
/// the default) is the machine's, not the gate's, and runs without the cache.
pub fn refuse_without_cache(asked: bool, has_fingerprint: bool) -> bool {
    asked && !has_fingerprint
}

/// The reference cache the gate reads, as [`cache_choice`] decides from the environment. Opened
/// once per process, which is when its cleanup runs; its lines go to stderr. A cache that cannot
/// be opened is reported in one line and the gate runs without it, unless
/// [`refuse_without_cache`] says otherwise. See `docs/gpu-tolerance.md`, "The reference cache".
pub fn ref_cache() -> Option<&'static RefCache> {
    static CACHE: std::sync::OnceLock<Option<RefCache>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            let setting = std::env::var("INFR_TOL_REFCACHE").ok();
            let choice = cache_choice(std::env::var_os("CI").is_some(), setting.as_deref())
                .unwrap_or_else(|e| panic!("{e}"));
            let root = RefCache::default_root();
            let (mode, asked) = match choice {
                CacheChoice::Off(why) => {
                    eprintln!("refcache: off, {why}");
                    return None;
                }
                CacheChoice::Clear => {
                    if let Some(root) = &root {
                        let removed = RefCache::clear(root);
                        eprintln!("refcache: cleared {removed} files under {}", root.display());
                    }
                    eprintln!("refcache: off after the clear");
                    return None;
                }
                CacheChoice::Open { mode, asked } => (mode, asked),
            };
            let has_fingerprint = infr_llama::tolerance::oracle::fingerprint().is_some();
            let Some(root) = root else {
                eprintln!("refcache: OFF, no home directory to keep it under");
                return None;
            };
            match RefCache::open(&root, mode, |l| eprintln!("{l}")) {
                Ok(cache) => {
                    eprintln!("refcache: {mode:?} at {}", root.display());
                    Some(cache)
                }
                Err(e) if refuse_without_cache(asked, has_fingerprint) => {
                    panic!("{e:#} (INFR_TOL_REFCACHE=off runs the gate without the cache)")
                }
                Err(e) => {
                    eprintln!(
                        "refcache: OFF, the cache at {} could not be opened, so every reference \
                         is computed: {e:#}",
                        root.display()
                    );
                    None
                }
            }
        })
        .as_ref()
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
    let reference = self::reference(reference);
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

/// `case` with its CPU reference's greedy run on each prompt ([`Reference::new`]).
pub fn reference(case: &Case) -> Reference<'_> {
    Reference::new(case).unwrap_or_else(|e| panic!("{e:#}"))
}

/// [`Reference::check_prefill`], what it could not exercise printed, for the planted-bug
/// demonstrations.
#[cfg(feature = "mutations")]
pub fn check_prefill(reference: &Reference, label: &str, failures: &mut Vec<String>) {
    let mut not_exercised = Vec::new();
    reference.check_prefill(label, failures, &mut not_exercised);
    for n in not_exercised {
        eprintln!("{n}");
    }
}

/// The library's evaluation ([`gate::evaluate_with`]) on test cases: its report lines printed,
/// the limits that were crossed returned, a run that could not be made a panic.
pub fn evaluate_with(
    key: &str,
    table: &str,
    reference: &Reference,
    test: &Case,
    on: ForcedOn,
    runs: &[(&str, PathClass)],
    seen: impl FnMut(usize, &str, &ForcedLogits, Option<&Scored>),
) -> Vec<String> {
    let cmp = gate::Comparison {
        key,
        table,
        reference,
        test,
        on,
    };
    gate::evaluate_with(&cmp, runs, seen, &mut |e| println!("{e}"))
        .unwrap_or_else(|e| panic!("{e:#}"))
        .failures
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
