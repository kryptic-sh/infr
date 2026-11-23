//! The mutation demonstrations (`docs/gpu-tolerance.md`, "Mutations"; `--features mutations`):
//! each bug planted in one loaded copy of a model, that copy driven on the backend under test and
//! scored by the gate against the clean copy's CPU reference.
//!
//! Graph bugs are planted with `SeamModel::plant`; weight bugs are written into a copy of the
//! model file ([`patch`]). A clean control runs first, twice: it must pass the gate, and its two
//! runs must agree bit for bit, so a planted run that differs from it differs because of the bug.
//! Every planted run must differ from the clean one on every decode row set it scores (the bug
//! reached decode); where the gate must catch the bug, the run must fail it, naming a metric;
//! where the doc records that it misses, the measured distance is printed and nothing more is
//! asserted, so a tighter table later cannot break the test.

mod patch;

use super::forced::{self, Case, Reference, Scored};
use infr_llama::mutation::Planted;
use infr_llama::seam::{ForcedLogits, ForcedOn};
use infr_llama::tolerance::{tolerance, PathClass};
use infr_testkit::gguf::{Fill, Meta, SyntheticModel, TempGguf, TensorSpec};
use patch::Patch;
use PathClass::{Float, FloatNoCoopmat, Int8};

/// The planted bugs.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Bug {
    RopeQueryAhead,
    MaskDropsLastKey,
    RmsEpsScaled,
    FfnDownScaled,
    FfnDownBlockScaled,
    GateUpSwapped,
    AttnResidualDropped,
    RopePairing,
    NibbleMap,
    RouterIgnoresBias,
}

/// What [`Bug::RmsEpsScaled`] multiplies the RMSNorm epsilon by.
const EPS_FACTOR: f32 = 10.0;
/// What [`Bug::FfnDownScaled`] and [`Bug::FfnDownBlockScaled`] scale their weights by.
const BLOCK_SCALE: f32 = 1.01;

impl Bug {
    /// The bug's name in the doc's table.
    fn label(self) -> &'static str {
        match self {
            Bug::RopeQueryAhead => "(a) rope query at position + 1",
            Bug::MaskDropsLastKey => "(b) mask drops the last visible key",
            Bug::RmsEpsScaled => "(c) RMSNorm epsilon x10",
            Bug::FfnDownScaled => "(d) one layer's FFN output x1.01, through its last weight",
            Bug::FfnDownBlockScaled => "(d1) one block of one ffn_down x1.01",
            Bug::GateUpSwapped => "(e) gate and up swapped",
            Bug::AttnResidualDropped => "(f) one layer's attention residual dropped",
            Bug::RopePairing => "(g) rope pairs the wrong elements",
            Bug::NibbleMap => "(h) Q4_K nibbles swapped in one ffn_down",
            Bug::RouterIgnoresBias => "(i) MoE router ignores its bias",
        }
    }

    /// The graph bug this plants, if it is one.
    fn planted(self, t: &Target) -> Option<Planted> {
        match self {
            Bug::RopeQueryAhead => Some(Planted::RopeQueryAhead),
            Bug::MaskDropsLastKey => Some(Planted::MaskDropsLastKey),
            Bug::RmsEpsScaled => Some(Planted::RmsEpsScaled(EPS_FACTOR)),
            Bug::AttnResidualDropped => Some(Planted::AttnResidualDropped {
                layer: t.resid_layer,
            }),
            _ => None,
        }
    }

    /// Whether the gate may miss this bug on `t` in `class`, on every device: the misses
    /// `docs/gpu-tolerance.md` records. Such a run is measured and printed, never required to
    /// pass or to fail. A row that only catches gross bugs (every Int8 row, and the rows a
    /// target lists in [`Target::smoke`]) may miss the fine ones; a fine row only the single
    /// mis-scaled block.
    fn may_miss(self, t: &Target, class: PathClass) -> bool {
        if class == Int8 || t.smoke.contains(&class) {
            matches!(
                self,
                Bug::RmsEpsScaled | Bug::FfnDownScaled | Bug::FfnDownBlockScaled
            )
        } else {
            self == Bug::FfnDownBlockScaled
        }
    }
}

/// A model to plant bugs in and where they go.
struct Target {
    /// A `forced::MODELS` key, or `synthetic-f32` / [`SYNTHETIC_MOE`].
    key: &'static str,
    /// The tolerance table's row the model is scored against.
    table: &'static str,
    /// The layer whose `ffn_down` is scaled or misread.
    ffn_layer: usize,
    /// The weight of that layer [`Bug::FfnDownScaled`] scales, by its name in the block: the last
    /// one the FFN sublayer's output passes through before the residual add, so that the scale
    /// reaches the residual stream. `ffn_down`, unless a norm follows it.
    ffn_last: &'static str,
    /// The layer whose attention residual is dropped.
    resid_layer: usize,
    /// The model's rope pairs `(p, p + head_dim/2)` (NEOX), not `(2p, 2p + 1)`.
    neox: bool,
    /// The model's vocabulary spells nothing; it is prompted with ids.
    synthetic: bool,
    /// The classes, besides Int8, whose row for this model the table labels a smoke check.
    smoke: &'static [PathClass],
}

const QWEN3: Target = Target {
    key: "qwen3-0.6b",
    table: "Qwen3-0.6B-Q4_K_M.gguf",
    ffn_layer: 10,
    ffn_last: "ffn_down",
    resid_layer: 14,
    neox: true,
    synthetic: false,
    smoke: &[],
};

/// gemma-3-1b: the fused per-head norm and rope of Qwen3 (so the same pairing), and a Q4_K_M file
/// whose `ffn_down` tensors alternate between `Q4_K` and `Q6_K`; the layer named here holds a
/// `Q4_K` one, which [`Bug::NibbleMap`] needs. Its sandwich norms renormalize each sublayer's
/// output before the residual add, so a scale of the whole `ffn_down` never reaches the residual
/// stream (planted there it moved the run's worst KL only 1.3 times off the clean run's,
/// `docs/gpu-tolerance.md`, "Mutations"); the post-FFN norm's gain is the weight whose scale
/// does. Its Float row is a smoke check of the
/// cooperative-matrix f16 prefill path (`TABLE`), FloatNoCoopmat the fine one.
const GEMMA3: Target = Target {
    key: "gemma3-1b",
    table: "gemma-3-1b-it-Q4_K_M.gguf",
    ffn_layer: 10,
    ffn_last: "post_ffw_norm",
    resid_layer: 13,
    neox: true,
    synthetic: false,
    smoke: &[Float],
};

const ERNIE_F16: Target = Target {
    key: "ernie-0.3b-f16",
    table: "ERNIE-4.5-0.3B-PT-F16.gguf",
    ffn_layer: 6,
    ffn_last: "ffn_down",
    resid_layer: 9,
    neox: false,
    synthetic: false,
    smoke: &[],
};

const SYNTHETIC: Target = Target {
    key: "synthetic-f32",
    table: "synthetic-f32",
    ffn_layer: 1,
    ffn_last: "ffn_down",
    resid_layer: 1,
    neox: false,
    synthetic: true,
    smoke: &[],
};

/// The synthetic MoE's key. It has no table row of its own: it is held to the synthetic f32
/// llama's Float row, whose arithmetic (f32 weights, f16 KV) it shares. Its clean control passes
/// that row on the CPU reference (a distance of 0) and not on the RX 7900 XTX, so only the CPU
/// demonstration is gated ([`Verdicts`]).
const SYNTHETIC_MOE: &str = "synthetic-ernie-moe";

const MOE: Target = Target {
    key: SYNTHETIC_MOE,
    table: "synthetic-f32",
    ffn_layer: 1,
    ffn_last: "ffn_down",
    resid_layer: 1,
    neox: false,
    synthetic: true,
    smoke: &[],
};

/// Experts of the synthetic MoE, how many each token uses, and their width.
const MOE_EXPERTS: usize = 8;
const MOE_USED: usize = 2;
const MOE_FF: usize = 128;
/// The router bias's amplitude: on the order of the gaps between the router's expert
/// probabilities, so it changes which experts some tokens pick.
const MOE_BIAS_AMP: f32 = 0.2;

/// [`forced::synthetic_llama`] as an `ernie4_5-moe`: every layer's FFN replaced by a softmax
/// router over [`MOE_EXPERTS`] f32 experts with a selection bias (`exp_probs_b.bias`) and an
/// ungated shared expert, the layout of ERNIE-4.5-21B-A3B (B174's model) at toy size.
fn synthetic_moe() -> SyntheticModel {
    const ARCH: &str = "ernie4_5-moe";
    let llama = forced::synthetic_llama(Fill::Rand(0.0));
    let ne = llama
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("token_embd")
        .shape[0];
    let mut meta: Vec<(String, Meta)> = llama
        .meta
        .into_iter()
        .map(|(k, v)| match k.strip_prefix("llama.") {
            Some(rest) => (format!("{ARCH}.{rest}"), v),
            None if k == "general.architecture" => (k, Meta::Str(ARCH.to_string())),
            None => (k, v),
        })
        .collect();
    let n_layer = meta
        .iter()
        .find_map(|(k, v)| match v {
            Meta::U32(n) if k == &format!("{ARCH}.block_count") => Some(*n as usize),
            _ => None,
        })
        .expect("block_count");
    for (k, v) in [
        ("expert_count", MOE_EXPERTS),
        ("expert_used_count", MOE_USED),
        ("expert_feed_forward_length", MOE_FF),
        ("expert_shared_feed_forward_length", MOE_FF),
        ("leading_dense_block_count", 0),
        ("interleave_moe_layer_step", 1),
    ] {
        meta.push((format!("{ARCH}.{k}"), Meta::U32(v as u32)));
    }
    let mut tensors: Vec<TensorSpec> = llama
        .tensors
        .into_iter()
        .filter(|t| {
            !["ffn_gate.", "ffn_up.", "ffn_down."]
                .iter()
                .any(|f| t.name.contains(f))
        })
        .collect();
    for l in 0..n_layer {
        let p = |s: &str| format!("blk.{l}.{s}");
        let r = |name: String, shape: Vec<usize>, amp: f32| {
            TensorSpec::new(name, shape, Fill::Rand(amp))
        };
        tensors.extend([
            r(p("ffn_gate_inp.weight"), vec![ne, MOE_EXPERTS], 0.3),
            r(
                p("ffn_gate_exps.weight"),
                vec![ne, MOE_FF, MOE_EXPERTS],
                0.2,
            ),
            r(p("ffn_up_exps.weight"), vec![ne, MOE_FF, MOE_EXPERTS], 0.2),
            r(
                p("ffn_down_exps.weight"),
                vec![MOE_FF, ne, MOE_EXPERTS],
                0.1,
            ),
            r(p("ffn_gate_shexp.weight"), vec![ne, MOE_FF], 0.2),
            r(p("ffn_up_shexp.weight"), vec![ne, MOE_FF], 0.2),
            r(p("ffn_down_shexp.weight"), vec![MOE_FF, ne], 0.1),
            r(p("exp_probs_b.bias"), vec![MOE_EXPERTS], MOE_BIAS_AMP),
        ]);
    }
    SyntheticModel { meta, tensors }
}

/// The synthetic models' files, written for the test that opens them.
fn synthetic_file(key: &str) -> TempGguf {
    match key {
        "synthetic-f32" => {
            TempGguf::write("planted-src", &forced::synthetic_llama(Fill::Rand(0.0)))
        }
        SYNTHETIC_MOE => TempGguf::write("planted-src-moe", &synthetic_moe()),
        _ => panic!("{key} is not synthetic"),
    }
}

/// The clean model of `t`, or `None` (printed) when it is not downloaded.
fn open_clean(t: &Target) -> Option<Case> {
    if t.synthetic {
        Some(forced::open_file(synthetic_file(t.key), true))
    } else {
        forced::open_or_skip(t.key)
    }
}

/// A copy of `t`'s model with `bug` in it: written into a copy of the file, or planted in the
/// loaded model's graph.
fn open_planted(t: &Target, bug: Bug) -> Case {
    let synth = t.synthetic.then(|| synthetic_file(t.key));
    let src = match &synth {
        Some(f) => f.path().to_path_buf(),
        None => forced::find_model(t.key).expect("the clean model opened"),
    };
    let mut p = Patch::open(&src);
    let n_layer = p.arch_u64("block_count") as usize;
    let down = format!("blk.{}.ffn_down.weight", t.ffn_layer);
    match bug {
        Bug::FfnDownScaled => p.scale(
            &format!("blk.{}.{}.weight", t.ffn_layer, t.ffn_last),
            None,
            BLOCK_SCALE,
        ),
        Bug::FfnDownBlockScaled => {
            // One block in the middle of the tensor.
            let mid = p.blocks(&down) / 2;
            p.scale(&down, Some(mid), BLOCK_SCALE);
        }
        Bug::GateUpSwapped => {
            for l in 0..n_layer {
                p.swap(
                    &format!("blk.{l}.ffn_gate.weight"),
                    &format!("blk.{l}.ffn_up.weight"),
                );
            }
        }
        Bug::RopePairing => {
            for (proj, norm, heads) in [
                ("attn_q", "attn_q_norm", p.arch_u64("attention.head_count")),
                (
                    "attn_k",
                    "attn_k_norm",
                    p.arch_u64("attention.head_count_kv"),
                ),
            ] {
                for l in 0..n_layer {
                    let w = format!("blk.{l}.{proj}.weight");
                    let n = format!("blk.{l}.{norm}.weight");
                    let head_dim = p.rows(&w) / heads as usize;
                    let norm = p.has(&n).then_some(n.as_str());
                    p.misorder_rope_pairs(&w, norm, head_dim, t.neox);
                }
            }
        }
        Bug::NibbleMap => p.swap_nibbles(&down),
        Bug::RouterIgnoresBias => {
            for l in 0..n_layer {
                p.zero(&format!("blk.{l}.exp_probs_b.bias"));
            }
        }
        Bug::RopeQueryAhead
        | Bug::MaskDropsLastKey
        | Bug::RmsEpsScaled
        | Bug::AttnResidualDropped => {}
    }
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let file = TempGguf::write_copy(dir, &format!("planted-{}-{bug:?}", t.key), &p.bytes);
    drop(p);
    drop(synth);
    let mut case = forced::open_file(file, t.synthetic);
    if let Some(planted) = bug.planted(t) {
        case.model.plant(planted);
    }
    case
}

/// One run of one prompt: the digests of its prefill rows and of its decode rows.
type Digests = std::collections::BTreeMap<(usize, String), (u64, u64)>;

/// What one evaluation measured: the gate's failures, each run's worst steady-phase KL and mean KL
/// over its bounds (infinite when a row could not be scored), and the rows' digests.
struct Outcome {
    failures: Vec<String>,
    ratios: std::collections::BTreeMap<String, (f64, f64)>,
    digests: Digests,
}

/// The digest of a run of logits, over their bit patterns.
fn digest(xs: &[f32]) -> u64 {
    infr_testkit::gguf::fnv1a64(bytemuck::cast_slice(xs))
}

/// Score `test`'s `runs` on `on` against `reference` with the gate, for the table row `table`.
fn run_case(
    label: &str,
    table: &str,
    reference: &Reference,
    test: &Case,
    on: ForcedOn,
    runs: &[(&str, PathClass)],
) -> Outcome {
    let _guard = forced::watchdog(label, forced::GATE_TIMEOUT);
    let mut ratios = std::collections::BTreeMap::new();
    let mut digests = Digests::new();
    let mut seen = |pi: usize, name: &str, g: &ForcedLogits, s: Option<&Scored>| {
        let split = g.prefill_rows * g.vocab;
        digests.insert(
            (pi, name.to_string()),
            (digest(&g.logits[..split]), digest(&g.logits[split..])),
        );
        let class = runs.iter().find(|r| r.0 == name).expect("a named run").1;
        let bound = tolerance(table, class).steady;
        let r: &mut (f64, f64) = ratios.entry(name.to_string()).or_default();
        match s {
            Some(s) => {
                for p in [&s.prefill, &s.decode]
                    .into_iter()
                    .filter(|p| p.positions > 0)
                {
                    r.0 = r.0.max(p.kl.worst.value / bound.kl_max);
                    r.1 = r.1.max(p.kl.mean() / bound.kl_mean);
                }
            }
            None => *r = (f64::INFINITY, f64::INFINITY),
        }
    };
    let mut failures = forced::evaluate_with(label, table, reference, test, on, runs, &mut seen);
    // A model whose prompt takes no batched prefill (the synthetic MoE) has no prefill rows in
    // the reference either; the gate's count of scored prefill rows then reports zero of zero,
    // which says nothing about the run. Only its decode is scored.
    if reference.runs.iter().all(|r| r.prefill_rows == 0) {
        eprintln!("{label}: the model takes no batched prefill; its decode rows alone are scored");
        failures.retain(|f| !f.contains(" prefill: only 0 of 0 positions"));
    }
    Outcome {
        failures,
        ratios,
        digests,
    }
}

/// The failures of `run` that a scored row raised (a metric over its bound, or a row that could
/// not be scored): the ones naming a prompt, not the vacuous-top-1 count over the whole run.
fn metric_failures<'a>(o: &'a Outcome, run: &str) -> Vec<&'a str> {
    let prefix = format!("{run} prompt ");
    o.failures
        .iter()
        .filter(|f| f.starts_with(&prefix))
        .map(String::as_str)
        .collect()
}

/// Whether a demonstration holds the gate's verdicts to what the doc records, or only measures.
#[derive(Clone, Copy, PartialEq)]
enum Verdicts {
    /// The clean control must pass the gate, and every bug not recorded as missed must fail it.
    Gated,
    /// The table has no row that fits the model (the synthetic MoE on a GPU, held to a borrowed
    /// row its clean run does not meet): every distance is printed, and only that the bug reached
    /// decode is asserted.
    Measured,
}

/// Plant each of `bugs` in `t`'s model in turn, drive it on `on` with `runs`, and check what the
/// gate does (see the module doc). Prints one `PLANTED` line per bug and run, the doc's table.
fn demonstrate(
    t: &Target,
    on: ForcedOn,
    runs: &[(&str, PathClass)],
    bugs: &[Bug],
    verdicts: Verdicts,
) {
    let Some(clean) = open_clean(t) else {
        return;
    };
    let device = match on {
        ForcedOn::Vulkan => match forced::device_for(t.key, 2 * clean.bytes) {
            Some(d) => d,
            None => return,
        },
        _ => format!("{on:?}"),
    };
    let t0 = std::time::Instant::now();
    let reference = Reference::new(&clean);
    let control = run_case("clean control", t.table, &reference, &clean, on, runs);
    let again = run_case(
        "clean control, again",
        t.table,
        &reference,
        &clean,
        on,
        runs,
    );
    if verdicts == Verdicts::Measured && !control.failures.is_empty() {
        println!(
            "PLANTED model={} device={device} the clean control is outside the {} row, so the \
             gate's verdicts are not asserted:\n  {}",
            t.key,
            t.table,
            control.failures.join("\n  ")
        );
    }
    assert!(
        verdicts == Verdicts::Measured || control.failures.is_empty(),
        "the clean {} fails the gate on {device}, so no planted bug can be measured against it:\n  {}",
        t.key,
        control.failures.join("\n  ")
    );
    assert_eq!(
        control.digests, again.digests,
        "two clean runs of {} on {device} differ, so a planted run differing from them proves \
         nothing",
        t.key
    );
    for (run, r) in &control.ratios {
        println!(
            "PLANTED model={} device={device} bug=clean run={run} kl_ratio={:.3} mean_ratio={:.3}",
            t.key, r.0, r.1
        );
    }
    let mut wrong = Vec::new();
    for &bug in bugs {
        let test = open_planted(t, bug);
        let o = run_case(bug.label(), t.table, &reference, &test, on, runs);
        drop(test);
        for &(run, class) in runs {
            // The prompts whose decode rows the bug left bit-identical to the clean run's.
            let unchanged: Vec<usize> = (0..clean.prompts.len())
                .filter(|&pi| {
                    let k = (pi, run.to_string());
                    o.digests[&k].1 == control.digests[&k].1
                })
                .collect();
            let caught = metric_failures(&o, run);
            let (kl, mean) = o.ratios[run];
            println!(
                "PLANTED model={} device={device} bug={:?} run={run} class={class:?} verdict={} \
                 kl_ratio={kl:.3} mean_ratio={mean:.3} first={}",
                t.key,
                bug,
                if caught.is_empty() {
                    "missed"
                } else {
                    "caught"
                },
                caught.first().copied().unwrap_or("-")
            );
            if !unchanged.is_empty() {
                wrong.push(format!(
                    "{} on {run}: decode rows identical to the clean run's on prompts \
                     {unchanged:?}: the bug did not reach decode",
                    bug.label()
                ));
            }
            if verdicts == Verdicts::Gated && caught.is_empty() && !bug.may_miss(t, class) {
                wrong.push(format!(
                    "{} on {run} ({class:?}) passed the gate: KL {kl:.3}x and mean KL {mean:.3}x \
                     its bound",
                    bug.label()
                ));
            }
        }
    }
    eprintln!(
        "time planted {}: {:.1}s on {device}",
        t.key,
        t0.elapsed().as_secs_f64()
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The bugs every dense model takes.
const DENSE_BUGS: &[Bug] = &[
    Bug::RopeQueryAhead,
    Bug::MaskDropsLastKey,
    Bug::RmsEpsScaled,
    Bug::FfnDownScaled,
    Bug::FfnDownBlockScaled,
    Bug::GateUpSwapped,
    Bug::AttnResidualDropped,
    Bug::RopePairing,
];

/// The hooks are wired, on the CPU reference (no GPU or model needed): every planted bug changes
/// the synthetic model's output, and the gate catches every one [`Bug::may_miss`] does not allow
/// it to miss, against the clean model's own reference, which passes at a distance of 0.
#[test]
fn planted_bugs_fail_the_gate_on_the_cpu() {
    let runs = &[("gpu", Float)];
    demonstrate(
        &SYNTHETIC,
        ForcedOn::CpuReference,
        runs,
        DENSE_BUGS,
        Verdicts::Gated,
    );
    demonstrate(
        &MOE,
        ForcedOn::CpuReference,
        runs,
        &[Bug::RouterIgnoresBias],
        Verdicts::Gated,
    );
}

/// Qwen3-0.6B Q4_K_M, the shipped int8 kernels and the int8-free ones.
#[test]
#[ignore = "needs Qwen3-0.6B Q4_K_M and a Vulkan GPU"]
fn gpu_planted_qwen3() {
    let mut bugs = DENSE_BUGS.to_vec();
    bugs.push(Bug::NibbleMap);
    demonstrate(
        &QWEN3,
        ForcedOn::Vulkan,
        &[("gpu", Int8), ("gpu-float", Float)],
        &bugs,
        Verdicts::Gated,
    );
}

/// The planted bugs' own size on gemma-3-1b: each planted copy on the CPU reference against the
/// clean copy's CPU reference, where the clean distance is exactly 0, held to the model's fine
/// row. No GPU's noise is in these distances, so they are the bug side of that row's gap.
#[test]
#[ignore = "needs gemma-3-1b-it Q4_K_M"]
fn planted_gemma3_bugs_fail_the_fine_row_on_the_cpu() {
    let mut bugs = DENSE_BUGS.to_vec();
    bugs.push(Bug::NibbleMap);
    demonstrate(
        &GEMMA3,
        ForcedOn::CpuReference,
        &[("gpu", FloatNoCoopmat)],
        &bugs,
        Verdicts::Gated,
    );
}

/// gemma-3-1b Q4_K_M in each class its rows are scored in: the shipped int8 kernels, the
/// int8-free ones, and the int8-free ones with the cooperative-matrix tier off.
#[test]
#[ignore = "needs gemma-3-1b-it Q4_K_M and a Vulkan GPU"]
fn gpu_planted_gemma3() {
    let mut bugs = DENSE_BUGS.to_vec();
    bugs.push(Bug::NibbleMap);
    demonstrate(
        &GEMMA3,
        ForcedOn::Vulkan,
        &[
            ("gpu", Int8),
            ("gpu-float", Float),
            ("gpu-float-nocoopmat", FloatNoCoopmat),
        ],
        &bugs,
        Verdicts::Gated,
    );
}

/// ERNIE-4.5-0.3B F16, a float-weight model (no nibbles to misread).
#[test]
#[ignore = "needs ERNIE-4.5-0.3B F16 and a Vulkan GPU"]
fn gpu_planted_ernie_f16() {
    demonstrate(
        &ERNIE_F16,
        ForcedOn::Vulkan,
        &[("gpu", Float)],
        DENSE_BUGS,
        Verdicts::Gated,
    );
}

/// The router bias B174 found ignored, on a synthetic ERNIE-4.5 MoE (the real 21B model is not
/// cached here). A measurement: the table has no MoE row, and the clean run on the RX 7900 XTX
/// (RADV) lands outside the synthetic f32 llama's row it borrows, so the gate's verdict means
/// nothing here; that the bug reaches decode is asserted, and both distances are printed.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn gpu_planted_router_bias() {
    demonstrate(
        &MOE,
        ForcedOn::Vulkan,
        &[("gpu", Float)],
        &[Bug::RouterIgnoresBias],
        Verdicts::Measured,
    );
}
