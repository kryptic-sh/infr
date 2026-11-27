//! The gate's evaluation: the CPU reference's greedy run on the gate's prompts, the same tokens
//! forced through another backend in each named configuration, the two scored row by row, and
//! every limit of the model's tolerance rows that a run crossed.
//!
//! This is the one evaluation: the `gpu_seam_tolerance_*` tests and `infr tune check` both call
//! [`evaluate_with`] (and [`super::greedy::greedy`]), so a model is held to its rows the same way
//! by the test suite and by the command. Nothing here prints: every measured value and timing
//! goes to the caller's [`Log`] as an [`Event`], and what was measured, what failed and what was
//! not exercised come back in an [`Evaluation`].

use super::greedy::{greedy, GreedyOutcome};
use super::oracle::{Oracle, Query, RefCache};
use super::{lookup, Check, Entry, PathClass, RouteScore, RunScore, FIRST_ROWS, MIN_TOP1_SCORED};
use crate::seam::{ForcedLogits, ForcedOn, PrefillHead, Routing};
use crate::{EngineConfig, SeamModel};
use anyhow::{anyhow, Context, Result};

/// One thing an evaluation reports while it runs.
#[derive(Clone, Copy, Debug)]
pub enum Event<'a> {
    /// The values one run measured on one prompt and phase: a block of lines.
    Values(&'a str),
    /// How long a part of the evaluation of the case labelled `key` took.
    Time {
        key: &'a str,
        what: &'a str,
        seconds: f64,
    },
}

impl std::fmt::Display for Event<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Event::Values(lines) => f.write_str(lines),
            Event::Time { key, what, seconds } => write!(f, "time {key}: {what} {seconds:.1}s"),
        }
    }
}

/// Where an evaluation's [`Event`]s go: the tests print them, the command sends them to its
/// diagnostics by kind.
pub type Log<'a> = &'a mut dyn FnMut(Event<'_>);

/// The table's entry for `table` and `class`, or the error a caller-supplied pair with no row
/// gets.
pub(super) fn entry(table: &str, class: PathClass) -> Result<&'static Entry> {
    lookup(table, class)
        .ok_or_else(|| anyhow!("the tolerance table has no {class:?} row for {table}"))
}

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

/// A model to score and the token ids of its prompts.
pub struct Case {
    pub model: SeamModel,
    pub prompts: Vec<Vec<u32>>,
    /// Whether the model's prompt goes through the batched prefill. `false` only for a model
    /// known to have none (the synthetic MoE: float expert banks have no batched kernel), whose
    /// prompt positions are then not scored on logits ([`Reference::check_prefill`]).
    pub batched_prefill: bool,
    /// Whether the model's greedy decode chains several steps into one submit on a device that
    /// allows it. `false` for the synthetic models, whose token embedding is gathered on the
    /// host, so every step has host work between it and the next.
    pub chains: bool,
    /// The prompt the greedy check also generates from: a chat-templated one
    /// ([`super::greedy::chat_prompt`]), for a model with a chat template. `None` for the
    /// synthetic models, whose vocabulary spells nothing.
    pub chat_prompt: Option<Vec<u32>>,
}

impl Case {
    /// The case of a real model, as the table's rows were measured: [`PROMPTS`] tokenized, the
    /// chat prompt rendered through the model's own template, a batched prefill and a chained
    /// decode expected.
    pub fn of_model(model: SeamModel) -> Result<Self> {
        Ok(Self {
            prompts: real_prompts(&model)?,
            chat_prompt: Some(super::greedy::chat_prompt(&model)?),
            model,
            batched_prefill: true,
            chains: true,
        })
    }
}

/// [`PROMPTS`], tokenized by `model`.
pub fn real_prompts(model: &SeamModel) -> Result<Vec<Vec<u32>>> {
    PROMPTS
        .iter()
        .map(|t| model.encode(t).context("encode a gate prompt"))
        .collect()
}

/// The argmax of a logit row, the lowest index on an exact tie.
pub fn argmax(row: &[f32]) -> u32 {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b }) as u32
}

/// The CPU reference's greedy run on `prompt`: `n` tokens, the logits at every position. Never
/// through the reference cache.
pub fn forced_reference(model: &SeamModel, prompt: &[u32], n: usize) -> Result<ForcedLogits> {
    Query::greedy(prompt, n)
        .compute(model)
        .context("cpu reference run")
}

/// `reference`'s tokens, and on a routed-expert model its expert selections, forced through
/// `on`, its engine configuration adjusted by `edit`.
pub fn forced(
    model: &SeamModel,
    on: ForcedOn,
    edit: impl FnOnce(&mut EngineConfig),
    reference: &ForcedLogits,
) -> Result<ForcedLogits> {
    forced_with(model, on, edit, PrefillHead::All, reference)
}

/// [`forced`] scoring the prefill rows `head` names.
pub fn forced_with(
    model: &SeamModel,
    on: ForcedOn,
    edit: impl FnOnce(&mut EngineConfig),
    head: PrefillHead,
    reference: &ForcedLogits,
) -> Result<ForcedLogits> {
    forced_along(model, on, edit, head, reference, reference.routing.as_ref())
}

/// [`forced_with`] along `reference`'s tokens and the expert selections `route` holds (`None`:
/// every router's own). An error when the run could not be made, or when it did not score the
/// reference's positions (a model or configuration that batches its prompt differently).
pub fn forced_along(
    model: &SeamModel,
    on: ForcedOn,
    edit: impl FnOnce(&mut EngineConfig),
    head: PrefillHead,
    reference: &ForcedLogits,
    route: Option<&Routing>,
) -> Result<ForcedLogits> {
    let steps = reference.positions.len() - reference.prefill_rows;
    let p = reference.seq.len() - steps;
    let got = model
        .forced_logits(on, edit, &reference.seq[..p], steps, head, route, |s, _| {
            reference.seq[p + s]
        })
        .with_context(|| format!("forced run on {on:?}"))?;
    check_along(on, head, reference, got)
}

/// `got`, a run on `on` forced along `reference`'s tokens scoring the prefill rows `head` names,
/// or the error [`forced_along`] gives when it did not score the reference's positions.
fn check_along(
    on: ForcedOn,
    head: PrefillHead,
    reference: &ForcedLogits,
    got: ForcedLogits,
) -> Result<ForcedLogits> {
    // The two runs may be of models loaded apart (another configuration, a planted bug), so
    // these are answers about the caller's pair, not invariants.
    if head != PrefillHead::All {
        anyhow::ensure!(
            got.positions
                .iter()
                .all(|q| reference.positions.contains(q)),
            "{on:?} scored positions the reference has no row for"
        );
        return Ok(got);
    }
    anyhow::ensure!(
        got.seq == reference.seq,
        "{on:?} was not fed the reference's tokens"
    );
    anyhow::ensure!(
        got.positions == reference.positions,
        "{on:?} scored other positions than the reference ({:?} against {:?})",
        got.positions,
        reference.positions
    );
    Ok(got)
}

/// One comparison, split by where the rows came from: the first [`FIRST_ROWS`] positions (whose
/// hidden states carry a model's largest activations, and which every quantized path misses by
/// the most), the rest of the batched prefill's rows, and the decode steps' rows.
#[derive(Clone, Debug)]
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
/// reference's. The error is a row that cannot be scored (a NaN or infinite logit, named with
/// its position), or a pair of runs that are not of the same positions.
pub fn try_score(
    reference: &ForcedLogits,
    test: &ForcedLogits,
    near_steady: f64,
    near_first: f64,
) -> Result<Scored, String> {
    let v = reference.vocab;
    let decode = |run: &ForcedLogits| run.positions.get(run.prefill_rows..).map(<[usize]>::to_vec);
    if decode(test).is_none() || decode(test) != decode(reference) {
        return Err("the two runs' decode positions differ".to_string());
    }
    let whole = |run: &ForcedLogits| run.logits.len() == run.positions.len() * v;
    if test.vocab != v || !whole(test) || !whole(reference) {
        return Err("the two runs' logit rows are of different shapes".to_string());
    }
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
            .ok_or_else(|| format!("the reference has no row at position {pos}"))?;
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

/// `test`'s own expert selections against `reference`'s at the near tie `near_tie`
/// ([`RouteScore`]); `None` for a model with no router-selected layer. An error when only one of
/// the two runs carries a routing, or the two cannot be compared.
pub fn score_routing(
    reference: &ForcedLogits,
    test: &ForcedLogits,
    near_tie: f64,
) -> Result<Option<RouteScore>, String> {
    match (&reference.routing, &test.routing) {
        (None, None) => Ok(None),
        (Some(want), Some(got)) => {
            let mut s = RouteScore::new(near_tie);
            s.add(want, got)?;
            Ok(Some(s))
        }
        _ => Err("one run of a pair read its routing back and the other did not".to_string()),
    }
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

/// A named Vulkan configuration of the gate.
#[derive(Clone, Copy)]
pub struct GpuConfig {
    pub name: &'static str,
    /// The edit it makes to the engine configuration the model under test was loaded with.
    pub edit: fn(&mut EngineConfig),
    /// The KV format (K, V) of the CPU reference it is scored against; `None`: the reference's
    /// own default.
    pub reference_kv: Option<(&'static str, &'static str)>,
}

const fn config(name: &'static str, edit: fn(&mut EngineConfig)) -> GpuConfig {
    GpuConfig {
        name,
        edit,
        reference_kv: None,
    }
}

const fn kv_config(
    name: &'static str,
    edit: fn(&mut EngineConfig),
    kv: (&'static str, &'static str),
) -> GpuConfig {
    GpuConfig {
        name,
        edit,
        reference_kv: Some(kv),
    }
}

/// Every configuration a gate run can name. Each is an edit on top of the engine configuration
/// the model was loaded with, so a tuning override in that configuration is in effect in every
/// one of them; a setting the edit itself writes is not ([`overridden`] says which).
pub const GPU_CONFIGS: &[GpuConfig] = &[
    config("gpu", |_| {}),
    config("gpu-ub32", |c| c.device.ubatch = Some(SMALL_UBATCH)),
    config("gpu-float", vulkan_float),
    config("gpu-float-ub32", |c| {
        vulkan_float(c);
        c.device.ubatch = Some(SMALL_UBATCH);
    }),
    config("gpu-float-static", |c| {
        vulkan_float(c);
        c.kernels.vulkan.no_replay = true;
    }),
    config("gpu-float-nocoopmat", vulkan_float_nocoopmat),
    config("gpu-float-nocoopmat-ub32", |c| {
        vulkan_float_nocoopmat(c);
        c.device.ubatch = Some(SMALL_UBATCH);
    }),
    config("gpu-float-nocoopmat-static", |c| {
        vulkan_float_nocoopmat(c);
        c.kernels.vulkan.no_replay = true;
    }),
    kv_config(
        "gpu-kv-q8",
        |c| kv_types(c, "q8_0", "q8_0"),
        ("q8_0", "q8_0"),
    ),
    kv_config(
        "gpu-kv-q4v",
        |c| kv_types(c, "f16", "q4_0"),
        ("f16", "q4_0"),
    ),
    kv_config(
        "gpu-kv-turbo3v",
        |c| kv_types(c, "f16", "turbo3"),
        ("f16", "turbo3"),
    ),
    kv_config(
        "gpu-kv-q8-float",
        |c| {
            kv_types(c, "q8_0", "q8_0");
            vulkan_float(c);
        },
        ("q8_0", "q8_0"),
    ),
];

/// The [`GPU_CONFIGS`] entry named `name`, or the error a run naming no configuration gets.
pub fn gpu_config(name: &str) -> Result<&'static GpuConfig> {
    GPU_CONFIGS
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| anyhow!("no gate configuration is named {name:?}"))
}

/// A setting of the configuration under test that a gate configuration's edit wrote over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Overridden {
    /// The setting's config path, as `--set` spells it.
    pub path: String,
    /// What the configuration under test had it at.
    pub set: String,
    /// What the run ran with.
    pub ran: String,
}

/// The settings of `user` that `config`'s edit wrote over: every config path whose value in
/// `user` is not the shipped default and is not what the run ran with. Read by walking every
/// path of the configuration before and after the edit, so it follows the edits as they change.
/// A setting given at its default value is not listed: nothing distinguishes it from one not
/// given.
pub fn overridden(config: &GpuConfig, user: &EngineConfig) -> Vec<Overridden> {
    let default = EngineConfig::default();
    let mut ran = user.clone();
    (config.edit)(&mut ran);
    EngineConfig::all_paths()
        .into_iter()
        .filter_map(|path| {
            let set = user.get_path(&path)?;
            let after = ran.get_path(&path)?;
            (Some(&set) != default.get_path(&path).as_ref() && after != set).then_some(Overridden {
                path,
                set,
                ran: after,
            })
        })
        .collect()
}

use PathClass::{Float, FloatNoCoopmat, Int8, KvQ8};

/// A quantized model: the production kernels at both prefill chunk heights, and the same with
/// every int8-activation kernel off, once more on the static per-token decode path.
const QUANT_RUNS: &[(&str, PathClass)] = &[
    ("gpu", Int8),
    ("gpu-ub32", Int8),
    ("gpu-float", Float),
    ("gpu-float-ub32", Float),
    ("gpu-float-static", Float),
];

/// The int8-free runs once more with the cooperative-matrix tier off, for a model whose Float row
/// cannot be tight on the f16 operands of the cooperative-matrix prefill GEMMs (gemma-3): the
/// prefill GEMMs then take f32 operands on every device, at both chunk heights and on the static
/// per-token decode path.
const NOCOOPMAT_RUNS: &[(&str, PathClass)] = &[
    ("gpu-float-nocoopmat", FloatNoCoopmat),
    ("gpu-float-nocoopmat-ub32", FloatNoCoopmat),
    ("gpu-float-nocoopmat-static", FloatNoCoopmat),
];

/// A float-weight model: the production kernels at both prefill chunk heights, and the static
/// per-token decode path.
const FLOAT_RUNS: &[(&str, PathClass)] = &[
    ("gpu", Float),
    ("gpu-ub32", Float),
    ("gpu-float-static", Float),
];

/// A routed-expert model with float expert banks: the production kernels on the replayed
/// per-token decode, and on the static one.
const MOE_RUNS: &[(&str, PathClass)] = &[("gpu", Float), ("gpu-float-static", Float)];

/// The runs the gate holds the model named `table` to, each a [`GPU_CONFIGS`] name with the
/// class whose row it is scored against; `None` for a model with no row. Read off the rows the
/// table has for the model, so that every row is scored by some run and no run lacks a row
/// (`every_row_is_scored_by_its_models_runs`):
///
/// - an Int8 row marks a quantized model: its shipped kernels are the int8-activation ones, held
///   to that row, and the same model with them off is held to its Float row;
/// - otherwise the shipped kernels are float ones, held to the Float row, on the two decode
///   paths only when the row is a routed-expert model's (which takes no batched prefill);
/// - a FloatNoCoopmat row adds the runs with the cooperative-matrix tier off, and a KvQ8 row the
///   int8-free run on the q8 KV cache.
pub fn runs_for(table: &str) -> Option<Vec<(&'static str, PathClass)>> {
    let float = lookup(table, Float)?;
    let mut runs = if lookup(table, Int8).is_some() {
        QUANT_RUNS.to_vec()
    } else if float.route_near_tie.is_some() {
        MOE_RUNS.to_vec()
    } else {
        FLOAT_RUNS.to_vec()
    };
    if lookup(table, FloatNoCoopmat).is_some() {
        runs.extend_from_slice(NOCOOPMAT_RUNS);
    }
    if lookup(table, KvQ8).is_some() {
        runs.push(("gpu-kv-q8-float", KvQ8));
    }
    Some(runs)
}

/// What a model file says of itself, beside the name its rows were found by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileIdentity {
    /// `general.architecture`.
    pub architecture: Option<String>,
    /// `general.name`.
    pub name: Option<String>,
    /// Whether any tensor of the file is in a quantized format.
    pub quantized: bool,
    /// How many tensors the file holds, and their bytes.
    pub tensors: usize,
    pub tensor_bytes: u64,
}

impl FileIdentity {
    /// Read off a loaded model's file.
    pub fn of(model: &SeamModel) -> Self {
        use infr_core::WeightSource;
        let g = model.gguf();
        let text = |key: &str| g.metadata().str(key).map(str::to_string);
        FileIdentity {
            architecture: text("general.architecture"),
            name: text("general.name"),
            quantized: g.tensors().iter().any(|t| t.dtype.is_quant()),
            tensors: g.tensors().len(),
            tensor_bytes: g.tensors().iter().map(|t| t.nbytes as u64).sum(),
        }
    }
}

/// Hold a file to the one thing the rows of `table` say about the file they were measured on:
/// a model with an Int8 row has quantized weights, and one without has float weights. The rows
/// are keyed by file name alone, so this is all that stands between a renamed or re-made file
/// and another file's rows; it refuses the mismatch it can see and nothing else. `Ok` also for
/// a model with no row, which [`runs_for`] refuses.
pub fn check_file(table: &str, file: &FileIdentity) -> Result<(), String> {
    if lookup(table, Float).is_none() {
        return Ok(());
    }
    let expected = lookup(table, Int8).is_some();
    if expected == file.quantized {
        return Ok(());
    }
    let kind = |quantized| {
        if quantized {
            "quantized weights"
        } else {
            "float weights"
        }
    };
    Err(format!(
        "the rows of `{table}` were measured on a file with {}, and this file has {}: it is \
         another file under that name",
        kind(expected),
        kind(file.quantized)
    ))
}

/// A case with its CPU reference's greedy run on each prompt, computed once for every
/// [`evaluate_with`] that scores against it, and every other reference run of the case's model
/// an evaluation asks for ([`Reference::oracle`]).
pub struct Reference<'a> {
    pub case: &'a Case,
    pub runs: Vec<ForcedLogits>,
    oracle: Oracle<'a>,
}

impl<'a> Reference<'a> {
    /// The reference of `case`, computed, never through the reference cache.
    pub fn new(case: &'a Case) -> Result<Self> {
        Self::with_cache(case, None)
    }

    /// The reference of `case` through `cache`: the greedy runs, and every reference run an
    /// evaluation asks of it later, are served from the cache where stored. Under
    /// [`super::oracle::Mode::On`] the shortest prompt's run among those served is recomputed
    /// and compared bit for bit, and a difference is the error.
    pub fn cached(case: &'a Case, cache: &'a RefCache) -> Result<Self> {
        Self::with_cache(case, Some(cache))
    }

    fn with_cache(case: &'a Case, cache: Option<&'a RefCache>) -> Result<Self> {
        let oracle = Oracle::new(&case.model, cache)?;
        let mut runs = Vec::with_capacity(case.prompts.len());
        let mut served = Vec::new();
        for (pi, ids) in case.prompts.iter().enumerate() {
            let (run, hit) = oracle.lookup(&Query::greedy(ids, N_DECODE))?;
            if hit {
                served.push(pi);
            }
            runs.push(run);
        }
        if let Some(&pi) = served.iter().min_by_key(|&&pi| case.prompts[pi].len()) {
            oracle.canary(&Query::greedy(&case.prompts[pi], N_DECODE), &runs[pi])?;
        }
        Ok(Self { case, runs, oracle })
    }

    /// Where every other reference run of the case's model goes: through the cache when the
    /// reference was made with one ([`Reference::cached`]).
    pub fn oracle(&self) -> &Oracle<'a> {
        &self.oracle
    }

    /// `run`'s tokens, and on a routed-expert model its expert selections, forced through the
    /// reference with the KV format pair `kv`: [`forced`] on the CPU reference, through the
    /// oracle.
    pub fn along_kv(&self, run: &ForcedLogits, kv: (&str, &str)) -> Result<ForcedLogits> {
        let on = ForcedOn::CpuReference;
        let got = self
            .oracle
            .run(&Query::along(
                run,
                PrefillHead::All,
                run.routing.as_ref(),
                Some(kv),
            ))
            .with_context(|| format!("forced run on {on:?}"))?;
        check_along(on, PrefillHead::All, run, got)
    }

    /// Hold the reference's prefill rows to what the case says of its model, adding to
    /// `failures`. A case marked as having no batched prefill ([`Case::batched_prefill`]) must
    /// have produced no prefill row on any prompt; [`evaluate_with`]'s count of scored prefill
    /// rows then reports zero of zero, which says nothing about the run, and that one failure is
    /// dropped, with a line added to `not_exercised` (the decode rows are scored, and a routed
    /// model's routing at every position). Any other case must have prefill rows on every
    /// prompt: a model that lost its batched prefill fails here by name.
    pub fn check_prefill(
        &self,
        label: &str,
        failures: &mut Vec<String>,
        not_exercised: &mut Vec<String>,
    ) {
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
        not_exercised.push(format!(
            "{label}: the model takes no batched prefill, so no prefill row is scored; its \
             decode rows are, and a routed model's routing at every position"
        ));
        failures.retain(|f| !f.contains(" prefill: only 0 of 0 positions"));
    }
}

/// What one run of an evaluation measured.
#[derive(Clone, Debug)]
pub struct RunOutcome {
    /// The [`GPU_CONFIGS`] name and the class whose row the run was held to.
    pub name: String,
    pub class: PathClass,
    /// That row.
    pub entry: &'static Entry,
    /// The settings of the configuration under test that the run's own edit wrote over, so that
    /// what ran is not what was set ([`overridden`]).
    pub overridden: Vec<Overridden>,
    /// Each prompt's score, in prompt order; `None` where a row could not be scored (a failure
    /// then names it).
    pub prompts: Vec<Option<Scored>>,
    /// The batched prefill's rows past the first, and the decode rows, each merged over every
    /// prompt: what the count of positions above the near-tie margin is taken over. `None`
    /// where no prompt's run scored.
    pub prefill: Option<RunScore>,
    pub decode: Option<RunScore>,
    /// A routed-expert model's routing comparison, merged over every prompt.
    pub routing: Option<RouteScore>,
}

impl RunOutcome {
    /// Every limit the run was held to under its row, as `(phase, check)`, each
    /// with the worst value over the prompts: per phase (`first`, `prefill`, `decode`) the
    /// row's [`super::Tolerance::checks`], then for the two steady phases the count of positions
    /// above the near-tie margin (where the phase scored any), then a routed-expert model's
    /// routing. A check here fails exactly when [`evaluate_with`] recorded a failure for it; a
    /// row that could not be scored is a failure with no line here.
    ///
    /// # Panics
    ///
    /// If a score in `prompts` was not made at the row's near-tie margins, which
    /// [`evaluate_with`] makes them at.
    pub fn checks(&self) -> Vec<(&'static str, Check)> {
        let entry = self.entry;
        let mut out = Vec::new();
        for phase in ["first", "prefill", "decode"] {
            let tol = if phase == "first" {
                &entry.first
            } else {
                &entry.steady
            };
            let mut worst: Option<[Check; 6]> = None;
            for scored in self.prompts.iter().flatten() {
                let ps = match phase {
                    "first" => &scored.first,
                    "prefill" => &scored.prefill,
                    _ => &scored.decode,
                };
                if ps.positions == 0 {
                    continue;
                }
                let checks = tol.checks(ps);
                worst = Some(match worst {
                    Some(w) => std::array::from_fn(|i| w[i].worse(checks[i])),
                    None => checks,
                });
            }
            out.extend(worst.into_iter().flatten().map(|c| (phase, c)));
            let merged = match phase {
                "prefill" => &self.prefill,
                "decode" => &self.decode,
                _ => &None,
            };
            if let Some(m) = merged.as_ref().filter(|m| m.positions > 0) {
                out.push((
                    phase,
                    Check::at_least("top1_scored", m.top1_scored as f64, MIN_TOP1_SCORED as f64),
                ));
            }
        }
        if let Some(r) = &self.routing {
            out.push((
                "routing",
                Check::at_most("route_misses", r.misses.len() as f64, 0.0),
            ));
            out.push((
                "routing",
                Check::at_least("route_scored", r.scored as f64, MIN_TOP1_SCORED as f64),
            ));
        }
        out
    }
}

/// What [`evaluate_with`] found: per run what was measured, and every limit that was crossed,
/// described. The evaluation passes when `failures` is empty.
#[derive(Clone, Debug, Default)]
pub struct Evaluation {
    pub runs: Vec<RunOutcome>,
    pub failures: Vec<String>,
    /// What the evaluation could not exercise on this model or device, described: a pass says
    /// nothing about these.
    pub not_exercised: Vec<String>,
}

/// What an evaluation compares: a case's CPU reference and the case driven on a backend, both
/// of the model whose rows are `table`'s. `key` labels the logged values. The two cases differ
/// where the reference is loaded apart from the configuration under test, and in the red-path
/// tests, where `test` carries a planted bug.
#[derive(Clone, Copy)]
pub struct Comparison<'a> {
    pub key: &'a str,
    pub table: &'a str,
    pub reference: &'a Reference<'a>,
    pub test: &'a Case,
    pub on: ForcedOn,
}

/// Score each run of `cmp.test` (driven on `cmp.on`) against the CPU reference's runs, along
/// the reference's own greedy tokens, against the table's entry for `cmp.table` and each run's
/// class. `runs` names [`GPU_CONFIGS`] entries. `seen(prompt, run name, run, score)` is called
/// after each run: the score is `None` when a row could not be scored, which is then a failure
/// naming it.
///
/// Per prompt, the first [`FIRST_ROWS`] positions must be within the entry's `first` bound, and
/// the batched prefill's other rows and the decode rows each within its `steady` bound; over a
/// case, each of those two phases must score at least [`MIN_TOP1_SCORED`] positions above the
/// near-tie margin.
///
/// A routed-expert model is forced along the reference's expert selections too, and its own
/// selections are held to the entry's `route_near_tie` at every position and layer; over a case,
/// at least [`MIN_TOP1_SCORED`] of the reference's selections must clear that margin. A row
/// with a routing near tie and a run without a routing, or the reverse, is a failure.
///
/// An error is a run that could not be made (the device, the model), a run naming no
/// configuration, or a class the table has no row of for the model; never a crossed limit.
pub fn evaluate_with(
    cmp: &Comparison,
    runs: &[(&str, PathClass)],
    mut seen: impl FnMut(usize, &str, &ForcedLogits, Option<&Scored>),
    log: Log,
) -> Result<Evaluation> {
    let Comparison {
        key,
        table,
        reference,
        test,
        on,
    } = *cmp;
    let mut failures = Vec::new();
    let mut outcomes = Vec::with_capacity(runs.len());
    // Per run: an Int8-class run that went without the packed int8 dot on some prompt.
    let mut int8_off = vec![false; runs.len()];
    for &(name, class) in runs {
        outcomes.push(RunOutcome {
            name: name.to_string(),
            class,
            entry: entry(table, class)?,
            overridden: overridden(gpu_config(name)?, test.model.engine_config()),
            prompts: Vec::new(),
            prefill: None,
            decode: None,
            routing: None,
        });
    }
    for (pi, (ids, r)) in reference
        .case
        .prompts
        .iter()
        .zip(&reference.runs)
        .enumerate()
    {
        for (ri, &(name, class)) in runs.iter().enumerate() {
            let &GpuConfig {
                edit, reference_kv, ..
            } = gpu_config(name)?;
            let entry = outcomes[ri].entry;
            let reference_run = match reference_kv {
                None => r.clone(),
                Some(kv) => reference.along_kv(r, kv)?,
            };
            let g = forced_with(&test.model, on, edit, PrefillHead::All, r)?;
            int8_off[ri] |= on == ForcedOn::Vulkan && class == Int8 && !g.i8_dot;
            match (entry.route_near_tie, g.routing.is_some()) {
                (Some(near), true) => {
                    let rs = score_routing(r, &g, near)
                        .map_err(|e| anyhow!("{name} prompt {pi} routing: {e}"))?
                        .ok_or_else(|| {
                            anyhow!("{name} prompt {pi} routing: the reference read back none")
                        })?;
                    log(Event::Values(&format!(
                        "{key} {name} ({class:?}) prompt {pi} ({} tokens):\n{}",
                        ids.len(),
                        rs.report()
                    )));
                    for v in rs.violations() {
                        failures.push(format!("{name} prompt {pi} routing: {v}"));
                    }
                    match &mut outcomes[ri].routing {
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
                    outcomes[ri].prompts.push(None);
                    continue;
                }
            };
            seen(pi, name, &g, Some(&s));
            for (phase, ps) in s.phases() {
                if ps.positions == 0 {
                    continue;
                }
                log(Event::Values(&format!(
                    "{key} {name} ({class:?}) prompt {pi} ({} tokens), {phase}:\n{}",
                    ids.len(),
                    ps.report()
                )));
                let tol = if phase == "first" {
                    &entry.first
                } else {
                    &entry.steady
                };
                for v in tol.violations(ps) {
                    failures.push(format!("{name} prompt {pi} {phase}: {v}"));
                }
            }
            let o = &mut outcomes[ri];
            for (slot, ps) in [(&mut o.prefill, &s.prefill), (&mut o.decode, &s.decode)] {
                match slot {
                    Some(m) => m.merge(ps),
                    None => *slot = Some(ps.clone()),
                }
            }
            o.prompts.push(Some(s));
        }
    }
    for o in &outcomes {
        if let Some(rs) = o.routing.as_ref().filter(|rs| rs.scored < MIN_TOP1_SCORED) {
            failures.push(format!(
                "{} routing: only {} of {} selections clear the {:.3e} near tie (at least \
                 {MIN_TOP1_SCORED} needed, or the routing check is vacuous)",
                o.name, rs.scored, rs.calls, rs.near_tie
            ));
        }
    }
    for o in &outcomes {
        for (phase, ps) in [("prefill", &o.prefill), ("decode", &o.decode)] {
            // No score: every prompt's row of this run failed to score, each a failure above.
            let Some(ps) = ps else { continue };
            if ps.top1_scored < MIN_TOP1_SCORED {
                failures.push(format!(
                    "{} {phase}: only {} of {} positions clear the {}-nat near-tie margin \
                     (at least {MIN_TOP1_SCORED} needed, or the top-1 check is vacuous)",
                    o.name, ps.top1_scored, ps.positions, ps.near_tie
                ));
            }
        }
    }
    // A device or setting that turns the packed int8 dot off runs such a configuration on the
    // float kernels, so its Int8 row is met without one int8 kernel being scored.
    let not_exercised = outcomes
        .iter()
        .zip(&int8_off)
        .filter(|&(_, &off)| off)
        .map(|(o, _)| {
            format!(
                "{}: ran without the packed int8 dot (`caps.i8_dot` off: `kernels.vulkan.i8_dot`, \
                 the `device.i8_dot` tuning parameter or the device), so its Int8 row scored \
                 the float kernels, not the int8 ones",
                o.name
            )
        })
        .collect();
    Ok(Evaluation {
        runs: outcomes,
        failures,
        not_exercised,
    })
}

/// What [`check`] found: the teacher-forced evaluation and the greedy check.
#[derive(Clone, Debug, Default)]
pub struct Verdict {
    pub forced: Evaluation,
    pub greedy: GreedyOutcome,
}

impl Verdict {
    /// Everything that failed, described: the forced runs' crossed limits, then the greedy
    /// check's. Empty when the device is within the model's rows.
    pub fn failures(&self) -> Vec<String> {
        let mut f = self.forced.failures.clone();
        f.extend(self.greedy.failures.iter().cloned());
        f
    }

    /// What neither half could exercise on this model and device, described. A pass says
    /// nothing about these, so whoever shows the verdict shows them with it.
    pub fn not_exercised(&self) -> Vec<String> {
        let mut n = self.forced.not_exercised.clone();
        n.extend(self.greedy.not_exercised.iter().cloned());
        n
    }
}

/// The gate: on every prompt, each of `runs` on the Vulkan device of `test`'s engine
/// configuration must be within the tolerance table's entry for (`table`, its class) of the CPU
/// reference of `reference` ([`evaluate_with`]), the reference's prefill rows must be what the
/// case says of its model ([`Reference::check_prefill`]), and the greedy check ([`greedy`]) must
/// hold the chained and the replayed decode, which no forced run exercises, to the per-token
/// path, and the greedy tokens to the reference. `key` labels the logged events: every measured
/// value, and the time each half took. What a pass does not cover on this model and device (a
/// device that does not chain, a model with no batched prefill) is in the verdict's
/// [`not_exercised`](Verdict::not_exercised).
///
/// `reference` and `test` are cases of the same model file. They are two so that the reference
/// can be loaded apart from the configuration under test: the CPU reference runs under the
/// engine configuration its case's model was loaded with, and a setting that changes its
/// arithmetic (a KV format) would otherwise move the reference along with the device. The tests
/// pass one case for both; their configuration differs from the default only in Vulkan knobs.
///
/// The `gpu_seam_tolerance_*` tests and `infr tune check` both call this, with
/// [`runs_for`]`(table)` as `runs`. An error is a run that could not be made, or `runs` naming
/// a configuration or a class the table has no row of; never a failed check.
pub fn check(
    key: &str,
    table: &str,
    reference: &Case,
    test: &Case,
    runs: &[(&str, PathClass)],
    log: Log,
) -> Result<Verdict> {
    check_cached(key, table, reference, test, runs, None, log)
}

/// [`check`] with the CPU reference served through `cache` when one is given
/// ([`Reference::cached`]), else computed. Either way the seconds the reference spent hashing the
/// model's weights for the key, computing and reading from the cache are logged as three
/// [`Event::Time`]s once the greedy check is done.
pub fn check_cached(
    key: &str,
    table: &str,
    reference: &Case,
    test: &Case,
    runs: &[(&str, PathClass)],
    cache: Option<&RefCache>,
    log: Log,
) -> Result<Verdict> {
    let t0 = std::time::Instant::now();
    let reference = Reference::with_cache(reference, cache)?;
    let cmp = Comparison {
        key,
        table,
        reference: &reference,
        test,
        on: ForcedOn::Vulkan,
    };
    let mut forced = evaluate_with(&cmp, runs, |_, _, _, _| {}, &mut *log)?;
    let Evaluation {
        failures,
        not_exercised,
        ..
    } = &mut forced;
    reference.check_prefill(key, failures, not_exercised);
    log(Event::Time {
        key,
        what: "the teacher-forced runs",
        seconds: t0.elapsed().as_secs_f64(),
    });
    let t0 = std::time::Instant::now();
    let greedy = greedy(key, table, &reference, test, runs, &mut *log)?;
    log(Event::Time {
        key,
        what: "the greedy check",
        seconds: t0.elapsed().as_secs_f64(),
    });
    let spent = reference.oracle().seconds();
    for (what, seconds) in [
        ("the CPU reference's weights hashed", spent.hashed),
        ("the CPU reference computed", spent.computed),
        ("the CPU reference read from the cache", spent.read),
    ] {
        log(Event::Time { key, what, seconds });
    }
    Ok(Verdict { forced, greedy })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerance::TABLE;

    /// The run lists are read off the table's rows: every row of a model is the row of one of
    /// its runs, and every run names a configuration and has a row.
    #[test]
    fn every_row_is_scored_by_its_models_runs() {
        for e in TABLE {
            let runs = runs_for(e.model).expect("a model with a row has runs");
            assert!(
                runs.iter().any(|r| r.1 == e.class),
                "no run of {} is scored against its {:?} row: {runs:?}",
                e.model,
                e.class
            );
            for (name, class) in runs {
                assert!(gpu_config(name).is_ok(), "{name} names no configuration");
                assert!(
                    lookup(e.model, class).is_some(),
                    "{}: the run {name} has no {class:?} row",
                    e.model
                );
            }
        }
        assert_eq!(runs_for("not-in-the-table.gguf"), None);
    }

    /// The lists themselves, name and class of every run, for every model the gate's tests were
    /// written against by hand before the lists were derived.
    #[test]
    fn run_lists_of_the_gated_models() {
        let quant = vec![
            ("gpu", Int8),
            ("gpu-ub32", Int8),
            ("gpu-float", Float),
            ("gpu-float-ub32", Float),
            ("gpu-float-static", Float),
        ];
        let float = vec![
            ("gpu", Float),
            ("gpu-ub32", Float),
            ("gpu-float-static", Float),
        ];
        let mut qwen3 = quant.clone();
        qwen3.push(("gpu-kv-q8-float", KvQ8));
        let mut gemma = quant.clone();
        gemma.extend([
            ("gpu-float-nocoopmat", FloatNoCoopmat),
            ("gpu-float-nocoopmat-ub32", FloatNoCoopmat),
            ("gpu-float-nocoopmat-static", FloatNoCoopmat),
        ]);
        let moe = vec![("gpu", Float), ("gpu-float-static", Float)];
        let expected = [
            ("Qwen3-0.6B-Q4_K_M.gguf", &qwen3),
            ("gemma-3-1b-it-Q4_K_M.gguf", &gemma),
            ("Qwen3.5-0.8B-Q4_K_M.gguf", &quant),
            ("LFM3-1.7B-Q4_K_M.gguf", &quant),
            ("ERNIE-4.5-0.3B-PT-Q4_K_M.gguf", &quant),
            ("LFM3-1.7B-F16.gguf", &float),
            ("ERNIE-4.5-0.3B-PT-F16.gguf", &float),
            ("synthetic-f32", &float),
            ("synthetic-f16", &float),
            ("synthetic-moe", &moe),
        ];
        for (model, runs) in expected {
            assert_eq!(runs_for(model).as_ref(), Some(runs), "{model}");
        }
        // Every model of the table is one of the above: a new row's runs get pinned here too.
        for e in TABLE {
            assert!(
                expected.iter().any(|(m, _)| *m == e.model),
                "{} has rows and no pinned run list",
                e.model
            );
        }
    }

    /// A file is held to what its name's rows say of the weights: quantized for a model with an
    /// Int8 row, float for one without.
    #[test]
    fn a_file_of_the_other_kind_is_refused() {
        let file = |quantized| FileIdentity {
            architecture: None,
            name: None,
            quantized,
            tensors: 0,
            tensor_bytes: 0,
        };
        assert!(check_file("Qwen3-0.6B-Q4_K_M.gguf", &file(true)).is_ok());
        let e = check_file("Qwen3-0.6B-Q4_K_M.gguf", &file(false)).expect_err("a float file");
        assert!(e.contains("another file under that name"), "{e}");
        assert!(check_file("LFM3-1.7B-F16.gguf", &file(false)).is_ok());
        assert!(check_file("LFM3-1.7B-F16.gguf", &file(true)).is_err());
        assert!(check_file("not-in-the-table.gguf", &file(true)).is_ok());
    }

    /// A run that names no configuration, and a class the model has no row of, are errors, not
    /// panics.
    #[test]
    fn unknown_names_are_errors() {
        assert!(gpu_config("gpu").is_ok());
        assert!(gpu_config("gpu-nope").is_err());
        assert!(entry("synthetic-f32", Float).is_ok());
        assert!(entry("synthetic-f32", Int8).is_err());
        assert!(entry("not-in-the-table.gguf", Float).is_err());
    }

    /// What a configuration's edit writes over a setting is reported, with both values; a
    /// setting the edit leaves alone, one at its default, and one the edit sets to the same
    /// value are not.
    #[test]
    fn overridden_settings_are_the_ones_the_edit_wrote_over() {
        let over = |name: &str, user: &EngineConfig| -> Vec<(String, String, String)> {
            overridden(gpu_config(name).expect("a configuration"), user)
                .into_iter()
                .map(|o| (o.path, o.set, o.ran))
                .collect()
        };
        let default = EngineConfig::default();
        for c in GPU_CONFIGS {
            assert_eq!(overridden(c, &default), [], "{}", c.name);
        }
        let mut user = EngineConfig::default();
        user.device.ubatch = Some(77);
        kv_types(&mut user, "f16", "q4_0");
        user.kernels.vulkan.no_replay = true;
        assert_eq!(over("gpu", &user), []);
        assert_eq!(
            over("gpu-ub32", &user),
            [(
                "device.ubatch".to_string(),
                "77".to_string(),
                SMALL_UBATCH.to_string()
            )]
        );
        // The static run sets `no_replay` to what the user already has.
        assert_eq!(over("gpu-float-static", &user), []);
        // Both halves of the KV pair the user gave are replaced by the q8 pair.
        let kv = over("gpu-kv-q8-float", &user);
        let paths: Vec<&str> = kv.iter().map(|o| o.0.as_str()).collect();
        assert_eq!(paths, ["kv.type_k", "kv.type_v"], "{kv:?}");
        assert!(kv.iter().all(|o| o.1 != o.2), "{kv:?}");
        // An int8 switch the user turned off is left as set by the int8-free runs, which turn
        // it off too.
        let mut user = EngineConfig::default();
        user.kernels.vulkan.mmq = false;
        assert_eq!(over("gpu-float", &user), []);
    }
}
