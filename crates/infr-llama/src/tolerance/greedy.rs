//! The greedy check: the production greedy decode on Vulkan, which no teacher-forced run
//! exercises, held to the per-token path those runs score, and its tokens to the CPU reference.
//!
//! A teacher-forced run builds, records and executes a fresh decode plan for every step, so it
//! never replays one recording at advancing positions and never chains several steps into one
//! submit (`docs/gpu-tolerance.md`). Here the same prompt is generated greedily on one device and
//! configuration by the production call (chained where the device chains), by the same call with
//! the chain off (one replay of a single recording per token), and by the teacher-forced driver
//! left to its own argmax. They must stop at the same step, choose the same tokens and leave the
//! same KV behind, bit for bit: one more per-token step after each generation, which attends over
//! every row it wrote, must give the same logits. That comparison is between runs of one device
//! and needs no tolerance. The tokens are then held to the CPU reference fed the same tokens, by
//! the table's top-1 rule ([`TokenScore`]).
//!
//! Like [`super::gate`], this is the one check the tests and `infr tune check` both run, and it
//! prints nothing itself: measured values go to the caller's [`Log`], and what it could not
//! compare on the device or model comes back in [`GreedyOutcome::not_exercised`].

use super::gate::{argmax, entry, gpu_config, kv_types, Case, Event, GpuConfig, Log, Reference};
use super::{Check, PathClass, TokenScore, MIN_TOP1_SCORED};
use crate::seam::{ForcedLogits, ForcedOn, GreedyTail, PrefillHead};
use crate::EngineConfig;
use anyhow::{Context, Result};

/// The configurations of a gate's runs the greedy check drives as well: one per set of decode
/// kernels a recording can hold (the shipped ones, the int8-free ones, and the q8 KV cache's).
/// The others differ from these in the prefill chunk height or the cooperative-matrix prefill
/// tier, which the teacher-forced runs score, or take the static decode, which records nothing.
pub const GREEDY_CONFIGS: &[&str] = &["gpu", "gpu-float", "gpu-kv-q8-float"];

/// Tokens each generation is asked for. One more than a whole number of chained submits at the
/// shipped `spec.decode_chain`, so that on a device that chains the call ends with a single
/// replayed step after its last chain: the hand-off from a chain to one replay inside a call,
/// which a multiple of the chain length never reaches ([`greedy`] fails a length that leaves no
/// such step). It is not the forced runs' `N_DECODE`, which the table's rows were measured at
/// and which this leaves alone.
pub const GREEDY_STEPS: usize = 33;

/// The user message of the chat prompt every real model also generates from, rendered through
/// its own chat template ([`chat_prompt`]).
const CHAT_MESSAGE: &str = "The capital of France is";

/// `CHAT_MESSAGE` rendered through `model`'s chat template and tokenized: a prompt whose
/// special tokens (the template's role and turn markers) go through the device's embedding
/// gather, which the plain-text prompts of the forced runs never feed.
pub fn chat_prompt(model: &crate::SeamModel) -> Result<Vec<u32>> {
    let rendered = model.render_chat(CHAT_MESSAGE).context("render chat")?;
    model.encode(&rendered).context("encode")
}

/// Whether the Vulkan device `cfg` selects packs several decode steps into one submit
/// (`Backend::max_decode_chain`): an integrated GPU, which has to keep each submit short, does
/// not, and its production greedy decode is one replay per token.
pub fn device_chains(cfg: &std::sync::Arc<EngineConfig>) -> Result<bool> {
    let vk = infr_vulkan::VulkanBackend::new_with(cfg.clone())
        .map_err(|e| anyhow::anyhow!("the gate's Vulkan device: {e}"))?;
    Ok(infr_core::backend::Backend::max_decode_chain(&vk) >= 2)
}

/// The most steps the runner puts in one chained submit, whatever `spec.decode_chain` asks.
const CHAIN_CAP: usize = 64;

/// How many of a generation's first `kept` tokens come from chained submits when it is asked for
/// `asked` at a chain of `chain` steps: the runner chains every submit of two steps or more and
/// takes a lone last step on its own.
fn chained_of(asked: usize, chain: usize, kept: usize) -> usize {
    let mut left = asked;
    while left > 0 {
        let n = chain.min(CHAIN_CAP).min(left);
        if n < 2 {
            break;
        }
        left -= n;
    }
    kept.min(asked - left)
}

/// Where two logit rows first differ in their bits, and at how many indices.
fn bit_diff(a: &[f32], b: &[f32]) -> Option<(usize, usize)> {
    assert_eq!(a.len(), b.len(), "rows of different vocabularies");
    let differs = |i: &usize| a[*i].to_bits() != b[*i].to_bits();
    let first = (0..a.len()).find(differs)?;
    Some((first, (0..a.len()).filter(differs).count()))
}

/// What the greedy check measured of one configuration: the production tokens held to the
/// reference by the top-1 rule, over every prompt. `tokens` is `None` for a routed-expert model,
/// whose tokens are not held to the reference's.
#[derive(Clone, Debug)]
pub struct GreedyRun {
    pub name: String,
    pub class: PathClass,
    pub tokens: Option<TokenScore>,
}

impl GreedyRun {
    /// The limits the configuration's greedy tokens were held to, each with what was measured:
    /// no counted token may differ from the reference's, and at least [`MIN_TOP1_SCORED`] steps
    /// must be counted. Empty for a routed-expert model. The bit-for-bit comparisons with the
    /// per-token path have no line: one that fails is in [`GreedyOutcome::failures`].
    pub fn checks(&self) -> Vec<Check> {
        let Some(t) = &self.tokens else {
            return Vec::new();
        };
        vec![
            Check::at_most("token_misses", t.misses.len() as f64, 0.0),
            Check::at_least("token_scored", t.scored as f64, MIN_TOP1_SCORED as f64),
        ]
    }
}

/// What [`greedy`] found: per configuration it drove what the token rule measured, and
/// everything that failed, described. The check passes when `failures` is empty.
#[derive(Clone, Debug, Default)]
pub struct GreedyOutcome {
    pub runs: Vec<GreedyRun>,
    pub failures: Vec<String>,
    /// What the check could not compare on this device or model, described (a device that does
    /// not chain, a routed-expert model's tokens): a pass says nothing about these.
    pub not_exercised: Vec<String>,
}

/// Run the greedy check on every prompt of `test`, its chat prompt included, in each of `runs`
/// that [`GREEDY_CONFIGS`] names, and return what failed, described. `reference` is the clean
/// model's (the two cases differ only where `test` carries a planted bug) and `table` its
/// tolerance row's name. No run among [`GREEDY_CONFIGS`] is itself a failure.
///
/// Per prompt and configuration:
/// - The teacher-forced driver, left to its own argmax for one step more than [`GREEDY_STEPS`],
///   says where a generation must stop: at its first end token, or at the length asked for. Each
///   generation must have sampled exactly that many tokens and report a natural stop exactly when
///   the driver's last of them is an end token, so one cut short, and one that ran past its end
///   token, both fail.
/// - Each generation ran the path it stands for: one decode recording; with the chain off no
///   chained token; on a device that chains, for a model [`Case::chains`] marks as chaining,
///   exactly the tokens the length and the chain imply (`chained_of`), and a full-length
///   generation must hold chained tokens and at least one that is not.
/// - Each generation's tokens are the driver's, and its tail row is the driver's row at that
///   position bit for bit.
///
/// Over a configuration, the production tokens are then held to the CPU reference fed the same
/// tokens, at the row's steady near-tie margin, with at least [`MIN_TOP1_SCORED`] steps above it.
/// A routed-expert model's tokens are not: left to its own routers a run legitimately parts from
/// the reference at a router near tie (`docs/gpu-tolerance.md`), which the token rule cannot tell
/// from a wrong token.
///
/// The device and the chain length are those of the engine configuration `test`'s model was
/// loaded with. An error is a run that could not be made, a run naming no configuration, or a
/// class the table has no row of for the model; never a failed check.
///
/// # Panics
///
/// If a run hands back fewer tokens or logit rows than it was asked for, or rows of another
/// vocabulary size than the model's: what `SeamModel::forced_logits` and
/// `SeamModel::greedy_tail_vulkan` guarantee of their results.
pub fn greedy(
    key: &str,
    table: &str,
    reference: &Reference,
    test: &Case,
    runs: &[(&str, PathClass)],
    log: Log,
) -> Result<GreedyOutcome> {
    let mut failures = Vec::new();
    let mut outcomes = Vec::new();
    let mut not_exercised = Vec::new();
    let ecfg = test.model.engine_config();
    let chains = device_chains(ecfg)?;
    if !chains {
        not_exercised.push(format!(
            "{key}: this device runs one decode step per submit; its greedy decode is one replay \
             per token, and no chained run is compared"
        ));
    }
    let chain = ecfg.spec.decode_chain;
    let cfg = test.model.config();
    let is_end = |id: u32| cfg.eos_ids.contains(&id) || id == cfg.eos;
    let prompts: Vec<(String, &Vec<u32>)> = test
        .prompts
        .iter()
        .enumerate()
        .map(|(pi, ids)| (format!("prompt {pi}"), ids))
        .chain(
            test.chat_prompt
                .iter()
                .map(|ids| ("chat prompt".into(), ids)),
        )
        .collect();
    // The reference's rows along a token sequence, kept for the configurations that chose the
    // same one: (prompt, the reference's KV format, the tokens) to the run.
    type Along = (usize, Option<(&'static str, &'static str)>, Vec<u32>);
    let mut along: Vec<(Along, ForcedLogits)> = Vec::new();
    let mut ran = 0;
    for &(name, class) in runs.iter().filter(|r| GREEDY_CONFIGS.contains(&r.0)) {
        ran += 1;
        let &GpuConfig {
            edit,
            reference_kv: kv,
            ..
        } = gpu_config(name)?;
        let near = entry(table, class)?.steady.near_tie;
        let mut tokens = TokenScore::new(near);
        let mut routed = false;
        for (pi, (label, ids)) in prompts.iter().enumerate() {
            let at = format!("{name} {label} greedy");
            let tail = |what: &str, chain: Option<usize>| -> Result<GreedyTail> {
                test.model
                    .greedy_tail_vulkan(
                        |c| {
                            edit(c);
                            if let Some(n) = chain {
                                c.spec.decode_chain = n;
                            }
                        },
                        ids,
                        GREEDY_STEPS,
                    )
                    .with_context(|| format!("{at}, {what}"))
            };
            let replayed = tail("one replay per token", Some(1))?;
            let chained = chains.then(|| tail("chained", None)).transpose()?;
            // One step more than the generations sample, so the driver has a row at the position
            // of each generation's tail row.
            let stepped = test
                .model
                .forced_logits(
                    ForcedOn::Vulkan,
                    edit,
                    ids,
                    GREEDY_STEPS + 1,
                    PrefillHead::Off,
                    None,
                    |_, row| argmax(row),
                )
                .with_context(|| format!("{at}, teacher-forced driver"))?;
            routed |= stepped.routing.is_some();
            let (plen, v) = (ids.len(), stepped.vocab);
            let decode_rows = &stepped.logits[stepped.prefill_rows * v..];
            // Where the per-token path says a generation stops.
            let own = &stepped.seq[plen..plen + GREEDY_STEPS];
            let end = own.iter().position(|&id| is_end(id));
            let want = end.map_or(GREEDY_STEPS, |s| s + 1);
            let own = &own[..want];

            if replayed.chained != 0 {
                failures.push(format!(
                    "{at}: with the chain off, {} tokens still came from chained submits",
                    replayed.chained
                ));
            }
            if let Some(c) = &chained {
                let implied = if test.chains {
                    chained_of(GREEDY_STEPS, chain, want)
                } else {
                    0
                };
                if c.chained != implied {
                    failures.push(format!(
                        "{at}: the production generation took {} of its {} tokens from chained \
                         submits, where a chain of {chain} over {GREEDY_STEPS} steps implies \
                         {implied} of {want}{}",
                        c.chained,
                        c.ids.len(),
                        if test.chains {
                            ""
                        } else {
                            " (the case's model is marked as one that does not chain)"
                        }
                    ));
                }
                if test.chains && want == GREEDY_STEPS && (implied == 0 || implied == want) {
                    failures.push(format!(
                        "{at}: a chain of {chain} over {GREEDY_STEPS} steps leaves {implied} \
                         chained and {} single steps, so the hand-off from a chain to one replay \
                         is not exercised",
                        want - implied
                    ));
                }
            }
            let named = [("chained", chained.as_ref()), ("replayed", Some(&replayed))];
            for (what, t) in named.iter().filter_map(|(w, t)| t.map(|t| (*w, t))) {
                if t.recordings != 1 {
                    failures.push(format!(
                        "{at}: the {what} generation made {} decode recordings, not one replayed \
                         for every step",
                        t.recordings
                    ));
                }
                // A token that differs is reported as that, also where it moved the stop.
                let shared = t.ids.len().min(want);
                if let Some(s) = (0..shared).find(|&s| t.ids[s] != own[s]) {
                    failures.push(format!(
                        "{at}: the {what} generation chose token {} at step {s} where the \
                         per-token path chose {} (the two agree on the {s} steps before it)",
                        t.ids[s], own[s]
                    ));
                    continue;
                }
                if (t.ids.len(), t.natural_stop) != (want, end.is_some()) {
                    failures.push(format!(
                        "{at}: the {what} generation sampled {} tokens{}, where the per-token \
                         path {} after {want}",
                        t.ids.len(),
                        if t.natural_stop {
                            " and stopped on an end token"
                        } else {
                            " and did not stop itself"
                        },
                        if end.is_some() {
                            "reaches its end token"
                        } else {
                            "is still generating"
                        },
                    ));
                    continue;
                }
                if let Some((first, n)) = bit_diff(&t.row, &decode_rows[want * v..(want + 1) * v]) {
                    failures.push(format!(
                        "{at}: after the {what} generation the next step's logits differ from \
                         the per-token path's at {n} of {v} tokens (first at token {first}): it \
                         left other KV behind"
                    ));
                }
            }
            log(Event::Values(&format!(
                "GREEDY {key} {name} {label} ({plen} tokens): {}; {} decode recording(s) for the \
                 per-token generation's {} tokens{}",
                match &chained {
                    Some(c) => format!("chained {} of {} tokens", c.chained, c.ids.len()),
                    None => "no chained run".to_string(),
                },
                replayed.recordings,
                replayed.ids.len(),
                if end.is_some() {
                    ", which end on an end token"
                } else {
                    ""
                },
            )));

            if routed {
                continue;
            }
            // The production path's tokens against the reference forced along them.
            let chosen = &chained.as_ref().unwrap_or(&replayed).ids;
            let m = chosen.len().min(GREEDY_STEPS);
            let seq = (pi, kv, chosen[..m].to_vec());
            let k = match along.iter().position(|(have, _)| *have == seq) {
                Some(k) => k,
                None => {
                    let run = reference
                        .case
                        .model
                        .forced_logits(
                            ForcedOn::CpuReference,
                            |c| {
                                if let Some((k, v)) = kv {
                                    kv_types(c, k, v);
                                }
                            },
                            ids,
                            m,
                            PrefillHead::Off,
                            None,
                            |s, _| chosen[s],
                        )
                        .with_context(|| format!("{at}, the reference along its tokens"))?;
                    along.push((seq, run));
                    along.len() - 1
                }
            };
            let run = &along[k].1;
            let rows = &run.logits[run.prefill_rows * v..];
            let mut one = TokenScore::new(near);
            for (s, &id) in chosen[..m].iter().enumerate() {
                if let Err(e) = one.add(plen - 1 + s, &rows[s * v..(s + 1) * v], id) {
                    failures.push(format!("{at}: position {}: {e}", plen - 1 + s));
                }
            }
            for viol in one.violations() {
                failures.push(format!("{at}: {viol}"));
            }
            tokens.merge(&one);
        }
        if routed {
            not_exercised.push(format!(
                "{key} {name}: a routed-expert model; its greedy tokens are not held to the \
                 reference's"
            ));
            outcomes.push(GreedyRun {
                name: name.to_string(),
                class,
                tokens: None,
            });
            continue;
        }
        log(Event::Values(&format!(
            "{key} {name} ({class:?}), greedy:\n{}",
            tokens.report()
        )));
        if tokens.scored < MIN_TOP1_SCORED {
            failures.push(format!(
                "{name} greedy: only {} of {} steps clear the {near}-nat near-tie margin (at \
                 least {MIN_TOP1_SCORED} needed, or the token check is vacuous)",
                tokens.scored,
                tokens.scored + tokens.near_ties
            ));
        }
        outcomes.push(GreedyRun {
            name: name.to_string(),
            class,
            tokens: Some(tokens),
        });
    }
    if ran == 0 {
        failures.push(format!(
            "greedy: none of the case's runs is one of {GREEDY_CONFIGS:?}, so the chained and \
             the replayed decode were not compared"
        ));
    }
    Ok(GreedyOutcome {
        runs: outcomes,
        failures,
        not_exercised,
    })
}
