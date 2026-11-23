//! The greedy check shown to fail (`docs/gpu-tolerance.md`, "The greedy check"): bugs in how a
//! decode recording is driven, which every teacher-forced run is blind to because it executes
//! each recording once, and what the check's token rule sees of a bug in the model.

use super::{open_clean, open_planted, run_case, Bug, QWEN3};
use crate::forced;
use infr_llama::seam::ForcedOn;
use infr_llama::tolerance::PathClass::{self, Float, Int8};

/// The model and configurations every demonstration here runs: Qwen3-0.6B Q4_K_M on the shipped
/// int8 kernels and on the int8-free ones.
const RUNS: &[(&str, PathClass)] = &[("gpu", Int8), ("gpu-float", Float)];

/// What a greedy-check failure says when a generation chose another token than the per-token
/// path, and when it left other KV behind.
const CHOSE: &str = "generation chose token";
const LEFT: &str = "left other KV behind";

/// The failures of the greedy check that say a generation parted from the per-token path: other
/// tokens, or other KV left behind.
fn path_failures(failures: &[String]) -> Vec<&str> {
    failures
        .iter()
        .filter(|f| f.contains(CHOSE) || f.contains(LEFT))
        .map(String::as_str)
        .collect()
}

/// The failures of the greedy check that say the greedy tokens are not the reference's.
fn token_failures(failures: &[String]) -> Vec<&str> {
    failures
        .iter()
        .filter(|f| f.contains("greedy token differs"))
        .map(String::as_str)
        .collect()
}

/// The generations the greedy check compares with the per-token path on this device: the chained
/// one only where the device chains.
fn generations() -> Vec<&'static str> {
    let mut g = vec!["replayed"];
    if forced::greedy::device_chains() {
        g.push("chained");
    }
    g
}

/// The prompts the greedy check labels its failures with, for a case of a real model.
fn prompt_labels(case: &forced::Case) -> Vec<String> {
    (0..case.prompts.len())
        .map(|pi| format!("prompt {pi}"))
        .chain(case.chat_prompt.iter().map(|_| "chat prompt".to_string()))
        .collect()
}

/// Plant `bug`, a bug in how a decode recording is driven, in Qwen3-0.6B and check both halves of
/// what the doc says of it. Every teacher-forced run must be the clean model's bit for bit and
/// pass the gate (a forced run executes each recording once). The greedy check must pass on the
/// clean model and, on the planted one, fail on every prompt and configuration for every
/// generation this device compares, each failure naming that generation and saying `says`; no
/// failure may say `never`.
fn recording_bug_fails_the_greedy_check(bug: Bug, says: &str, never: &str) {
    let t = &QWEN3;
    let Some(clean) = open_clean(t) else {
        return;
    };
    let Some(device) = forced::device_for(t.key, 2 * clean.bytes) else {
        return;
    };
    let _guard = forced::watchdog("a recording bug's red path", forced::GATE_TIMEOUT);
    let reference = forced::reference(&clean);
    let on = ForcedOn::Vulkan;
    let control = run_case("clean control", t.table, &reference, &clean, on, RUNS);
    assert!(
        control.failures.is_empty(),
        "the clean {} fails the gate on {device}:\n  {}",
        t.key,
        control.failures.join("\n  ")
    );
    let green = forced::greedy(t.key, t.table, &reference, &clean, RUNS);
    assert!(
        green.is_empty(),
        "the clean {} fails the greedy check on {device}:\n  {}",
        t.key,
        green.join("\n  ")
    );
    let test = open_planted(t, bug);
    let o = run_case(bug.label(), t.table, &reference, &test, on, RUNS);
    assert_eq!(
        o.digests,
        control.digests,
        "{} reached a teacher-forced run, which executes each recording once",
        bug.label()
    );
    assert!(
        o.failures.is_empty(),
        "bit-identical runs failed the gate: {:?}",
        o.failures
    );
    let red = forced::greedy(t.key, t.table, &reference, &test, RUNS);
    println!(
        "PLANTED model={} device={device} bug={bug:?} greedy:\n  {}",
        t.key,
        red.join("\n  ")
    );
    let unwanted: Vec<&String> = red.iter().filter(|f| f.contains(never)).collect();
    assert!(
        unwanted.is_empty(),
        "{} is not a bug that shows as {never:?}: {unwanted:?}",
        bug.label()
    );
    let mut passed = Vec::new();
    for &(run, _) in RUNS {
        for label in prompt_labels(&clean) {
            for generation in generations() {
                let at = format!("{run} {label} greedy: ");
                let the = format!("the {generation} generation");
                let caught = red
                    .iter()
                    .any(|f| f.starts_with(&at) && f.contains(&the) && f.contains(says));
                if !caught {
                    passed.push(format!("{run}, {label}, the {generation} generation"));
                }
            }
        }
    }
    assert!(
        passed.is_empty(),
        "{} passed the greedy check ({says:?} not reported) on:\n  {}",
        bug.label(),
        passed.join("\n  ")
    );
}

/// The token half: with [`Bug::ReplayFeedsStaleToken`] planted, the chained generation (on a
/// device that chains) and the replayed one each choose another token than the per-token path.
#[test]
#[ignore = "needs Qwen3-0.6B Q4_K_M and a Vulkan GPU"]
fn gpu_planted_replay_feeds_stale_token() {
    recording_bug_fails_the_greedy_check(Bug::ReplayFeedsStaleToken, CHOSE, LEFT);
}

/// The KV half: with [`Bug::ReplayRewritesLastRow`] planted, every generation samples the
/// per-token path's tokens, so nothing is caught by a token; the row one step later must differ
/// in its bits, for the chained generation (on a device that chains) and for the replayed one.
#[test]
#[ignore = "needs Qwen3-0.6B Q4_K_M and a Vulkan GPU"]
fn gpu_planted_replay_rewrites_last_row() {
    recording_bug_fails_the_greedy_check(Bug::ReplayRewritesLastRow, LEFT, CHOSE);
}

/// The bugs [`gpu_planted_greedy_tokens`] plants: those that leave the replayed decode in place
/// (a bug baked at a position opts its graphs out of it, and the greedy check then fails on the
/// missing recording before it reads a token).
const GREEDY_BUGS: &[Bug] = &[
    Bug::RmsEpsScaled,
    Bug::FfnDownScaled,
    Bug::GateUpSwapped,
    Bug::AttnResidualDropped,
    Bug::RopePairing,
    Bug::NibbleMap,
];

/// The bugs among [`GREEDY_BUGS`] the greedy tokens must show in both classes: those that put
/// several tokens of a prompt past the near-tie margin. The others move one token or none
/// (`docs/gpu-tolerance.md`, "The greedy check").
const GREEDY_GROSS: &[Bug] = &[Bug::GateUpSwapped, Bug::RopePairing];

/// What the greedy check's token rule sees of a bug in the model, on Qwen3-0.6B Q4_K_M: the
/// planted copy generates greedily on the GPU and its tokens are held to the clean copy's CPU
/// reference fed the same tokens. A bug in the model is in every decode path alike, so no planted
/// run may part from the per-token path; each bug of [`GREEDY_GROSS`] must put a token past the
/// near-tie margin; for the others the verdict is printed and nothing more is asserted (tokens
/// see far less than the logit rows the gate scores).
#[test]
#[ignore = "needs Qwen3-0.6B Q4_K_M and a Vulkan GPU"]
fn gpu_planted_greedy_tokens() {
    let t = &QWEN3;
    let Some(clean) = open_clean(t) else {
        return;
    };
    let Some(device) = forced::device_for(t.key, 2 * clean.bytes) else {
        return;
    };
    let _guard = forced::watchdog("the greedy token demonstration", forced::GATE_TIMEOUT);
    let reference = forced::reference(&clean);
    let green = forced::greedy(t.key, t.table, &reference, &clean, RUNS);
    assert!(
        green.is_empty(),
        "the clean {} fails the greedy check on {device}:\n  {}",
        t.key,
        green.join("\n  ")
    );
    let mut wrong = Vec::new();
    for &bug in GREEDY_BUGS {
        let test = open_planted(t, bug);
        let red = forced::greedy(t.key, t.table, &reference, &test, RUNS);
        drop(test);
        for f in path_failures(&red) {
            wrong.push(format!("{}: {f}", bug.label()));
        }
        for &(run, class) in RUNS {
            let at = format!("{run} ");
            let caught: Vec<&str> = token_failures(&red)
                .into_iter()
                .filter(|f| f.starts_with(&at))
                .collect();
            println!(
                "PLANTED model={} device={device} bug={bug:?} run={run} class={class:?} \
                 greedy_tokens={} first={}",
                t.key,
                if caught.is_empty() {
                    "missed"
                } else {
                    "caught"
                },
                caught.first().copied().unwrap_or("-")
            );
            if caught.is_empty() && GREEDY_GROSS.contains(&bug) {
                wrong.push(format!(
                    "{} on {run} ({class:?}): every greedy token is within the near tie of the \
                     reference's",
                    bug.label()
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
