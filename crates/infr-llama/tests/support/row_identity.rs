//! Row-count identity of a whole forward on the CPU: the same token stream gives the same bits
//! whether its prompt is prefilled in one chunk, in several, or one row at a time.
//!
//! [`assert_row_count_identity`] generates greedily from one prompt at several prefill chunk
//! heights (`device.ubatch`; height 1 is a teacher-forced decode of the whole prompt) with the
//! CPU op-hash tap on (`debug.cpu_op_hash`), and compares, between every height and the first:
//! the generated ids, and every tensor the last `n_gen` graphs wrote, bit for bit (their hashes,
//! labels and sizes). Those graphs are each one row (the prompt's last token always runs as a
//! one-row graph, then each decode step), so they are the same graphs at every height; what
//! differs is how the KV caches and recurrent state they read were filled. The first of them
//! computes the logits for the prompt's last position, the rest the decode positions', so the
//! final-position logits are among the compared tensors.
//!
//! Shared by `synthetic_lfm3.rs` and `cpu_backend.rs` through `#[path]`.

use std::path::Path;
use std::sync::Arc;

/// One generation: the ids, and per graph the write lines of the tap (`op<i> W t<id> n=<n>
/// <hash> <label>`, the backend and graph numbers stripped).
struct Run {
    ids: Vec<u32>,
    graphs: Vec<Vec<String>>,
}

fn run(
    load: &dyn Fn(Arc<infr_llama::EngineConfig>) -> infr_llama::SeamModel,
    base: &infr_llama::EngineConfig,
    prompt: &[u32],
    n_gen: usize,
    ubatch: usize,
    tap: &Path,
) -> Run {
    let _ = std::fs::remove_file(tap);
    let mut cfg = base.clone();
    cfg.device.ubatch = Some(ubatch);
    cfg.device.ubatch_specified = true;
    cfg.debug.cpu_op_hash = Some(tap.to_path_buf());
    let model = load(Arc::new(cfg));
    let ids = model
        .generate_cpu_ids(prompt, n_gen, |_| {})
        .expect("cpu generate");
    let text = std::fs::read_to_string(tap).expect("op-hash tap file");
    std::fs::remove_file(tap).expect("remove the tap file");
    let mut graphs: Vec<(String, Vec<String>)> = Vec::new();
    for line in text.lines() {
        let mut f = line.splitn(3, ' ');
        let (_backend, graph, rest) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
        if !rest.split(' ').nth(1).is_some_and(|t| t == "W") {
            continue;
        }
        match graphs.last_mut() {
            Some((g, lines)) if g == graph => lines.push(rest.to_string()),
            _ => graphs.push((graph.to_string(), vec![rest.to_string()])),
        }
    }
    Run {
        ids,
        graphs: graphs.into_iter().map(|(_, l)| l).collect(),
    }
}

/// Generate `n_gen` tokens after `prompt` at each prefill chunk height of `ubatches` and assert
/// every height gives the first one's ids and the bits of every tensor the last `n_gen` graphs
/// wrote. `load` builds the model on a configuration (`base` with the height and the tap set).
pub fn assert_row_count_identity(
    what: &str,
    load: &dyn Fn(Arc<infr_llama::EngineConfig>) -> infr_llama::SeamModel,
    base: &infr_llama::EngineConfig,
    prompt: &[u32],
    n_gen: usize,
    ubatches: &[usize],
) {
    let tap = std::env::temp_dir().join(format!(
        "infr-row-identity-{}-{}.txt",
        what.replace(|c: char| !c.is_ascii_alphanumeric(), "-"),
        std::process::id()
    ));
    let runs: Vec<(usize, Run)> = ubatches
        .iter()
        .map(|&u| (u, run(load, base, prompt, n_gen, u, &tap)))
        .collect();
    let (u0, first) = &runs[0];
    assert_eq!(
        first.ids.len(),
        n_gen,
        "{what}: generated {} ids",
        first.ids.len()
    );
    assert!(
        first.graphs.len() > n_gen,
        "{what}: the tap recorded {} graphs, no prefill before the last {n_gen}",
        first.graphs.len()
    );
    let tail = |r: &Run| r.graphs[r.graphs.len() - n_gen..].to_vec();
    let want = tail(first);
    let written: usize = want.iter().map(Vec::len).sum();
    assert!(written > n_gen, "{what}: the compared graphs wrote nothing");
    for (u, r) in &runs[1..] {
        assert_eq!(
            r.ids, first.ids,
            "{what}: ids at ubatch {u} differ from ubatch {u0}"
        );
        assert!(
            r.graphs.len() > n_gen,
            "{what}: ubatch {u} recorded {} graphs",
            r.graphs.len()
        );
        for (g, (got, want)) in tail(r).iter().zip(&want).enumerate() {
            let diff = got.iter().zip(want).position(|(a, b)| a != b);
            assert!(
                diff.is_none() && got.len() == want.len(),
                "{what}: graph {g} of the last {n_gen} at ubatch {u} differs from ubatch {u0}: \
                 {:?} against {:?}",
                diff.map(|i| &got[i]),
                diff.map(|i| &want[i]),
            );
        }
    }
    println!(
        "{what}: ubatch {ubatches:?} give the same {n_gen} ids and the same bits in the {written} \
         tensors the last {n_gen} graphs wrote"
    );
}
