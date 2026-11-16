//! The synthetic V4 file prefilled through a host weight arena smaller than its expert banks
//! (`paging.dram`): what a prompt of several chunks reads from the file in each prefill order.

use super::super::dsv4_fixture::dsv4_dims;
use super::session::{ids, session_model, Model};
use crate::seam::HOST_LAYER_MAJOR_MIN_ROWS;
use crate::EngineConfig;
use infr_cpu::op_tap::TappedOp;

/// Blocks read from the file so far, over every pool of `store`.
fn reads(store: &infr_cpu::paged::PagedWeights) -> u64 {
    store.pool_stats().iter().map(|p| p.stats.reads).sum()
}

/// **A prompt of several tall chunks reads each expert it routes to about once when the experts do
/// not fit the arena: the CPU prefills it layer-major on its own.** The canonical layout with 32
/// experts of 32 KiB per bank (480 expert blocks over five layers), a 600-token prompt in chunks
/// of [`HOST_LAYER_MAJOR_MIN_ROWS`] (three chunks), and four runs of the same prompt:
///
/// - an arena that holds every block: nothing is read twice, so its reads are the blocks the
///   prompt routes to. Every block fits, so the automatic rule keeps chunk-major (nothing to save);
/// - an arena of 120 expert slots, a quarter of the blocks, left to the automatic rule
///   (`paging.layer_major` unset): it must run layer-major (the op tap shows the order) and read
///   each routed block once in its prefill. Its count may pass the first run's by the decode
///   step's misses — the prompt's last position runs in the decode loop after the prefill, and
///   finds early layers' experts evicted — so it is held to that bound;
/// - the same small arena forced chunk-major: every chunk sweeps the experts again;
/// - the same small arena in chunks of 37 left to the automatic rule: chunk-major, the shape the
///   real file measured slower layer-major ([`crate::seam::host_layer_major`]).
///
/// All four give the same logits, bit for bit. Measured 2026-11-17: printed on failure.
#[test]
fn layer_major_prefill_reads_each_expert_once_through_the_host_arena() {
    let mut d = dsv4_dims();
    d.n_expert = 32;
    d.n_used = 4;
    d.n_ff_exp = 128;
    let file = session_model(&d);
    let open = |tag: &str, ubatch: usize, layer_major: Option<bool>| {
        let mut ec = EngineConfig::default();
        ec.device.ubatch = Some(ubatch);
        ec.device.ubatch_specified = true;
        ec.paging.layer_major = layer_major;
        Model::open_with(tag, &file, ec)
    };
    let prompt = ids(600, 21);
    let tall = HOST_LAYER_MAJOR_MIN_ROWS;
    let slot = (d.n_embd * d.n_ff_exp * 4) as u64;
    let blocks = (3 * d.n_expert * d.compress_ratios.len()) as u64;
    let run = |m: &Model, ubatch: usize, slots: u64| {
        let store = m.host_arena(slots * slot).expect("an expert pool");
        let pools = store.pool_stats();
        assert!(
            pools.len() == 1 && pools[0].experts && pools[0].n_slots as u64 == slots.min(blocks),
            "the arena is not one expert pool of {slots} slots: {} pools",
            pools.len()
        );
        infr_cpu::op_tap::start();
        let got = m
            .generate_paged(&store, &prompt, 1, &mut None)
            .expect("generation");
        let ops = infr_cpu::op_tap::take();
        let rows: Vec<u32> = ops
            .iter()
            .filter_map(|o| match *o {
                TappedOp::Attention { rows } => Some(rows),
                _ => None,
            })
            .collect();
        // Layer-major: every chunk through the first layer before any reaches the second.
        let n_layer = d.compress_ratios.len();
        let chunks: Vec<u32> = (0..prompt.len() - 1)
            .step_by(ubatch)
            .map(|s| ubatch.min(prompt.len() - 1 - s) as u32)
            .collect();
        let mut by_layer: Vec<u32> = (0..n_layer).flat_map(|_| chunks.clone()).collect();
        by_layer.extend(std::iter::repeat_n(1, n_layer));
        (got.logits, reads(&store), rows == by_layer)
    };
    let auto = open("dsv4-pg-auto", tall, None);
    let (want, once, once_lm) = run(&auto, tall, blocks);
    let (lm_logits, lm, lm_lm) = run(&auto, tall, 120);
    let forced = open("dsv4-pg-cm", tall, Some(false));
    let (cm_logits, cm, cm_lm) = run(&forced, tall, 120);
    let short = open("dsv4-pg-short", 37, None);
    let (short_logits, short_reads, short_lm) = run(&short, 37, 120);
    let decode = (3 * d.n_used * d.compress_ratios.len()) as u64;
    let counts = format!(
        "{blocks} expert blocks; read with every block cached {once}, with 120 slots {lm} \
         layer-major and {cm} chunk-major; in chunks of 37, {short_reads}"
    );
    assert!(
        !short_lm,
        "chunks of 37 ran layer-major by default: {counts}"
    );
    assert!(
        !once_lm,
        "an arena holding every block ran layer-major: {counts}"
    );
    assert!(lm_lm, "the small arena did not run layer-major: {counts}");
    assert!(
        !cm_lm,
        "paging.layer_major = false ran layer-major: {counts}"
    );
    assert!(once <= blocks, "{counts}");
    assert!(
        (once..=once + decode).contains(&lm),
        "the layer-major prefill read experts again: {counts}"
    );
    assert!(
        cm > 2 * lm,
        "chunk-major did not re-read the experts: {counts}"
    );
    assert!(lm_logits == want, "layer-major moved the logits: {counts}");
    assert!(cm_logits == want, "chunk-major moved the logits: {counts}");
    assert!(
        short_logits == want,
        "chunks of 37 moved the logits: {counts}"
    );
}
