//! Batched V4 prefill against one token per graph: the same prompt prefilled in chunks of several
//! positions per graph must leave the same logits, the same per-row indexer selections and the
//! same compressed caches and rings as one position per graph, and decode on from there alike.
//! One token per graph is what `device.ubatch = 1` (`INFR_UBATCH=1`) gives, the switch every
//! model's batched prefill takes.

use super::super::dsv4_fixture::{dsv4_dims, Dsv4Dims, IndexerDims};
use super::super::{prefill_chunk_rows, DSV4_PREFILL_SCORE_BYTES};
use super::session::{ids, session_model, Model, SESSION_CTX};
use crate::seam::weights::SeamKv;
use crate::EngineConfig;
use infr_core::tensor::DType;
use infr_cpu::op_tap::TappedOp;
use infr_testkit::gguf::{Fill, SyntheticModel};

const CSA: usize = crate::seam::dsv4_plan::DSV4_CSA_RATIO as usize;

/// One fixture layout: the llama.cpp oracle cases' layer patterns (at the oracle's
/// hyper-connection count and indexer shape) and the canonical `dsv4_dims` file.
struct Layout {
    name: &'static str,
    dims: Dsv4Dims,
}

/// A layout at the oracle files' shape (`synthetic_deepseek4_llama_cpp.rs`' `oracle_dims`, no
/// null tail: that layer only exists on llama.cpp's side).
fn oracle_layout(
    name: &'static str,
    ratios: &[usize],
    hash_layer_count: usize,
    top_k: usize,
    n_head: usize,
) -> Layout {
    Layout {
        name,
        dims: Dsv4Dims {
            swiglu_clamp_exp: (0..ratios.len()).map(|l| 7.0 + 0.5 * l as f32).collect(),
            compress_ratios: ratios.to_vec(),
            hash_layer_count,
            hc_mult: 4,
            indexer: IndexerDims {
                head_size: 128,
                n_head,
                top_k,
            },
            ..dsv4_dims()
        },
    }
}

fn layouts() -> Vec<Layout> {
    vec![
        oracle_layout("ratio0", &[0, 0, 0, 0, 0], 2, 512, 4),
        oracle_layout("hca", &[128, 128, 128, 128], 1, 512, 4),
        oracle_layout("csa", &[4, 4, 4, 4], 1, 5, 16),
        oracle_layout("mixed", &[0, 4, 128, 4, 0], 2, 512, 4),
        Layout {
            name: "canonical",
            dims: dsv4_dims(),
        },
    ]
}

/// `layout`'s file opened with prefill chunks of `ubatch` rows.
fn open(layout: &Layout, m: &SyntheticModel, ubatch: usize) -> Model {
    let mut ec = EngineConfig::default();
    ec.device.ubatch = Some(ubatch);
    ec.device.ubatch_specified = true;
    Model::open_with(&format!("dsv4-pf-{}-{ubatch}", layout.name), m, ec)
}

/// `layout`'s file, with its token embedding stored as f16 when `f16_embedding`. The fixture's
/// f32 table has no embedding-gather kernel, so the prefill embeds on the host and hands the
/// hash-routed layers their token ids in a buffer of their own (`PfChunk::ids`); an f16 table is
/// gathered in the graph from the token ids the prefill uploads in place of embedded rows.
fn layout_file(layout: &Layout, f16_embedding: bool) -> SyntheticModel {
    let mut m = session_model(&layout.dims);
    if f16_embedding {
        let t = m
            .tensors
            .iter_mut()
            .find(|t| t.name == "token_embd.weight")
            .expect("the fixture's token embedding");
        t.fill = Fill::ExactF16(t.values());
    }
    m
}

/// One `(position, layer)` row an indexer scored: the visible blocks it kept, sorted, or `None`
/// when it kept them all.
type Selection = (usize, usize, Option<Vec<u32>>);

/// What one prefill of a prompt, and the decode after it, left behind.
#[derive(PartialEq)]
struct Observed {
    /// The last prompt position's logits.
    logits: Vec<f32>,
    /// Every scoring row of the prefill, ordered by position, then layer.
    selections: Vec<Selection>,
    /// The compressed caches and rings after the prompt.
    state: Vec<Vec<u8>>,
    /// The greedy ids of the first generation and of the decode that extends it.
    decoded: Vec<u32>,
    /// The logits the extending decode sampled its first id from.
    decode_logits: Vec<f32>,
    /// The compressed caches and rings after the decode.
    end_state: Vec<Vec<u8>>,
}

/// How many ids the extending decode generates.
const DECODE: usize = 6;

/// The rows of each graph a `len`-token prompt runs in, prefilled in chunks of `chunk`
/// positions: the batched prefill takes every position but the last in chunks (when more than
/// two are left), the decode loop the last. At the fixtures' lengths V4's memory bound on the
/// chunk (`dsv4_emit::prefill_chunk_rows`) never bites.
fn graph_rows(len: usize, chunk: usize) -> Vec<u32> {
    let chunk = if len > 2 { chunk } else { 1 };
    let mut rows: Vec<u32> = (0..len - 1)
        .step_by(chunk)
        .map(|s| (chunk.min(len - 1 - s)) as u32)
        .collect();
    rows.push(1);
    rows
}

/// Prefill `prompt` on a fresh session of `m` (the prompt's last position is the decode loop's,
/// as on every model), then decode [`DECODE`] more ids on the same session one token per graph.
/// Checks, through `infr_cpu::op_tap`, that the prompt ran in the graphs a `chunk`-position
/// prefill runs (every layer's attention at each graph's row count) and that each graph gathered
/// its token embedding from the f16 table in the graph exactly when `gathered`.
fn observe(m: &Model, prompt: &[u32], chunk: usize, gathered: bool) -> Observed {
    let mut s: Option<SeamKv> = None;
    infr_cpu::indexer_tap::start();
    infr_cpu::op_tap::start();
    let first = m.generate(prompt, 1, &mut s, None);
    let tapped = infr_cpu::indexer_tap::take();
    let ops = infr_cpu::op_tap::take();
    let first = first.expect("prefill");
    let want_rows = graph_rows(prompt.len(), chunk);
    let attention: Vec<u32> = ops
        .iter()
        .filter_map(|o| match *o {
            TappedOp::Attention { rows } => Some(rows),
            _ => None,
        })
        .collect();
    let per_layer: Vec<u32> = want_rows
        .iter()
        .flat_map(|&r| std::iter::repeat_n(r, m.cfg.n_layer))
        .collect();
    assert_eq!(
        attention,
        per_layer,
        "a {}-token prompt in chunks of {chunk} ran in other graphs",
        prompt.len()
    );
    let embedded: Vec<u32> = ops
        .iter()
        .filter_map(|o| match *o {
            TappedOp::EmbedGather {
                rows,
                table: DType::F16,
            } => Some(rows),
            _ => None,
        })
        .collect();
    let want_embedded = if gathered { want_rows } else { Vec::new() };
    assert_eq!(
        embedded, want_embedded,
        "the token embedding was not gathered in the graph exactly when the table is f16"
    );
    assert_eq!(first.n_cached, 0);
    let state = m.comp_state(&s, false);
    let csa_layers: Vec<usize> = (0..m.cfg.n_layer)
        .filter(|&l| m.cfg.layer_compress_ratio(l) == CSA)
        .collect();
    let mut selections: Vec<Selection> = tapped
        .iter()
        .filter_map(|r| {
            // Every graph that scores runs one indexer per ratio-4 layer, in layer order; a row
            // that sees no block is scored over nothing and kept for no one.
            let visible = (r.pos + 1) / CSA;
            (visible > 0).then(|| {
                let mut kept: Vec<u32> = r
                    .kept
                    .iter()
                    .copied()
                    .filter(|&j| (j as usize) < visible)
                    .collect();
                kept.sort_unstable();
                kept.dedup();
                let layer = csa_layers[r.op % csa_layers.len()];
                (r.pos, layer, (kept.len() != visible).then_some(kept))
            })
        })
        .collect();
    selections.sort_by_key(|s| (s.0, s.1));
    let scoring_rows = (0..prompt.len()).filter(|&p| (p + 1) / CSA > 0).count();
    assert_eq!(
        selections.len(),
        scoring_rows * csa_layers.len(),
        "the indexer scored another number of rows than the prompt has"
    );
    let extended = [prompt, &first.ids[..]].concat();
    let next = m.generate(&extended, DECODE, &mut s, None).expect("decode");
    assert_eq!(
        next.n_cached,
        prompt.len(),
        "the decode did not extend the prefilled session"
    );
    Observed {
        logits: first.logits,
        selections,
        state,
        decoded: [first.ids, next.ids].concat(),
        decode_logits: next.logits,
        end_state: m.comp_state(&s, false),
    }
}

/// The prompt lengths each layout is prefilled at: inside the raw window, then past it and across
/// several HCA blocks, longest on the layouts with an HCA tier; on `csa`, well past the first
/// position whose top-k drops a block.
fn lens(layout: &Layout) -> &'static [usize] {
    match layout.name {
        "hca" | "mixed" => &[70, 300, 520],
        "csa" => &[70, 200],
        _ => &[70, 300],
    }
}

/// The chunk heights every layout is prefilled at: under a CSA block, unaligned to every block
/// and longer than a CSA ring, longer than an HCA block and its ring, and the whole prompt.
const CHUNKS: [usize; 4] = [3, 37, 150, SESSION_CTX];

/// What differs between `a` and `b`, one line per field, prefixed by `what`.
fn report(what: &str, a: &Observed, b: &Observed) -> Vec<String> {
    let max = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max)
    };
    let mut d = Vec::new();
    if a.logits != b.logits {
        d.push(format!(
            "{what}: last-position logits differ by up to {:e}",
            max(&a.logits, &b.logits)
        ));
    }
    if a.selections != b.selections {
        let rows: Vec<_> = a
            .selections
            .iter()
            .zip(&b.selections)
            .filter(|(x, y)| x != y)
            .map(|(x, _)| (x.0, x.1))
            .collect();
        d.push(format!(
            "{what}: the selection differs at (position, layer) {rows:?}"
        ));
    }
    let buffers = |x: &[Vec<u8>], y: &[Vec<u8>]| -> Vec<usize> {
        (0..x.len()).filter(|&i| x[i] != y[i]).collect()
    };
    if a.state != b.state {
        d.push(format!(
            "{what}: compressed buffers {:?} differ after the prompt",
            buffers(&a.state, &b.state)
        ));
    }
    if a.decoded != b.decoded {
        d.push(format!(
            "{what}: decoded {:?}, one position per graph {:?}",
            a.decoded, b.decoded
        ));
    }
    if a.decode_logits != b.decode_logits {
        d.push(format!(
            "{what}: the decode's logits differ by up to {:e}",
            max(&a.decode_logits, &b.decode_logits)
        ));
    }
    if a.end_state != b.end_state {
        d.push(format!(
            "{what}: compressed buffers {:?} differ after the decode",
            buffers(&a.end_state, &b.end_state)
        ));
    }
    d
}

/// **A prompt prefilled in chunks equals the same prompt prefilled one position per graph, bit
/// for bit**: the last position's logits, every row's indexer selection, every compressed cache
/// and ring after the prompt, and the greedy decode after it with its logits and buffers. On the
/// layout named `name` ([`layouts`]: each tier alone, all of them in one model, hash and bias
/// routing, the oracle's and the canonical hyper-connection stream counts; one test each below),
/// at [`lens`], at every [`CHUNKS`] height. A chunk boundary lands inside blocks of both tiers; a
/// chunk holds rows that see no block yet beside rows that do, commits several blocks, mixes ring
/// rows earlier graphs persisted with its own rows and persists more rows than the ring holds.
/// Every layout is hash-routed on its first layers, so each also runs with the embedding table
/// gathered in the graph ([`layout_file`]): there the token ids the hash routing reads are the
/// chunk's only input.
///
/// Bit-exact, not close: on the CPU every op this file's graph runs computes a row the same way
/// whatever the graph's row count (the f32 weights take one dot per output and row), so equal
/// visibility, selection and routing give equal bits. A difference of any size is a row seeing
/// what it should not, or missing what it should see.
fn batched_equals_one_position_per_graph(name: &str) {
    let layout = layouts()
        .into_iter()
        .find(|l| l.name == name)
        .unwrap_or_else(|| panic!("no layout {name}"));
    assert!(
        layout.dims.hash_layer_count > 0,
        "{name} is not hash-routed"
    );
    let host = layout_file(&layout, false);
    let gathered = layout_file(&layout, true);
    // (what, gathered in the graph, reference, chunked runs): every chunk height on the
    // host-embedded file, one unaligned height on the gathered one.
    let k = CHUNKS[1];
    let cases = [
        (
            "host embedding",
            false,
            open(&layout, &host, 1),
            CHUNKS.map(|k| (k, open(&layout, &host, k))).into(),
        ),
        (
            "gathered embedding",
            true,
            open(&layout, &gathered, 1),
            vec![(k, open(&layout, &gathered, k))],
        ),
    ];
    let mut diffs = Vec::new();
    for &len in lens(&layout) {
        let prompt = ids(len, 11);
        for (what, gathered, one, runs) in &cases {
            let want = observe(one, &prompt, 1, *gathered);
            for (k, m) in runs {
                let got = observe(m, &prompt, *k, *gathered);
                diffs.extend(report(
                    &format!("{name} at {len} tokens, {what}, chunk {k}"),
                    &got,
                    &want,
                ));
            }
        }
    }
    assert!(diffs.is_empty(), "{}", diffs.join("\n"));
}

#[test]
fn batched_prefill_equals_one_position_per_graph_ratio0() {
    batched_equals_one_position_per_graph("ratio0");
}

#[test]
fn batched_prefill_equals_one_position_per_graph_hca() {
    batched_equals_one_position_per_graph("hca");
}

#[test]
fn batched_prefill_equals_one_position_per_graph_csa() {
    batched_equals_one_position_per_graph("csa");
}

#[test]
fn batched_prefill_equals_one_position_per_graph_mixed() {
    batched_equals_one_position_per_graph("mixed");
}

#[test]
fn batched_prefill_equals_one_position_per_graph_canonical() {
    batched_equals_one_position_per_graph("canonical");
}

/// **One position per graph through the batched prefill is the decode loop's own computation**:
/// `device.ubatch = 1` prefills a position in a one-row graph without the model head, the decode
/// loop in a one-row graph with it, and a prompt fed one turn per token (every turn extending the
/// session by one, so the decode loop runs every position) leaves the same logits and compressed
/// buffers. This is what makes `ubatch = 1` the reference of the test above.
#[test]
fn one_position_per_graph_equals_one_position_per_turn() {
    let (one, _) = canonical(1);
    let prompt = ids(70, 11);
    let want = observe(&one, &prompt, 1, false);
    let mut s = None;
    let mut last = Vec::new();
    for len in 1..=prompt.len() {
        let turn = one.generate(&prompt[..len], 1, &mut s, None).expect("turn");
        assert_eq!(
            turn.n_cached,
            len - 1,
            "the turn to {len} tokens re-prefilled"
        );
        last = turn.logits;
    }
    assert!(last == want.logits, "logits");
    assert!(
        one.comp_state(&s, false) == want.state,
        "compressed buffers"
    );
}

/// The canonical layout (every tier, hash routing, its own stream count) opened at chunk height
/// `chunk`, and the same file one position per graph.
fn canonical(chunk: usize) -> (Model, Model) {
    let layout = layouts()
        .into_iter()
        .find(|l| l.name == "canonical")
        .expect("the canonical layout");
    let file = session_model(&layout.dims);
    (open(&layout, &file, chunk), open(&layout, &file, 1))
}

/// **An extending turn prefills only its suffix, in chunks, from where the turn before left the
/// rings.** The first turn stops at 150 tokens, inside an HCA block and a CSA block; the second
/// extends it to 300 in chunks of 37, completing the HCA block at position 255 from ring rows the
/// first turn persisted plus its own, and must equal one prefill of the 300 tokens one position
/// per graph: logits, and every compressed buffer.
#[test]
fn batched_extending_turn_prefills_only_its_suffix() {
    let (chunked, one) = canonical(37);
    let prompt = ids(300, 12);
    let mut s = None;
    let first = chunked
        .generate(&prompt[..150], 1, &mut s, None)
        .expect("first turn");
    assert_eq!(first.n_cached, 0);
    let second = chunked
        .generate(&prompt, 1, &mut s, None)
        .expect("second turn");
    assert_eq!(
        second.n_cached, 150,
        "the extending turn re-prefilled instead of reusing the session"
    );
    let mut fresh = None;
    let want = one
        .generate(&prompt, 1, &mut fresh, None)
        .expect("one prefill");
    assert!(second.logits == want.logits, "logits");
    assert!(
        chunked.comp_state(&s, false) == one.comp_state(&fresh, false),
        "compressed buffers"
    );
}

/// **A turn that does not extend the session resets it before a chunked prefill**: after a
/// 300-token prompt, a 40-token prompt sharing nothing gives a fresh session's logits and leaves
/// the rings byte-equal to a fresh session's (40 positions rewrite only part of the HCA ring).
#[test]
fn batched_non_extending_turn_resets() {
    let (chunked, _) = canonical(7);
    let (long, short) = (ids(300, 13), ids(40, 14));
    assert_ne!(long[0], short[0], "the prompts share no prefix");
    let mut s = None;
    chunked.generate(&long, 1, &mut s, None).expect("long turn");
    let got = chunked
        .generate(&short, 1, &mut s, None)
        .expect("short turn");
    assert_eq!(got.n_cached, 0, "a turn sharing nothing reused the session");
    let mut fresh = None;
    let want = chunked
        .generate(&short, 1, &mut fresh, None)
        .expect("fresh");
    assert!(got.logits == want.logits, "logits");
    assert!(
        chunked.comp_state(&s, true) == chunked.comp_state(&fresh, true),
        "the reset left a ring row dirty"
    );
}

/// **A prefill cancelled between two chunks leaves nothing the next request reuses.** The first
/// chunk of an extending turn advances the rings past what the session records; the request after
/// the cancel, extending the same first turn, must re-prefill from position 0 and equal a fresh
/// session.
///
/// Two guards in the runner refuse that reuse, deliberately redundant (each is the backstop of the
/// other): a session with unrewindable state records no prompt while a turn runs (`cached` is
/// emptied before the prefill), and `written_end` is raised past each chunk before it executes. A
/// reuse needs both to fail, so the outcome alone cannot show either; the state the cancel leaves
/// is asserted for each. The CPU's execute cannot be interrupted part way (it has no abort check),
/// so a cancel always lands between chunks; an execute that failed part way would leave the same
/// state, since `written_end` is raised before it starts.
#[test]
fn batched_prefill_cancelled_between_chunks_is_not_reused() {
    let (chunked, _) = canonical(7);
    let prompt = ids(80, 15);
    let then = [&prompt[..30], &ids(20, 16)[..]].concat();
    let mut s = None;
    chunked
        .generate(&prompt[..30], 1, &mut s, None)
        .expect("first turn");
    let req = crate::sampling::RequestCtx::new(Default::default()).abort_at_poll(1);
    let err = match chunked.generate(&prompt, 1, &mut s, Some(&req)) {
        Ok(_) => panic!("the prefill was not cancelled"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("cancelled during prompt prefill"),
        "{err}"
    );
    let kv = s.as_ref().expect("a session");
    assert!(
        kv.cached.is_empty(),
        "the cancelled turn left the session recording {} tokens",
        kv.cached.len()
    );
    assert!(
        kv.written_end >= 30 + 7,
        "the cancelled turn's first chunk ran without raising written_end ({})",
        kv.written_end
    );
    let after = chunked
        .generate(&then, 1, &mut s, None)
        .expect("turn after the cancel");
    assert_eq!(
        after.n_cached, 0,
        "the turn after the cancel reused the session"
    );
    let mut fresh = None;
    let want = chunked.generate(&then, 1, &mut fresh, None).expect("fresh");
    assert!(after.logits == want.logits, "logits");
    assert!(
        chunked.comp_state(&s, false) == chunked.comp_state(&fresh, false),
        "compressed buffers"
    );
}

/// **V4's prefill chunk shrinks so a CSA layer's `rows × blocks` buffers stay within
/// [`DSV4_PREFILL_SCORE_BYTES`]**, at the heights worked out by hand for a session chunk of 1024
/// rows (a chunk starting at `s` with `r` rows holds `r × ((s + r) / 4)` f32 per buffer, against
/// 64 MiB = 67 108 864 bytes): at 8k, 1024 rows take 1024 × 2304 × 4 = 9 437 184 bytes, so the
/// chunk keeps its height; at 64k, 1008 × 16 636 × 4 = 67 076 352 fits and 1009 × 16 636 × 4 =
/// 67 142 896 does not; at 160k, 408 × 41 062 × 4 = 67 013 184 fits and 409 × 41 062 × 4 =
/// 67 177 432 does not; at 1M, 63 × 262 159 × 4 = 66 064 068 fits and 64 × 262 160 × 4 =
/// 67 112 960 does not. A file with no CSA layer keeps the session's height at any context.
#[test]
fn prefill_chunk_rows_bound_the_csa_buffers() {
    assert_eq!(
        DSV4_PREFILL_SCORE_BYTES,
        64 << 20,
        "the heights below are for 64 MiB"
    );
    let csa = Model::open_with(
        "dsv4-pf-bound-csa",
        &session_model(&dsv4_dims()),
        EngineConfig::default(),
    );
    for (start, rows) in [
        (0, 1024),
        (8 << 10, 1024),
        (64 << 10, 1008),
        (160 << 10, 408),
    ] {
        assert_eq!(
            prefill_chunk_rows(&csa.cfg, start, 1024),
            rows,
            "at {start}"
        );
    }
    assert_eq!(prefill_chunk_rows(&csa.cfg, 1 << 20, 1024), 63, "at 1M");
    assert_eq!(
        prefill_chunk_rows(&csa.cfg, 64 << 20, 1024),
        1,
        "never below one row"
    );
    let mut d = dsv4_dims();
    d.compress_ratios = vec![0, 128, 0];
    d.swiglu_clamp_exp.truncate(d.compress_ratios.len());
    let no_csa = Model::open_with(
        "dsv4-pf-bound-hca",
        &session_model(&d),
        EngineConfig::default(),
    );
    assert_eq!(prefill_chunk_rows(&no_csa.cfg, 1 << 20, 1024), 1024);
}

/// **A prompt chunk is never taller than the sliding-window ring of the session it runs in.** The
/// raw window's ring holds `window + chunk` rows for the chunk height the session was MADE with;
/// `ubatch_rows` answers for the placement scope current at each call, so a session made under a
/// pinned short chunk and extended outside that scope would get the default, far taller chunk and
/// overwrite ring rows its own earlier rows still attend. The extending turn must equal a fresh
/// prefill of the whole prompt.
#[test]
fn prefill_chunk_never_outgrows_the_window_ring() {
    let layout = layouts()
        .into_iter()
        .find(|l| l.name == "canonical")
        .expect("the canonical layout");
    let file = session_model(&layout.dims);
    let m = Model::open_with("dsv4-pf-ring", &file, EngineConfig::default());
    let prompt = ids(300, 17);
    const PINNED: usize = 8;
    let mut s: Option<SeamKv> = None;
    {
        let _scope = crate::seam::PlacementScope::enter(std::sync::Arc::default());
        crate::seam::pin_ubatch(PINNED);
        m.generate(&prompt[..20], 1, &mut s, None)
            .expect("first turn");
    }
    let kv = s.as_ref().expect("a session");
    assert_eq!(
        kv.ubatch, PINNED,
        "the session's rings were sized for another chunk"
    );
    assert!(
        crate::seam::ubatch_rows(&EngineConfig::default()) > PINNED,
        "outside the scope the chunk height is not taller, so nothing is tested"
    );
    let got = m
        .generate(&prompt, 1, &mut s, None)
        .expect("extending turn");
    assert_eq!(
        got.n_cached, 20,
        "the second turn did not extend the session"
    );
    let want = m.generate(&prompt, 1, &mut None, None).expect("fresh");
    assert!(
        got.logits == want.logits,
        "the extending turn read a clobbered ring"
    );
}

/// The canonical layout with its routed expert banks stored as Q8_0, the shipped expert bank's
/// sibling that `Op::MoeFfn` feeds int8-quantized activations once a graph has two rows or more
/// (its `int8_ok`; `infr_cpu`'s `expert_acts_kind`), where one row keeps f32 activations.
fn quantized_banks() -> SyntheticModel {
    let mut m = session_model(&dsv4_dims());
    let mut n = 0;
    for t in m
        .tensors
        .iter_mut()
        .filter(|t| t.name.ends_with("_exps.weight"))
    {
        t.fill = Fill::Quant(DType::Q8_0);
        n += 1;
    }
    assert!(n > 0, "the fixture has no routed expert banks");
    m
}

/// `quantized_banks` opened at chunk height `ubatch`, in reference mode when `reference`.
fn open_quantized(m: &SyntheticModel, ubatch: usize, reference: bool) -> Model {
    let mut ec = EngineConfig::default();
    ec.device.ubatch = Some(ubatch);
    ec.device.ubatch_specified = true;
    ec.kernels.cpu.reference = reference;
    Model::open_with(&format!("dsv4-pf-q8-{ubatch}-{reference}"), m, ec)
}

/// The largest |logit| difference chunked prefill may make against one position per graph on
/// [`quantized_banks`] in the production kernels, at the last prompt position and at the first
/// decoded one: the int8 rounding of the expert activations, which a one-row graph does not
/// apply. Measured at 300 tokens: 1.71e-4 and 2.37e-4.
const QUANTIZED_BANKS_MAX_DIFF: f32 = 1e-3;

/// **With quantized expert banks, chunked prefill is close to one position per graph, not equal,
/// and the CPU's reference mode makes it equal again.** A chunk sends the Q8_0 banks int8
/// activations and one position per graph f32 ones, so the two differ by that rounding,
/// everywhere downstream of the first MoE layer: the logits stay within
/// [`QUANTIZED_BANKS_MAX_DIFF`] (and DO differ, or the int8 path would not be running), while the
/// indexer's selections may flip at near-ties (the canonical indexer's four heads make many) and
/// the compressed buffers differ. Every chunk height gives the same bits as every other: each
/// row's activations are quantized on their own. Reference mode keeps every row count on f32
/// activations, so there chunked and one-position prefill agree bit for bit on everything
/// [`observe`] records, and its one-position run equals the production one-position run (no
/// int8 activation reaches a one-row graph).
#[test]
fn quantized_banks_chunked_prefill_is_close_and_reference_is_exact() {
    let m = quantized_banks();
    let prompt = ids(300, 11);
    let one = observe(&open_quantized(&m, 1, false), &prompt, 1, false);
    let max = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let chunked: Vec<Observed> = CHUNKS
        .iter()
        .map(|&k| observe(&open_quantized(&m, k, false), &prompt, k, false))
        .collect();
    let (logits, decode) = (
        max(&chunked[0].logits, &one.logits),
        max(&chunked[0].decode_logits, &one.decode_logits),
    );
    let flipped = (chunked[0].selections.iter())
        .zip(&one.selections)
        .filter(|(a, b)| a != b)
        .count();
    let measured = format!(
        "max |logit diff| {logits:e} at the prompt's end, {decode:e} at the first decoded \
         position; {flipped} of {} selections differ",
        one.selections.len()
    );
    assert!(
        logits > 0.0,
        "the chunks took the one-row f32 path, no int8 activation ran: {measured}"
    );
    assert!(
        logits <= QUANTIZED_BANKS_MAX_DIFF && decode <= QUANTIZED_BANKS_MAX_DIFF,
        "chunked prefill moved the logits too far: {measured}"
    );
    for (k, c) in CHUNKS.iter().zip(&chunked).skip(1) {
        let d = report(
            &format!("chunk {k} against chunk {}", CHUNKS[0]),
            c,
            &chunked[0],
        );
        assert!(
            d.is_empty(),
            "{}",
            d.join(
                "
"
            )
        );
    }
    let ref_one = observe(&open_quantized(&m, 1, true), &prompt, 1, false);
    let ref_chunked = observe(&open_quantized(&m, 37, true), &prompt, 37, false);
    let d = report("reference mode, chunk 37", &ref_chunked, &ref_one);
    assert!(d.is_empty(), "{}", d.join("\n"));
    let d = report(
        "one position per graph, production against reference",
        &one,
        &ref_one,
    );
    assert!(d.is_empty(), "{}", d.join("\n"));
}

/// The canonical layout stored in the shipped V4-Flash file's dtypes (`docs/deepseek.md`, "The
/// shipped file's dtypes"): every attention, compressor, indexer, hyper-connection and
/// shared-expert matmul weight, the APE tables, `token_embd` and `output_hc_fn` Q8_0, the head
/// `output` Q6_K, the routers BF16 and the routed expert banks Q2_K; norms, sinks, biases and the
/// hash tables stay as the fixture has them. Widened so every one of those rows is a whole
/// number of its dtype's blocks (`n_embd` and `n_ff_exp` 256 for the Q6_K head and the Q2_K
/// banks, `head_dim` and the indexer's head 64 and 32 for the Q8_0 output and APE rows).
fn shipped_dtypes() -> SyntheticModel {
    let d = Dsv4Dims {
        n_embd: 256,
        head_dim: 64,
        n_ff_exp: 256,
        indexer: IndexerDims {
            head_size: 32,
            ..super::super::dsv4_fixture::dsv4_indexer()
        },
        ..dsv4_dims()
    };
    const Q8_0: &[&str] = &[
        "attn_q_a",
        "attn_q_b",
        "attn_kv",
        "attn_output_a",
        "attn_output_b",
        "hc_attn_fn",
        "hc_ffn_fn",
        "attn_compressor_kv",
        "attn_compressor_gate",
        "attn_compressor_ape",
        "indexer.proj",
        "indexer.attn_q_b",
        "indexer_compressor_kv",
        "indexer_compressor_gate",
        "indexer_compressor_ape",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
    ];
    let mut m = session_model(&d);
    let mut per_dtype = std::collections::BTreeMap::<String, usize>::new();
    for t in &mut m.tensors {
        let Some(base) = t.name.strip_suffix(".weight") else {
            continue;
        };
        // `blk.{l}.attn_q_b` → `attn_q_b`; the model-level tensors keep their whole name.
        let name = match base.strip_prefix("blk.").and_then(|r| r.split_once('.')) {
            Some((_, local)) => local,
            None => base,
        };
        let fill = if Q8_0.contains(&name) || name == "token_embd" || name == "output_hc_fn" {
            Fill::Quant(DType::Q8_0)
        } else if name == "output" {
            Fill::Quant(DType::Q6K)
        } else if name.ends_with("_exps") {
            Fill::Quant(DType::Q2K)
        } else if name == "ffn_gate_inp" {
            Fill::ExactBf16(t.values())
        } else {
            continue;
        };
        *per_dtype.entry(name.to_string()).or_default() += 1;
        t.fill = fill;
    }
    // Every name above matched on every layer that has it: a renamed tensor would silently stay
    // f32 and leave its dtype untested.
    let ratio4 = d.compress_ratios.iter().filter(|&&r| r == 4).count();
    let compressed = d.compress_ratios.iter().filter(|&&r| r != 0).count();
    let want = |n: &str| match n {
        "token_embd" | "output" | "output_hc_fn" => 1,
        n if n.starts_with("indexer") => ratio4,
        n if n.starts_with("attn_compressor") => compressed,
        _ => d.n_layer(),
    };
    for (n, &got) in &per_dtype {
        assert_eq!(got, want(n), "{n}: re-typed on {got} layers");
    }
    assert_eq!(per_dtype.len(), Q8_0.len() + 7, "{per_dtype:?}");
    m
}

/// **On the shipped file's dtypes, chunked prefill equals one position per graph, bit for bit.**
/// Every Q8_0 matmul runs `Op::Linear` at the chunk's row count on the batched int8 kernels and at
/// one row on the one-row kernels, which give the same bits (`infr-cpu`'s
/// `linear_batch_identity`); the Q2_K expert banks keep f32 activations at every row count; the
/// BF16 routers take an f32 dot. So unlike [`quantized_banks`], nothing here may differ: the
/// logits, every row's kept blocks, every compressed cache and ring, and the decode after, at a
/// chunk height under a CSA block, an unaligned one, and the whole prompt, in the production
/// kernels.
#[test]
fn shipped_dtypes_chunked_prefill_equals_one_position_per_graph() {
    let m = shipped_dtypes();
    let open = |ubatch: usize| {
        let mut ec = EngineConfig::default();
        ec.device.ubatch = Some(ubatch);
        ec.device.ubatch_specified = true;
        Model::open_with(&format!("dsv4-pf-shipped-{ubatch}"), &m, ec)
    };
    // Past the raw window, an HCA block and the first position whose top-k drops a block.
    let prompt = ids(200, 11);
    let want = observe(&open(1), &prompt, 1, false);
    let mut diffs = Vec::new();
    for k in [3, 37, SESSION_CTX] {
        let got = observe(&open(k), &prompt, k, false);
        diffs.extend(report(&format!("shipped dtypes, chunk {k}"), &got, &want));
    }
    assert!(diffs.is_empty(), "{}", diffs.join("\n"));
}
