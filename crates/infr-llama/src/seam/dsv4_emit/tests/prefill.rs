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
    open_as(layout, m, ubatch, None)
}

/// `layout`'s file opened with prefill chunks of `ubatch` rows, prefilling layer-major in
/// segments of at most `segment` rows when `layer_major` is `Some(segment)` (`usize::MAX`: one
/// segment), chunk-major when `None`.
fn open_as(
    layout: &Layout,
    m: &SyntheticModel,
    ubatch: usize,
    layer_major: Option<usize>,
) -> Model {
    let mut ec = EngineConfig::default();
    ec.device.ubatch = Some(ubatch);
    ec.device.ubatch_specified = true;
    if let Some(rows) = layer_major {
        ec.paging.layer_major = Some(true);
        ec.paging.layer_major_rows = Some(rows);
    }
    let order = layer_major.map_or("cm".to_string(), |r| format!("lm{r}"));
    Model::open_with(&format!("dsv4-pf-{}-{ubatch}-{order}", layout.name), m, ec)
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

/// The row count of every `Op::Attention` a `len`-token prompt runs, in execution order, on a
/// model of `n_layer` layers prefilled in chunks of `chunk`: chunk-major, every graph runs every
/// layer; layer-major in segments of at most `segment` rows (`layer_major: Some(segment)`), each
/// segment runs layer by layer, every chunk of the segment through a layer before the next layer.
/// The decode loop's one-row graph runs every layer last either way.
fn attention_rows(
    len: usize,
    chunk: usize,
    n_layer: usize,
    layer_major: Option<usize>,
) -> Vec<u32> {
    let rows = graph_rows(len, chunk);
    let (prefill, decode) = rows.split_at(rows.len() - 1);
    let mut out = Vec::new();
    match layer_major {
        None => {
            for &r in prefill {
                out.extend(std::iter::repeat_n(r, n_layer));
            }
        }
        Some(segment) => {
            // Segments by hand: consecutive chunks while their rows fit, a lone chunk otherwise.
            let mut segs: Vec<Vec<u32>> = Vec::new();
            let mut held = 0usize;
            for &r in prefill {
                match segs.last_mut() {
                    Some(seg) if held + r as usize <= segment => seg.push(r),
                    _ => {
                        segs.push(vec![r]);
                        held = 0;
                    }
                }
                held += r as usize;
            }
            for seg in segs {
                for _ in 0..n_layer {
                    out.extend(&seg);
                }
            }
        }
    }
    for &r in decode {
        out.extend(std::iter::repeat_n(r, n_layer));
    }
    out
}

/// Prefill `prompt` on a fresh session of `m` (the prompt's last position is the decode loop's,
/// as on every model), then decode [`DECODE`] more ids on the same session one token per graph.
/// Checks, through `infr_cpu::op_tap`, that the prompt ran in the graphs a `chunk`-position
/// chunk-major prefill runs (every layer's attention at each graph's row count) and that each
/// graph gathered its token embedding from the f16 table in the graph exactly when `gathered`.
fn observe(m: &Model, prompt: &[u32], chunk: usize, gathered: bool) -> Observed {
    observe_as(m, prompt, chunk, None, gathered)
}

/// [`observe`], for a model prefilling layer-major in segments of at most `segment` rows when
/// `layer_major` is `Some(segment)`: the op tap must then show every layer's attention in that
/// order ([`attention_rows`]), so a run that fell back to chunk-major cannot pass. The embedding
/// is gathered by the first layer span, chunk by chunk, so its gathers come in the same order as
/// chunk-major's.
fn observe_as(
    m: &Model,
    prompt: &[u32],
    chunk: usize,
    layer_major: Option<usize>,
    gathered: bool,
) -> Observed {
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
    assert_eq!(
        attention,
        attention_rows(prompt.len(), chunk, m.cfg.n_layer, layer_major),
        "a {}-token prompt in chunks of {chunk} ({}) ran in other graphs",
        prompt.len(),
        layer_major.map_or("chunk-major".into(), |s| format!(
            "layer-major in segments of {s}"
        )),
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
    // A position is scored once per ratio-4 layer, in layer order whichever way the prompt runs:
    // chunk-major, a graph runs its layers in order; layer-major, a layer runs every chunk of a
    // segment before the next layer starts. So a row's layer is how many rows of its position came
    // before it.
    let mut seen: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut selections: Vec<Selection> = tapped
        .iter()
        .filter_map(|r| {
            let nth = seen.entry(r.pos).or_insert(0);
            let layer = csa_layers[*nth];
            *nth += 1;
            // A row that sees no block is scored over nothing and kept for no one.
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

/// The layer-major orders every layout is prefilled in, as (chunk height, segment rows): one
/// segment at an unaligned small height, a mid height and the whole prompt (one chunk, still run
/// as one layer span per graph); segments of two chunks at two heights; and segments shorter
/// than a chunk, so each chunk is a segment of its own.
const LAYER_MAJOR: [(usize, usize); 6] = [
    (3, usize::MAX),
    (37, usize::MAX),
    (SESSION_CTX, usize::MAX),
    (3, 7),
    (37, 80),
    (150, 40),
];

/// **A prompt prefilled layer-major equals the same prompt prefilled one position per graph, bit
/// for bit** — and so equals the chunk-major prefill of [`batched_equals_one_position_per_graph`]:
/// the last position's logits, every row's indexer selection, every compressed cache and ring
/// after the prompt, and the greedy decode after it. On the layout named `name`, at [`lens`], in
/// every [`LAYER_MAJOR`] order, with the embedding on the host and gathered in the graph.
///
/// Layer-major runs one layer for every chunk of a segment before the next layer, carrying each
/// chunk's `hc_mult`-wide residual stream between the layer spans (`DecodeHandles::hc_resid`).
/// Every piece of V4's state belongs to one layer — its raw cache, compressed caches, rings and
/// indexer cache — and each layer still sees its chunks in ascending order, so each layer's
/// state receives the same writes in the same order as chunk-major, and each op the same rows:
/// equal bits, not merely close ones. The op tap checks every run took the order it names.
fn layer_major_equals_one_position_per_graph(name: &str) {
    let layout = layouts()
        .into_iter()
        .find(|l| l.name == name)
        .unwrap_or_else(|| panic!("no layout {name}"));
    let host = layout_file(&layout, false);
    let gathered = layout_file(&layout, true);
    let lm =
        |m: &SyntheticModel, (k, seg): (usize, usize)| (k, seg, open_as(&layout, m, k, Some(seg)));
    let cases = [
        (
            "host embedding",
            false,
            open(&layout, &host, 1),
            LAYER_MAJOR.map(|o| lm(&host, o)).into(),
        ),
        (
            "gathered embedding",
            true,
            open(&layout, &gathered, 1),
            vec![lm(&gathered, LAYER_MAJOR[1]), lm(&gathered, LAYER_MAJOR[4])],
        ),
    ];
    let mut diffs = Vec::new();
    for &len in lens(&layout) {
        let prompt = ids(len, 11);
        for (what, gathered, one, runs) in &cases {
            let want = observe(one, &prompt, 1, *gathered);
            for (k, seg, m) in runs {
                let got = observe_as(m, &prompt, *k, Some(*seg), *gathered);
                diffs.extend(report(
                    &format!("{name} at {len} tokens, {what}, layer-major chunk {k} segment {seg}"),
                    &got,
                    &want,
                ));
            }
        }
    }
    assert!(diffs.is_empty(), "{}", diffs.join("\n"));
}

#[test]
fn layer_major_prefill_equals_one_position_per_graph_ratio0() {
    layer_major_equals_one_position_per_graph("ratio0");
}

#[test]
fn layer_major_prefill_equals_one_position_per_graph_hca() {
    layer_major_equals_one_position_per_graph("hca");
}

#[test]
fn layer_major_prefill_equals_one_position_per_graph_csa() {
    layer_major_equals_one_position_per_graph("csa");
}

#[test]
fn layer_major_prefill_equals_one_position_per_graph_mixed() {
    layer_major_equals_one_position_per_graph("mixed");
}

#[test]
fn layer_major_prefill_equals_one_position_per_graph_canonical() {
    layer_major_equals_one_position_per_graph("canonical");
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

/// The canonical layout opened layer-major (one segment) at chunk height `chunk`, and the same
/// file one position per graph.
fn canonical_layer_major(chunk: usize) -> (Model, Model) {
    let layout = layouts()
        .into_iter()
        .find(|l| l.name == "canonical")
        .expect("the canonical layout");
    let file = session_model(&layout.dims);
    (
        open_as(&layout, &file, chunk, Some(usize::MAX)),
        open(&layout, &file, 1),
    )
}

/// **A layer-major extending turn prefills only its suffix, from where the turn before left
/// every layer's state**: [`batched_extending_turn_prefills_only_its_suffix`] with the second
/// turn's suffix run layer-major in chunks of 37 (the op tap shows the order), equal to one
/// prefill of the 300 tokens one position per graph: logits and every compressed buffer.
#[test]
fn layer_major_extending_turn_prefills_only_its_suffix() {
    let (lm, one) = canonical_layer_major(37);
    let prompt = ids(300, 12);
    let mut s = None;
    let first = lm
        .generate(&prompt[..150], 1, &mut s, None)
        .expect("first turn");
    assert_eq!(first.n_cached, 0);
    infr_cpu::op_tap::start();
    let second = lm.generate(&prompt, 1, &mut s, None).expect("second turn");
    let ops = infr_cpu::op_tap::take();
    assert_eq!(
        second.n_cached, 150,
        "the extending turn re-prefilled instead of reusing the session"
    );
    let attention: Vec<u32> = ops
        .iter()
        .filter_map(|o| match *o {
            TappedOp::Attention { rows } => Some(rows),
            _ => None,
        })
        .collect();
    // The suffix: positions 150..298 (149 rows) in chunks of 37 and the one row left, layer by
    // layer, then the decode row.
    let n_layer = lm.cfg.n_layer;
    let mut want = Vec::new();
    for _ in 0..n_layer {
        want.extend([37, 37, 37, 37, 1]);
    }
    want.extend(std::iter::repeat_n(1, n_layer));
    assert_eq!(attention, want, "the suffix did not run layer-major");
    let mut fresh = None;
    let want = one
        .generate(&prompt, 1, &mut fresh, None)
        .expect("one prefill");
    assert!(second.logits == want.logits, "logits");
    assert!(
        lm.comp_state(&s, false) == one.comp_state(&fresh, false),
        "compressed buffers"
    );
}

/// **A layer-major turn that does not extend the session resets it first**, as the chunk-major
/// one does ([`batched_non_extending_turn_resets`]).
#[test]
fn layer_major_non_extending_turn_resets() {
    let (lm, _) = canonical_layer_major(7);
    let (long, short) = (ids(300, 13), ids(40, 14));
    let mut s = None;
    lm.generate(&long, 1, &mut s, None).expect("long turn");
    let got = lm.generate(&short, 1, &mut s, None).expect("short turn");
    assert_eq!(got.n_cached, 0, "a turn sharing nothing reused the session");
    let mut fresh = None;
    let want = lm.generate(&short, 1, &mut fresh, None).expect("fresh");
    assert!(got.logits == want.logits, "logits");
    assert!(
        lm.comp_state(&s, true) == lm.comp_state(&fresh, true),
        "the reset left a ring row dirty"
    );
}

/// **A layer-major prefill cancelled between two layers leaves nothing the next request
/// reuses.** The cancel lands part-way through the canonical layout's layer 1 (a CSA layer): every
/// chunk has run layer 0, two chunks layer 1, none the layers after. So layer 1's compressed state
/// has advanced past the first turn and layers 2 and 3's have not — asserted, since that half-
/// advanced state is what must not be reused. The two guards of
/// [`batched_prefill_cancelled_between_chunks_is_not_reused`] hold here too, each asserted: the
/// session records no prompt, and `written_end` is past the whole suffix (raised before each
/// chunk's first span). The next request re-prefills from position 0, equal to a fresh session:
/// logits and every compressed buffer. (The ring zeroing that reset does is not what this shows:
/// a prefill from position 0 rewrites a ring before reading it, and the test that sees the zeroing
/// is [`layer_major_non_extending_turn_resets`].)
#[test]
fn layer_major_prefill_cancelled_between_layers_is_not_reused() {
    let chunk = 7;
    let (lm, _) = canonical_layer_major(chunk);
    let prompt = ids(80, 15);
    // Longer than what the cancelled run wrote (its first two chunks reach position 43), so a
    // fresh prefill rewrites every compressed row that run left, and the buffers compare whole.
    let then = [&prompt[..30], &ids(20, 16)[..]].concat();
    let mut s = None;
    lm.generate(&prompt[..30], 1, &mut s, None)
        .expect("first turn");
    let after_first: Vec<Vec<Vec<u8>>> = (0..lm.cfg.n_layer)
        .map(|l| lm.layer_comp_state(&s, l, false))
        .collect();
    // The suffix is positions 30..79 in chunks of 7: 7 chunks a layer.
    let chunks = (prompt.len() - 1 - 30).div_ceil(chunk);
    let req = crate::sampling::RequestCtx::new(Default::default()).abort_at_poll(chunks + 2);
    let err = match lm.generate(&prompt, 1, &mut s, Some(&req)) {
        Ok(_) => panic!("the prefill was not cancelled"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("cancelled during prompt prefill"),
        "{err}"
    );
    let ratios: Vec<usize> = (0..lm.cfg.n_layer)
        .map(|l| lm.cfg.layer_compress_ratio(l))
        .collect();
    assert_eq!(ratios, [0, 4, 128, 4, 0], "the canonical layout moved");
    assert!(
        lm.layer_comp_state(&s, 1, false) != after_first[1],
        "layer 1 did not advance before the cancel"
    );
    for l in [2, 3] {
        assert!(
            lm.layer_comp_state(&s, l, false) == after_first[l],
            "layer {l} advanced: the cancel did not land between layers 1 and 2"
        );
    }
    let kv = s.as_ref().expect("a session");
    assert!(
        kv.cached.is_empty(),
        "the cancelled turn left the session recording {} tokens",
        kv.cached.len()
    );
    assert!(
        kv.written_end >= prompt.len() - 1,
        "the cancelled turn's chunks ran without raising written_end ({})",
        kv.written_end
    );
    let after = lm
        .generate(&then, 1, &mut s, None)
        .expect("turn after the cancel");
    assert_eq!(
        after.n_cached, 0,
        "the turn after the cancel reused the session"
    );
    let mut fresh = None;
    let want = lm.generate(&then, 1, &mut fresh, None).expect("fresh");
    assert!(after.logits == want.logits, "logits");
    assert!(
        lm.comp_state(&s, false) == lm.comp_state(&fresh, false),
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

/// **The host-arena layer-major gate follows the first chunk's height, so the prefill's start
/// position** (`crate::seam::host_layer_major` over `prefill_chunk_rows`). On a file with a CSA
/// layer at the default 1024-row chunk, a prefill from position 0 qualifies (whatever the
/// prompt's length: later chunks are not consulted), and so does one starting at 64k or 160k;
/// the first chunk falls under `HOST_LAYER_MAJOR_MIN_ROWS` (256) rows between start positions
/// 261 891 and 261 892, worked by hand: 256 rows at 261 891 hold 256 x (262 147 / 4) x 4 =
/// 67 108 864 bytes, exactly `DSV4_PREFILL_SCORE_BYTES`, and at 261 892 256 x 65 537 x 4 =
/// 67 109 888, past it. An extending turn starting later prefills chunk-major.
#[test]
fn layer_major_gate_follows_the_first_chunk() {
    let csa = Model::open_with(
        "dsv4-pf-gate",
        &session_model(&dsv4_dims()),
        EngineConfig::default(),
    );
    let min = crate::seam::HOST_LAYER_MAJOR_MIN_ROWS;
    assert_eq!(min, 256, "the boundary below is for 256 rows");
    let gate = |start: usize| {
        crate::seam::host_layer_major(&csa.cfg, prefill_chunk_rows(&csa.cfg, start, 1024), false)
    };
    for start in [0, 64 << 10, 160 << 10, 261_891] {
        assert!(gate(start), "a prefill starting at {start} lost the order");
    }
    assert_eq!(prefill_chunk_rows(&csa.cfg, 261_891, 1024), min);
    for start in [261_892, 400 << 10, 1 << 20] {
        assert!(!gate(start), "a prefill starting at {start} kept the order");
    }
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
/// sibling that `Op::MoeFfn` feeds int8-quantized activations at every row count (`infr_cpu`'s
/// `expert_acts_kind`), each row quantized on its own.
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

/// **With quantized expert banks, chunked prefill equals one position per graph**, bit for bit,
/// in the production kernels and in reference mode. The Q8_0 banks take int8 activations at
/// every row count, each row quantized on its own, and the expert kernels give a row the same
/// bits in a bucket of any size, so every chunk height gives the same logits, indexer
/// selections, compressed state and decode as one position per graph (until 2026-11 one row took
/// f32 activations, and chunked prefill sat 1.71e-4 from it in the logits). Reference mode keeps
/// every row count on f32 activations, and there too chunked and one-position prefill agree; the
/// production one-position run DIFFERS from the reference one, which shows one row really takes
/// the int8 path.
#[test]
fn quantized_banks_chunked_prefill_is_exact_in_both_modes() {
    let m = quantized_banks();
    let prompt = ids(300, 11);
    let one = observe(&open_quantized(&m, 1, false), &prompt, 1, false);
    for k in CHUNKS {
        let c = observe(&open_quantized(&m, k, false), &prompt, k, false);
        let d = report(
            &format!("chunk {k} against one position per graph"),
            &c,
            &one,
        );
        assert!(d.is_empty(), "{}", d.join("\n"));
    }
    let ref_one = observe(&open_quantized(&m, 1, true), &prompt, 1, false);
    let ref_chunked = observe(&open_quantized(&m, 37, true), &prompt, 37, false);
    let d = report("reference mode, chunk 37", &ref_chunked, &ref_one);
    assert!(d.is_empty(), "{}", d.join("\n"));
    assert!(
        one.logits != ref_one.logits,
        "one position per graph gave the reference logits: no int8 activation ran at one row"
    );
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
/// BF16 routers take an f32 dot. So, as with [`quantized_banks`], nothing here may differ: the
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
    // And layer-major (`paging.layer_major` forced), in one segment: the same ops on the same
    // rows in another order, so the same bits.
    for k in [37, SESSION_CTX] {
        let mut ec = EngineConfig::default();
        ec.device.ubatch = Some(k);
        ec.device.ubatch_specified = true;
        ec.paging.layer_major = Some(true);
        let lm = Model::open_with(&format!("dsv4-pf-shipped-lm-{k}"), &m, ec);
        let got = observe_as(&lm, &prompt, k, Some(usize::MAX), false);
        diffs.extend(report(
            &format!("shipped dtypes, layer-major chunk {k}"),
            &got,
            &want,
        ));
    }
    assert!(diffs.is_empty(), "{}", diffs.join("\n"));
}

/// **With quantized expert banks, a layer-major prefill equals the chunk-major one at the same
/// chunk height bit for bit**, int8 activations included: the reorder hands every op the same
/// rows, and each row's activations are quantized on their own. So the layer-major path equals
/// one position per graph as chunk-major does (the test above), in both modes.
#[test]
fn quantized_banks_layer_major_equals_chunk_major() {
    let m = quantized_banks();
    let prompt = ids(300, 11);
    let open_lm = |reference: bool| {
        let mut ec = EngineConfig::default();
        ec.device.ubatch = Some(37);
        ec.device.ubatch_specified = true;
        ec.kernels.cpu.reference = reference;
        ec.paging.layer_major = Some(true);
        Model::open_with(&format!("dsv4-pf-q8-lm-{reference}"), &m, ec)
    };
    let chunked = observe(&open_quantized(&m, 37, false), &prompt, 37, false);
    let lm = observe_as(&open_lm(false), &prompt, 37, Some(usize::MAX), false);
    let d = report("layer-major against chunk-major, chunk 37", &lm, &chunked);
    assert!(d.is_empty(), "{}", d.join("\n"));
    let ref_one = observe(&open_quantized(&m, 1, true), &prompt, 1, false);
    let ref_lm = observe_as(&open_lm(true), &prompt, 37, Some(usize::MAX), false);
    let d = report("reference mode, layer-major chunk 37", &ref_lm, &ref_one);
    assert!(d.is_empty(), "{}", d.join("\n"));
}
