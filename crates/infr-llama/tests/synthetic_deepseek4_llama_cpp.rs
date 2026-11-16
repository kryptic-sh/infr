//! llama.cpp as an external oracle for the synthetic `deepseek4` (V4) model, with no download: the
//! real V4-Flash file does not load in llama.cpp on a 64 GB machine (`docs/plan/09-deepseek-v4-
//! compressed.md` § 4), but llama.cpp runs the fixture `dsv4_fixture` builds in seconds.
//!
//! Each [`OracleCase`] is one synthetic file. `tests/fixtures/synthetic_dsv4_<case>_llama_cpp.txt`
//! holds the logits llama.cpp computed for it at every length in its [`OracleCase::prompt_lens`],
//! each a prefix of one fixed id sequence per case ([`oracle_prompt`]). For the cases with compressed layers,
//! `..._llama_cpp_sums.txt` holds llama.cpp's per-tensor sums at a few positions of a
//! token-by-token run, to bisect a mismatch layer by layer. Both headers carry the byte length and
//! FNV-1a digest of the GGUF they were made from, and
//! [`synthetic_dsv4_llama_cpp_fixtures_match_their_gguf`] fails as soon as the fixture builder
//! writes other bytes: change the fixture, regenerate.
//!
//! What llama.cpp (at the commit the headers name) demands of a V4 file, and what the cases
//! therefore do differently from the plain `dsv4_fixture` model:
//!
//! - `hyper_connection.count` must be 4 (`build_hc_mixes` asserts it), so every case uses
//!   [`LLAMA_CPP_HC_MULT`].
//! - The indexer's key width must be at least 64 ([`ORACLE_INDEXER_WIDTH`]).
//! - At least one layer must have compress ratio 4. Without one the lightning-indexer cache has no
//!   layers, so `llama_kv_cache`'s constructor collects a head width of 0 for the Hadamard rotation
//!   it forces on that cache, doubles `n_rot_k` until the `u32` wraps to 0 and takes a remainder
//!   by it: the process dies with an integer division by zero. A case with no ratio-4 layer
//!   therefore gets a NULL tail layer ([`OracleCase::null_tail`]): ratio 4 in llama.cpp's file,
//!   ratio 0 in infr's, and an all-zero `attn_output_b` in both, so its attention adds exactly
//!   zero on either side and the two files compute the same logits.
//!
//! Every stored point where the indexer's top-k drops blocks is checked, at generation, to keep
//! its selection under noise on the indexer weights ([`indexer_robustness`]): a near-tie in the
//! top-k would flip under the f16 rounding either engine applies to the indexer keys, and a point
//! that flips under rounding cannot judge infr's indexer.
//!
//! ```text
//! cargo test -p infr-llama --test synthetic_deepseek4_llama_cpp
//! INFR_LLAMA_DEBUG=<llama.cpp>/build/bin/Release/llama-debug.exe [INFR_LLAMA_FIXTURE_OUT=<dir>] \
//!   cargo test -p infr-llama --test synthetic_deepseek4_llama_cpp -- --ignored \
//!   synthetic_dsv4_regenerate_llama_cpp_fixtures
//! INFR_DSV4_GGUF_DIR=<dir> cargo test -p infr-llama --test synthetic_deepseek4_llama_cpp -- \
//!   --ignored synthetic_dsv4_write_oracle_ggufs   # the files, for llama-eval-callback by hand
//! ```

mod dsv4_fixture;

use dsv4_fixture::{dsv4_dims, dsv4_indexer, dsv4_model, dsv4_token_text, Dsv4Dims, IndexerDims};
use infr_testkit::gguf::{fnv1a64, splitmix64, Fill, Meta, SyntheticModel, TempGguf};
use infr_testkit::llama_cpp::{
    llama_cpp_version, llama_debug_logits, run_headless, write_or_compare_fixture,
};

/// One synthetic V4 file run through both engines.
struct OracleCase {
    /// The fixture files' and the temp GGUF's name.
    name: &'static str,
    /// `compress_ratios`, before any null tail.
    ratios: &'static [usize],
    hash_layer_count: usize,
    /// `attention.indexer.top_k`.
    top_k: usize,
    /// The prompt lengths the case is scored at, all prefixes of [`oracle_prompt`].
    prompt_lens: &'static [usize],
    /// Which id sequence the case's prompts are prefixes of ([`oracle_prompt`]). Chosen so that no
    /// stored point sits on an indexer near-tie (the module doc); 0 for a case the indexer does not
    /// select for.
    prompt_salt: u64,
}

impl OracleCase {
    /// Whether the case needs the null ratio-4 tail layer llama.cpp's loader demands (module doc):
    /// exactly when no layer of its own has ratio 4.
    fn null_tail(&self) -> bool {
        !self.ratios.contains(&4)
    }

    /// Whether infr generates this case today. The compressed tiers are refused until slices S6
    /// (ratio 128) and S7 (ratio 4) land; a case's infr file has only its own ratios, because the
    /// null tail is ratio 0 there.
    fn infr_generates(&self) -> bool {
        self.ratios.iter().all(|&r| r == 0)
    }

    /// Whether the indexer's top-k drops blocks anywhere in a `len`-token prompt and that changes
    /// the logits: a ratio-4 layer of the case's own (the null tail's output is zeroed), and more
    /// blocks visible at the last position than `top_k` keeps.
    fn top_k_bites(&self, len: usize) -> bool {
        self.ratios.contains(&4) && len / CSA_RATIO > self.top_k
    }

    /// The length of the token-by-token run the sums come from: [`SUMS_LEN`], or the case's
    /// longest prompt when that is shorter, so every sum comes from a prefix the robustness check
    /// covered.
    fn sums_len(&self) -> usize {
        SUMS_LEN.min(*self.prompt_lens.last().expect("a case has prompts"))
    }

    /// The positions of the sums run that are kept: [`SUM_POSITIONS`] inside the run, and its last.
    fn sum_positions(&self) -> Vec<usize> {
        let last = self.sums_len() - 1;
        let mut p: Vec<usize> = SUM_POSITIONS.into_iter().filter(|&p| p < last).collect();
        p.push(last);
        p
    }

    fn fixture_path(&self, suffix: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "tests/fixtures/synthetic_dsv4_{}_llama_cpp{suffix}.txt",
            self.name
        ))
    }
}

/// Every case. `ratio0` is the tier infr already computes, compared in CI; `mixed` is the canonical
/// fixture's layer pattern; `csa` and `hca` put every layer on one compressed tier. Hash routing
/// covers the first layers of each, as on the real file.
///
/// Only `csa` lets the indexer's top-k drop blocks (`top_k` [`BITING_TOP_K`]), and only up to 131
/// tokens: past that some row of some layer always sat on a near-tie (module doc). Measured with
/// llama.cpp over 40 prompt sequences: none kept its selection under the noise at 300 tokens, and
/// 30 sequences none at 520, while the first one tried held at 30, 70 and 131. The other compressed
/// cases keep every visible block (`top_k` [`REAL_TOP_K`], the real file's), which is what lets
/// them reach 520 tokens: past two blocks of 128 and llama.cpp's ubatch boundary.
const CASES: [OracleCase; 4] = [
    OracleCase {
        name: "ratio0",
        ratios: &[0, 0, 0, 0, 0],
        hash_layer_count: 2,
        top_k: REAL_TOP_K,
        prompt_lens: &PROMPT_LENS,
        prompt_salt: 0,
    },
    OracleCase {
        name: "mixed",
        ratios: &[0, 4, 128, 4, 0],
        hash_layer_count: 2,
        top_k: REAL_TOP_K,
        prompt_lens: &PROMPT_LENS,
        prompt_salt: 0,
    },
    OracleCase {
        name: "csa",
        ratios: &[4, 4, 4, 4],
        hash_layer_count: 1,
        top_k: BITING_TOP_K,
        prompt_lens: &[7, 30, 70, 131],
        prompt_salt: 0,
    },
    OracleCase {
        name: "hca",
        ratios: &[128, 128, 128, 128],
        hash_layer_count: 1,
        top_k: REAL_TOP_K,
        prompt_lens: &PROMPT_LENS,
        prompt_salt: 0,
    },
];

/// The real file's `attention.indexer.top_k`: more blocks than any oracle prompt has, so the
/// selection keeps every visible block.
const REAL_TOP_K: usize = 512;

/// The canonical fixture's `top_k` (`dsv4_indexer`), which the indexer exceeds from the 24th token
/// on.
const BITING_TOP_K: usize = 5;

/// The only hyper-connection count llama.cpp's `build_hc_mixes` accepts.
const LLAMA_CPP_HC_MULT: usize = 4;

/// The indexer key width of every oracle file: the real file's. llama.cpp forces a Hadamard
/// rotation on the indexer's cache and builds the matrices only from 64 up, so the canonical
/// fixture's narrower width does not load there (`dsv4_indexer` says why that one is narrow).
const ORACLE_INDEXER_WIDTH: usize = 128;

/// The CSA block length (`DSV4_CSA_RATIO` in `deepseek4.cpp`).
const CSA_RATIO: usize = 4;

/// `context_length`, which is also the `-c` llama.cpp runs with: room for the longest prompt.
const ORACLE_CTX: usize = 1024;

/// The prompt lengths of the cases that reach 520 tokens: below and past the sliding window,
/// across several blocks of 128, and past llama.cpp's default 512-token ubatch, so its compressor
/// state crosses a ubatch boundary as well.
const PROMPT_LENS: [usize; 6] = [7, 30, 70, 131, 300, 520];

/// The longest token-by-token run the per-tensor sums come from ([`OracleCase::sums_len`]), and the
/// positions kept: the first ratio-4 commit and the step after it, the last position before a
/// biting `top_k` drops a block and the first where it does, the first ratio-128 commit and the
/// step after it, and (added per case) the run's last position.
const SUMS_LEN: usize = 300;
const SUM_POSITIONS: [usize; 6] = [3, 4, 22, 23, 127, 128];

/// The tensors whose sums are kept, as `llama-debug --tensor-filter` prefixes: the compressors
/// (`build_overlap_compressed_kv_from_state` and `build_hca_compressed_kv_from_state` name every
/// intermediate after their `name` argument), the indexer's selection, each attention tier's
/// output, and every layer's attention and FFN outputs to locate the first layer that differs.
/// `result_norm` comes last in every step's graph, so it also ends a position in the log.
const SUM_TENSORS: [&str; 10] = [
    "csa_state_compress",
    "lid_state_compress",
    "hca_state_compress",
    "lid_top_k",
    "attn_csa_lid",
    "attn_hca",
    "attn_raw",
    "attn_out",
    "ffn_out",
    "result_norm",
];

/// The indexer weights the robustness check perturbs: its query projection and its compressor's
/// key projection, on every ratio-4 layer.
const INDEXER_WEIGHTS: [&str; 2] = ["indexer.attn_q_b.weight", "indexer_compressor_kv.weight"];

/// The relative noise the stored points must keep their top-k selection under: about twice the
/// f16 rounding (2^-11 relative) that either engine applies to the indexer's keys.
const INDEXER_NOISE: f32 = 1e-3;

/// How many noise draws a stored point must survive.
const INDEXER_NOISE_SEEDS: u64 = 3;

/// The largest logit move the noise may cause at a stored point. A kept selection moves nothing
/// at all (the scores only choose which blocks the attention sees), so any move is a flip.
const INDEXER_NOISE_MAX: f32 = 1e-4;

/// A perturbation large enough to change the selection, and the logit move it must cause at every
/// point where the top-k drops blocks: proof that the indexer is load-bearing there.
const INDEXER_REDRAW_NOISE: f32 = 0.5;
const INDEXER_REDRAW_MIN: f32 = 1e-2;

/// The rope-scaling keys the real file declares (type, factor, original context), at values that
/// make the YaRN ramp cover the fixture's fast pairs: with a 16-dim rope at compress base 40000
/// and an original context of 512, every pair but the first is scaled at least partly, so the
/// compressed tiers' rope differs from an unscaled one within a few positions. The real file's
/// own original context (65536) would leave the first five of eight pairs unscaled
/// (`synthetic_deepseek4_rope_scaling_keys_are_read` has those divisors). Ratio-0 layers must
/// ignore all of it.
fn yarn_keys() -> [(&'static str, Meta); 3] {
    [
        ("rope.scaling.type", Meta::Str("yarn".into())),
        ("rope.scaling.factor", Meta::F32(16.0)),
        ("rope.scaling.original_context_length", Meta::U32(512)),
    ]
}

/// The dims of `case`'s file for llama.cpp (`for_llama`) or for infr: identical except the null
/// tail's ratio.
fn oracle_dims(case: &OracleCase, for_llama: bool) -> Dsv4Dims {
    let mut ratios = case.ratios.to_vec();
    if case.null_tail() {
        ratios.push(if for_llama { 4 } else { 0 });
    }
    Dsv4Dims {
        // Per-layer and all different, as in `dsv4_dims`, at this case's layer count.
        swiglu_clamp_exp: (0..ratios.len()).map(|l| 7.0 + 0.5 * l as f32).collect(),
        compress_ratios: ratios,
        hash_layer_count: case.hash_layer_count,
        hc_mult: LLAMA_CPP_HC_MULT,
        indexer: IndexerDims {
            head_size: ORACLE_INDEXER_WIDTH,
            top_k: case.top_k,
            ..dsv4_indexer()
        },
        ..dsv4_dims()
    }
}

/// `case`'s GGUF for llama.cpp (`for_llama`) or for infr.
fn oracle_model(case: &OracleCase, for_llama: bool) -> SyntheticModel {
    let d = oracle_dims(case, for_llama);
    let mut m = dsv4_model(&d);
    let ctx_key = "deepseek4.context_length";
    let ctx = m
        .meta
        .iter_mut()
        .find(|(k, _)| k == ctx_key)
        .unwrap_or_else(|| panic!("the fixture declares no {ctx_key}"));
    ctx.1 = Meta::U32(ORACLE_CTX as u32);
    for (k, v) in yarn_keys() {
        m.meta.push((format!("deepseek4.{k}"), v));
    }
    m.meta.sort_by(|a, b| a.0.cmp(&b.0));
    if case.null_tail() {
        let name = format!("blk.{}.attn_output_b.weight", d.n_layer() - 1);
        let t = m
            .tensors
            .iter_mut()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("no {name} in the fixture"));
        t.fill = Fill::Exact(vec![0.0; t.numel()]);
    }
    m
}

/// A value in `[-1, 1)` from `seed` and `i`, the testkit's fill rule.
fn uniform(seed: u64, i: usize) -> f32 {
    let h = splitmix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    ((h >> 40) as f32) / f32::from(1u16 << 12) / f32::from(1u16 << 12) * 2.0 - 1.0
}

/// `model` with every [`INDEXER_WEIGHTS`] value scaled by `1 + amp·u`, `u` uniform in `[-1, 1)`
/// and drawn from `seed` and the tensor's name.
fn with_indexer_noise(model: &SyntheticModel, amp: f32, seed: u64) -> SyntheticModel {
    let mut n = 0;
    let tensors = model
        .tensors
        .iter()
        .map(|t| {
            let mut t = t.clone();
            if INDEXER_WEIGHTS.iter().any(|w| t.name.ends_with(w)) {
                let s = seed ^ fnv1a64(t.name.as_bytes());
                let v = t.values();
                t.fill = Fill::Exact(
                    v.iter()
                        .enumerate()
                        .map(|(i, x)| x * (1.0 + amp * uniform(s, i)))
                        .collect(),
                );
                n += 1;
            }
            t
        })
        .collect();
    assert!(n > 0, "no indexer weight to perturb");
    SyntheticModel {
        meta: model.meta.clone(),
        tensors,
    }
}

/// The ids `case`'s prompts are prefixes of. Drawn from the vocabulary minus the double quote and
/// the backslash, which the command line and llama.cpp's prompt escaping would rewrite, and minus
/// the EOS id.
fn oracle_prompt(case: &OracleCase, len: usize) -> Vec<u32> {
    let vocab = dsv4_dims().vocab as u32;
    let skip = |id: u32| matches!(dsv4_token_text(id), '"' | '\\') || id == 2;
    let ids: Vec<u32> = (0..vocab).filter(|&id| !skip(id)).collect();
    (0..len as u64)
        .map(|i| {
            ids[(splitmix64(i.wrapping_add(case.prompt_salt << 32)) % ids.len() as u64) as usize]
        })
        .collect()
}

/// The byte length and FNV-1a digest of `model`'s GGUF, as a fixture header records them.
fn gguf_identity(model: &SyntheticModel) -> String {
    let bytes = model.to_gguf_bytes();
    format!("{} bytes, fnv1a64 {:#018x}", bytes.len(), fnv1a64(&bytes))
}

/// A fixture file's `# gguf:` identity.
fn read_identity(path: &std::path::Path) -> String {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    text.lines()
        .find_map(|l| l.strip_prefix("# gguf: "))
        .unwrap_or_else(|| panic!("{} names no `# gguf:` identity", path.display()))
        .to_string()
}

/// One llama.cpp run in a logits fixture: the prompt ids and the last position's logits.
struct OracleRun {
    tokens: Vec<u32>,
    logits: Vec<f32>,
}

/// A logits fixture's runs: `# tokens:` lists a run's prompt ids, and the value lines after it are
/// its logits.
fn read_logits_fixture(case: &OracleCase) -> Vec<OracleRun> {
    let path = case.fixture_path("");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut runs: Vec<OracleRun> = Vec::new();
    for line in text.lines() {
        if let Some(ids) = line.strip_prefix("# tokens:") {
            runs.push(OracleRun {
                tokens: ids
                    .split_whitespace()
                    .map(|t| t.parse().expect("a token id"))
                    .collect(),
                logits: Vec::new(),
            });
        } else if !line.starts_with('#') && !line.trim().is_empty() {
            runs.last_mut()
                .unwrap_or_else(|| panic!("{}: a logit before any `# tokens:`", path.display()))
                .logits
                .push(line.trim().parse().expect("a logit"));
        }
    }
    runs
}

/// One position of a sums fixture: its 0-based position and its `<tensor header> sum <sum>` lines.
type SumsPosition = (usize, Vec<String>);

/// The data body of a sums fixture: `# position: <p>` before each position's lines.
fn format_sums(positions: &[SumsPosition]) -> String {
    let mut text = String::new();
    for (pos, lines) in positions {
        text.push_str(&format!("# position: {pos}\n"));
        for l in lines {
            text.push_str(l);
            text.push('\n');
        }
    }
    text
}

/// A sums fixture's positions, the inverse of [`format_sums`] over the file's non-header lines.
fn read_sums_fixture(path: &std::path::Path) -> Vec<SumsPosition> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut positions: Vec<SumsPosition> = Vec::new();
    for line in text.lines() {
        if let Some(p) = line.strip_prefix("# position: ") {
            positions.push((p.parse().expect("a position"), Vec::new()));
        } else if !line.starts_with('#') && !line.trim().is_empty() {
            positions
                .last_mut()
                .unwrap_or_else(|| panic!("{}: a sum before any `# position:`", path.display()))
                .1
                .push(line.to_string());
        }
    }
    positions
}

/// A sums line's tensor name (with its `-<il>` layer suffix and any view suffix) and its sum.
fn parse_sum_line(line: &str) -> (&str, f64) {
    let (head, sum) = line
        .rsplit_once(" sum ")
        .unwrap_or_else(|| panic!("no ` sum ` in {line:?}"));
    let name = head
        .split_once(" = ")
        .unwrap_or_else(|| panic!("no tensor name in {line:?}"))
        .0;
    (
        name,
        sum.parse()
            .unwrap_or_else(|e| panic!("sum {sum:?} in {line:?}: {e}")),
    )
}

fn load(tmp: &TempGguf) -> infr_llama::SeamModel {
    infr_llama::SeamModel::load_with(
        tmp.path(),
        None,
        std::sync::Arc::new(infr_llama::EngineConfig::default()),
    )
    .expect("synthetic model load")
}

fn softmax(v: &[f32]) -> Vec<f64> {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = v.iter().map(|&x| f64::from(x - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|x| x / sum).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit vector lengths");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |m, d| {
            if m.is_nan() || d.is_nan() {
                f32::NAN
            } else {
                m.max(d)
            }
        })
}

/// How far infr's logits are from llama.cpp's on one run.
struct Score {
    max_abs: f32,
    prob_cosine: f64,
    top1_agrees: bool,
}

fn score(ours: &[f32], theirs: &[f32]) -> Score {
    let (p, q) = (softmax(ours), softmax(theirs));
    let dot: f64 = p.iter().zip(&q).map(|(a, b)| a * b).sum();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    Score {
        max_abs: max_abs_diff(ours, theirs),
        prob_cosine: dot / (norm(&p) * norm(&q)),
        top1_agrees: argmax(ours) == argmax(theirs),
    }
}

impl std::fmt::Display for Score {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max |logit diff| {:e}, probability cosine {}, top-1 {}",
            self.max_abs,
            self.prob_cosine,
            if self.top1_agrees {
                "agrees"
            } else {
                "DIFFERS"
            }
        )
    }
}

/// The bound on infr's CPU logits against llama.cpp's on a case infr generates. The scores
/// measured when the data was generated are recorded, as generated lines, in the case's fixture
/// header (`# infr CPU at ...`); docs/plan/09-deepseek-v4-compressed.md § S5a records them with
/// the gaps a window off by one and dropped attention sinks give, which this bound separates.
/// Probability cosine is printed, not asserted: it stays close to 1 under both breaks.
const ORACLE_MAX_ABS: f32 = 2e-3;

/// infr's CPU prefill of `case` against each of `runs` (one fixture's), returning one score per run.
fn infr_scores(case: &OracleCase, runs: &[OracleRun]) -> Vec<(usize, Score)> {
    let tmp = TempGguf::write(
        &format!("dsv4-oracle-{}", case.name),
        &oracle_model(case, false),
    );
    let model = load(&tmp);
    let vocab = model.config().vocab;
    runs.iter()
        .map(|run| {
            let len = run.tokens.len();
            assert_eq!(
                run.tokens,
                oracle_prompt(case, len),
                "the fixture is for another prompt"
            );
            assert_eq!(
                run.logits.len(),
                vocab,
                "the fixture is for another vocabulary"
            );
            let ours = model.prefill_logits_cpu(&run.tokens).expect("cpu prefill");
            (len, score(&ours, &run.logits))
        })
        .collect()
}

/// infr's CPU prefill of `case` against every run in its checked-in llama.cpp fixture.
fn cpu_matches_llama_cpp(case: &OracleCase) -> Vec<(usize, Score)> {
    let runs = read_logits_fixture(case);
    let lens: Vec<usize> = runs.iter().map(|r| r.tokens.len()).collect();
    assert_eq!(
        lens, case.prompt_lens,
        "the {} fixture is for other lengths",
        case.name
    );
    let scores = infr_scores(case, &runs);
    for (len, s) in &scores {
        println!("synthetic dsv4 {} at {len} tokens: {s}", case.name);
    }
    scores
}

/// **The external oracle on the tier infr generates, in CI.** infr's CPU prefill of the ratio-0
/// case (hash- and bias-routed layers, hyper-connections at the real stream count, sinks, the
/// grouped output projection, the de-rope, YaRN keys that ratio-0 layers must ignore) against
/// llama.cpp's logits at every prompt length, below and past the sliding window.
#[test]
fn synthetic_dsv4_ratio0_cpu_matches_llama_cpp() {
    let case = &CASES[0];
    assert!(case.infr_generates() && case.null_tail());
    for (len, s) in cpu_matches_llama_cpp(case) {
        assert!(
            s.max_abs < ORACLE_MAX_ABS,
            "infr's CPU misses llama.cpp by {} at {len} tokens on the synthetic ratio-0 V4",
            s.max_abs
        );
    }
}

/// **The compressed cases are refused until S6 and S7 lift the refusal, and then this test must
/// become a comparison.** Each compressed case's llama.cpp data is checked in already; when infr
/// generates a case, this test fails for it, and the fix is to score it with
/// [`cpu_matches_llama_cpp`] as [`synthetic_dsv4_ratio0_cpu_matches_llama_cpp`] does.
#[test]
fn synthetic_dsv4_compressed_cases_are_still_refused() {
    for case in CASES.iter().filter(|c| !c.infr_generates()) {
        let tmp = TempGguf::write(
            &format!("dsv4-refused-{}", case.name),
            &oracle_model(case, false),
        );
        let err = match load(&tmp).prefill_logits_cpu(&oracle_prompt(case, case.prompt_lens[0])) {
            Ok(_) => panic!(
                "infr now generates the {} case: score it against its llama.cpp fixture",
                case.name
            ),
            Err(e) => format!("{e:#}"),
        };
        assert!(
            err.contains("only the ratio-0 tier (pure sliding window) generates"),
            "the {} case failed some other way: {err}",
            case.name
        );
    }
}

/// **The oracle files' YaRN keys scale the compressed tiers' rope unevenly across the pairs**, as
/// [`yarn_keys`] intends: infr's own divisors for them leave the first pair unscaled, scale every
/// other pair, divide the slowest by the full factor, and put at least one pair on the ramp
/// between. A YaRN error that only touched the ramp would otherwise have nothing to show on.
#[test]
fn synthetic_dsv4_oracle_yarn_ramp_varies() {
    let tmp = TempGguf::write("dsv4-oracle-yarn", &oracle_model(&CASES[0], false));
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let d = cfg
        .dsv4_compressed_rope_divisors()
        .expect("a deepseek4 config");
    println!("oracle compressed-tier rope divisors: {d:?}");
    let factor = cfg.rope_scaling_factor;
    assert!(cfg.rope_scaling_yarn, "the oracle files declare YaRN");
    assert_eq!(d[0], 1.0, "the fastest pair stays unscaled: {d:?}");
    assert!(
        d[1..].iter().all(|&x| x > 1.0),
        "a pair past the first is unscaled: {d:?}"
    );
    assert_eq!(*d.last().expect("rope pairs"), factor, "{d:?}");
    assert!(
        d.iter().any(|&x| x > 1.0 && x < factor),
        "no pair on the ramp: {d:?}"
    );
}

/// The command that regenerates every fixture this file reads.
const REGENERATE: &str = "INFR_LLAMA_DEBUG=<llama-debug> INFR_LLAMA_FIXTURE_OUT=<dir> cargo test \
                          -p infr-llama --test synthetic_deepseek4_llama_cpp -- --ignored \
                          synthetic_dsv4_regenerate_llama_cpp_fixtures";

/// **The checked-in llama.cpp data was made from the files the fixture builder writes today.** A
/// change to `dsv4_fixture`, to a case or to the testkit's GGUF writer that moves a single byte of
/// llama.cpp's file fails here, naming the case and the command that regenerates the data, rather
/// than leaving infr scored against data for another model.
#[test]
fn synthetic_dsv4_llama_cpp_fixtures_match_their_gguf() {
    for case in &CASES {
        let want = gguf_identity(&oracle_model(case, true));
        let mut files = vec![case.fixture_path("")];
        if !case.infr_generates() {
            files.push(case.fixture_path("_sums"));
        }
        for path in files {
            assert_eq!(
                read_identity(&path),
                want,
                "{} was made from another GGUF than the {} case builds now; regenerate it with \
                 `{REGENERATE}` and copy the files over the checked-in ones",
                path.display(),
                case.name
            );
        }
    }
}

/// **The checked-in sums files parse, and say what a bisect needs.** Each compressed case's file
/// holds exactly its [`OracleCase::sum_positions`], every line names a tensor and a number, every position has one
/// `attn_out` and one `ffn_out` per layer of llama.cpp's file, and [`format_sums`] writes the parsed
/// positions back to the file's own data lines, so the reader and the writer cannot drift.
#[test]
fn synthetic_dsv4_llama_cpp_sums_files_parse() {
    for case in CASES.iter().filter(|c| !c.infr_generates()) {
        let path = case.fixture_path("_sums");
        let positions = read_sums_fixture(&path);
        let got: Vec<usize> = positions.iter().map(|p| p.0).collect();
        assert_eq!(
            got,
            case.sum_positions(),
            "{}: other positions",
            path.display()
        );
        let n_layer = oracle_dims(case, true).n_layer();
        for (pos, lines) in &positions {
            let names: Vec<&str> = lines.iter().map(|l| parse_sum_line(l).0).collect();
            for kind in ["attn_out", "ffn_out"] {
                for il in 0..n_layer {
                    let want = format!("{kind}-{il}");
                    let n = names.iter().filter(|n| **n == want).count();
                    assert_eq!(n, 1, "{}: position {pos} has {n} {want}", path.display());
                }
            }
            assert_eq!(
                names.last(),
                Some(&"result_norm"),
                "{}: position {pos} does not end with result_norm",
                path.display()
            );
        }
        let text = std::fs::read_to_string(&path).expect("read the sums fixture");
        let body: String = text
            .lines()
            .filter(|l| !l.starts_with('#') || l.starts_with("# position:"))
            .map(|l| format!("{l}\n"))
            .collect();
        assert_eq!(
            format_sums(&positions),
            body,
            "{}: the reader and the writer disagree",
            path.display()
        );
    }
}

/// [`sums_by_position`] on a hand-written log in llama-debug's format: two one-token steps, with
/// value rows between headers and sums, and a log line that is neither.
#[test]
fn synthetic_dsv4_sums_log_splits_into_positions() {
    let log = "\
llama_model_loader: something else
common_debug_cb_eval:            attn_out-0 = (f32)    MUL_MAT(w{2, 1, 1, 1}, x{2, 1, 1, 1}}) = {2, 1, 1, 1}
    [
        [
            [      0.5000,       0.2500  ],
        ],
    ]
    sum = 0.750000
common_debug_cb_eval:           result_norm = (f32)        MUL(norm{2, 0, 1, 1}, w{2, 1, 1, 1}}) = {2, 0, 1, 1}
    sum = 0.000000
common_debug_cb_eval:            attn_out-0 = (f32)    MUL_MAT(w{2, 1, 1, 1}, x{2, 1, 1, 1}}) = {2, 1, 1, 1}
    sum = -inf
common_debug_cb_eval:           result_norm = (f32)        MUL(norm{2, 1, 1, 1}, w{2, 1, 1, 1}}) = {2, 1, 1, 1}
    sum = 1.250000
";
    let got = sums_by_position(log, 2);
    assert_eq!(
        got,
        vec![
            vec![
                "attn_out-0 = (f32)    MUL_MAT(w{2, 1, 1, 1}, x{2, 1, 1, 1}}) = {2, 1, 1, 1} sum 0.750000"
                    .to_string(),
                "result_norm = (f32)        MUL(norm{2, 0, 1, 1}, w{2, 1, 1, 1}}) = {2, 0, 1, 1} sum 0.000000"
                    .to_string(),
            ],
            vec![
                "attn_out-0 = (f32)    MUL_MAT(w{2, 1, 1, 1}, x{2, 1, 1, 1}}) = {2, 1, 1, 1} sum -inf"
                    .to_string(),
                "result_norm = (f32)        MUL(norm{2, 1, 1, 1}, w{2, 1, 1, 1}}) = {2, 1, 1, 1} sum 1.250000"
                    .to_string(),
            ],
        ]
    );
    assert_eq!(
        parse_sum_line(&got[1][0]),
        ("attn_out-0", f64::NEG_INFINITY)
    );
}

/// The text a prompt of `ids` is typed as.
fn prompt_text(ids: &[u32]) -> String {
    ids.iter().map(|&id| dsv4_token_text(id)).collect()
}

/// llama.cpp's logits for `model` (written to a temp file tagged `tag`) on `ids`, checking it
/// tokenized the prompt to exactly those ids.
fn llama_logits(exe: &str, tag: &str, model: &SyntheticModel, ids: &[u32]) -> Vec<f32> {
    let tmp = TempGguf::write(tag, model);
    let ctx = ORACLE_CTX.to_string();
    let (tokens, logits) = llama_debug_logits(
        exe,
        tmp.path(),
        &prompt_text(ids),
        &["-c".as_ref(), ctx.as_ref()],
    );
    assert_eq!(
        tokens,
        ids,
        "llama.cpp tokenized the {}-token prompt differently",
        ids.len()
    );
    assert_eq!(
        logits.len(),
        dsv4_dims().vocab,
        "llama.cpp returned another vocabulary"
    );
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "non-finite llama.cpp logits"
    );
    logits
}

/// How firmly llama.cpp's indexer holds its selection on one prompt: the largest logit move under
/// [`INDEXER_NOISE`] over [`INDEXER_NOISE_SEEDS`] draws (`noise`), and the move under
/// [`INDEXER_REDRAW_NOISE`] (`redraw`). A near-tie in any row's top-k shows as a non-zero `noise`.
struct IndexerRobustness {
    noise: f32,
    redraw: f32,
}

fn indexer_robustness(
    exe: &str,
    case: &OracleCase,
    model: &SyntheticModel,
    ids: &[u32],
    base: &[f32],
) -> IndexerRobustness {
    let moved = |amp: f32, seed: u64| {
        let noisy = with_indexer_noise(model, amp, seed);
        max_abs_diff(
            base,
            &llama_logits(exe, &format!("dsv4-noise-{}", case.name), &noisy, ids),
        )
    };
    IndexerRobustness {
        noise: (0..INDEXER_NOISE_SEEDS)
            .map(|seed| moved(INDEXER_NOISE, seed))
            .fold(0.0, f32::max),
        redraw: moved(INDEXER_REDRAW_NOISE, INDEXER_NOISE_SEEDS),
    }
}

/// Recompute every case's fixtures with llama.cpp (the binary `INFR_LLAMA_DEBUG` names): the
/// logits at each of the case's prompt lengths, and for the compressed cases the sums of
/// [`SUM_TENSORS`] at [`OracleCase::sum_positions`] of a run fed one token per ubatch (`-ub 1`, the
/// step infr's prefill takes). At every point where the top-k drops blocks it also re-measures
/// [`indexer_robustness`] and fails on a near-tie or an inert indexer, so a regeneration cannot
/// bring one back silently. The headers record the build, the commands, the GGUF's identity, the
/// measured robustness, and for a case infr generates infr's own scores against the new data.
///
/// The files are compared with the checked-in ones and written only into the directory
/// `INFR_LLAMA_FIXTURE_OUT` names (`write_or_compare_fixture`), so running every ignored test with
/// llama.cpp configured never rewrites the tree. Without `INFR_LLAMA_DEBUG` it returns at once: the
/// GPU gate runs every ignored test on machines with no llama.cpp build, and this is a tool, while
/// the checks that must run everywhere are the non-ignored tests above.
#[test]
#[ignore = "recomputes the llama.cpp fixtures: needs INFR_LLAMA_DEBUG (a llama-debug binary)"]
fn synthetic_dsv4_regenerate_llama_cpp_fixtures() {
    let Ok(exe) = std::env::var("INFR_LLAMA_DEBUG") else {
        eprintln!("skip: INFR_LLAMA_DEBUG not set");
        return;
    };
    let version = llama_cpp_version(&exe);
    let ctx = ORACLE_CTX.to_string();
    for case in &CASES {
        let model = oracle_model(case, true);
        let identity = gguf_identity(&model);
        let header = |what: &str, cmd: &str| {
            format!(
                "# {what} for the synthetic deepseek4 oracle case `{}` \
                 (synthetic_deepseek4_llama_cpp.rs).\n\
                 # llama.cpp {version}\n\
                 # Regenerate: {REGENERATE}\n\
                 # which runs: {cmd}\n\
                 # gguf: {identity}\n",
                case.name
            )
        };

        let mut runs = Vec::new();
        let mut notes = String::new();
        for &len in case.prompt_lens {
            let ids = oracle_prompt(case, len);
            let logits = llama_logits(&exe, &format!("dsv4-{}", case.name), &model, &ids);
            let lo = logits.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            notes.push_str(&format!("# logits at {len} tokens span [{lo}, {hi}]\n"));
            if case.ratios.contains(&4) && !case.top_k_bites(len) {
                notes.push_str(&format!(
                    "# indexer at {len} tokens: top_k {} keeps every visible block\n",
                    case.top_k
                ));
            }
            if case.top_k_bites(len) {
                let r = indexer_robustness(&exe, case, &model, &ids, &logits);
                println!(
                    "{} at {len} tokens: indexer noise moves {:e}, redraw moves {:e}",
                    case.name, r.noise, r.redraw
                );
                notes.push_str(&format!(
                    "# indexer at {len} tokens: ±{INDEXER_NOISE} relative noise on {}, \
                     {INDEXER_NOISE_SEEDS} draws, moves the logits by at most {:e}; \
                     ±{INDEXER_REDRAW_NOISE} moves them by {:e}\n",
                    INDEXER_WEIGHTS.join(" and "),
                    r.noise,
                    r.redraw
                ));
                assert!(
                    r.noise < INDEXER_NOISE_MAX,
                    "the {} case at {len} tokens sits on an indexer near-tie: ±{INDEXER_NOISE} \
                     noise moves llama.cpp's logits by {}; pick another prompt_salt",
                    case.name,
                    r.noise
                );
                assert!(
                    r.redraw > INDEXER_REDRAW_MIN,
                    "the indexer is inert in the {} case at {len} tokens: ±{INDEXER_REDRAW_NOISE} \
                     noise moves the logits by only {}",
                    case.name,
                    r.redraw
                );
            }
            runs.push(OracleRun {
                tokens: ids,
                logits,
            });
        }
        if case.infr_generates() {
            for (len, s) in infr_scores(case, &runs) {
                notes.push_str(&format!("# infr CPU at {len} tokens: {s}\n"));
            }
        }

        let mut text = header(
            "llama.cpp logits at the last position, one per vocabulary id, for each prompt",
            &format!(
                "llama-debug -m <synthetic gguf> -p <prompt ids as text> -c {ctx} --save-logits \
                 --logits-output-dir <dir>"
            ),
        );
        text.push_str(&notes);
        for run in &runs {
            let ids: Vec<String> = run.tokens.iter().map(u32::to_string).collect();
            text.push_str(&format!("# tokens: {}\n", ids.join(" ")));
            for v in &run.logits {
                text.push_str(&format!("{v}\n"));
            }
        }
        if let Some(written) = write_or_compare_fixture(&case.fixture_path(""), &text) {
            println!("wrote {}", written.display());
        }

        if case.infr_generates() {
            continue;
        }
        let tmp = TempGguf::write(&format!("dsv4-sums-{}", case.name), &model);
        let mut args: Vec<String> = vec![
            "-m".into(),
            tmp.path().to_str().expect("a UTF-8 temp path").into(),
            "-p".into(),
            prompt_text(&oracle_prompt(case, case.sums_len())),
            "-c".into(),
            ctx.clone(),
            "-ub".into(),
            "1".into(),
            "--no-warmup".into(),
        ];
        for t in SUM_TENSORS {
            args.push("--tensor-filter".into());
            args.push(t.into());
        }
        let log = run_headless(
            &exe,
            &args
                .iter()
                .map(|a| a.as_ref())
                .collect::<Vec<&std::ffi::OsStr>>(),
        );
        let filters: Vec<String> = SUM_TENSORS
            .iter()
            .flat_map(|t| ["--tensor-filter".to_string(), (*t).to_string()])
            .collect();
        let mut text = header(
            "llama.cpp per-tensor sums, one ubatch per token, at the positions below",
            &format!(
                "llama-debug -m <synthetic gguf> -p <the first {} prompt ids as text> -c \
                 {ctx} -ub 1 --no-warmup {}",
                case.sums_len(),
                filters.join(" ")
            ),
        );
        text.push_str(
            "# `# position: p` starts the lines for 0-based position p (the step that feeds the\n\
             # prompt's (p+1)-th id). Each line is `<tensor> = (<type>) <op>(<sources>) = {<shape>}\n\
             # sum <sum>`, in graph order, the name carrying the layer as `-<il>`. Sums are\n\
             # llama.cpp's `%f`: six decimals, absolute, so compare them with an absolute\n\
             # tolerance. `result_norm` has no rows (sum 0) except at the last position, the only\n\
             # one llama-debug asks logits for.\n",
        );
        if case.null_tail() {
            text.push_str(&format!(
                "# Layer {} is the null tail (synthetic_deepseek4_llama_cpp.rs): its csa_*/lid_*\n\
                 # tensors and its all-zero attn_out exist only in llama.cpp's file; infr's file\n\
                 # has a ratio-0 layer there.\n",
                oracle_dims(case, true).n_layer() - 1
            ));
        }
        let keep = case.sum_positions();
        let kept: Vec<SumsPosition> = sums_by_position(&log, case.sums_len())
            .into_iter()
            .enumerate()
            .filter(|(pos, _)| keep.contains(pos))
            .collect();
        text.push_str(&format_sums(&kept));
        if let Some(written) = write_or_compare_fixture(&case.fixture_path("_sums"), &text) {
            println!("wrote {}", written.display());
        }
    }
    println!("recomputed the deepseek4 llama.cpp fixtures ({version})");
}

/// Split a `llama-debug --tensor-filter` log of a one-token-per-ubatch run into one list per
/// position of `<tensor header> sum <sum>` lines. A position opens at the first tensor after the
/// previous one closed, and `result_norm` closes it; the count is checked against `n_positions`
/// rather than assumed.
fn sums_by_position(log: &str, n_positions: usize) -> Vec<Vec<String>> {
    const TAG: &str = "common_debug_cb_eval: ";
    let mut positions: Vec<Vec<String>> = Vec::new();
    let mut open = false;
    let mut header: Option<String> = None;
    for line in log.lines() {
        if let Some(h) = line.split_once(TAG).map(|(_, h)| h.trim().to_string()) {
            if !open {
                positions.push(Vec::new());
                open = true;
            }
            header = Some(h);
        } else if let Some(sum) = line.trim().strip_prefix("sum = ") {
            let h = header
                .take()
                .expect("a sum line follows its tensor's header line");
            let done = h.starts_with("result_norm");
            positions
                .last_mut()
                .expect("a position is open")
                .push(format!("{h} sum {sum}"));
            if done {
                open = false;
            }
        }
    }
    assert_eq!(
        positions.len(),
        n_positions,
        "the one-token-per-ubatch run printed another number of positions"
    );
    positions
}

/// Write every case's two GGUFs (`<case>-llama.gguf`, `<case>-infr.gguf`) into the directory
/// `INFR_DSV4_GGUF_DIR` names, for running `llama-eval-callback` or `llama-debug --verbose` on them
/// by hand when a comparison has to be bisected.
#[test]
#[ignore = "writes the oracle GGUFs: needs INFR_DSV4_GGUF_DIR (a directory)"]
fn synthetic_dsv4_write_oracle_ggufs() {
    let Ok(dir) = std::env::var("INFR_DSV4_GGUF_DIR") else {
        eprintln!("skip: INFR_DSV4_GGUF_DIR not set");
        return;
    };
    let dir = std::path::Path::new(&dir);
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    for case in &CASES {
        for (side, for_llama) in [("llama", true), ("infr", false)] {
            let path = dir.join(format!("{}-{side}.gguf", case.name));
            std::fs::write(&path, oracle_model(case, for_llama).to_gguf_bytes())
                .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
            println!("wrote {}", path.display());
        }
    }
}
