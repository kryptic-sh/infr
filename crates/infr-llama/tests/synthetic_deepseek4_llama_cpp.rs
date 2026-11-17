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
    /// `attention.indexer.head_count`, when not the canonical fixture's (`dsv4_indexer`).
    indexer_heads: Option<usize>,
}

impl OracleCase {
    /// Whether the case needs the null ratio-4 tail layer llama.cpp's loader demands (module doc):
    /// exactly when no layer of its own has ratio 4.
    fn null_tail(&self) -> bool {
        !self.ratios.contains(&4)
    }

    /// Whether the case has a compressed layer of its own, and so a sums file to bisect with.
    fn compressed(&self) -> bool {
        self.ratios.iter().any(|&r| r != 0)
    }

    /// Whether the case has a ratio-4 layer of its own, and so a selection fixture
    /// ([`read_selection_fixture`]).
    fn selects(&self) -> bool {
        self.ratios.contains(&4)
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

/// Every case, each compared in CI. `ratio0` is the plain tier; `mixed` is the canonical
/// fixture's layer pattern; `csa` and `hca` put every layer on one compressed tier. Hash routing
/// covers the first layers of each, as on the real file.
///
/// Only `csa` lets the indexer's top-k drop blocks (`top_k` [`BITING_TOP_K`]), and only up to 131
/// tokens: past that some row of some layer always sat on a near-tie (module doc). Measured with
/// llama.cpp (at 4 indexer heads) over 40 prompt sequences: none kept its selection under the noise
/// at 300 tokens, and 30 sequences none at 520. At [`BITING_INDEXER_HEADS`], salts 0 to 4 each sat
/// on a near-tie at 70 or 131 tokens and salt 5 moved by 6.4e-5 under the noise; salt 6 moves by
/// exactly 0 at 30, 70 and 131. The other compressed
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
        indexer_heads: None,
    },
    OracleCase {
        name: "mixed",
        ratios: &[0, 4, 128, 4, 0],
        hash_layer_count: 2,
        top_k: REAL_TOP_K,
        prompt_lens: &PROMPT_LENS,
        prompt_salt: 0,
        indexer_heads: None,
    },
    OracleCase {
        name: "csa",
        ratios: &[4, 4, 4, 4],
        hash_layer_count: 1,
        top_k: BITING_TOP_K,
        prompt_lens: &[7, 30, 70, 131],
        prompt_salt: 6,
        indexer_heads: Some(BITING_INDEXER_HEADS),
    },
    OracleCase {
        name: "hca",
        ratios: &[128, 128, 128, 128],
        hash_layer_count: 1,
        top_k: REAL_TOP_K,
        prompt_lens: &PROMPT_LENS,
        prompt_salt: 0,
        indexer_heads: None,
    },
];

/// The real file's `attention.indexer.top_k`: more blocks than any oracle prompt has, so the
/// selection keeps every visible block.
const REAL_TOP_K: usize = 512;

/// The canonical fixture's `top_k` (`dsv4_indexer`), which the indexer exceeds from the 24th token
/// on.
const BITING_TOP_K: usize = 5;

/// The indexer head count of the case whose top-k drops blocks: enough heads that no block's score
/// is exactly 0. A score is `Σ_h w·relu(q_h·k)`, so it is exactly 0 whenever every head's dot is
/// negative, which with the canonical fixture's four heads happens to one block in sixteen; two
/// such blocks competing for the last kept slot are an exact tie, which llama.cpp breaks by
/// `std::partial_sort`'s order (unspecified, and not infr's lower-index rule), and no noise on the
/// weights can show it, since a negative dot stays negative. Four heads gave nine exact ties at
/// the boundary over 131 tokens, which moved llama.cpp's selection away from infr's at those rows
/// and, through them, later ones (plan 09 § S7). The real file has 64 heads.
const BITING_INDEXER_HEADS: usize = 16;

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
            n_head: case.indexer_heads.unwrap_or(dsv4_indexer().n_head),
            top_k: case.top_k,
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
    load_with(tmp, infr_llama::EngineConfig::default())
}

fn load_with(tmp: &TempGguf, ec: infr_llama::EngineConfig) -> infr_llama::SeamModel {
    infr_llama::SeamModel::load_with(tmp.path(), None, std::sync::Arc::new(ec))
        .expect("synthetic model load")
}

/// The chunk height the layer-major leg of [`infr_scores`] prefills at: short enough that every
/// stored length past it runs several chunks through each layer.
const LAYER_MAJOR_UBATCH: usize = 64;

/// The engine configuration that prefills layer-major (`paging.layer_major` forced on, as a CPU
/// DeepSeek V4 run streaming its experts from disk takes it) in chunks of [`LAYER_MAJOR_UBATCH`].
fn layer_major_config() -> infr_llama::EngineConfig {
    let mut ec = infr_llama::EngineConfig::default();
    ec.paging.layer_major = Some(true);
    ec.device.ubatch = Some(LAYER_MAJOR_UBATCH);
    ec.device.ubatch_specified = true;
    ec
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

/// The bound on infr's CPU logits against llama.cpp's. The scores
/// measured when the data was generated are recorded, as generated lines, in the case's fixture
/// header (`# infr CPU at ...`); docs/plan/09-deepseek-v4-compressed.md § S5a records them with
/// the gaps a window off by one and dropped attention sinks give, which this bound separates.
/// Probability cosine is printed, not asserted: it stays close to 1 under both breaks.
const ORACLE_MAX_ABS: f32 = 2e-3;

/// The row count of every `Op::Attention` a `len`-token prompt runs layer-major in chunks of
/// [`LAYER_MAJOR_UBATCH`] on `n_layer` layers: every chunk of the batched prefill (all positions
/// but the last, when more than two) through a layer before the next, then the decode loop's one
/// row through every layer.
fn layer_major_attention(len: usize, n_layer: usize) -> Vec<u32> {
    let prefill = if len > 2 { len - 1 } else { 0 };
    let chunks: Vec<u32> = (0..prefill)
        .step_by(LAYER_MAJOR_UBATCH)
        .map(|s| LAYER_MAJOR_UBATCH.min(prefill - s) as u32)
        .collect();
    let mut out: Vec<u32> = (0..n_layer).flat_map(|_| chunks.clone()).collect();
    let decode = if len > 2 { 1 } else { len };
    for _ in 0..decode {
        out.extend(std::iter::repeat_n(1, n_layer));
    }
    out
}

/// infr's CPU prefill of `case` against each of `runs` (one fixture's), returning one score per run.
///
/// Each prompt is prefilled twice: chunk-major at the default height, and layer-major in chunks of
/// [`LAYER_MAJOR_UBATCH`] ([`layer_major_config`]). The two must give the same logits bit for bit
/// (on these f32 files every op computes a row the same way whatever the order and the row count),
/// so the score, held to the oracle's bound by the callers, is the layer-major path's as much as
/// the chunk-major one's.
fn infr_scores(case: &OracleCase, runs: &[OracleRun]) -> Vec<(usize, Score)> {
    let tmp = TempGguf::write(
        &format!("dsv4-oracle-{}", case.name),
        &oracle_model(case, false),
    );
    let model = load(&tmp);
    let layer_major = load_with(&tmp, layer_major_config());
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
            infr_cpu::op_tap::start();
            let lm = layer_major
                .prefill_logits_cpu(&run.tokens)
                .expect("layer-major cpu prefill");
            let attention: Vec<u32> = infr_cpu::op_tap::take()
                .into_iter()
                .filter_map(|o| match o {
                    infr_cpu::op_tap::TappedOp::Attention { rows } => Some(rows),
                    _ => None,
                })
                .collect();
            assert_eq!(
                attention,
                layer_major_attention(len, model.config().n_layer),
                "{} at {len} tokens: the prefill did not run layer-major",
                case.name
            );
            assert!(
                lm == ours,
                "{} at {len} tokens: the layer-major prefill's logits differ from chunk-major's \
                 by up to {}",
                case.name,
                max_abs_diff(&lm, &ours)
            );
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

/// **The external oracle on the plain tier, in CI.** infr's CPU prefill of the ratio-0
/// case (hash- and bias-routed layers, hyper-connections at the real stream count, sinks, the
/// grouped output projection, the de-rope, YaRN keys that ratio-0 layers must ignore) against
/// llama.cpp's logits at every prompt length, below and past the sliding window.
#[test]
fn synthetic_dsv4_ratio0_cpu_matches_llama_cpp() {
    let case = &CASES[0];
    assert!(case.null_tail());
    for (len, s) in cpu_matches_llama_cpp(case) {
        assert!(
            s.max_abs < ORACLE_MAX_ABS,
            "infr's CPU misses llama.cpp by {} at {len} tokens on the synthetic ratio-0 V4",
            s.max_abs
        );
    }
}

/// **The HCA tier against llama.cpp, in CI.** infr's CPU prefill of the all-ratio-128 case (four
/// HCA layers and the null tail) at every stored length: before the first block completes (7, 30
/// and 70 tokens, where the compressor only fills its ring), past the first block (131), across
/// two (300) and four (520) blocks and past the raw window, at the bound the ratio-0 tier is held
/// to. Plan 09 § S6 records the measured scores and the breaks this bound separates.
#[test]
fn synthetic_dsv4_hca_cpu_matches_llama_cpp() {
    let case = &CASES[3];
    assert_eq!(case.name, "hca");
    assert!(case.null_tail());
    for (len, s) in cpu_matches_llama_cpp(case) {
        assert!(
            s.max_abs < ORACLE_MAX_ABS,
            "infr's CPU misses llama.cpp by {} at {len} tokens on the synthetic HCA V4",
            s.max_abs
        );
    }
}

/// **The CSA tier against llama.cpp, in CI, with the indexer's top-k dropping blocks.** infr's
/// CPU prefill of the all-ratio-4 case at `top_k` [`BITING_TOP_K`]: before the first block (7
/// tokens), while every visible block is kept (30 tokens, under the top-k's six blocks), and where
/// the top-k drops blocks (70 and 131 tokens), at the bound the ratio-0 tier is held to. A point
/// where the top-k drops blocks was checked at generation to keep llama.cpp's selection under
/// noise twice the f16 rounding of the indexer keys, so a flipped selection is infr's, not a
/// near-tie. Plan 09 § S7 records the measured scores and the breaks this bound separates.
#[test]
fn synthetic_dsv4_csa_cpu_matches_llama_cpp() {
    let case = &CASES[2];
    assert_eq!(case.name, "csa");
    assert!(!case.null_tail() && case.top_k_bites(*case.prompt_lens.last().expect("lengths")));
    for (len, s) in cpu_matches_llama_cpp(case) {
        assert!(
            s.max_abs < ORACLE_MAX_ABS,
            "infr's CPU misses llama.cpp by {} at {len} tokens on the synthetic CSA V4",
            s.max_abs
        );
    }
}

/// **Every tier in one model against llama.cpp, in CI.** infr's CPU prefill of the canonical
/// fixture's layer pattern (ratio 0, 4, 128, 4, 0, hash routing on the first two) at the real
/// file's `top_k` at every stored length, past the raw window and four HCA blocks: the CSA tier
/// beside the HCA one, at the bound the ratio-0 tier is held to.
#[test]
fn synthetic_dsv4_mixed_cpu_matches_llama_cpp() {
    let case = &CASES[1];
    assert_eq!(case.name, "mixed");
    assert!(!case.null_tail());
    for (len, s) in cpu_matches_llama_cpp(case) {
        assert!(
            s.max_abs < ORACLE_MAX_ABS,
            "infr's CPU misses llama.cpp by {} at {len} tokens on the synthetic mixed V4",
            s.max_abs
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
        if case.compressed() {
            files.push(case.fixture_path("_sums"));
        }
        if case.selects() {
            files.push(case.fixture_path("_top_k"));
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
    for case in CASES.iter().filter(|c| c.compressed()) {
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

/// [`llama_selections`] and [`format_selections`] on a hand-written log in llama-debug's format:
/// eight one-token steps with a selection at positions 0 (no block visible yet: dropped), 3 (values
/// elided, so `all`) and 7 (a short top-k filled with a masked block), each beside the
/// `(reshaped)` view llama-debug prints too, which is ignored.
#[test]
fn synthetic_dsv4_selection_log_parses() {
    let row = |name: &str, vals: &str| {
        format!(
            "common_debug_cb_eval:   {name} = (i32)       CONT(node_1{{2, 1, 1, 1}}, }}) = \
             {{2, 1, 1, 1}}\n    [\n        [\n            [ {vals} ],\n        ],\n    ]\n    \
             sum = 1.000000\n"
        )
    };
    let norm = "common_debug_cb_eval:   result_norm = (f32) MUL(norm{2, 1, 1, 1}, \
                w{2, 1, 1, 1}}) = {2, 1, 1, 1}\n    sum = 0.000000\n";
    let mut log = String::new();
    for pos in 0..8 {
        let vals = match pos {
            0 => Some("0.0000, 1.0000"),
            3 => Some("0.0000,    ...,  1.0000"),
            7 => Some("1.0000, 3.0000"),
            _ => None,
        };
        if let Some(vals) = vals {
            log.push_str(&row("lid_top_k-2", vals));
            log.push_str(&row("lid_top_k-2 (reshaped)", vals));
        }
        log.push_str(norm);
    }
    let rows = llama_selections(&log, 2);
    assert_eq!(
        rows,
        vec![
            SelectionRow {
                pos: 3,
                layer: 2,
                kept: None
            },
            SelectionRow {
                pos: 7,
                layer: 2,
                kept: Some(vec![1])
            },
        ]
    );
    assert_eq!(format_selections(&rows), "3 2 all\n7 2 1\n");
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
/// measured robustness, and infr's own scores against the new data.
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
        for (len, s) in infr_scores(case, &runs) {
            notes.push_str(&format!("# infr CPU at {len} tokens: {s}\n"));
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

        if !case.compressed() {
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
        let sums_cmd = format!(
            "llama-debug -m <synthetic gguf> -p <the first {} prompt ids as text> -c {ctx} -ub 1 \
             --no-warmup {}",
            case.sums_len(),
            filters.join(" ")
        );
        if case.selects() {
            // The gap between the last kept and first dropped score is read off infr's own
            // scores: llama-debug prints only six values of a row of scores, and where the two
            // engines keep the same blocks their scores agree to f32 rounding.
            if let Err(e) = check_near_ties(case, &infr_selections(case)) {
                panic!("{e}");
            }
            let mut text = header(
                "llama.cpp's lightning-indexer selection, one ubatch per token, at every row",
                &sums_cmd,
            );
            text.push_str(SELECTION_HEADER);
            if !case.top_k_bites(case.sums_len()) {
                text.push_str(SELECTION_ALL_HEADER);
            }
            let rows = llama_selections(&log, case.top_k);
            assert_eq!(
                rows.len(),
                selection_rows(case),
                "the run printed another number of selections"
            );
            text.push_str(&format_selections(&rows));
            if let Some(written) = write_or_compare_fixture(&case.fixture_path("_top_k"), &text) {
                println!("wrote {}", written.display());
            }
        }
        let mut text = header(
            "llama.cpp per-tensor sums, one ubatch per token, at the positions below",
            &sums_cmd,
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

/// One row's selection in a selection fixture: its position, its layer, and the visible blocks it
/// kept, ascending; `None` is every visible block.
#[derive(Debug, PartialEq)]
struct SelectionRow {
    pos: usize,
    layer: usize,
    kept: Option<Vec<u32>>,
}

/// The visible-block count of the row at `pos`: a block completes at the token that ends it.
fn visible_blocks(pos: usize) -> usize {
    (pos + 1) / CSA_RATIO
}

/// `kept`, restricted to the row's visible blocks and sorted, or `None` when that is all of them.
fn selection_of(pos: usize, kept: impl Iterator<Item = u32>) -> Option<Vec<u32>> {
    let vis = visible_blocks(pos);
    let mut v: Vec<u32> = kept.filter(|&j| (j as usize) < vis).collect();
    v.sort_unstable();
    v.dedup();
    (v.len() != vis).then_some(v)
}

fn format_selections(rows: &[SelectionRow]) -> String {
    rows.iter()
        .map(|r| {
            let kept = r.kept.as_ref().map_or_else(
                || "all".to_string(),
                |k| k.iter().map(u32::to_string).collect::<Vec<_>>().join(" "),
            );
            format!("{} {} {kept}\n", r.pos, r.layer)
        })
        .collect()
}

/// `case`'s checked-in selection fixture, the inverse of [`format_selections`].
fn read_selection_fixture(case: &OracleCase) -> Vec<SelectionRow> {
    let path = case.fixture_path("_top_k");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let num = |i: usize| -> usize {
                f.get(i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| panic!("{}: bad line {l:?}", path.display()))
            };
            let kept = (f[2..] != ["all"]).then(|| {
                f[2..]
                    .iter()
                    .map(|v| v.parse().expect("a block index"))
                    .collect()
            });
            SelectionRow {
                pos: num(0),
                layer: num(1),
                kept,
            }
        })
        .collect()
}

/// The row count of `case`'s one-token-per-step run that score blocks: every position that sees
/// one, once per ratio-4 layer.
fn selection_rows(case: &OracleCase) -> usize {
    let layers = case.ratios.iter().filter(|&&r| r == 4).count();
    (0..case.sums_len())
        .filter(|&p| visible_blocks(p) > 0)
        .count()
        * layers
}

/// llama.cpp's selection at every row of the one-token-per-ubatch run whose `log` holds the
/// `lid_top_k-<layer>` values (`--tensor-filter lid_top_k`), for the rows that see a block. Where
/// llama-debug elides values (more than six), the row must keep every visible block by
/// construction (`top_k` at least the visible count), and is recorded as `all`.
fn llama_selections(log: &str, top_k: usize) -> Vec<SelectionRow> {
    const TAG: &str = "common_debug_cb_eval: ";
    let mut rows = Vec::new();
    let mut pos = 0;
    let mut values_of: Option<usize> = None;
    for line in log.lines() {
        if let Some(h) = line.split_once(TAG).map(|(_, h)| h.trim()) {
            values_of = h
                .strip_prefix("lid_top_k-")
                .and_then(|r| r.split_once(" = "))
                .and_then(|(layer, _)| layer.parse().ok());
            if h.starts_with("result_norm") {
                pos += 1;
            }
            continue;
        }
        let Some(layer) = values_of else { continue };
        let t = line
            .trim()
            .trim_matches(|c| c == '[' || c == ']' || c == ',')
            .trim();
        if t.is_empty() {
            continue;
        }
        values_of = None;
        if visible_blocks(pos) == 0 {
            continue;
        }
        let kept = if t.contains("...") {
            assert!(
                top_k >= visible_blocks(pos),
                "llama-debug elided the selection of a row whose top-k drops blocks (position \
                 {pos}, layer {layer})"
            );
            None
        } else {
            selection_of(
                pos,
                t.split(',')
                    .map(|v| v.trim().parse::<f32>().expect("an index") as u32),
            )
        };
        rows.push(SelectionRow { pos, layer, kept });
    }
    rows
}

/// The relative gap between a row's last kept and first dropped score under which the row is a
/// near-tie. The f16 rounding of an indexer key moves a score by about 2^-11 relative, and the
/// reference rounds Hadamard-rotated keys where infr rounds unrotated ones (docs/deepseek.md §
/// "The CSA tier"), so under this gap the two engines may keep different blocks without either
/// being wrong.
const SELECTION_MARGIN: f64 = 1e-3;

/// The near-ties of the checked-in data ([`SELECTION_MARGIN`]), each with the relative margin infr
/// measured there when it was listed: `(case, position, layer, margin)`. On these rows infr may
/// keep other blocks than llama.cpp; on every other row it must keep the same ones.
///
/// The `csa` rows below are the last layer at earlier positions, so no stored logit reads them.
/// llama.cpp keeps another block at both; a temporary Hadamard rotation of infr's LID keys and
/// queries made both agree (plan 09 § S7).
const NEAR_TIES: &[(&str, usize, usize, f64)] = &[
    ("csa", 111, 3, 3.6955584767615923e-4),
    ("csa", 122, 3, 4.0974090716636846e-4),
];

/// One row infr's CPU indexer scored: the selection, and the relative gap between its last kept
/// and first dropped score (`None` when it keeps every visible block; 0 for a tie).
struct InfrSelection {
    row: SelectionRow,
    margin: Option<f64>,
}

/// infr's CPU selection at every row of a prefill of `case`'s sums prompt, read through
/// `infr_cpu::indexer_tap`, ordered as the fixture is: by position, then layer. The prefill runs
/// in chunks of several positions per graph; each graph that scores runs one indexer per ratio-4
/// layer, in layer order, over every row of the graph, so an op's ordinal names its layer. A row
/// that sees no block is scored over nothing, and the fixture has no line for it.
fn infr_selections(case: &OracleCase) -> Vec<InfrSelection> {
    let tmp = TempGguf::write(
        &format!("dsv4-selection-{}", case.name),
        &oracle_model(case, false),
    );
    let model = load(&tmp);
    infr_cpu::indexer_tap::start();
    let run = model.prefill_logits_cpu(&oracle_prompt(case, case.sums_len()));
    let tapped = infr_cpu::indexer_tap::take();
    run.expect("cpu prefill");
    let layers: Vec<usize> = (0..case.ratios.len())
        .filter(|&l| case.ratios[l] == 4)
        .collect();
    let ops = tapped.last().map_or(0, |r| r.op + 1);
    assert!(
        ops.is_multiple_of(layers.len()),
        "the graphs ran {ops} indexers, not a whole number per ratio-4 layer"
    );
    let mut rows: Vec<InfrSelection> = tapped
        .iter()
        .filter(|r| visible_blocks(r.pos) > 0)
        .map(|r| {
            let mut sorted = r.scores.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let k = case.top_k.min(sorted.len());
            let margin = (sorted.len() > k).then(|| {
                let (kept, dropped) = (f64::from(sorted[k - 1]), f64::from(sorted[k]));
                if kept == dropped {
                    0.0
                } else {
                    (kept - dropped) / kept.abs().max(dropped.abs())
                }
            });
            InfrSelection {
                row: SelectionRow {
                    pos: r.pos,
                    layer: layers[r.op % layers.len()],
                    kept: selection_of(r.pos, r.kept.iter().copied()),
                },
                margin,
            }
        })
        .collect();
    rows.sort_by_key(|s| (s.row.pos, s.row.layer));
    assert_eq!(
        rows.len(),
        selection_rows(case),
        "infr's indexer scored another number of rows"
    );
    rows
}

/// Every row of `infr` whose kept-versus-dropped margin is under [`SELECTION_MARGIN`] (an exact
/// tie among them) must be listed in [`NEAR_TIES`], and every entry of `case` there must still be
/// such a row.
fn check_near_ties(case: &OracleCase, infr: &[InfrSelection]) -> Result<(), String> {
    let listed: Vec<(usize, usize)> = NEAR_TIES
        .iter()
        .filter(|t| t.0 == case.name)
        .map(|t| (t.1, t.2))
        .collect();
    let mut errors = Vec::new();
    for s in infr {
        let at = (s.row.pos, s.row.layer);
        let Some(m) = s.margin else {
            if listed.contains(&at) {
                errors.push(format!("{at:?} is listed but keeps every visible block"));
            }
            continue;
        };
        match (m < SELECTION_MARGIN, listed.contains(&at)) {
            (true, false) if m == 0.0 => errors.push(format!(
                "position {} layer {}: its last kept and first dropped scores tie exactly",
                at.0, at.1
            )),
            (true, false) => errors.push(format!(
                "position {} layer {}: relative margin {m:e}",
                at.0, at.1
            )),
            (false, true) => errors.push(format!(
                "position {} layer {} is listed but its margin is {m:e}",
                at.0, at.1
            )),
            _ => {}
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{}: indexer near-ties among the stored rows (an exact tie llama.cpp's \
             `std::partial_sort` breaks in no defined order, or a gap under {SELECTION_MARGIN} \
             that f16 rounding can flip). Pick another prompt_salt, or list a measured f16-scale \
             near-tie in NEAR_TIES:\n{}",
            case.name,
            errors.join("\n")
        ))
    }
}

/// **infr's CPU indexer keeps, at every row, the blocks llama.cpp keeps**, on the cases with a
/// ratio-4 layer: every (position, layer) of the one-token-per-step run, against the checked-in
/// selection. The logits cannot show this: a selection at a row nothing later reads (the last
/// layer, at an earlier position) moves no logit. A row in [`NEAR_TIES`] may differ and must still
/// be a near-tie; no other row may differ, or come within [`SELECTION_MARGIN`] of a tie.
#[test]
fn synthetic_dsv4_indexer_keeps_what_llama_cpp_keeps() {
    for case in CASES.iter().filter(|c| c.selects()) {
        let want = read_selection_fixture(case);
        let got = infr_selections(case);
        assert_eq!(
            got.len(),
            want.len(),
            "{}: the fixture is for another run",
            case.name
        );
        if let Err(e) = check_near_ties(case, &got) {
            panic!("{e}");
        }
        let mut differ = Vec::new();
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(
                (g.row.pos, g.row.layer),
                (w.pos, w.layer),
                "{}: rows out of step",
                case.name
            );
            let listed = NEAR_TIES
                .iter()
                .any(|t| (t.0, t.1, t.2) == (case.name, w.pos, w.layer));
            if g.row.kept != w.kept && !listed {
                differ.push(format!(
                    "position {} layer {}: infr keeps {:?}, llama.cpp {:?} (margin {:?})",
                    w.pos, w.layer, g.row.kept, w.kept, g.margin
                ));
            }
        }
        assert!(
            differ.is_empty(),
            "{}: infr's indexer keeps other blocks than llama.cpp's:\n{}",
            case.name,
            differ.join("\n")
        );
    }
}

/// The selection fixture's format, under its header.
const SELECTION_HEADER: &str = "\
# One line per (position, layer) whose row sees a block: `<position> <layer> <kept blocks>`,
# ascending, or `all` when every visible block is kept. Read off the `lid_top_k-<layer>` values
# llama-debug prints, restricted to the visible blocks (a short top-k is filled with masked ones).
";

/// Added when the case's `top_k` never drops a block.
const SELECTION_ALL_HEADER: &str = "\
# Here `top_k` is at least the padded width llama.cpp selects from, so every row keeps every
# visible block by construction; llama-debug prints only six of the selected indices.
";

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
