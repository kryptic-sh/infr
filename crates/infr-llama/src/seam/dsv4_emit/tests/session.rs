//! The synthetic V4 file through the runner's sessions: which weights move which logits, prefix
//! reuse across turns, and the reset a turn that does not extend the session takes.

use super::super::dsv4_fixture::{dsv4_dims, dsv4_model, Dsv4Dims};
use super::rand_vals;
use crate::seam::weights::SeamKv;
use crate::seam::TokenEmbd;
use crate::{Config, EngineConfig};
use infr_core::backend::Backend;
use infr_core::tensor::DType;
use infr_cpu::CpuBackend;
use infr_gguf::Gguf;
use infr_testkit::gguf::{fnv1a64, splitmix64, Fill, Meta, SyntheticModel, TempGguf, TensorSpec};

const HCA: usize = crate::seam::dsv4_plan::DSV4_HCA_RATIO as usize;
const CSA: usize = crate::seam::dsv4_plan::DSV4_CSA_RATIO as usize;

/// The session tests' context: room for every prompt below.
pub(super) const SESSION_CTX: usize = 640;

/// A two-layer V4 file: a compressed layer at `ratio`, then a ratio-0 one, with the oracle files'
/// YaRN keys so the compressed layer's rope differs from the plain one, and the canonical
/// indexer (`top_k` 5) unless `top_k` names another. `edit` rewrites any tensor.
fn v4_model(ratio: usize, top_k: Option<usize>, edit: impl Fn(&mut TensorSpec)) -> SyntheticModel {
    let mut d = dsv4_dims();
    d.compress_ratios = vec![ratio, 0];
    d.hash_layer_count = 1;
    d.swiglu_clamp_exp = vec![7.0, 7.5];
    if let Some(k) = top_k {
        d.indexer.top_k = k;
    }
    let mut m = session_model(&d);
    m.tensors.iter_mut().for_each(edit);
    m
}

/// `d`'s file with a [`SESSION_CTX`] context and the oracle files' YaRN keys.
pub(super) fn session_model(d: &Dsv4Dims) -> SyntheticModel {
    let mut m = dsv4_model(d);
    for (k, v) in m.meta.iter_mut() {
        if k == "deepseek4.context_length" {
            *v = Meta::U32(SESSION_CTX as u32);
        }
    }
    m.meta.extend([
        (
            "deepseek4.rope.scaling.type".to_string(),
            Meta::Str("yarn".into()),
        ),
        ("deepseek4.rope.scaling.factor".to_string(), Meta::F32(16.0)),
        (
            "deepseek4.rope.scaling.original_context_length".to_string(),
            Meta::U32(512),
        ),
    ]);
    m.meta.sort_by(|a, b| a.0.cmp(&b.0));
    m
}

/// `t` with each value multiplied by its own factor drawn from `[1 - amp, 1 + amp)`. Not one
/// factor for the whole tensor: a compressed row is RMS-normed, so a uniform scale of the rows it
/// pools is invisible, and a uniform positive scale of an indexer score keeps its order.
fn perturb(t: &mut TensorSpec, amp: f32) {
    let noise = rand_vals(t.numel(), fnv1a64(t.name.as_bytes()), amp);
    let v = t
        .values()
        .iter()
        .zip(noise)
        .map(|(x, u)| x * (1.0 + u))
        .collect();
    t.fill = Fill::Exact(v);
}

/// `v4_model(ratio, None, ..)`, with every `attn_compressor_kv` [`perturb`]ed when `perturbed`.
fn comp_kv_perturbed(ratio: usize, perturbed: bool) -> SyntheticModel {
    v4_model(ratio, None, |t| {
        if perturbed && t.name.ends_with("attn_compressor_kv.weight") {
            perturb(t, 0.5);
        }
    })
}

/// Whether `t` is one of a CSA layer's indexer tensors: its projections and its compressor.
fn is_indexer(t: &TensorSpec) -> bool {
    t.name.contains(".indexer.") || t.name.contains(".indexer_compressor_")
}

/// One loaded synthetic V4 file on the CPU backend.
pub(super) struct Model {
    g: Gguf,
    pub(super) cfg: Config,
    embd: std::sync::OnceLock<Vec<f32>>,
    be: CpuBackend,
    ec: EngineConfig,
    /// The file on disk, kept for as long as the model: a host arena reads weights from it.
    _file: TempGguf,
}

/// What one generation returned: the ids it generated, the logits its first one was sampled
/// from (the prompt's last position), and how many of the prompt's positions it took from the
/// session (`GenStats::n_cached`).
pub(super) struct Gen {
    pub(super) ids: Vec<u32>,
    pub(super) logits: Vec<f32>,
    pub(super) n_cached: usize,
}

impl Model {
    fn open(tag: &str, m: &SyntheticModel) -> Self {
        let m = Self::open_with(tag, m, EngineConfig::default());
        assert!(m.cfg.has_unrewindable_state());
        m
    }

    pub(super) fn open_with(tag: &str, m: &SyntheticModel, ec: EngineConfig) -> Self {
        let tmp = TempGguf::write(tag, m);
        let g = Gguf::open(tmp.path()).expect("open synthetic GGUF");
        let cfg = Config::from_gguf(&g).expect("synthetic config");
        Model {
            g,
            cfg,
            embd: std::sync::OnceLock::new(),
            be: CpuBackend::new_with(std::sync::Arc::new(ec.clone())),
            ec,
            _file: tmp,
        }
    }

    /// A host weight arena of `budget` bytes over this model's file (`paging.dram`), planned and
    /// sized as a CPU run's is; `None` when the budget seats no pool.
    pub(super) fn host_arena(
        &self,
        budget: u64,
    ) -> Option<std::sync::Arc<infr_cpu::paged::PagedWeights>> {
        let mut ec = self.ec.clone();
        ec.paging.dram = Some(infr_core::SizeSpec::Bytes(budget));
        crate::seam::cpu_paged_store(&ec, &self.g, &self.cfg).expect("host arena")
    }

    /// [`Self::generate`] with the weights bound through `store` (registered on first use, so a
    /// fresh `state` and a fresh store go together) — the CPU run's own binder.
    pub(super) fn generate_paged(
        &self,
        store: &std::sync::Arc<infr_cpu::paged::PagedWeights>,
        prompt: &[u32],
        max_new: usize,
        state: &mut Option<SeamKv>,
    ) -> anyhow::Result<Gen> {
        let n_expert = self.cfg.moe.as_ref().map(|m| m.n_expert);
        let bind = crate::seam::cpu_bind_with(&self.be, Some(store.clone()), n_expert);
        self.generate_bound(&bind, prompt, max_new, state, None)
    }

    /// The last position's logits of `prompt` on `state` (a fresh session when it is `None`).
    fn logits(&self, prompt: &[u32], state: &mut Option<SeamKv>) -> Vec<f32> {
        self.turn(prompt, state).0
    }

    /// [`Self::logits`] and how many of `prompt`'s positions the turn took from the session
    /// instead of prefilling them (`GenStats::n_cached`): what tells a reuse from a re-prefill.
    fn turn(&self, prompt: &[u32], state: &mut Option<SeamKv>) -> (Vec<f32>, usize) {
        let r = self
            .generate(prompt, 1, state, None)
            .expect("synthetic V4 generation");
        (r.logits, r.n_cached)
    }

    /// Generate up to `max_new` greedy ids after `prompt` on `state`, under `req` when given.
    pub(super) fn generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        state: &mut Option<SeamKv>,
        req: Option<&crate::sampling::RequestCtx>,
    ) -> anyhow::Result<Gen> {
        self.generate_bound(
            &crate::seam::cpu_upload_bind(&self.be),
            prompt,
            max_new,
            state,
            req,
        )
    }

    /// [`Self::generate`] with the weights bound by `bind`.
    fn generate_bound(
        &self,
        bind: &crate::seam::BindWeight<'_>,
        prompt: &[u32],
        max_new: usize,
        state: &mut Option<SeamKv>,
        req: Option<&crate::sampling::RequestCtx>,
    ) -> anyhow::Result<Gen> {
        let mut logits = Vec::new();
        let (ids, stats) = crate::seam::generate_dense_backend(
            &self.be,
            bind,
            &self.g,
            &self.cfg,
            &self.ec,
            TokenEmbd::new(&self.embd, &self.g),
            None,
            prompt,
            max_new,
            |_| {},
            state,
            SESSION_CTX,
            None,
            None,
            None,
            Some(&mut logits),
            None,
            None,
            req,
        )?;
        assert_eq!(logits.len(), self.cfg.vocab);
        Ok(Gen {
            ids,
            logits,
            n_cached: stats.n_cached,
        })
    }

    fn fresh(&self, prompt: &[u32]) -> Vec<f32> {
        self.logits(prompt, &mut None)
    }

    /// The bytes of every compressor ring a session holds.
    fn rings(&self, state: &Option<SeamKv>) -> Vec<Vec<u8>> {
        let out = self.comp_state(state, true);
        assert!(!out.is_empty(), "the session holds no ring");
        out
    }

    /// The bytes of the compressed-tier buffers a session holds, layer by layer in
    /// `crate::seam::dsv4_comp_bufs` order: the rings only when `rings_only`, else the block
    /// caches too.
    pub(super) fn comp_state(&self, state: &Option<SeamKv>, rings_only: bool) -> Vec<Vec<u8>> {
        (0..self.cfg.n_layer)
            .flat_map(|l| self.layer_comp_state(state, l, rings_only))
            .collect()
    }

    /// [`Self::comp_state`] of layer `l` alone.
    pub(super) fn layer_comp_state(
        &self,
        state: &Option<SeamKv>,
        l: usize,
        rings_only: bool,
    ) -> Vec<Vec<u8>> {
        let kv = state.as_ref().expect("a session");
        let geometry = crate::seam::dsv4_comp_bufs(&self.cfg, l, kv.max_ctx, kv.k_fmt);
        let mut out = Vec::new();
        for (b, buf) in geometry.iter().zip(&kv.cbufs[l]) {
            if b.ring || !rings_only {
                let mut v = vec![0u8; buf.len_bytes()];
                self.be.download(buf.as_ref(), &mut v).expect("download");
                out.push(v);
            }
        }
        out
    }
}

/// `n` prompt ids drawn from `salt`.
pub(super) fn ids(n: usize, salt: u64) -> Vec<u32> {
    let vocab = dsv4_dims().vocab as u64;
    (0..n as u64)
        .map(|i| (splitmix64(i ^ (salt << 32)) % vocab) as u32)
        .collect()
}

/// Run `prompt` one more token per turn on one session per model, each turn extending the one
/// before (asserted), and hand `check` each length with both models' logits.
fn extend_both(a: &Model, b: &Model, prompt: &[u32], mut check: impl FnMut(usize, &[f32], &[f32])) {
    let (mut sa, mut sb) = (None, None);
    for len in 1..=prompt.len() {
        let (la, reused) = a.turn(&prompt[..len], &mut sa);
        assert_eq!(
            reused,
            len - 1,
            "the turn to {len} tokens did not extend the session"
        );
        let lb = b.logits(&prompt[..len], &mut sb);
        check(len, &la, &lb);
    }
}

/// **A compressor's weights are inert until its first block completes, and load-bearing from the
/// token that completes it**, on either tier. Perturbing `attn_compressor_kv` changes only the
/// rows the compressor pools; before the block's last position none is visible, so every logit before
/// that length is bit-identical, and from the length whose last position completes block 0 (it
/// attends the block in the same step) the logits move. One session per file, extended a token at
/// a time, so each length is also a reuse of the session before it.
#[test]
fn compressor_weights_are_inert_until_the_first_block() {
    for (ratio, n) in [(HCA, 140), (CSA, 30)] {
        let base = Model::open(
            &format!("dsv4-inert-{ratio}-base"),
            &comp_kv_perturbed(ratio, false),
        );
        let scaled = Model::open(
            &format!("dsv4-inert-{ratio}-perturbed"),
            &comp_kv_perturbed(ratio, true),
        );
        extend_both(&base, &scaled, &ids(n, 1), |len, la, lb| {
            if len < ratio {
                assert_eq!(
                    la, lb,
                    "ratio {ratio}: the compressor's kv weights moved the logits at {len} tokens"
                );
            } else {
                assert_ne!(
                    la, lb,
                    "ratio {ratio}: the compressor's kv weights left the logits at {len} tokens \
                     unmoved"
                );
            }
        });
    }
}

/// The canonical indexer's `top_k`, and the first prompt length whose last position sees more
/// blocks than it keeps.
const TOP_K: usize = 5;
const FIRST_DROP: usize = CSA * (TOP_K + 1);

/// **The indexer is inert while it keeps every visible block, and load-bearing from the first
/// position where it drops one.** Redrawing every indexer tensor of a CSA layer (both
/// projections and the LID compressor; each value times a factor from `[-1, 3)`, signs flipping)
/// changes only which blocks the top-k keeps: while a row
/// sees at most `top_k` blocks it keeps them all whatever the scores, so every logit before
/// [`FIRST_DROP`] tokens is bit-identical — which also shows the indexer's tensors and its LID
/// cache reach nothing but the selection — and from that length on the logits move.
#[test]
fn csa_indexer_is_inert_until_the_top_k_drops_a_block() {
    let base = Model::open("dsv4-ix-inert-base", &v4_model(CSA, None, |_| {}));
    assert_eq!(base.cfg.indexer_top_k, TOP_K);
    let noisy = v4_model(CSA, None, |t| {
        if is_indexer(t) {
            perturb(t, 2.0);
        }
    });
    assert!(noisy.tensors.iter().filter(|t| is_indexer(t)).count() == 6);
    let noisy = Model::open("dsv4-ix-inert-noisy", &noisy);
    extend_both(&base, &noisy, &ids(FIRST_DROP + 16, 2), |len, la, lb| {
        if len < FIRST_DROP {
            assert_eq!(
                la, lb,
                "the indexer's weights moved the logits at {len} tokens"
            );
        } else {
            assert_ne!(
                la, lb,
                "the indexer's weights left the logits at {len} tokens unmoved"
            );
        }
    });
}

/// **A `top_k` no row's visible blocks exceed is the same model as one with no indexer at all.**
/// At `top_k` 512 every visible block is kept, so the file whose indexer tensors are all zero
/// (every score 0, a tie the selection breaks by index) computes the same logits, bit for bit, at
/// every length past the first blocks.
#[test]
fn csa_top_k_past_every_block_equals_a_zeroed_indexer() {
    let keep_all = Model::open("dsv4-ix-all", &v4_model(CSA, Some(512), |_| {}));
    let zeroed = v4_model(CSA, Some(512), |t| {
        if is_indexer(t) {
            t.fill = Fill::Exact(vec![0.0; t.numel()]);
        }
    });
    let zeroed = Model::open("dsv4-ix-zero", &zeroed);
    extend_both(&keep_all, &zeroed, &ids(60, 3), |len, la, lb| {
        assert_eq!(
            la, lb,
            "a zeroed indexer changed the logits at {len} tokens"
        );
    });
}

/// **A prompt prefilled in two calls ends where one call does, bit for bit**, on either tier. The
/// first call stops part way into a block, so the second call — which must take those positions
/// from the session and prefill only the rest — completes that block from ring rows the first
/// call persisted. On HCA, 150 tokens into block 1. On CSA, 10 tokens into block 2: that block's
/// previous window (block 1) and its first two positions come from the ring, the straddle every
/// CSA block past the first makes at a turn boundary.
#[test]
fn split_prefill_equals_one_prefill() {
    for (ratio, split, n) in [(HCA, 150, 270), (CSA, 10, 50)] {
        let m = Model::open(
            &format!("dsv4-split-{ratio}"),
            &v4_model(ratio, None, |_| {}),
        );
        let prompt = ids(n, 4);
        let mut s = None;
        assert_eq!(m.turn(&prompt[..split], &mut s).1, 0);
        let (got, reused) = m.turn(&prompt, &mut s);
        assert_eq!(
            reused, split,
            "ratio {ratio}: the extending turn re-prefilled instead of reusing the session"
        );
        assert_eq!(got, m.fresh(&prompt), "ratio {ratio}");
    }
}

/// **A new prompt after a long one starts from a reset state**, on either tier. The session first
/// runs a prompt that commits blocks and writes the whole ring; a short prompt with nothing in
/// common then gives a fresh session's logits AND leaves the session's rings byte-equal to a fresh
/// session's. The short prompt is shorter than the ring (128 rows on HCA, 8 on CSA), so ring rows
/// past it hold the first prompt's rows unless the reset zeroed them.
#[test]
fn new_prompt_after_a_long_one_starts_from_a_reset_state() {
    for (ratio, long, short) in [(HCA, 200, 50), (CSA, 40, 6)] {
        let m = Model::open(
            &format!("dsv4-reset-{ratio}"),
            &v4_model(ratio, None, |_| {}),
        );
        let (long, short) = (ids(long, 5), ids(short, 6));
        assert_ne!(long[0], short[0], "the prompts share no prefix");
        let mut s = None;
        m.logits(&long, &mut s);
        let (got, reused) = m.turn(&short, &mut s);
        assert_eq!(
            reused, 0,
            "ratio {ratio}: a turn sharing nothing reused {reused} positions"
        );
        let mut fresh = None;
        let want = m.logits(&short, &mut fresh);
        assert_eq!(got, want, "ratio {ratio}");
        assert!(
            m.rings(&s) == m.rings(&fresh),
            "ratio {ratio}: the reset left a ring row dirty"
        );
    }
}

/// **A prompt that shares a prefix with a longer earlier one re-prefills instead of rewinding**,
/// on either tier. The session first runs a prompt long enough that the ring rows for the shared
/// prefix's last positions now hold later positions of that first prompt; the second prompt keeps
/// the prefix and runs past the block those rows belong to, so a rewind to the shared prefix would
/// pool the first prompt's later positions instead. The turn must re-prefill from position 0, not
/// merely end on the right logits.
#[test]
fn shared_prefix_does_not_rewind_into_a_stale_ring() {
    for (ratio, first, shared, more) in [(HCA, 300, 150, 110), (CSA, 40, 10, 20)] {
        let m = Model::open(
            &format!("dsv4-rewind-{ratio}"),
            &v4_model(ratio, None, |_| {}),
        );
        let first = ids(first, 7);
        let second = [&first[..shared], &ids(more, 8)[..]].concat();
        let mut s = None;
        m.logits(&first, &mut s);
        let (got, reused) = m.turn(&second, &mut s);
        assert_eq!(
            reused, 0,
            "ratio {ratio}: a non-extending turn rewound to position {reused}"
        );
        assert_eq!(got, m.fresh(&second), "ratio {ratio}");
    }
}

/// **A V4 file with compressed layers keeps its caches in f16 on the CPU unless a quantized KV
/// format is asked for by name.** f16 is what the llama.cpp oracle checks; a quantized cache would
/// round the unrotated compressed and LID rows where the reference rotates them first, which no
/// oracle has measured. The file's widths are whole 32-element blocks (head 64, indexer 32), so a
/// q8 cache fits it: with auto-q8 pinned for the session scope (the Vulkan placement's own
/// default) the cache stays f16, and `INFR_KV_TYPE_K=q8_0` (explicit) does select q8, which shows
/// the format was reachable.
#[test]
fn compressed_caches_default_to_f16_on_the_cpu() {
    let mut d = dsv4_dims();
    d.compress_ratios = vec![CSA, HCA];
    d.hash_layer_count = 1;
    d.swiglu_clamp_exp = vec![7.0, 7.5];
    d.head_dim = 64;
    d.indexer.head_size = 32;
    let m = dsv4_model(&d);
    let prompt = ids(9, 9);
    let fmt = |m: &Model| {
        let mut s = None;
        m.logits(&prompt, &mut s);
        s.expect("a session").k_fmt
    };
    let auto = Model::open("dsv4-kv-auto", &m);
    let _scope = crate::seam::PlacementScope::enter(std::sync::Arc::default());
    crate::seam::pin_kv_auto_q8();
    assert!(crate::seam::kv_auto_q8());
    assert_eq!(
        fmt(&auto),
        DType::F16,
        "an unasked-for format reached a V4 compressed cache"
    );
    let asked = EngineConfig {
        kv: infr_core::config::KvCfg {
            type_k: Some(DType::Q8_0),
            type_k_specified: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let asked = Model::open_with("dsv4-kv-q8", &m, asked);
    assert_eq!(
        fmt(&asked),
        DType::Q8_0,
        "a q8 cache asked for by name was not taken"
    );
}
