//! Dense layer streaming of FLOAT weights, end to end on a synthetic llama: an all-f16 and an
//! all-f32 checkpoint, with the dense cache forced far below the model's projections
//! (`paging.cache`), must stream their per-layer projections through the arena and generate the
//! same tokens as the all-resident run.
//!
//! A float checkpoint used to have no streamable dense weights at all — the placement planner took
//! only the native quant formats — so a capped budget was silently ignored and the whole model
//! went resident. Both assertions below fail on that: no streaming session exists to report
//! counters.
//!
//! The model is shaped so the projections take every float route: a prefill long enough for the
//! tiled GEMMs (the f16 coopmat GEMM, the split-K warptile on the deep `ffn_down`, the fma
//! warptile without coopmat), decode through the GEMVs, and three layers so each pool holds more
//! blocks than slots and recycles them. Two streamed runs: one with the host DRAM tier off, so
//! misses read the mapping, and one with a host arena under the VRAM one (`paging.dram`), so the
//! float blocks are also read back off the model file.
//!
//! Two more tests pin what a streamed session that HOLDS every block skips: layer-major prefill
//! (counted in static executes per prefill) and the arena barriers around staging copies (counted
//! per forward), each against an evicting budget that still pays for them.
//!
//! ```text
//! INFR_DEV=Vulkan0 cargo test -p infr-llama --release --test synthetic_float_stream -- --ignored
//! ```

use infr_testkit::gguf::{Fill, Meta, SyntheticModel, TempGguf, TensorSpec};

const N_LAYER: usize = 3;
const NE: usize = 128;
const N_HEAD: usize = 4;
const N_KV: usize = 2;
const HD: usize = NE / N_HEAD;
/// Deep enough that `ffn_down`'s narrow GEMM takes the f16 split-K warptile.
const NFF: usize = 1024;
const VOCAB: usize = 64;
/// Tokens to generate.
const N_GEN: usize = 12;

/// Token `id`'s text: the printable ASCII characters from `!` on, each its own GPT-2 byte-level
/// symbol, so a string of them encodes back to exactly these ids.
fn token_text(id: usize) -> char {
    char::from(b'!' + u8::try_from(id).expect("a vocab id"))
}

/// A llama whose every projection (and embedding and output) is `float`; the norms stay f32, as
/// in a real f16 checkpoint.
fn llama(float: Float) -> SyntheticModel {
    let k = |key: &str| format!("llama.{key}");
    let u = |key: &str, v: usize| (k(key), Meta::U32(v as u32));
    let meta = vec![
        (
            "general.architecture".to_string(),
            Meta::Str("llama".to_string()),
        ),
        u("block_count", N_LAYER),
        u("context_length", 256),
        u("embedding_length", NE),
        u("feed_forward_length", NFF),
        u("attention.head_count", N_HEAD),
        u("attention.head_count_kv", N_KV),
        u("rope.dimension_count", HD),
        (k("rope.freq_base"), Meta::F32(1.0e4)),
        (k("attention.layer_norm_rms_epsilon"), Meta::F32(1.0e-5)),
        (
            "tokenizer.ggml.model".to_string(),
            Meta::Str("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Meta::StrArr((0..VOCAB).map(|i| token_text(i).to_string()).collect()),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            Meta::StrArr(Vec::new()),
        ),
        ("tokenizer.ggml.eos_token_id".to_string(), Meta::U32(0)),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            Meta::Bool(false),
        ),
    ];
    let w = |name: String, shape: Vec<usize>, amp: f32| float.tensor(name, shape, amp);
    let mut tensors = vec![
        w("token_embd.weight".into(), vec![NE, VOCAB], 1.0),
        TensorSpec::new("output_norm.weight", vec![NE], Fill::Gain),
        w("output.weight".into(), vec![NE, VOCAB], 0.5),
    ];
    for l in 0..N_LAYER {
        let n = |s: &str| format!("blk.{l}.{s}.weight");
        tensors.push(TensorSpec::new(n("attn_norm"), vec![NE], Fill::Gain));
        tensors.push(TensorSpec::new(n("ffn_norm"), vec![NE], Fill::Gain));
        tensors.push(w(n("attn_q"), vec![NE, N_HEAD * HD], 0.3));
        tensors.push(w(n("attn_k"), vec![NE, N_KV * HD], 0.3));
        tensors.push(w(n("attn_v"), vec![NE, N_KV * HD], 0.3));
        tensors.push(w(n("attn_output"), vec![N_HEAD * HD, NE], 0.2));
        tensors.push(w(n("ffn_gate"), vec![NE, NFF], 0.2));
        tensors.push(w(n("ffn_up"), vec![NE, NFF], 0.2));
        tensors.push(w(n("ffn_down"), vec![NFF, NE], 0.1));
    }
    SyntheticModel { meta, tensors }
}

#[derive(Clone, Copy, Debug)]
enum Float {
    F16,
    F32,
}

impl Float {
    /// A pseudo-random tensor in `[-amp, amp]`, written as this dtype.
    fn tensor(self, name: String, shape: Vec<usize>, amp: f32) -> TensorSpec {
        let t = TensorSpec::new(name, shape, Fill::Rand(amp));
        match self {
            Float::F32 => t,
            Float::F16 => {
                let v = t.values();
                TensorSpec::new(t.name, t.shape, Fill::ExactF16(v))
            }
        }
    }
}

fn load(tmp: &TempGguf, edit: impl FnOnce(&mut infr_llama::EngineConfig)) -> infr_llama::SeamModel {
    let mut cfg = infr_llama::EngineConfig::default();
    edit(&mut cfg);
    infr_llama::SeamModel::load_with(tmp.path(), None, std::sync::Arc::new(cfg))
        .expect("synthetic model load")
}

/// Greedy text for `prompt` on a fresh Vulkan session of `model`, and the dense streaming
/// session's counters after it (`None` when nothing streamed).
fn generate(
    model: &infr_llama::SeamModel,
    prompt: &str,
) -> (String, Option<infr_vulkan::pager::DenseTotals>) {
    let mut session = model.vulkan_session(256).expect("vulkan session");
    let mut text = String::new();
    model
        .generate_vulkan_session(&mut session, prompt, N_GEN, None, |p| text.push_str(p))
        .expect("vulkan generation");
    (text, session.dense_pager_totals())
}

fn check(float: Float) {
    let tag = format!("float-stream-{float:?}");
    let tmp = TempGguf::write(&tag, &llama(float));
    // Long enough for a tiled prefill GEMM (well past the multi-row GEMV bands).
    let prompt: String = (0..40).map(|i| token_text((i * 7 + 3) % VOCAB)).collect();

    // The fused-gate/up capability the seam's group enumeration reads (`fuse_gu_decision`).
    let combined_gu = {
        use infr_core::Backend;
        infr_vulkan::VulkanBackend::new()
            .expect("vulkan device")
            .capabilities()
            .combined_gu
    };
    let (resident, none) = generate(&load(&tmp, |_| {}), &prompt);
    assert!(
        none.is_none(),
        "{float:?}: the resident run streamed ({none:?})"
    );
    assert!(
        resident.chars().count() >= N_GEN / 2,
        "{float:?}: the resident run stopped early ({resident:?}) — too few tokens to compare"
    );

    // 64 KiB is below the staging ring's own 256 MiB floor (`infr_core::pager::ring_bytes`), so
    // nothing is left for the arenas and every pool runs at its 2-slot floor — below the three
    // or more blocks each pool holds — so blocks are evicted and re-staged every forward.
    let cap = |c: &mut infr_llama::EngineConfig| {
        c.paging.cache = Some(infr_core::SizeSpec::Bytes(64 * 1024));
    };
    // `paging.dram = 0` turns the host tier off (misses read the mapping); 1 MiB is below the
    // model's streamed bytes, so the host arena evicts too.
    for (leg, dram) in [("over the mmap", 0), ("over the host tier", 1024 * 1024)] {
        let streamed_model = load(&tmp, |c| {
            cap(c);
            c.paging.dram = Some(infr_core::SizeSpec::Bytes(dram));
        });
        let (streamed, totals) = generate(&streamed_model, &prompt);
        let t = totals.unwrap_or_else(|| panic!("{float:?} ({leg}): no weight streamed"));
        println!("{float:?} ({leg}): {t:?}");
        // Every layer's q, k, v, output, gate and up (one group when the backend uploads them
        // combined, as `fuse_gu_decision` asks of it — every layer here shares one dtype) and
        // down.
        let gate_up_groups = if combined_gu { 1 } else { 2 };
        assert_eq!(
            t.groups,
            (5 + gate_up_groups) * N_LAYER,
            "{float:?} ({leg}): streamed groups"
        );
        assert!(
            t.vram.evictions > 0,
            "{float:?} ({leg}): nothing was evicted — the cap did not bite ({t:?})"
        );
        assert_eq!(
            t.file_reads > 0,
            dram > 0,
            "{float:?} ({leg}): file reads with the host tier, none without it ({t:?})"
        );
        assert_eq!(
            streamed, resident,
            "{float:?} ({leg}): the streamed run diverged from the resident one"
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_f16_llama_streams_and_matches_resident() {
    check(Float::F16);
}

#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_f32_llama_streams_and_matches_resident() {
    check(Float::F32);
}

/// A dense cache far above this model's projections: what is left after the staging ring takes
/// its share (`infr_core::pager::ring_bytes`) still gives every pool a slot per block. The tests
/// assert it: no evictions at this budget.
const HOLDS_ALL: u64 = 512 * 1024 * 1024;
/// Below what `infr_core::pager::ring_bytes` takes for the ring, so the arenas get nothing and
/// every pool runs at the seam's per-pool slot floor (where it builds each `DensePoolSpec`), below
/// its block count. The tests assert it evicts.
const EVICTS: u64 = 64 * 1024;

/// A prompt of `n` tokens; its prefill is `n - 1` rows (the decode loop feeds the last one).
fn prompt_of(n: usize, salt: usize) -> String {
    (0..n)
        .map(|i| token_text((i * 7 + 3 + salt) % VOCAB))
        .collect()
}

fn load_at(tmp: &TempGguf, cache: Option<u64>, ubatch: usize) -> infr_llama::SeamModel {
    load(tmp, |c| {
        c.device.ubatch = Some(ubatch);
        c.device.ubatch_specified = true;
        if let Some(b) = cache {
            c.paging.cache = Some(infr_core::SizeSpec::Bytes(b));
            c.paging.dram = Some(infr_core::SizeSpec::Bytes(0));
        }
    })
}

/// Layer-major prefill runs only where it can save a read: several chunks over a cache that
/// cannot hold every block. One chunk, or a cache with a slot per block, prefills chunk-major —
/// one static execute per chunk — instead of one per (layer, chunk). Before that rule a streamed
/// session always ran layer-major, so the first two cases made `N_LAYER` times their chunk count.
/// Each case also generates the resident run's tokens at the same chunk height.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_streamed_prefill_is_layer_major_only_where_it_saves_reads() {
    let tmp = TempGguf::write("float-stream-layer-major", &llama(Float::F16));
    const TOKENS: usize = 40;
    let prompt = prompt_of(TOKENS, 0);
    let rows = TOKENS - 1;
    for (cache, ubatch, layer_major, what) in [
        (
            HOLDS_ALL,
            64,
            false,
            "one chunk, a cache that holds every block",
        ),
        (
            HOLDS_ALL,
            16,
            false,
            "several chunks, a cache that holds every block",
        ),
        (EVICTS, 64, false, "one chunk, an evicting cache"),
        (EVICTS, 16, true, "several chunks, an evicting cache"),
    ] {
        let chunks = rows.div_ceil(ubatch) as u64;
        let want = if layer_major {
            N_LAYER as u64 * chunks
        } else {
            chunks
        };
        let model = load_at(&tmp, Some(cache), ubatch);
        let mut session = model.vulkan_session(256).expect("vulkan session");
        // Every static execute comes from `Backend::execute`, and with `max_new = 0` the runner
        // calls it only per prefill span: loading the weights and picking the slot upload
        // without executing, and the decode loop stops before feeding the last prompt token.
        let before = session.static_executes();
        model
            .generate_vulkan_session(&mut session, &prompt, 0, None, |_| {})
            .expect("prefill");
        let executes = session.static_executes() - before;
        let t = session
            .dense_pager_totals()
            .unwrap_or_else(|| panic!("{what}: no weight streamed"));
        println!("{what}: {executes} static executes, {t:?}");
        assert_eq!(
            t.vram.evictions == 0,
            cache == HOLDS_ALL,
            "{what}: the cache did not hold (or evict) as the case says ({t:?})"
        );
        assert_eq!(
            executes, want,
            "{what}: static executes per prefill ({chunks} chunks, {N_LAYER} layers)"
        );

        let (streamed, _) = generate(&model, &prompt);
        let (resident, none) = generate(&load_at(&tmp, None, ubatch), &prompt);
        assert!(none.is_none(), "{what}: the resident run streamed");
        assert_eq!(streamed, resident, "{what}: diverged from the resident run");
    }
}

/// The arena barriers bracket a staging COPY and nothing else: on a session that holds every
/// block, a second generation (every block resident) records none, and on an evicting one each
/// forward records exactly two per miss. Before that rule every streamed Linear recorded two, hit
/// or miss, so both counts read twice the stagings instead.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_streamed_arena_barriers_only_around_copies() {
    let tmp = TempGguf::write("float-stream-barriers", &llama(Float::F16));
    for (cache, what) in [
        (HOLDS_ALL, "a cache that holds every block"),
        (EVICTS, "an evicting cache"),
    ] {
        let model = load_at(&tmp, Some(cache), 64);
        let mut session = model.vulkan_session(256).expect("vulkan session");
        let mut run = |salt: usize| {
            let b0 = session.arena_barriers();
            let t0 = session.dense_pager_totals().unwrap_or_default();
            model
                .generate_vulkan_session(&mut session, &prompt_of(40, salt), N_GEN, None, |_| {})
                .expect("vulkan generation");
            let t = session
                .dense_pager_totals()
                .unwrap_or_else(|| panic!("{what}: no weight streamed"));
            let barriers = session.arena_barriers() - b0;
            let (hits, misses) = (t.vram.hits - t0.vram.hits, t.vram.misses - t0.vram.misses);
            println!("{what}, run {salt}: {barriers} arena barriers, {hits} hits, {misses} misses");
            (barriers, hits, misses, t)
        };
        let (first, _, first_misses, t) = run(0);
        assert_eq!(
            first,
            2 * first_misses,
            "{what}: the first generation records two barriers per copy"
        );
        let (second, hits, misses, t2) = run(1);
        assert!(hits > 0, "{what}: the second generation hit nothing");
        assert_eq!(
            second,
            2 * misses,
            "{what}: the second generation records two barriers per copy and none per hit"
        );
        if cache == HOLDS_ALL {
            // One miss per group: the seam numbers each pool's blocks from its own counter and
            // registers each group once, so no two registered buffers share a block.
            assert_eq!(
                (first_misses, t.vram.evictions),
                (t.groups as u64, 0),
                "{what}: every block is staged exactly once"
            );
            assert_eq!(
                (second, misses),
                (0, 0),
                "{what}: a session holding every block records no barrier once it is filled"
            );
        } else {
            assert!(
                misses > 0 && t2.vram.evictions > 0,
                "{what}: the cap did not bite ({t2:?})"
            );
        }
    }
}

/// What one greedy generation on a fresh session showed: its text, and how far the backend's
/// counters moved (decode recordings, static executes, arena barriers) along with the dense
/// cache's totals after it.
struct Run {
    text: String,
    recordings: u64,
    static_executes: u64,
    barriers: u64,
    totals: Option<infr_vulkan::pager::DenseTotals>,
}

/// Generate `N_GEN` tokens for `prompt` on a fresh session of `model`, moving the dense cache's
/// placement epoch before the `epoch_move`th decode replay when that is `Some`.
fn run(model: &infr_llama::SeamModel, prompt: &str, epoch_move: Option<u64>) -> Run {
    let mut session = model.vulkan_session(256).expect("vulkan session");
    let (r0, s0, b0) = (
        session.decode_recordings(),
        session.static_executes(),
        session.arena_barriers(),
    );
    if let Some(n) = epoch_move {
        session.force_dense_epoch_move_after(n);
    }
    let mut text = String::new();
    model
        .generate_vulkan_session(&mut session, prompt, N_GEN, None, |p| text.push_str(p))
        .expect("vulkan generation");
    Run {
        text,
        recordings: session.decode_recordings() - r0,
        static_executes: session.static_executes() - s0,
        barriers: session.arena_barriers() - b0,
        totals: session.dense_pager_totals(),
    }
}

/// `load_at` at the default chunk height, with the decode chain at `chain` steps (`0` = off) and
/// the record-once replay off when `no_replay`.
fn load_decode(
    tmp: &TempGguf,
    cache: Option<u64>,
    chain: usize,
    no_replay: bool,
) -> infr_llama::SeamModel {
    load(tmp, |c| {
        c.spec.decode_chain = chain;
        c.kernels.vulkan.no_replay = no_replay;
        if let Some(b) = cache {
            c.paging.cache = Some(infr_core::SizeSpec::Bytes(b));
            c.paging.dram = Some(infr_core::SizeSpec::Bytes(0));
        }
    })
}

/// The default decode chain, and the chain off.
const CHAINS: [usize; 2] = [8, 0];

/// A streamed session that holds every block decodes from ONE record-once recording, as the
/// resident path does, and one that evicts records none (it re-stages every token). The tokens
/// are the resident run's in every case, and the streamed run's with the replay switched off.
/// Before streamed sessions were admitted to the replay, the holding case recorded nothing.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_streamed_decode_replays_while_every_block_is_held() {
    for float in [Float::F16, Float::F32] {
        let tmp = TempGguf::write(&format!("float-stream-replay-{float:?}"), &llama(float));
        let prompt = prompt_of(40, 0);
        for chain in CHAINS {
            let what = |case: &str| format!("{float:?}, chain {chain}, {case}");
            let resident = run(&load_decode(&tmp, None, chain, false), &prompt, None);
            assert!(resident.totals.is_none(), "{}: streamed", what("resident"));
            assert_eq!(resident.recordings, 1, "{}", what("resident"));
            for (cache, no_replay, recordings, case) in [
                (HOLDS_ALL, false, 1, "holding every block"),
                (HOLDS_ALL, true, 0, "holding every block, replay off"),
                (EVICTS, false, 0, "evicting"),
            ] {
                let streamed = run(
                    &load_decode(&tmp, Some(cache), chain, no_replay),
                    &prompt,
                    None,
                );
                let t = streamed
                    .totals
                    .unwrap_or_else(|| panic!("{}: no weight streamed", what(case)));
                println!(
                    "{}: {} recordings, {} static executes, {t:?}",
                    what(case),
                    streamed.recordings,
                    streamed.static_executes
                );
                assert_eq!(
                    t.vram.evictions == 0,
                    cache == HOLDS_ALL,
                    "{}: the cache did not hold (or evict) as the case says",
                    what(case)
                );
                assert_eq!(
                    streamed.recordings,
                    recordings,
                    "{}: decode recordings",
                    what(case)
                );
                assert_eq!(
                    streamed.text,
                    resident.text,
                    "{}: diverged from the resident run",
                    what(case)
                );
            }
        }
    }
}

/// A placement change mid-decode invalidates the recording: forcing the epoch to move before the
/// second replay makes that replay record again (two recordings in all), and the tokens stay the
/// resident run's. A replay that ignored the epoch would keep its one recording.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_streamed_replay_records_again_when_placements_move() {
    let tmp = TempGguf::write("float-stream-epoch", &llama(Float::F16));
    let prompt = prompt_of(40, 0);
    for chain in CHAINS {
        let resident = run(&load_decode(&tmp, None, chain, false), &prompt, None);
        let model = load_decode(&tmp, Some(HOLDS_ALL), chain, false);
        let moved = run(&model, &prompt, Some(2));
        println!("chain {chain}: {} recordings", moved.recordings);
        assert_eq!(moved.recordings, 2, "chain {chain}: decode recordings");
        assert_eq!(
            moved.text, resident.text,
            "chain {chain}: diverged from the resident run after the epoch moved"
        );
    }
}

/// A decode that starts on an empty cache, with no prefill before it (a one-token prompt), stages
/// every streamed block in a pass of its own before recording: each block misses once, with two
/// arena barriers per copy, no static execute runs, and the decode replays from one recording with
/// the resident run's tokens. Without the staging pass the recording finds no resident block.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_streamed_replay_stages_an_empty_cache_first() {
    let tmp = TempGguf::write("float-stream-cold-replay", &llama(Float::F16));
    let prompt = prompt_of(1, 0);
    for chain in CHAINS {
        let resident = run(&load_decode(&tmp, None, chain, false), &prompt, None);
        let cold = run(
            &load_decode(&tmp, Some(HOLDS_ALL), chain, false),
            &prompt,
            None,
        );
        let t = cold
            .totals
            .unwrap_or_else(|| panic!("chain {chain}: no weight streamed"));
        println!(
            "chain {chain}: {} recordings, {} static executes, {} barriers, {t:?}",
            cold.recordings, cold.static_executes, cold.barriers
        );
        assert_eq!(
            (t.vram.misses, t.vram.evictions),
            (t.groups as u64, 0),
            "chain {chain}: every block staged exactly once"
        );
        assert_eq!(
            cold.static_executes, 0,
            "chain {chain}: a static forward ran"
        );
        assert_eq!(
            cold.barriers,
            2 * t.vram.misses,
            "chain {chain}: barriers per copy"
        );
        assert_eq!(cold.recordings, 1, "chain {chain}: decode recordings");
        assert_eq!(
            cold.text, resident.text,
            "chain {chain}: diverged from the resident run"
        );
    }
}

/// Two sequences decoding at once on one `serve --parallel` engine whose streamed cache holds
/// every block, while the placement epoch moves under them. Each must generate what it generates
/// alone, and the move must make the recordings made before it be rebuilt, not replayed stale.
///
/// The engine's warmup (`ParallelSeam::new`) decodes a token first, so every block is staged
/// before any request: under `serve --parallel` the cache is never empty when two sequences run,
/// and the epoch then moves only on a taken-back or rolled-back placement. The forced move stands
/// in for that. Counted per replay check over both sequences, so it lands between two checks
/// while at least one sequence holds a recording made before it and has a check left: at least
/// one rebuild, two when both are mid-decode.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_streamed_parallel_sequences_rebuild_on_an_epoch_move() {
    let tmp = TempGguf::write("float-stream-parallel", &llama(Float::F16));
    let prompts = [prompt_of(40, 0), prompt_of(40, 5)];
    // The move lands at this replay check: the second when chained (two checks per sequence), the
    // sixth per token (twelve per sequence).
    for (chain, move_at) in [(8usize, 2u64), (0, 6)] {
        let serial: Vec<String> = prompts
            .iter()
            .map(|p| run(&load_decode(&tmp, Some(HOLDS_ALL), chain, false), p, None).text)
            .collect();
        assert_ne!(serial[0], serial[1], "the two prompts decode alike");
        let engine = infr_llama::parallel::ParallelSeam::new(
            load_decode(&tmp, Some(HOLDS_ALL), chain, false),
            2,
            None,
        )
        .expect("parallel engine");
        let be = engine.backend();
        let r0 = be.decode_recordings();
        be.force_dense_epoch_move_after(move_at);
        let greedy = infr_llama::sampling::RequestSampling {
            temp: Some(0.0),
            ..Default::default()
        };
        let texts: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = prompts
                .iter()
                .map(|p| {
                    let req = engine.request_ctx(greedy.clone());
                    let engine = &engine;
                    s.spawn(move || {
                        let mut text = String::new();
                        engine
                            .generate(p, N_GEN, None, &req, |piece| text.push_str(piece))
                            .expect("parallel generation");
                        text
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("sequence thread"))
                .collect()
        });
        let recordings = be.decode_recordings() - r0;
        let t = be.dense_pager_totals().expect("no weight streamed");
        println!("chain {chain}: {recordings} recordings over two sequences, {t:?}");
        assert_eq!(t.vram.evictions, 0, "chain {chain}: the cache did not hold");
        assert!(
            recordings >= 3,
            "chain {chain}: {recordings} recordings — the epoch move rebuilt nothing"
        );
        assert_eq!(
            texts, serial,
            "chain {chain}: a concurrent sequence diverged from its serial run"
        );
    }
}
