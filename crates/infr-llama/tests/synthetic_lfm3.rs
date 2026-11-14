//! Synthetic-GGUF harness for `lfm3`: a tiny model with both layer kinds (short convolution, then
//! full attention, then short convolution, then window attention), a tied output and the
//! `rope_freqs` divisors the full layers read, written to a temp file and driven through the REAL
//! loader and seam — on CPU unconditionally, and on Vulkan behind `#[ignore]`.
//!
//! The real 1.7B file is checked against llama.cpp in `cpu_backend.rs` (and docs/plan/03-lfm3.md),
//! but only where it is in the HF cache; this file is what CI runs. Its window is four tokens, so
//! the eight-token prompt reaches past it on the window layer.
//!
//! The same model is also scored against llama.cpp itself, with no model download:
//! `fixtures/synthetic_lfm3_llama_cpp.txt` holds the logits llama.cpp computed for it, and
//! [`synthetic_lfm3_cpu_matches_llama_cpp`] compares infr's CPU prefill with them. The fixture's
//! header names the llama.cpp build and the command that wrote it; to regenerate it, point
//! `INFR_LLAMA_DEBUG` at a `llama-debug` binary and run the ignored
//! [`synthetic_lfm3_regenerate_llama_cpp_fixture`].
//!
//! ```text
//! cargo test -p infr-llama --test synthetic_lfm3
//! cargo test -p infr-llama --test synthetic_lfm3 -- --include-ignored   # + Vulkan
//! INFR_LLAMA_DEBUG=<llama.cpp>/build/bin/Release/llama-debug.exe \
//!   cargo test -p infr-llama --test synthetic_lfm3 -- --ignored synthetic_lfm3_regenerate_llama_cpp_fixture
//! ```

use infr_testkit::gguf::{Fill, Meta, SyntheticModel, TempGguf, TensorSpec};

const N_LAYER: usize = 4;
const NE: usize = 64;
const N_HEAD: usize = 4;
const N_KV: usize = 2;
const HD: usize = 16;
const NFF: usize = 96;
const VOCAB: usize = 64;
const L_CACHE: usize = 4;
const WINDOW: usize = 4;
/// `attention.head_count_kv`: zero marks a short-convolution layer.
const KV_PER_LAYER: [u32; N_LAYER] = [0, N_KV as u32, 0, N_KV as u32];
/// `attention.sliding_window_pattern`: layer 3 is the window layer.
const SWA: [bool; N_LAYER] = [false, false, false, true];
const PROMPT: &[u32] = &[3, 11, 5, 40, 27, 8, 61, 14];
/// The prompt scored against llama.cpp: longer than the window and the convolution's history
/// several times over. Its ids avoid the backslash and the double quote, which llama.cpp's prompt
/// escaping and the command line would rewrite.
const ORACLE_PROMPT: &[u32] = &[
    3, 11, 5, 40, 27, 8, 61, 14, 33, 20, 47, 9, 52, 36, 18, 62, 25, 44, 7, 30, 56, 12, 39, 21,
];
/// The llama.cpp logits for [`ORACLE_PROMPT`]'s last position on [`lfm3_model`].
const ORACLE_FIXTURE: &str = "tests/fixtures/synthetic_lfm3_llama_cpp.txt";
/// The bound on infr's CPU logits against the fixture: f32 on both sides, so only summation
/// order separates them (docs/plan/03-lfm3.md records the measured gap).
const ORACLE_MAX_ABS: f32 = 1e-4;

/// Token `id`'s text: the printable ASCII characters from `!` on, each its own GPT-2 byte-level
/// symbol, so with no merges llama.cpp's tokenizer reads a string of them back as exactly these
/// ids.
fn token_text(id: u32) -> char {
    char::from(b'!' + u8::try_from(id).expect("a vocab id"))
}

fn lfm3_model() -> SyntheticModel {
    let k = |key: &str| format!("lfm3.{key}");
    let u = |key: &str, v: usize| (k(key), Meta::U32(v as u32));
    let meta = vec![
        (
            "general.architecture".to_string(),
            Meta::Str("lfm3".to_string()),
        ),
        u("block_count", N_LAYER),
        u("context_length", 64),
        u("embedding_length", NE),
        u("feed_forward_length", NFF),
        u("attention.head_count", N_HEAD),
        (
            k("attention.head_count_kv"),
            Meta::U32Arr(KV_PER_LAYER.to_vec()),
        ),
        u("attention.key_length", HD),
        u("attention.value_length", HD),
        u("rope.dimension_count", HD),
        (k("rope.freq_base"), Meta::F32(1.0e6)),
        (k("rope.freq_base_swa"), Meta::F32(1.0e4)),
        (k("attention.layer_norm_rms_epsilon"), Meta::F32(1.0e-5)),
        u("attention.sliding_window", WINDOW),
        (
            k("attention.sliding_window_pattern"),
            Meta::BoolArr(SWA.to_vec()),
        ),
        u("shortconv.l_cache", L_CACHE),
        // The minimum tokenizer `build_tokenizer` accepts; the tests hand token ids to the seam.
        (
            "tokenizer.ggml.model".to_string(),
            Meta::Str("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Meta::StrArr(
                (0..VOCAB as u32)
                    .map(|i| token_text(i).to_string())
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            Meta::StrArr(Vec::new()),
        ),
        ("tokenizer.ggml.eos_token_id".to_string(), Meta::U32(2)),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            Meta::Bool(false),
        ),
    ];
    // The 1.7B file's divisors: the first half of the pairs rope, the second half stand still.
    let rope_freqs: Vec<f32> = (0..HD / 2)
        .map(|p| if p < HD / 4 { 1.0 } else { 1.0e30 })
        .collect();
    let mut tensors = vec![
        TensorSpec::new("token_embd.weight", vec![NE, VOCAB], Fill::Rand(0.5)),
        TensorSpec::new("output_norm.weight", vec![NE], Fill::Gain),
        TensorSpec::new("rope_freqs.weight", vec![HD / 2], Fill::Exact(rope_freqs)),
    ];
    let w = |l: usize, name: &str, shape: Vec<usize>, fill: Fill| {
        TensorSpec::new(format!("blk.{l}.{name}.weight"), shape, fill)
    };
    for (l, &kv) in KV_PER_LAYER.iter().enumerate() {
        for norm in [
            "attn_norm",
            "ffn_norm",
            "post_attention_norm",
            "post_ffw_norm",
        ] {
            tensors.push(w(l, norm, vec![NE], Fill::Gain));
        }
        tensors.push(w(l, "ffn_gate", vec![NE, NFF], Fill::Rand(0.2)));
        tensors.push(w(l, "ffn_up", vec![NE, NFF], Fill::Rand(0.2)));
        tensors.push(w(l, "ffn_down", vec![NFF, NE], Fill::Rand(0.2)));
        if kv == 0 {
            tensors.push(w(l, "shortconv.conv", vec![L_CACHE, NE], Fill::Rand(0.5)));
            tensors.push(w(l, "shortconv.in_proj", vec![NE, 3 * NE], Fill::Rand(0.3)));
            tensors.push(w(l, "shortconv.out_proj", vec![NE, NE], Fill::Rand(0.2)));
        } else {
            tensors.push(w(l, "attn_q", vec![NE, N_HEAD * HD], Fill::Rand(0.3)));
            tensors.push(w(l, "attn_k", vec![NE, N_KV * HD], Fill::Rand(0.3)));
            tensors.push(w(l, "attn_v", vec![NE, N_KV * HD], Fill::Rand(0.3)));
            tensors.push(w(l, "attn_q_norm", vec![HD], Fill::Gain));
            tensors.push(w(l, "attn_k_norm", vec![HD], Fill::Gain));
            tensors.push(w(l, "attn_output", vec![N_HEAD * HD, NE], Fill::Rand(0.2)));
            tensors.push(w(l, "attn_gate", vec![NE, N_HEAD], Fill::Rand(1.0)));
        }
    }
    SyntheticModel { meta, tensors }
}

fn load(tmp: &TempGguf) -> anyhow::Result<infr_llama::SeamModel> {
    infr_llama::SeamModel::load_with(
        tmp.path(),
        None,
        std::sync::Arc::new(infr_llama::EngineConfig::default()),
    )
}

fn cpu_logits(tag: &str, model: &SyntheticModel) -> Vec<f32> {
    let tmp = TempGguf::write(tag, model);
    load(&tmp)
        .expect("synthetic model load")
        .prefill_logits_cpu(PROMPT)
        .expect("cpu prefill")
}

/// `model` with every value of tensor `name` replaced by `v`.
fn with_constant(model: &SyntheticModel, name: &str, v: f32) -> SyntheticModel {
    let mut m = SyntheticModel {
        meta: model.meta.clone(),
        tensors: model.tensors.clone(),
    };
    let t = m
        .tensors
        .iter_mut()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("no tensor {name}"));
    t.fill = Fill::Exact(vec![v; t.shape.iter().product()]);
    m
}

/// Max |a - b|, NaN if any difference is NaN (`f32::max` would drop a NaN operand).
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit vector lengths");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, |m, d| {
            if m.is_nan() || d.is_nan() {
                f32::NAN
            } else {
                m.max(d)
            }
        })
}

/// The loader reads the layer kinds, the window pattern and the kernel width the way
/// `lfm3.cpp::load_arch_hparams` does, and sizes the recurrent state from them.
#[test]
fn synthetic_lfm3_parses_its_layer_kinds() {
    let tmp = TempGguf::write("lfm3-parse", &lfm3_model());
    let model = load(&tmp).expect("synthetic model load");
    let c = model.config();
    assert!(c.lfm3 && c.qk_norm && c.sandwich_norm && c.rope_freq_factors);
    assert_eq!(c.n_kv, N_KV);
    assert_eq!(c.shortconv_l_cache, L_CACHE);
    assert_eq!(c.swa_window, WINDOW);
    assert_eq!(c.swa_rope_theta, 1.0e4);
    for l in 0..N_LAYER {
        let conv = KV_PER_LAYER[l] == 0;
        assert_eq!(c.is_shortconv_layer(l), conv, "layer {l} kind");
        assert_eq!(c.is_recurrent_layer(l), conv, "layer {l} recurrence");
        assert_eq!(c.is_swa_layer(l), SWA[l], "layer {l} window");
        assert_eq!(
            c.recurrent_state_elems(l),
            conv.then_some(((L_CACHE - 1) * NE, 1)),
            "layer {l} state"
        );
    }
    assert!(c.has_recurrent_state());
}

/// A file that declares the non-causal ("d1") variant is refused, not run as the causal model.
#[test]
fn synthetic_lfm3_refuses_the_non_causal_variant() {
    let mut m = lfm3_model();
    m.meta
        .push(("lfm3.attention.causal".to_string(), Meta::Bool(false)));
    let tmp = TempGguf::write("lfm3-noncausal", &m);
    let err = load(&tmp).err().expect("a non-causal LFM3 must be refused");
    assert!(
        format!("{err:#}").contains("non-causal"),
        "unexpected refusal: {err:#}"
    );
}

/// `lfm3.cpp` refuses a layer carrying only one of the two sandwich norms.
#[test]
fn synthetic_lfm3_refuses_half_a_sandwich() {
    let mut m = lfm3_model();
    m.tensors.retain(|t| t.name != "blk.2.post_ffw_norm.weight");
    let tmp = TempGguf::write("lfm3-half-sandwich", &m);
    let err = load(&tmp).err().expect("half a sandwich must be refused");
    assert!(
        format!("{err:#}").contains("layer 2"),
        "unexpected refusal: {err:#}"
    );
}

/// The CPU prefill is finite, and each LFM3-specific tensor reaches the logits: zeroing a head
/// gate's projection or a short-convolution kernel would leave them unmoved if the graph dropped
/// it.
#[test]
fn synthetic_lfm3_cpu_prefill_reads_every_mixer() {
    let base = lfm3_model();
    let logits = cpu_logits("lfm3-base", &base);
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
    for (what, name) in [
        ("the head gate", "blk.1.attn_gate.weight"),
        (
            "the short convolution's kernel",
            "blk.2.shortconv.conv.weight",
        ),
        ("the window layer's head gate", "blk.3.attn_gate.weight"),
    ] {
        let moved = max_abs_diff(
            &logits,
            &cpu_logits(&format!("lfm3-{name}"), &with_constant(&base, name, 0.0)),
        );
        assert!(
            moved > 1e-3,
            "zeroing {what} ({name}) moved the logits by only {moved}"
        );
    }
}

/// The Vulkan seam against the CPU on the synthetic model: both mixers, the window mask past four
/// tokens, the full layers' `rope_freqs` and the tied output, in one prefill.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_lfm3_vulkan_matches_cpu() {
    let model = lfm3_model();
    let want = cpu_logits("lfm3-cpu", &model);
    let tmp = TempGguf::write("lfm3-vk", &model);
    let got = load(&tmp)
        .expect("synthetic model load")
        .prefill_logits_vulkan(PROMPT)
        .expect("vulkan prefill");
    let err = max_abs_diff(&want, &got);
    println!("synthetic lfm3 vulkan-vs-cpu max |logit diff| {err:e}");
    assert!(err < 2e-2, "Vulkan misses the CPU by {err}");
}

/// `lfm3.cpp` reads the RMSNorm epsilon with a plain `get_key`, so a file without it is refused
/// rather than run at a default.
#[test]
fn synthetic_lfm3_refuses_a_missing_rms_epsilon() {
    let mut m = lfm3_model();
    m.meta
        .retain(|(k, _)| k != "lfm3.attention.layer_norm_rms_epsilon");
    let tmp = TempGguf::write("lfm3-no-eps", &m);
    let err = load(&tmp).err().expect("a missing epsilon must be refused");
    assert!(
        format!("{err:#}").contains("layer_norm_rms_epsilon"),
        "unexpected refusal: {err:#}"
    );
}

/// `lfm3.cpp` creates `rope_freqs` as a required tensor for the full-attention layers, so a file
/// with a full layer and no divisors is refused rather than roped without them.
#[test]
fn synthetic_lfm3_refuses_missing_rope_freqs() {
    let mut m = lfm3_model();
    m.tensors.retain(|t| t.name != "rope_freqs.weight");
    let tmp = TempGguf::write("lfm3-no-rope-freqs", &m);
    let err = load(&tmp)
        .err()
        .expect("missing rope_freqs must be refused");
    assert!(
        format!("{err:#}").contains("rope_freqs"),
        "unexpected refusal: {err:#}"
    );
}

/// A file whose short-convolution kernels are f16: the GPU conv shaders read the kernel as f32,
/// so the loader widens it, and Vulkan still matches the CPU (which reads either dtype).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn synthetic_lfm3_vulkan_reads_an_f16_conv_kernel() {
    let mut model = lfm3_model();
    for t in &mut model.tensors {
        if t.name.ends_with("shortconv.conv.weight") {
            t.fill = Fill::ExactF16(t.values());
        }
    }
    let want = cpu_logits("lfm3-f16conv-cpu", &model);
    let tmp = TempGguf::write("lfm3-f16conv-vk", &model);
    let got = load(&tmp)
        .expect("synthetic model load")
        .prefill_logits_vulkan(PROMPT)
        .expect("vulkan prefill");
    let err = max_abs_diff(&want, &got);
    println!("synthetic lfm3 f16 conv kernel, vulkan-vs-cpu max |logit diff| {err:e}");
    assert!(err < 2e-2, "Vulkan misses the CPU by {err}");
}

/// The fixture's token ids and logits: `# tokens:` names the ids, every other non-`#` line is one
/// logit.
fn read_oracle_fixture() -> (Vec<u32>, Vec<f32>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(ORACLE_FIXTURE);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut tokens = None;
    let mut logits = Vec::new();
    for line in text.lines() {
        if let Some(ids) = line.strip_prefix("# tokens:") {
            tokens = Some(
                ids.split_whitespace()
                    .map(|t| t.parse().expect("a token id"))
                    .collect(),
            );
        } else if !line.starts_with('#') && !line.trim().is_empty() {
            logits.push(line.trim().parse().expect("a logit"));
        }
    }
    (tokens.expect("the fixture names its token ids"), logits)
}

/// **The external oracle that runs in CI.** infr's CPU prefill of the synthetic LFM3 against the
/// logits llama.cpp computed for the same file and ids (the fixture): both short convolutions, the
/// head gates, the window mask past four tokens, `rope_freqs` on the full layer, the sandwich
/// norms and the tied output, in f32 on both sides.
#[test]
fn synthetic_lfm3_cpu_matches_llama_cpp() {
    let (tokens, want) = read_oracle_fixture();
    assert_eq!(tokens, ORACLE_PROMPT, "the fixture is for another prompt");
    assert_eq!(want.len(), VOCAB, "the fixture is for another vocabulary");
    let tmp = TempGguf::write("lfm3-oracle-cpu", &lfm3_model());
    let got = load(&tmp)
        .expect("synthetic model load")
        .prefill_logits_cpu(ORACLE_PROMPT)
        .expect("cpu prefill");
    let err = max_abs_diff(&want, &got);
    println!("synthetic lfm3 cpu-vs-llama.cpp max |logit diff| {err:e}");
    assert!(
        err < ORACLE_MAX_ABS,
        "infr's CPU misses llama.cpp by {err} on the synthetic LFM3"
    );
}

/// Rewrite the fixture [`synthetic_lfm3_cpu_matches_llama_cpp`] reads: write the synthetic GGUF,
/// run `llama-debug --save-logits` on [`ORACLE_PROMPT`]'s text (the binary `INFR_LLAMA_DEBUG`
/// names), check llama.cpp tokenized it to exactly those ids, and record its last-position logits
/// with the build's version line and the command.
#[test]
#[ignore = "regenerates the llama.cpp fixture: needs INFR_LLAMA_DEBUG (a llama-debug binary)"]
fn synthetic_lfm3_regenerate_llama_cpp_fixture() {
    let Ok(exe) = std::env::var("INFR_LLAMA_DEBUG") else {
        eprintln!("skip: INFR_LLAMA_DEBUG not set");
        return;
    };
    let tmp = TempGguf::write("lfm3-oracle", &lfm3_model());
    let out_dir = tmp.path().with_extension("llama-debug");
    let prompt: String = ORACLE_PROMPT.iter().map(|&id| token_text(id)).collect();
    let version = run_headless(&exe, &["--version".as_ref()]);
    let version = version
        .lines()
        .find(|l| l.starts_with("version:"))
        .unwrap_or_else(|| panic!("no version line from {exe} --version:\n{version}"))
        .to_string();
    run_headless(
        &exe,
        &[
            "-m".as_ref(),
            tmp.path().as_os_str(),
            "-p".as_ref(),
            prompt.as_ref(),
            "--save-logits".as_ref(),
            "--logits-output-dir".as_ref(),
            out_dir.as_os_str(),
        ],
    );
    let stem = tmp
        .path()
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("a UTF-8 file stem");
    let read = |suffix: &str| {
        let path = out_dir.join(format!("llamacpp-{stem}{suffix}"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert_eq!(bytes.len() % 4, 0, "{} is not 4-byte words", path.display());
        bytes
    };
    let logits: Vec<f32> = read(".bin")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();
    let tokens: Vec<u32> = read("-tokens.bin")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| u32::from_le_bytes(c))
        .collect();
    for suffix in [".bin", ".txt", "-prompt.txt", "-tokens.bin"] {
        std::fs::remove_file(out_dir.join(format!("llamacpp-{stem}{suffix}")))
            .expect("remove a llama-debug output");
    }
    std::fs::remove_dir(&out_dir).expect("remove the llama-debug output directory");
    assert_eq!(
        tokens, ORACLE_PROMPT,
        "llama.cpp tokenized the prompt differently"
    );
    assert_eq!(logits.len(), VOCAB, "llama.cpp returned another vocabulary");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "non-finite llama.cpp logits"
    );

    let ids: Vec<String> = tokens.iter().map(u32::to_string).collect();
    let mut text = format!(
        "# llama.cpp logits for the synthetic LFM3 model (`lfm3_model` in synthetic_lfm3.rs) at\n\
         # the last position of ORACLE_PROMPT, one per vocabulary id.\n\
         # llama.cpp {version}\n\
         # Regenerate: INFR_LLAMA_DEBUG=<llama-debug> cargo test -p infr-llama --test \
         synthetic_lfm3 -- --ignored synthetic_lfm3_regenerate_llama_cpp_fixture\n\
         # which runs: llama-debug -m <synthetic gguf> -p <ORACLE_PROMPT as text> --save-logits \
         --logits-output-dir <dir>\n\
         # tokens: {}\n",
        ids.join(" ")
    );
    for v in &logits {
        text.push_str(&format!("{v}\n"));
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(ORACLE_FIXTURE);
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!("wrote {} ({version})", path.display());
}

/// Run `exe` with `args` and no console window, panicking on a failure; returns its stdout and
/// stderr together (llama.cpp logs to stderr).
fn run_headless(exe: &str, args: &[&std::ffi::OsStr]) -> String {
    let mut cmd = std::process::Command::new(exe);
    cmd.args(args).stdin(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        /// `CREATE_NO_WINDOW` from the Win32 process-creation flags: a console program started
        /// from a test runner opens no window of its own. Other platforms have no such window.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let out = cmd.output().unwrap_or_else(|e| panic!("run {exe}: {e}"));
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success(),
        "{exe} failed ({}):\n{text}",
        out.status
    );
    text
}
