//! GPU logits within tolerance of the CPU reference (`docs/gpu-tolerance.md`).
//!
//! Each case runs the CPU reference backend (`kernels.cpu.reference`) greedily on a prompt, then
//! teacher-forces the very same tokens through the Vulkan seam and compares the two backends'
//! full-vocabulary logits at every position — the batched prefill's rows (at two chunk heights)
//! and one decode step per generated token — with `infr_llama::tolerance`'s metrics, against the
//! tolerance table for the model's path class. Each case then generates greedily on the
//! production path and holds the chained and the replayed decode to that per-token one, and the
//! tokens to the reference (`support/greedy.rs`).
//!
//! ```text
//! INFR_DEV=Vulkan0 cargo nextest run -p infr-llama --release --test gpu_tolerance \
//!   --run-ignored all --test-threads 1 --no-capture
//! ```

#[path = "support/forced.rs"]
mod forced;

/// The planted-bug demonstrations, built only with the `mutations` feature.
#[cfg(feature = "mutations")]
#[path = "planted/mod.rs"]
mod planted;

#[test]
fn forced_run_reproduces_the_cpu_reference_on_every_path() {
    forced::self_check_on_a_synthetic_llama();
}

#[test]
fn forced_run_reads_back_and_takes_a_routing_on_the_cpu() {
    forced::self_check_on_a_synthetic_moe();
}

/// A measurement, not a gate: CPU production and every Vulkan configuration against the CPU
/// reference on the models named in `INFR_TOL_MODELS` (keys of [`forced::MODELS`]), printed as one
/// line per pair and phase (see [`forced::measure`]).
#[test]
#[ignore = "measurement: needs cached models and a Vulkan GPU"]
fn measure_gpu_tolerance() {
    let Ok(keys) = std::env::var("INFR_TOL_MODELS") else {
        eprintln!("skip: INFR_TOL_MODELS unset");
        return;
    };
    for key in keys.split(',') {
        forced::measure(key);
    }
}

use infr_llama::tolerance::PathClass::{self, Float, FloatNoCoopmat, Int8, KvQ8};

/// A quantized model: the production kernels at both prefill chunk heights, and the same with
/// every int8-activation kernel off, once more on the static per-token decode path.
const QUANT_RUNS: &[(&str, PathClass)] = &[
    ("gpu", Int8),
    ("gpu-ub32", Int8),
    ("gpu-float", Float),
    ("gpu-float-ub32", Float),
    ("gpu-float-static", Float),
];

/// The int8-free runs once more with the cooperative-matrix tier off, for a model whose Float row
/// cannot be tight on the f16 operands of the cooperative-matrix prefill GEMMs (gemma-3): the
/// prefill GEMMs then take f32 operands on every device, at both chunk heights and on the static
/// per-token decode path.
const NOCOOPMAT_RUNS: &[(&str, PathClass)] = &[
    ("gpu-float-nocoopmat", FloatNoCoopmat),
    ("gpu-float-nocoopmat-ub32", FloatNoCoopmat),
    ("gpu-float-nocoopmat-static", FloatNoCoopmat),
];

/// A float-weight model: the production kernels at both prefill chunk heights, and the static
/// per-token decode path.
const FLOAT_RUNS: &[(&str, PathClass)] = &[
    ("gpu", Float),
    ("gpu-ub32", Float),
    ("gpu-float-static", Float),
];

/// A routed-expert model with float expert banks: the production kernels on the replayed
/// per-token decode, and on the static one.
const MOE_RUNS: &[(&str, PathClass)] = &[("gpu", Float), ("gpu-float-static", Float)];

/// The red path, in the normal test job: a synthetic model with one projection's weights scaled
/// by [`RED_SCALE`], driven on the CPU reference against the unmodified model's CPU reference, must
/// fail the gate's evaluation, naming the limits it crossed; the unmodified model against itself
/// must pass it. CI runs this on every system, so a gate that cannot go red fails here.
///
/// How sensitive that is: on the CPU a 1.01 scale of the same tensor trips the gate and 1.008 and
/// below pass (measured in review), so [`RED_SCALE`] sits about twice the threshold above 1. The
/// green control is the reference against itself, a distance of exactly 0; it shows the
/// evaluation accepts a match, not that it tolerates legitimate noise (the GPU cases do that).
#[test]
fn tolerance_gate_goes_red_on_a_scaled_weight_cpu() {
    let clean = forced::open_case("synthetic-f32").expect("synthetic model");
    let scaled = forced::open_case_scaled("synthetic-f32", Some((RED_TENSOR, RED_SCALE)))
        .expect("synthetic model");
    let runs = [("gpu", Float)];
    let on = infr_llama::seam::ForcedOn::CpuReference;
    let green = forced::evaluate("synthetic-f32", &clean, &clean, on, &runs);
    assert!(green.is_empty(), "the clean model failed: {green:?}");
    let red = forced::evaluate("synthetic-f32", &clean, &scaled, on, &runs);
    eprintln!("the scaled model's failures:\n  {}", red.join("\n  "));
    assert!(
        red.iter().any(|f| f.contains("KL")),
        "a {RED_SCALE}x {RED_TENSOR} passed the gate: {red:?}"
    );
}

/// The tensor the red-path tests scale, and by how much: one layer's FFN down projection, by 2%.
const RED_TENSOR: &str = "blk.1.ffn_down.weight";
const RED_SCALE: f32 = 1.02;

/// The red path on the GPU: the scaled synthetic model on the Vulkan seam, against the unmodified
/// model's CPU reference, must fail the gate.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_tolerance_goes_red_on_a_scaled_weight() {
    let _guard = forced::watchdog("the red-path case", forced::GATE_TIMEOUT);
    let clean = forced::open_case("synthetic-f32").expect("synthetic model");
    let scaled = forced::open_case_scaled("synthetic-f32", Some((RED_TENSOR, RED_SCALE)))
        .expect("synthetic model");
    if infr_testkit::gpu_or_skip(infr_vulkan::VulkanBackend::new_with(std::sync::Arc::new(
        forced::env_cfg(),
    )))
    .is_none()
    {
        eprintln!("skip: no Vulkan device");
        return;
    }
    let on = infr_llama::seam::ForcedOn::Vulkan;
    let red = forced::evaluate("synthetic-f32", &clean, &scaled, on, &[("gpu", Float)]);
    eprintln!("the scaled model's failures:\n  {}", red.join("\n  "));
    assert!(
        red.iter().any(|f| f.contains("KL")),
        "a {RED_SCALE}x {RED_TENSOR} passed the gate on the GPU: {red:?}"
    );
}

/// The prefill head the harness adds is a vocabulary-wide GEMM production never runs; it must not
/// change anything else: the decode rows, which read the KV the prefill wrote, are bit-identical
/// with the head on every row, on each chunk's last row (`INFR_TOL_HEAD=last`) and off.
/// The chunk height is forced to [`forced::SMALL_UBATCH`] so the prompt spans several chunks and
/// the first decode row follows a multi-chunk prefill.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_tolerance_prefill_head_leaves_decode_unchanged() {
    use infr_llama::seam::{ForcedOn, PrefillHead};
    let _guard = forced::watchdog("the prefill-head check", forced::GATE_TIMEOUT);
    let case = forced::open_case("synthetic-f32").expect("synthetic model");
    if infr_testkit::gpu_or_skip(infr_vulkan::VulkanBackend::new_with(std::sync::Arc::new(
        forced::env_cfg(),
    )))
    .is_none()
    {
        eprintln!("skip: no Vulkan device");
        return;
    }
    let ids = &case.prompts[2];
    let r = forced::forced_reference(&case.model, ids, forced::N_DECODE);
    assert!(
        ids.len() > 2 * forced::SMALL_UBATCH,
        "the prompt must span several chunks"
    );
    let small_chunks =
        |c: &mut infr_llama::EngineConfig| c.device.ubatch = Some(forced::SMALL_UBATCH);
    let decode = |head| {
        let g = forced::forced_with(&case.model, ForcedOn::Vulkan, small_chunks, head, &r);
        let v = g.vocab;
        g.logits[g.prefill_rows * v..]
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    };
    let without = decode(PrefillHead::Off);
    assert_eq!(without.len(), forced::N_DECODE * r.vocab);
    // Where two decodes first differ: (decode row, token), or None when bit-identical.
    let first_diff = |a: &[u32]| {
        (0..a.len())
            .find(|&i| a[i] != without[i])
            .map(|i| (i / r.vocab, i % r.vocab))
    };
    // A second run without the head first: if the decode itself is not run-to-run
    // deterministic on this device, that is what failed, not the head.
    let again = first_diff(&decode(PrefillHead::Off));
    assert!(
        again.is_none(),
        "two runs WITHOUT the prefill head differ (first at decode row, token {again:?}): the \
         GPU decode is not deterministic here"
    );
    for head in [PrefillHead::All, PrefillHead::LastOfChunk] {
        let d = first_diff(&decode(head));
        assert!(
            d.is_none(),
            "the prefill head {head:?} changed the decode rows' bits (first at decode row, \
             token {d:?})"
        );
    }
}

/// What makes a kernel an int8-activation one, by its name: the dp4a `mmq` GEMMs, the int8
/// coopmat GEMMs (`i8cm`), the dp4a K-quant GEMMs, the activation quantizer, and the int8
/// GEMVs (`native_mmv_*`; the float GEMVs are `native_gemv_*` and `native_<dtype>*`). Typed by
/// hand from the kernel names: the Vulkan crate has no central list of int8-activation kernels,
/// so a new one named otherwise would pass unseen (B207).
const INT8_KERNEL_MARKERS: &[&str] = &["mmq", "i8", "dp4a", "quant_q8", "native_mmv"];

/// The Float class is int8-free: a Float run of a quantized model builds no int8-activation
/// kernel, while the shipped configuration on the same model does (so the check can fail).
/// Kernels are built lazily, when first recorded, so the built set is a superset of what ran and
/// the negative check is the conservative one.
#[test]
#[ignore = "needs Qwen3-0.6B Q4_K_M and a Vulkan GPU"]
fn gpu_seam_tolerance_float_runs_build_no_int8_kernel() {
    use infr_llama::seam::ForcedOn;
    let _guard = forced::watchdog("the int8-free check", forced::GATE_TIMEOUT);
    let Some(case) = forced::open_case("qwen3-0.6b") else {
        eprintln!("skip: qwen3-0.6b not in the HF cache");
        return;
    };
    if infr_testkit::gpu_or_skip(infr_vulkan::VulkanBackend::new_with(std::sync::Arc::new(
        forced::env_cfg(),
    )))
    .is_none()
    {
        eprintln!("skip: no Vulkan device");
        return;
    }
    let r = forced::forced_reference(&case.model, &case.prompts[1], 4);
    let int8 = |k: &str| INT8_KERNEL_MARKERS.iter().any(|m| k.contains(m));
    let shipped = forced::forced(&case.model, ForcedOn::Vulkan, |_| {}, &r).kernels;
    let float = forced::forced(&case.model, ForcedOn::Vulkan, forced::vulkan_float, &r).kernels;
    eprintln!("shipped: {shipped:?}\nfloat: {float:?}");
    assert!(
        shipped.iter().any(|k| int8(k)),
        "the shipped run built no int8 kernel, so this check cannot fail: {shipped:?}"
    );
    let bad: Vec<_> = float.iter().filter(|k| int8(k)).collect();
    assert!(bad.is_empty(), "the Float run built int8 kernels: {bad:?}");
}

/// What makes a kernel a cooperative-matrix prefill GEMM, by its name. Typed by hand, like
/// [`INT8_KERNEL_MARKERS`] and with the same gap: one named otherwise would pass unseen (B207).
const COOPMAT_GEMM_MARKER: &str = "gemm_warp";

/// The FloatNoCoopmat class runs no cooperative-matrix prefill GEMM: such a run of gemma-3 builds
/// none, while the Float run on the same model does wherever the device has the tier (so the
/// check can fail there; on a device without it the two runs build the same kernels, which is
/// printed and is all that can be said).
#[test]
#[ignore = "needs gemma-3-1b-it Q4_K_M and a Vulkan GPU"]
fn gpu_seam_tolerance_nocoopmat_runs_build_no_coopmat_gemm() {
    use infr_llama::seam::ForcedOn;
    let _guard = forced::watchdog("the coopmat-free check", forced::GATE_TIMEOUT);
    let Some(case) = forced::open_or_skip("gemma3-1b") else {
        return;
    };
    if forced::device_for("gemma3-1b", case.bytes).is_none() {
        return;
    }
    let r = forced::forced_reference(&case.model, &case.prompts[1], 4);
    let coopmat = |k: &str| k.contains(COOPMAT_GEMM_MARKER);
    let float = forced::forced(&case.model, ForcedOn::Vulkan, forced::vulkan_float, &r).kernels;
    let without = forced::forced(
        &case.model,
        ForcedOn::Vulkan,
        forced::vulkan_float_nocoopmat,
        &r,
    )
    .kernels;
    eprintln!("float: {float:?}\nwithout the tier: {without:?}");
    let bad: Vec<_> = without.iter().filter(|k| coopmat(k)).collect();
    assert!(
        bad.is_empty(),
        "the run with the cooperative-matrix tier off built its GEMMs: {bad:?}"
    );
    if float.iter().any(|k| coopmat(k)) {
        assert_ne!(float, without, "the tier's switch changed no kernel");
    } else {
        eprintln!(
            "this device's Float run builds no cooperative-matrix GEMM: the two classes run the \
             same kernels here, and the check above could not have failed"
        );
        assert_eq!(
            float, without,
            "the tier's switch changed kernels it does not name"
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_tolerance_synthetic_f32() {
    forced::gate("synthetic-f32", FLOAT_RUNS);
}

#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_tolerance_synthetic_f16() {
    forced::gate("synthetic-f16", FLOAT_RUNS);
}

/// The synthetic routed-expert model, forced along the reference's expert selections, on the
/// replayed and on the static per-token decode; its own selections are held to the row's routing
/// near tie. It takes no batched prefill, so only its decode rows are scored.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_tolerance_synthetic_moe() {
    forced::gate(forced::SYNTHETIC_MOE, MOE_RUNS);
}

/// What a forced run reads back as a router's own selection is the selection its expert layer
/// runs on when left to itself, and running on a supplied selection is the same arithmetic: a
/// Vulkan run left on its own routers, and the same run forced along the selections that run
/// read back, give the same logits bit for bit and read back the same selections, on the
/// replayed and on the static decode. (`Op::MoeRoute` repeats the router dispatch of the
/// `Op::MoeFfn` lowering; this fails when the two part.) The control: with one of those
/// selections replaced by an expert the router did not pick, at the first decode position, the
/// run's bits must change, or the supplied selection is not what ran.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_tolerance_own_routing_is_the_selection_it_runs() {
    use infr_llama::seam::{ForcedOn, PrefillHead};
    let _guard = forced::watchdog("the own-routing check", forced::GATE_TIMEOUT);
    let case = forced::open_case(forced::SYNTHETIC_MOE).expect("synthetic model");
    if infr_testkit::gpu_or_skip(infr_vulkan::VulkanBackend::new_with(std::sync::Arc::new(
        forced::env_cfg(),
    )))
    .is_none()
    {
        eprintln!("skip: no Vulkan device");
        return;
    }
    let r = forced::forced_reference(&case.model, &case.prompts[2], forced::N_DECODE);
    let bits = |g: &infr_llama::seam::ForcedLogits| {
        g.logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    };
    for &(name, _) in MOE_RUNS {
        let edit = forced::gpu_config_edit(name);
        let run = |route| {
            forced::forced_along(
                &case.model,
                ForcedOn::Vulkan,
                edit,
                PrefillHead::All,
                &r,
                route,
            )
        };
        let own = run(None);
        let routing = own.routing.as_ref().expect("a routed model's routing");
        let again = run(Some(routing));
        assert!(
            bits(&again) == bits(&own),
            "{name}: forced along its own selections, the run's logits changed"
        );
        assert_eq!(again.routing, own.routing, "{name}: its selections moved");
        let mut other = routing.clone();
        let at = r.positions[r.prefill_rows] * other.layers.len() * other.n_used;
        let picked = other.ids[at..at + other.n_used].to_vec();
        other.ids[at] = (0..other.n_expert as u32)
            .find(|e| !picked.contains(e))
            .expect("more experts than a token uses");
        assert!(
            bits(&run(Some(&other))) != bits(&own),
            "{name}: forced along another selection, the run's logits did not change"
        );
    }
}

#[test]
#[ignore = "needs Qwen3-0.6B Q4_K_M and a Vulkan GPU"]
fn gpu_seam_tolerance_qwen3() {
    let mut runs = QUANT_RUNS.to_vec();
    runs.push(("gpu-kv-q8-float", KvQ8));
    forced::gate("qwen3-0.6b", &runs);
}

#[test]
#[ignore = "needs gemma-3-1b-it Q4_K_M and a Vulkan GPU"]
fn gpu_seam_tolerance_gemma3() {
    let mut runs = QUANT_RUNS.to_vec();
    runs.extend_from_slice(NOCOOPMAT_RUNS);
    forced::gate("gemma3-1b", &runs);
}

#[test]
#[ignore = "needs Qwen3.5-0.8B Q4_K_M and a Vulkan GPU"]
fn gpu_seam_tolerance_qwen35() {
    forced::gate("qwen35-0.8b", QUANT_RUNS);
}

#[test]
#[ignore = "needs LFM3-1.7B Q4_K_M and F16 and a Vulkan GPU"]
fn gpu_seam_tolerance_lfm3() {
    forced::gate("lfm3-q4", QUANT_RUNS);
    forced::gate("lfm3-f16", FLOAT_RUNS);
}

#[test]
#[ignore = "needs ERNIE-4.5-0.3B Q4_K_M and F16 and a Vulkan GPU"]
fn gpu_seam_tolerance_ernie() {
    forced::gate("ernie-0.3b-q4", QUANT_RUNS);
    forced::gate("ernie-0.3b-f16", FLOAT_RUNS);
}
