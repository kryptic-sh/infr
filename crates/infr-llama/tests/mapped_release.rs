//! A model resident on a DISCRETE GPU does not keep its mapped file pages in the host's resident
//! set (`paging.release_uploaded`).
//!
//! Without the release, every page the upload read stays in the process working set for as long as
//! the model is loaded, and on Windows those pages count as in use: the host loses about the
//! model's size in available memory beside the copy in VRAM. The test reads the mechanism directly,
//! as the residency of every tensor's mapped bytes (`TensorBytes::resident_bytes`), with the
//! release off (it must be high, or the probe sees nothing) and on (it must be low, during and
//! after a generation), then checks the released mapping still reads the file's bytes.
//!
//! Run: `INFR_DEV=Vulkan0 cargo nextest run --release -p infr-llama --test mapped_release
//! --run-ignored all`. Self-skips on an integrated GPU, where nothing is released.

use infr_core::WeightSource;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Locate a cached GGUF the way `infr` does, or `None` (the test self-skips).
fn find_gguf(repo: &str, file: &str) -> Option<PathBuf> {
    infr_hub::Store::discover().ok()?.snapshot_file(repo, file)
}

/// The device these tests load on (`INFR_DEV`, else the default pick).
fn engine_cfg(release: bool) -> infr_llama::EngineConfig {
    let mut cfg = infr_llama::EngineConfig::default();
    cfg.device.dev = std::env::var("INFR_DEV").ok();
    cfg.paging.release_uploaded = release;
    infr_testkit::honour_env(&mut cfg);
    cfg
}

/// Bytes of the model's tensors that are in this process's resident set, and their total size.
fn tensor_residency(g: &infr_gguf::Gguf) -> (u64, u64) {
    let (mut resident, mut total) = (0u64, 0u64);
    for t in g.tensors() {
        let view = g.tensor_bytes_arc(&t.name).expect("tensor view");
        resident += view.resident_bytes().expect("residency probe");
        total += view.len() as u64;
    }
    (resident, total)
}

/// Load the model at `path` with the release on or off, generate on a fresh Vulkan session, and
/// return the generated text with the residency fraction after the first generation and after a
/// second one on the same session (the steady state).
fn run(path: &Path, release: bool) -> (String, f64, f64, infr_llama::SeamModel) {
    let model =
        infr_llama::SeamModel::load_with(path, None, std::sync::Arc::new(engine_cfg(release)))
            .expect("model load");
    let mut session = model.vulkan_session(512).expect("vulkan session");
    let mut text = String::new();
    let prompt = model
        .render_chat("What is 2+2? Answer briefly.")
        .expect("render");
    model
        .generate_vulkan_session(&mut session, &prompt, 8, None, |p| text.push_str(p))
        .expect("first generation");
    let (r1, total) = tensor_residency(model.gguf());
    let prompt2 = model.render_chat("Name three colours.").expect("render");
    model
        .generate_vulkan_session(&mut session, &prompt2, 16, None, |_| {})
        .expect("second generation");
    let (r2, _) = tensor_residency(model.gguf());
    drop(session);
    (
        text,
        r1 as f64 / total as f64,
        r2 as f64 / total as f64,
        model,
    )
}

#[test]
#[ignore = "requires a Vulkan GPU and Qwen3-1.7B: run with --run-ignored on a GPU box"]
fn gpu_seam_resident_upload_releases_mapped_pages() {
    let Some(path) = find_gguf("unsloth--Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf") else {
        eprintln!("skip: Qwen3-1.7B not in the HF cache");
        return;
    };
    let probe_cfg = engine_cfg(true);
    let Some(dev) = infr_vulkan::VulkanBackend::enumerate_devices(&probe_cfg)
        .expect("enumerate devices")
        .into_iter()
        .find(|d| d.is_default_pick)
    else {
        panic!("no default Vulkan device");
    };
    if dev.integrated {
        eprintln!(
            "skip: Vulkan{} ({}) is unified memory; nothing is released there",
            dev.index, dev.name
        );
        return;
    }

    // Off first, and dropped before the second load: each model has its own mapping, and the
    // probe reads only the mapping of the model it is given.
    let (kept_text, kept, kept_steady, kept_model) = run(&path, false);
    drop(kept_model);
    eprintln!(
        "release off: {:.1}% resident, {:.1}% after a second turn",
        kept * 100.0,
        kept_steady * 100.0
    );
    assert!(
        kept > 0.5,
        "with the release off the uploaded weights should stay resident; the probe saw {:.1}%",
        kept * 100.0
    );

    let (text, released, steady, model) = run(&path, true);
    eprintln!(
        "release on: {:.1}% resident, {:.1}% after a second turn",
        released * 100.0,
        steady * 100.0
    );
    assert!(
        released < 0.05,
        "{:.1}% of the weights' mapped bytes are still resident after a resident upload",
        released * 100.0
    );
    assert!(
        steady < 0.05,
        "decode faulted the released weights back in: {:.1}% resident after a second turn",
        steady * 100.0
    );
    assert_eq!(
        text, kept_text,
        "releasing the mapped pages changed the generation"
    );

    // The mapping is still valid and still the file: every tensor reads the bytes at its file range.
    let g = model.gguf();
    let mut file = std::fs::File::open(&path).expect("open model file");
    for t in g.tensors() {
        let view = g.tensor_bytes_arc(&t.name).expect("tensor view");
        let (off, len) = view.file_range();
        let mut want = vec![0u8; len];
        file.seek(SeekFrom::Start(off)).expect("seek");
        file.read_exact(&mut want).expect("read");
        assert!(
            view[..] == want[..],
            "{} reads differently after its release",
            t.name
        );
    }
}
