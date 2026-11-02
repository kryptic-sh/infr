//! Run: cargo run -p infr-llama --example exl3_smoke --release -- <model-dir> [prompt]

use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let dir = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: exl3_smoke <model-dir> [prompt]"))?;
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The capital of France is".into());
    let dir = std::path::Path::new(&dir);
    let start = std::time::Instant::now();
    let model = infr_llama::SeamModel::load_with(
        &dir.join("model.safetensors"),
        None,
        Arc::new(infr_llama::EngineConfig::default()),
    )?;
    tracing::info!(
        seconds = start.elapsed().as_secs_f32(),
        "opened EXL3 model metadata"
    );
    let stats = model.generate_cpu(&prompt, 8, None, |piece| {
        // print-ok: generated text is the example's output.
        print!("{piece}");
        std::io::Write::flush(&mut std::io::stdout()).ok();
    })?;
    // print-ok: terminate the generated text line.
    println!();
    tracing::info!(
        ?stats,
        seconds = start.elapsed().as_secs_f32(),
        "EXL3 smoke completed"
    );
    Ok(())
}
