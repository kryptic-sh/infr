//! cargo run -p infr-llama --example run -- <model.gguf> <tokenizer.json> "<prompt>" [max_new]
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let gguf = args
        .next()
        .expect("usage: run <gguf> <tokenizer.json> <prompt> [max_new]");
    let tok = args.next().expect("need tokenizer.json path");
    let prompt = args
        .next()
        .unwrap_or_else(|| "What is the capital of France?".into());
    let max_new: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(48);

    let t0 = std::time::Instant::now();
    let llama = infr_llama::Llama::load(std::path::Path::new(&gguf), std::path::Path::new(&tok))?;
    eprintln!(
        "loaded {} layers in {:?}",
        llama.config().n_layer,
        t0.elapsed()
    );

    let full = llama.render_chat(&prompt)?;
    eprintln!("--- prompt ---\n{full}\n--- response ---");
    let t1 = std::time::Instant::now();
    let mut n = 0usize;
    let _ = llama.generate(&full, max_new, |piece| {
        print!("{piece}");
        let _ = std::io::stdout().flush();
        n += 1;
    })?;
    println!();
    let dt = t1.elapsed();
    eprintln!(
        "\n{} tokens in {:?} ({:.2} tok/s)",
        n,
        dt,
        n as f32 / dt.as_secs_f32()
    );
    Ok(())
}
