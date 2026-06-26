//! Decode-rate-vs-context benchmark.
//! cargo run -p infr-llama --example decode_bench --release -- <model.gguf> <tokenizer.json>
//!
//! Fills the KV cache to various depths with dummy tokens (content irrelevant for timing), then
//! times single-token decodes at each depth — isolating decode throughput at a given context.
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let gguf = args
        .next()
        .expect("usage: decode_bench <gguf> <tokenizer.json>");
    let tok = args.next().expect("need tokenizer.json");
    let llama = infr_llama::Llama::load(std::path::Path::new(&gguf), std::path::Path::new(&tok))?;

    // Prefill ONCE incrementally; measure decode throughput at each milestone (so the O(n²)
    // prefill is paid a single time, not per depth).
    let milestones = [128usize, 1024, 4096, 8000, 12000, 16000];
    let max_ctx = 16384;
    let mut kv = llama.new_kv(max_ctx)?;
    let mut pos = 0usize;
    for &depth in &milestones {
        while pos < depth {
            let chunk = (depth - pos).min(256);
            let toks: Vec<u32> = (0..chunk).map(|i| ((pos + i) % 100) as u32).collect();
            let _ = llama.forward_resident_kv(&toks, &mut kv)?;
            pos += chunk;
        }
        let iters = 24;
        let t = std::time::Instant::now();
        for i in 0..iters {
            let _ = llama.forward_resident_kv(&[(i % 100) as u32], &mut kv)?;
            pos += 1;
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        println!(
            "ctx={depth:5}: {:.3} ms/token  {:.1} tok/s",
            dt * 1e3,
            1.0 / dt
        );
    }
    Ok(())
}
