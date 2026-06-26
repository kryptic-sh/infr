//! Prefill throughput vs chunk size: process an N-token prompt in fixed CHUNK-size submits, timed.
//! One chunk size per run so a device-loss doesn't poison the next.
//! cargo run -p infr-llama --example prefill_bench --release -- <gguf> <tok> <N> <chunk>
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let gguf = args
        .next()
        .expect("usage: prefill_bench <gguf> <tok> <N> <chunk>");
    let tok = args.next().expect("need tokenizer.json");
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let chunk: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let llama = infr_llama::Llama::load(std::path::Path::new(&gguf), std::path::Path::new(&tok))?;

    let mut kv = llama.new_kv(n + 8)?;
    let t = std::time::Instant::now();
    let mut pos = 0usize;
    while pos < n {
        let c = chunk.min(n - pos);
        let toks: Vec<u32> = (0..c).map(|i| ((pos + i) % 100) as u32).collect();
        if let Err(e) = llama.forward_resident_kv(&toks, &mut kv) {
            println!("N={n} chunk={chunk:5}: DEVICE-LOST ({e})");
            return Ok(());
        }
        pos += c;
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "N={n} chunk={chunk:5}: {:.3}s  {:.0} tok/s  ({} submits)",
        dt,
        n as f64 / dt,
        n.div_ceil(chunk)
    );
    Ok(())
}
