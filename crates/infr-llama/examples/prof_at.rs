//! Per-op profile at a given context depth. Prefills to N, then profiles ONE deep prefill chunk
//! and ONE decode step at depth N (run with INFR_PROF2=1 to get the [prof2] breakdowns).
//! cargo run -p infr-llama --example prof_at --release -- <gguf> <tok> <N>
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let gguf = args.next().expect("usage: prof_at <gguf> <tok> <N>");
    let tok = args.next().expect("need tokenizer.json");
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32768);
    let llama = infr_llama::Llama::load(std::path::Path::new(&gguf), std::path::Path::new(&tok))?;

    // Prefill quietly to N - (realistic adaptive chunk), then profile that final chunk + a decode.
    let kv_target = n + 64;
    let mut kv = llama.new_kv(kv_target)?;
    let final_chunk = llama.prefill_chunk(n.saturating_sub(1)).min(n);
    let warm = n - final_chunk;
    let mut pos = 0usize;
    while pos < warm {
        let c = llama.prefill_chunk(pos).min(warm - pos);
        let toks: Vec<u32> = (0..c).map(|i| ((pos + i) % 100) as u32).collect();
        let _ = llama.forward_resident_kv(&toks, &mut kv)?;
        pos += c;
    }
    eprintln!(
        "\n========== PREFILL CHUNK at ctx~{pos} ({} tokens) ==========",
        n - pos
    );
    let toks: Vec<u32> = (0..(n - pos)).map(|i| ((pos + i) % 100) as u32).collect();
    let _ = llama.forward_resident_kv(&toks, &mut kv)?;

    eprintln!("\n========== DECODE STEP at ctx~{n} ==========");
    for _ in 0..3 {
        let _ = llama.forward_resident_kv(&[7u32], &mut kv)?;
    }
    Ok(())
}
