//! Reliable per-stage breakdown of ONE real prefill chunk at depth N. Warms KV quietly (profiling
//! OFF), then enables INFR_PROF2 for exactly the final chunk so the [prof2] block is that chunk
//! alone (prof_at profiles every forward, mixing warm chunks of different sizes). Run twice:
//!   prefill_stage_prof <gguf> <tok> <N>                   # overlap mode: total == true GPU time
//!   INFR_FULLBARRIER=1 prefill_stage_prof <gguf> <tok> <N># serialized: clean per-op durations
//! Compare the two totals to see overlap; compare overlap-total to wall-clock to confirm it reconciles.
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let gguf = args
        .next()
        .expect("usage: prefill_stage_prof <gguf> <tok> <N>");
    let tok = args.next().expect("need tokenizer.json");
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32768);
    // Warm with profiling OFF so only the final chunk prints a [prof2] block.
    std::env::remove_var("INFR_PROF2");
    let llama = infr_llama::Llama::load(std::path::Path::new(&gguf), std::path::Path::new(&tok))?;
    let mut kv = llama.new_kv(n + 64)?;
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
        "\n===== ONE PREFILL CHUNK: m={final_chunk} at ctx~{pos} (kv_len={}) =====",
        pos
    );
    // Profile exactly this chunk (recorder reads INFR_PROF2 at construction).
    std::env::set_var("INFR_PROF2", "1");
    let toks: Vec<u32> = (0..final_chunk).map(|i| ((pos + i) % 100) as u32).collect();
    let t = std::time::Instant::now();
    let _ = llama.forward_resident_kv(&toks, &mut kv)?;
    eprintln!(
        "[wall] chunk forward incl. submit+wait: {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}
