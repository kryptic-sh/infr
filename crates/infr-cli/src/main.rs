//! `infr` CLI — `pull` / `run` / `serve`, all over the same engine + backend.
//! See PLAN.md "Product surface".
#![allow(dead_code, unused_variables)]

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "infr",
    version,
    about = "Pure-Rust, Vulkan-first LLM inference engine"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download + cache a model (hf:org/repo[:file] | ollama:name[:tag] | path).
    Pull { model: String },
    /// Interactive terminal chat (auto-pulls if missing).
    Run {
        model: String,
        /// Optional one-shot message (otherwise drop into a REPL).
        message: Option<String>,
    },
    /// Start the OpenAI-compatible HTTP API (auto-pulls if missing).
    Serve {
        model: String,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
    },
    /// Benchmark prefill/decode tok/s — same interface as llama.cpp's `llama-bench` (-p/-n/-d/-r),
    /// so the two are directly comparable. Prefill (pp) when -n 0; decode (tg) when -p 0.
    Bench {
        model: String,
        /// Prompt tokens to process (prefill). pp throughput = n_prompt / time.
        #[arg(short = 'p', long = "n-prompt", default_value_t = 512)]
        n_prompt: usize,
        /// Tokens to generate (decode). tg throughput = n_gen / time. Set -p 0 to measure decode.
        #[arg(short = 'n', long = "n-gen", default_value_t = 0)]
        n_gen: usize,
        /// Context depth pre-filled (untimed) before measuring — matches llama-bench -d.
        #[arg(short = 'd', long = "n-depth", default_value_t = 0)]
        depth: usize,
        /// Logical batch size (matches llama-bench -b). Accepted for flag-parity; the engine
        /// chunks by ubatch, so only -ub affects per-forward work.
        #[arg(short = 'b', long = "batch-size", default_value_t = 2048)]
        batch: usize,
        /// Physical batch = tokens per forward = our prefill chunk (matches llama-bench -ub). 0 =
        /// the engine's adaptive chunk policy; >0 pins the chunk so both tools sweep identically.
        #[arg(short = 'u', long = "ubatch-size", default_value_t = 0)]
        ubatch: usize,
        /// Repetitions (reported value is the average).
        #[arg(short = 'r', long, default_value_t = 3)]
        reps: usize,
        /// Emit `[{"avg_ts": X}]` (same shape as `llama-bench -o json`) for scripted comparison.
        #[arg(long)]
        json: bool,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Pull { model } => cmd_pull(&model),
        Cmd::Run { model, message } => cmd_run(&model, message.as_deref()),
        Cmd::Serve { model, addr } => cmd_serve(&model, &addr),
        Cmd::Bench {
            model,
            n_prompt,
            n_gen,
            depth,
            batch,
            ubatch,
            reps,
            json,
        } => cmd_bench(&model, n_prompt, n_gen, depth, ubatch, reps, json),
    }
}

use anyhow::{anyhow, Context};
use std::path::{Path, PathBuf};

/// Resolve a model arg to (gguf_path, optional tokenizer_json_path).
/// Accept a path to a `.gguf` or an `hf:`/`ollama:` ref resolved via infr-hub. The tokenizer is the
/// `tokenizer.json` beside the GGUF if present, else `None` → derived from the GGUF's embedded vocab
/// (ollama blobs are content-addressed with no sidecar).
fn resolve(model: &str) -> anyhow::Result<(PathBuf, Option<PathBuf>)> {
    let gguf = if Path::new(model).exists() {
        PathBuf::from(model)
    } else {
        let r = infr_hub::ModelRef::parse(model).map_err(|e| anyhow!("{e}"))?;
        infr_hub::ensure(&r).map_err(|e| anyhow!("{e}"))?
    };
    let tok = gguf
        .parent()
        .map(|d| d.join("tokenizer.json"))
        .filter(|p| p.exists());
    Ok((gguf, tok))
}

fn cmd_pull(model: &str) -> anyhow::Result<()> {
    let r = infr_hub::ModelRef::parse(model).map_err(|e| anyhow!("{e}"))?;
    let path = infr_hub::ensure(&r).map_err(|e| anyhow!("{e}"))?;
    println!("{}", path.display());
    Ok(())
}

/// Streams generated text, dimming the model's `<think>…</think>` reasoning so it's visually
/// distinct from the answer. Handles tags split across deltas via a small carry buffer; ANSI styling
/// only on a TTY.
struct ThinkRender {
    in_think: bool,
    carry: String,
    tty: bool,
}
impl ThinkRender {
    fn new() -> Self {
        use std::io::IsTerminal;
        Self {
            in_think: false,
            carry: String::new(),
            tty: std::io::stdout().is_terminal(),
        }
    }
    fn feed(&mut self, delta: &str) {
        use std::io::Write;
        if !self.tty {
            // Not a terminal: pass through (the literal <think>…</think> tags stay, so the reasoning
            // is still delimited in piped/redirected output).
            print!("{delta}");
            let _ = std::io::stdout().flush();
            return;
        }
        self.carry.push_str(delta);
        loop {
            let tag = if self.in_think { "</think>" } else { "<think>" };
            let Some(i) = self.carry.find(tag) else { break };
            let before = self.carry[..i].to_string();
            print!("{before}");
            self.carry.replace_range(..i + tag.len(), "");
            self.in_think = !self.in_think;
            if self.tty {
                // dim+italic on entering the think block; reset on leaving.
                print!(
                    "{}",
                    if self.in_think {
                        "\x1b[2;3m"
                    } else {
                        "\x1b[0m"
                    }
                );
            }
        }
        // Flush all but a tail that might be the start of a tag (keep < longest tag length).
        let keep = "</think>".len() - 1;
        if self.carry.len() > keep {
            let mut cut = self.carry.len() - keep;
            while cut > 0 && !self.carry.is_char_boundary(cut) {
                cut -= 1;
            }
            print!("{}", &self.carry[..cut]);
            self.carry.replace_range(..cut, "");
        }
        let _ = std::io::stdout().flush();
    }
    fn finish(&mut self) {
        use std::io::Write;
        print!("{}", self.carry);
        self.carry.clear();
        if self.in_think && self.tty {
            print!("\x1b[0m");
        }
        self.in_think = false;
        println!();
        let _ = std::io::stdout().flush();
    }
}

/// Print prefill / decode rates separately (like `ollama run --verbose`), splitting at the
/// first emitted token. `prefill` = prompt tokens over time-to-first-token; `decode` = the
/// remaining tokens over the time after the first. This avoids the misleading single amortized
/// rate, which folds prefill into decode and tanks for short generations.
fn print_run_stats(
    t0: std::time::Instant,
    t_first: Option<std::time::Instant>,
    n_gen: usize,
    prompt_toks: usize,
    ctx: Option<(usize, usize)>,
) {
    let now = std::time::Instant::now();
    let ttft = t_first.unwrap_or(now).duration_since(t0).as_secs_f32();
    let decode_dt = now.duration_since(t_first.unwrap_or(now)).as_secs_f32();
    let decode_n = n_gen.saturating_sub(1); // tokens produced after the first
    let pf_rate = if ttft > 0.0 {
        prompt_toks as f32 / ttft
    } else {
        0.0
    };
    let dec_rate = if decode_dt > 0.0 {
        decode_n as f32 / decode_dt
    } else {
        0.0
    };
    let ctxs = ctx
        .map(|(c, m)| format!(" | ctx {c}/{m}"))
        .unwrap_or_default();
    eprintln!(
        "[prefill {prompt_toks} tok @ {pf_rate:.0} tok/s ({:.0} ms) | decode {n_gen} tok @ {dec_rate:.1} tok/s{ctxs}]",
        ttft * 1000.0,
    );
}

fn cmd_run(model: &str, message: Option<&str>) -> anyhow::Result<()> {
    use std::io::Write;
    // Default 8192; override with INFR_MAX_CTX (e.g. 32768 to exercise high-ctx prefill).
    let max_ctx: usize = std::env::var("INFR_MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    #[allow(non_snake_case)]
    let MAX_CTX = max_ctx;
    let envf = |k: &str, d: f32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let envu = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    // Generation ceiling per reply (a turn also caps to remaining context). High enough for long
    // answers (lists/stories); override with INFR_MAX_NEW.
    let max_new = envu("INFR_MAX_NEW", 2048);
    let (gguf, tok) = resolve(model)?;
    let llama = infr_llama::Llama::load_opt(&gguf, tok.as_deref())?;
    // Qwen3's recommended sampling — pure greedy makes thinking models degenerate (unterminated
    // <think>, no answer). Tune via INFR_TEMP / INFR_TOP_K / INFR_TOP_P.
    llama.set_sampling(
        envf("INFR_TEMP", 0.6),
        envu("INFR_TOP_K", 20),
        envf("INFR_TOP_P", 0.95),
    );

    // One-shot message: a single chat turn (via the session path so user content is encoded safely).
    let mut session = llama.chat_session(MAX_CTX)?;
    if let Some(m) = message {
        let t0 = std::time::Instant::now();
        let mut n = 0usize;
        let mut t_first: Option<std::time::Instant> = None;
        let mut render = ThinkRender::new();
        session.turn(m, max_new, |piece| {
            if t_first.is_none() {
                t_first = Some(std::time::Instant::now());
            }
            n += 1;
            render.feed(piece);
        })?;
        render.finish();
        print_run_stats(t0, t_first, n, session.last_prompt_tokens(), None);
        return Ok(());
    }

    // REPL: a persistent chat session keeps prior turns in the KV cache (multi-turn context).
    let stdin = std::io::stdin();
    loop {
        print!("\n[ctx {}/{}] > ", session.ctx_len(), session.max_ctx());
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "exit" | "quit" | ":q" | ":quit") {
            break;
        }
        let t0 = std::time::Instant::now();
        let mut n = 0usize;
        let mut t_first: Option<std::time::Instant> = None;
        let mut render = ThinkRender::new();
        let res = session.turn(line, max_new, |piece| {
            if t_first.is_none() {
                t_first = Some(std::time::Instant::now());
            }
            n += 1;
            render.feed(piece);
        });
        render.finish();
        match res {
            Ok(_) => {
                print_run_stats(
                    t0,
                    t_first,
                    n,
                    session.last_prompt_tokens(),
                    Some((session.ctx_len(), session.max_ctx())),
                );
            }
            Err(e) => eprintln!("error: {e}"),
        }
    }
    Ok(())
}

/// Adapter: drive `infr-llama` through the server's `ChatGenerator` seam.
struct LlamaGenerator {
    llama: infr_llama::Llama,
}

impl infr_server::ChatGenerator for LlamaGenerator {
    fn chat(
        &mut self,
        messages: &[infr_engine::ChatMessage],
        _tools_json: Option<&str>,
        on_delta: &mut dyn FnMut(infr_engine::Delta),
    ) -> anyhow::Result<()> {
        // Bring-up: use the last user message; full chat-template/tools wiring comes later.
        let user = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();
        let prompt = self.llama.chatml(&user);
        self.llama.generate(&prompt, 256, |piece| {
            on_delta(infr_engine::Delta::Content(piece.to_string()));
        })?;
        Ok(())
    }
}

/// Benchmark prefill (pp) or decode (tg) tok/s with the same -p/-n/-d/-r interface as
/// `llama-bench`, so `infr bench` and `llama-bench` are directly comparable. Dummy tokens (timing
/// is data-independent), `prefill_chunk` policy for the prefill batching (the engine's real path).
fn cmd_bench(
    model: &str,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    ubatch: usize,
    reps: usize,
    json: bool,
) -> anyhow::Result<()> {
    let (gguf, tok) = resolve(model)?;
    let llama = infr_llama::Llama::load_opt(&gguf, tok.as_deref())?;
    let measure_tg = n_gen > 0;
    let dummy =
        |pos: usize, c: usize| -> Vec<u32> { (0..c).map(|i| ((pos + i) % 100) as u32).collect() };
    // ubatch>0 pins the prefill chunk (= llama-bench -ub); 0 = the engine's adaptive policy.
    let chunk = |pos: usize| {
        if ubatch > 0 {
            ubatch
        } else {
            llama.prefill_chunk(pos)
        }
    };
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut kv = llama.new_kv(depth + n_prompt + n_gen + 64)?;
        // warm to `depth` (untimed)
        let mut pos = 0usize;
        while pos < depth {
            let c = chunk(pos).min(depth - pos);
            llama.forward_resident_kv(&dummy(pos, c), &mut kv)?;
            pos += c;
        }
        let t = std::time::Instant::now();
        if measure_tg {
            for _ in 0..n_gen {
                llama.forward_resident_kv(&[7u32], &mut kv)?;
            }
            samples.push(n_gen as f64 / t.elapsed().as_secs_f64());
        } else {
            let mut done = 0usize;
            while done < n_prompt {
                let c = chunk(pos).min(n_prompt - done);
                llama.forward_resident_kv(&dummy(pos, c), &mut kv)?;
                pos += c;
                done += c;
            }
            samples.push(n_prompt as f64 / t.elapsed().as_secs_f64());
        }
    }
    let avg = samples.iter().sum::<f64>() / samples.len() as f64;
    if json {
        println!("[{{\"avg_ts\": {avg:.2}}}]");
    } else {
        let label = if measure_tg {
            format!("tg{n_gen}")
        } else {
            format!("pp{n_prompt}")
        };
        let d = if depth > 0 {
            format!(" @ d{depth}")
        } else {
            String::new()
        };
        println!("{label}{d}: {avg:.1} t/s  ({reps} reps)");
    }
    Ok(())
}

fn cmd_serve(model: &str, addr: &str) -> anyhow::Result<()> {
    let (gguf, tok) = resolve(model)?;
    let llama = infr_llama::Llama::load_opt(&gguf, tok.as_deref())?;
    let model_id = gguf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let sockaddr: std::net::SocketAddr = addr.parse().context("invalid --addr")?;
    let generator: Box<dyn infr_server::ChatGenerator> = Box::new(LlamaGenerator { llama });

    let rt = tokio::runtime::Runtime::new()?;
    println!("infr serve: {model_id} on http://{sockaddr}  (OpenAI /v1)");
    rt.block_on(infr_server::serve(generator, model_id, sockaddr))
}
