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
    /// Download + cache a model (`org/repo[:quant]` from HuggingFace, or a path to a `.gguf`).
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
        /// Combined prompt+gen turn `P,G` (matches llama-bench -pg): time ingesting P tokens THEN
        /// generating G; throughput = (P+G)/time. Models one coding-agent turn (read a file/tool
        /// result, then emit a reply). Overrides -p/-n when set.
        #[arg(long = "pg")]
        pg: Option<String>,
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
    /// Compare infr vs llama.cpp on coding-agent-shaped workloads (long context, replies at depth,
    /// whole turns). Shells out to `infr bench` and `llama-bench` with matching flags, same model+GPU.
    Compare {
        model: String,
        /// llama-bench device for the same GPU (override if device order differs).
        #[arg(long, default_value = "Vulkan0")]
        dev: String,
        /// Repetitions per measurement (reported value is the average).
        #[arg(short = 'r', long, default_value_t = 3)]
        reps: usize,
        /// Pin the ubatch (per-forward chunk) on both tools. 0 = each tool's own default.
        #[arg(short = 'u', long = "ubatch-size", default_value_t = 0)]
        ubatch: usize,
        /// Session depths / prefill sizes (coding-agent scale). Stay within the model's context.
        #[arg(long, value_delimiter = ',', default_values_t = [8000usize, 16000, 32000])]
        ctx: Vec<usize>,
        /// Reply length for the decode-at-depth scenario.
        #[arg(long, default_value_t = 256)]
        gen: usize,
        /// Session turns as `P,G` (ingest P tokens, generate G). Repeat the flag for several shapes.
        #[arg(long = "turn", default_values_t = ["2048,256".to_string(), "8192,512".to_string()])]
        turns: Vec<String>,
        /// Path to the llama-bench binary.
        #[arg(long, default_value = "llama-bench")]
        llama_bench: String,
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
            pg,
            batch,
            ubatch,
            reps,
            json,
        } => cmd_bench(&model, n_prompt, n_gen, depth, pg, ubatch, reps, json),
        Cmd::Compare {
            model,
            dev,
            reps,
            ubatch,
            ctx,
            gen,
            turns,
            llama_bench,
        } => cmd_compare(&model, &dev, reps, ubatch, &ctx, gen, &turns, &llama_bench),
    }
}

use anyhow::{anyhow, Context};
use std::path::{Path, PathBuf};

/// Resolve a model arg to (gguf_path, optional tokenizer_json_path).
/// Accept a path to a `.gguf` or an `org/repo[:quant]` HuggingFace ref resolved via infr-hub. The
/// tokenizer is the `tokenizer.json` beside the GGUF if present, else `None` → derived from the
/// GGUF's embedded vocab (HF Hub blobs are content-addressed with no sidecar).
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

    // Qwen3.5/3.6 (qwen35 / Qwen3-Next) run a hybrid path: linear projections on the GPU (f16),
    // the SSM conv + gated-delta recurrence + hd=256 attention on the CPU (no GPU kernels yet — see
    // docs/QWEN35.md). `Q35_CPU=1` forces the pure-CPU oracle. One-shot only for now.
    if infr_llama::qwen35::is_qwen35(&gguf) {
        let Some(msg) = message else {
            anyhow::bail!("qwen35 (Qwen3-Next) currently supports one-shot only: pass a message");
        };
        let mode = if std::env::var("Q35_CPU").is_ok() {
            "CPU oracle"
        } else {
            "hybrid GPU"
        };
        eprintln!("[qwen35 Qwen3-Next — {mode}: GPU linear + CPU SSM]");
        let t0 = std::time::Instant::now();
        let mut render = ThinkRender::new();
        let mut t_first: Option<std::time::Instant> = None;
        let (n_prompt, n_gen) = infr_llama::qwen35::generate_chat(&gguf, msg, max_new, |piece| {
            if t_first.is_none() {
                t_first = Some(std::time::Instant::now());
            }
            render.feed(piece);
        })?;
        render.finish();
        print_run_stats(t0, t_first, n_gen, n_prompt, None);
        return Ok(());
    }

    let llama = infr_llama::Llama::load_opt(&gguf, tok.as_deref())?;
    // Qwen3's recommended sampling — pure greedy makes thinking models degenerate (unterminated
    // <think>, no answer). Tune via INFR_TEMP / INFR_TOP_K / INFR_TOP_P.
    llama.set_sampling(
        envf("INFR_TEMP", 0.6),
        envu("INFR_TOP_K", 20),
        envf("INFR_TOP_P", 0.95),
    );

    // MoE (qwen3moe): eager CPU-orchestrated forward (router top-k + per-expert FFN), no KV cache.
    // One-shot only for now.
    if llama.is_moe() {
        let Some(m) = message else {
            anyhow::bail!("qwen3moe currently supports one-shot only: pass a message");
        };
        eprintln!("[qwen3moe — eager MoE forward: GPU matmuls + GPU KV cache + CPU router/top-k + auto-fit]");
        let prompt = llama.chatml(m);
        let t0 = std::time::Instant::now();
        let mut n = 0usize;
        let mut t_first: Option<std::time::Instant> = None;
        let mut render = ThinkRender::new();
        llama.generate_moe(&prompt, max_new, |piece| {
            if t_first.is_none() {
                t_first = Some(std::time::Instant::now());
            }
            n += 1;
            render.feed(piece);
        })?;
        render.finish();
        print_run_stats(t0, t_first, n, prompt.len(), None);
        return Ok(());
    }

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
#[allow(clippy::too_many_arguments)]
fn cmd_bench(
    model: &str,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    pg: Option<String>,
    ubatch: usize,
    reps: usize,
    json: bool,
) -> anyhow::Result<()> {
    // -pg "P,G": a coding-agent turn (ingest P then generate G); throughput = (P+G)/time.
    let pg = pg
        .map(|s| -> anyhow::Result<(usize, usize)> {
            let (p, g) = s.split_once(',').context("--pg expects `P,G`")?;
            Ok((p.trim().parse()?, g.trim().parse()?))
        })
        .transpose()?;
    let (gguf, tok) = resolve(model)?;
    let llama = infr_llama::Llama::load_opt(&gguf, tok.as_deref())?;
    let measure_tg = pg.is_none() && n_gen > 0;
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
    let cap = depth + pg.map_or(n_prompt + n_gen, |(p, g)| p + g) + 64;
    let mut samples = Vec::with_capacity(reps);
    // MoE models use the eager forward + GPU MoE KV cache; dense models the resident GPU forward.
    // Same prefill→measure shape, different forward/cache types, so the rep body is branched.
    if llama.is_moe() {
        let prefill = |llama: &infr_llama::Llama,
                       kv: &mut infr_llama::MoeKv,
                       count: usize|
         -> anyhow::Result<()> {
            let mut done = 0usize;
            while done < count {
                let pos = kv.len();
                let c = chunk(pos).min(count - done);
                llama.forward_moe_chunk(&dummy(pos, c), kv)?;
                done += c;
            }
            Ok(())
        };
        // Pipelines are compiled at load (Llama::warmup); the timed reps measure compute only.
        for _ in 0..reps {
            let mut kv = llama.new_moe_kv(cap)?;
            prefill(&llama, &mut kv, depth)?; // warm to `depth` (untimed)
            let t = std::time::Instant::now();
            if let Some((p, g)) = pg {
                prefill(&llama, &mut kv, p)?;
                for _ in 0..g {
                    llama.forward_moe_chunk(&[7u32], &mut kv)?;
                }
                samples.push((p + g) as f64 / t.elapsed().as_secs_f64());
            } else if measure_tg {
                for _ in 0..n_gen {
                    llama.forward_moe_chunk(&[7u32], &mut kv)?;
                }
                samples.push(n_gen as f64 / t.elapsed().as_secs_f64());
            } else {
                prefill(&llama, &mut kv, n_prompt)?;
                samples.push(n_prompt as f64 / t.elapsed().as_secs_f64());
            }
        }
    } else {
        // prefill `count` tokens at the cache head, chunked.
        let prefill = |kv: &mut infr_llama::KvCache, count: usize| -> anyhow::Result<()> {
            let mut done = 0usize;
            while done < count {
                let pos = kv.len();
                let c = chunk(pos).min(count - done);
                llama.forward_resident_kv(&dummy(pos, c), kv)?;
                done += c;
            }
            Ok(())
        };
        // Pipelines are compiled at load (Llama::warmup); the timed reps measure compute only.
        for _ in 0..reps {
            let mut kv = llama.new_kv(cap)?;
            prefill(&mut kv, depth)?; // warm to `depth` (untimed)
            let t = std::time::Instant::now();
            if let Some((p, g)) = pg {
                // coding-agent turn: time prompt ingest + reply generation together.
                prefill(&mut kv, p)?;
                for _ in 0..g {
                    llama.forward_resident_kv(&[7u32], &mut kv)?;
                }
                samples.push((p + g) as f64 / t.elapsed().as_secs_f64());
            } else if measure_tg {
                for _ in 0..n_gen {
                    llama.forward_resident_kv(&[7u32], &mut kv)?;
                }
                samples.push(n_gen as f64 / t.elapsed().as_secs_f64());
            } else {
                prefill(&mut kv, n_prompt)?;
                samples.push(n_prompt as f64 / t.elapsed().as_secs_f64());
            }
        }
    }
    let avg = samples.iter().sum::<f64>() / samples.len() as f64;
    if json {
        println!("[{{\"avg_ts\": {avg:.2}}}]");
    } else {
        let label = if let Some((p, g)) = pg {
            format!("pg{p}+{g}")
        } else if measure_tg {
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

/// Compare infr vs llama.cpp on coding-agent-shaped workloads. Shells out to `infr bench` (this same
/// binary) and `llama-bench` with matching flags, so both run the SAME model + GPU under one driver.
/// Scenarios (the target workload — see memory infr-optimization-priority):
///   • CONTEXT LOAD — cold prefill of a repo/file dump (pp at each ctx size)
///   • REPLY @depth — decode a `gen`-token reply with a session already in context (tg @ depth)
///   • SESSION TURN — ingest P then generate G at session depth (pg, the realistic per-turn unit)
#[allow(clippy::too_many_arguments)]
fn cmd_compare(
    model: &str,
    dev: &str,
    reps: usize,
    ubatch: usize,
    ctx: &[usize],
    gen: usize,
    turns: &[String],
    llama_bench: &str,
) -> anyhow::Result<()> {
    use std::process::Command;
    let exe = std::env::current_exe().context("locating the infr binary")?;
    let reps_s = reps.to_string();
    // infr and llama.cpp share the HF Hub cache and the same `org/repo:quant` ref grammar, so hand
    // BOTH tools the same reference: `infr bench` takes `model` verbatim, and llama-bench gets the
    // matching `-hf`/`--hf-file` (or `-m` for a local path). Pull once up front so `--offline` holds.
    let resolved = resolve(model)?.0; // ensures the model is cached; also the `-m` path for a local file
    let llama_model_args: Vec<String> = match infr_hub::ModelRef::parse(model)? {
        infr_hub::ModelRef::Repo { repo, sel } => {
            let mut a = vec!["--offline".to_string()];
            match sel {
                None => a.extend(["-hf".into(), repo]),
                Some(s) if s.to_lowercase().ends_with(".gguf") => {
                    a.extend(["--hf-repo".into(), repo, "--hf-file".into(), s])
                }
                Some(s) => a.extend(["-hf".into(), format!("{repo}:{s}")]),
            }
            a
        }
        infr_hub::ModelRef::Path(_) => {
            vec!["-m".to_string(), resolved.to_string_lossy().into_owned()]
        }
    };

    // Run `infr bench` (this binary) and read its single-row [{"avg_ts":X}].
    let infr_b = |args: &[&str]| -> anyhow::Result<f64> {
        let mut c = Command::new(&exe);
        c.arg("bench").arg(model).args(["-r", &reps_s]);
        if ubatch > 0 {
            c.args(["-u", &ubatch.to_string()]);
        }
        c.args(args).arg("--json");
        let out = c.output().context("running `infr bench`")?;
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).with_context(|| {
            format!(
                "parsing infr bench output: {}",
                String::from_utf8_lossy(&out.stdout)
            )
        })?;
        v[0]["avg_ts"]
            .as_f64()
            .context("infr bench: missing avg_ts")
    };

    // Run `llama-bench -o json` and pick the row matching (n_prompt, n_gen): -pg adds extra rows.
    let llama_b = |np: usize, ng: usize, args: &[&str]| -> Option<f64> {
        let mut c = Command::new(llama_bench);
        c.args(&llama_model_args);
        c.args([
            "-ngl", "99", "-dev", dev, "-fa", "auto", "-r", &reps_s, "-o", "json",
        ]);
        if ubatch > 0 {
            c.args(["-ub", &ubatch.to_string()]);
        }
        c.args(args);
        let out = c.output().ok()?;
        let rows: serde_json::Value = match serde_json::from_slice(&out.stdout) {
            Ok(v) => v,
            Err(_) => {
                eprintln!(
                    "llama-bench failed (np={np} ng={ng}): {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                );
                return None;
            }
        };
        rows.as_array()?.iter().find_map(|r| {
            let p = r["n_prompt"].as_u64()? as usize;
            let g = r["n_gen"].as_u64()? as usize;
            (p == np && g == ng).then(|| r["avg_ts"].as_f64())?
        })
    };

    let model_name = Path::new(model)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model);
    let ub_s = if ubatch > 0 {
        ubatch.to_string()
    } else {
        "tool-default".into()
    };
    println!("\nmodel: {model_name}   reps: {reps}   ubatch: {ub_s}");

    let row = |label: String, i: anyhow::Result<f64>, l: Option<f64>| {
        let is = i
            .as_ref()
            .map(|v| format!("{v:.0}"))
            .unwrap_or_else(|_| "ERR".into());
        let ls = l.map(|v| format!("{v:.0}")).unwrap_or_else(|| "NA".into());
        let ratio = match (i.as_ref().ok(), l) {
            (Some(&iv), Some(lv)) if lv > 0.0 => format!("{:.2}x", iv / lv),
            _ => "-".into(),
        };
        println!("{label:<17} | {is:>10} | {ls:>10} | {ratio:>8}");
    };
    let hdr = |title: &str| {
        println!(
            "\n{title:<17} | {:>10} | {:>10} | {:>8}",
            "infr", "llama.cpp", "infr/llama"
        );
        println!("{:-<18}+{:-<12}+{:-<12}+{:-<10}", "", "", "", "");
    };

    hdr("CONTEXT LOAD"); // cold prefill of a repo/file dump
    for &n in ctx {
        let np = n.to_string();
        row(
            format!("pp{n}"),
            infr_b(&["-p", &np, "-n", "0"]),
            llama_b(n, 0, &["-p", &np, "-n", "0"]),
        );
    }

    hdr("REPLY @depth"); // decode a reply with a session already in context
    let g = gen.to_string();
    for &d in ctx {
        let ds = d.to_string();
        row(
            format!("tg{gen}@{d}"),
            infr_b(&["-p", "0", "-n", &g, "-d", &ds]),
            llama_b(0, gen, &["-p", "0", "-n", &g, "-d", &ds]),
        );
    }

    for t in turns {
        let (p, gg) = t.split_once(',').context("--turn expects `P,G`")?;
        let (pn, gn): (usize, usize) = (p.trim().parse()?, gg.trim().parse()?);
        hdr(&format!("TURN {t}")); // ingest P then generate G at session depth
        for &d in ctx {
            let ds = d.to_string();
            row(
                format!("pg{t}@{d}"),
                infr_b(&["--pg", t, "-d", &ds]),
                llama_b(pn, gn, &["-p", "0", "-n", "0", "-pg", t, "-d", &ds]),
            );
        }
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
