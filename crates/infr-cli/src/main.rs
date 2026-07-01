//! `infr` CLI — `pull` / `run` / `serve`, all over the same engine + backend.
//! See docs/PLAN.md "Product surface".
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
        /// GPU layers (matches llama-bench -ngl): >0 = Vulkan GPU forward; 0 = run on the CPU
        /// reference backend (no GPU), so `infr bench -ngl 0` is directly comparable to llama.cpp CPU.
        #[arg(long = "n-gpu-layers", visible_alias = "ngl", default_value_t = 999)]
        ngl: usize,
        /// CPU threads for the `-ngl 0` path (matches llama-bench -t). 0 = all cores.
        #[arg(short = 't', long, default_value_t = 0)]
        threads: usize,
        /// GPU device for the Vulkan forward (matches llama-bench --dev, e.g. Vulkan0/Vulkan1).
        #[arg(long, default_value = "Vulkan0")]
        dev: String,
        /// Emit `[{"avg_ts": X}]` (same shape as `llama-bench -o json`) for scripted comparison.
        #[arg(long)]
        json: bool,
    },
    /// Compare infr vs llama.cpp on coding-agent-shaped workloads (long context, replies at depth,
    /// whole turns). Shells out to `infr bench` and `llama-bench` with matching flags, same model+GPU.
    Compare {
        model: String,
        /// GPU device for both tools (matches llama-bench --dev; override if device order differs).
        #[arg(long, default_value = "Vulkan0")]
        dev: String,
        /// GPU layers for both tools (matches llama-bench -ngl): >0 = GPU; 0 = CPU comparison
        /// (infr CPU reference backend vs llama.cpp CPU). 0 lets `infr compare -ngl 0` bench CPU directly.
        #[arg(long = "n-gpu-layers", visible_alias = "ngl", default_value_t = 999)]
        ngl: usize,
        /// CPU threads for the -ngl 0 path on both tools (matches llama-bench -t). 0 = all cores.
        #[arg(short = 't', long, default_value_t = 0)]
        threads: usize,
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
            ngl,
            threads,
            dev,
            json,
        } => cmd_bench(
            &model, n_prompt, n_gen, depth, pg, ubatch, reps, ngl, threads, &dev, json,
        ),
        Cmd::Compare {
            model,
            dev,
            ngl,
            threads,
            reps,
            ubatch,
            ctx,
            gen,
            turns,
            llama_bench,
        } => cmd_compare(
            &model,
            &dev,
            ngl,
            threads,
            reps,
            ubatch,
            &ctx,
            gen,
            &turns,
            &llama_bench,
        ),
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
    // `pull` checks HF for the repo's latest commit and updates a stale cache (run/serve stay
    // cache-first via `ensure`). Offline → falls back to the cached copy.
    let path = infr_hub::ensure_latest(&r).map_err(|e| anyhow!("{e}"))?;
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
    // Context window: default to the model's own trained context length (`<arch>.context_length`);
    // override with INFR_MAX_CTX (e.g. shrink to fit VRAM, or extend to exercise high-ctx prefill).
    let ctx_override: Option<usize> = std::env::var("INFR_MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok());
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

    // Build the per-backend generation primitive (`ChatModel`), then wrap it in the ONE shared `Chat`
    // (infr_llama::model) that owns history + `<think>`-stripping and drives the single REPL below:
    // INFR_CPU (dense/MoE or qwen35 on the agnostic compute graph, no Vulkan/VRAM), qwen35 GPU,
    // qwen3moe, dense Qwen3/Llama/Gemma. Every backend now does history-based multi-turn — no
    // per-arch one-shot special-case. The CLI owns the Llama; the boxed trait object borrows it (so
    // the borrow-based dense `ChatSession` needs no ownership change).
    let is_q35 = infr_llama::qwen35::is_qwen35(&gguf);
    let llama; // declared here so a borrowing ChatSession / MoeChat outlives `chat`
    let model: Box<dyn infr_llama::model::ChatModel + '_> = if std::env::var("INFR_METAL").is_ok() {
        if is_q35 {
            eprintln!(
                "[metal backend — qwen35/Qwen3-Next on the agnostic seam, Apple GPU (reference)]"
            );
            Box::new(infr_llama::model::CpuQwen35Chat::new_metal(gguf.clone()))
        } else {
            eprintln!(
                "[metal backend — dense/MoE forward on Apple GPU via the agnostic compute graph (reference)]"
            );
            Box::new(infr_llama::model::CpuDenseChat::new_metal(
                infr_llama::CpuModel::load(&gguf, tok.as_deref())?,
            ))
        }
    } else if std::env::var("INFR_CPU").is_ok() {
        if is_q35 {
            eprintln!("[cpu backend — qwen35/Qwen3-Next on the agnostic seam, no GPU]");
            Box::new(infr_llama::model::CpuQwen35Chat::new(gguf.clone()))
        } else {
            eprintln!(
                "[cpu backend — dense/MoE forward on CPU via the agnostic compute graph, no GPU]"
            );
            Box::new(infr_llama::model::CpuDenseChat::new(
                infr_llama::CpuModel::load(&gguf, tok.as_deref())?,
            ))
        }
    } else if is_q35 {
        let mode = if std::env::var("Q35_CPU").is_ok() {
            "CPU oracle"
        } else {
            "GPU linear + SSM + attention"
        };
        eprintln!("[qwen35 Qwen3-Next — {mode}]");
        Box::new(infr_llama::model::Qwen35Chat::new(gguf.clone()))
    } else {
        llama = infr_llama::Llama::load_opt(&gguf, tok.as_deref())?;
        // Qwen3's recommended sampling — pure greedy makes thinking models degenerate
        // (unterminated <think>, no answer). Tune via INFR_TEMP / INFR_TOP_K / INFR_TOP_P.
        llama.set_sampling(
            envf("INFR_TEMP", 0.6),
            envu("INFR_TOP_K", 20),
            envf("INFR_TOP_P", 0.95),
        );
        if llama.is_moe() {
            eprintln!("[qwen3moe — eager MoE forward: GPU matmuls + GPU KV cache + CPU router/top-k + auto-fit]");
            Box::new(infr_llama::model::MoeChat::new(&llama))
        } else {
            // Honor the model's default context length; INFR_MAX_CTX overrides it.
            let max_ctx = ctx_override.unwrap_or_else(|| llama.config().n_ctx_train);
            eprintln!(
                "[ctx {max_ctx}{}]",
                if ctx_override.is_some() {
                    " (override)"
                } else {
                    " (model default)"
                }
            );
            Box::new(llama.chat_session(max_ctx)?)
        }
    };
    let mut chat = infr_llama::model::Chat::new(model);

    // One-shot (a message) or an interactive multi-turn REPL (every backend now supports it).
    if let Some(m) = message {
        run_chat_turn(&mut chat, m, max_new)?;
        return Ok(());
    }
    let stdin = std::io::stdin();
    loop {
        match chat.repl_status() {
            Some(s) => print!("\n[{s}] > "),
            None => print!("\n> "),
        }
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
        if let Err(e) = run_chat_turn(&mut chat, line, max_new) {
            eprintln!("error: {e}");
        }
    }
    Ok(())
}

/// Run one chat turn through the shared [`Chat`]: stream pieces via the `<think>` renderer, then
/// print the prefill/decode stats line.
fn run_chat_turn(
    chat: &mut infr_llama::model::Chat,
    message: &str,
    max_new: usize,
) -> anyhow::Result<()> {
    let mut render = ThinkRender::new();
    let stats = chat.turn(message, max_new, &mut |p| render.feed(p))?;
    render.finish();
    let rate = |n: usize, s: f64| if s > 0.0 { n as f64 / s } else { 0.0 };
    eprintln!(
        "[prefill {} tok @ {:.0} tok/s ({:.0} ms) | decode {} tok @ {:.1} tok/s]",
        stats.n_prompt,
        rate(stats.n_prompt, stats.prompt_secs),
        stats.prompt_secs * 1000.0,
        stats.n_gen,
        rate(stats.n_gen, stats.decode_secs),
    );
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
        tools_json: Option<&str>,
        tool_choice: Option<&str>,
        on_delta: &mut dyn FnMut(infr_engine::Delta),
    ) -> anyhow::Result<()> {
        // Render the FULL conversation (system/user/assistant-with-tool_calls/tool results) + the
        // request's tool spec through the model's own chat template — the template emits each model's
        // native tool syntax, so infr never hardcodes a format.
        let tools: Option<serde_json::Value> = tools_json
            .map(serde_json::from_str)
            .transpose()
            .context("parsing request `tools`")?;
        let prompt = self.llama.render_chat_oai(messages, tools.as_ref())?;
        // INFR_DUMP_REQ=<file>: append the exact rendered prompt for each request (debugging the
        // serve-only garbage). Replay it verbatim to reproduce/root-cause outside the server.
        if let Ok(path) = std::env::var("INFR_DUMP_REQ") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = write!(
                    f,
                    "===REQ n_msgs={} has_tools={} chars={}===\n{prompt}\n===END===\n",
                    messages.len(),
                    tools.is_some(),
                    prompt.len(),
                );
            }
        }
        let max_new = std::env::var("INFR_MAX_NEW")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048usize);

        // For tool_choice "required"/named, build a grammar constraint that FORCES a valid,
        // schema-conforming tool call (llama.cpp-parity reliability). "auto"/"none"/absent → None.
        if let Some(mut constraint) = self.llama.tool_constraint(tools.as_ref(), tool_choice)? {
            // Forced path: prime the assistant turn with the `<tool_call>` opener and grammar-constrain
            // the JSON body. On any grammar/parser error, OR if the constrained body is empty/unparseable
            // (e.g. the toktrie bridge masked out the whole call), fall through to unconstrained
            // generation rather than failing the request or returning an empty `stop`.
            let primed = format!("{prompt}<tool_call>\n");
            let emitted =
                match self
                    .llama
                    .generate_ids(&primed, max_new, Some(&mut constraint), |_| {})
                {
                    Ok(ids) => {
                        let body = self.llama.decode_ids(&ids, false)?;
                        // The constrained output IS the JSON call object — parse it straight; strip any
                        // trailing `</tool_call>` past the grammar.
                        let body = body.trim().trim_end_matches("</tool_call>").trim();
                        match serde_json::from_str::<serde_json::Value>(body) {
                            Ok(val) => val.get("name").and_then(|v| v.as_str()).map(|name| {
                                let arguments = val
                                    .get("arguments")
                                    .cloned()
                                    .unwrap_or(serde_json::json!({}));
                                on_delta(infr_engine::Delta::ToolCall {
                                    name: name.to_string(),
                                    arguments: serde_json::to_string(&arguments)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                });
                            }),
                            Err(_) => None,
                        }
                        .is_some()
                    }
                    Err(e) => {
                        eprintln!("[tools] grammar-constrained generation failed ({e})");
                        false
                    }
                };
            if emitted {
                return Ok(());
            }
            eprintln!("[tools] forced tool call produced no parseable call; falling back to unconstrained");
            // fall through to the unconstrained path below
        }

        // Auto path: unconstrained, STREAMED. Each decoded piece is routed live — `<think>…</think>`
        // → Reasoning deltas, the rest → Content deltas — via `ChatStream`, which holds back a
        // marker-length tail so a partial `<think>`/`</think>`/`<tool_call>` never streams. `tool_choice
        // "none"` forbids tool calls (all output is content); otherwise, once a `<tool_call>` opener
        // appears the tail is buffered and parsed into ToolCall deltas at finish().
        let mut stream = ChatStream::new(tool_choice != Some("none"));
        {
            let od = &mut *on_delta;
            self.llama
                .generate_ids(&prompt, max_new, None, |piece: &str| {
                    stream.push(piece, &mut *od)
                })?;
        }
        stream.finish(on_delta);
        Ok(())
    }
}

/// Incremental splitter for the streaming server path. Accumulates the raw decoded text and, on each
/// piece, emits the newly-stable Reasoning (`<think>…</think>`) and Content deltas — holding back a
/// marker-length tail so a partial `<think>`/`</think>`/`<tool_call>` marker is never emitted. Once a
/// `<tool_call>` opener appears (and tool calls are allowed) it stops streaming content; `finish()`
/// flushes the held-back tails and parses the buffered tool call(s) into ToolCall deltas.
struct ChatStream {
    raw: String,
    sent_r: usize, // reasoning bytes already emitted (offset within the reasoning region)
    sent_c: usize, // content bytes already emitted (offset within the content region)
    allow_tools: bool,
}

impl ChatStream {
    fn new(allow_tools: bool) -> Self {
        Self {
            raw: String::new(),
            sent_r: 0,
            sent_c: 0,
            allow_tools,
        }
    }

    fn push(&mut self, piece: &str, on_delta: &mut dyn FnMut(infr_engine::Delta)) {
        self.raw.push_str(piece);
        self.emit(false, on_delta);
    }

    fn finish(&mut self, on_delta: &mut dyn FnMut(infr_engine::Delta)) {
        self.emit(true, on_delta); // flush the held-back tails (no marker can still be forming)
        if self.allow_tools {
            let (_r, body) = infr_engine::split_think(&self.raw);
            let (_content, calls) = infr_engine::parse_hermes_tool_calls(&body);
            for call in calls {
                on_delta(infr_engine::Delta::ToolCall {
                    name: call.name,
                    arguments: serde_json::to_string(&call.arguments)
                        .unwrap_or_else(|_| "{}".to_string()),
                });
            }
        }
    }

    fn emit(&mut self, final_flush: bool, on_delta: &mut dyn FnMut(infr_engine::Delta)) {
        const TO: &str = "<think>";
        const TC: &str = "</think>";
        const TL: &str = "<tool_call>";
        let raw = &self.raw;
        let think_open = raw.find(TO);
        let think_close = raw.find(TC);
        let tool_open = if self.allow_tools { raw.find(TL) } else { None };
        // Reasoning region: between `<think>` and `</think>` (or end, while still thinking).
        if let Some(to) = think_open {
            let rs = to + TO.len();
            let (r_end, hold) = match think_close {
                Some(tc) => (tc, false),
                None => (raw.len(), !final_flush),
            };
            emit_region(raw, rs, r_end, hold, &mut self.sent_r, true, on_delta);
        }
        // Content region: after `</think>` (or from the start when there's no `<think>` at all), up to
        // a `<tool_call>` opener (whose block is buffered, not streamed) or the end.
        let c_start = match think_close {
            Some(tc) => Some(tc + TC.len()),
            None if think_open.is_none() => Some(0),
            None => None,
        };
        if let Some(cs) = c_start {
            let (c_end, hold) = match tool_open {
                Some(t) if t >= cs => (t, false),
                _ => (raw.len(), !final_flush),
            };
            emit_region(raw, cs, c_end, hold, &mut self.sent_c, false, on_delta);
        }
    }
}

/// Emit the not-yet-sent slice of `raw[region_start .. region_end]` (past `*sent` bytes), holding back
/// a marker-length tail when `hold` (so a partial marker isn't emitted mid-stream), clamped to a UTF-8
/// boundary. Advances `*sent`.
fn emit_region(
    raw: &str,
    region_start: usize,
    region_end: usize,
    hold: bool,
    sent: &mut usize,
    reasoning: bool,
    on_delta: &mut dyn FnMut(infr_engine::Delta),
) {
    const HOLD: usize = 12; // > the longest marker, so a partial one never streams
    let abs = region_start + *sent;
    if abs >= region_end {
        return;
    }
    let mut end = if hold {
        region_end.saturating_sub(HOLD).max(abs)
    } else {
        region_end
    };
    while end > abs && !raw.is_char_boundary(end) {
        end -= 1;
    }
    if end <= abs {
        return;
    }
    let text = &raw[abs..end];
    if text.is_empty() {
        return;
    }
    on_delta(if reasoning {
        infr_engine::Delta::Reasoning(text.to_owned())
    } else {
        infr_engine::Delta::Content(text.to_owned())
    });
    *sent += text.len();
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
    ngl: usize,
    threads: usize,
    dev: &str,
    json: bool,
) -> anyhow::Result<()> {
    // -t: pin the CPU thread count (the backend's rayon pool reads RAYON_NUM_THREADS on first use).
    // Must be set before any parallel work spins the pool up — do it here, before the model loads.
    if threads > 0 {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }
    // -pg "P,G": a coding-agent turn (ingest P then generate G); throughput = (P+G)/time.
    let pg = pg
        .map(|s| -> anyhow::Result<(usize, usize)> {
            let (p, g) = s.split_once(',').context("--pg expects `P,G`")?;
            Ok((p.trim().parse()?, g.trim().parse()?))
        })
        .transpose()?;
    let (gguf, tok) = resolve(model)?;
    // qwen35 (Qwen3-Next): a hybrid gated-DeltaNet + GQA model with a bespoke per-token forward —
    // it is NOT a `Llama` (dense/MoE), so `load_opt` below can't load or run it. Route it through
    // its own loader + `forward`, reusing the same pp/tg/pg warmup-then-time methodology.
    if infr_llama::qwen35::is_qwen35(&gguf) {
        return cmd_bench_qwen35(&gguf, n_prompt, n_gen, depth, pg, reps, ngl == 0, json);
    }
    // INFR_GPU_SEAM=1: bench the dense forward on the Vulkan backend THROUGH THE AGNOSTIC SEAM
    // instead of the production Recorder path. An eligible qwen3-style dense decode now records the
    // graph ONCE and replays it per token (params-driven `_dyn` kernels), matching the production
    // Recorder's record-once behavior. Reuses -p/-n/-r; reports pp/tg like the others.
    if std::env::var("INFR_GPU_SEAM").is_ok() {
        let model = infr_llama::CpuModel::load(&gguf, tok.as_deref())?;
        let mut pps = Vec::new();
        let mut tgs = Vec::new();
        for _ in 0..reps.max(1) {
            let s = model.bench_vulkan(n_prompt, n_gen)?;
            if s.prompt_secs > 0.0 {
                pps.push(s.n_prompt as f64 / s.prompt_secs);
            }
            if s.decode_secs > 0.0 {
                tgs.push(s.n_gen as f64 / s.decode_secs);
            }
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        };
        println!(
            "seam (record-once replay): pp={:.1} tok/s  tg={:.1} tok/s  (p={n_prompt} n={n_gen} r={reps})",
            med(pps),
            med(tgs),
        );
        return Ok(());
    }
    // -ngl 0: run on the CPU reference backend (no GPU), comparable to `llama-bench -ngl 0`.
    if ngl == 0 {
        return cmd_bench_cpu(
            &gguf,
            tok.as_deref(),
            n_prompt,
            n_gen,
            depth,
            pg,
            reps,
            json,
        );
    }
    let _ = dev; // GPU device selection: VulkanBackend uses the default adapter (--dev reserved for parity).
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

/// CPU-backend bench (`infr bench -ngl 0`): the GPU bench's pp/tg/pg metrics on the agnostic CPU
/// reference path, using `CpuModel`'s token-level timing — directly comparable to `llama-bench -ngl 0`.
#[allow(clippy::too_many_arguments)]
fn cmd_bench_cpu(
    gguf: &Path,
    tok: Option<&Path>,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    pg: Option<(usize, usize)>,
    reps: usize,
    json: bool,
) -> anyhow::Result<()> {
    let model = infr_llama::CpuModel::load(gguf, tok)?;
    let measure_tg = pg.is_none() && n_gen > 0;
    // One untimed warmup (page-cache the mmap'd weights) before the timed reps.
    let _ = model.bench(depth.max(1), if measure_tg || pg.is_some() { 1 } else { 0 });
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let ts = if let Some((p, g)) = pg {
            let s = model.bench(p, g)?; // pg: prefill p + decode g, throughput (p+g)/total
            (p + g) as f64 / (s.prompt_secs + s.decode_secs)
        } else if measure_tg {
            let s = model.bench(depth.max(1), n_gen)?; // tg@depth: decode at `depth` context
            n_gen as f64 / s.decode_secs
        } else {
            let s = model.bench(n_prompt, 0)?; // pp: prefill only
            n_prompt as f64 / s.prompt_secs
        };
        samples.push(ts);
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
        println!("{label}{d} [cpu]: {avg:.1} t/s  ({reps} reps)");
    }
    Ok(())
}

/// qwen35 (Qwen3-Next) bench: the hybrid gated-DeltaNet + GQA forward is a bespoke per-token path
/// (no batched-prefill kernel, and not a [`infr_llama::Llama`]), so it can't ride the dense/MoE
/// bench above. Same pp/tg/pg methodology as [`cmd_bench`] — warm the pipelines outside the timed
/// region, then time prefill and decode SEPARATELY — driven through the model's public
/// [`infr_llama::qwen35::Model::forward`]. Dummy tokens (timing is data-independent). `cpu` selects
/// the `Q35_CPU=1` reference forward (`-ngl 0`) vs the GPU-resident forward.
#[allow(clippy::too_many_arguments)]
fn cmd_bench_qwen35(
    gguf: &Path,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    pg: Option<(usize, usize)>,
    reps: usize,
    cpu: bool,
    json: bool,
) -> anyhow::Result<()> {
    // -ngl 0: force the pure-CPU oracle forward (no Vulkan), matching `llama-bench -ngl 0`.
    if cpu {
        std::env::set_var("Q35_CPU", "1");
    }
    let model = infr_llama::qwen35::load_path(gguf)?;
    let measure_tg = pg.is_none() && n_gen > 0;
    let dummy = |i: usize| (i % 100) as u32;
    // Untimed warmup: one forward on a throwaway state compiles the lazy GPU pipelines (and pages in
    // the weights), keeping that one-time cost out of the timed reps — the qwen35 analogue of the
    // dense path's `Llama::warmup` at load.
    {
        let mut st = model.new_state();
        let _ = model.forward(dummy(0), &mut st);
    }
    let mut samples = Vec::with_capacity(reps.max(1));
    for _ in 0..reps.max(1) {
        // Fresh state per rep; advance it to `depth` (untimed) so tg is measured at that context.
        let mut st = model.new_state();
        for i in 0..depth {
            let _ = model.forward(dummy(i), &mut st);
        }
        let t = std::time::Instant::now();
        let ts = if let Some((p, g)) = pg {
            // coding-agent turn: time prompt ingest + reply generation together.
            for i in 0..p {
                let _ = model.forward(dummy(depth + i), &mut st);
            }
            for _ in 0..g {
                let _ = model.forward(7u32, &mut st);
            }
            (p + g) as f64 / t.elapsed().as_secs_f64()
        } else if measure_tg {
            for _ in 0..n_gen {
                let _ = model.forward(7u32, &mut st);
            }
            n_gen as f64 / t.elapsed().as_secs_f64()
        } else {
            for i in 0..n_prompt {
                let _ = model.forward(dummy(depth + i), &mut st);
            }
            n_prompt as f64 / t.elapsed().as_secs_f64()
        };
        samples.push(ts);
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
        let tag = if cpu { " [cpu]" } else { "" };
        println!("{label}{d}{tag}: {avg:.1} t/s  ({reps} reps)");
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
    ngl: usize,
    threads: usize,
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
    let ngl_s = ngl.to_string();
    let threads_s = threads.to_string();
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
        c.args(["--ngl", &ngl_s, "--dev", dev]);
        if threads > 0 {
            c.args(["-t", &threads_s]);
        }
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
            "-ngl", &ngl_s, "-dev", dev, "-fa", "auto", "-r", &reps_s, "-o", "json",
        ]);
        if threads > 0 {
            c.args(["-t", &threads_s]);
        }
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
    // Qwen3's recommended sampling — pure greedy makes thinking models degenerate (unterminated
    // `<think>`, repeated tokens). Mirrors `cmd_run`; tune via INFR_TEMP / INFR_TOP_K / INFR_TOP_P.
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
    llama.set_sampling(
        envf("INFR_TEMP", 0.6),
        envu("INFR_TOP_K", 20),
        envf("INFR_TOP_P", 0.95),
    );
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

#[cfg(test)]
mod chat_stream_tests {
    use super::*;
    use infr_engine::Delta;

    /// Feed `pieces` through a `ChatStream` and collect every emitted delta (streaming + finish).
    fn run(pieces: &[&str], allow_tools: bool) -> (String, String, Vec<(String, String)>) {
        let mut out: Vec<Delta> = Vec::new();
        let mut s = ChatStream::new(allow_tools);
        {
            let mut od = |d: Delta| out.push(d);
            for p in pieces {
                s.push(p, &mut od);
            }
        }
        s.finish(&mut |d: Delta| out.push(d));
        let (mut r, mut c, mut t) = (String::new(), String::new(), Vec::new());
        for d in &out {
            match d {
                Delta::Reasoning(x) => r.push_str(x),
                Delta::Content(x) => c.push_str(x),
                Delta::ToolCall { name, arguments } => t.push((name.clone(), arguments.clone())),
            }
        }
        (r, c, t)
    }

    #[test]
    fn streams_think_then_content() {
        let (r, c, t) = run(
            &["<think>", "reason", "ing", "</think>", "the ", "answer"],
            true,
        );
        assert_eq!(r.trim(), "reasoning");
        assert_eq!(c.trim(), "the answer");
        assert!(t.is_empty());
    }

    #[test]
    fn plain_content_no_think() {
        let (r, c, _) = run(&["hello ", "world, this is a longer reply"], true);
        assert!(r.trim().is_empty());
        assert_eq!(c.trim(), "hello world, this is a longer reply");
    }

    #[test]
    fn tool_call_buffered_and_parsed() {
        let (r, _c, t) = run(
            &[
                "<think>plan</think>",
                "<tool_call>\n{\"name\": \"run_bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>",
            ],
            true,
        );
        assert_eq!(r.trim(), "plan");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].0, "run_bash");
        assert!(t[0].1.contains("ls"));
    }

    #[test]
    fn tool_choice_none_keeps_tool_text_as_content() {
        // allow_tools=false → a <tool_call> block is NOT extracted; it stays content.
        let (_r, c, t) = run(&["<tool_call>{\"name\":\"x\"}</tool_call>"], false);
        assert!(t.is_empty());
        assert!(c.contains("tool_call"));
    }
}
