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
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,

    /// Print shell completions to stdout and exit (packaging helper).
    #[arg(long, value_enum, value_name = "SHELL", hide = true)]
    completions: Option<CompletionShell>,

    /// Print the man page (troff) to stdout and exit (packaging helper).
    #[arg(long, hide = true)]
    man: bool,
}

/// Shells `--completions` can generate for: clap_complete's five core shells plus nushell
/// (separate generator crate). Mirrors gpur's packaging-helper flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CompletionShell {
    Bash,
    Zsh,
    Fish,
    Powershell,
    Elvish,
    Nushell,
}

impl CompletionShell {
    fn generate(self, cmd: &mut clap::Command) {
        use clap_complete::Shell;
        let out = &mut std::io::stdout();
        match self {
            CompletionShell::Bash => clap_complete::generate(Shell::Bash, cmd, "infr", out),
            CompletionShell::Zsh => clap_complete::generate(Shell::Zsh, cmd, "infr", out),
            CompletionShell::Fish => clap_complete::generate(Shell::Fish, cmd, "infr", out),
            CompletionShell::Powershell => {
                clap_complete::generate(Shell::PowerShell, cmd, "infr", out)
            }
            CompletionShell::Elvish => clap_complete::generate(Shell::Elvish, cmd, "infr", out),
            CompletionShell::Nushell => {
                clap_complete::generate(clap_complete_nushell::Nushell, cmd, "infr", out)
            }
        }
    }
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
        /// Model(s): one for the deep coding-agent scenarios, several with --sweep.
        #[arg(num_args = 1.., required = true)]
        models: Vec<String>,
        /// Survey mode: for EVERY model given, measure pp512 / tg128 / tg@depth on both tools and
        /// print a gap matrix + the worst ratios — the recurring "where are we behind llama.cpp"
        /// sweep. Without it, the single model gets the deep coding-agent scenarios below.
        #[arg(long)]
        sweep: bool,
        /// Decode depth for the sweep's at-depth metric.
        #[arg(long, default_value_t = 4096)]
        sweep_depth: usize,
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

    // Packaging helpers (hidden): emit completions / man page and exit. The man page includes
    // every subcommand and flag straight from the clap definitions, so it can't drift.
    if let Some(shell) = cli.completions {
        use clap::CommandFactory;
        shell.generate(&mut Cli::command());
        return Ok(());
    }
    if cli.man {
        use clap::CommandFactory;
        clap_mangen::Man::new(Cli::command()).render(&mut std::io::stdout())?;
        return Ok(());
    }

    let Some(cmd) = cli.cmd else {
        // arg_required_else_help covers the bare invocation; a flag-only call lands here.
        use clap::CommandFactory;
        Cli::command().print_help()?;
        return Ok(());
    };
    match cmd {
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
            models,
            sweep,
            sweep_depth,
            dev,
            ngl,
            threads,
            reps,
            ubatch,
            ctx,
            gen,
            turns,
            llama_bench,
        } => {
            if sweep {
                cmd_compare_sweep(
                    &models,
                    sweep_depth,
                    &dev,
                    ngl,
                    threads,
                    reps,
                    ubatch,
                    &llama_bench,
                )
            } else {
                if models.len() > 1 {
                    anyhow::bail!(
                        "compare without --sweep takes ONE model (got {}); pass --sweep for the                          multi-model survey",
                        models.len()
                    );
                }
                cmd_compare(
                    &models[0],
                    &dev,
                    ngl,
                    threads,
                    reps,
                    ubatch,
                    &ctx,
                    gen,
                    &turns,
                    &llama_bench,
                )
            }
        }
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

/// Streams generated text through THE shared reasoning splitter (`infr_engine::ChatStream` — the
/// same one `infr serve` emits OpenAI deltas from), dimming the reasoning so it's visually
/// distinct from the answer. Because run and serve consume one splitter, every thinking model is
/// exposed identically on both surfaces — a new reasoning format lands in `infr-chat` once. On a
/// non-TTY the raw model text passes through untouched (markers preserved for piped output).
struct ThinkRender {
    split: infr_engine::ChatStream,
    in_think: bool,
    tty: bool,
}
impl ThinkRender {
    fn new() -> Self {
        use std::io::IsTerminal;
        Self {
            split: infr_engine::ChatStream::new(false),
            in_think: false,
            tty: std::io::stdout().is_terminal(),
        }
    }
    fn feed(&mut self, delta: &str) {
        use std::io::Write;
        if !self.tty {
            print!("{delta}");
            let _ = std::io::stdout().flush();
            return;
        }
        let in_think = &mut self.in_think;
        self.split.push(delta, &mut |d| Self::render(d, in_think));
        let _ = std::io::stdout().flush();
    }
    fn finish(&mut self) {
        use std::io::Write;
        if self.tty {
            let in_think = &mut self.in_think;
            self.split.finish(&mut |d| Self::render(d, in_think));
            if *in_think {
                print!("[0m");
                self.in_think = false;
            }
        }
        println!();
        let _ = std::io::stdout().flush();
    }
    /// Style transitions ride the delta KIND: entering Reasoning dims+italicizes, entering
    /// Content resets. The splitter already stripped the markers.
    fn render(d: infr_engine::Delta, in_think: &mut bool) {
        match d {
            infr_engine::Delta::Reasoning(t) => {
                if !*in_think {
                    print!("[2;3m");
                    *in_think = true;
                }
                print!("{t}");
            }
            infr_engine::Delta::Content(t) => {
                if *in_think {
                    print!("[0m");
                    *in_think = false;
                }
                print!("{t}");
            }
            infr_engine::Delta::ToolCall { .. } => {}
        }
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
    // Chat-default sampling for every backend (the bespoke branch reads the same envs below).
    set_default_sampling_env();
    let model: Box<dyn infr_llama::model::ChatModel + '_> = if std::env::var("INFR_METAL").is_ok() {
        if is_q35 {
            eprintln!(
                "[metal backend — qwen35/Qwen3-Next on the agnostic seam, Apple GPU (reference)]"
            );
            Box::new(infr_llama::model::Qwen35Chat::new_metal(gguf.clone()))
        } else {
            eprintln!(
                "[metal backend — dense/MoE forward on Apple GPU via the agnostic compute graph, persistent KV session]"
            );
            #[cfg(target_os = "macos")]
            {
                // INFR_SPEC_DRAFT=<gguf>: speculative decoding — a small same-tokenizer draft
                // proposes k tokens (INFR_SPEC_K, default 4), one batched target forward
                // verifies. Greedy-only; pays off for ~8B-class targets (issue #16).
                if let Ok(draft_path) = std::env::var("INFR_SPEC_DRAFT") {
                    let target = infr_llama::CpuModel::load(&gguf, tok.as_deref())?;
                    let draft =
                        infr_llama::CpuModel::load(std::path::Path::new(&draft_path), None)?;
                    // Verify cost is nearly flat in k (one batched forward), so a larger k
                    // amortizes it; 8 balances the draft cost against acceptance decay.
                    let k = std::env::var("INFR_SPEC_K")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(8);
                    let (tb, db) = (
                        std::fs::metadata(&gguf).map(|m| m.len()).unwrap_or(0),
                        std::fs::metadata(&draft_path).map(|m| m.len()).unwrap_or(0),
                    );
                    if db * 4 > tb {
                        eprintln!(
                            "[spec] warning: draft is more than 1/4 the target's size — \
                             speculation only pays when the target is much larger (see #16)"
                        );
                    }
                    std::env::set_var("INFR_TEMP", "0");
                    eprintln!(
                        "[metal spec — target + {k}-token draft verify, greedy (INFR_TEMP=0)]"
                    );
                    Box::new(infr_llama::model::SpecMetalChat::new(target, draft, k))
                } else {
                    Box::new(infr_llama::model::MetalSeamChat::new(
                        infr_llama::CpuModel::load(&gguf, tok.as_deref())?,
                    ))
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                Box::new(infr_llama::model::CpuDenseChat::new_metal(
                    infr_llama::CpuModel::load(&gguf, tok.as_deref())?,
                ))
            }
        }
    } else if std::env::var("INFR_CPU").is_ok() {
        if is_q35 {
            eprintln!("[cpu backend — qwen35/Qwen3-Next on the agnostic seam, no GPU]");
            Box::new(infr_llama::model::Qwen35Chat::new_cpu(gguf.clone()))
        } else {
            eprintln!(
                "[cpu backend — dense/MoE forward on CPU via the agnostic compute graph, no GPU]"
            );
            Box::new(infr_llama::model::CpuDenseChat::new(
                infr_llama::CpuModel::load(&gguf, tok.as_deref())?,
            ))
        }
    } else if is_q35 {
        eprintln!("[qwen35 Qwen3-Next — Vulkan agnostic seam]");
        Box::new(infr_llama::model::Qwen35Chat::new(gguf.clone()))
    } else {
        // The default: dense/MoE on the VULKAN agnostic seam — persistent multi-slot KV sessions
        // (per-turn suffix-only prefill), record-once decode replay, MoE expert auto-fit.
        eprintln!("[vulkan seam — dense/MoE on the agnostic compute graph, persistent KV session]");
        Box::new(infr_llama::model::DenseSeamChat::new(
            infr_llama::CpuModel::load(&gguf, tok.as_deref())?,
        ))
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

/// Serve adapter for the seam-backed [`ChatModel`]s (qwen35 on any backend, dense/MoE on the
/// Vulkan seam or the CPU/Metal reference): renders the FULL OpenAI conversation — including tool
/// specs and prior tool calls/results — through the model's own chat template
/// (`infr_chat::render_chat_oai`, model-independent), generates through the SAME `ChatModel`
/// primitive `infr run`/`bench` drive (persistent session ⇒ per-request suffix-only prefill), and
/// streams through the same [`ChatStream`] splitter (reasoning/content/auto-parsed tool calls).
/// Grammar-FORCED tool_choice builds an llguidance constraint and generates through
/// `generate_constrained` (llama.cpp-parity reliability); auto/none stream through the parser.
struct SeamGenerator {
    model: Box<dyn infr_llama::model::ChatModel + Send>,
    renderer: infr_llama::model::OaiRenderer,
}

impl SeamGenerator {
    fn new(
        gguf_path: &Path,
        model: Box<dyn infr_llama::model::ChatModel + Send>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            model,
            renderer: infr_llama::model::OaiRenderer::open(gguf_path)?,
        })
    }
}

impl infr_server::ChatGenerator for SeamGenerator {
    fn chat(
        &mut self,
        messages: &[infr_engine::ChatMessage],
        tools_json: Option<&str>,
        tool_choice: Option<&str>,
        max_tokens: Option<u32>,
        on_delta: &mut dyn FnMut(infr_engine::Delta),
    ) -> anyhow::Result<()> {
        let tools: Option<serde_json::Value> = tools_json
            .map(serde_json::from_str)
            .transpose()
            .context("parsing request `tools`")?;
        let prompt = self.renderer.render(messages, tools.as_ref())?;
        // The request's `max_tokens` wins; INFR_MAX_NEW (default 2048) is the server-side
        // default for requests that don't set one.
        let max_new = max_tokens.map(|v| v as usize).unwrap_or_else(|| {
            std::env::var("INFR_MAX_NEW")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2048usize)
        });
        // Forced tool_choice ("required"/named): grammar-constrain the call body (the same
        // llguidance machinery as the bespoke path — grammar::constrained_step runs inside the
        // seam decode). Prime the assistant turn with the <tool_call> opener and parse the
        // constrained JSON; on any failure fall back to unconstrained (mirrors LlamaGenerator).
        if let Some(mut constraint) = self.renderer.tool_constraint(tools.as_ref(), tool_choice)? {
            let primed = format!("{prompt}<tool_call>\n");
            let mut body = String::new();
            let emitted = match self.model.generate_constrained(
                &primed,
                max_new,
                &mut constraint,
                &mut |p: &str| body.push_str(p),
            ) {
                Ok(_) => {
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
                    eprintln!("[tools] seam grammar-constrained generation failed ({e})");
                    false
                }
            };
            if emitted {
                return Ok(());
            }
            eprintln!(
                "[tools] forced tool call produced no parseable call; falling back to unconstrained"
            );
        }
        let mut stream = infr_engine::ChatStream::new(tool_choice != Some("none"));
        {
            let od = &mut *on_delta;
            // Template-prefilled thinking (the PROMPT ends with the `<think>` opener): inject a
            // synthetic opener so the splitter emits the head as Reasoning deltas, mirroring
            // `Chat::turn` — run and serve expose thinking identically.
            if infr_engine::prompt_prefills_think(&prompt) {
                stream.push("<think>", &mut *od);
            }
            self.model.generate(&prompt, max_new, &mut |piece: &str| {
                stream.push(piece, &mut *od)
            })?;
        }
        stream.finish(on_delta);
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
    // Benchmarks decode a FIXED token count (llama-bench semantics): never stop at EOS — a model
    // that emits EOS instantly on the dummy context would otherwise report fictional tok/s.
    std::env::set_var("INFR_IGNORE_EOS", "1");
    // -pg "P,G": a coding-agent turn (ingest P then generate G); throughput = (P+G)/time.
    let pg = pg
        .map(|s| -> anyhow::Result<(usize, usize)> {
            let (p, g) = s.split_once(',').context("--pg expects `P,G`")?;
            Ok((p.trim().parse()?, g.trim().parse()?))
        })
        .transpose()?;
    let (gguf, tok) = resolve(model)?;
    // qwen35 (Qwen3-Next): a hybrid gated-DeltaNet + GQA model on its own agnostic-seam graph — it
    // is NOT a `Llama` (dense/MoE), so `load_opt` below can't load or run it. Route it through the
    // seam's own `ChatModel` (`cmd_bench_qwen35`), reusing the same pp/tg warmup-then-time
    // methodology.
    if infr_llama::qwen35::is_qwen35(&gguf) {
        return cmd_bench_qwen35(&gguf, n_prompt, n_gen, depth, pg, reps, ngl == 0, json);
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
    // INFR_METAL=1 (or --dev Metal): bench the dense forward on the Metal backend through the
    // agnostic seam — same pp/tg/pg + depth methodology as the CPU arm, directly comparable to
    // `llama-bench` on the Metal build.
    if std::env::var("INFR_METAL").is_ok() || dev.eq_ignore_ascii_case("metal") {
        return cmd_bench_metal(
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
                 // ubatch>0 pins the seam's prefill chunk (= llama-bench -ub); 0 = the default (1024).
    if ubatch > 0 {
        std::env::set_var("INFR_UBATCH", ubatch.to_string());
    }
    let model = infr_llama::CpuModel::load(&gguf, tok.as_deref())?;
    let samples = model.bench_vulkan(n_prompt, n_gen, depth, pg, reps)?;
    let label = if let Some((p, g)) = pg {
        format!("pg{p}+{g}")
    } else if n_gen > 0 {
        format!("tg{n_gen}")
    } else {
        format!("pp{n_prompt}")
    };
    print_bench_avg(&samples, &label, depth, "", reps, json);
    Ok(())
}

/// Shared bench-result reporter: average the per-rep t/s samples and print either the JSON shape
/// (`[{"avg_ts": ..}]`, llama-bench-comparable) or `label[ @ dN][tag]: X t/s (N reps)`. One
/// implementation for the dense-CPU / qwen35 bench tails (they previously each had a copy).
fn print_bench_avg(samples: &[f64], label: &str, depth: usize, tag: &str, reps: usize, json: bool) {
    let avg = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
    if json {
        println!("[{{\"avg_ts\": {avg:.2}}}]");
    } else {
        let d = if depth > 0 {
            format!(" @ d{depth}")
        } else {
            String::new()
        };
        println!("{label}{d}{tag}: {avg:.1} t/s  ({reps} reps)");
    }
}

/// Metal twin of [`cmd_bench_cpu`]: same pp/tg/pg + depth methodology on the Apple-GPU seam
/// backend (`CpuModel::bench_metal`). On non-macOS this arm is unreachable (the backend crate
/// compiles to nothing), so the whole body is cfg-gated.
#[allow(clippy::too_many_arguments)]
fn cmd_bench_metal(
    gguf: &Path,
    tok: Option<&Path>,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    pg: Option<(usize, usize)>,
    reps: usize,
    json: bool,
) -> anyhow::Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (gguf, tok, n_prompt, n_gen, depth, pg, reps, json);
        anyhow::bail!("metal bench requires macOS");
    }
    #[cfg(target_os = "macos")]
    {
        let model = infr_llama::CpuModel::load(gguf, tok)?;
        let measure_tg = pg.is_none() && n_gen > 0;
        // ONE session for warmup + every rep: backend, uploaded weights, compiled pipelines and
        // the dequant/repack weight caches persist (each rep still measures a full prefill —
        // `bench_metal` resets the materialized tokens). A fresh backend per rep re-paid those
        // one-time costs inside the measurement.
        let ctx = pg.map(|(p, g)| p + g).unwrap_or(if measure_tg {
            depth.max(1) + n_gen
        } else {
            n_prompt
        }) + 2;
        let mut sess = model.metal_session(ctx)?;
        // One untimed warmup: page-cache the mmap + build the weight caches + compile pipelines.
        let _ = model.bench_metal(
            &mut sess,
            depth.max(1),
            if measure_tg || pg.is_some() { 1 } else { 0 },
        );
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let ts = if let Some((p, g)) = pg {
                let s = model.bench_metal(&mut sess, p, g)?;
                (p + g) as f64 / (s.prompt_secs + s.decode_secs)
            } else if measure_tg {
                let s = model.bench_metal(&mut sess, depth.max(1), n_gen)?;
                n_gen as f64 / s.decode_secs
            } else {
                let s = model.bench_metal(&mut sess, n_prompt, 0)?;
                n_prompt as f64 / s.prompt_secs
            };
            samples.push(ts);
        }
        let label = if let Some((p, g)) = pg {
            format!("pg{p}+{g}")
        } else if measure_tg {
            format!("tg{n_gen}")
        } else {
            format!("pp{n_prompt}")
        };
        print_bench_avg(&samples, &label, depth, " [metal]", reps, json);
        Ok(())
    }
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
    let label = if let Some((p, g)) = pg {
        format!("pg{p}+{g}")
    } else if measure_tg {
        format!("tg{n_gen}")
    } else {
        format!("pp{n_prompt}")
    };
    print_bench_avg(&samples, &label, depth, " [cpu]", reps, json);
    Ok(())
}

/// qwen35 (Qwen3-Next) bench: drives the PRODUCTION path through the SAME `ChatModel` structs
/// `infr run` builds (`Qwen35Chat`, on the Vulkan or CPU seam backend), timing `ChatModel::generate`
/// itself — bench and run share one engine BY CONSTRUCTION, so a production-path change can never
/// leave the bench measuring a dead path. The seam ingests a text prompt, so synthesize one near
/// `n_prompt` tokens and report the actual token counts from its stats. `cpu` selects the CPU seam
/// backend (`-ngl 0`) vs the Vulkan GPU-resident forward.
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
    if pg.is_some() {
        anyhow::bail!(
            "qwen35 bench does not support -pg (combined prompt+gen); use separate -p/-n"
        );
    }
    use infr_llama::model::ChatModel;
    std::env::set_var("INFR_Q35_IGNORE_EOS", "1"); // fixed tg count, no early stop
    let mut m: Box<dyn ChatModel> = if cpu {
        Box::new(infr_llama::model::Qwen35Chat::new_cpu(gguf.to_path_buf()))
    } else {
        Box::new(infr_llama::model::Qwen35Chat::new(gguf.to_path_buf()))
    };
    let sentence = "The quick brown fox jumps over the lazy dog. "; // ~10 tokens
                                                                    // Depth rides the PROMPT: `GenStats` splits prompt_secs from decode_secs, so decoding at
                                                                    // depth d = a ~d-token prompt whose (untimed-for-tg) prefill fills the state, then the
                                                                    // timed decode runs at that depth. pp is only reported at depth 0 (a deeper prompt can't
                                                                    // split its prefill time into "depth warm" vs "measured n_prompt").
    let prompt_toks = n_prompt + depth;
    let prompt = if prompt_toks > 0 {
        sentence.repeat(prompt_toks.div_ceil(10))
    } else {
        sentence.to_string()
    };
    let n = n_gen.max(1);
    // untimed warmup: loads the model once (weights + pipeline compile stay warm across reps)
    m.generate(sentence, 1, &mut |_| {})?;
    let (mut pps, mut tgs) = (Vec::new(), Vec::new());
    let (mut np, mut ng) = (0usize, 0usize);
    for _ in 0..reps.max(1) {
        let st = m.generate(&prompt, n, &mut |_| {})?;
        pps.push(st.n_prompt as f64 / st.prompt_secs.max(1e-9));
        tgs.push(st.n_gen as f64 / st.decode_secs.max(1e-9));
        (np, ng) = (st.n_prompt, st.n_gen);
    }
    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    if json {
        let a = if n_gen > 0 { avg(&tgs) } else { avg(&pps) };
        println!("[{{\"avg_ts\": {a:.2}}}]");
    } else if n_gen > 0 && n_prompt > 0 && depth == 0 {
        println!(
            "pp{np}: {:.1} t/s | tg{ng}: {:.1} t/s  ({reps} reps)",
            avg(&pps),
            avg(&tgs)
        );
    } else if n_gen > 0 {
        println!("tg{ng}: {:.1} t/s  ({reps} reps)", avg(&tgs));
    } else {
        println!("pp{np}: {:.1} t/s  ({reps} reps)", avg(&pps));
    }
    Ok(())
}

/// The recurring "where are we behind llama.cpp" survey: for EVERY model given, measure
/// pp512 (prefill), tg128 (decode) and tg64 at `--sweep-depth` on both tools, print the matrix
/// as it fills (a sweep is long — partial results beat silence), and finish with the ratios
/// ranked worst-first so the next perf target is the top of the list.
#[allow(clippy::too_many_arguments)]
fn cmd_compare_sweep(
    models: &[String],
    depth: usize,
    dev: &str,
    ngl: usize,
    threads: usize,
    reps: usize,
    ubatch: usize,
    llama_bench: &str,
) -> anyhow::Result<()> {
    const METRICS: [&str; 3] = ["pp512", "tg128", "tg64@d"];
    println!(
        "\n{:<22} {:<10} | {:>9} | {:>9} | {:>10}",
        "model", "metric", "infr", "llama.cpp", "infr/llama"
    );
    println!("{:-<23}{:-<11}+{:-<11}+{:-<11}+{:-<12}", "", "", "", "", "");
    // (model, metric, infr, llama, ratio) for the ranked summary.
    let mut rows: Vec<(String, String, f64, f64)> = Vec::new();
    for model in models {
        let short = model.rsplit('/').next().unwrap_or(model);
        let mb = match ModelBench::new(model, dev, ngl, threads, reps, ubatch, llama_bench) {
            Ok(mb) => mb,
            Err(e) => {
                println!("{short:<22} {:<10} | resolve failed: {e}", "-");
                continue;
            }
        };
        let ds = depth.to_string();
        let runs: [(&str, Vec<&str>, (usize, usize)); 3] = [
            ("pp512", vec!["-p", "512", "-n", "0"], (512, 0)),
            ("tg128", vec!["-p", "0", "-n", "128"], (0, 128)),
            (METRICS[2], vec!["-p", "0", "-n", "64", "-d", &ds], (0, 64)),
        ];
        for (metric, args, (np, ng)) in runs {
            let metric_label = if metric == "tg64@d" {
                format!("tg64@d{depth}")
            } else {
                metric.to_string()
            };
            // A cold/low-power GPU can flub Vulkan device init on the first subprocess launch
            // after a while-idle; one retry absorbs that without forcing a manual re-run of the
            // whole sweep for one bad cell.
            let mut iv = mb.infr(&args);
            if let Err(e) = &iv {
                eprintln!("infr bench failed ({short} {metric_label}), retrying once: {e:#}");
                std::thread::sleep(std::time::Duration::from_secs(3));
                iv = mb.infr(&args);
            }
            let mut lv = if metric == "tg64@d" {
                mb.llama(np, ng, &["-p", "0", "-n", "64", "-d", &ds])
            } else {
                mb.llama(np, ng, &args)
            };
            if lv.is_none() {
                eprintln!("llama-bench failed ({short} {metric_label}), retrying once");
                std::thread::sleep(std::time::Duration::from_secs(3));
                lv = if metric == "tg64@d" {
                    mb.llama(np, ng, &["-p", "0", "-n", "64", "-d", &ds])
                } else {
                    mb.llama(np, ng, &args)
                };
            }
            let is = iv.as_ref().map(|v| format!("{v:.0}")).unwrap_or_else(|e| {
                eprintln!("infr bench failed ({short} {metric_label}): {e:#}");
                "ERR".into()
            });
            let ls = lv.map(|v| format!("{v:.0}")).unwrap_or_else(|| "NA".into());
            let ratio = match (iv.as_ref().ok(), lv) {
                (Some(&i), Some(l)) if l > 0.0 => {
                    rows.push((short.to_string(), metric_label.clone(), i, l));
                    format!("{:.2}x", i / l)
                }
                _ => "-".into(),
            };
            println!("{short:<22} {metric_label:<10} | {is:>9} | {ls:>9} | {ratio:>10}");
        }
    }
    // Worst-first: the top of this list is the next perf target.
    rows.sort_by(|a, b| (a.2 / a.3).partial_cmp(&(b.2 / b.3)).unwrap());
    println!("\nBIGGEST GAPS (infr/llama, worst first):");
    for (m, metric, i, l) in rows.iter().take(10) {
        println!("  {:>5.2}x  {m:<22} {metric:<10} ({i:.0} vs {l:.0})", i / l);
    }
    Ok(())
}

/// One model's infr-vs-llama.cpp bench harness: resolves the shared model ref once and shells
/// out to `infr bench --json` / `llama-bench -o json` with MATCHED flags. Both the deep
/// coding-agent scenarios (`cmd_compare`) and the multi-model survey (`cmd_compare_sweep`) run
/// through this, so the two tools are always measured identically.
struct ModelBench {
    exe: PathBuf,
    model: String,
    llama_model_args: Vec<String>,
    llama_bench: String,
    dev: String,
    ngl: usize,
    threads: usize,
    reps: usize,
    ubatch: usize,
}

impl ModelBench {
    fn new(
        model: &str,
        dev: &str,
        ngl: usize,
        threads: usize,
        reps: usize,
        ubatch: usize,
        llama_bench: &str,
    ) -> anyhow::Result<Self> {
        let exe = std::env::current_exe().context("locating the infr binary")?;
        // infr and llama.cpp share the HF Hub cache and the same `org/repo:quant` ref grammar, so
        // hand BOTH tools the same reference: `infr bench` takes `model` verbatim, and llama-bench
        // gets the matching `-hf`/`--hf-file` (or `-m` for a local path). Pull once up front so
        // `--offline` holds.
        let resolved = resolve(model)?.0;
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
        Ok(Self {
            exe,
            model: model.to_string(),
            llama_model_args,
            llama_bench: llama_bench.to_string(),
            dev: dev.to_string(),
            ngl,
            threads,
            reps,
            ubatch,
        })
    }

    /// Run `infr bench` (this binary) and read its single-row [{"avg_ts":X}].
    fn infr(&self, args: &[&str]) -> anyhow::Result<f64> {
        use std::process::Command;
        let mut c = Command::new(&self.exe);
        c.arg("bench")
            .arg(&self.model)
            .args(["-r", &self.reps.to_string()]);
        c.args(["--ngl", &self.ngl.to_string(), "--dev", &self.dev]);
        if self.threads > 0 {
            c.args(["-t", &self.threads.to_string()]);
        }
        if self.ubatch > 0 {
            c.args(["-u", &self.ubatch.to_string()]);
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
    }

    /// Run `llama-bench -o json` and pick the row matching (n_prompt, n_gen): -pg adds extra rows.
    fn llama(&self, np: usize, ng: usize, args: &[&str]) -> Option<f64> {
        use std::process::Command;
        let mut c = Command::new(&self.llama_bench);
        c.args(&self.llama_model_args);
        c.args([
            "-ngl",
            &self.ngl.to_string(),
            "-dev",
            &self.dev,
            "-fa",
            "auto",
            "-r",
            &self.reps.to_string(),
            "-o",
            "json",
        ]);
        if self.threads > 0 {
            c.args(["-t", &self.threads.to_string()]);
        }
        if self.ubatch > 0 {
            c.args(["-ub", &self.ubatch.to_string()]);
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
    }
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
    let mb = ModelBench::new(model, dev, ngl, threads, reps, ubatch, llama_bench)?;
    let infr_b = |args: &[&str]| mb.infr(args);
    let llama_b = |np: usize, ng: usize, args: &[&str]| mb.llama(np, ng, args);

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

/// Seam paths read sampling from INFR_TEMP / INFR_TOP_K / INFR_TOP_P (library default = greedy so
/// tests stay deterministic). For chat UX, run/serve set the qwen3-recommended defaults when the
/// user hasn't — pure greedy makes thinking models degenerate. Mirrors the bespoke
/// `Llama::set_sampling(0.6, 20, 0.95)` defaults.
fn set_default_sampling_env() {
    if std::env::var("INFR_TEMP").is_err() {
        std::env::set_var("INFR_TEMP", "0.6");
    }
    if std::env::var("INFR_TOP_K").is_err() {
        std::env::set_var("INFR_TOP_K", "20");
    }
    if std::env::var("INFR_TOP_P").is_err() {
        std::env::set_var("INFR_TOP_P", "0.95");
    }
}

fn cmd_serve(model: &str, addr: &str) -> anyhow::Result<()> {
    let (gguf, tok) = resolve(model)?;
    let model_id = gguf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let sockaddr: std::net::SocketAddr = addr.parse().context("invalid --addr")?;

    // Seam-backed serve — the ONE engine: the SAME ChatModel + multi-slot session `infr run`
    // uses, so serve gets per-request suffix-only prefill and cross-conversation prefix seeding
    // for free. INFR_CPU / INFR_METAL select the reference backends; Vulkan is the default.
    let is_q35 = infr_llama::qwen35::is_qwen35(&gguf);
    let mut m: Box<dyn infr_llama::model::ChatModel + Send> = if is_q35 {
        Box::new(if std::env::var("INFR_METAL").is_ok() {
            infr_llama::model::Qwen35Chat::new_metal(gguf.clone())
        } else if std::env::var("INFR_CPU").is_ok() {
            infr_llama::model::Qwen35Chat::new_cpu(gguf.clone())
        } else {
            infr_llama::model::Qwen35Chat::new(gguf.clone())
        })
    } else if std::env::var("INFR_METAL").is_ok() {
        // Metal: the persistent-session seam chat on macOS (the Vulkan session's Apple twin);
        // the stateless reference wrapper elsewhere (the arm is unreachable off-macOS anyway).
        #[cfg(target_os = "macos")]
        {
            Box::new(infr_llama::model::MetalSeamChat::new(
                infr_llama::CpuModel::load(&gguf, tok.as_deref())?,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Box::new(infr_llama::model::CpuDenseChat::new_metal(
                infr_llama::CpuModel::load(&gguf, tok.as_deref())?,
            ))
        }
    } else if std::env::var("INFR_CPU").is_ok() {
        Box::new(infr_llama::model::CpuDenseChat::new(
            infr_llama::CpuModel::load(&gguf, tok.as_deref())?,
        ))
    } else {
        Box::new(infr_llama::model::DenseSeamChat::new(
            infr_llama::CpuModel::load(&gguf, tok.as_deref())?,
        ))
    };
    {
        set_default_sampling_env();
        // Compile every lazily-built pipeline NOW (a tiny throwaway generation) so the first
        // request doesn't pay seconds of pipeline builds on top of its own prefill.
        let t0 = std::time::Instant::now();
        m.warmup()?;
        eprintln!(
            "warmup: pipelines compiled in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        let generator: Box<dyn infr_server::ChatGenerator> =
            Box::new(SeamGenerator::new(&gguf, m)?);
        let rt = tokio::runtime::Runtime::new()?;
        println!("infr serve: {model_id} on http://{sockaddr}  (OpenAI /v1, agnostic seam)");
        rt.block_on(infr_server::serve(generator, model_id, sockaddr))
    }
}
