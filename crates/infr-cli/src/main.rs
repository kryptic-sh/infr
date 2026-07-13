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
        /// Concurrent generation slots (llama-server's `-np`). N requests generate at once, each
        /// with its own KV cache, taking turns on the GPU at token granularity; the (N+1)'th queues.
        ///
        /// Each slot owns a full KV cache, so the DEFAULT per-slot context is the VRAM-fit window
        /// divided by N: the N slots together stay inside the same VRAM budget one slot is held to,
        /// and raising `-np` can never OOM a box that `-np 1` fit. The visible cost is a smaller
        /// per-request window. Pass `--ctx` to pin the per-slot window instead (then `N * ctx` must
        /// fit, and the Vulkan budget guard will say so if it doesn't).
        /// `--np` is accepted as an alias (llama-server spells this `-np`; clap shorts are a single
        /// character, so `-n` is the short form and `--np` the familiar long one).
        #[arg(
            long = "parallel",
            visible_alias = "np",
            short = 'n',
            default_value_t = 4,
            value_name = "N"
        )]
        parallel: usize,
        /// Per-slot context window in tokens (`8192`, `256k`, or `50%` of the free-VRAM KV
        /// capacity). Default: the model's trained context, clamped to VRAM and divided by
        /// `--parallel`. Overrides INFR_CTX.
        #[arg(long, value_name = "TOKENS")]
        ctx: Option<String>,
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
        Cmd::Serve {
            model,
            addr,
            parallel,
            ctx,
        } => cmd_serve(&model, &addr, parallel, ctx.as_deref()),
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
                    // Close the dim reasoning with a blank line: `<think>` models emit their own
                    // whitespace around `</think>`, but channel-format models (diffusion-gemma,
                    // E2B) end reasoning at a bare `<channel|>` — without this the answer renders
                    // GLUED to the last thinking line ("…clearly.The capital of France is Paris.").
                    print!("[0m\n\n");
                    *in_think = false;
                    // The splitter strips markers, not whitespace: a think-model's own newline
                    // after `</think>` shouldn't stack a third blank line on top of ours.
                    print!("{}", t.trim_start_matches('\n'));
                    return;
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
    // Context window: the model's trained context by default; INFR_CTX overrides (shared size
    // grammar — tokens, `256k`, or `%` of the free-VRAM KV capacity), read by the chat sessions.
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
    let (gguf, tok) = resolve(model)?;
    // diffusion-gemma (block text-diffusion, Phase 3 — docs/DIFFUSIONGEMMA.md): a cheap arch peek
    // (no full SeamModel load) so the default token budget below and the ChatModel selection further
    // down can both branch on it. -n/max_new drives `blocks = ceil(n_predict / canvas_length)`
    // (256-token canvas) rather than autoregressive tokens, so the AR default of 2048 would run 8
    // whole blocks for a "Hi" reply; 1024 (4 blocks) is the same order of magnitude as a normal
    // chat reply and still overridable via INFR_MAX_NEW.
    let is_dg = infr_llama::diffusion::is_diffusion_gemma(&gguf);
    // Generation ceiling per reply (a turn also caps to remaining context). High enough for long
    // answers (lists/stories); override with INFR_MAX_NEW.
    let max_new = envu("INFR_MAX_NEW", if is_dg { 1024 } else { 2048 });

    // Build the per-backend generation primitive (`ChatModel`), then wrap it in the ONE shared `Chat`
    // (infr_llama::model) that owns history + `<think>`-stripping and drives the single REPL below:
    // INFR_CPU (dense/MoE/qwen35 on the agnostic compute graph, no Vulkan/VRAM), Vulkan/Metal GPU,
    // qwen3moe, dense Qwen3/Llama/Gemma. Every backend now does history-based multi-turn — no
    // per-arch one-shot special-case. The CLI owns the Llama; the boxed trait object borrows it (so
    // the borrow-based dense `ChatSession` needs no ownership change).
    // qwen35 (Qwen3.5) runs through the SAME standard `ChatModel` structs as every other arch below
    // — `SeamModel::load` + the CPU/Vulkan/Metal sessions drive any `Config` arch (including
    // `MixerW::DeltaNet`) — so there is no qwen35-only branch (the old bespoke seam and its
    // env-gated escape hatch were deleted once the unified path was validated; issue #30).
    // Chat-default sampling for every backend (the bespoke branch reads the same envs below).
    set_default_sampling_env();
    let model: Box<dyn infr_llama::chat::ChatModel + '_> = if is_dg {
        // diffusion-gemma (Phase 3/D): the entropy-bound block-diffusion loop
        // (`infr_llama::diffusion`) over a persistent session — Vulkan by default, CPU under
        // INFR_CPU, Metal under INFR_METAL (Phase D added the Metal DG session; macOS only — the
        // non-macOS build still compiles this arm, `DiffusionGemmaChat::generate` errors clearly
        // at runtime there instead, matching every other INFR_METAL arm's convention).
        let cpu = std::env::var("INFR_CPU").is_ok();
        let metal = std::env::var("INFR_METAL").is_ok();
        eprintln!(
            "[{} — diffusion-gemma entropy-bound block decode]",
            if cpu {
                "cpu backend"
            } else if metal {
                "metal backend"
            } else {
                "vulkan seam"
            }
        );
        let loaded = infr_llama::SeamModel::load(&gguf, tok.as_deref())?;
        Box::new(if cpu {
            infr_llama::chat::DiffusionGemmaChat::new_cpu(loaded)
        } else if metal {
            infr_llama::chat::DiffusionGemmaChat::new_metal(loaded)
        } else {
            infr_llama::chat::DiffusionGemmaChat::new(loaded)
        })
    } else if std::env::var("INFR_METAL").is_ok() {
        eprintln!(
            "[metal backend — dense/MoE forward on Apple GPU via the agnostic compute graph, persistent KV session]"
        );
        #[cfg(target_os = "macos")]
        {
            metal_chat_model(&gguf, tok.as_deref())?
        }
        #[cfg(not(target_os = "macos"))]
        {
            Box::new(infr_llama::chat::CpuDenseChat::new_metal(
                infr_llama::SeamModel::load(&gguf, tok.as_deref())?,
            ))
        }
    } else if std::env::var("INFR_CPU").is_ok() {
        eprintln!(
            "[cpu backend — dense/MoE forward on CPU via the agnostic compute graph, no GPU]"
        );
        Box::new(infr_llama::chat::CpuDenseChat::new(
            infr_llama::SeamModel::load(&gguf, tok.as_deref())?,
        ))
    } else {
        // The default: dense/MoE on the VULKAN agnostic seam — persistent multi-slot KV sessions
        // (per-turn suffix-only prefill), record-once decode replay, MoE expert auto-fit (fully
        // resident when experts fit; the paged expert cache — INFR_CACHE — otherwise). qwen35 (Qwen3.5) lands here too — same seam, same `Config::from_gguf` +
        // `MixerW::DeltaNet` unified runner (see `unified_qwen35_*` tests). llama4 (Scout) lands
        // here too now: the paged expert cache lets its 37 GB Q2_K bank run on a 24 GB card.
        eprintln!("[vulkan seam — dense/MoE on the agnostic compute graph, persistent KV session]");
        Box::new(infr_llama::chat::DenseSeamChat::new(
            infr_llama::SeamModel::load(&gguf, tok.as_deref())?,
        ))
    };
    let mut model = model;
    // Compile the lazily-built pipelines NOW (like `serve` does before its first request) so the
    // first turn's reported prefill rate measures prefill, not one-time pipeline builds — a cold
    // diffusion-gemma prefill measured 26 t/s vs 1424 t/s warm, all compile.
    model.warmup()?;
    let mut chat = infr_llama::chat::Chat::new(model);
    // Live denoise canvas view (diffusion-gemma only — see `DiffusionVisual`'s doc); `None` when
    // unset/not-DG/not-a-tty leaves `run_chat_turn` on the exact pre-existing `chat.turn` path.
    let mut visual = if is_dg {
        DiffusionVisual::new(&gguf)?
    } else {
        None
    };

    // One-shot (a message) or an interactive multi-turn REPL (every backend now supports it).
    if let Some(m) = message {
        run_chat_turn(&mut chat, m, max_new, visual.as_mut())?;
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
        if let Err(e) = run_chat_turn(&mut chat, line, max_new, visual.as_mut()) {
            eprintln!("error: {e}");
        }
    }
    Ok(())
}

/// Run one chat turn through the shared [`Chat`]: stream pieces via the `<think>` renderer, then
/// print the prefill/decode stats line. `visual: Some` (diffusion-gemma, `INFR_DIFFUSION_VISUAL`
/// set, stdout a tty) drives the turn through `turn_with_step_hook` instead, so the live canvas
/// view redraws in a reserved terminal region while a block denoises; `None` is the exact
/// pre-existing `chat.turn` call, byte-for-byte.
///
/// `on_piece` now fires once PER FINISHED BLOCK (`DiffusionGemmaChat::generate_impl` streams each
/// block's committed text through the shared detok as soon as `diffusion_generate`'s `on_block`
/// hands it over — see that fn's doc), not once for the whole reply at turn end. So each call
/// erases whatever live region is currently reserved (`DiffusionVisual::end`, a no-op if none is —
/// e.g. before the very first block), which lets the just-finished block's text print normally
/// and PERMANENTLY via `render.feed` right where the region's cursor was left, scrolling up
/// naturally like any other transcript text. `DiffusionVisual::step` then lazily reserves a FRESH
/// region below that printed text the next time a block starts denoising — so the live-updating
/// region only ever shows the CURRENT in-progress block, never a finished one.
fn run_chat_turn(
    chat: &mut infr_llama::chat::Chat,
    message: &str,
    max_new: usize,
    visual: Option<&mut DiffusionVisual>,
) -> anyhow::Result<()> {
    let mut render = ThinkRender::new();
    let stats = match visual {
        Some(v) => {
            // Both closures below only need transient access (never concurrently — generation is
            // single-threaded and `on_step`/`on_piece` never interleave), so a `RefCell` lets them
            // share `v` without two simultaneous `&mut` captures.
            let visual = std::cell::RefCell::new(v);
            // Buffer the streamed pieces instead of feeding ThinkRender per block: the
            // template-aware rendering (think/channel-marker handling) only formats correctly
            // over the COMPLETE response — per-block feeding split markers across pieces and
            // broke the formatting (user report). The live region is the in-progress display;
            // the one-shot render below is the permanent transcript.
            let buffered = std::cell::RefCell::new(String::new());
            let mut on_piece = |p: &str| {
                buffered.borrow_mut().push_str(p);
            };
            let mut on_step =
                |view: infr_llama::diffusion::StepView| visual.borrow_mut().step(view);
            let result =
                chat.turn_with_step_hook(message, max_new, &mut on_piece, Some(&mut on_step));
            // Erase the live region first, then render the whole response through the
            // template-aware renderer exactly once — correct formatting, no duplication.
            visual.borrow_mut().end();
            render.feed(&buffered.borrow());
            result?
        }
        None => chat.turn(message, max_new, &mut |p| render.feed(p))?,
    };
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

/// Fixed-size scratch region (rows) the live view redraws in place — capped rather than sized to
/// the real terminal so there's no ioctl/terminal-size dependency: worst case on a narrower
/// terminal a long line wraps, which only affects this scratch region's own redraw math, never the
/// surrounding transcript.
const DG_VISUAL_ROWS: usize = 16;
/// Default/fallback column cap per rendered line (item-truncated — see [`DiffusionVisual::cols`]),
/// same conservative-width rationale as [`DG_VISUAL_ROWS`]. A parseable `$COLUMNS` narrower than
/// this clamps `cols` down further (audit finding 2: without that clamp, a narrower real terminal
/// WRAPS a `DG_VISUAL_COLS`-wide line, so the region occupies more than `DG_VISUAL_ROWS` actual
/// terminal rows and the fixed cursor-up math in `step`/`end` desyncs — frames smear into each
/// other instead of overwriting cleanly).
const DG_VISUAL_COLS: usize = 120;

use std::io::{IsTerminal, Write as _};

/// `INFR_DIFFUSION_VISUAL` live denoise canvas view for `infr run` (diffusion-gemma only — see
/// `docs/DIFFUSIONGEMMA.md`, ports the UX idea of the oracle's `--diffusion-visual` without
/// depending on it): per step, decode the block's CURRENT canvas fresh with a throwaway tokenizer
/// ([`OaiRenderer::decode_ids`] — cheap, ≤ canvas_len tokens, no GPU work), render accepted
/// (committed) runs as normal text and not-yet-accepted (still renoising — this sampler has no
/// literal mask token, see `crate::diffusion`'s module doc) runs as a dim `·` placeholder, and
/// redraw a fixed-height region in place (cursor-up + erase, DEC synchronized-update framing) so
/// the terminal never scrolls mid-block.
///
/// The region is reserved LAZILY (the first `step` call after construction, or after the previous
/// region was erased by `end`) and erased every time a block finishes (`run_chat_turn`'s
/// `on_piece`, once per block now — see `DiffusionGemmaChat::generate_impl`'s doc) so that block's
/// permanent transcript text prints cleanly, scrolling up naturally; the NEXT block's first `step`
/// then reserves a fresh region below it. So at any moment at most one region is on screen, and it
/// only ever shows the block currently denoising.
struct DiffusionVisual {
    oai: infr_llama::chat::OaiRenderer,
    /// Rows the previous frame's redraw advanced past the region's top — 0 before the first frame
    /// (mirrors the oracle's `cb_data.vis_prev_rows`).
    prev_rows: usize,
    /// Whether a region is currently reserved on screen. `step` reserves one lazily when this is
    /// `false`; `end` erases the region and clears this back to `false` (a no-op if already
    /// clear — `run_chat_turn` calls `end` unconditionally on every `on_piece` and once more as a
    /// post-turn safety net).
    active: bool,
    /// Column cap for one rendered line, in DISPLAY ITEMS (audit finding 1: never raw chars/bytes
    /// of a pre-rendered ANSI string) — `min(DG_VISUAL_COLS, $COLUMNS - 1)` when `COLUMNS` parses
    /// to something usable, else `DG_VISUAL_COLS`. Best-effort: infr has no ioctl-based terminal
    /// size probe, and `$COLUMNS` is a shell-maintained convention, not guaranteed exported to a
    /// child process — this only ever narrows the conservative default, never widens past it.
    cols: usize,
}

impl DiffusionVisual {
    /// `None` unless `INFR_DIFFUSION_VISUAL=1` (stdout must be a tty) or `=force` (bypasses the
    /// tty check — for scripted verification against piped stdout, e.g. `... | tail -20`).
    fn new(gguf: &Path) -> anyhow::Result<Option<Self>> {
        let mode = std::env::var("INFR_DIFFUSION_VISUAL").unwrap_or_default();
        let force = mode == "force";
        if !force && mode != "1" {
            return Ok(None);
        }
        if !force && !std::io::stdout().is_terminal() {
            return Ok(None);
        }
        let cols = std::env::var("COLUMNS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .and_then(|c| c.checked_sub(1))
            .filter(|&c| c > 0)
            .map(|c| DG_VISUAL_COLS.min(c))
            .unwrap_or(DG_VISUAL_COLS);
        Ok(Some(Self {
            oai: infr_llama::chat::OaiRenderer::open(gguf)?,
            prev_rows: 0,
            active: false,
            cols,
        }))
    }

    /// Reserve the region: hide the cursor, print `DG_VISUAL_ROWS` blank lines (scrolls once if
    /// already at the bottom, exactly like a normal print would — scrollback stays intact), then
    /// park the cursor back at the region's top.
    fn begin(&mut self) {
        self.prev_rows = 0;
        print!("\x1b[?25l{}", "\n".repeat(DG_VISUAL_ROWS));
        print!("\x1b[{DG_VISUAL_ROWS}A");
        std::io::stdout().flush().ok();
        self.active = true;
    }

    /// Redraw the region for one denoise step — lazily reserving it first (`begin`) if the last
    /// finished block's `on_piece` erased it (or this is the turn's very first step).
    fn step(&mut self, view: infr_llama::diffusion::StepView) {
        if !self.active {
            self.begin();
        }
        // One flat (display_char, is_dim) sequence: accepted runs decode to real text, sanitized
        // per audit finding 3 (`\r` dropped — never jumps the cursor mid-line; `\t` -> a single
        // space, never a literal tab); not-yet-accepted runs render as a dim `·` placeholder.
        // Kept as tagged chars, NOT pre-rendered ANSI text, so the truncation/line-splitting below
        // always operates on DISPLAY items and never raw escape bytes (audit finding 1).
        let mut items: Vec<(char, bool)> = Vec::new();
        let mut i = 0;
        while i < view.canvas.len() {
            if view.accepted[i] {
                let start = i;
                while i < view.canvas.len() && view.accepted[i] {
                    i += 1;
                }
                if let Ok(s) = self.oai.decode_ids(&view.canvas[start..i]) {
                    for c in s.chars() {
                        match c {
                            '\r' => {}                        // dropped: never jump the cursor
                            '\t' => items.push((' ', false)), // one space, never a literal tab
                            _ => items.push((c, false)),
                        }
                    }
                }
            } else {
                let start = i;
                while i < view.canvas.len() && !view.accepted[i] {
                    i += 1;
                }
                for _ in start..i {
                    items.push(('·', true));
                }
            }
        }

        // Split into display lines on '\n' only (audit finding 3's other half — a literal '\n' in
        // decoded text is the sole line-break signal), each capped at `self.cols` ITEMS (audit
        // findings 1/2), then rendered with dim runs balanced one `\x1b[2m`/`\x1b[0m` pair at a
        // time (`render_dim_line`) — never a bare escape fragment, never a leaked dim mode.
        let mut lines: Vec<Vec<(char, bool)>> = vec![Vec::new()];
        for &(c, dim) in &items {
            if c == '\n' {
                lines.push(Vec::new());
            } else {
                lines.last_mut().unwrap().push((c, dim));
            }
        }
        // No header line (user feedback: the "[diffusion] block N step S/M" rows read as
        // noise) — the resolving canvas itself is the progress indicator.
        let _ = (view.block, view.step, view.max_steps, view.committed_before);
        let mut rows: Vec<String> = Vec::new();
        for line in &lines {
            let cut = line.len().min(self.cols);
            rows.push(render_dim_line(&line[..cut]));
        }
        let keep = rows.len().min(DG_VISUAL_ROWS);
        let shown = &rows[rows.len() - keep..];

        // ?7l: auto-wrap OFF for the frame — a too-long line truncates at the margin instead
        // of wrapping to a second physical row, which would desync the fixed-row cursor math
        // (COLUMNS is a shell variable most shells don't export, so the item clamp can't know
        // the real width). Restored (?7h) at frame end.
        let mut frame = String::from("\x1b[?2026h\x1b[?7l"); // begin synchronized update
        if self.prev_rows > 0 {
            frame.push_str(&format!("\x1b[{}A", self.prev_rows));
        }
        frame.push('\r');
        for r in 0..DG_VISUAL_ROWS {
            if let Some(line) = shown.get(r) {
                frame.push_str(line);
            }
            frame.push_str("\x1b[K"); // erase to end of line — clears a shorter previous frame
            if r + 1 < DG_VISUAL_ROWS {
                frame.push('\n');
            }
        }
        frame.push_str("\x1b[?7h\x1b[?2026l"); // auto-wrap back on, end synchronized update
        self.prev_rows = DG_VISUAL_ROWS - 1;
        print!("{frame}");
        std::io::stdout().flush().ok();
    }

    /// Erase the region and restore the cursor, if one is currently reserved (a no-op otherwise —
    /// safe to call unconditionally, e.g. after every finished block AND once more at turn end).
    fn end(&mut self) {
        if !self.active {
            return;
        }
        let mut frame = String::new();
        if self.prev_rows > 0 {
            frame.push_str(&format!("\x1b[{}A", self.prev_rows));
        }
        frame.push_str("\r\x1b[J\x1b[?25h"); // erase from cursor to end of screen, show cursor
        print!("{frame}");
        std::io::stdout().flush().ok();
        self.prev_rows = 0;
        self.active = false;
    }
}

/// Render one line's `(display_char, is_dim)` items, wrapping each maximal DIM RUN in exactly one
/// `\x1b[2m`/`\x1b[0m` pair — the fix for audit finding 1: the old code truncated a pre-rendered
/// `"\x1b[2m·\x1b[0m"` string by raw `char` count, which could cut an escape sequence in half
/// (printing a literal `[2m` fragment) and/or drop the trailing reset, leaking dim mode into
/// whatever printed next. Operating on tagged items instead of ANSI bytes makes that class of bug
/// unrepresentable: every emitted `\x1b[2m` here is always paired with an `\x1b[0m` on the same
/// line, and truncation (by the caller, before this runs) only ever drops whole items.
fn render_dim_line(items: &[(char, bool)]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < items.len() {
        if items[i].1 {
            out.push_str("\x1b[2m");
            while i < items.len() && items[i].1 {
                out.push(items[i].0);
                i += 1;
            }
            out.push_str("\x1b[0m");
        } else {
            while i < items.len() && !items[i].1 {
                out.push(items[i].0);
                i += 1;
            }
        }
    }
    out
}

/// The Metal dense/MoE [`ChatModel`] for run AND serve: the persistent-session seam chat, or —
/// with `INFR_SPEC_DRAFT=<gguf>` — speculative decoding (a small same-tokenizer draft proposes up
/// to `INFR_SPEC_K` tokens per round, default 6; one batched target forward verifies; greedy-only,
/// pays off for ~8B-class targets — issue #16). One selection funnel so run and serve can never
/// disagree on how the Metal model is built.
#[cfg(target_os = "macos")]
fn metal_chat_model(
    gguf: &Path,
    tok: Option<&Path>,
) -> anyhow::Result<Box<dyn infr_llama::chat::ChatModel + Send>> {
    if let Ok(draft_path) = std::env::var("INFR_SPEC_DRAFT") {
        let target = infr_llama::SeamModel::load(gguf, tok)?;
        let draft = infr_llama::SeamModel::load(std::path::Path::new(&draft_path), None)?;
        // Upper bound on the draft length; the driver adapts the actual k per round to recent
        // acceptance (verify cost scales with rows on this hardware, so over-drafting
        // low-acceptance text costs real time).
        let k = std::env::var("INFR_SPEC_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6);
        let (tb, db) = (
            std::fs::metadata(gguf).map(|m| m.len()).unwrap_or(0),
            std::fs::metadata(&draft_path).map(|m| m.len()).unwrap_or(0),
        );
        if db * 4 > tb {
            eprintln!(
                "[spec] warning: draft is more than 1/4 the target's size — \
                 speculation only pays when the target is much larger (see #16)"
            );
        }
        std::env::set_var("INFR_TEMP", "0");
        eprintln!("[metal spec — target + {k}-token draft verify, greedy (INFR_TEMP=0)]");
        Ok(Box::new(infr_llama::chat::SpecMetalChat::new(
            target, draft, k,
        )))
    } else {
        Ok(Box::new(infr_llama::chat::MetalSeamChat::new(
            infr_llama::SeamModel::load(gguf, tok)?,
        )))
    }
}

/// Serve adapter for the seam-backed [`ChatModel`]s (qwen35 on any backend, dense/MoE on the
/// Vulkan seam or the CPU/Metal reference): renders the FULL OpenAI conversation — including tool
/// specs and prior tool calls/results — through the model's own chat template
/// (`infr_chat::render_chat_oai`, model-independent), generates through the SAME `ChatModel`
/// primitive `infr run`/`bench` drive (persistent session ⇒ per-request suffix-only prefill), and
/// streams through the same [`ChatStream`] splitter (reasoning/content/auto-parsed tool calls).
/// Grammar-FORCED tool_choice builds an llguidance constraint and generates through
/// `generate_constrained` (llama.cpp-parity reliability); auto/none stream through the parser.
/// SERIALISED: the backend is a `&mut`-only `ChatModel` with a single KV session, so concurrent
/// requests take turns behind a `Mutex`. Used for CPU / Metal / diffusion-gemma, which have no
/// multi-slot engine — `cmd_serve` pins those to `--parallel 1` so the queueing is explicit rather
/// than a silent serialisation of a server the user asked to parallelise.
struct SeamGenerator {
    model: std::sync::Mutex<Box<dyn infr_llama::chat::ChatModel + Send>>,
    renderer: infr_llama::chat::OaiRenderer,
}

impl SeamGenerator {
    fn new(
        gguf_path: &Path,
        model: Box<dyn infr_llama::chat::ChatModel + Send>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            model: std::sync::Mutex::new(model),
            renderer: infr_llama::chat::OaiRenderer::open(gguf_path)?,
        })
    }
}

/// CONCURRENT: N KV slots on the Vulkan seam, round-robin on the GPU at token granularity — see
/// [`infr_llama::parallel::ParallelSeam`]. The `infr serve --parallel N` default path.
struct ParallelGenerator {
    engine: infr_llama::parallel::ParallelSeam,
    renderer: infr_llama::chat::OaiRenderer,
}

impl ParallelGenerator {
    fn new(gguf_path: &Path, engine: infr_llama::parallel::ParallelSeam) -> anyhow::Result<Self> {
        Ok(Self {
            engine,
            renderer: infr_llama::chat::OaiRenderer::open(gguf_path)?,
        })
    }
}

/// The two things a serve backend must do, so the OpenAI wire logic (rendering, forced tool calls,
/// stop sequences, reasoning/content splitting) lives in ONE place — [`run_chat`] — instead of being
/// duplicated per backend.
trait GenBackend: Send + Sync {
    fn renderer(&self) -> &infr_llama::chat::OaiRenderer;

    /// This sequence's own state: sampling config, abort latch, and (when the engine is
    /// multi-slot) its turn on the GPU baton.
    fn request_ctx(
        &self,
        sampling: infr_llama::sampling::RequestSampling,
    ) -> infr_llama::sampling::RequestCtx;

    fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut infr_llama::grammar::Constraint>,
        req: &infr_llama::sampling::RequestCtx,
        on_piece: &mut dyn FnMut(&str),
    ) -> anyhow::Result<infr_llama::GenStats>;
}

impl GenBackend for SeamGenerator {
    fn renderer(&self) -> &infr_llama::chat::OaiRenderer {
        &self.renderer
    }

    fn request_ctx(
        &self,
        sampling: infr_llama::sampling::RequestSampling,
    ) -> infr_llama::sampling::RequestCtx {
        // No gate: this backend is serialised by the Mutex below, so there is never more than one
        // sequence on the GPU to take turns with.
        infr_llama::sampling::RequestCtx::new(sampling)
    }

    fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut infr_llama::grammar::Constraint>,
        req: &infr_llama::sampling::RequestCtx,
        on_piece: &mut dyn FnMut(&str),
    ) -> anyhow::Result<infr_llama::GenStats> {
        let mut model = self.model.lock().expect("serve generator poisoned");
        match constraint {
            Some(c) => model.generate_constrained(prompt, max_new, c, Some(req), on_piece),
            None => model.generate(prompt, max_new, Some(req), on_piece),
        }
    }
}

impl GenBackend for ParallelGenerator {
    fn renderer(&self) -> &infr_llama::chat::OaiRenderer {
        &self.renderer
    }

    fn request_ctx(
        &self,
        sampling: infr_llama::sampling::RequestSampling,
    ) -> infr_llama::sampling::RequestCtx {
        self.engine.request_ctx(sampling)
    }

    fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut infr_llama::grammar::Constraint>,
        req: &infr_llama::sampling::RequestCtx,
        on_piece: &mut dyn FnMut(&str),
    ) -> anyhow::Result<infr_llama::GenStats> {
        self.engine
            .generate(prompt, max_new, constraint, req, |p| on_piece(p))
    }
}

/// Map the validated request params onto the seam's per-request sampling scope. An ABSENT request
/// field stays `None`/neutral, so the decode loop keeps inheriting the CLI's `INFR_TEMP` /
/// `INFR_TOP_K` / `INFR_TOP_P` defaults for it — only an EXPLICIT field overrides.
fn request_sampling(p: &infr_server::GenParams) -> infr_llama::sampling::RequestSampling {
    infr_llama::sampling::RequestSampling {
        temp: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        seed: p.seed,
        presence_penalty: p.presence_penalty.unwrap_or(0.0),
        frequency_penalty: p.frequency_penalty.unwrap_or(0.0),
        repeat_penalty: p.repeat_penalty.unwrap_or(1.0),
        ..Default::default()
    }
}

impl infr_server::ChatGenerator for SeamGenerator {
    fn chat(
        &self,
        messages: &[infr_engine::ChatMessage],
        tools_json: Option<&str>,
        tool_choice: Option<&str>,
        params: &infr_server::GenParams,
        on_delta: &mut dyn FnMut(infr_engine::Delta),
    ) -> anyhow::Result<infr_server::Finish> {
        run_chat(self, messages, tools_json, tool_choice, params, on_delta)
    }
}

impl infr_server::ChatGenerator for ParallelGenerator {
    fn chat(
        &self,
        messages: &[infr_engine::ChatMessage],
        tools_json: Option<&str>,
        tool_choice: Option<&str>,
        params: &infr_server::GenParams,
        on_delta: &mut dyn FnMut(infr_engine::Delta),
    ) -> anyhow::Result<infr_server::Finish> {
        run_chat(self, messages, tools_json, tool_choice, params, on_delta)
    }
}

/// The OpenAI chat body, shared by every serve backend: render the conversation through the model's
/// own template, honour a FORCED tool_choice with an llguidance constraint, then stream the reply
/// through the reasoning/content/tool-call splitter with stop sequences applied to the raw text.
///
/// Backend-agnostic by construction — all the per-sequence state (sampling, RNG seed, penalties,
/// stop matcher, abort latch) is created HERE, per call, and handed to the backend explicitly. Two
/// concurrent calls therefore share nothing: that is what makes request A's `temperature` unable to
/// leak into request B (the old thread-local could not offer that guarantee once one thread stepped
/// several sequences).
fn run_chat(
    be: &dyn GenBackend,
    messages: &[infr_engine::ChatMessage],
    tools_json: Option<&str>,
    tool_choice: Option<&str>,
    params: &infr_server::GenParams,
    on_delta: &mut dyn FnMut(infr_engine::Delta),
) -> anyhow::Result<infr_server::Finish> {
    {
        let tools: Option<serde_json::Value> = tools_json
            .map(serde_json::from_str)
            .transpose()
            .context("parsing request `tools`")?;
        let prompt = be.renderer().render(messages, tools.as_ref())?;
        // The request's `max_tokens`/`max_completion_tokens` wins; INFR_MAX_NEW (default 2048) is
        // the server-side default for requests that don't set one.
        let max_new = params.max_tokens.map(|v| v as usize).unwrap_or_else(|| {
            std::env::var("INFR_MAX_NEW")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2048usize)
        });
        // THIS sequence's own sampling (temperature/top_p/top_k/seed/penalties) + abort latch + GPU
        // turn. Owned by this call — not installed anywhere ambient — so it cannot be observed by,
        // or leak into, any other in-flight request.
        let req = be.request_ctx(request_sampling(params));
        // Forced tool_choice ("required"/named): grammar-constrain the call body (the same
        // llguidance machinery as the bespoke path — grammar::constrained_step runs inside the
        // seam decode). Prime the assistant turn with the <tool_call> opener and parse the
        // constrained JSON; on any failure fall back to unconstrained (mirrors LlamaGenerator).
        if let Some(mut constraint) = be.renderer().tool_constraint(tools.as_ref(), tool_choice)? {
            let primed = format!("{prompt}<tool_call>\n");
            let mut body = String::new();
            let emitted = match be.generate(
                &primed,
                max_new,
                Some(&mut constraint),
                &req,
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
                return Ok(infr_server::Finish::ToolCalls);
            }
            eprintln!(
                "[tools] forced tool call produced no parseable call; falling back to unconstrained"
            );
        }
        let mut stream = infr_engine::ChatStream::new(tool_choice != Some("none"));
        // Stop sequences run on the RAW decoded text, BEFORE the reasoning/tool splitter — so a
        // stop string that spans two tokens still fires, and a partial stop prefix is held back
        // instead of being streamed out (see `StopMatcher`). A hit latches an abort on THIS
        // sequence's own `RequestCtx`, which its decode loop polls after the current token. The
        // matcher and the latch are both per-call locals, so N concurrent requests each stop on
        // their OWN stop strings and nobody else's.
        let mut stops = infr_server::StopMatcher::new(params.stop.clone());
        let stats = {
            let od = &mut *on_delta;
            // Template-prefilled thinking (the PROMPT ends with the `<think>` opener): inject a
            // synthetic opener so the splitter emits the head as Reasoning deltas, mirroring
            // `Chat::turn` — run and serve expose thinking identically.
            if infr_engine::prompt_prefills_think(&prompt) {
                stream.push("<think>", &mut *od);
            }
            let stats = be.generate(&prompt, max_new, None, &req, &mut |piece: &str| {
                let safe = stops.push(piece);
                if !safe.is_empty() {
                    stream.push(&safe, &mut *od);
                }
                if stops.hit() {
                    req.abort();
                }
            })?;
            // No stop fired: whatever the matcher was holding back was never a stop prefix.
            let tail = stops.flush();
            if !tail.is_empty() {
                stream.push(&tail, &mut *od);
            }
            stats
        };
        stream.finish(on_delta);
        Ok(if stops.hit() {
            infr_server::Finish::Stop
        } else if stats.n_gen >= max_new {
            // The budget was exhausted (EOS would have broken the loop earlier).
            infr_server::Finish::Length
        } else {
            infr_server::Finish::Stop
        })
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
    // diffusion-gemma (Phase 4/D, docs/DIFFUSIONGEMMA.md): llama-bench has no diffusion mode, so
    // `infr bench` measures infr's OWN decode shape (block prefill + canvas denoise, see
    // `cmd_bench_diffusion_gemma`'s doc) instead of routing through the AR pp/tg arms below.
    // Backend selection mirrors `cmd_run`/`cmd_serve`: -ngl 0 or INFR_CPU picks the CPU reference
    // session; INFR_METAL (or --dev Metal) picks the Metal session (Phase D — macOS only, see
    // `cmd_bench_diffusion_gemma`'s own cfg-gated dispatch).
    if infr_llama::diffusion::is_diffusion_gemma(&gguf) {
        let metal = std::env::var("INFR_METAL").is_ok() || dev.eq_ignore_ascii_case("metal");
        let cpu = ngl == 0 || std::env::var("INFR_CPU").is_ok();
        if ubatch > 0 {
            std::env::set_var("INFR_UBATCH", ubatch.to_string());
        }
        return cmd_bench_diffusion_gemma(
            &gguf,
            tok.as_deref(),
            n_prompt,
            n_gen,
            depth,
            pg,
            reps,
            cpu,
            metal,
            json,
        );
    }
    // qwen35 (Qwen3.5) benches through the STANDARD arms below (`cmd_bench_cpu` / the seam's
    // `bench_vulkan` / `cmd_bench_metal`) — `SeamModel::load` drives it through the unified runner
    // (`Config::from_gguf` + `MixerW::DeltaNet`), reusing the exact same pp/tg/depth methodology
    // every other arch gets (no more qwen35-only bench arm or depth-accounting artifacts).
    // -ngl 0: run on the CPU reference backend (no GPU), comparable to `llama-bench -ngl 0`.
    // llama4 benches through the standard Vulkan arm below like every other model now (the paged
    // expert cache) — only -ngl 0 forces it onto this CPU arm.
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
    let model = infr_llama::SeamModel::load(&gguf, tok.as_deref())?;
    let samples = model.bench_vulkan(n_prompt, n_gen, depth, pg, reps)?;
    let label = if let Some((p, g)) = pg {
        format!("pg{p}+{g}")
    } else if n_gen > 0 {
        format!("tg{n_gen}")
    } else {
        format!("pp{n_prompt}")
    };
    // MTP arm (issue #33, phase 4 — perf-bottleneck visibility): a model that ships an MTP head
    // gets its self-speculative decode measured ADDITIONALLY alongside the baseline tg above (not
    // instead — `36 t/s vs 100 t/s baseline` is the whole point of this pass being visible in one
    // run). Only when there's a decode length to measure (`-n > 0`) and no `--pg` (that's a
    // combined ingest+reply shape the MTP arm doesn't have an equivalent for). Models without a
    // head take the exact path above unchanged — `mtp` stays `None`, `print_bench_avg_mtp` falls
    // straight through to the pre-existing `print_bench_avg` line.
    let mtp = if infr_llama::mtp::mtp_enabled()
        && pg.is_none()
        && n_gen > 0
        && model.config().n_layer_nextn > 0
    {
        Some(bench_mtp_tg(&model, n_prompt, depth, n_gen, reps)?)
    } else {
        None
    };
    print_bench_avg_mtp(&samples, &label, depth, "", reps, json, mtp.as_ref());
    Ok(())
}

/// The MTP-spec-decode measurement `infr bench` reports alongside the baseline tg rate (issue #33,
/// phase 4) — for models whose GGUF ships an MTP head. Runs
/// [`infr_llama::mtp::generate_mtp_spec_vulkan_timed`] on a synthetic text prompt sized to match
/// `-p`/`-d` (the same "repeat a fixed sentence to ~N tokens" convention used below —
/// the MTP driver takes a rendered PROMPT, not raw token ids, unlike `bench_vulkan`'s dummy-id
/// arm), once per rep (no persistent MTP session yet — `docs/MTP.md`'s Phase 3 doc on
/// `generate_mtp_spec_vulkan`'s per-call fresh trunk+head — so each rep re-pays the full weight
/// upload; keep `-r` small for this arm), and aggregates the per-cycle draft/verify/catchup wall
/// time into phase shares + the accept rate (alpha) via [`infr_llama::mtp::MtpTiming`].
struct MtpBenchStats {
    n_gen: usize,
    ts: f64,
    alpha: f64,
    draft_pct: f64,
    verify_pct: f64,
    catchup_pct: f64,
}

/// The one sentence (~10 tokens/rep) every MTP measurement decodes from, on BOTH engines
/// ([`bench_mtp_tg`] for infr, `ModelBench::MTP_PROMPT` → llama-cli `-no-cnv` for the oracle) —
/// α is content-sensitive, so cross-engine mtp ratios are only meaningful on shared
/// un-templated content.
const MTP_SENTENCE: &str = "The quick brown fox jumps over the lazy dog. ";

fn bench_mtp_tg(
    model: &infr_llama::SeamModel,
    n_prompt: usize,
    depth: usize,
    n_gen: usize,
    reps: usize,
) -> anyhow::Result<MtpBenchStats> {
    let head = infr_llama::mtp::load_mtp_head(model.gguf(), model.config())?;
    let want = (n_prompt + depth).max(1);
    let prompt = MTP_SENTENCE.repeat(want.div_ceil(10));
    let mut samples = Vec::with_capacity(reps.max(1));
    let mut timing = infr_llama::mtp::MtpTiming::default();
    // Backend selection mirrors the rest of the CLI: INFR_METAL routes the MTP driver onto the
    // Apple-GPU trunk+head (issue #39 — measuring Metal's accept rate needs the Metal timed path,
    // not Vulkan's), everything else stays on Vulkan. The two timed fns share a signature.
    let use_metal = std::env::var("INFR_METAL").is_ok();
    for _ in 0..reps.max(1) {
        let (stats, t) = if use_metal {
            #[cfg(target_os = "macos")]
            {
                infr_llama::mtp::generate_mtp_spec_metal_timed(
                    model,
                    &head,
                    &prompt,
                    n_gen,
                    |_| {},
                )?
            }
            #[cfg(not(target_os = "macos"))]
            {
                anyhow::bail!("INFR_METAL MTP bench requires macOS");
            }
        } else {
            infr_llama::mtp::generate_mtp_spec_vulkan_timed(model, &head, &prompt, n_gen, |_| {})?
        };
        samples.push(stats.n_gen as f64 / stats.decode_secs.max(1e-9));
        timing.add(&t);
    }
    let ts = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
    let (draft_pct, verify_pct, catchup_pct) = timing.phase_shares();
    Ok(MtpBenchStats {
        n_gen,
        ts,
        alpha: timing.alpha(),
        draft_pct,
        verify_pct,
        catchup_pct,
    })
}

/// [`print_bench_avg`] plus the optional MTP segment: `mtp` is `None` for every model without a
/// head (or `-pg`/`-n 0` bench calls), in which case this is BYTE-IDENTICAL to the pre-MTP output
/// (falls straight through to `print_bench_avg`) — the "no new flags, unchanged output" guarantee
/// for non-MTP models.
fn print_bench_avg_mtp(
    samples: &[f64],
    label: &str,
    depth: usize,
    // Backend tag passed through to the no-MTP fallback (" [metal]" / " [cpu]" / "") — PR #42
    // routed cmd_bench_metal here and the hardcoded "" silently dropped its " [metal]" tag.
    suffix: &str,
    reps: usize,
    json: bool,
    mtp: Option<&MtpBenchStats>,
) {
    let Some(m) = mtp else {
        print_bench_avg(samples, label, depth, suffix, reps, json);
        return;
    };
    let avg = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
    let ratio = m.ts / avg.max(1e-9);
    if json {
        println!(
            "[{{\"avg_ts\": {avg:.2}, \"mtp_ts\": {:.2}, \"mtp_ratio\": {ratio:.4}, \"alpha\": {:.4}, \"draft_pct\": {:.1}, \"verify_pct\": {:.1}, \"catchup_pct\": {:.1}}}]",
            m.ts, m.alpha, m.draft_pct, m.verify_pct, m.catchup_pct
        );
        return;
    }
    let d = if depth > 0 {
        format!(" @ d{depth}")
    } else {
        String::new()
    };
    println!(
        "{label}{d}: {avg:.1} t/s | mtp{}: {:.1} t/s ({ratio:.2}x, alpha={:.2}, draft {:.0}% verify {:.0}% catchup {:.0}%)  ({reps} reps)",
        m.n_gen, m.ts, m.alpha, m.draft_pct, m.verify_pct, m.catchup_pct
    );
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
/// backend (`SeamModel::bench_metal`). On non-macOS this arm is unreachable (the backend crate
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
        let model = infr_llama::SeamModel::load(gguf, tok)?;
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
        // MTP arm (issue #33/#39): twin of the Vulkan path's arm — a model shipping an MTP head
        // gets its self-speculative Metal decode (accept rate + phase split) measured alongside the
        // baseline tg above. `bench_mtp_tg` reads INFR_METAL (set here) to route onto the Metal
        // timed driver. Models without a head keep `mtp = None` → byte-identical to the old output.
        let mtp = if infr_llama::mtp::mtp_enabled()
            && pg.is_none()
            && n_gen > 0
            && model.config().n_layer_nextn > 0
        {
            Some(bench_mtp_tg(&model, n_prompt, depth, n_gen, reps)?)
        } else {
            None
        };
        print_bench_avg_mtp(
            &samples,
            &label,
            depth,
            " [metal]",
            reps,
            json,
            mtp.as_ref(),
        );
        Ok(())
    }
}

/// CPU-backend bench (`infr bench -ngl 0`): the GPU bench's pp/tg/pg metrics on the agnostic CPU
/// reference path, using `SeamModel`'s token-level timing — directly comparable to `llama-bench -ngl 0`.
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
    let model = infr_llama::SeamModel::load(gguf, tok)?;
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

/// Aggregated result of one diffusion-gemma decode measurement (Phase 4/E,
/// `docs/DIFFUSIONGEMMA.md`): the same numbers `cmd_bench_diffusion_gemma` prints, factored out of
/// [`dg_bench_run`] so the compare arm (`ModelBench::dg_infr`) can read them directly instead of
/// scraping this command's own stdout. `pp_ts`/`gen_ts`/`parallel_ts`/`steps` are already averaged
/// over `reps`; `last_np`/`last_ng` are the LAST rep's actual prompt/gen token counts (matches
/// every other bench arm's "label from the last rep" convention).
struct DgBenchResult {
    pp_ts: f64,
    /// n_gen / decode_secs — end-to-end, THIS run's own step count. NOT apples-to-apples across
    /// implementations (entropy-bound step counts are content/impl-sensitive) — see the compare
    /// arm's dg-step/dg-e2e comment for why `parallel_ts` is the metric that ratio is built on.
    gen_ts: f64,
    /// canvas_length * steps / decode_secs — the in-step-parallel rate: the number that reflects
    /// actual forward-pass work regardless of how many steps the entropy-bound sampler took.
    /// Directly comparable to llama.cpp's own `llama-diffusion-cli` "in-step parallel N tok/s".
    parallel_ts: f64,
    steps: f64,
    last_np: usize,
    last_ng: usize,
}

/// Core diffusion-gemma decode loop (Phase 4, `docs/DIFFUSIONGEMMA.md`): drives
/// `crate::diffusion::diffusion_generate` directly over a persistent
/// `DiffusionGemmaCpuSession`/`DiffusionGemmaVulkanSession` — the SAME primitive
/// `DiffusionGemmaChat::generate` (run/serve) uses — rather than going through the generic
/// `ChatModel::generate`, because bench needs the step/block counts `GenStats` alone doesn't carry.
/// Used by BOTH `cmd_bench_diffusion_gemma` (prints the result) and `ModelBench::dg_infr` (Phase
/// 4/E, the `compare`/`compare --sweep` DG arm — reads the result directly instead of shelling out
/// to `infr bench` and parsing its printed text, which is what every OTHER arch's compare arm does
/// via `infr bench --json`; DG's bench has no `--json` support for the per-step fields).
///
/// `-p P`/`-d D` behave like every other bench arm: `D` extra prompt tokens are prefilled UNTIMED
/// first, then `P` more (an exact-prefix extension of the same real-text token sequence — see the
/// in-body comment on why NOT the AR arms' `i % 100` dummy ids) through the SAME session — its
/// reuse forwards only the new `P` suffix, so `prompt_secs` times exactly that, matching the
/// `-d`/`-p` split every other arch's bench reports. `-n N` then times the block-diffusion decode of
/// N tokens (`ceil(N / canvas_length)` whole canvas blocks — a `-n 64` request still denoises a
/// full `canvas_length`-token canvas per step, so besides the naive "committed tokens / decode
/// secs" rate this also reports the oracle's own "in-step parallel" rate
/// (`canvas_length * steps / decode_secs`, see the reference runs captured in
/// `docs/DIFFUSIONGEMMA.md`) — the number that actually reflects how much forward-pass work ran.
/// `-n 0` measures prefill only (pp, like every other arch's `-n 0`).
///
/// `--pg` has no diffusion-shaped meaning (a denoise block isn't an ingest-then-reply AR turn) —
/// `cmd_bench_diffusion_gemma` errors clearly on it rather than silently mis-measuring; this
/// function itself has no `pg` parameter at all. `-t`/`-u` keep their generic meaning (rayon
/// threads for the entropy-bound sampler's per-position reduction; the shared prefill chunk size —
/// DG's causal prefill rides the exact same `generate_dense_backend` chunked loop as every other
/// arch) with no DG-specific wiring needed.
#[allow(clippy::too_many_arguments)]
fn dg_bench_run(
    gguf: &Path,
    tok: Option<&Path>,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    reps: usize,
    // `Some(text)`: chat-template-render + encode THIS prompt instead of the synthetic repeated
    // sentence, and ignore `n_prompt` (requires depth == 0). The compare arm passes its shared
    // natural prompt so BOTH sides' entropy-bound loops converge on the same (templated) content —
    // with the synthetic prompt infr always rode the 48-step cap while the oracle converged in ~23
    // (steps are content-sensitive, so e2e was double-penalized by prompt choice alone). See the
    // `prompt_ids` binding below for why this is templated, not raw-encoded.
    prompt_text: Option<&str>,
    cpu: bool,
    // Phase D: the Metal DG session (macOS only — see the `one_rep!` dispatch below). `cpu` wins
    // if both are set (matches `cmd_run`/`cmd_serve`'s precedence).
    metal: bool,
) -> anyhow::Result<DgBenchResult> {
    use infr_llama::diffusion::{diffusion_generate, EbConfig};
    let model = infr_llama::SeamModel::load(gguf, tok)?;
    let cfg = model.config();
    let canvas_len = cfg.canvas_length.max(1);
    let vocab = cfg.vocab;
    let eb = EbConfig::from_config(cfg);
    // INFR_IGNORE_EOS (cmd_bench always sets it): a fixed generation budget should never
    // early-stop on an EOS id, matching every AR bench arm's semantics for the same env.
    let eos_ids: Vec<u32> = if std::env::var("INFR_IGNORE_EOS").is_ok() {
        Vec::new()
    } else {
        cfg.eos_ids.clone()
    };
    let seed: u64 = std::env::var("INFR_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);

    if prompt_text.is_some() && depth > 0 {
        anyhow::bail!("dg_bench_run: prompt_text override is depth-0 only (compare's DG arm)");
    }
    // Real-prompt mode (compare's DG arm): the reference fork's OWN CLI applies the model's chat
    // template by default before it ever decodes `-p`'s text (`common_params::enable_chat_template`
    // defaults `true`; `diffusion-cli.cpp`'s `apply_template`/`common_chat_templates_apply`, called
    // unless `--no-chat-template` is passed — which the compare arm's `llama_diffusion` invocation
    // does NOT pass). So the oracle's ~23-step convergence is measured on a rendered
    // `<start_of_turn>user ... <start_of_turn>model` turn, never on the bare sentence. Encoding the
    // bare sentence here (as this used to) fed our side a badly out-of-distribution prompt for an
    // instruction-tuned model: entropy never settled — `INFR_EB_TRACE=1` showed it oscillating
    // (not monotonically falling) with acceptance stuck near 1/256 for dozens of steps — so infr
    // rode the full step cap on EVERY block while the fork, decoding the SAME text but templated,
    // converged normally. Render the identical single-user-turn template `cmd_run`/
    // `DiffusionGemmaChat` already use for a real chat reply, so both tools decode the model's
    // actual instruction-tuned input shape instead of two different prompts that merely share text.
    let prompt_ids = prompt_text
        .map(|text| model.encode(&model.render_chat_messages(&[("user", text)])?))
        .transpose()?;
    let p_eff = match &prompt_ids {
        Some(ids) => ids.len().max(1),
        None => n_prompt.max(1), // DG always needs >=1 prompt token (the canvas's causal prefix)
    };
    let blocks_wanted = if n_gen > 0 {
        n_gen.div_ceil(canvas_len).max(1)
    } else {
        1
    };
    let max_ctx = depth + p_eff + blocks_wanted * canvas_len + 64;

    // Unlike the AR bench arms' `i % 100` dummy ids (matmul timing there truly IS
    // content-independent), DG's entropy-bound sampler converges (steps run, trim point) on the
    // ACTUAL logits — an out-of-distribution raw-id prompt made the model collapse the whole
    // canvas to one repeated token in 2 steps (`trim_canvas` then cut it to 0 committed tokens),
    // which measures a degenerate path instead of the shape `-n` asks for. Encode a real repeated
    // sentence instead (same "fixed synthetic prompt" convention `bench_mtp_tg` uses) and slice
    // it: `dummy(depth)` is then an exact prefix of `dummy(depth + p_eff)` by construction, so the
    // untimed depth warm + timed suffix prefill still gets the session's prefix-diff reuse.
    let full_ids = match prompt_ids {
        Some(ids) => ids,
        None => {
            let long_text =
                "The quick brown fox jumps over the lazy dog. ".repeat((depth + p_eff) / 4 + 8);
            model.encode(&long_text)?
        }
    };
    if full_ids.len() < depth + p_eff {
        anyhow::bail!(
            "internal: synthetic bench prompt too short ({} ids for depth {depth} + p {p_eff})",
            full_ids.len()
        );
    }
    let dummy = |n: usize| -> Vec<u32> { full_ids[..n.max(1)].to_vec() };

    let mut pps = Vec::with_capacity(reps.max(1));
    let mut gens = Vec::with_capacity(reps.max(1));
    let mut steps_v = Vec::with_capacity(reps.max(1));
    let mut parallel_v = Vec::with_capacity(reps.max(1));
    let (mut last_np, mut last_ng) = (0usize, 0usize);
    // Untimed pipeline-warmup tokens: NOT a prefix of the timed sequence (reversed), so the
    // timed prefill still re-prefills from scratch after the prefix-diff — the warmup only
    // pre-compiles the lazily-built pipelines, exactly like the AR bench arms' untimed warmup
    // (a cold DG prefill measured 26 t/s vs 1424 t/s warm — all one-time compile).
    let warm_ids: Vec<u32> = full_ids[..8.min(full_ids.len())]
        .iter()
        .rev()
        .copied()
        .collect();

    for _ in 0..reps.max(1) {
        macro_rules! one_rep {
            ($sess:expr) => {{
                let mut sess = $sess;
                sess.prefill(&model, &warm_ids)?; // untimed pipeline warmup (see warm_ids)
                if depth > 0 {
                    sess.prefill(&model, &dummy(depth))?; // untimed depth warm
                }
                if n_gen == 0 {
                    // pp-only: no canvas denoise, just the timed prefill (matches every other
                    // arch's `-n 0` meaning).
                    let t0 = std::time::Instant::now();
                    sess.prefill(&model, &dummy(depth + p_eff))?;
                    let secs = t0.elapsed().as_secs_f64();
                    pps.push(p_eff as f64 / secs.max(1e-9));
                    last_np = p_eff;
                } else {
                    let result = diffusion_generate(
                        &mut sess,
                        &model,
                        &dummy(depth + p_eff),
                        canvas_len,
                        vocab,
                        &eos_ids,
                        &eb,
                        n_gen,
                        seed,
                        max_ctx,
                        None,
                        None,
                    )?;
                    pps.push(p_eff as f64 / result.stats.prompt_secs.max(1e-9));
                    gens.push(result.stats.n_gen as f64 / result.stats.decode_secs.max(1e-9));
                    parallel_v.push(
                        (canvas_len * result.steps) as f64 / result.stats.decode_secs.max(1e-9),
                    );
                    steps_v.push(result.steps as f64);
                    last_np = p_eff;
                    last_ng = result.stats.n_gen;
                }
            }};
        }
        if cpu {
            one_rep!(model.diffusion_gemma_cpu_session(max_ctx));
        } else if metal {
            // Phase D: Metal DG session, macOS only — the non-macOS build still compiles this
            // arm (errors clearly at runtime instead), matching every other INFR_METAL arm.
            #[cfg(target_os = "macos")]
            {
                one_rep!(model.diffusion_gemma_metal_session(max_ctx)?);
            }
            #[cfg(not(target_os = "macos"))]
            {
                anyhow::bail!("the Metal backend is only available on macOS");
            }
        } else {
            one_rep!(model.diffusion_gemma_vulkan_session(max_ctx)?);
        }
    }

    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    Ok(DgBenchResult {
        pp_ts: avg(&pps),
        gen_ts: avg(&gens),
        parallel_ts: avg(&parallel_v),
        steps: avg(&steps_v),
        last_np,
        last_ng,
    })
}

/// diffusion-gemma bench (Phase 4, `docs/DIFFUSIONGEMMA.md`): llama-bench has no diffusion mode
/// (no llama.cpp comparison possible via `llama-bench` — this is infr-only reporting; `infr
/// compare`'s DG arm instead shells `llama-diffusion-cli` from the fork directly, see
/// `ModelBench::llama_diffusion`), so this drives [`dg_bench_run`] and formats its result. See
/// `dg_bench_run`'s doc comment for the -p/-n/-d semantics.
#[allow(clippy::too_many_arguments)]
fn cmd_bench_diffusion_gemma(
    gguf: &Path,
    tok: Option<&Path>,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    pg: Option<(usize, usize)>,
    reps: usize,
    cpu: bool,
    metal: bool,
    json: bool,
) -> anyhow::Result<()> {
    if pg.is_some() {
        anyhow::bail!(
            "diffusion-gemma bench has no --pg equivalent (a denoise block isn't an ingest-then-reply \
             AR turn); use separate -p/-n instead — `infr bench <model> -p P -n N`"
        );
    }
    let r = dg_bench_run(gguf, tok, n_prompt, n_gen, depth, reps, None, cpu, metal)?;
    let tag = if cpu {
        " [cpu]"
    } else if metal {
        " [metal]"
    } else {
        ""
    };
    if json {
        let a = if n_gen > 0 { r.gen_ts } else { r.pp_ts };
        println!("[{{\"avg_ts\": {a:.2}}}]");
    } else if n_gen > 0 {
        println!(
            "pp{}: {:.1} t/s | gen{}: {:.1} t/s (end-to-end) | {:.1} steps | in-step parallel {:.1} t/s{tag}  ({reps} reps)",
            r.last_np, r.pp_ts, r.last_ng, r.gen_ts, r.steps, r.parallel_ts,
        );
    } else {
        println!("pp{}: {:.1} t/s{tag}  ({reps} reps)", r.last_np, r.pp_ts);
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
    const METRICS: [&str; 4] = ["pp512", "tg128", "tg64@d", "pp4@d"];
    println!(
        "\n{:<22} {:<10} | {:>9} | {:>9} | {:>10}",
        "model", "metric", "infr", "llama.cpp", "infr/llama"
    );
    println!("{:-<23}{:-<11}+{:-<11}+{:-<11}+{:-<12}", "", "", "", "", "");
    // (model, metric, infr, llama, ratio) for the ranked summary.
    let mut rows: Vec<(String, String, f64, f64)> = Vec::new();
    for model in models {
        let short = model.rsplit('/').next().unwrap_or(model);
        // Cooldown between models to reduce thermal skew across the sweep.
        std::thread::sleep(std::time::Duration::from_secs(10));
        let mb = match ModelBench::new(model, dev, ngl, threads, reps, ubatch, llama_bench) {
            Ok(mb) => mb,
            Err(e) => {
                println!("{short:<22} {:<10} | resolve failed: {e}", "-");
                continue;
            }
        };
        // diffusion-gemma (Phase 4/E): entirely different measurement shape (no upstream
        // llama-bench support — see `ModelBench::is_dg`'s doc comment), so it prints its own two
        // rows and skips the standard pp/tg/mtp metrics below entirely.
        if mb.is_dg {
            print_dg_sweep_rows(&mb, short, &mut rows);
            continue;
        }
        let ds = depth.to_string();
        // pp4@d: the tiny-suffix-turn shape (m=2..8 Linears at session depth) — the multi-row
        // GEMV / spec-verify path; invisible in pp512/tg but it IS multi-turn serve TTFT.
        let runs: [(&str, Vec<&str>, (usize, usize)); 4] = [
            ("pp512", vec!["-p", "512", "-n", "0"], (512, 0)),
            ("tg128", vec!["-p", "0", "-n", "128"], (0, 128)),
            (METRICS[2], vec!["-p", "0", "-n", "64", "-d", &ds], (0, 64)),
            (METRICS[3], vec!["-p", "4", "-n", "0", "-d", &ds], (4, 0)),
        ];
        for (metric, args, (np, ng)) in runs {
            let metric_label = if metric.ends_with("@d") {
                format!("{metric}{depth}") // "tg64@d" -> "tg64@d4096"
            } else {
                metric.to_string()
            };
            // Cooldown between metric pairs (infr + llama.cpp) to reduce thermal coupling.
            std::thread::sleep(std::time::Duration::from_secs(10));
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
        // mtp column (issue #33, phase 4): infr-vs-llama at the SAME tg128 shape above, but with
        // both tools' MTP spec decode on — MTP-capable models only (the cell costs one llama-cli
        // run + one infr mtp run; every other model prints a blank row instantly, keeping the
        // sweep's runtime dominated by the 4 metrics above, not this column).
        let metric_label = "mtp128".to_string();
        // MTP is PARKED (`infr_llama::mtp::mtp_enabled`'s doc) — the column stays in the table shape
        // (a blank row, same as any model without a head) so the sweep archive's columns keep lining
        // up with older archives, but neither engine is measured.
        if infr_llama::mtp::mtp_enabled() && mb.has_mtp_head() {
            let iv = mb.infr_mtp(&["-p", "0", "-n", "128"]);
            let lv = mb.llama_cli_mtp(128);
            let is = iv.as_ref().map(|v| format!("{v:.0}")).unwrap_or_else(|e| {
                eprintln!("infr mtp bench failed ({short}): {e:#}");
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
        } else {
            println!(
                "{short:<22} {metric_label:<10} | {:>9} | {:>9} | {:>10}",
                "-", "-", "-"
            );
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

/// `llama-cli`'s MTP arm has no JSON output (unlike `llama-bench -o json`) — it prints
/// `[ Prompt: X t/s | Generation: Y t/s ]` at exit, so pull the LAST `Generation: <float> t/s`
/// occurrence out of its combined stdout+stderr by hand (no regex dependency for one float). The
/// LAST match matters: some llama.cpp builds print an interim perf line before the final summary.
fn parse_llama_cli_gen_rate(output: &str) -> Option<f64> {
    const NEEDLE: &str = "Generation:";
    let mut last = None;
    let mut rest = output;
    while let Some(idx) = rest.find(NEEDLE) {
        let after = rest[idx + NEEDLE.len()..].trim_start();
        let end = after
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(after.len());
        if let Ok(v) = after[..end].parse::<f64>() {
            last = Some(v);
        }
        rest = &rest[idx + NEEDLE.len()..];
    }
    last
}

/// One model's infr-vs-llama.cpp bench harness: resolves the shared model ref once and shells
/// out to `infr bench --json` / `llama-bench -o json` with MATCHED flags. Both the deep
/// coding-agent scenarios (`cmd_compare`) and the multi-model survey (`cmd_compare_sweep`) run
/// through this, so the two tools are always measured identically.
struct ModelBench {
    exe: PathBuf,
    model: String,
    /// The resolved local GGUF path — `llama-cli`'s MTP arm (issue #33, phase 4) shells the
    /// binary directly (no `llama-bench` JSON plumbing for spec decode: `llama-bench` doesn't run
    /// spec at all), so it needs a real `-m <path>` regardless of whether `model` was an `-hf` ref.
    gguf_path: PathBuf,
    /// Sidecar tokenizer.json beside the GGUF, if any — DG's compare arm (`dg_infr`) loads the
    /// model in-process (`dg_bench_run`) rather than shelling out, so it needs this the same way
    /// `cmd_bench`/`cmd_run` do (see `resolve`'s doc comment).
    tok_path: Option<PathBuf>,
    /// arch=diffusion-gemma (Phase 4/E, `docs/DIFFUSIONGEMMA.md`): gates the DG rows in
    /// `cmd_compare`/`cmd_compare_sweep` — no upstream-merged `llama-bench` support exists for
    /// this arch, so it takes a completely different pair of measurement paths (`dg_infr` +
    /// `llama_diffusion`) instead of the standard `infr`/`llama` pp/tg arms.
    is_dg: bool,
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
        let (resolved, tok_path) = resolve(model)?;
        // diffusion-gemma (Phase 4/E, docs/DIFFUSIONGEMMA.md): the arch's PR isn't merged into
        // mainline llama.cpp, so `llama-bench` can't run it — but the reference fork at
        // `~/Projects/mxaddict/llama.cpp-dg` builds `llama-diffusion-cli`, which IS a usable
        // oracle (see `ModelBench::llama_diffusion`/`llama_diffusion_cli_path`). `is_dg` routes
        // `cmd_compare`/`cmd_compare_sweep` to the DG-shaped measurement pair instead of bailing.
        let is_dg = infr_llama::diffusion::is_diffusion_gemma(&resolved);
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
            gguf_path: resolved,
            tok_path,
            is_dg,
            llama_model_args,
            llama_bench: llama_bench.to_string(),
            dev: dev.to_string(),
            ngl,
            threads,
            reps,
            ubatch,
        })
    }

    /// Whether this model's GGUF ships an MTP head (issue #33, phase 4) — gates the extra
    /// `mtp`/`MTP DECODE` measurements in `cmd_compare`/`cmd_compare_sweep` so non-MTP models pay
    /// nothing extra.
    fn has_mtp_head(&self) -> bool {
        infr_llama::mtp::has_mtp_head(&self.gguf_path)
    }

    /// Run `infr bench --json` and return the parsed row object (`[{"avg_ts": .., ..}]`'s first
    /// element) — shared by [`infr`](Self::infr) (`avg_ts`) and [`infr_mtp`](Self::infr_mtp)
    /// (`mtp_ts`), so both read one shelled-out call's worth of plumbing.
    fn infr_json(&self, args: &[&str]) -> anyhow::Result<serde_json::Value> {
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
        serde_json::from_slice(&out.stdout).with_context(|| {
            format!(
                "parsing infr bench output: {}",
                String::from_utf8_lossy(&out.stdout)
            )
        })
    }

    /// Run `infr bench` (this binary) and read its single-row [{"avg_ts":X}].
    fn infr(&self, args: &[&str]) -> anyhow::Result<f64> {
        self.infr_json(args)?[0]["avg_ts"]
            .as_f64()
            .context("infr bench: missing avg_ts")
    }

    /// Run `infr bench` and read the `mtp_ts` field `bench_mtp_tg`'s JSON output adds for
    /// MTP-capable models (issue #33, phase 4) — `args` should include `-n <N>` matching the
    /// baseline `infr` call this is compared against (the sweep's/compare's tg128 shape).
    fn infr_mtp(&self, args: &[&str]) -> anyhow::Result<f64> {
        self.infr_json(args)?[0]["mtp_ts"]
            .as_f64()
            .context("infr bench: missing mtp_ts (model has no MTP head, or -n was 0)")
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

    /// The fixed prompt both tools' MTP arm decodes from — the SAME raw sentence
    /// [`bench_mtp_tg`] feeds infr's side, passed to llama-cli with `-no-cnv` so neither side
    /// chat-templates it. MTP throughput is dominated by the accept rate α, and α is strongly
    /// content-sensitive (greedy drafting collapses on some regimes) — the two engines are only
    /// comparable decoding the SAME un-templated content. The previous asymmetry (infr: raw fox
    /// sentence, α≈0.51 on the 9B; llama-cli: chat-templated "capital of France", a friendlier
    /// regime) manufactured ~0.05-0.08 of phantom ratio gap in the sweep's mtp128 row.
    const MTP_PROMPT: &str = MTP_SENTENCE;

    /// `llama-cli` sits alongside `llama-bench` in the same install (`/usr/sbin` on this box) —
    /// derive its path from `--llama-bench` instead of adding a second CLI flag: replace the
    /// binary name when `llama_bench` names it explicitly, else assume it's on `PATH` like the
    /// `llama-bench` default is.
    fn llama_cli_path(&self) -> String {
        if self.llama_bench.contains("llama-bench") {
            self.llama_bench.replace("llama-bench", "llama-cli")
        } else {
            "llama-cli".to_string()
        }
    }

    /// Shell `llama-cli` directly with MTP spec decode on (`llama-bench` has no spec-decode mode
    /// at all — issue #33's phase 4 context) and parse the LAST `Generation: X t/s` line from its
    /// combined stdout+stderr (robust to which stream a given llama.cpp build writes the perf
    /// summary to). `-r` reps average, matching every other measurement this tool prints.
    fn llama_cli_mtp(&self, n_gen: usize) -> Option<f64> {
        use std::process::Command;
        let cli = self.llama_cli_path();
        let gguf = self.gguf_path.to_string_lossy().into_owned();
        let n = n_gen.to_string();
        let mut samples = Vec::with_capacity(self.reps.max(1));
        for _ in 0..self.reps.max(1) {
            let out = Command::new(&cli)
                .args(["-m", &gguf])
                .args(["-ngl", "99"])
                .args(["-p", Self::MTP_PROMPT])
                .args(["-n", &n])
                .args(["--temp", "0", "--single-turn", "-no-cnv"])
                .args(["--spec-type", "draft-mtp", "--spec-draft-n-max", "6"])
                .output()
                .ok()?;
            let combined = format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            match parse_llama_cli_gen_rate(&combined) {
                Some(v) => samples.push(v),
                None => {
                    eprintln!(
                        "llama-cli MTP run produced no parseable `Generation: X t/s` line: {}",
                        combined.trim()
                    );
                    return None;
                }
            }
        }
        Some(samples.iter().sum::<f64>() / samples.len().max(1) as f64)
    }

    /// infr's side of the DG compare arm (Phase 4/E): drives [`dg_bench_run`] — the SAME core
    /// `cmd_bench_diffusion_gemma` uses — in-process, at a fixed `-n <n_gen>` decode with a
    /// minimal 1-token prompt and no extra depth (this arm measures decode throughput only, not
    /// a coding-agent-shaped prefill scenario). `cpu` follows `ngl == 0`, matching every other
    /// arm's `-ngl 0` = CPU-reference-backend convention; Metal has no wiring here (`compare`
    /// doesn't expose a Metal device selector — out of scope for this arm).
    fn dg_infr(&self, n_gen: usize) -> anyhow::Result<DgBenchResult> {
        dg_bench_run(
            &self.gguf_path,
            self.tok_path.as_deref(),
            1, // ignored — the shared natural prompt below defines the prefix
            n_gen,
            0,
            self.reps,
            // The SAME prompt the fork oracle decodes from (`Self::DG_PROMPT`) — entropy-bound
            // step counts are content-sensitive, so e2e is only comparable on shared content.
            Some(Self::DG_PROMPT),
            self.ngl == 0,
            false,
        )
    }

    /// Fixed prompt for [`llama_diffusion`](Self::llama_diffusion)'s oracle run — arbitrary but
    /// fixed content, matching `Self::MTP_PROMPT`'s convention of one shared literal both sides
    /// decode from (entropy-bound step counts are content-sensitive, see `dg_bench_run`'s comment
    /// on why `infr`'s own dummy prompt is a fixed real sentence rather than raw ids).
    const DG_PROMPT: &str = "Write a short story about a robot learning to paint.";

    /// The reference fork's own build directories — `~/Projects/mxaddict/llama.cpp-dg`, the only
    /// place `arch=diffusion-gemma` support actually exists (it isn't merged into mainline
    /// llama.cpp): `build-vulkan` (GPU, `cmake -DGGML_VULKAN=ON`) or `build` (CPU-only). `cpu`
    /// picks between them, mirroring `-ngl 0` vs `-ngl >0` on every other arm. Used both as
    /// [`llama_diffusion_cli_path`](Self::llama_diffusion_cli_path)'s last-resort candidate AND as
    /// [`llama_diffusion`](Self::llama_diffusion)'s fallback target (see its doc comment).
    fn fork_diffusion_cli_path(cpu: bool) -> PathBuf {
        let fork = PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("Projects/mxaddict/llama.cpp-dg");
        if cpu {
            fork.join("build/bin/llama-diffusion-cli")
        } else {
            fork.join("build-vulkan/bin/llama-diffusion-cli")
        }
    }

    /// Resolve `llama-diffusion-cli`'s binary path (Phase 4/E compare arm) — precedence, in order:
    ///   1. `INFR_LLAMA_DIFFUSION_CLI` env var (explicit override, e.g. a custom build location).
    ///   2. `llama-diffusion-cli` on `PATH` (a manual PATH walk — no extra `which` dependency for
    ///      one binary lookup).
    ///   3. [`fork_diffusion_cli_path`](Self::fork_diffusion_cli_path) — the reference fork's build.
    ///
    /// CAVEAT: mainline llama.cpp already ships a generic `llama-diffusion-cli` (LLaDA/Dream
    /// support), so tier 2 can resolve to a REAL binary that nonetheless doesn't know
    /// `arch=diffusion-gemma` (the fork's own unmerged addition) — it loads and errors "unknown
    /// model architecture" rather than failing to resolve at all, so this function alone can't
    /// detect the mismatch. [`llama_diffusion`](Self::llama_diffusion) is the one that actually
    /// runs the binary, so it catches that specific failure and falls through to tier 3 itself.
    fn llama_diffusion_cli_path(cpu: bool) -> PathBuf {
        if let Ok(p) = std::env::var("INFR_LLAMA_DIFFUSION_CLI") {
            return PathBuf::from(p);
        }
        if let Ok(path_var) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path_var) {
                let cand = dir.join("llama-diffusion-cli");
                if cand.is_file() {
                    return cand;
                }
            }
        }
        Self::fork_diffusion_cli_path(cpu)
    }

    /// Shell `llama-diffusion-cli` (see [`llama_diffusion_cli_path`](Self::llama_diffusion_cli_path)
    /// for the resolution precedence) at a fixed `-n <n_gen>` decode and parse its own
    /// throughput/in-step-parallel/step-count numbers straight out of combined stdout+stderr (this
    /// binary has no `-o json` mode). `-ngl 999 -dev <dev>` for the GPU compare, `-ngl 0` for CPU —
    /// mirrors [`Self::llama`]'s handling.
    ///
    /// FALLBACK: if the resolved binary loads but rejects the model with "unknown model
    /// architecture" (the tier-2 PATH caveat documented on `llama_diffusion_cli_path` — a mainline
    /// install's generic diffusion CLI lacking this arch), retry against
    /// [`fork_diffusion_cli_path`](Self::fork_diffusion_cli_path) directly instead of reporting a
    /// hard failure. Detected once per call and then stuck to for the remaining reps.
    fn llama_diffusion(&self, n_gen: usize) -> Option<DgLlamaResult> {
        use std::process::Command;
        let cpu = self.ngl == 0;
        let mut cli = Self::llama_diffusion_cli_path(cpu);
        if !cli.is_file() {
            eprintln!(
                "llama-diffusion-cli not found ({}): set INFR_LLAMA_DIFFUSION_CLI, put it on PATH, \
                 or build the fork at ~/Projects/mxaddict/llama.cpp-dg (cmake -DGGML_VULKAN=ON \
                 -B build-vulkan && cmake --build build-vulkan --target llama-diffusion-cli)",
                cli.display()
            );
            return None;
        }
        let n = n_gen.to_string();
        let run = |cli: &Path| -> Option<String> {
            let mut c = Command::new(cli);
            c.args(&self.llama_model_args);
            c.args(["-p", Self::DG_PROMPT, "-n", &n]);
            if cpu {
                c.args(["-ngl", "0"]);
            } else {
                c.args(["-ngl", "999", "-dev", &self.dev]);
            }
            let out = c.output().ok()?;
            Some(format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ))
        };
        let mut e2e = Vec::with_capacity(self.reps.max(1));
        let mut step = Vec::with_capacity(self.reps.max(1));
        let mut steps_v = Vec::with_capacity(self.reps.max(1));
        let mut fell_back = false;
        for _ in 0..self.reps.max(1) {
            let mut combined = run(&cli)?;
            if !fell_back && combined.contains("unknown model architecture") {
                let fork = Self::fork_diffusion_cli_path(cpu);
                eprintln!(
                    "{} doesn't support arch=diffusion-gemma (mainline install?) — falling back \
                     to the fork build at {}",
                    cli.display(),
                    fork.display()
                );
                cli = fork;
                fell_back = true;
                combined = run(&cli)?;
            }
            match parse_llama_diffusion_cli_output(&combined) {
                Some(r) => {
                    e2e.push(r.e2e_ts);
                    step.push(r.step_ts);
                    steps_v.push(r.steps);
                }
                None => {
                    eprintln!(
                        "llama-diffusion-cli produced no parseable throughput line: {}",
                        combined.trim()
                    );
                    return None;
                }
            }
        }
        let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        Some(DgLlamaResult {
            e2e_ts: avg(&e2e),
            step_ts: avg(&step),
            steps: avg(&steps_v),
        })
    }
}

/// llama.cpp's `llama-diffusion-cli` (the fork's DG oracle, Phase 4/E) prints its summary as two
/// plain-text lines (no `-o json` mode on this binary), e.g.:
///   `total time: 39559.95ms, time per step: 1582.40ms (25 steps over 1 blocks, entropy-bound)`
///   `throughput: 6.5 tok/s (256 tok in 39559.95ms), in-step parallel 162 tok/s (256-tok canvas x 25.0 steps/block)`
/// `e2e_ts` ("throughput:") and `step_ts` ("in-step parallel ... tok/s") both come off the SECOND
/// line; `steps` (total denoising steps run) comes off the FIRST line's "N steps over" — parsed
/// separately since it's the one number not repeated on the throughput line.
struct DgLlamaResult {
    e2e_ts: f64,
    step_ts: f64,
    steps: f64,
}

/// Pull a `float` out of `s` immediately after the first occurrence of `needle` (shared by
/// [`parse_llama_diffusion_cli_output`]'s two fields — same "walk forward from a fixed label"
/// approach as [`parse_llama_cli_gen_rate`], factored out here since this arm needs it twice).
fn parse_float_after(s: &str, needle: &str) -> Option<f64> {
    let idx = s.find(needle)?;
    let after = s[idx + needle.len()..].trim_start();
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(after.len());
    after[..end].parse::<f64>().ok()
}

/// Parse `llama-diffusion-cli`'s two-line summary — see [`DgLlamaResult`]'s doc comment for the
/// exact format. `steps` is scraped from " steps over " on the first line (walking BACKWARD to
/// the start of the number, the mirror image of [`parse_float_after`]'s forward walk, since the
/// number precedes the needle here); `e2e_ts`/`step_ts` come off the "throughput:"/"in-step
/// parallel " labels on the second line via `parse_float_after`.
fn parse_llama_diffusion_cli_output(output: &str) -> Option<DgLlamaResult> {
    let steps = {
        const NEEDLE: &str = " steps over ";
        let idx = output.find(NEEDLE)?;
        let before = &output[..idx];
        let start = before
            .rfind(|c: char| !(c.is_ascii_digit() || c == '.'))
            .map(|i| i + 1)
            .unwrap_or(0);
        before[start..].parse::<f64>().ok()?
    };
    let line = output.lines().find(|l| l.contains("in-step parallel"))?;
    let e2e_ts = parse_float_after(line, "throughput:")?;
    let step_ts = parse_float_after(line, "in-step parallel ")?;
    Some(DgLlamaResult {
        e2e_ts,
        step_ts,
        steps,
    })
}

/// diffusion-gemma's fixed decode length for the compare arm (Phase 4/E) — mirrors the oracle
/// invocation quoted in `docs/DIFFUSIONGEMMA.md` (`-n 256`), so numbers seen here line up with
/// numbers already captured there.
const DG_N_GEN: usize = 256;

/// diffusion-gemma's compare arm (Phase 4/E, `docs/DIFFUSIONGEMMA.md`): shared by
/// `cmd_compare`/`cmd_compare_sweep` at the point their old hard bail on `arch=diffusion-gemma`
/// used to sit (`ModelBench::new` no longer bails — see `is_dg`'s doc comment). Runs BOTH tools'
/// DG decode once at a fixed `-n 256` and returns the raw measurements; callers format their own
/// rows because `cmd_compare` prints a labeled table and `cmd_compare_sweep` needs `(model,
/// metric, infr, llama)` tuples for the ranked summary.
///
/// METRIC HONESTY: DG is entropy-bound — both implementations run a DIFFERENT NUMBER of denoise
/// steps for the same `-n 256` request (the early-stop point is content/implementation-sensitive:
/// see `dg_bench_run`'s comment on why the dummy prompt is a fixed real sentence, not raw ids), so
/// wall-clock end-to-end tok/s is NOT apples-to-apples between infr and llama.cpp — whichever tool
/// happens to trim its canvas earlier looks faster for free, independent of actual GPU work done.
/// The number that DOES reflect forward-pass work is per-step throughput
/// (`canvas_length * steps / decode_secs`): llama.cpp's own oracle already reports this as
/// "in-step parallel N tok/s", and infr's `dg_bench_run` computes the IDENTICAL formula
/// (`DgBenchResult::parallel_ts`), so `dg-step` is the metric this arm treats as the real
/// infr-vs-llama ratio (it's what feeds the sweep's "BIGGEST GAPS" ranking). `dg-e2e` is still
/// printed for visibility — informational only, each side's OWN step count folded into the row
/// text so the mismatch is visible instead of hidden.
fn dg_compare_measure(mb: &ModelBench) -> (anyhow::Result<DgBenchResult>, Option<DgLlamaResult>) {
    let infr = mb.dg_infr(DG_N_GEN);
    let llama = mb.llama_diffusion(DG_N_GEN);
    (infr, llama)
}

/// `cmd_compare_sweep`'s DG rows: `dg-step` (in-step-parallel ratio — the real gap metric, feeds
/// `rows`/the ranked summary) and `dg-e2e` (informational, own-step-count e2e, NOT fed into
/// `rows`). See [`dg_compare_measure`]'s doc comment for why the split.
fn print_dg_sweep_rows(mb: &ModelBench, short: &str, rows: &mut Vec<(String, String, f64, f64)>) {
    let (infr, llama) = dg_compare_measure(mb);
    let step_label = "dg-step".to_string();
    let is = infr
        .as_ref()
        .map(|r| format!("{:.0}", r.parallel_ts))
        .unwrap_or_else(|e| {
            eprintln!("infr DG bench failed ({short}): {e:#}");
            "ERR".into()
        });
    let ls = llama
        .as_ref()
        .map(|r| format!("{:.0}", r.step_ts))
        .unwrap_or_else(|| "NA".into());
    let ratio = match (infr.as_ref().ok(), llama.as_ref()) {
        (Some(i), Some(l)) if l.step_ts > 0.0 => {
            rows.push((
                short.to_string(),
                step_label.clone(),
                i.parallel_ts,
                l.step_ts,
            ));
            format!("{:.2}x", i.parallel_ts / l.step_ts)
        }
        _ => "-".into(),
    };
    println!("{short:<22} {step_label:<10} | {is:>9} | {ls:>9} | {ratio:>10}");

    let e2e_label = "dg-e2e".to_string();
    match (&infr, &llama) {
        (Ok(i), Some(l)) => println!(
            "{short:<22} {e2e_label:<10} | {:>6.1}t/s@{:.0}st | {:>6.1}t/s@{:.0}st | informational (steps differ)",
            i.gen_ts, i.steps, l.e2e_ts, l.steps
        ),
        _ => println!(
            "{short:<22} {e2e_label:<10} | {:>9} | {:>9} | {:>10}",
            "-", "-", "-"
        ),
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

    // diffusion-gemma (Phase 4/E): entirely different measurement shape (no upstream llama-bench
    // support for this arch — see `ModelBench::is_dg`'s doc comment), so it skips every AR
    // scenario below (CONTEXT LOAD / REPLY @depth / MTP / SHORT TURN / TURN all assume a
    // prefill-then-decode AR turn shape DG doesn't have) and returns after its own two rows.
    if mb.is_dg {
        hdr("DIFFUSION-GEMMA DECODE"); // entropy-bound block-diffusion decode, fixed -n 256
        let (infr, llama) = dg_compare_measure(&mb);
        let infr_steps = infr.as_ref().ok().map(|r| r.steps);
        let infr_e2e = infr.as_ref().ok().map(|r| r.gen_ts);
        row(
            "dg-step".to_string(),
            infr.map(|r| r.parallel_ts),
            llama.as_ref().map(|r| r.step_ts),
        );
        let fmt_st = |ts: Option<f64>, st: Option<f64>| match (ts, st) {
            (Some(t), Some(s)) => format!("{t:.1} t/s @ {s:.0} steps"),
            _ => "-".into(),
        };
        println!(
            "  dg-e2e (informational — step counts differ, not apples-to-apples): infr {}   llama {}",
            fmt_st(infr_e2e, infr_steps),
            fmt_st(llama.as_ref().map(|r| r.e2e_ts), llama.as_ref().map(|r| r.steps)),
        );
        return Ok(());
    }

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

    // MTP DECODE (issue #33, phase 4 — perf-bottleneck visibility for the self-speculative decode
    // path, currently NET NEGATIVE at 36 t/s vs a ~100 t/s baseline): four numbers — infr/llama at
    // baseline tg, infr/llama with MTP on — plus each tool's own mtp/base speedup, so the gap AND
    // its shape (is infr's speculation slower to begin with, or is its baseline just slower) are
    // both visible in one section. Absent entirely for a GGUF with no `nextn.*` head.
    if mb.has_mtp_head() {
        hdr("MTP DECODE"); // self-speculative decode: baseline tg vs MTP-spec tg, both tools
        const N: usize = 128;
        let ns = N.to_string();
        let infr_base = infr_b(&["-p", "0", "-n", &ns]);
        let infr_mtp = mb.infr_mtp(&["-p", "0", "-n", &ns]);
        let llama_base = llama_b(0, N, &["-p", "0", "-n", &ns]);
        let llama_mtp = mb.llama_cli_mtp(N);
        // Each tool's own speedup (mtp/base) — NOT an infr-vs-llama ratio, so computed here rather
        // than through `row`'s ratio column (which is always infr/llama at one shape).
        let infr_speedup = match (infr_base.as_ref(), infr_mtp.as_ref()) {
            (Ok(&b), Ok(&m)) if b > 0.0 => Some(m / b),
            _ => None,
        };
        let llama_speedup = match (llama_base, llama_mtp) {
            (Some(b), Some(m)) if b > 0.0 => Some(m / b),
            _ => None,
        };
        row(format!("tg{N}"), infr_base, llama_base); // infr-vs-llama @ baseline
        row(format!("mtp{N}"), infr_mtp, llama_mtp); // infr-vs-llama @ MTP
        let fmt_x = |v: Option<f64>| v.map(|v| format!("{v:.2}x")).unwrap_or_else(|| "-".into());
        println!(
            "  own-tool speedup (mtp/base): infr {}   llama {}",
            fmt_x(infr_speedup),
            fmt_x(llama_speedup),
        );
    }

    // SHORT TURN — the tiny suffix prefill (m = 2..8 Linears): what a brief follow-up costs in a
    // warm multi-turn session (KV prefix reused, only the new tokens forward). This is the
    // multi-row-GEMV / spec-verify shape; 16 sits past the mrow gate (the first GEMM point) as
    // the cliff-edge reference. Cold (@0) isolates the kernels; @depth adds the attention term.
    hdr("SHORT TURN"); // tiny suffix prefill on a warm session
    for &n in &[2usize, 4, 8, 16] {
        let np = n.to_string();
        row(
            format!("pp{n}"),
            infr_b(&["-p", &np, "-n", "0"]),
            llama_b(n, 0, &["-p", &np, "-n", "0"]),
        );
        for &d in ctx {
            let ds = d.to_string();
            row(
                format!("pp{n}@{d}"),
                infr_b(&["-p", &np, "-n", "0", "-d", &ds]),
                llama_b(n, 0, &["-p", &np, "-n", "0", "-d", &ds]),
            );
        }
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

fn cmd_serve(model: &str, addr: &str, parallel: usize, ctx: Option<&str>) -> anyhow::Result<()> {
    let (gguf, tok) = resolve(model)?;
    let model_id = gguf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let sockaddr: std::net::SocketAddr = addr.parse().context("invalid --addr")?;
    let parallel = parallel.max(1);

    // `--ctx` is the per-slot context. It shares the size grammar (`8192`, `256k`, `50%`) with
    // INFR_CTX, which it overrides — one grammar, one meaning (`infr_core::parse_size`).
    if let Some(c) = ctx {
        if infr_core::parse_size(c).is_none() {
            anyhow::bail!("invalid --ctx `{c}` (expected e.g. 8192, 256k, or 50%)");
        }
        std::env::set_var("INFR_CTX", c);
    }

    let is_dg = infr_llama::diffusion::is_diffusion_gemma(&gguf);
    let is_vulkan =
        !is_dg && std::env::var("INFR_METAL").is_err() && std::env::var("INFR_CPU").is_err();

    set_default_sampling_env();

    // ── the CONCURRENT path: dense/MoE/qwen35 on the Vulkan seam ────────────────────────────────
    // N KV slots off ONE weight upload, round-robin on the GPU at token granularity. This is the
    // default `infr serve` engine.
    if is_vulkan {
        let loaded = infr_llama::SeamModel::load(&gguf, tok.as_deref())?;
        // `--ctx` (or INFR_CTX) is the PER-SLOT window: an explicit token count is used verbatim,
        // a `%` is a fraction of the whole device's KV capacity split across the slots, and unset
        // derives it from the VRAM fit divided by N. See `SeamModel::vulkan_slot_ctx`.
        let want_ctx = std::env::var("INFR_CTX")
            .ok()
            .as_deref()
            .and_then(infr_core::parse_size);
        let t0 = std::time::Instant::now();
        let engine = infr_llama::parallel::ParallelSeam::new(loaded, parallel, want_ctx)?;
        eprintln!(
            "warmup: pipelines compiled in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        let n_slots = engine.n_slots();
        let max_ctx = engine.max_ctx();
        let generator: std::sync::Arc<dyn infr_server::ChatGenerator> =
            std::sync::Arc::new(ParallelGenerator::new(&gguf, engine)?);
        let rt = tokio::runtime::Runtime::new()?;
        println!(
            "infr serve: {model_id} on http://{sockaddr}  (OpenAI /v1, {n_slots} slot{} x {max_ctx} ctx)",
            if n_slots == 1 { "" } else { "s" },
        );
        return rt.block_on(infr_server::serve(generator, model_id, sockaddr, n_slots));
    }

    // ── the SERIALISED path: CPU / Metal / diffusion-gemma ──────────────────────────────────────
    // These have no multi-slot engine (one `&mut` ChatModel, one KV session), so concurrent
    // requests would only queue behind a Mutex. Say so rather than pretending to parallelise.
    if parallel > 1 {
        eprintln!(
            "note: --parallel {parallel} ignored on the CPU/Metal/diffusion backends (no \
             multi-slot engine); serving 1 request at a time. The Vulkan seam is the concurrent \
             engine."
        );
    }

    // Seam-backed serve — the ONE engine: the SAME ChatModel + multi-slot session `infr run`
    // uses, so serve gets per-request suffix-only prefill and cross-conversation prefix seeding
    // for free. INFR_CPU / INFR_METAL select the reference backends; Vulkan is the default.
    // qwen35 shares the SAME selection funnel as every other arch below — see the matching
    // comment in `cmd_run` (the old standalone `Qwen35Chat` branch + its env-gated escape hatch
    // were deleted once the unified path was validated; issue #30).
    let mut m: Box<dyn infr_llama::chat::ChatModel + Send> = if is_dg {
        // diffusion-gemma (Phase 3/D): same selection as `cmd_run` — see its matching comment.
        let cpu = std::env::var("INFR_CPU").is_ok();
        let metal = std::env::var("INFR_METAL").is_ok();
        let loaded = infr_llama::SeamModel::load(&gguf, tok.as_deref())?;
        Box::new(if cpu {
            infr_llama::chat::DiffusionGemmaChat::new_cpu(loaded)
        } else if metal {
            infr_llama::chat::DiffusionGemmaChat::new_metal(loaded)
        } else {
            infr_llama::chat::DiffusionGemmaChat::new(loaded)
        })
    } else if std::env::var("INFR_METAL").is_ok() {
        // Metal: the SAME selection funnel as `infr run` — persistent-session seam chat, or
        // speculative decoding with INFR_SPEC_DRAFT (serve requests then decode through the
        // draft-verify driver; greedy-only). Stateless reference wrapper elsewhere (the arm is
        // unreachable off-macOS anyway).
        #[cfg(target_os = "macos")]
        {
            metal_chat_model(&gguf, tok.as_deref())?
        }
        #[cfg(not(target_os = "macos"))]
        {
            Box::new(infr_llama::chat::CpuDenseChat::new_metal(
                infr_llama::SeamModel::load(&gguf, tok.as_deref())?,
            ))
        }
    } else if std::env::var("INFR_CPU").is_ok() {
        Box::new(infr_llama::chat::CpuDenseChat::new(
            infr_llama::SeamModel::load(&gguf, tok.as_deref())?,
        ))
    } else {
        Box::new(infr_llama::chat::DenseSeamChat::new(
            infr_llama::SeamModel::load(&gguf, tok.as_deref())?,
        ))
    };
    {
        // Compile every lazily-built pipeline NOW (a tiny throwaway generation) so the first
        // request doesn't pay seconds of pipeline builds on top of its own prefill.
        let t0 = std::time::Instant::now();
        m.warmup()?;
        eprintln!(
            "warmup: pipelines compiled in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        let generator: std::sync::Arc<dyn infr_server::ChatGenerator> =
            std::sync::Arc::new(SeamGenerator::new(&gguf, m)?);
        let rt = tokio::runtime::Runtime::new()?;
        println!("infr serve: {model_id} on http://{sockaddr}  (OpenAI /v1, agnostic seam)");
        rt.block_on(infr_server::serve(generator, model_id, sockaddr, 1))
    }
}
