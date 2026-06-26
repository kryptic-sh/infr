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

fn cmd_run(model: &str, message: Option<&str>) -> anyhow::Result<()> {
    use std::io::Write;
    const MAX_CTX: usize = 8192;
    const MAX_NEW: usize = 512;
    let (gguf, tok) = resolve(model)?;
    let llama = infr_llama::Llama::load_opt(&gguf, tok.as_deref())?;

    // One-shot message: single fresh generation, no persisted context.
    if let Some(m) = message {
        let t0 = std::time::Instant::now();
        let mut n = 0usize;
        let mut render = ThinkRender::new();
        llama.generate(&llama.chatml(m), MAX_NEW, |piece| {
            n += 1;
            render.feed(piece);
        })?;
        render.finish();
        let dt = t0.elapsed().as_secs_f32();
        eprintln!("[{n} tokens, {dt:.2}s, {:.1} tok/s]", n as f32 / dt);
        return Ok(());
    }

    // REPL: a persistent chat session keeps prior turns in the KV cache (multi-turn context).
    let mut session = llama.chat_session(MAX_CTX)?;
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
        let t0 = std::time::Instant::now();
        let mut n = 0usize;
        let mut render = ThinkRender::new();
        let res = session.turn(line, MAX_NEW, |piece| {
            n += 1;
            render.feed(piece);
        });
        render.finish();
        match res {
            Ok(_) => {
                let dt = t0.elapsed().as_secs_f32();
                eprintln!(
                    "[{n} tokens, {dt:.2}s, {:.1} tok/s | ctx {}/{}]",
                    n as f32 / dt,
                    session.ctx_len(),
                    session.max_ctx()
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
