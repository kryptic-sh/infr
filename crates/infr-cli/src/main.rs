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

fn cmd_run(model: &str, message: Option<&str>) -> anyhow::Result<()> {
    use std::io::Write;
    const MAX_CTX: usize = 8192;
    const MAX_NEW: usize = 256;
    let (gguf, tok) = resolve(model)?;
    let llama = infr_llama::Llama::load_opt(&gguf, tok.as_deref())?;

    // One-shot message: single fresh generation, no persisted context.
    if let Some(m) = message {
        let t0 = std::time::Instant::now();
        let mut n = 0usize;
        llama.generate(&llama.chatml(m), MAX_NEW, |piece| {
            print!("{piece}");
            let _ = std::io::stdout().flush();
            n += 1;
        })?;
        let dt = t0.elapsed().as_secs_f32();
        println!();
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
        match session.turn(line, MAX_NEW, |piece| {
            print!("{piece}");
            let _ = std::io::stdout().flush();
            n += 1;
        }) {
            Ok(_) => {
                let dt = t0.elapsed().as_secs_f32();
                println!();
                eprintln!(
                    "[{n} tokens, {dt:.2}s, {:.1} tok/s | ctx {}/{}]",
                    n as f32 / dt,
                    session.ctx_len(),
                    session.max_ctx()
                );
            }
            Err(e) => {
                println!();
                eprintln!("error: {e}");
            }
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
