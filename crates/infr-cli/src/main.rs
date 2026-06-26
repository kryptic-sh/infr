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

fn cmd_pull(model: &str) -> anyhow::Result<()> {
    // TODO(sonnet): ModelRef::parse -> infr_hub::ensure -> print resolved path.
    todo!("infr pull")
}

fn cmd_run(model: &str, message: Option<&str>) -> anyhow::Result<()> {
    // TODO(sonnet): ensure model -> VulkanBackend::new -> Engine::load -> REPL loop calling
    // engine.chat, printing reasoning dimmed + content normally.
    todo!("infr run")
}

fn cmd_serve(model: &str, addr: &str) -> anyhow::Result<()> {
    // TODO(sonnet): ensure model -> VulkanBackend::new -> Engine::load -> build a tokio
    // runtime -> infr_server::serve(engine, addr).
    todo!("infr serve")
}
