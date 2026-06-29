//! Decode seam + diffusion strategy. Turns compiled forward passes into tokens.
//!
//! Reference: `~/Projects/llama.cpp/examples/diffusion/diffusion.{cpp,h}`
//! (`diffusion_generate_entropy_bound`). See docs/PLAN.md "decode".
#![allow(dead_code, unused_variables)]

use infr_core::error::Result;

/// Which channel a streamed token belongs to (DiffusionGemma emits a thought channel
/// before the final answer; see PLAN "DiffusionGemma spec").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Reasoning,
    Answer,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub id: u32,
    pub text: String,
    pub channel: Channel,
}

/// Tunables for the entropy-bound diffusion decoder (defaults observed from the model).
#[derive(Clone, Debug)]
pub struct DiffusionParams {
    pub canvas_len: usize,
    pub max_steps: u32,
    pub temperature: f32,
    pub t_min: f32,
    pub t_max: f32,
    pub entropy_bound: f32,
    pub confidence: f32,
    pub mask_token: u32,
}

impl Default for DiffusionParams {
    fn default() -> Self {
        Self {
            canvas_len: 256,
            max_steps: 48,
            temperature: 0.8,
            t_min: 0.4,
            t_max: 0.8,
            entropy_bound: 0.1,
            confidence: 0.005,
            mask_token: 4,
        }
    }
}

/// Strategy seam: diffusion now, autoregressive later — both produce `Token`s.
pub trait DecodeStrategy {
    /// Generate from a tokenized prompt, streaming tokens to `on_token`.
    fn generate(&mut self, prompt: &[u32], on_token: &mut dyn FnMut(Token)) -> Result<()>;
}

/// Block (canvas) diffusion decoder.
///
/// TODO(sonnet): hold the backend + compiled model plan + KV state + params. Port the
/// entropy-bound denoise loop: init masked canvas, run N steps of forward+sample with
/// self-conditioning, early-stop on the entropy bound, commit blocks.
pub struct DiffusionDecoder {
    pub params: DiffusionParams,
}

impl DiffusionDecoder {
    pub fn new(params: DiffusionParams) -> Self {
        Self { params }
    }
}

impl DecodeStrategy for DiffusionDecoder {
    fn generate(&mut self, prompt: &[u32], on_token: &mut dyn FnMut(Token)) -> Result<()> {
        todo!("port the entropy-bound diffusion decode loop")
    }
}
