//! Phase 3: the entropy-bound block-diffusion decode loop (backend-agnostic — drives either
//! [`crate::seam::model::DiffusionGemmaCpuSession`] or its Vulkan twin through the small
//! [`DiffusionSession`] trait). Ports `diffusion_generate_entropy_bound` + `run_turn`'s block loop
//! from the oracle reference (`~/Projects/mxaddict/llama.cpp-dg/examples/diffusion/diffusion.cpp`
//! and `diffusion-cli.cpp`) — see `docs/DIFFUSIONGEMMA.md`'s "Decode loop" section. Line refs in
//! comments below point at `diffusion.cpp`'s `diffusion_generate_entropy_bound` (the sampler) and
//! `diffusion-cli.cpp`'s `run_turn` (the block/commit/trim loop), both read 2026-07-05.
//!
//! Not bit-for-bit RNG-identical to the reference's `std::mt19937` (a house xorshift64 stands in —
//! see [`Rng`]): the design doc's validation ladder explicitly does NOT require token-identical
//! output (a 128-expert top-8 MoE model's CPU/Vulkan routing already diverges legitimately), only
//! the SAME schedule/acceptance/stop semantics under a fixed seed.

use crate::seam::model::{DiffusionGemmaCpuSession, DiffusionGemmaVulkanSession, SeamModel};
use crate::{Config, GenStats};
use anyhow::Result;
use rayon::prelude::*;

/// The two DiffusionGemma sessions' shared shape (Phase 2, `seam/model.rs`): causal prefill of the
/// committed prefix, then a canvas denoise forward. One decode loop below drives either backend.
pub trait DiffusionSession {
    fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()>;
    fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
    ) -> Result<Vec<f32>>;
}

impl DiffusionSession for DiffusionGemmaCpuSession {
    fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()> {
        DiffusionGemmaCpuSession::prefill(self, model, tokens)
    }
    fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
    ) -> Result<Vec<f32>> {
        DiffusionGemmaCpuSession::denoise(self, model, canvas_tokens, sc_logits, temp_inv)
    }
}

impl DiffusionSession for DiffusionGemmaVulkanSession {
    fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()> {
        DiffusionGemmaVulkanSession::prefill(self, model, tokens)
    }
    fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
    ) -> Result<Vec<f32>> {
        DiffusionGemmaVulkanSession::denoise(self, model, canvas_tokens, sc_logits, temp_inv)
    }
}

#[cfg(target_os = "macos")]
impl DiffusionSession for crate::seam::model::DiffusionGemmaMetalSession {
    fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()> {
        crate::seam::model::DiffusionGemmaMetalSession::prefill(self, model, tokens)
    }
    fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
    ) -> Result<Vec<f32>> {
        crate::seam::model::DiffusionGemmaMetalSession::denoise(
            self,
            model,
            canvas_tokens,
            sc_logits,
            temp_inv,
        )
    }
}

/// The entropy-bound sampler's tunables (`diffusion.eb_*` GGUF metadata — see `Config`'s fields
/// and `diffusion-cli.cpp`'s `meta_f`/`meta_i` fallbacks).
#[derive(Clone, Copy, Debug)]
pub struct EbConfig {
    pub max_steps: usize,
    pub t_min: f32,
    pub t_max: f32,
    pub entropy_bound: f32,
    pub stability_threshold: usize,
    pub confidence_threshold: f32,
}

impl EbConfig {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            max_steps: cfg.eb_max_steps,
            t_min: cfg.eb_t_min,
            t_max: cfg.eb_t_max,
            entropy_bound: cfg.eb_entropy_bound,
            stability_threshold: cfg.eb_stability_threshold,
            confidence_threshold: cfg.eb_confidence_threshold,
        }
    }
}

/// A whole-turn diffusion generation: the committed response tokens (prompt excluded, already
/// trimmed) plus the counts the CLI/tests report (steps/blocks — `diffusion-cli.cpp`'s
/// `cb_data.steps_seen`/`blocks_seen`).
pub struct DiffusionGenResult {
    pub tokens: Vec<u32>,
    pub stats: GenStats,
    pub steps: usize,
    pub blocks: usize,
}

/// One denoise step's observable state, handed to the optional `on_step` hook in
/// [`diffusion_generate`] — everything a live TTY renderer (`INFR_DIFFUSION_VISUAL`, `infr-cli`'s
/// `cmd_run`) needs to redraw the canvas without reaching into the sampler's internals. Purely
/// additive: the hook is `Option`, and `None` (every existing caller) skips construction of this
/// struct entirely — no behavior or timing change on the hot path.
pub struct StepView<'a> {
    /// 0-based index of the block currently denoising (`diffusion_generate`'s `b`).
    pub block: usize,
    /// 0-based step index WITHIN this block (how many denoise steps have run so far, this one
    /// included).
    pub step: usize,
    /// This block's step budget (`eb.max_steps`) — lets a renderer show "step N/max".
    pub max_steps: usize,
    /// This block's current argmax canvas (`diffusion.cpp:658`'s `argmax_canvas`) — the same
    /// observable output the reference visualizer draws every step, regardless of acceptance.
    pub canvas: &'a [u32],
    /// Per-canvas-position: `true` once this step accepted (committed) the position's low-entropy
    /// sample; `false` means the position is still being renoised each step — this sampler has no
    /// literal mask token (see the module doc), so "accepted" stands in for "decided" and
    /// "not yet accepted" for "still masked/undecided".
    pub accepted: &'a [bool],
    /// Response tokens already committed BEFORE this block started (`response.len()` at the top
    /// of the block loop) — the prompt-relative position of this block's canvas, so a renderer can
    /// place it after prior committed text without re-deriving it from `prefix`/`prompt_tokens`.
    pub committed_before: usize,
}

/// Deterministic xorshift64 PRNG (seeded per block — see [`diffusion_generate`]'s doc). Stands in
/// for the reference's `std::mt19937`: same *role* (canvas init, per-step multinomial draw,
/// renoise), not bit-identical output — see this module's doc comment.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1) // xorshift64 never leaves the zero state, but also never enters it from a
                       // nonzero seed — the |1 just guards a literal seed=0
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform draw in `[0, 1)` (`diffusion.cpp:468`'s `uni01`).
    fn next_f32_01(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform token id in `[0, vocab)` (`diffusion.cpp:469`'s `vocab_dist`).
    fn next_token(&mut self, vocab: usize) -> u32 {
        (self.next_u64() % vocab as u64) as u32
    }
}

/// Reborrow the per-block `on_step` hook with a fresh, shorter lifetime for one `denoise_block`
/// call. A bare `on_step.as_deref_mut()` inside `diffusion_generate`'s per-block `for` loop trips
/// E0499 (no Polonius yet): each iteration's reborrow of the `&mut dyn FnMut` needs to be shorter
/// than the OUTER lifetime the function signature ties it to, which a method call inline can't
/// express — routing through this free function gives the reborrow its own elided lifetime.
fn reborrow_step_hook<'b>(
    hook: &'b mut Option<&mut dyn FnMut(StepView)>,
) -> Option<&'b mut dyn FnMut(StepView)> {
    match hook {
        Some(cb) => Some(&mut **cb),
        None => None,
    }
}

/// ONE block's entropy-bound denoise (`diffusion_generate_entropy_bound`, `diffusion.cpp:442-683`)
/// against a session already `prefill`ed with the committed prefix. Returns the block's argmax
/// canvas (the observable output — `diffusion.cpp:658` writes `argmax_canvas` to `output_tokens`
/// every step regardless of acceptance) and the number of steps actually run (early stop —
/// `diffusion.cpp:665`).
#[allow(clippy::too_many_arguments)]
fn denoise_block(
    session: &mut impl DiffusionSession,
    model: &SeamModel,
    canvas_len: usize,
    vocab: usize,
    eb: &EbConfig,
    rng: &mut Rng,
    block: usize,
    committed_before: usize,
    mut on_step: Option<&mut dyn FnMut(StepView)>,
) -> Result<(Vec<u32>, usize)> {
    let c = canvas_len;
    let s = eb.max_steps.max(1);

    // Random-initialized (NOT mask-token) working canvas — `diffusion.cpp:471-474`. Renoised
    // (non-accepted) positions get a fresh random token each step, never the mask id: this
    // sampler has no notion of "still masked", unlike the origin/LLaDA path.
    let mut current_canvas: Vec<u32> = (0..c).map(|_| rng.next_token(vocab)).collect();
    // Previous step's raw canvas logits for self-conditioning; `None` on step 0 (`sc_use=0`,
    // `diffusion.cpp:559`'s `step_idx == 0 ? 0.0f : 1.0f` gate) — the session's `denoise` already
    // treats `None` as that gate (see `DenoiseReq`'s doc), so we don't track `step_idx` separately.
    let mut sc_buffer: Option<Vec<f32>> = None;
    let mut argmax_canvas = vec![0u32; c];
    let mut prev_argmax: Option<Vec<u32>> = None; // None == "step 0, never stable" (-1 sentinel upstream)
    let mut entropy = vec![0f32; c];
    let mut denoiser = vec![0u32; c];

    // `prev_temp_inv` starts at 1.0 (`diffusion.cpp:525`) and is what SC divides by — the
    // PREVIOUS step's temperature, not the current one (`diffusion.cpp:558-559`).
    let mut prev_temp_inv = 1.0f32;
    let mut held = 0usize;
    let mut steps_run = 0usize;
    // Diagnostic (INFR_EB_TRACE=1): per-step schedule/entropy trace, kept permanently since it's
    // small/clean and env-gated (one std::env::var read per BLOCK, not per step — zero cost when
    // unset). Grew out of chasing a convergence-speed gap between this sampler and the fork's
    // (root cause turned out to be upstream of this loop — the compare/bench harness feeding an
    // untemplated prompt, see `dg_bench_run`'s `prompt_ids` comment — not the sampler itself, which
    // this trace helped confirm already matches `diffusion.cpp` step-for-step). Useful again next
    // time infr's/the fork's entropy trajectories need diffing.
    let eb_trace = std::env::var_os("INFR_EB_TRACE").is_some();

    // cur_step: S downto 1 (`diffusion.cpp:530`); step_idx = S - cur_step is the 0-based step.
    for cur_step in (1..=s).rev() {
        let t = eb.t_min + (eb.t_max - eb.t_min) * (cur_step as f32 / s as f32); // line 532
        let temp_inv = 1.0 / t;

        let logits =
            session.denoise(model, &current_canvas, sc_buffer.as_deref(), prev_temp_inv)?;
        steps_run += 1;

        // Pre-draw the step's randomness single-threaded BEFORE the parallel reduction, so the
        // result doesn't depend on thread scheduling (`diffusion.cpp:576-580`).
        let mut u = vec![0f32; c];
        let mut renoise = vec![0u32; c];
        for pos in 0..c {
            u[pos] = rng.next_f32_01();
            renoise[pos] = rng.next_token(vocab);
        }

        // Per-position argmax / entropy(softmax(raw*temp_inv)) / one multinomial draw from that
        // softmax using the pre-drawn `u[pos]` (`diffusion.cpp:583-612`'s `worker`).
        //
        // Perf (profiled via samply: this loop's `exp`/`ln` calls — glibc's correctly-rounded
        // expf/logf helper, `f32subf64x` — were >25% of ALL sampled thread-time on a 256-row
        // canvas × 262144-vocab step, dwarfing every GPU kernel's share; see docs/PERF.md's class-5
        // "host-in-the-loop" entry). The original computed `exp(raw*temp_inv - m)` TWICE per vocab
        // element (once to accumulate `z_sum`, again — bit-for-bit the same value, since `exp` is a
        // pure function of its input bits — to get `e` for the entropy/cumsum pass). Caching that
        // first `exp` in a per-thread scratch buffer (`map_init`, reused across this worker's
        // positions instead of a fresh per-position `Vec`) drops the loop from 2 exp passes + 1 ln
        // pass to 1 exp pass + 1 ln pass over the row — same values, same order, bit-identical
        // output, ~1/3 fewer transcendental calls and one fewer full 1 MB/row traversal.
        let per_pos: Vec<(u32, f32, u32)> = (0..c)
            .into_par_iter()
            .map_init(
                || vec![0f32; vocab],
                |escratch, pos| {
                    let row = &logits[pos * vocab..(pos + 1) * vocab];
                    let mut m = f32::NEG_INFINITY;
                    let mut amax = 0u32;
                    for (v, &raw) in row.iter().enumerate() {
                        let z = raw * temp_inv;
                        if z > m {
                            m = z;
                            amax = v as u32;
                        }
                    }
                    let mut z_sum = 0f32;
                    for (v, &raw) in row.iter().enumerate() {
                        let e = (raw * temp_inv - m).exp();
                        escratch[v] = e;
                        z_sum += e;
                    }
                    let target = u[pos] * z_sum;
                    let mut cum = 0f32;
                    let mut h = 0f32;
                    let mut sampled = (vocab - 1) as u32;
                    let mut picked = false;
                    for (v, &e) in escratch.iter().enumerate() {
                        let p = e / z_sum;
                        if p > 0.0 {
                            h -= p * p.ln();
                        }
                        cum += e;
                        if !picked && cum >= target {
                            sampled = v as u32;
                            picked = true;
                        }
                    }
                    (amax, h, sampled)
                },
            )
            .collect();
        for (pos, &(amax, h, sampled)) in per_pos.iter().enumerate() {
            argmax_canvas[pos] = amax;
            entropy[pos] = h;
            denoiser[pos] = sampled;
        }

        // Accept the lowest-entropy positions whose STRICTLY-EARLIER cumulative entropy stays
        // within the MI bound (`diffusion.cpp:644-652`).
        let mut order: Vec<usize> = (0..c).collect();
        order.sort_by(|&a, &b| entropy[a].partial_cmp(&entropy[b]).unwrap());
        let mut accepted = vec![false; c];
        let mut cum_e = 0f64;
        for &pos in &order {
            cum_e += entropy[pos] as f64;
            if cum_e - entropy[pos] as f64 <= eb.entropy_bound as f64 {
                accepted[pos] = true;
            }
        }

        // Renoise: accepted -> sampled token, rest -> fresh random; the OUTPUT canvas is always
        // the argmax, whether or not a position was accepted this step (`diffusion.cpp:654-660`).
        let mut entropy_sum = 0f32;
        for pos in 0..c {
            current_canvas[pos] = if accepted[pos] {
                denoiser[pos]
            } else {
                renoise[pos]
            };
            entropy_sum += entropy[pos];
        }
        sc_buffer = Some(logits); // this step's raw logits self-condition the next

        if let Some(cb) = on_step.as_deref_mut() {
            cb(StepView {
                block,
                step: steps_run - 1,
                max_steps: s,
                canvas: &argmax_canvas,
                accepted: &accepted,
                committed_before,
            });
        }

        // Adaptive stop: argmax stable for `stability_threshold` steps AND confident (mean
        // entropy below the bound) — `diffusion.cpp:662-667`.
        let stable = prev_argmax.as_deref() == Some(argmax_canvas.as_slice());
        held = if stable { held + 1 } else { 0 };
        let mean_entropy = entropy_sum / c as f32;
        let confident = mean_entropy < eb.confidence_threshold;
        prev_argmax = Some(argmax_canvas.clone());
        prev_temp_inv = temp_inv;
        if eb_trace {
            let n_accepted = accepted.iter().filter(|&&a| a).count();
            eprintln!(
                "[eb_trace] block={block} step={} temp_inv={temp_inv:.6} mean_entropy={mean_entropy:.6} accepted={n_accepted}/{c} held={held} stable={stable} confident={confident}",
                steps_run - 1,
            );
        }
        if held >= eb.stability_threshold && confident {
            break;
        }
    }

    Ok((argmax_canvas, steps_run))
}

/// Cut a denoised canvas at the first end-of-generation token, or (many checkpoints emit no stop
/// token) at the onset of a repetition loop — a token recurring at stride 1-2 for >= 6 reps
/// (`diffusion-cli.cpp:388-411`'s `trim_canvas`). A cut shorter than the canvas ends the turn.
fn trim_canvas(canvas: &[u32], is_eog: impl Fn(u32) -> bool) -> usize {
    let n = canvas.len();
    let mut cut = n;
    for (i, &tok) in canvas.iter().enumerate() {
        if is_eog(tok) {
            cut = i;
            break;
        }
    }
    for i in 0..cut.saturating_sub(1) {
        let mut looped = false;
        for stride in 1..=2usize {
            if looped {
                break;
            }
            let mut reps = 0u32;
            let mut j = i;
            while j + stride < n && canvas[j] == canvas[j + stride] {
                reps += 1;
                j += stride;
            }
            looped = reps >= 6;
        }
        if looped {
            cut = i;
            break;
        }
    }
    cut
}

/// The block-diffusion decode loop (`diffusion-cli.cpp:417-485`'s `run_turn`, canvas branch):
/// prefill the committed prefix, denoise a block, trim it, commit (causal re-prefill next
/// iteration treats it as prompt), repeat until an end token/repetition loop, the block budget
/// (`ceil(n_predict / canvas_len)`), or the session's KV cache runs out of room. The RNG reseeds
/// to `seed` at the start of EVERY block (matching the reference: `diffusion_generate_entropy_bound`
/// constructs a fresh `std::mt19937(params.seed)` on each call, and `run_turn` calls it once per
/// block — `diffusion.cpp:467`).
#[allow(clippy::too_many_arguments)]
pub fn diffusion_generate(
    session: &mut impl DiffusionSession,
    model: &SeamModel,
    prompt_tokens: &[u32],
    canvas_len: usize,
    vocab: usize,
    eos_ids: &[u32],
    eb: &EbConfig,
    n_predict: usize,
    seed: u64,
    max_ctx: usize,
    mut on_step: Option<&mut dyn FnMut(StepView)>,
) -> Result<DiffusionGenResult> {
    let blocks_wanted = n_predict.div_ceil(canvas_len.max(1)).max(1);
    let mut prefix: Vec<u32> = prompt_tokens.to_vec();
    let mut response: Vec<u32> = Vec::new();
    let mut steps_total = 0usize;
    let mut blocks_run = 0usize;
    let mut prompt_secs = 0f64;
    let mut decode_secs = 0f64; // every block's prefill + denoise AFTER the first prompt ingest

    for b in 0..blocks_wanted {
        let max_length = prefix.len() + canvas_len;
        if max_length > max_ctx {
            if b == 0 {
                anyhow::bail!(
                    "diffusion-gemma needs the whole [prompt | canvas] to fit the session's KV \
                     cache: prefix {} + canvas {canvas_len} = {max_length} > max_ctx {max_ctx}",
                    prefix.len()
                );
            }
            break; // out of KV room: stop here, keep what already generated (line 447-455)
        }

        let t_pf = std::time::Instant::now();
        session.prefill(model, &prefix)?;
        if b == 0 {
            prompt_secs = t_pf.elapsed().as_secs_f64();
        } else {
            decode_secs += t_pf.elapsed().as_secs_f64();
        }

        // Reseed every block (see this fn's doc) — `run_turn`'s per-block sampler call.
        let mut rng = Rng::new(seed);
        let t_dn = std::time::Instant::now();
        let (canvas, steps) = denoise_block(
            session,
            model,
            canvas_len,
            vocab,
            eb,
            &mut rng,
            b,
            response.len(),
            reborrow_step_hook(&mut on_step),
        )?;
        decode_secs += t_dn.elapsed().as_secs_f64();
        steps_total += steps;
        blocks_run += 1;

        let cut = trim_canvas(&canvas, |t| eos_ids.contains(&t));
        response.extend_from_slice(&canvas[..cut]);
        if cut < canvas_len {
            break; // end token or repetition loop: answer complete (line 478-480)
        }
        prefix.extend_from_slice(&canvas[..cut]); // commit the block, denoise the next (line 481)
    }

    let n_gen = response.len();
    Ok(DiffusionGenResult {
        tokens: response,
        stats: GenStats {
            n_prompt: prompt_tokens.len(),
            prompt_secs,
            n_gen,
            decode_secs,
        },
        steps: steps_total,
        blocks: blocks_run,
    })
}

/// Cheap architecture peek (mirrors [`crate::qwen35::is_qwen35`]): open the GGUF and read
/// `general.architecture` without building a full [`SeamModel`] — lets `infr run`/`serve` pick the
/// diffusion decode loop (and its own default token budget) before paying a full model load.
pub fn is_diffusion_gemma(path: &std::path::Path) -> bool {
    use infr_core::WeightSource;
    infr_gguf::Gguf::open(path)
        .ok()
        .map(|g| g.metadata().str("general.architecture") == Some(crate::arch::DIFFUSION_GEMMA))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::trim_canvas;

    #[test]
    fn trim_canvas_cuts_at_eog() {
        let canvas = [1u32, 2, 3, 99, 4, 5];
        assert_eq!(trim_canvas(&canvas, |t| t == 99), 3);
    }

    #[test]
    fn trim_canvas_no_cut_keeps_whole_block() {
        let canvas = [1u32, 2, 3, 4, 5];
        assert_eq!(trim_canvas(&canvas, |t| t == 99), 5);
    }

    #[test]
    fn trim_canvas_cuts_at_stride1_repetition() {
        // Token 7 repeats 6+ times in a row from index 2 — the onset of the loop, not its end.
        let mut canvas = vec![1u32, 2];
        canvas.extend(std::iter::repeat_n(7u32, 8));
        assert_eq!(trim_canvas(&canvas, |t| t == 99), 2);
    }

    #[test]
    fn trim_canvas_cuts_at_stride2_repetition() {
        // a b a b a b a b a b a b... (period-2 loop) starting at index 0.
        let canvas: Vec<u32> = (0..16).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect();
        assert_eq!(trim_canvas(&canvas, |t| t == 99), 0);
    }

    #[test]
    fn trim_canvas_short_canvas_no_panic() {
        // cut.saturating_sub(1) must not underflow when cut is 0 or 1.
        assert_eq!(trim_canvas(&[], |t| t == 99), 0);
        assert_eq!(trim_canvas(&[5], |t| t == 5), 0);
    }
}
