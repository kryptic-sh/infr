//! Gemma 4 vision encoder (`clip.vision.projector_type = "gemma4v"`) — the `mmproj-*.gguf` sidecar
//! that turns an image into soft tokens the Gemma 4 text model reads in place of `<|image|>`.
//!
//! A CPU reference, written against llama.cpp's `tools/mtmd/models/gemma4v.cpp` +
//! `clip.cpp::build_vit` (the ground truth this was checked against), op for op:
//!
//! ```text
//! pixels  : RGB, resized (aspect kept) so both sides are multiples of patch*merge = 48 and the
//!           soft-token count lands in [min_tokens, max_tokens]; scaled to [-1, 1]
//! patches : 16x16 conv, stride 16, no bias  ==  a Linear over each flattened (c, ky, kx) patch
//!         + pos_x table[col] + pos_y table[row]          (learned 2-D position lookup)
//! 16 x    : h = rms(x)*ln1
//!           q,k,v = clip(Linear(clip(h)))                (Gemma4ClippableLinear, per tensor)
//!           q = rms_head(q)*q_norm ; k = rms_head(k)*k_norm ; v = rms_head(v)   (per 64-dim head)
//!           q,k : NEOX rope, dims [0,32) by col, [32,64) by row, theta 100
//!           a = softmax(q·kᵀ) v                          (scale 1, bidirectional)
//!           x += rms(clip(Linear(clip(a))))*attn_post_norm
//!           h = rms(x)*ln2
//!           x += rms(clip(Linear_down(gelu_tanh(clip(gate)) * clip(up))))*ffn_post_norm
//! pool    : 3x3 average over the patch grid, x sqrt(n_embd)
//! project : Linear_mm(rms(pooled))  -> [n_tokens, text n_embd], NOT scaled by sqrt(n_embd)
//! ```
//!
//! The output rows go straight into the text model's residual stream at the image's positions
//! (see `crate::media`); the text model's own `sqrt(n_embd)` embedding scale is NOT applied to
//! them, matching llama.cpp ("do not normalize weights for raw embeddings input").
//!
//! Weights stay f16 in the mmap and are widened to f32 one matrix at a time, right before its GEMM,
//! so the encoder adds ~10 MB of RSS rather than the 600 MB an eager f32 copy would.

use anyhow::{anyhow, bail, Context, Result};
use infr_core::{DType, WeightSource};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// llama.cpp `hparams.set_limit_image_tokens(70, 1120)` for gemma4v: the per-image soft-token
/// floor. The ceiling is configurable (`serve.image_max_tokens`); llama.cpp allows up to 1120 but
/// the CPU encoder's attention is quadratic in patches, and 280 is the Gemma 4 default budget.
pub const MIN_IMAGE_TOKENS: usize = 70;
pub const MAX_IMAGE_TOKENS: usize = 1120;

/// `Gemma4ClippableLinear`'s four scalars; `±inf` where the GGUF carries none.
#[derive(Clone, Copy, Debug)]
struct Clamp {
    in_lo: f32,
    in_hi: f32,
    out_lo: f32,
    out_hi: f32,
}

impl Clamp {
    const NONE: Clamp = Clamp {
        in_lo: f32::NEG_INFINITY,
        in_hi: f32::INFINITY,
        out_lo: f32::NEG_INFINITY,
        out_hi: f32::INFINITY,
    };
}

/// One projection: its f16/f32 weight `[out, in]` row-major (ggml `[in, out]`) plus its clamp.
struct Proj {
    name: String,
    n_in: usize,
    n_out: usize,
    clamp: Clamp,
}

struct Layer {
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    attn_post_norm: Vec<f32>,
    ffn_post_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    gate: Proj,
    up: Proj,
    down: Proj,
}

/// A loaded Gemma 4 vision tower. Thread-safe: [`encode`](Self::encode) takes `&self`.
pub struct Gemma4Vision {
    src: infr_gguf::Gguf,
    path: PathBuf,
    n_embd: usize,
    n_head: usize,
    patch: usize,
    merge: usize,
    eps: f32,
    rope_theta: f32,
    /// Text model width the projector maps to.
    pub n_embd_out: usize,
    patch_embd: Proj,
    /// `[2][n_pos][n_embd]`: x table then y table.
    pos_embd: Vec<f32>,
    n_pos: usize,
    layers: Vec<Layer>,
    mm_proj: Proj,
    std_bias: Option<Vec<f32>>,
    std_scale: Option<Vec<f32>>,
    /// Encoded images keyed by content hash — a chat client resends the same image on every turn
    /// (and on every tool round), and re-encoding it each time would cost seconds per request.
    cache: Mutex<Vec<CacheEntry>>,
}

const CACHE_IMAGES: usize = 16;

/// One cached encode: (content hash, token budget, rows).
type CacheEntry = (u64, usize, Arc<Vec<f32>>);

fn meta_f64(g: &infr_gguf::Gguf, key: &str) -> Option<f64> {
    match g.metadata().get(key)? {
        infr_core::MetaValue::F64(v) => Some(*v),
        infr_core::MetaValue::U64(v) => Some(*v as f64),
        infr_core::MetaValue::I64(v) => Some(*v as f64),
        _ => None,
    }
}

fn meta_usize(g: &infr_gguf::Gguf, key: &str) -> Result<usize> {
    g.metadata()
        .u64(key)
        .map(|v| v as usize)
        .ok_or_else(|| anyhow!("mmproj: missing metadata {key}"))
}

/// Any f32/f16/bf16 tensor as f32.
fn tensor_f32(g: &infr_gguf::Gguf, name: &str) -> Result<Vec<f32>> {
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("mmproj: missing tensor {name}"))?;
    let bytes = g
        .tensor_bytes(name)
        .map_err(|e| anyhow!("mmproj: {name}: {e}"))?;
    widen(info.dtype, bytes).with_context(|| format!("mmproj: {name}"))
}

fn widen(dtype: DType, bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        DType::F32 => bytemuck::pod_collect_to_vec::<u8, f32>(bytes),
        DType::F16 => bytes
            .par_chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        DType::Bf16 => bytes
            .par_chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        other => bail!("unsupported weight dtype {other:?} (want F32/F16/BF16)"),
    })
}

fn scalar(g: &infr_gguf::Gguf, name: &str) -> Option<f32> {
    tensor_f32(g, name).ok().and_then(|v| v.first().copied())
}

impl Gemma4Vision {
    /// Open an mmproj GGUF. Errors when it has no gemma4v vision tower.
    pub fn open(path: &Path) -> Result<Self> {
        let src =
            infr_gguf::Gguf::open(path).map_err(|e| anyhow!("mmproj {}: {e}", path.display()))?;
        let proj_type = src
            .metadata()
            .str("clip.vision.projector_type")
            .or_else(|| src.metadata().str("clip.projector_type"))
            .unwrap_or("");
        if proj_type != "gemma4v" {
            bail!(
                "mmproj {}: vision projector type {proj_type:?} is not supported (only gemma4v)",
                path.display()
            );
        }
        let n_embd = meta_usize(&src, "clip.vision.embedding_length")?;
        let n_head = meta_usize(&src, "clip.vision.attention.head_count")?;
        let n_layer = meta_usize(&src, "clip.vision.block_count")?;
        let patch = meta_usize(&src, "clip.vision.patch_size")?;
        let n_embd_out = meta_usize(&src, "clip.vision.projection_dim")?;
        let eps = meta_f64(&src, "clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6) as f32;
        // llama.cpp reads an optional KEY_PROJ_SCALE_FACTOR, defaulting to the pooling kernel 3.
        let merge = src
            .metadata()
            .u64("clip.vision.projector.scale_factor")
            .map(|v| v as usize)
            .unwrap_or(3);
        if !n_embd.is_multiple_of(n_head) || !(n_embd / n_head).is_multiple_of(4) {
            bail!("mmproj: head dim {n_embd}/{n_head} is not a multiple of 4");
        }
        // gemma4v's image_mean/std are 0/1 — anything else would need a normalize step we skip.
        for (key, want) in [
            ("clip.vision.image_mean", 0.0),
            ("clip.vision.image_std", 1.0),
        ] {
            if let Some(infr_core::MetaValue::Arr(a)) = src.metadata().get(key) {
                let ok = a.iter().all(|v| match v {
                    infr_core::MetaValue::F64(x) => (*x - want).abs() < 1e-6,
                    _ => false,
                });
                if !ok {
                    bail!("mmproj: {key} is not all {want} — unsupported preprocessing");
                }
            }
        }

        let proj = |name: &str| -> Result<Proj> {
            let w = format!("{name}.weight");
            let info = src
                .tensors()
                .iter()
                .find(|t| t.name == w)
                .ok_or_else(|| anyhow!("mmproj: missing tensor {w}"))?;
            let n_in = info.shape[0];
            let n_out = info.shape[1..].iter().product();
            let get =
                |suffix: &str, dflt: f32| scalar(&src, &format!("{name}.{suffix}")).unwrap_or(dflt);
            Ok(Proj {
                name: w,
                n_in,
                n_out,
                clamp: Clamp {
                    in_lo: get("input_min", f32::NEG_INFINITY),
                    in_hi: get("input_max", f32::INFINITY),
                    out_lo: get("output_min", f32::NEG_INFINITY),
                    out_hi: get("output_max", f32::INFINITY),
                },
            })
        };

        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let p = format!("v.blk.{l}");
            let t = |s: &str| tensor_f32(&src, &format!("{p}.{s}.weight"));
            layers.push(Layer {
                ln1: t("ln1")?,
                ln2: t("ln2")?,
                attn_post_norm: t("attn_post_norm")?,
                ffn_post_norm: t("ffn_post_norm")?,
                q_norm: t("attn_q_norm")?,
                k_norm: t("attn_k_norm")?,
                q: proj(&format!("{p}.attn_q"))?,
                k: proj(&format!("{p}.attn_k"))?,
                v: proj(&format!("{p}.attn_v"))?,
                o: proj(&format!("{p}.attn_out"))?,
                gate: proj(&format!("{p}.ffn_gate"))?,
                up: proj(&format!("{p}.ffn_up"))?,
                down: proj(&format!("{p}.ffn_down"))?,
            });
        }

        // `v.patch_embd.weight` is ggml [kw, kh, c, out] — i.e. `out` rows of (c, ky, kx) — so it is
        // exactly a Linear over the flattened patch.
        let patch_embd = Proj {
            name: "v.patch_embd.weight".into(),
            n_in: 3 * patch * patch,
            n_out: n_embd,
            clamp: Clamp::NONE,
        };
        let pe_info = src
            .tensors()
            .iter()
            .find(|t| t.name == "v.position_embd.weight")
            .ok_or_else(|| anyhow!("mmproj: missing v.position_embd.weight"))?;
        if pe_info.shape.len() != 3 || pe_info.shape[0] != n_embd || pe_info.shape[2] != 2 {
            bail!(
                "mmproj: unexpected v.position_embd shape {:?}",
                pe_info.shape
            );
        }
        let n_pos = pe_info.shape[1];
        let pos_embd = tensor_f32(&src, "v.position_embd.weight")?;
        let mut mm_proj = proj("mm.input_projection")?;
        mm_proj.clamp = Clamp::NONE; // never clipped (no scalars in the GGUF either)
        let std_bias = tensor_f32(&src, "v.std_bias").ok();
        let std_scale = tensor_f32(&src, "v.std_scale").ok();

        tracing::info!(
            "[vision] gemma4v encoder from {}: {n_layer} layers, {n_embd} wide, patch {patch}, merge {merge} -> {n_embd_out}",
            path.display()
        );
        Ok(Self {
            src,
            path: path.to_path_buf(),
            n_embd,
            n_head,
            patch,
            merge,
            eps,
            rope_theta: 100.0,
            n_embd_out,
            patch_embd,
            pos_embd,
            n_pos,
            layers,
            mm_proj,
            std_bias,
            std_scale,
            cache: Mutex::new(Vec::new()),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The first `mmproj*.gguf` beside `model` (the HF snapshot directory holds both).
    pub fn sibling_of(model: &Path) -> Option<PathBuf> {
        let dir = model.parent()?;
        let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    n.to_ascii_lowercase().starts_with("mmproj") && n.ends_with(".gguf")
                })
            })
            .collect();
        // Prefer the widest float (F32 > BF16 > F16) when several are present: deterministic, and
        // the encoder widens to f32 anyway.
        found.sort();
        found.into_iter().next()
    }

    /// Target `(width, height)` for an image of `w x h`: llama.cpp's `calc_size_preserved_ratio`
    /// (round-half-away, `align = patch * merge`, pixel budget from the token limits).
    pub fn target_size(&self, w: usize, h: usize, max_tokens: usize) -> (usize, usize) {
        let align = (self.patch * self.merge) as f32;
        let area = (self.patch * self.patch * self.merge * self.merge) as f32;
        let max_tokens = max_tokens.clamp(MIN_IMAGE_TOKENS, MAX_IMAGE_TOKENS);
        let min_px = MIN_IMAGE_TOKENS as f32 * area;
        let max_px = max_tokens as f32 * area;
        let (wf, hf) = (w as f32, h as f32);
        let round = |x: f32| (x / align).round() * align;
        let ceil = |x: f32| (x / align).ceil() * align;
        let floor = |x: f32| (x / align).floor() * align;
        let mut wb = round(wf).max(align);
        let mut hb = round(hf).max(align);
        if wb * hb > max_px {
            let beta = (hf * wf / max_px).sqrt();
            hb = floor(hf / beta).max(align);
            wb = floor(wf / beta).max(align);
        } else if wb * hb < min_px {
            let beta = (min_px / (hf * wf)).sqrt();
            hb = ceil(hf * beta);
            wb = ceil(wf * beta);
        }
        (wb as usize, hb as usize)
    }

    /// Decode `bytes` (PNG/JPEG/GIF/WebP/BMP) and encode it: `[n_tokens * n_embd_out]` row-major.
    /// Cached by content, so a resent image costs nothing.
    pub fn encode(&self, bytes: &[u8], max_tokens: usize) -> Result<Arc<Vec<f32>>> {
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut h);
            h.finish()
        };
        if let Some(hit) = self
            .cache
            .lock()
            .unwrap()
            .iter()
            .find(|(k, t, _)| *k == key && *t == max_tokens)
            .map(|(_, _, v)| v.clone())
        {
            return Ok(hit);
        }
        let t0 = std::time::Instant::now();
        let img = image::load_from_memory(bytes)
            .map_err(|e| anyhow!("image decode failed: {e}"))?
            .to_rgb8();
        let (w0, h0) = (img.width() as usize, img.height() as usize);
        let (w, h) = self.target_size(w0, h0, max_tokens);
        let img = image::imageops::resize(
            &img,
            w as u32,
            h as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let rgb: Vec<f32> = img.as_raw().iter().map(|&b| b as f32 / 255.0).collect();
        let out = Arc::new(self.encode_pixels(&rgb, w, h)?);
        tracing::info!(
            "[vision] encoded {w0}x{h0} image as {w}x{h} -> {} tokens in {:.2}s",
            out.len() / self.n_embd_out,
            t0.elapsed().as_secs_f32()
        );
        let mut cache = self.cache.lock().unwrap();
        if cache.len() >= CACHE_IMAGES {
            cache.remove(0);
        }
        cache.push((key, max_tokens, out.clone()));
        Ok(out)
    }

    /// Encode an already-resized image: `rgb` is HWC, values in [0, 1], `w`/`h` multiples of
    /// `patch * merge`.
    pub fn encode_pixels(&self, rgb: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
        let (p, m, ne) = (self.patch, self.merge, self.n_embd);
        if !w.is_multiple_of(p * m) || !h.is_multiple_of(p * m) || rgb.len() != w * h * 3 {
            bail!("vision: bad image geometry {w}x{h} ({} values)", rgb.len());
        }
        let (cols, rows) = (w / p, h / p);
        if cols > self.n_pos || rows > self.n_pos {
            bail!(
                "vision: {cols}x{rows} patches exceeds the {} position table",
                self.n_pos
            );
        }
        let n = cols * rows;

        // Patchify: one (c, ky, kx) vector per patch, scaled to [-1, 1].
        let pin = 3 * p * p;
        let mut patches = vec![0f32; n * pin];
        patches
            .par_chunks_mut(pin)
            .enumerate()
            .for_each(|(i, dst)| {
                let (px, py) = (i % cols, i / cols);
                for c in 0..3 {
                    for ky in 0..p {
                        for kx in 0..p {
                            let (x, y) = (px * p + kx, py * p + ky);
                            dst[c * p * p + ky * p + kx] = rgb[(y * w + x) * 3 + c] * 2.0 - 1.0;
                        }
                    }
                }
            });
        let mut x = self.linear(&self.patch_embd, &patches, n)?;
        drop(patches);
        let pos_x: Vec<usize> = (0..n).map(|i| i % cols).collect();
        let pos_y: Vec<usize> = (0..n).map(|i| i / cols).collect();
        x.par_chunks_mut(ne).enumerate().for_each(|(i, row)| {
            let tx = &self.pos_embd[pos_x[i] * ne..(pos_x[i] + 1) * ne];
            let ty = &self.pos_embd[(self.n_pos + pos_y[i]) * ne..(self.n_pos + pos_y[i] + 1) * ne];
            for j in 0..ne {
                row[j] += tx[j] + ty[j];
            }
        });

        for layer in &self.layers {
            self.block(layer, &mut x, n, &pos_x, &pos_y)?;
        }

        // 3x3 average pool over the patch grid (token = oy * out_x + ox), times sqrt(n_embd).
        let (ox_n, oy_n) = (cols / m, rows / m);
        let n_tok = ox_n * oy_n;
        let scale = (ne as f32).sqrt() / (m * m) as f32;
        let mut pooled = vec![0f32; n_tok * ne];
        pooled.par_chunks_mut(ne).enumerate().for_each(|(t, dst)| {
            let (ox, oy) = (t % ox_n, t / ox_n);
            for dy in 0..m {
                for dx in 0..m {
                    let src = (oy * m + dy) * cols + ox * m + dx;
                    for (d, s) in dst.iter_mut().zip(&x[src * ne..(src + 1) * ne]) {
                        *d += s;
                    }
                }
            }
            dst.iter_mut().for_each(|v| *v *= scale);
        });
        drop(x);
        if let (Some(b), Some(s)) = (&self.std_bias, &self.std_scale) {
            pooled.par_chunks_mut(ne).for_each(|row| {
                for j in 0..ne {
                    row[j] = (row[j] - b[j]) * s[j];
                }
            });
        }
        rms_rows(&mut pooled, ne, None, self.eps);
        self.linear(&self.mm_proj, &pooled, n_tok)
    }

    fn block(
        &self,
        l: &Layer,
        x: &mut [f32],
        n: usize,
        pos_x: &[usize],
        pos_y: &[usize],
    ) -> Result<()> {
        let ne = self.n_embd;
        let hd = ne / self.n_head;

        let mut h = x.to_vec();
        rms_rows(&mut h, ne, Some(&l.ln1), self.eps);
        let mut q = self.linear(&l.q, &h, n)?;
        let mut k = self.linear(&l.k, &h, n)?;
        let mut v = self.linear(&l.v, &h, n)?;
        rms_rows(&mut q, hd, Some(&l.q_norm), self.eps);
        rms_rows(&mut k, hd, Some(&l.k_norm), self.eps);
        rms_rows(&mut v, hd, None, self.eps);
        self.rope_2d(&mut q, n, pos_x, pos_y);
        self.rope_2d(&mut k, n, pos_x, pos_y);
        let a = attention(&q, &k, &v, n, self.n_head, hd)?;
        drop((q, k, v));
        let mut o = self.linear(&l.o, &a, n)?;
        rms_rows(&mut o, ne, Some(&l.attn_post_norm), self.eps);
        x.par_iter_mut().zip(&o).for_each(|(a, b)| *a += b);

        h.copy_from_slice(x);
        rms_rows(&mut h, ne, Some(&l.ln2), self.eps);
        let up = self.linear(&l.up, &h, n)?;
        let mut gate = self.linear(&l.gate, &h, n)?;
        gate.par_iter_mut()
            .zip(&up)
            .for_each(|(g, u)| *g = gelu_tanh(*g) * u);
        drop(up);
        let mut d = self.linear(&l.down, &gate, n)?;
        rms_rows(&mut d, ne, Some(&l.ffn_post_norm), self.eps);
        x.par_iter_mut().zip(&d).for_each(|(a, b)| *a += b);
        Ok(())
    }

    /// NEOX rope on each head: dims `[0, hd/2)` rotated by the patch column, `[hd/2, hd)` by its
    /// row — two `ggml_rope_ext(n_dims = hd/2)` calls, the second offset by `hd/2`.
    fn rope_2d(&self, t: &mut [f32], n: usize, pos_x: &[usize], pos_y: &[usize]) {
        let ne = self.n_embd;
        let hd = ne / self.n_head;
        let half = hd / 2; // rotated span per axis
        let pairs = half / 2;
        let inv: Vec<f32> = (0..pairs)
            .map(|i| self.rope_theta.powf(-2.0 * i as f32 / half as f32))
            .collect();
        t.par_chunks_mut(ne)
            .take(n)
            .enumerate()
            .for_each(|(r, row)| {
                for head in row.chunks_mut(hd) {
                    for (axis, pos) in [(0usize, pos_x[r]), (1usize, pos_y[r])] {
                        let span = &mut head[axis * half..(axis + 1) * half];
                        for i in 0..pairs {
                            let (s, c) = (pos as f32 * inv[i]).sin_cos();
                            let (a, b) = (span[i], span[i + pairs]);
                            span[i] = a * c - b * s;
                            span[i + pairs] = a * s + b * c;
                        }
                    }
                }
            });
    }

    /// `clip(clip(x) · Wᵀ)` for `rows` rows.
    fn linear(&self, p: &Proj, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        debug_assert_eq!(x.len(), rows * p.n_in);
        let info = self
            .src
            .tensors()
            .iter()
            .find(|t| t.name == p.name)
            .ok_or_else(|| anyhow!("mmproj: missing tensor {}", p.name))?;
        let w = widen(
            info.dtype,
            self.src
                .tensor_bytes(&p.name)
                .map_err(|e| anyhow!("mmproj: {}: {e}", p.name))?,
        )?;
        let c = p.clamp;
        let clipped;
        let x = if c.in_lo.is_finite() || c.in_hi.is_finite() {
            clipped = x
                .par_iter()
                .map(|v| v.clamp(c.in_lo, c.in_hi))
                .collect::<Vec<_>>();
            &clipped[..]
        } else {
            x
        };
        let mut y = vec![0f32; rows * p.n_out];
        matmul_nt(&mut y, x, &w, rows, p.n_in, p.n_out);
        if c.out_lo.is_finite() || c.out_hi.is_finite() {
            y.par_iter_mut()
                .for_each(|v| *v = v.clamp(c.out_lo, c.out_hi));
        }
        Ok(y)
    }
}

/// `y[m, n] = a[m, k] · b[n, k]ᵀ` (both row-major), via `gemm` on all cores.
pub(crate) fn matmul_nt(y: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    assert!(y.len() >= m * n && a.len() >= m * k && b.len() >= n * k);
    // SAFETY: the asserts bound every access gemm makes for these shapes/strides; `y` does not
    // alias `a`/`b` (distinct borrows).
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            y.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            k as isize,
            1,
            0.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }
}

/// Full (bidirectional) attention, scale 1: `q,k,v` are `[n, n_head * hd]`.
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    n_head: usize,
    hd: usize,
) -> Result<Vec<f32>> {
    let ne = n_head * hd;
    let mut out = vec![0f32; n * ne];
    let mut s = vec![0f32; n * n];
    for h in 0..n_head {
        // S = Q_h · K_hᵀ : strided views into the packed [n, ne] rows.
        // SAFETY: every index stays inside q/k (rows < n, cols < hd from offset h*hd < ne).
        unsafe {
            gemm::gemm(
                n,
                n,
                hd,
                s.as_mut_ptr(),
                1,
                n as isize,
                false,
                q.as_ptr().add(h * hd),
                1,
                ne as isize,
                k.as_ptr().add(h * hd),
                ne as isize,
                1,
                0.0,
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
        s.par_chunks_mut(n).for_each(|row| {
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - mx).exp();
                sum += *v;
            }
            let inv = 1.0 / sum;
            row.iter_mut().for_each(|v| *v *= inv);
        });
        // O_h = S · V_h, written straight into this head's columns of `out`.
        // SAFETY: as above; `out` columns [h*hd, (h+1)*hd) are disjoint per head.
        unsafe {
            gemm::gemm(
                n,
                hd,
                n,
                out.as_mut_ptr().add(h * hd),
                1,
                ne as isize,
                false,
                s.as_ptr(),
                1,
                n as isize,
                v.as_ptr().add(h * hd),
                1,
                ne as isize,
                0.0,
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
    }
    Ok(out)
}

/// RMS-normalize every `width`-wide row in place, then multiply by `w` (when given).
fn rms_rows(x: &mut [f32], width: usize, w: Option<&[f32]>, eps: f32) {
    x.par_chunks_mut(width).for_each(|row| {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let r = 1.0 / (ms + eps).sqrt();
        match w {
            Some(w) => row.iter_mut().zip(w).for_each(|(v, g)| *v *= r * g),
            None => row.iter_mut().for_each(|v| *v *= r),
        }
    });
}

/// ggml's `gelu` (tanh approximation), which `ggml_geglu_split` applies to the gate.
fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_nt_matches_naive() {
        let (m, k, n) = (7, 13, 5);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let mut y = vec![0f32; m * n];
        matmul_nt(&mut y, &a, &b, m, k, n);
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum();
                assert!((y[i * n + j] - want).abs() < 1e-4, "({i},{j})");
            }
        }
    }

    #[test]
    fn attention_matches_naive() {
        let (n, nh, hd) = (6, 2, 4);
        let ne = nh * hd;
        let f = |s: f32| {
            (0..n * ne)
                .map(move |i| (i as f32 * s).sin())
                .collect::<Vec<f32>>()
        };
        let (q, k, v) = (f(0.3), f(0.7), f(1.1));
        let got = attention(&q, &k, &v, n, nh, hd).unwrap();
        for h in 0..nh {
            for i in 0..n {
                let sc: Vec<f32> = (0..n)
                    .map(|j| {
                        (0..hd)
                            .map(|d| q[i * ne + h * hd + d] * k[j * ne + h * hd + d])
                            .sum()
                    })
                    .collect();
                let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                let z: f32 = e.iter().sum();
                for d in 0..hd {
                    let want: f32 = (0..n).map(|j| e[j] / z * v[j * ne + h * hd + d]).sum();
                    assert!((got[i * ne + h * hd + d] - want).abs() < 1e-5);
                }
            }
        }
    }
}
