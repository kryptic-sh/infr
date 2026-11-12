//! Synthetic GGUF files for tests: a model described as data ([`SyntheticModel`]: metadata
//! values and tensors with deterministic contents) and written to a temp file ([`TempGguf`]), so a
//! test can drive the real loader and runner without a downloaded model.
//!
//! A tensor's pseudo-random fill is seeded by its NAME, not by its position in the file, so two
//! variants of a model that add or remove one tensor differ only in that tensor.

use std::path::{Path, PathBuf};

// ─── GGUF metadata values ────────────────────────────────────────────────────────

/// A GGUF metadata value, restricted to the type tags this harness writes. Tags are the ggml
/// `gguf_type` enum (see `infr-gguf`'s `read_meta_value_at_depth`).
#[derive(Clone, Debug, PartialEq)]
pub enum Meta {
    U32(u32),
    /// An array of UINT32s — e.g. a per-layer `head_count_kv`.
    U32Arr(Vec<u32>),
    /// An INT32 — what `gguf-py`'s `add_int32` writes, e.g. `general.sampling.top_k`.
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    /// An array of STRINGs — the vocab and merge lists.
    StrArr(Vec<String>),
    /// An array of INT32s. This is what `gguf-py`'s `add_array` writes for a Python `list[int]`
    /// (`GGUFValueType.get_type` maps every `int` to `INT32`), so it is the on-disk type of
    /// `deepseek4.attention.compress_ratios`.
    I32Arr(Vec<i32>),
    /// An array of FLOAT32s — `deepseek4.swiglu_clamp_exp` / `_shexp` in their per-layer form.
    F32Arr(Vec<f32>),
}

fn push_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

/// A GGUF string: u64 byte length, then the UTF-8 bytes (no NUL).
fn push_gguf_str(b: &mut Vec<u8>, s: &str) {
    push_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}

impl Meta {
    fn write(&self, b: &mut Vec<u8>) {
        match self {
            Meta::U32(v) => {
                push_u32(b, 4);
                push_u32(b, *v);
            }
            Meta::U32Arr(a) => {
                push_u32(b, 9);
                push_u32(b, 4); // element type: UINT32
                push_u64(b, a.len() as u64);
                for v in a {
                    push_u32(b, *v);
                }
            }
            Meta::I32(v) => {
                push_u32(b, 5);
                b.extend_from_slice(&v.to_le_bytes());
            }
            Meta::F32(v) => {
                push_u32(b, 6);
                b.extend_from_slice(&v.to_le_bytes());
            }
            Meta::Bool(v) => {
                push_u32(b, 7);
                b.push(u8::from(*v));
            }
            Meta::Str(s) => {
                push_u32(b, 8);
                push_gguf_str(b, s);
            }
            Meta::StrArr(a) => {
                push_u32(b, 9);
                push_u32(b, 8); // element type: STRING
                push_u64(b, a.len() as u64);
                for s in a {
                    push_gguf_str(b, s);
                }
            }
            Meta::I32Arr(a) => {
                push_u32(b, 9);
                push_u32(b, 5); // element type: INT32
                push_u64(b, a.len() as u64);
                for v in a {
                    b.extend_from_slice(&v.to_le_bytes());
                }
            }
            Meta::F32Arr(a) => {
                push_u32(b, 9);
                push_u32(b, 6); // element type: FLOAT32
                push_u64(b, a.len() as u64);
                for v in a {
                    b.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
}

// ─── deterministic tensor values ─────────────────────────────────────────────────

/// Stable FNV-1a-64 — the per-tensor RNG seed. Not `DefaultHasher`, which is not stable across
/// toolchains and would make the "same file every run" promise a lie on the next compiler.
fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// SplitMix64 finalizer — the whole RNG. Stateless, so element `i` of a tensor depends only on
/// `(name, i)` and never on how many tensors came before it.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic value in `[-1, 1)` for element `i` of the tensor seeded by `seed`.
fn uniform(seed: u64, i: usize) -> f32 {
    let h = splitmix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    // 24 bits → an exactly-representable f32 in [0, 1).
    ((h >> 40) as f32) / f32::from(1u16 << 12) / f32::from(1u16 << 12) * 2.0 - 1.0
}

/// How a tensor's values are produced.
#[derive(Clone, Debug)]
pub enum Fill {
    /// Pseudo-random in `[-amp, amp]`.
    Rand(f32),
    /// `1 + 0.1·u` — an RMSNorm gain, kept near 1 so activations stay in a sane range.
    Gain,
    /// Verbatim values, for a tensor whose exact numbers are the point (a router bias) or whose
    /// values the caller generates itself.
    Exact(Vec<f32>),
    /// [`Fill::Exact`] written as `GGML_TYPE_F16` — for a test that needs a half-precision tensor,
    /// such as an f16 embedding a device-side gather reads.
    ExactF16(Vec<f32>),
    /// [`Fill::Exact`] written as `GGML_TYPE_BF16` — the dtype of every `ffn_gate_inp` router in
    /// the shipped DeepSeek V4 conversion. Values not representable in bf16 are rounded to nearest.
    ExactBf16(Vec<f32>),
    /// Expert ids drawn from `0..n_expert`, written as a `GGML_TYPE_I32` tensor. The one fill in
    /// this harness that is not f32 on disk, because the one tensor that uses it —
    /// `deepseek4`'s `ffn_gate_tid2eid` hash-routing table — is i32 in the reference conversion
    /// (`ggml-org/DeepSeek-V4-Flash-0731-GGUF`). Writing it f32 like everything else is what let
    /// the loader's missing `GGML_TYPE_I32` arm go unnoticed until a real file was opened.
    ExpertIds(usize),
    /// [`Fill::Exact`]'s i32 twin — verbatim expert ids, written as `GGML_TYPE_I32`. What the
    /// hash-routing tests build a `ffn_gate_tid2eid` with when WHICH expert each token names is
    /// the thing under test.
    ExactIds(Vec<i32>),
}

/// One tensor of the synthetic model. `shape` is in GGUF/ggml order — `ne0` (the fastest,
/// in-features dimension) FIRST, which is what a loader reads off `token_embd.weight`.
#[derive(Clone, Debug)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub fill: Fill,
}

impl TensorSpec {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, fill: Fill) -> Self {
        Self {
            name: name.into(),
            shape,
            fill,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn values(&self) -> Vec<f32> {
        let seed = fnv1a64(&self.name);
        match &self.fill {
            Fill::Rand(amp) => (0..self.numel()).map(|i| uniform(seed, i) * amp).collect(),
            Fill::Gain => (0..self.numel())
                .map(|i| 1.0 + 0.1 * uniform(seed, i))
                .collect(),
            Fill::Exact(v) | Fill::ExactF16(v) | Fill::ExactBf16(v) => {
                assert_eq!(v.len(), self.numel(), "{}: Exact fill length", self.name);
                v.clone()
            }
            // `uniform` is in [-1, 1); fold to [0, 1) before scaling so every id is in range.
            Fill::ExpertIds(n_expert) => (0..self.numel())
                .map(|i| ((uniform(seed, i) * 0.5 + 0.5) * *n_expert as f32).floor())
                .collect(),
            Fill::ExactIds(v) => {
                assert_eq!(v.len(), self.numel(), "{}: ExactIds fill length", self.name);
                v.iter().map(|&e| e as f32).collect()
            }
        }
    }

    /// The `ggml_type` this tensor is written as — `GGML_TYPE_I32` for the integer fills,
    /// `GGML_TYPE_F16` for [`Fill::ExactF16`], `GGML_TYPE_BF16` for [`Fill::ExactBf16`],
    /// `GGML_TYPE_F32` for every other.
    pub fn ggml_type(&self) -> u32 {
        match self.fill {
            Fill::ExpertIds(_) | Fill::ExactIds(_) => 26,
            Fill::ExactF16(_) => 1,
            Fill::ExactBf16(_) => 30,
            _ => 0,
        }
    }

    /// The length of [`Self::bytes`], without generating them.
    pub fn byte_len(&self) -> usize {
        match self.fill {
            Fill::ExactF16(_) | Fill::ExactBf16(_) => self.numel() * 2,
            _ => self.numel() * 4,
        }
    }

    /// The tensor's on-disk bytes: [`Self::values`] encoded in [`Self::ggml_type`]. The two are
    /// separate so the differential tests can compare LOGICAL values across variants without
    /// caring which encoding each tensor landed in.
    pub fn bytes(&self) -> Vec<u8> {
        let v = self.values();
        match self.fill {
            Fill::ExpertIds(_) | Fill::ExactIds(_) => {
                v.iter().flat_map(|x| (*x as i32).to_le_bytes()).collect()
            }
            Fill::ExactF16(_) => v
                .iter()
                .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
                .collect(),
            Fill::ExactBf16(_) => v
                .iter()
                .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
                .collect(),
            _ => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
}

// ─── the file ────────────────────────────────────────────────────────────────────

/// GGUF's default `general.alignment`, which this harness does not override.
const ALIGN: usize = 32;

/// A whole synthetic GGUF as data: the metadata KVs in write order, and the tensors.
pub struct SyntheticModel {
    pub meta: Vec<(String, Meta)>,
    pub tensors: Vec<TensorSpec>,
}

impl SyntheticModel {
    /// Serialize to GGUF v3 bytes, each tensor in its [`TensorSpec::ggml_type`]. Nothing is
    /// quantized: a quantized fixture would test the dequantizers rather than whatever the model
    /// was built to test.
    pub fn to_gguf_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        push_u32(&mut b, 0x4655_4747); // "GGUF"
        push_u32(&mut b, 3); // version
        push_u64(&mut b, self.tensors.len() as u64);
        push_u64(&mut b, self.meta.len() as u64);
        for (k, v) in &self.meta {
            push_gguf_str(&mut b, k);
            v.write(&mut b);
        }
        // Tensor directory. Offsets are relative to the data region and each is ALIGN-aligned.
        let mut offset = 0usize;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            offsets.push(offset);
            offset += t.byte_len().div_ceil(ALIGN) * ALIGN;
        }
        for (t, off) in self.tensors.iter().zip(&offsets) {
            push_gguf_str(&mut b, &t.name);
            push_u32(&mut b, t.shape.len() as u32);
            for d in &t.shape {
                push_u64(&mut b, *d as u64);
            }
            push_u32(&mut b, t.ggml_type());
            push_u64(&mut b, *off as u64);
        }
        while !b.len().is_multiple_of(ALIGN) {
            b.push(0);
        }
        let data_start = b.len();
        b.resize(data_start + offset, 0);
        for (t, off) in self.tensors.iter().zip(&offsets) {
            let base = data_start + off;
            let bytes = t.bytes();
            b[base..base + bytes.len()].copy_from_slice(&bytes);
        }
        b
    }
}

/// A synthetic GGUF on disk, removed when the test drops it. A `std::env::temp_dir()` file and an
/// explicit unlink rather than a `tempfile` dependency; the counter keeps concurrently-running
/// tests in one binary from colliding on a name.
pub struct TempGguf(PathBuf);

impl TempGguf {
    pub fn write(tag: &str, model: &SyntheticModel) -> Self {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("infr-synth-{tag}-{}-{n}.gguf", std::process::id()));
        std::fs::write(&path, model.to_gguf_bytes()).expect("write synthetic GGUF");
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempGguf {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
