//! GGUF loader — `WeightSource` impl.
//!
//! Parses the GGUF binary format (little-endian) by mmap-ping the file and
//! walking a byte cursor through header → metadata KV pairs → tensor directory.
//! Quantised weight blocks are returned as-is; the backend owns dequantisation.
//!
//! References:
//!   `~/Projects/llama.cpp/ggml/include/gguf.h`  (format spec)
//!   `~/Projects/llama.cpp/ggml/include/ggml.h`  (ggml_type enum)
//!   `~/Projects/llama.cpp/conversion/diffusion_gemma.py` (tensor names / keys)

use infr_core::{
    error::{Error, Result},
    loader::{MetaValue, Metadata, TensorInfo},
    tensor::DType,
    WeightSource,
};
use memmap2::Mmap;
use std::{collections::HashMap, fs::File, path::Path};

// ─── constants ────────────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // b"GGUF" little-endian
const DEFAULT_ALIGNMENT: usize = 32; // GGUF_DEFAULT_ALIGNMENT

// ─── public struct ────────────────────────────────────────────────────────────

/// A parsed, mmap-backed GGUF file.
///
/// The `Mmap` handle keeps the backing memory alive for the lifetime of this
/// struct; `tensor_bytes` returns slices directly into that region.
pub struct Gguf {
    mmap: Mmap,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    /// Absolute byte offset into `mmap` where tensor data begins.
    data_region_start: usize,
}

// ─── byte-cursor ──────────────────────────────────────────────────────────────

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn ensure(&self, n: usize) -> Result<()> {
        if self.pos + n > self.buf.len() {
            Err(Error::Loader(format!(
                "GGUF: unexpected EOF at offset {} (need {n} more bytes, file is {} bytes)",
                self.pos,
                self.buf.len()
            )))
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        self.ensure(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        self.ensure(2)?;
        let v = u16::from_le_bytes(self.buf[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        self.ensure(4)?;
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        self.ensure(8)?;
        let v = u64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        self.ensure(4)?;
        let v = f32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_f64(&mut self) -> Result<f64> {
        self.ensure(8)?;
        let v = f64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    /// Read a GGUF string: u64 length prefix then UTF-8 bytes (no NUL).
    fn read_gguf_str(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        self.ensure(len)?;
        let s = std::str::from_utf8(&self.buf[self.pos..self.pos + len])
            .map_err(|e| {
                Error::Loader(format!(
                    "GGUF: invalid UTF-8 string at offset {}: {e}",
                    self.pos
                ))
            })?
            .to_owned();
        self.pos += len;
        Ok(s)
    }

    /// Recursively parse a metadata value given its GGUF type tag.
    fn read_meta_value(&mut self, vtype: u32) -> Result<MetaValue> {
        match vtype {
            0 => Ok(MetaValue::U64(self.read_u8()? as u64)), // UINT8
            1 => Ok(MetaValue::I64(self.read_i8()? as i64)), // INT8
            2 => Ok(MetaValue::U64(self.read_u16()? as u64)), // UINT16
            3 => Ok(MetaValue::I64(self.read_i16()? as i64)), // INT16
            4 => Ok(MetaValue::U64(self.read_u32()? as u64)), // UINT32
            5 => Ok(MetaValue::I64(self.read_i32()? as i64)), // INT32
            6 => Ok(MetaValue::F64(self.read_f32()? as f64)), // FLOAT32
            7 => Ok(MetaValue::Bool(self.read_bool()?)),     // BOOL
            8 => Ok(MetaValue::Str(self.read_gguf_str()?)),  // STRING
            9 => {
                // ARRAY: u32 elem_type, u64 count, then count × elem
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(self.read_meta_value(elem_type)?);
                }
                Ok(MetaValue::Arr(arr))
            }
            10 => Ok(MetaValue::U64(self.read_u64()?)), // UINT64
            11 => Ok(MetaValue::I64(self.read_i64()?)), // INT64
            12 => Ok(MetaValue::F64(self.read_f64()?)), // FLOAT64
            _ => Err(Error::Loader(format!("GGUF: unknown value type {vtype}"))),
        }
    }
}

// ─── ggml_type → DType + block sizing ────────────────────────────────────────

fn ggml_type_to_dtype(t: u32) -> Result<DType> {
    match t {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        8 => Ok(DType::Q8_0),
        12 => Ok(DType::Q4K),
        13 => Ok(DType::Q5K),
        14 => Ok(DType::Q6K),
        _ => Err(Error::Unsupported(format!("ggml type {t}"))),
    }
}

/// Returns `(elements_per_block, bytes_per_block)` for the GGUF block layout.
///
/// GGUF dim order: `ne[0]` is the fastest-varying axis (innermost / columns).
fn block_layout(dtype: DType) -> (usize, usize) {
    match dtype {
        DType::F32 => (1, 4),
        DType::F16 => (1, 2),
        DType::Q8_0 => (32, 34),
        DType::Q4K => (256, 144),
        DType::Q5K => (256, 176),
        DType::Q6K => (256, 210),
        // I32 / U32 are not reachable via ggml_type_to_dtype; kept for exhaustiveness
        DType::I32 | DType::U32 => (1, 4),
    }
}

fn tensor_nbytes(dtype: DType, numel: usize) -> usize {
    let (be, bb) = block_layout(dtype);
    (numel / be) * bb
}

// ─── Gguf::open ───────────────────────────────────────────────────────────────

impl Gguf {
    /// Open and parse a GGUF file.
    ///
    /// The file is memory-mapped; no tensor bytes are copied into RAM.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the file is not modified while this Mmap is live.
        let mmap = unsafe { Mmap::map(&file) }?;

        // All parsing happens in this block so the borrow of `mmap` ends
        // before we move `mmap` into the returned struct.
        let (metadata, tensors, data_region_start) = {
            let buf: &[u8] = &mmap;
            let mut r = Reader::new(buf);

            // ── header ────────────────────────────────────────────────────────
            let magic = r.read_u32()?;
            if magic != GGUF_MAGIC {
                return Err(Error::Loader(format!(
                    "not a GGUF file (magic 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X})"
                )));
            }

            let version = r.read_u32()?;
            if version != 2 && version != 3 {
                return Err(Error::Loader(format!(
                    "unsupported GGUF version {version} (supported: 2, 3)"
                )));
            }

            let tensor_count = r.read_u64()? as usize;
            let kv_count = r.read_u64()? as usize;

            // ── metadata KV pairs ─────────────────────────────────────────────
            let mut kv: HashMap<String, MetaValue> = HashMap::with_capacity(kv_count);
            for _ in 0..kv_count {
                let key = r.read_gguf_str()?;
                let vtype = r.read_u32()?;
                let val = r.read_meta_value(vtype)?;
                kv.insert(key, val);
            }
            let metadata = Metadata { kv };

            // alignment from metadata, defaulting to GGUF_DEFAULT_ALIGNMENT (32)
            let alignment = metadata
                .u64("general.alignment")
                .unwrap_or(DEFAULT_ALIGNMENT as u64) as usize;

            // ── tensor info entries ───────────────────────────────────────────
            // Collect raw fields first (name, shape, ggml_type, offset); then
            // convert after parsing so errors are reported before we consume the
            // remaining header bytes.
            let mut raw: Vec<(String, Vec<usize>, u32, u64)> = Vec::with_capacity(tensor_count);
            for _ in 0..tensor_count {
                let name = r.read_gguf_str()?;
                let n_dims = r.read_u32()? as usize;
                let mut shape = Vec::with_capacity(n_dims);
                for _ in 0..n_dims {
                    shape.push(r.read_u64()? as usize);
                }
                let ggml_type = r.read_u32()?;
                let offset = r.read_u64()?;
                raw.push((name, shape, ggml_type, offset));
            }

            // ── data region: align cursor position to `alignment` ─────────────
            let data_region_start = r.pos.div_ceil(alignment) * alignment;

            // ── convert raw tensor entries to TensorInfo ─────────────────────
            let mut tensors: Vec<TensorInfo> = Vec::with_capacity(tensor_count);
            for (name, shape, ggml_type, offset) in raw {
                let dtype = ggml_type_to_dtype(ggml_type)?;
                let numel: usize = shape.iter().product();
                let nbytes = tensor_nbytes(dtype, numel);
                tensors.push(TensorInfo {
                    name,
                    shape,
                    dtype,
                    offset,
                    nbytes,
                });
            }

            (metadata, tensors, data_region_start)
        }; // ← borrow of `mmap` ends here

        Ok(Gguf {
            mmap,
            metadata,
            tensors,
            data_region_start,
        })
    }
}

// ─── WeightSource impl ────────────────────────────────────────────────────────

impl WeightSource for Gguf {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Returns a slice into the mmap'd data region for the named tensor.
    ///
    /// The slice lifetime is tied to `&self` (i.e. the `Gguf` struct keeps
    /// the `Mmap` alive).
    fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::Loader(format!("tensor not found: '{name}'")))?;

        let start = self.data_region_start + info.offset as usize;
        let end = start + info.nbytes;
        let buf: &[u8] = &self.mmap;

        if end > buf.len() {
            return Err(Error::Loader(format!(
                "tensor '{name}' byte range {start}..{end} exceeds file size {}",
                buf.len()
            )));
        }

        Ok(&buf[start..end])
    }

    fn chat_template(&self) -> Option<&str> {
        self.metadata.str("tokenizer.chat_template")
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── fixture builder ───────────────────────────────────────────────────────

    fn u32_le(v: u32) -> [u8; 4] {
        v.to_le_bytes()
    }
    fn u64_le(v: u64) -> [u8; 8] {
        v.to_le_bytes()
    }

    fn push_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&u32_le(v));
    }
    fn push_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&u64_le(v));
    }
    fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
        push_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }

    /// Build a minimal valid GGUF v3 file in memory:
    ///   - 2 metadata KVs:
    ///       "test.block_count"    = UINT32 30
    ///       "general.architecture" = STRING "diffusion-gemma"
    ///   - 1 tensor: "tensor0", F32, shape [4], 16 bytes of zeros
    fn build_fixture() -> Vec<u8> {
        let mut b: Vec<u8> = Vec::new();

        // header
        push_u32(&mut b, GGUF_MAGIC); // magic
        push_u32(&mut b, 3); // version
        push_u64(&mut b, 1); // tensor_count
        push_u64(&mut b, 2); // kv_count

        // KV 1: "test.block_count" = UINT32 30
        push_gguf_str(&mut b, "test.block_count");
        push_u32(&mut b, 4); // GGUF_TYPE_UINT32
        push_u32(&mut b, 30);

        // KV 2: "general.architecture" = STRING "diffusion-gemma"
        push_gguf_str(&mut b, "general.architecture");
        push_u32(&mut b, 8); // GGUF_TYPE_STRING
        push_gguf_str(&mut b, "diffusion-gemma");

        // tensor info: "tensor0" F32 [4] offset=0
        push_gguf_str(&mut b, "tensor0");
        push_u32(&mut b, 1); // n_dims
        push_u64(&mut b, 4); // dim[0] = 4 elements
        push_u32(&mut b, 0); // ggml_type = F32
        push_u64(&mut b, 0); // offset = 0

        // pad to 32-byte alignment
        while b.len() % 32 != 0 {
            b.push(0);
        }

        // tensor data: 4 × f32 = 16 bytes
        b.extend_from_slice(&[0u8; 16]);

        b
    }

    // ── helper: write fixture to a named temp file ────────────────────────────

    fn write_temp_gguf(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(bytes).expect("write fixture");
        f.flush().expect("flush");
        f
    }

    // ── offline fixture test ──────────────────────────────────────────────────

    #[test]
    fn fixture_gguf_round_trip() {
        let bytes = build_fixture();
        let tmp = write_temp_gguf(&bytes);
        let gguf = Gguf::open(tmp.path()).expect("open fixture");

        // metadata
        assert_eq!(
            gguf.metadata().u64("test.block_count"),
            Some(30),
            "test.block_count should be 30"
        );
        assert_eq!(
            gguf.metadata().str("general.architecture"),
            Some("diffusion-gemma"),
            "general.architecture should be 'diffusion-gemma'"
        );

        // tensor directory
        let tensors = gguf.tensors();
        assert_eq!(tensors.len(), 1, "should have 1 tensor");
        let t = &tensors[0];
        assert_eq!(t.name, "tensor0");
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.shape, vec![4]);
        assert_eq!(t.nbytes, 16);

        // data region
        let data = gguf.tensor_bytes("tensor0").expect("tensor_bytes");
        assert_eq!(data.len(), 16, "tensor bytes should be 16");

        // chat_template absent → None
        assert!(gguf.chat_template().is_none());
    }

    // ── gated real-model test (skipped offline) ───────────────────────────────

    #[test]
    fn real_model_if_path_set() {
        let path = match std::env::var("INFR_TEST_GGUF") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => return, // skip when env var is not set
        };
        let gguf = Gguf::open(&path).expect("open real GGUF");
        assert!(
            !gguf.tensors().is_empty(),
            "real model should have at least one tensor"
        );
        assert!(
            !gguf.metadata().kv.is_empty(),
            "real model should have non-empty metadata"
        );
    }
}
