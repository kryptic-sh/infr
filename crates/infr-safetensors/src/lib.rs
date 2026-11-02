//! Zero-copy safetensors weight loader: the [`WeightSource`] impl for Hugging Face
//! checkpoints — EXL3 quant files (`trellis`/`suh`/`svh` tensors) and dense models.
//!
//! The file is memory-mapped; tensor bytes are borrowed slices, never copied. Layout
//! follows the safetensors spec: `u64` LE header length, JSON header
//! (`__metadata__` + per-tensor `{dtype, shape, data_offsets}`), then the data
//! region. `data_offsets` are relative to the data region's start.
//!
//! Single-file only: a sharded model (`model-00001-of-00002.safetensors`) must be
//! opened per shard by the caller — EXL3 checkpoints ship single-file, and the
//! multi-file index (`model.safetensors.index.json`) is future scope.

use infr_core::error::Result;
use infr_core::loader::{MetaValue, Metadata, TensorInfo, WeightSource};
use infr_core::{DType, Error};
use std::collections::HashMap;
use std::path::Path;

/// A memory-mapped safetensors file implementing [`WeightSource`].
pub struct Safetensors {
    mmap: memmap2::Mmap,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Safetensors {
    /// Open and parse a safetensors file. The file is memory-mapped; no tensor
    /// bytes are copied.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Loader(format!("open {}: {e}", path.display())))?;
        // SAFETY: same contract as the GGUF loader — the file must not be written to or
        // truncated while the mapping is live.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::Loader(format!("mmap {}: {e}", path.display())))?;
        Self::parse(mmap, path)
    }

    fn parse(mmap: memmap2::Mmap, path: &Path) -> Result<Self> {
        let file_len = mmap.len();
        let hlen_size: usize = 8;
        if file_len < hlen_size {
            return Err(Error::Loader(format!(
                "{} is {file_len} bytes, shorter than the 8-byte safetensors header length",
                path.display()
            )));
        }
        let hlen = usize::try_from(u64::from_le_bytes(
            mmap[..8].try_into().expect("len checked"),
        ))
        .map_err(|_| Error::Loader("safetensors header length overflows usize".into()))?;
        let data_start = hlen_size
            .checked_add(hlen)
            .ok_or_else(|| Error::Loader("safetensors header length overflows usize".into()))?;
        if data_start > file_len {
            return Err(Error::Loader(format!(
                "{} claims a {hlen}-byte header but is only {file_len} bytes",
                path.display()
            )));
        }
        let header: serde_json::Value = serde_json::from_slice(&mmap[hlen_size..data_start])
            .map_err(|e| Error::Loader(format!("{}: invalid header JSON: {e}", path.display())))?;
        let obj = header.as_object().ok_or_else(|| {
            Error::Loader(format!("{}: header is not a JSON object", path.display()))
        })?;

        let mut metadata = Metadata { kv: HashMap::new() };
        if let Some(meta) = obj.get("__metadata__") {
            let meta_obj = meta.as_object().ok_or_else(|| {
                Error::Loader(format!(
                    "{}: `__metadata__` is not an object",
                    path.display()
                ))
            })?;
            for (k, v) in meta_obj {
                let s = v.as_str().ok_or_else(|| {
                    Error::Loader(format!(
                        "{}: `__metadata__[{k}]` is not a string",
                        path.display()
                    ))
                })?;
                metadata.kv.insert(k.clone(), MetaValue::Str(s.to_string()));
            }
        }

        let mut tensors = Vec::new();
        // `preserve_order` (workspace serde_json) keeps file order — deterministic enumeration.
        for (name, entry) in obj {
            if name == "__metadata__" {
                continue;
            }
            let shape = entry
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| {
                    Error::Loader(format!(
                        "{name}: missing `shape` array in {}",
                        path.display()
                    ))
                })?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|u| usize::try_from(u).ok())
                        .ok_or_else(|| {
                            Error::Loader(format!("{name}: bad shape entry in {}", path.display()))
                        })
                })
                .collect::<Result<Vec<usize>>>()?;
            let dtype_str = entry.get("dtype").and_then(|d| d.as_str()).ok_or_else(|| {
                Error::Loader(format!("{name}: missing `dtype` in {}", path.display()))
            })?;
            let dtype = map_dtype(dtype_str)
                .ok_or_else(|| Error::Loader(format!("{name}: unsupported dtype `{dtype_str}`")))?;
            let offsets = entry
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .ok_or_else(|| {
                    Error::Loader(format!(
                        "{name}: missing `data_offsets` in {}",
                        path.display()
                    ))
                })?;
            if offsets.len() != 2 {
                return Err(Error::Loader(format!(
                    "{name}: `data_offsets` must be [begin, end] in {}",
                    path.display()
                )));
            }
            let begin = offsets[0].as_u64().ok_or_else(|| {
                Error::Loader(format!("{name}: bad data_offsets in {}", path.display()))
            })?;
            let end = offsets[1].as_u64().ok_or_else(|| {
                Error::Loader(format!("{name}: bad data_offsets in {}", path.display()))
            })?;
            if end < begin {
                return Err(Error::Loader(format!(
                    "{name}: data_offsets end {end} < begin {begin} in {}",
                    path.display()
                )));
            }
            let nbytes = usize::try_from(end - begin)
                .map_err(|_| Error::Loader(format!("{name}: tensor byte range overflows usize")))?;
            let numel = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| Error::Loader(format!("{name}: shape product overflows usize")))?;
            let width = dtype.dense_bytes(1).ok_or_else(|| {
                Error::Loader(format!("{name}: quant dtype {dtype:?} in safetensors"))
            })?;
            let expect = numel.checked_mul(width).ok_or_else(|| {
                Error::Loader(format!("{name}: tensor byte size overflows usize"))
            })?;
            if nbytes != expect {
                return Err(Error::Loader(format!(
                    "{name}: byte range is {nbytes} bytes but {numel} x {dtype:?} needs {expect}"
                )));
            }
            let offset = (data_start as u64)
                .checked_add(begin)
                .ok_or_else(|| Error::Loader(format!("{name}: tensor offset overflows u64")))?;
            let abs_end = offset
                .checked_add(nbytes as u64)
                .ok_or_else(|| Error::Loader(format!("{name}: tensor end overflows u64")))?;
            if abs_end > file_len as u64 {
                return Err(Error::Loader(format!(
                    "{name}: bytes {offset}..{abs_end} exceed file length {file_len}"
                )));
            }
            tensors.push(TensorInfo {
                name: name.clone(),
                shape,
                dtype,
                offset,
                nbytes,
            });
        }
        let mut ranges: Vec<_> = tensors.iter().map(|t| (t.offset, t.nbytes)).collect();
        ranges.sort_unstable();
        let mut cursor = data_start as u64;
        for (offset, size) in ranges {
            if offset != cursor {
                return Err(Error::Loader(
                    "safetensors tensor ranges overlap or leave a gap".into(),
                ));
            }
            cursor += size as u64;
        }
        if cursor != file_len as u64 {
            return Err(Error::Loader(
                "safetensors has unclaimed trailing data".into(),
            ));
        }
        Ok(Self {
            mmap,
            metadata,
            tensors,
        })
    }
}

/// safetensors dtype string → [`DType`]. `None` for types infr has no dense
/// representation for (`U8`, `I8`, `I64`, `F64`, `BOOL`, …) — the loader rejects
/// those tensors rather than mis-typing them.
fn map_dtype(s: &str) -> Option<DType> {
    match s {
        "F32" => Some(DType::F32),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::Bf16),
        "I32" => Some(DType::I32),
        "U32" => Some(DType::U32),
        "I16" => Some(DType::I16),
        _ => None,
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl WeightSource for Safetensors {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Returns a slice into the mmap'd data region. Lifetime tied to `&self`.
    fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::Loader(format!("tensor not found: '{name}'")))?;
        let start = info.offset as usize;
        Ok(&self.mmap[start..start + info.nbytes])
    }

    /// HF safetensors files carry no chat template (it lives in
    /// `tokenizer_config.json` beside the weights).
    fn chat_template(&self) -> Option<&str> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal valid safetensors file: `__metadata__` with one string entry, one F16
    /// `[2]` tensor (`w`, bytes 0x00..0x04) and one I16 `[3]` trellis-like tensor (`t`, 6 bytes).
    fn build_fixture() -> Vec<u8> {
        let header = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
            "t": {"dtype": "I16", "shape": [3], "data_offsets": [4, 10]},
        });
        let hbytes = serde_json::to_vec(&header).unwrap();
        let mut b = Vec::new();
        b.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
        b.extend_from_slice(&hbytes);
        b.extend_from_slice(&[0x00, 0x3c, 0x00, 0x3e]); // w: 1.0, 1.5 in f16 LE
        b.extend_from_slice(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06]); // t: raw words
        b
    }

    fn write_temp(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(bytes).expect("write fixture");
        f.flush().unwrap();
        f
    }

    fn file_for_header(header: serde_json::Value, bytes: &[u8]) -> tempfile::NamedTempFile {
        let h = serde_json::to_vec(&header).unwrap();
        let mut file = (h.len() as u64).to_le_bytes().to_vec();
        file.extend(h);
        file.extend_from_slice(bytes);
        write_temp(&file)
    }

    #[test]
    fn rejects_size_overflows_without_panicking() {
        for shape in [vec![u64::MAX, 4], vec![usize::MAX as u64 / 2 + 1]] {
            let file = file_for_header(
                serde_json::json!({
                    "w":{"dtype":"F16","shape":shape,"data_offsets":[0,0]}
                }),
                &[],
            );
            assert!(Safetensors::open(file.path()).is_err());
        }
    }

    #[test]
    fn rejects_overlapping_gapped_and_unclaimed_ranges() {
        for (begin, end, total) in [(0, 2, 4), (3, 5, 5), (2, 4, 6)] {
            let file = file_for_header(
                serde_json::json!({
                    "a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},
                    "b":{"dtype":"F16","shape":[1],"data_offsets":[begin,end]}
                }),
                &vec![0; total],
            );
            assert!(Safetensors::open(file.path()).is_err());
        }
    }

    #[test]
    fn accepts_empty_and_scalar_tensors() {
        let file = file_for_header(
            serde_json::json!({
                "a":{"dtype":"F16","shape":[0],"data_offsets":[0,0]},
                "b":{"dtype":"F16","shape":[],"data_offsets":[0,2]}
            }),
            &[0, 0],
        );
        let source = Safetensors::open(file.path()).unwrap();
        assert!(source.tensor_bytes("a").unwrap().is_empty());
        assert_eq!(source.tensor_bytes("b").unwrap(), &[0, 0]);
    }

    #[test]
    fn parses_header_tensors_and_metadata() {
        let f = write_temp(&build_fixture());
        let st = Safetensors::open(f.path()).unwrap();
        assert_eq!(
            st.metadata().str("format"),
            Some("pt"),
            "the __metadata__ string map must surface through metadata()"
        );
        assert_eq!(st.tensors().len(), 2);
        let w = st.tensors().iter().find(|t| t.name == "w").unwrap();
        assert_eq!(
            (w.shape.as_slice(), w.dtype, w.nbytes),
            (&[2][..], DType::F16, 4)
        );
        let t = st.tensors().iter().find(|t| t.name == "t").unwrap();
        assert_eq!(
            (t.shape.as_slice(), t.dtype, t.nbytes),
            (&[3][..], DType::I16, 6)
        );
        assert_eq!(st.tensor_bytes("w").unwrap(), &[0x00, 0x3c, 0x00, 0x3e]);
        assert_eq!(
            st.tensor_bytes("t").unwrap(),
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06]
        );
        assert!(st.chat_template().is_none());
        assert!(st.tensor_bytes("missing").is_err());
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let header = serde_json::json!({
            "q": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        });
        let hbytes = serde_json::to_vec(&header).unwrap();
        let mut b = Vec::new();
        b.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
        b.extend_from_slice(&hbytes);
        b.extend_from_slice(&[0u8; 4]);
        let f = write_temp(&b);
        let err = match Safetensors::open(f.path()) {
            Ok(_) => panic!("U8 tensor must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("U8"), "must name the rejected dtype: {err}");
    }

    #[test]
    fn rejects_size_mismatch_and_overrun() {
        // Declared 4 bytes for an F16 [3] (needs 6).
        let header = serde_json::json!({
            "w": {"dtype": "F16", "shape": [3], "data_offsets": [0, 4]},
        });
        let hbytes = serde_json::to_vec(&header).unwrap();
        let mut b = Vec::new();
        b.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
        b.extend_from_slice(&hbytes);
        b.extend_from_slice(&[0u8; 4]);
        let f = write_temp(&b);
        assert!(Safetensors::open(f.path()).is_err());
        // Offsets past EOF.
        let header = serde_json::json!({
            "w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 400]},
        });
        let hbytes = serde_json::to_vec(&header).unwrap();
        let mut b = Vec::new();
        b.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
        b.extend_from_slice(&hbytes);
        b.extend_from_slice(&[0u8; 4]);
        let f = write_temp(&b);
        assert!(Safetensors::open(f.path()).is_err());
        // Truncated file (header claims more than exists).
        let f = write_temp(&[0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7b, 0x7d]);
        assert!(Safetensors::open(f.path()).is_err());
    }
}
