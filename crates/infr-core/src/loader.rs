//! The weight-source seam — formats (GGUF now, safetensors later) implement this.

use crate::error::Result;
use crate::tensor::DType;
use std::collections::HashMap;

/// A typed metadata value from a model file's KV store.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
    Str(String),
    Arr(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetaValue::U64(v) => Some(*v),
            MetaValue::I64(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetaValue::F64(v) => Some(*v),
            MetaValue::U64(v) => Some(*v as f64),
            MetaValue::I64(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Arr(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Metadata {
    pub kv: HashMap<String, MetaValue>,
}

impl Metadata {
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.kv.get(key)
    }
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(MetaValue::as_u64)
    }
    pub fn str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(MetaValue::as_str)
    }
}

/// Where a tensor lives in the backing file.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Byte offset into the tensor-data region of the file.
    pub offset: u64,
    pub nbytes: usize,
}

/// A model weight file. GGUF is the MVP impl (`infr-gguf`).
pub trait WeightSource: Send + Sync {
    fn metadata(&self) -> &Metadata;
    fn tensors(&self) -> &[TensorInfo];
    /// Raw bytes for a named tensor (quantized blocks returned as-is).
    fn tensor_bytes(&self, name: &str) -> Result<&[u8]>;
    /// Embedded jinja chat template, if present.
    fn chat_template(&self) -> Option<&str>;
}
