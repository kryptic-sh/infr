//! Tensor descriptors and data types (incl. GGUF quant types).

/// Element / block type of a tensor.
///
/// Quantized variants are stored as GGUF blocks; the backend owns dequant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    Bf16,
    I32,
    U32,
    // legacy scalar quants (needed to parse DiffusionGemma rope_freqs + scale tensors)
    Q5_0,
    Q5_1,
    // GGUF k-quants we care about for the MVP (extend as needed).
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

impl DType {
    /// True for block-quantized weight types.
    pub fn is_quant(self) -> bool {
        matches!(
            self,
            DType::Q5_0 | DType::Q5_1 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Q8_0
        )
    }

    /// Bytes for `n` elements of a non-quant dtype. Returns `None` for quant types
    /// (those are sized by block, computed by the loader from the GGUF layout).
    pub fn dense_bytes(self, n: usize) -> Option<usize> {
        let sz = match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::Bf16 => 2,
            _ => return None,
        };
        Some(n * sz)
    }
}

pub type Shape = Vec<usize>;

/// Shape + dtype of a tensor value flowing through the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDesc {
    pub shape: Shape,
    pub dtype: DType,
}

impl TensorDesc {
    pub fn new(shape: impl Into<Shape>, dtype: DType) -> Self {
        Self {
            shape: shape.into(),
            dtype,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Handle to a node's output value within a single [`crate::graph::Graph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub u32);
