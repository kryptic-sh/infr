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
    // legacy round quants
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    // GGUF k-quants
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    // i-quants (codebook)
    Iq1S,
    Iq1M,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq4Nl,
    Iq4Xs,
    // ternary quants
    Tq1_0,
    Tq2_0,
    // fp4 quants
    Mxfp4,
    Nvfp4,
    // TurboQuant KV-cache-only formats (WHT rotation + PolarQuant centroids). NOT weight dtypes —
    // only used for the KV cache (like Q8_0-for-KV). turbo3 = 128-elem block, 50 bytes (3.125 bpw).
    Turbo3,
}

impl DType {
    /// True for block-quantized weight types.
    pub fn is_quant(self) -> bool {
        matches!(
            self,
            DType::Q4_0
                | DType::Q4_1
                | DType::Q5_0
                | DType::Q5_1
                | DType::Q8_0
                | DType::Q2K
                | DType::Q3K
                | DType::Q4K
                | DType::Q5K
                | DType::Q6K
                | DType::Iq1S
                | DType::Iq1M
                | DType::Iq2Xxs
                | DType::Iq2Xs
                | DType::Iq2S
                | DType::Iq3Xxs
                | DType::Iq3S
                | DType::Iq4Nl
                | DType::Iq4Xs
                | DType::Tq1_0
                | DType::Tq2_0
                | DType::Mxfp4
                | DType::Nvfp4
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
