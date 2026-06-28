//! `infr-core` — shared types + the four pluggable seams:
//! [`backend::Backend`] (GPU), [`loader::WeightSource`] (format), plus the
//! [`graph::Graph`]/[`tensor`] vocabulary the model layer builds against.
//!
//! Nothing here is GPU- or model-specific. See PLAN.md.

pub mod backend;
pub mod error;
pub mod graph;
pub mod iquant_grids;
pub mod loader;
pub mod progress;
pub mod tensor;

pub use backend::{Backend, Buffer, BufferUsage, Capabilities, Plan};
pub use error::{Error, Result};
pub use graph::{AttnMask, Bindings, Graph, Node, Op};
pub use loader::{MetaValue, Metadata, TensorInfo, WeightSource};
pub use tensor::{DType, Shape, TensorDesc, TensorId};
