//! Shared error/result types for the whole engine.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("backend: {0}")]
    Backend(String),
    #[error("loader: {0}")]
    Loader(String),
    #[error("model: {0}")]
    Model(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Error {
    /// Construct a backend error from anything Display — the constructor each backend used to
    /// reinvent (Vulkan's local `fn be`).
    pub fn backend(msg: impl std::fmt::Display) -> Self {
        Error::Backend(msg.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
