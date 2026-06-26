//! Parsed model-reference grammar: `hf:org/repo[:file]`, `ollama:name[:tag]`, plain path.

use infr_core::error::{Error, Result};
use std::path::PathBuf;

/// A parsed model reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRef {
    /// `hf:org/repo[:file.gguf]`
    Hf { repo: String, file: Option<String> },
    /// `ollama:name[:tag]` — tag defaults to `"latest"`.
    Ollama { name: String, tag: String },
    /// A plain filesystem path to a `.gguf`.
    Path(PathBuf),
}

impl ModelRef {
    /// Parse `hf:…`, `ollama:…`, or a plain filesystem path.
    ///
    /// - `hf:org/repo`              → `Hf { repo: "org/repo", file: None }`
    /// - `hf:org/repo:file.gguf`    → `Hf { repo: "org/repo", file: Some("file.gguf") }`
    /// - `ollama:name`              → `Ollama { name: "name", tag: "latest" }`
    /// - `ollama:name:tag`          → `Ollama { name: "name", tag: "tag" }`
    /// - `ollama:ns/name:tag`       → `Ollama { name: "ns/name", tag: "tag" }`
    /// - anything else              → `Path(PathBuf::from(s))`
    pub fn parse(s: &str) -> Result<Self> {
        if let Some(rest) = s.strip_prefix("hf:") {
            parse_hf(rest)
        } else if let Some(rest) = s.strip_prefix("ollama:") {
            parse_ollama(rest)
        } else {
            Ok(ModelRef::Path(PathBuf::from(s)))
        }
    }
}

/// Parse everything after the `hf:` prefix.
fn parse_hf(rest: &str) -> Result<ModelRef> {
    if rest.is_empty() {
        return Err(Error::Other("hf: reference requires org/repo".into()));
    }
    // HF repo names cannot contain `:`, so the first `:` (if any) separates repo from file.
    match rest.find(':') {
        Some(colon) => {
            let repo = &rest[..colon];
            let file = &rest[colon + 1..];
            if !repo.contains('/') {
                return Err(Error::Other(format!(
                    "hf: repo must be org/repo, got: {repo}"
                )));
            }
            if file.is_empty() {
                return Err(Error::Other("hf: file name after ':' is empty".into()));
            }
            Ok(ModelRef::Hf {
                repo: repo.to_owned(),
                file: Some(file.to_owned()),
            })
        }
        None => {
            if !rest.contains('/') {
                return Err(Error::Other(format!(
                    "hf: repo must be org/repo, got: {rest}"
                )));
            }
            Ok(ModelRef::Hf {
                repo: rest.to_owned(),
                file: None,
            })
        }
    }
}

/// Parse everything after the `ollama:` prefix.
///
/// The name may include a namespace slash (e.g. `library/qwen`).  The last `:`
/// segment is treated as the tag only when it contains no `/`.
fn parse_ollama(rest: &str) -> Result<ModelRef> {
    if rest.is_empty() {
        return Err(Error::Other(
            "ollama: reference requires a model name".into(),
        ));
    }
    match rest.rfind(':') {
        Some(colon) => {
            let tag_candidate = &rest[colon + 1..];
            // Only split on this colon when the candidate is non-empty and has no '/'
            // (a '/' in the candidate means the ':' was part of some URL or unusual name).
            if !tag_candidate.is_empty() && !tag_candidate.contains('/') {
                let name = &rest[..colon];
                if name.is_empty() {
                    return Err(Error::Other("ollama: name is empty".into()));
                }
                Ok(ModelRef::Ollama {
                    name: name.to_owned(),
                    tag: tag_candidate.to_owned(),
                })
            } else {
                // Treat the whole string as the name; use "latest" tag.
                Ok(ModelRef::Ollama {
                    name: rest.to_owned(),
                    tag: "latest".to_owned(),
                })
            }
        }
        None => Ok(ModelRef::Ollama {
            name: rest.to_owned(),
            tag: "latest".to_owned(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_no_file() {
        let r = ModelRef::parse("hf:unsloth/diffusiongemma-26B-A4B-it-GGUF").unwrap();
        assert_eq!(
            r,
            ModelRef::Hf {
                repo: "unsloth/diffusiongemma-26B-A4B-it-GGUF".into(),
                file: None,
            }
        );
    }

    #[test]
    fn hf_with_file() {
        let r =
            ModelRef::parse("hf:unsloth/diffusiongemma-26B-A4B-it-GGUF:model-Q4_K_M.gguf").unwrap();
        assert_eq!(
            r,
            ModelRef::Hf {
                repo: "unsloth/diffusiongemma-26B-A4B-it-GGUF".into(),
                file: Some("model-Q4_K_M.gguf".into()),
            }
        );
    }

    #[test]
    fn hf_missing_slash_is_err() {
        assert!(ModelRef::parse("hf:noslash").is_err());
    }

    #[test]
    fn hf_empty_file_is_err() {
        assert!(ModelRef::parse("hf:org/repo:").is_err());
    }

    #[test]
    fn ollama_no_tag() {
        let r = ModelRef::parse("ollama:qwen2.5").unwrap();
        assert_eq!(
            r,
            ModelRef::Ollama {
                name: "qwen2.5".into(),
                tag: "latest".into(),
            }
        );
    }

    #[test]
    fn ollama_with_tag() {
        let r = ModelRef::parse("ollama:qwen2.5:7b").unwrap();
        assert_eq!(
            r,
            ModelRef::Ollama {
                name: "qwen2.5".into(),
                tag: "7b".into(),
            }
        );
    }

    #[test]
    fn ollama_namespaced_no_tag() {
        let r = ModelRef::parse("ollama:library/qwen2.5").unwrap();
        assert_eq!(
            r,
            ModelRef::Ollama {
                name: "library/qwen2.5".into(),
                tag: "latest".into(),
            }
        );
    }

    #[test]
    fn ollama_namespaced_with_tag() {
        let r = ModelRef::parse("ollama:library/qwen2.5:latest").unwrap();
        assert_eq!(
            r,
            ModelRef::Ollama {
                name: "library/qwen2.5".into(),
                tag: "latest".into(),
            }
        );
    }

    #[test]
    fn ollama_custom_namespace() {
        let r = ModelRef::parse("ollama:myorg/mymodel:v2").unwrap();
        assert_eq!(
            r,
            ModelRef::Ollama {
                name: "myorg/mymodel".into(),
                tag: "v2".into(),
            }
        );
    }

    #[test]
    fn plain_path() {
        let r = ModelRef::parse("/home/user/models/mymodel.gguf").unwrap();
        assert_eq!(
            r,
            ModelRef::Path(PathBuf::from("/home/user/models/mymodel.gguf"))
        );
    }

    #[test]
    fn plain_path_relative() {
        let r = ModelRef::parse("./mymodel.gguf").unwrap();
        assert_eq!(r, ModelRef::Path(PathBuf::from("./mymodel.gguf")));
    }
}
