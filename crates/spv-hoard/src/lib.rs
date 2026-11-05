//! Content-addressed cache around `glslc` for build scripts that compile many shader variants.

use std::ffi::OsString;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::atomic::{AtomicUsize, Ordering};

/// One shader variant to compile.
#[derive(Clone, Debug)]
pub struct Job {
    /// Shader source, e.g. `shaders/exl3_dot.comp`.
    pub src: PathBuf,
    /// Extra `glslc` arguments that change the output, e.g. `-DK_SLICES=16`.
    pub defines: Vec<String>,
    /// Destination of the compiled SPIR-V.
    pub out: PathBuf,
}

impl Job {
    pub fn new(src: impl Into<PathBuf>, out: impl Into<PathBuf>) -> Self {
        Self {
            src: src.into(),
            defines: Vec::new(),
            out: out.into(),
        }
    }

    pub fn define(mut self, define: impl Into<String>) -> Self {
        self.defines.push(define.into());
        self
    }
}

/// Caches each compiled output under `<key>.spv`, keyed on everything that determines the bytes.
pub struct Cache {
    glslc: PathBuf,
    cache_dir: PathBuf,
    flags: Vec<String>,
    include_dirs: Vec<PathBuf>,
    tool: Vec<u8>,
}

impl Cache {
    /// `glslc` is the compiler to run; its `--version` output is part of every key.
    pub fn new(glslc: impl Into<PathBuf>, cache_dir: impl Into<PathBuf>) -> io::Result<Self> {
        let glslc = glslc.into();
        let tool = Command::new(&glslc).arg("--version").output()?.stdout;
        let cache_dir = cache_dir.into();
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self {
            glslc,
            cache_dir,
            flags: Vec::new(),
            include_dirs: Vec::new(),
            tool,
        })
    }

    /// Arguments shared by every job, placed before the per-job defines.
    pub fn flags(mut self, flags: impl IntoIterator<Item = String>) -> Self {
        self.flags.extend(flags);
        self
    }

    /// An `-I` directory, also searched when hashing `#include` directives.
    pub fn include(mut self, dir: impl Into<PathBuf>) -> Self {
        self.include_dirs.push(dir.into());
        self
    }

    pub fn compile(&self, job: &Job) -> Result<(), Error> {
        let key = key(&self.tool, &self.flags, &self.include_dirs, job);
        let cached = self.cache_dir.join(format!("{key:016x}.spv"));
        if !cached.is_file() {
            let tmp = self.temp_path(key);
            let mut command = Command::new(&self.glslc);
            command.args(&self.flags).args(&job.defines);
            for dir in &self.include_dirs {
                let mut arg = OsString::from("-I");
                arg.push(dir);
                command.arg(arg);
            }
            let output = command.arg(&job.src).arg("-o").arg(&tmp).output()?;
            if !output.status.success() {
                let _ = std::fs::remove_file(&tmp);
                return Err(Error::Glslc {
                    src: job.src.clone(),
                    status: output.status,
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                });
            }
            // Equal keys can race each other; whoever publishes first wins.
            if std::fs::rename(&tmp, &cached).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        }
        std::fs::copy(&cached, &job.out)?;
        Ok(())
    }

    /// Compile every job, keeping at most `jobs_limit` `glslc` processes running.
    pub fn compile_all(&self, jobs: &[Job], jobs_limit: usize) -> Result<(), Error> {
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..jobs_limit.max(1).min(jobs.len()) {
                handles.push(scope.spawn(|| -> Result<(), Error> {
                    while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        self.compile(job)?;
                    }
                    Ok(())
                }));
            }
            let mut first = None;
            for handle in handles {
                match handle.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => {
                        first.get_or_insert(err);
                    }
                    Err(panic) => std::panic::resume_unwind(panic),
                }
            }
            match first {
                Some(err) => Err(err),
                None => Ok(()),
            }
        })
    }

    fn temp_path(&self, key: u64) -> PathBuf {
        static TEMP: AtomicUsize = AtomicUsize::new(0);
        self.cache_dir.join(format!(
            ".{key:016x}.{}.{}.tmp",
            std::process::id(),
            TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }
}

/// The shared cache directory for a build script: `<target>/shader-cache/v2` for `OUT_DIR` at
/// `<target>/<profile>/build/<pkg>-<hash>/out`.
pub fn default_cache_dir(out_dir: &Path) -> PathBuf {
    out_dir
        .ancestors()
        .nth(4)
        .map(|target| target.join("shader-cache").join("v2"))
        .unwrap_or_else(|| out_dir.join("spv-cache"))
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Glslc {
        src: PathBuf,
        status: ExitStatus,
        stderr: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(err) => write!(f, "{err}"),
            Error::Glslc {
                src,
                status,
                stderr,
            } => write!(f, "glslc failed for {} ({status}): {stderr}", src.display()),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn key(tool: &[u8], flags: &[String], include_dirs: &[PathBuf], job: &Job) -> u64 {
    let mut h = FNV_OFFSET;
    hash(&mut h, b"spv-hoard-v1");
    hash(&mut h, tool);
    for flag in flags {
        hash(&mut h, flag.as_bytes());
        hash(&mut h, b"\0");
    }
    for define in &job.defines {
        hash(&mut h, define.as_bytes());
        hash(&mut h, b"\0");
    }
    hash_file(&mut h, &job.src);
    let mut seen = vec![job.src.clone()];
    hash_includes(&mut h, &job.src, include_dirs, &mut seen);
    h
}

fn hash(h: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *h ^= byte as u64;
        *h = h.wrapping_mul(FNV_PRIME);
    }
}

fn hash_file(h: &mut u64, path: &Path) {
    match std::fs::read(path) {
        Ok(bytes) => hash(h, &bytes),
        Err(_) => hash(h, b"\0"),
    }
}

/// Hash every `#include "..."` reachable from `src`, searching the source's directory first.
fn hash_includes(h: &mut u64, src: &Path, include_dirs: &[PathBuf], seen: &mut Vec<PathBuf>) {
    let Ok(text) = std::fs::read_to_string(src) else {
        return;
    };
    let dir = src.parent().unwrap_or_else(|| Path::new(""));
    for line in text.lines() {
        let Some(rest) = line.trim_start().strip_prefix("#include") else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('"') else {
            continue;
        };
        let Some((name, _)) = rest.split_once('"') else {
            continue;
        };
        let found = std::iter::once(dir.join(name))
            .chain(include_dirs.iter().map(|include| include.join(name)))
            .find(|candidate| candidate.is_file());
        let Some(path) = found else {
            continue;
        };
        if seen.contains(&path) {
            continue;
        }
        seen.push(path.clone());
        hash_file(h, &path);
        hash_includes(h, &path, include_dirs, seen);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("spv-hoard-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn key_of(src: &Path, defines: &[&str], include_dirs: &[PathBuf]) -> u64 {
        let job = Job {
            src: src.to_path_buf(),
            defines: defines.iter().map(|define| define.to_string()).collect(),
            out: PathBuf::new(),
        };
        key(b"test-tool", &["-O".to_string()], include_dirs, &job)
    }

    #[test]
    fn key_is_stable_for_equal_inputs() {
        let dir = temp_dir("stable");
        let src = dir.join("kernel.comp");
        std::fs::write(&src, "void main() {}\n").unwrap();
        assert_eq!(key_of(&src, &["-DA"], &[]), key_of(&src, &["-DA"], &[]));
    }

    #[test]
    fn defines_change_the_key() {
        let dir = temp_dir("defines");
        let src = dir.join("kernel.comp");
        std::fs::write(&src, "void main() {}\n").unwrap();
        assert_ne!(key_of(&src, &["-DA"], &[]), key_of(&src, &["-DB"], &[]));
        assert_ne!(
            key_of(&src, &["-DA"], &[]),
            key_of(&src, &["-DA", "-DB"], &[])
        );
    }

    #[test]
    fn source_edits_change_the_key() {
        let dir = temp_dir("source");
        let src = dir.join("kernel.comp");
        std::fs::write(&src, "void main() {}\n").unwrap();
        let before = key_of(&src, &[], &[]);
        std::fs::write(&src, "void main() { int x; }\n").unwrap();
        assert_ne!(before, key_of(&src, &[], &[]));
    }

    #[test]
    fn include_edits_change_the_key() {
        let dir = temp_dir("include");
        std::fs::write(dir.join("common.glsl"), "float x;\n").unwrap();
        std::fs::write(
            dir.join("kernel.comp"),
            "#include \"common.glsl\"\nvoid main() {}\n",
        )
        .unwrap();
        let src = dir.join("kernel.comp");
        let before = key_of(&src, &[], &[]);
        std::fs::write(dir.join("common.glsl"), "float y;\n").unwrap();
        assert_ne!(before, key_of(&src, &[], &[]));
    }

    #[test]
    fn nested_includes_are_hashed() {
        let dir = temp_dir("nested");
        std::fs::write(dir.join("inner.glsl"), "float inner;\n").unwrap();
        std::fs::write(dir.join("outer.glsl"), "#include \"inner.glsl\"\n").unwrap();
        std::fs::write(dir.join("kernel.comp"), "#include \"outer.glsl\"\n").unwrap();
        let src = dir.join("kernel.comp");
        let before = key_of(&src, &[], &[]);
        std::fs::write(dir.join("inner.glsl"), "float changed;\n").unwrap();
        assert_ne!(before, key_of(&src, &[], &[]));
    }

    #[test]
    fn include_dirs_are_searched() {
        let dir = temp_dir("include-dir");
        let headers = dir.join("generated");
        std::fs::create_dir_all(&headers).unwrap();
        std::fs::write(headers.join("grids.glsl"), "float grid;\n").unwrap();
        std::fs::write(dir.join("kernel.comp"), "#include \"grids.glsl\"\n").unwrap();
        let src = dir.join("kernel.comp");
        let before = key_of(&src, &[], std::slice::from_ref(&headers));
        std::fs::write(headers.join("grids.glsl"), "float changed;\n").unwrap();
        assert_ne!(before, key_of(&src, &[], std::slice::from_ref(&headers)));
    }
}
