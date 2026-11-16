//! Running llama.cpp's `llama-debug` as an external oracle over a synthetic GGUF: the tests that
//! regenerate a checked-in llama.cpp fixture call these, with the binary named by an environment
//! variable, so nothing here runs in an ordinary test pass.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Run `exe` with `args` and no console window, panicking on a failure; returns its stdout and
/// stderr together (llama.cpp logs to stderr).
pub fn run_headless(exe: &str, args: &[&OsStr]) -> String {
    let mut cmd = std::process::Command::new(exe);
    cmd.args(args).stdin(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        /// `CREATE_NO_WINDOW` from the Win32 process-creation flags: a console program started
        /// from a test runner opens no window of its own. Other platforms have no such window.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let out = cmd.output().unwrap_or_else(|e| panic!("run {exe}: {e}"));
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success(),
        "{exe} failed ({}):\n{text}",
        out.status
    );
    text
}

/// The `version:` line `exe --version` prints, which names the build and its commit.
pub fn llama_cpp_version(exe: &str) -> String {
    let text = run_headless(exe, &["--version".as_ref()]);
    text.lines()
        .find(|l| l.starts_with("version:"))
        .unwrap_or_else(|| panic!("no version line from {exe} --version:\n{text}"))
        .to_string()
}

/// Run `llama-debug` (`exe`) on `gguf` with `prompt` and `--save-logits`, plus `extra` arguments,
/// and return the token ids it tokenized the prompt to and its logits at the last position. The
/// files it writes go to a directory beside `gguf` and are removed before returning.
pub fn llama_debug_logits(
    exe: &str,
    gguf: &Path,
    prompt: &str,
    extra: &[&OsStr],
) -> (Vec<u32>, Vec<f32>) {
    let out_dir = gguf.with_extension("llama-debug");
    let mut args: Vec<&OsStr> = vec![
        "-m".as_ref(),
        gguf.as_os_str(),
        "-p".as_ref(),
        prompt.as_ref(),
        "--save-logits".as_ref(),
        "--logits-output-dir".as_ref(),
        out_dir.as_os_str(),
    ];
    args.extend_from_slice(extra);
    run_headless(exe, &args);
    let stem = gguf
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("a UTF-8 file stem");
    let read = |suffix: &str| {
        let path = out_dir.join(format!("llamacpp-{stem}{suffix}"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert_eq!(bytes.len() % 4, 0, "{} is not 4-byte words", path.display());
        bytes
    };
    let logits: Vec<f32> = read(".bin")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();
    let tokens: Vec<u32> = read("-tokens.bin")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| u32::from_le_bytes(c))
        .collect();
    for suffix in [".bin", ".txt", "-prompt.txt", "-tokens.bin"] {
        std::fs::remove_file(out_dir.join(format!("llamacpp-{stem}{suffix}")))
            .expect("remove a llama-debug output");
    }
    std::fs::remove_dir(&out_dir).expect("remove the llama-debug output directory");
    (tokens, logits)
}

/// The environment variable naming the directory a fixture regenerator writes into. Without it a
/// regenerator only compares what llama.cpp computes now with the checked-in fixture, so running
/// every ignored test (`--run-ignored all`) with a llama.cpp binary configured never rewrites a
/// file in the tree.
pub const FIXTURE_OUT_ENV: &str = "INFR_LLAMA_FIXTURE_OUT";

/// The lines of a llama.cpp fixture that carry its data: every line that is not a `#` comment, and
/// the `# tokens:`, `# position:` and `# gguf:` lines that frame it. The other comment lines (the
/// build, the commands, measurements made at generation time) are left out of a comparison.
pub fn fixture_data_lines(text: &str) -> Vec<&str> {
    text.lines()
        .filter(|l| {
            !l.starts_with('#')
                || ["# tokens:", "# position:", "# gguf:"]
                    .iter()
                    .any(|p| l.starts_with(p))
        })
        .collect()
}

/// Deliver a regenerated fixture: write `text` into the directory [`FIXTURE_OUT_ENV`] names (under
/// `checked_in`'s file name) and return the path written, or, when it is unset, panic unless
/// `text` carries the same data ([`fixture_data_lines`]) as the checked-in file and return `None`.
pub fn write_or_compare_fixture(checked_in: &Path, text: &str) -> Option<PathBuf> {
    let name = checked_in.file_name().expect("a fixture file name");
    if let Some(dir) = std::env::var_os(FIXTURE_OUT_ENV) {
        let path = Path::new(&dir).join(name);
        std::fs::write(&path, text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        return Some(path);
    }
    let old = std::fs::read_to_string(checked_in)
        .unwrap_or_else(|e| panic!("read {}: {e}", checked_in.display()));
    assert!(
        fixture_data_lines(&old) == fixture_data_lines(text),
        "llama.cpp no longer reproduces {}: set {FIXTURE_OUT_ENV} to a directory to write the new \
         data there, and review it before copying it over the checked-in file",
        checked_in.display()
    );
    None
}
