// INFR_PROFILE=1 at build time -> cfg(infr_profile) -> #[cfg_attr(infr_profile,
// infr_prof::instrument)] annotations become live and inject profiling spans into every fn.
// Default builds get NO cfg, the attributes vanish, and zero profiling code is compiled in.
// See docs/perf/playbook.md § "Build-time auto-instrumentation".
//
// It also writes the code fingerprint of the tolerance gate's reference cache
// (`src/tolerance/oracle.rs`, `docs/gpu-tolerance.md`): a SHA-256 over everything the CPU
// reference's output can depend on that is fixed at build time. A manually bumped constant fails
// silently when someone forgets it, and a git commit needs git at build time, so the content
// itself is hashed. `infr-vulkan` and `infr-metal` are left out on purpose: the CPU reference runs
// none of their code, and hashing them would turn every shader edit into a cold cache.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// The workspace crates whose code the CPU reference runs, by directory name.
const REFERENCE_CRATES: &[&str] = &[
    "infr-core",
    "infr-cpu",
    "infr-gguf",
    "infr-llama",
    "infr-chat",
    "infr-plat",
    "infr-prof",
    "infr-prof-rt",
];

/// The environment the compiler's output depends on, beyond the sources and every `CARGO_CFG_*`
/// variable (hashed all, by name): the target, the profile and its settings, and the flags. The
/// `CARGO_CFG_*` set is what tells two builds of one profile apart where the code itself
/// branches: debug assertions, overflow checks, the panic strategy, the target features.
const BUILD_ENV: &[&str] = &[
    "TARGET",
    "HOST",
    "PROFILE",
    "OPT_LEVEL",
    "DEBUG",
    "CARGO_ENCODED_RUSTFLAGS",
    "INFR_PROFILE",
];

fn main() {
    println!("cargo:rerun-if-env-changed=INFR_PROFILE");
    println!("cargo:rustc-check-cfg=cfg(infr_profile)");
    if std::env::var("INFR_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0") {
        println!("cargo:rustc-cfg=infr_profile");
    }
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    // An empty value is "no fingerprint": a build without its sibling sources (a published crate)
    // has nothing to hash, and the cache refuses to run on it.
    let fingerprint = fingerprint().unwrap_or_default();
    println!("cargo:rustc-env=REFCACHE_CODE_FINGERPRINT={fingerprint}");
    // For the test that holds this list to the crates infr-llama actually depends on.
    println!(
        "cargo:rustc-env=REFCACHE_CODE_CRATES={}",
        REFERENCE_CRATES.join(",")
    );
}

/// The fingerprint as lowercase hex, or `None` when a reference crate's sources, the workspace
/// manifest or the lockfile is not there to hash.
fn fingerprint() -> Option<String> {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR")?);
    let crates = manifest.parent()?;
    let root = crates.parent()?;
    let mut h = Sha256::new();
    // Every field is written with its length first, so no two different inputs hash the same
    // byte stream.
    let mut field = |tag: &str, bytes: &[u8]| {
        for part in [tag.as_bytes(), bytes] {
            h.update((part.len() as u64).to_le_bytes());
            h.update(part);
        }
    };
    for name in ["Cargo.toml", "Cargo.lock"] {
        let path = root.join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        field(name, &std::fs::read(&path).ok()?);
    }
    for krate in REFERENCE_CRATES {
        let dir = crates.join(krate);
        for name in ["Cargo.toml", "build.rs"] {
            let path = dir.join(name);
            println!("cargo:rerun-if-changed={}", path.display());
            if let Ok(bytes) = std::fs::read(&path) {
                field(&format!("{krate}/{name}"), &bytes);
            } else if name == "Cargo.toml" {
                return None;
            }
        }
        let src = dir.join("src");
        println!("cargo:rerun-if-changed={}", src.display());
        let mut files = Vec::new();
        walk(&src, &mut files).ok()?;
        if files.is_empty() {
            return None;
        }
        files.sort();
        for (file, linked_dir) in files {
            let rel = file.strip_prefix(&dir).ok()?;
            // `/` on every platform, so one tree hashes the same wherever it is checked out.
            let rel: Vec<_> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect();
            let rel = format!("{krate}/{}", rel.join("/"));
            match linked_dir {
                Some(target) => field(&format!("{rel} -> dir"), target.as_bytes()),
                None => field(&rel, &std::fs::read(&file).ok()?),
            }
        }
    }
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = std::process::Command::new(rustc).arg("-vV").output().ok()?;
    if !version.status.success() {
        return None;
    }
    field("rustc -vV", &version.stdout);
    for var in BUILD_ENV {
        field(var, std::env::var(var).unwrap_or_default().as_bytes());
    }
    let mut cfgs: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| k.starts_with("CARGO_CFG_"))
        .collect();
    cfgs.sort();
    for (k, v) in &cfgs {
        field(k, v.as_bytes());
    }
    let mut features: Vec<String> = std::env::vars()
        .filter_map(|(k, _)| k.strip_prefix("CARGO_FEATURE_").map(str::to_string))
        .collect();
    features.sort();
    field("features", features.join(",").as_bytes());
    Some(
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
    )
}

/// Every file under `dir`, recursively, each with `None`; and every symlink to a directory with
/// the text of its target, which is not followed: a link loop would never end, and a tree that
/// links a directory in is hashed by where the link points, not by what is there today. A
/// symlink to a file is read as the file.
fn walk(dir: &Path, out: &mut Vec<(PathBuf, Option<String>)>) -> std::io::Result<()> {
    for e in std::fs::read_dir(dir)? {
        let path = e?.path();
        let is_link = std::fs::symlink_metadata(&path)?.file_type().is_symlink();
        if is_link && path.is_dir() {
            let target = std::fs::read_link(&path)?;
            out.push((path, Some(target.to_string_lossy().into_owned())));
        } else if path.is_dir() {
            walk(&path, out)?;
        } else {
            out.push((path, None));
        }
    }
    Ok(())
}
