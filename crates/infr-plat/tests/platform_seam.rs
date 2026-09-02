//! **The seam guard**: no crate but `infr-plat` names an operating-system library.
//!
//! This is what `docs/infr-plat.md` set out to make true and what keeps it true. The rule is
//! stated over Cargo manifests rather than over `cfg` attributes in source, and that choice is the
//! point: a `cfg(target_os)` is two different things wearing one syntax.
//!
//! - **OS plumbing** — one capability, a different spelling per platform: `flock` versus
//!   `LockFileEx`, `pread` versus `seek_read`. Writing it requires `libc` or `windows`, so a
//!   manifest edge is exactly where it shows up, and this test forbids that edge everywhere but
//!   the seam.
//! - **Backend availability** — `#[cfg(target_os = "macos")]` meaning "the Metal backend exists
//!   here". Those gates need no OS library at all; they are a feature-gating question that
//!   relocating into a platform crate would move rather than solve. Forbidding them here would
//!   force an exception list longer than the rule.
//!
//! A grep for `cfg(target_os)` cannot tell the two apart, so it would either fail on the second
//! kind or carry so many exceptions that it stopped meaning anything. This can fail, and does:
//! adding `libc.workspace = true` to any other crate turns it red.

use std::path::{Path, PathBuf};

/// Crates permitted to depend on `libc`, with the reason each is here. Anything else is a bug.
const LIBC_ALLOWED: &[(&str, &str)] = &[
    (
        "infr-plat",
        "the platform seam itself — this is its whole job",
    ),
    (
        // `p2p.rs` / `tp_sem.rs` share Vulkan device memory and semaphores across processes by
        // duplicating POSIX fds (`vkGetMemoryFdKHR`, `libc::dup`). The Windows arm is a stub —
        // `type RawFd = c_int`, with no `HANDLE` and no `VK_KHR_external_memory_win32` anywhere in
        // the tree — so hoisting it into the seam would launder a stub into an abstraction. It
        // stays put until someone implements the Win32 external-memory extension, which is feature
        // work rather than a move. See docs/backlog.md.
        "infr-vulkan",
        "cross-process Vulkan fd export; the Windows side is an unimplemented stub",
    ),
];

/// Crates permitted to depend on the `windows` crate.
const WINDOWS_ALLOWED: &[&str] = &["infr-plat"];

/// The OS libraries this rule is about. Not `metal`/`objc`: those are a GPU BACKEND's bindings,
/// and the hardware seam in this tree is `infr_core::backend::Backend`, not this crate.
const OS_LIBRARIES: &[&str] = &["libc", "windows"];

#[test]
fn only_the_seam_depends_on_an_os_library() {
    let crates = crates_dir();
    let mut manifests: Vec<PathBuf> = std::fs::read_dir(&crates)
        .expect("read crates/")
        .filter_map(|e| {
            let p = e.expect("crates/ entry").path().join("Cargo.toml");
            p.is_file().then_some(p)
        })
        .collect();
    manifests.sort();

    // A path glob that matches nothing is the classic vacuous guard, so prove the scan ran and saw
    // the crate the rule exists for.
    assert!(
        manifests.len() >= 10,
        "expected the whole workspace, found {} manifests under {}",
        manifests.len(),
        crates.display()
    );
    assert!(
        manifests
            .iter()
            .any(|m| m.ends_with("infr-plat/Cargo.toml")),
        "the scan did not reach infr-plat, so it proves nothing"
    );

    let mut offenders = Vec::new();
    let mut seam_edges = 0usize;
    for manifest in &manifests {
        let name = manifest
            .parent()
            .and_then(Path::file_name)
            .and_then(|n| n.to_str())
            .expect("crate directory name")
            .to_string();
        let text = std::fs::read_to_string(manifest).expect("read Cargo.toml");
        for lib in OS_LIBRARIES {
            if !declares_dependency(&text, lib) {
                continue;
            }
            let allowed = match *lib {
                "libc" => LIBC_ALLOWED.iter().any(|(c, _)| *c == name),
                "windows" => WINDOWS_ALLOWED.contains(&name.as_str()),
                _ => unreachable!("unlisted OS library {lib}"),
            };
            if name == "infr-plat" {
                seam_edges += 1;
            }
            if !allowed {
                offenders.push(format!(
                    "{name} depends on `{lib}`: platform code belongs in infr-plat \
                     (see docs/infr-plat.md), or add {name} to the allow-list here with a reason"
                ));
            }
        }
    }

    // The seam must hold BOTH edges. Without this the test passes just as well against a workspace
    // that has no platform code left to guard, which is not the same thing as one that is tidy.
    assert_eq!(
        seam_edges,
        OS_LIBRARIES.len(),
        "infr-plat should declare every OS library ({OS_LIBRARIES:?}); found {seam_edges}"
    );
    assert!(offenders.is_empty(), "{}", offenders.join("\n"));
}

/// Does `text` declare a dependency on `lib`, in any of the dependency tables?
///
/// Deliberately line-based rather than a TOML parse: the manifests use both
/// `libc.workspace = true` and `windows = { version = .. }`, in plain, `[target.'cfg(..)']` and
/// dev tables, and every one of those spellings is an edge this rule cares about.
fn declares_dependency(text: &str, lib: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim();
        line.starts_with(&format!("{lib} ")) || line.starts_with(&format!("{lib}."))
    })
}

/// The workspace's `crates/` directory, from this test's own manifest location.
fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/infr-plat has a parent")
        .to_path_buf()
}
