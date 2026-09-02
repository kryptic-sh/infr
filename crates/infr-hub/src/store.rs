//! infr's on-disk model store — the **standard HuggingFace Hub cache** (`~/.cache/huggingface/hub`),
//! shared with llama.cpp, `huggingface_hub`, and `transformers`. So `infr run hf:org/repo:Q4_K_M` and
//! `llama-cli -hf org/repo:Q4_K_M` hit the same files — one download, no duplication:
//!
//! ```text
//! <hub>/models--<org>--<repo>/
//!   refs/main                       -> <commit>
//!   blobs/<sha256>                     the file bytes (content-addressed; bare hex, no prefix)
//!   snapshots/<commit>/<file.gguf>  -> ../../blobs/<sha256>   (symlink with the real filename)
//!   snapshots/<commit>/<dir>/<file.gguf> -> ../../../blobs/<sha256>
//! ```
//!
//! A file's `rfilename` may name a SUBDIRECTORY (unsloth ships its Dynamic quants as
//! `UD-Q4_K_XL/<model>.gguf`), so the snapshot mirrors the repo's tree and the link target carries
//! one `..` per level — exactly what `huggingface_hub` writes, which is what keeps the cache shared.

use crate::model_ref::ModelRef;
use infr_core::error::{Error, Result};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// The `-NNNNN-of-MMMMM.gguf` expansion lives in [`infr_core::gguf_split`], not here: the GGUF
/// loader follows the same convention to find a model's other shards at open, and two parsers are
/// two conventions that can disagree about which files a model is. Re-exported at its old path so
/// the download side (`pull.rs`) names it where it uses it.
pub(crate) use infr_core::gguf_split::shard_set;

/// Default quant when an `hf:` ref gives only `org/repo` (matches llama.cpp's `-hf`).
pub(crate) const DEFAULT_QUANT: &str = "Q4_K_M";

/// The HuggingFace Hub cache.
pub struct Store {
    pub hub: PathBuf,
}

/// `huggingface_hub`'s own precedence, as a pure function of the four inputs.
///
/// Takes the environment's values rather than reading them, so the chain is testable without
/// `set_var` racing every other thread in the test binary — and so the same code proves the answer
/// is identical on all three platforms, which is the entire point of the layout.
///
/// An empty variable is treated as unset: `HF_HOME=` from a cleared shell export would otherwise
/// resolve the cache to the relative path `hub`.
fn hub_dir(
    hf_hub_cache: Option<std::ffi::OsString>,
    huggingface_hub_cache: Option<std::ffi::OsString>,
    hf_home: Option<std::ffi::OsString>,
    cache_home: Option<PathBuf>,
) -> Option<PathBuf> {
    let set = |v: Option<std::ffi::OsString>| v.filter(|s| !s.is_empty()).map(PathBuf::from);
    if let Some(p) = set(hf_hub_cache).or_else(|| set(huggingface_hub_cache)) {
        return Some(p);
    }
    if let Some(home) = set(hf_home) {
        return Some(home.join("hub"));
    }
    Some(cache_home?.join("huggingface").join("hub"))
}

impl Store {
    /// Locate the Hugging Face hub cache, agreeing with `huggingface_hub` on every platform:
    /// `$HF_HUB_CACHE`, else `$HUGGINGFACE_HUB_CACHE`, else `$HF_HOME/hub`, else
    /// `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`.
    ///
    /// **Agreeing matters more than being conventional.** The layout is not infr's to choose: the
    /// `models--org--repo/snapshots/<commit>/<file>` tree is `huggingface_hub`'s, llama.cpp reads
    /// the same one, and a user who ran `hf download` must not have to fetch the same 30 GB again.
    /// So this deliberately does NOT use each platform's native cache directory —
    /// `huggingface_hub` reads `$XDG_CACHE_HOME` and falls back to `~/.cache` on Windows and macOS
    /// too, with no `%LOCALAPPDATA%` or `~/Library/Caches` arm (verified against
    /// `huggingface_hub/constants.py`).
    ///
    /// A cache an older infr left in the OS-native location is still honoured when the standard
    /// one does not exist yet — see [`Self::legacy_hub_dir`].
    pub fn discover() -> Result<Self> {
        let hub = hub_dir(
            std::env::var_os("HF_HUB_CACHE"),
            std::env::var_os("HUGGINGFACE_HUB_CACHE"),
            std::env::var_os("HF_HOME"),
            infr_plat::paths::cache_home(),
        )
        .ok_or_else(|| Error::Other("cannot determine cache directory".into()))?;
        Ok(Store {
            hub: Self::legacy_hub_dir(hub),
        })
    }

    /// Redirect to a cache an OLDER infr wrote to the OS-native directory, but only while the
    /// standard location does not exist.
    ///
    /// Before this agreed with `huggingface_hub`, the fallback was `dirs::cache_dir()` — the same
    /// path on Linux, but `~/Library/Caches` on macOS and `%LOCALAPPDATA%` on Windows. Switching
    /// without looking would leave those models on disk, unseen, and re-download them. The
    /// redirect is one-way and conditional, so a fresh install and anyone already on the standard
    /// path are unaffected, and it disappears by itself once the standard directory exists.
    fn legacy_hub_dir(standard: PathBuf) -> PathBuf {
        if standard.exists() {
            return standard;
        }
        let Some(legacy) = infr_plat::paths::os_cache_home()
            .map(|b| b.join("huggingface").join("hub"))
            .filter(|p| *p != standard && p.is_dir())
        else {
            return standard;
        };
        tracing::warn!(
            "using the Hugging Face cache at {} left by an older infr; the standard location is \
             {}. Move it there (or set HF_HOME) — this fallback stops applying as soon as the \
             standard directory exists.",
            legacy.display(),
            standard.display(),
        );
        legacy
    }

    /// `<hub>/models--<org>--<repo>` — the HF Hub repo dir (HF replaces `/` with `--`).
    pub fn repo_dir(&self, repo: &str) -> PathBuf {
        self.hub
            .join(format!("models--{}", repo.replace('/', "--")))
    }

    /// Resolve a cached GGUF for `repo` selecting `sel` (a quant like `Q4_K_M`, or an explicit
    /// `*.gguf` filename; `None` → [`DEFAULT_QUANT`]). Uses the SAME selection routine as the
    /// download path ([`pick_gguf`]) so a repo that downloaded once is judged cached on the next
    /// run (a divergence otherwise re-pulls multi-GB every invocation). Snapshots are tried in
    /// [`refs/main`][Self::ordered_snapshots] order first. A sharded GGUF only counts as cached when
    /// the WHOLE shard set is present (a lone shard 1 fails at load), and its blobs must not be
    /// dangling (garbage-collected).
    pub fn resolve_repo(&self, repo: &str, sel: Option<&str>) -> Option<PathBuf> {
        for snap in self.ordered_snapshots(repo) {
            let names = snapshot_ggufs(&snap);
            let Some(chosen) = pick_gguf(&names, sel) else {
                continue;
            };
            // Every shard of the chosen file must be present (and non-dangling) to be usable; hand
            // back the canonical first shard (what a GGUF loader opens), matching the download path.
            let set = shard_set(&chosen);
            if set.iter().all(|f| snap.join(f).exists()) {
                return Some(snap.join(&set[0]));
            }
        }
        None
    }

    /// Snapshot dirs for `repo`, the one named by `refs/main` FIRST (when present), then the rest.
    /// HF leaves stale snapshots in place across commits, so preferring `refs/main` avoids returning
    /// an arbitrary older snapshot for the current model.
    fn ordered_snapshots(&self, repo: &str) -> Vec<PathBuf> {
        let dir = self.repo_dir(repo);
        let snaps = dir.join("snapshots");
        let main = fs::read_to_string(dir.join("refs").join("main"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let mut out = Vec::new();
        if let Some(commit) = &main {
            let p = snaps.join(commit);
            if p.is_dir() {
                out.push(p);
            }
        }
        for e in fs::read_dir(&snaps).into_iter().flatten().flatten() {
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            let is_main = main
                .as_deref()
                .zip(p.file_name().and_then(|n| n.to_str()))
                .is_some_and(|(c, n)| c == n);
            if !is_main {
                out.push(p);
            }
        }
        out
    }

    /// If the referenced model already exists locally, return its GGUF path.
    pub fn resolve(&self, r: &ModelRef) -> Result<Option<PathBuf>> {
        Ok(match r {
            ModelRef::Path(p) => p.exists().then(|| p.clone()),
            ModelRef::Repo { repo, sel } => self.resolve_repo(repo, sel.as_deref()),
        })
    }
}

/// How many directory levels below a snapshot root are descended when listing its GGUFs.
///
/// A snapshot mirrors the repo's own tree, and real GGUF repos put weights either at the root or one
/// directory down (unsloth's `UD-Q4_K_XL/`, `BF16/`, `Q4_K_M/` Dynamic-quant dirs; `original/` on
/// some conversions). Three levels covers those with room to spare while keeping the walk cheap and
/// bounded — this runs on the `infr run` fast path, once per snapshot, and must not turn into a full
/// traversal of whatever an arbitrary repo happens to ship.
const MAX_SNAPSHOT_DEPTH: usize = 3;

/// Every `.gguf` in `snap`, named RELATIVE to the snapshot root (`UD-Q4_K_XL/model.gguf`).
///
/// Snapshot-relative — not bare basenames — because these names are fed to [`pick_gguf`], which must
/// return exactly what the download path's `pick_gguf` selected from the HF API's `rfilename`s, and
/// they are re-joined onto `snap` to test for existence. A NON-recursive listing here was the bug
/// that made a subdirectory GGUF invisible to the cache check, so `infr run` re-downloaded multiple
/// GB on every single invocation.
fn snapshot_ggufs(snap: &Path) -> Vec<String> {
    let mut out = Vec::new();
    collect_ggufs(snap, "", 0, &mut out);
    out
}

fn collect_ggufs(dir: &Path, prefix: &str, depth: usize, out: &mut Vec<String>) {
    for e in fs::read_dir(dir).into_iter().flatten().flatten() {
        // `DirEntry::file_type` does NOT follow symlinks, which is what we want twice over: a
        // snapshot's GGUF entries are symlinks into `blobs/` (still files to us, and a dangling one
        // is caught later by the `exists()` shard check), and only a REAL directory is descended, so
        // a symlinked directory cannot send this walk round a cycle.
        let Ok(ft) = e.file_type() else { continue };
        let name = e.file_name().to_string_lossy().into_owned();
        let rel = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        if ft.is_dir() {
            if depth < MAX_SNAPSHOT_DEPTH {
                collect_ggufs(&e.path(), &rel, depth + 1, out);
            }
            continue;
        }
        if rel.to_lowercase().ends_with(".gguf") {
            out.push(rel);
        }
    }
}

/// The final path component of an `rfilename` (`UD-Q4_K_XL/m.gguf` → `m.gguf`); the whole string
/// when it names no directory.
fn base_name(s: &str) -> &str {
    s.rsplit('/').next().unwrap_or(s)
}

/// How well a cached `.gguf` filename matches a selector.
enum Match {
    /// An explicit filename matched exactly, or the quant is the file's suffix (`…-Q4_K_M.gguf`).
    Exact,
    /// The quant appears somewhere in the name (weaker; e.g. an oddly-named or split file).
    Loose,
    No,
}

/// Pick the best `.gguf` from `names` for selector `sel` (quant or filename; `None` → default quant).
/// The SINGLE selection routine shared by the download path ([`repo_info`][crate::pull]) and the
/// cache-hit path ([`Store::resolve_repo`]) — they must agree or a downloaded model reads as "not
/// cached" and re-pulls every run.
///
/// Exact match wins; else a loose (substring) match; else — only for the *default* quant (no explicit
/// selector) — a fallback file (llama.cpp's "fall back to the first file"). The fallback NEVER picks
/// an `mmproj*` sidecar (a vision projector, not the LM weights) and prefers a real quant over an
/// `F16`/`F32`/`BF16` master when both are present.
pub(crate) fn pick_gguf(names: &[String], sel: Option<&str>) -> Option<String> {
    let want = sel.unwrap_or(DEFAULT_QUANT);
    let mut loose: Option<&String> = None;
    let mut fallback_quant: Option<&String> = None; // a non-mmproj, non-float-master gguf
    let mut fallback_any: Option<&String> = None; // any non-mmproj gguf (incl. F16 masters)
    for n in names {
        if !n.to_lowercase().ends_with(".gguf") {
            continue;
        }
        match gguf_match(n, want) {
            Match::Exact => return Some(n.clone()),
            Match::Loose => loose = loose.or(Some(n)),
            Match::No => {}
        }
        if is_mmproj(n) {
            continue; // never a weights fallback
        }
        fallback_any = fallback_any.or(Some(n));
        if !is_float_master(n) {
            fallback_quant = fallback_quant.or(Some(n));
        }
    }
    if let Some(l) = loose {
        return Some(l.clone());
    }
    if sel.is_none() {
        return fallback_quant.or(fallback_any).cloned();
    }
    None
}

/// True for an `mmproj` sidecar (multimodal projector: `mmproj-model-f16.gguf`, `mmproj-*.gguf`) —
/// never the language-model weights, so it must not be served as the model.
fn is_mmproj(name: &str) -> bool {
    name.to_lowercase()
        .rsplit('/')
        .next()
        .unwrap_or("")
        .starts_with("mmproj")
}

/// True when the file is a full-precision master (`F16`/`F32`/`BF16` on a token boundary) rather than
/// a real quant — deprioritised in the default fallback (a quant is what `infr run` wants).
fn is_float_master(name: &str) -> bool {
    matches!(gguf_match(name, "f16"), Match::Exact | Match::Loose)
        || matches!(gguf_match(name, "f32"), Match::Exact | Match::Loose)
        || matches!(gguf_match(name, "bf16"), Match::Exact | Match::Loose)
}

/// Match a cached `.gguf` filename against a selector (an explicit `*.gguf` name, or a quant).
///
/// The quant must sit on **token boundaries** in the filename, else neighbouring formats collide:
/// `…-PQ2_0.gguf` / `…-TQ2_0.gguf` / `…-Q2_0_g64.gguf` are all DIFFERENT weight layouts from `Q2_0`
/// and must never satisfy a `Q2_0` selector. A token starts after `-`/`_`/`.` (or the name start) and
/// ends before `-`/`.` (or the stem end) — `_` does not end it, since quant names embed it (`Q4_K_M`).
///
/// An explicit `*.gguf` selector matches either the full `rfilename` OR its final path component, so
/// `org/repo:Qwen3-30B-UD-Q4_K_XL.gguf` still selects the API's
/// `UD-Q4_K_XL/Qwen3-30B-UD-Q4_K_XL.gguf` — a user reading the repo's file list types the name they
/// see, not the directory it lives in, and before this an explicit filename could never match a
/// subdirectory GGUF at all.
fn gguf_match(fname: &str, sel: &str) -> Match {
    if sel.to_lowercase().ends_with(".gguf") {
        return if fname.eq_ignore_ascii_case(sel)
            || base_name(fname).eq_ignore_ascii_case(base_name(sel))
        {
            Match::Exact
        } else {
            Match::No
        };
    }
    let (f, q) = (fname.to_lowercase(), sel.to_lowercase());
    let Some(stem) = f.strip_suffix(".gguf") else {
        return Match::No;
    };
    let starts_token = |i: usize| i == 0 || matches!(stem.as_bytes()[i - 1], b'-' | b'_' | b'.');
    let ends_token = |i: usize| i == stem.len() || matches!(stem.as_bytes()[i], b'-' | b'.');

    let mut loose = false;
    for (i, _) in stem.match_indices(q.as_str()) {
        let end = i + q.len();
        if !starts_token(i) || !ends_token(end) {
            continue;
        }
        if end == stem.len() {
            return Match::Exact; // the quant IS the trailing token: `…-Q4_K_M.gguf`
        }
        loose = true; // a delimited hit elsewhere: split shards, `…-Q4_K_M-00001-of-00003.gguf`
    }
    if loose {
        Match::Loose
    } else {
        Match::No
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fake HF Hub repo dir: blobs/<sha> + snapshots/<commit>/<file> -> blob + refs/main.
    ///
    /// `file` may name a subdirectory (`UD-Q4_K_XL/model.gguf`), exactly as an HF `rfilename` can:
    /// the parent is created and the link target is built by the SAME `blob_link_target` the real
    /// download path uses, so these fixtures stay byte-identical to a `huggingface_hub` tree and a
    /// wrong `..` depth shows up here as a dangling link.
    fn fake_hf(hub: &std::path::Path, repo: &str, commit: &str, file: &str, sha: &str) {
        let dir = hub.join(format!("models--{}", repo.replace('/', "--")));
        let blobs = dir.join("blobs");
        let snap = dir.join("snapshots").join(commit);
        fs::create_dir_all(&blobs).unwrap();
        fs::create_dir_all(&snap).unwrap();
        fs::write(blobs.join(sha), b"fake gguf bytes").unwrap();
        let link = snap.join(file);
        fs::create_dir_all(link.parent().unwrap()).unwrap();
        infr_plat::link::link_blob(crate::pull::blob_link_target(file, sha), &link).unwrap();
        fs::create_dir_all(dir.join("refs")).unwrap();
        fs::write(dir.join("refs").join("main"), commit).unwrap();
    }

    fn store_at(hub: PathBuf) -> Store {
        Store { hub }
    }

    #[test]
    fn resolve_hf_default_quant() {
        let tmp = tempfile::tempdir().unwrap();
        fake_hf(
            tmp.path(),
            "unsloth/Qwen3-14B-GGUF",
            "abc123",
            "Qwen3-14B-Q4_K_M.gguf",
            "deadbeef",
        );
        let store = store_at(tmp.path().to_path_buf());
        let got = store.resolve_repo("unsloth/Qwen3-14B-GGUF", None).unwrap();
        assert!(got.ends_with("Qwen3-14B-Q4_K_M.gguf"));
    }

    #[test]
    fn resolve_hf_quant_selector() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path();
        fake_hf(hub, "u/r", "c", "model-Q4_K_M.gguf", "aa");
        fake_hf(hub, "u/r", "c", "model-Q8_0.gguf", "bb");
        let store = store_at(hub.to_path_buf());
        assert!(store
            .resolve_repo("u/r", Some("Q8_0"))
            .unwrap()
            .ends_with("model-Q8_0.gguf"));
        assert!(store
            .resolve_repo("u/r", Some("q4_k_m")) // case-insensitive
            .unwrap()
            .ends_with("model-Q4_K_M.gguf"));
    }

    /// prism-ml/Ternary-Bonsai-*-gguf ships Q2_0 next to PQ2_0 and Q2_0_g64 — all different layouts.
    /// A `Q2_0` selector must land on Q2_0 regardless of listing order, never on its neighbours.
    #[test]
    fn pick_gguf_quant_neighbours_never_collide() {
        let names: Vec<String> = [
            "Ternary-Bonsai-1.7B-F16.gguf",
            "Ternary-Bonsai-1.7B-PQ2_0.gguf",
            "Ternary-Bonsai-1.7B-Q2_0.gguf",
            "Ternary-Bonsai-1.7B-Q2_0_g64.gguf",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            pick_gguf(&names, Some("Q2_0")).as_deref(),
            Some("Ternary-Bonsai-1.7B-Q2_0.gguf")
        );
        assert_eq!(
            pick_gguf(&names, Some("PQ2_0")).as_deref(),
            Some("Ternary-Bonsai-1.7B-PQ2_0.gguf")
        );
        assert_eq!(
            pick_gguf(&names, Some("Q2_0_g64")).as_deref(),
            Some("Ternary-Bonsai-1.7B-Q2_0_g64.gguf")
        );
        // No Q2_0 in the repo at all → a PQ2_0/TQ2_0 sibling must NOT be served as a fallback.
        let only_p = vec!["Ternary-Bonsai-4B-TQ2_0.gguf".to_string()];
        assert_eq!(pick_gguf(&only_p, Some("Q2_0")), None);
    }

    #[test]
    fn pick_gguf_split_shards_are_loose() {
        let names: Vec<String> = ["m-Q4_K_M-00001-of-00002.gguf", "m-Q8_0.gguf"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            pick_gguf(&names, Some("Q4_K_M")).as_deref(),
            Some("m-Q4_K_M-00001-of-00002.gguf")
        );
        // A quant that is a strict prefix of another must not match it.
        assert_eq!(pick_gguf(&names, Some("Q4_K")), None);
    }

    #[test]
    fn pick_gguf_excludes_mmproj_from_fallback() {
        // Repo ships an oddly-named weights file next to an mmproj projector. The default fallback
        // must pick the weights, never the mmproj.
        let names: Vec<String> = ["mmproj-model-f16.gguf", "weird-weights.gguf"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            pick_gguf(&names, None).as_deref(),
            Some("weird-weights.gguf")
        );
        // mmproj alone, no real weights → no fallback rather than the projector.
        assert_eq!(
            pick_gguf(&["mmproj-model-f16.gguf".to_string()], None),
            None
        );
    }

    #[test]
    fn pick_gguf_prefers_quant_over_f16_master() {
        // Listing order F16-first must still yield the quant for the default selector.
        let names: Vec<String> = ["model-F16.gguf", "model-oddquant.gguf"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            pick_gguf(&names, None).as_deref(),
            Some("model-oddquant.gguf")
        );
        // Only an F16 master present → it is served (better than nothing).
        assert_eq!(
            pick_gguf(&["only-F16.gguf".to_string()], None).as_deref(),
            Some("only-F16.gguf")
        );
    }

    /// `huggingface_hub`'s precedence, in order, and the fallback shape.
    ///
    /// The literal `.cache/huggingface/hub` is the point of the test, not an implementation
    /// detail: it is `huggingface_hub`'s layout (`constants.py` — `$XDG_CACHE_HOME`, else
    /// `~/.cache`, then `huggingface`, then `hub`) and infr has to match it or it re-downloads
    /// models `hf download` already fetched.
    #[test]
    fn hub_dir_follows_huggingface_hubs_precedence() {
        let cache_home = || Some(PathBuf::from("/xdg-cache"));
        let expected_default = PathBuf::from("/xdg-cache/huggingface/hub");

        // Nothing set: the XDG cache base, then HF's two path segments.
        assert_eq!(
            hub_dir(None, None, None, cache_home()),
            Some(expected_default.clone())
        );
        // HF_HOME appends `hub`, and beats the base.
        assert_eq!(
            hub_dir(None, None, Some("/hfhome".into()), cache_home()),
            Some(PathBuf::from("/hfhome/hub"))
        );
        // HF_HUB_CACHE is the full hub dir, no `hub` appended, and beats HF_HOME.
        assert_eq!(
            hub_dir(
                Some("/explicit".into()),
                None,
                Some("/hfhome".into()),
                cache_home()
            ),
            Some(PathBuf::from("/explicit"))
        );
        // The legacy variable is honoured, but loses to the current one.
        assert_eq!(
            hub_dir(None, Some("/legacy".into()), None, cache_home()),
            Some(PathBuf::from("/legacy"))
        );
        assert_eq!(
            hub_dir(
                Some("/new".into()),
                Some("/legacy".into()),
                None,
                cache_home()
            ),
            Some(PathBuf::from("/new"))
        );
        // An empty variable is unset, not a relative path — `HF_HOME=` must not yield `hub`.
        assert_eq!(
            hub_dir(
                Some("".into()),
                Some("".into()),
                Some("".into()),
                cache_home()
            ),
            Some(expected_default)
        );
        // No home directory at all is "cannot tell", not a relative path.
        assert_eq!(hub_dir(None, None, None, None), None);
    }

    /// The layout is `huggingface_hub`'s, so it must NOT vary by platform — no `%LOCALAPPDATA%`,
    /// no `~/Library/Caches`. This is what the whole fix is about, so assert it directly: the
    /// resolver is a pure function of the XDG base, and the base is the same shape everywhere.
    #[test]
    fn the_hub_layout_is_identical_on_every_platform() {
        let base = PathBuf::from("/base");
        let got = hub_dir(None, None, None, Some(base.clone())).expect("a base always resolves");
        assert_eq!(got, base.join("huggingface").join("hub"));

        // ...and the live resolution uses that same base, rather than the OS-native cache dir,
        // which differs per platform and is only consulted for an older infr's leftovers.
        let live = hub_dir(None, None, None, infr_plat::paths::cache_home())
            .expect("this platform resolves a cache home");
        assert!(
            live.ends_with("huggingface/hub"),
            "expected <base>/huggingface/hub, got {live:?}"
        );
        if let Some(os) = infr_plat::paths::os_cache_home() {
            assert_eq!(
                cfg!(target_os = "linux"),
                os == infr_plat::paths::cache_home().unwrap(),
                "the OS cache dir coincides with the XDG one on Linux and nowhere else"
            );
        }

        // And that `discover` passes the XDG base rather than the OS one. On Linux the two are
        // equal, so this assertion is only load-bearing on the macOS and Windows CI legs — which
        // is exactly where getting it wrong costs a re-download. Skipped when the environment
        // names a cache, since then neither base is consulted.
        let env_names_a_cache = ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"]
            .iter()
            .any(|k| std::env::var_os(k).is_some_and(|v| !v.is_empty()));
        if !env_names_a_cache {
            let expected = infr_plat::paths::cache_home()
                .expect("a cache home")
                .join("huggingface")
                .join("hub");
            let store = Store::discover().expect("discover");
            // `discover` may redirect to an older infr's cache; accept either, reject a third.
            let legacy = infr_plat::paths::os_cache_home().map(|b| b.join("huggingface/hub"));
            assert!(
                store.hub == expected || Some(store.hub.clone()) == legacy,
                "discover resolved {:?}, expected the XDG path {expected:?} (or the legacy \
                 {legacy:?})",
                store.hub,
            );
        }
    }

    #[test]
    fn resolve_prefers_refs_main_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path();
        // Two snapshots each with the same-named gguf; refs/main points at `new`.
        fake_hf(hub, "u/r", "old", "model-Q4_K_M.gguf", "oldblob");
        fake_hf(hub, "u/r", "new", "model-Q4_K_M.gguf", "newblob");
        // fake_hf sets refs/main to the last commit written; assert it wins.
        let store = store_at(hub.to_path_buf());
        let got = store.resolve_repo("u/r", None).unwrap();
        // Compare components, not a substring: the separator is `\` on Windows, so a literal
        // "snapshots/new/" matches nothing there however right the answer is.
        let mut components = got.components().map(|c| c.as_os_str());
        assert!(
            components.any(|c| c == "snapshots") && components.next() == Some("new".as_ref()),
            "expected a snapshots/new path, got {got:?}"
        );
    }

    #[test]
    fn resolve_incomplete_shard_set_is_not_cached() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path();
        // Only shard 1 of 2 present → not usable, must resolve to None (triggering a re-pull).
        fake_hf(hub, "u/r", "c", "m-Q4_K_M-00001-of-00002.gguf", "aa");
        let store = store_at(hub.to_path_buf());
        assert_eq!(store.resolve_repo("u/r", Some("Q4_K_M")), None);
    }

    #[test]
    fn resolve_hf_explicit_filename() {
        let tmp = tempfile::tempdir().unwrap();
        fake_hf(tmp.path(), "u/r", "c", "weird-name.gguf", "aa");
        let store = store_at(tmp.path().to_path_buf());
        assert!(store
            .resolve_repo("u/r", Some("weird-name.gguf"))
            .unwrap()
            .ends_with("weird-name.gguf"));
        assert_eq!(store.resolve_repo("u/r", Some("Q4_K_M")), None);
    }

    #[test]
    fn resolve_hf_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_at(tmp.path().to_path_buf());
        assert_eq!(store.resolve_repo("nope/missing", None), None);
    }

    /// unsloth's Dynamic quants live in a per-quant SUBDIRECTORY
    /// (`UD-Q4_K_XL/Qwen3-30B-A3B-UD-Q4_K_XL.gguf`). Such a GGUF must read as cached — a
    /// non-recursive snapshot listing made it invisible, so `infr run` re-downloaded ~18 GB every
    /// invocation. Reading THROUGH the returned symlink also pins the `..` depth of its target: a
    /// hardcoded `../../blobs/<sha>` one directory too shallow dangles and this read fails.
    #[test]
    fn resolve_subdirectory_gguf_is_cached() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path();
        let rel = "UD-Q4_K_XL/Qwen3-30B-A3B-UD-Q4_K_XL.gguf";
        fake_hf(hub, "unsloth/Qwen3-30B-A3B-GGUF", "c1", rel, "aa");
        let store = store_at(hub.to_path_buf());
        for sel in [
            "Q4_K_XL",                                  // quant token in the filename
            "UD-Q4_K_XL",                               // the fuller quant token
            "Qwen3-30B-A3B-UD-Q4_K_XL.gguf",            // explicit name as the user reads it on HF
            "UD-Q4_K_XL/Qwen3-30B-A3B-UD-Q4_K_XL.gguf", // explicit full rfilename
        ] {
            let got = store
                .resolve_repo("unsloth/Qwen3-30B-A3B-GGUF", Some(sel))
                .unwrap_or_else(|| panic!("selector {sel} should resolve as cached"));
            assert!(got.ends_with(rel), "{sel} → {got:?}");
            assert_eq!(
                fs::read_to_string(&got).unwrap(),
                "fake gguf bytes",
                "{sel}: snapshot symlink must point at the blob"
            );
        }
    }

    /// The whole-shard-set rule must hold for shards living in a subdirectory too — half a split
    /// model in `UD-Q4_K_XL/` is as unloadable as half a split model at the snapshot root.
    #[test]
    fn resolve_subdirectory_shard_set_completeness() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path();
        let one = "UD-Q4_K_XL/m-UD-Q4_K_XL-00001-of-00002.gguf";
        let two = "UD-Q4_K_XL/m-UD-Q4_K_XL-00002-of-00002.gguf";
        fake_hf(hub, "u/r", "c", one, "aa");
        let store = store_at(hub.to_path_buf());
        assert_eq!(
            store.resolve_repo("u/r", Some("UD-Q4_K_XL")),
            None,
            "shard 1 of 2 alone must not count as cached"
        );
        fake_hf(hub, "u/r", "c", two, "bb");
        let got = store.resolve_repo("u/r", Some("UD-Q4_K_XL")).unwrap();
        assert!(got.ends_with(one), "must hand back shard 1: {got:?}");
    }

    /// The snapshot walk is recursive but deliberately depth-bounded (see [`MAX_SNAPSHOT_DEPTH`]) so
    /// the `infr run` fast path can't be turned into a full traversal by an arbitrary repo tree.
    #[test]
    fn snapshot_walk_is_recursive_and_depth_bounded() {
        let tmp = tempfile::tempdir().unwrap();
        let snap = tmp.path();
        for rel in ["root.gguf", "a/one.gguf", "a/b/c/deep.gguf"] {
            let p = snap.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, b"x").unwrap();
        }
        let mut got = snapshot_ggufs(snap);
        got.sort();
        assert_eq!(got, vec!["a/b/c/deep.gguf", "a/one.gguf", "root.gguf"]);
        // One level past the bound is not listed. Raising MAX_SNAPSHOT_DEPTH is a deliberate change,
        // so this assertion is meant to be updated with it, never silently.
        let too_deep = snap.join("a/b/c/d/deeper.gguf");
        fs::create_dir_all(too_deep.parent().unwrap()).unwrap();
        fs::write(&too_deep, b"x").unwrap();
        assert!(!snapshot_ggufs(snap).iter().any(|n| n.contains("deeper")));
    }

    #[test]
    fn resolve_path_variants() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("model.gguf");
        fs::write(&gguf, b"x").unwrap();
        let store = store_at(tmp.path().to_path_buf());
        assert_eq!(
            store.resolve(&ModelRef::Path(gguf.clone())).unwrap(),
            Some(gguf)
        );
        assert_eq!(
            store
                .resolve(&ModelRef::Path(tmp.path().join("nope.gguf")))
                .unwrap(),
            None
        );
    }
}
