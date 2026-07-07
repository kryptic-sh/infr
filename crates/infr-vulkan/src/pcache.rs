//! On-disk `vkPipelineCache` persistence. Pipeline creation (`vkCreateComputePipelines`) is the
//! driver's GPU-specific codegen and was measured at ~5s across a cold DG forward's kernel set —
//! all one-time work that previously re-ran every process launch (we passed
//! `vk::PipelineCache::null()` everywhere and relied incidentally on Mesa's own shader cache,
//! which softens but does not eliminate it). This module seeds ONE `vk::PipelineCache` from a
//! per-device file at backend init and writes it back (debounced, and finally on drop), so every
//! launch after the first creates pipelines from cached binaries.
//!
//! Invalidation is two-layer:
//! - The DRIVER validates its own header inside the blob (vendor/device/driverVersion/cacheUUID)
//!   and silently ignores data it can't use — so a driver upgrade never corrupts, at worst it
//!   recompiles once and the next save replaces the file.
//! - OUR envelope carries the build-time SHADER_SET_FINGERPRINT (FNV-1a over every compiled
//!   SPIR-V blob — see build.rs): any shader change discards the old file WHOLESALE instead of
//!   letting entries for retired pipeline variants accumulate in the blob forever.
//!
//! `INFR_NO_PIPELINE_CACHE=1` disables persistence (the in-process cache handle still works).

use ash::vk;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

include!(concat!(env!("OUT_DIR"), "/shader_fingerprint.rs"));

const MAGIC: &[u8; 8] = b"INFRVPC1";
/// Debounce for mid-run saves: pipeline creation comes in bursts (warmup, a new arch's first
/// forward) — one save per burst-second is plenty, and the final Drop save catches the tail.
const SAVE_DEBOUNCE_SECS: u64 = 1;

/// Handle for one device's persisted pipeline cache: where it lives on disk and when it was last
/// written. `None`-able at the call sites (env-disabled or no writable cache dir).
pub(crate) struct PcachePersist {
    path: PathBuf,
    /// Driver version folded into the envelope alongside the shader-set fingerprint: the driver
    /// already ignores stale blobs itself, but a version flip also means retired entries would
    /// sit in the file forever — treat it like a shader-set change and start fresh.
    driver_version: u32,
    last_save: Mutex<Instant>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl PcachePersist {
    /// `~/.cache/infr/vk-pipeline-cache-{vendor:08x}-{device:08x}.bin` (XDG-aware) — keyed per
    /// device so a multi-GPU box never clobbers one GPU's cache with another's.
    pub(crate) fn new(props: &vk::PhysicalDeviceProperties) -> Option<Self> {
        if std::env::var_os("INFR_NO_PIPELINE_CACHE").is_some() {
            return None;
        }
        let base = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
        let dir = base.join("infr");
        std::fs::create_dir_all(&dir).ok()?;
        Some(Self {
            path: dir.join(format!(
                "vk-pipeline-cache-{:08x}-{:08x}.bin",
                props.vendor_id, props.device_id
            )),
            driver_version: props.driver_version,
            last_save: Mutex::new(Instant::now()),
        })
    }

    /// Read + validate the persisted blob. Any mismatch (magic, fingerprint, driver version,
    /// truncation) returns `None` — the stale file is simply replaced by the next save.
    pub(crate) fn load(&self) -> Option<Vec<u8>> {
        let data = std::fs::read(&self.path).ok()?;
        if data.len() < 28 || &data[..8] != MAGIC {
            return None;
        }
        let fp = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let drv = u32::from_le_bytes(data[16..20].try_into().unwrap());
        let len = u64::from_le_bytes(data[20..28].try_into().unwrap()) as usize;
        if fp != SHADER_SET_FINGERPRINT || drv != self.driver_version || data.len() != 28 + len {
            return None;
        }
        Some(data[28..].to_vec())
    }

    /// Serialize the live cache to disk atomically (tmp + rename — a crash mid-write never
    /// leaves a torn file; `load`'s length check catches anything else).
    pub(crate) fn save(&self, device: &ash::Device, cache: vk::PipelineCache) {
        if cache == vk::PipelineCache::null() {
            return;
        }
        let Ok(blob) = (unsafe { device.get_pipeline_cache_data(cache) }) else {
            return;
        };
        if blob.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(28 + blob.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&SHADER_SET_FINGERPRINT.to_le_bytes());
        out.extend_from_slice(&self.driver_version.to_le_bytes());
        out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        out.extend_from_slice(&blob);
        let tmp = self
            .path
            .with_extension(format!("tmp.{}", std::process::id()));
        if std::fs::write(&tmp, &out).is_ok() {
            let _ = std::fs::rename(&tmp, &self.path);
        }
        *self.last_save.lock().unwrap() = Instant::now();
    }

    /// Debounced save for mid-run persistence (called after each NEW pipeline lands) — covers
    /// long-lived processes that never Drop cleanly (serve under SIGKILL).
    pub(crate) fn maybe_save(&self, device: &ash::Device, cache: vk::PipelineCache) {
        {
            let last = self.last_save.lock().unwrap();
            if last.elapsed().as_secs() < SAVE_DEBOUNCE_SECS {
                return;
            }
        }
        self.save(device, cache);
    }
}
