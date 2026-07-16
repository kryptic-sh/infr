//! Vulkan backend (`ash` + SPIR-V). The MVP `Backend` impl.
//!
//! Reference: `~/Projects/llama.cpp/ggml/src/ggml-vulkan/` and its `vulkan-shaders/*.comp`
//! (reuse the tuned quant matmul / dequant / attention shaders). Enable device features
//! `VK_KHR_cooperative_matrix`, `shaderFloat16`, `VK_KHR_16bit_storage`,
//! `VK_KHR_shader_subgroup_extended_types`. See docs/PLAN.md.
#![allow(dead_code)]
// GPU kernel record/dispatch APIs bind many distinct buffers (weights, scales, activations,
// scratch) — wide signatures are inherent here, not a refactor smell.
#![allow(clippy::too_many_arguments)]

mod adapter;
mod gemm;
pub mod linear;
mod matmul;
mod ops;
pub mod pager;
mod pcache;
mod recorder;

pub use recorder::{RecordedCmd, Recorder};

/// Shared-memory bytes consumed per query row of a flash-attention prefill tile
/// (`Ss` + `Ps` + `Os` + softmax state, at `BN=64` / `HD=128`). The tile height is chosen so
/// `rows * FLASH_SHARED_PER_ROW <= maxComputeSharedMemorySize`; `use_flash` needs the smallest
/// tile (`BM=32`) to fit. Keep in sync with `attn_flash{,_warp,_partial}.comp`.
pub const FLASH_SHARED_PER_ROW: u32 = 908;
/// Same, for the register-O flash tile (`sfsh` + `Psh` + `pvsh` + state); smallest tile is `BR=64`.
/// Keep in sync with `attn_flash_reg.comp`.
pub const FLASH_REG_SHARED_PER_ROW: u32 = 460;

use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;

use infr_core::{
    backend::{Bindings, Buffer, BufferUsage, Capabilities, Plan},
    error::{Error, Result},
    graph::Graph,
    Backend,
};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Terse local shorthand for the shared [`Error::backend`] constructor.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn be(s: impl std::fmt::Display) -> Error {
    Error::backend(s)
}

/// Downcast `&dyn Buffer` → `&VkBuffer`.
///
/// # Safety
/// Must only be called with buffers returned by `VulkanBackend::alloc`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn as_vk_buf(b: &dyn Buffer) -> &VkBuffer {
    // Fat pointer (data_ptr, vtable_ptr) → thin data_ptr → &VkBuffer.
    &*(b as *const dyn Buffer as *const () as *const VkBuffer)
}

// ── device class (process-global) ─────────────────────────────────────────────

/// The class of the Vulkan device this PROCESS opened — see [`device_class`].
#[derive(Clone, Copy, Debug)]
pub struct DeviceClass {
    /// `deviceType == INTEGRATED_GPU` (see [`Capabilities::integrated`]).
    pub integrated: bool,
    /// Compute units, or 0 = unknown (see [`Capabilities::compute_units`]).
    pub compute_units: u32,
}

/// Set ONCE by the first [`VulkanBackend::new`] in the process.
static DEVICE_CLASS: std::sync::OnceLock<DeviceClass> = std::sync::OnceLock::new();

/// The class of the Vulkan device this process opened, or `None` when no Vulkan backend has been
/// constructed (a CPU/Metal run, or a GPU-less box).
///
/// A PROCESS-GLOBAL because its one consumer, the seam's `ubatch_rows`, is itself a process-global
/// funnel: the prefill loop, the activation reserve, and the SWA ring sizing must all agree on ONE
/// chunk height, and they are reached from call sites that hold no backend handle. Same shape and
/// lifetime as the seam's existing `PINNED_UBATCH`. A multi-GPU process mixing an iGPU and a dGPU
/// would pin whichever opened first; infr opens exactly one device per process today.
pub fn device_class() -> Option<DeviceClass> {
    DEVICE_CLASS.get().copied()
}

// ── shared GPU state ──────────────────────────────────────────────────────────

/// Device memory snapshot from [`VulkanBackend::vram`]. `available` is live free bytes when
/// `live` is true (VK_EXT_memory_budget present), otherwise it equals `total` (best-effort).
///
/// WHICH HEAPS THIS COUNTS depends on the device class (see [`vram_info`]): device-local only on a
/// discrete card, ALL heaps on a unified-memory part where they are the same physical DDR.
#[derive(Clone, Copy, Debug)]
pub struct VramInfo {
    pub total: u64,
    pub available: u64,
    pub live: bool,
    /// True when this snapshot counted every heap because the device has unified memory (see
    /// [`Capabilities::unified_memory`]) — only affects how the guard words its error.
    pub uma: bool,
}

struct VulkanShared {
    // NOTE: field declaration order matters for drop.
    // Rust drops struct fields in *declaration order*.  We keep `allocator`
    // in a `ManuallyDrop` so we can drop it explicitly before calling
    // `destroy_device` in the `Drop` impl.
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    /// Serialises all one-shot command-buffer submissions.
    cmd_pool: Mutex<vk::CommandPool>,
    /// Must be dropped before the device is destroyed.
    allocator: ManuallyDrop<Mutex<Allocator>>,
    caps: Capabilities,
    /// VK_EXT_memory_budget enabled → `vram()` can report live free bytes (else total only).
    has_mem_budget: bool,
    /// `maxMemoryAllocationSize` — the largest single `vkAllocateMemory` this device accepts
    /// (`VkPhysicalDeviceMaintenance3Properties`, ~4 GiB on RADV). The weight arena
    /// (`reserve_weights`) splits its up-front reservation into blocks no larger than this, since a
    /// single alloc of a whole multi-GiB model is impossible. Falls back to a conservative 1 GiB
    /// (the Vulkan-guaranteed floor) if the device reports 0.
    max_mem_alloc_size: u64,
    /// `VK_KHR_push_descriptor` loader, when the device supports it — every dispatch's
    /// descriptor binding then records via `cmd_push_descriptor_set` (recorder.rs
    /// `bind_descriptors`) instead of a pooled `alloc_set` + `update_descriptor_sets` +
    /// `cmd_bind_descriptor_sets` per op. `None` falls back to the pooled path.
    push_descriptor: Option<ash::khr::push_descriptor::Device>,
    /// Pre-reserved bump arena for load-once weights (see `reserve_weights`). `None` until reserved;
    /// weight allocs then sub-allocate from it instead of the gpu-allocator.
    weight_arena: Mutex<Option<WeightArena>>,
    /// Lazily-built, reused compute pipeline for the linear op (see `linear.rs`).
    linear_kernel: std::sync::OnceLock<crate::linear::LinearKernel>,
    /// Generic cache of compute kernels by name (see `ops.rs`).
    kernels: Mutex<HashMap<&'static str, crate::ops::ComputeKernel>>,
    /// Device pipeline cache, seeded from disk at init and persisted back (see `pcache.rs`) so
    /// pipeline creation after the first-ever launch reuses cached driver binaries. Null when
    /// creation failed (caching is then simply off — Vulkan accepts a null cache everywhere).
    pipeline_cache: vk::PipelineCache,
    /// Disk persistence for `pipeline_cache`; `None` = INFR_NO_PIPELINE_CACHE or no cache dir.
    pcache: Option<crate::pcache::PcachePersist>,
    /// Active weight-load progress bar (see [`VulkanBackend::weight_progress`]). Every
    /// `BufferUsage::Weights` allocation advances it, so no model loader can forget to tick it.
    weight_pb: Mutex<Option<indicatif::ProgressBar>>,
    /// Cumulative device-local bytes THIS backend has committed (pooled/dedicated allocations +
    /// weight-arena blocks). The VRAM budget guard's fallback accounting when
    /// VK_EXT_memory_budget is absent — the live per-heap budget is preferred when present
    /// (it also sees other processes' VRAM).
    device_used: AtomicU64,
    /// SUBMIT SPLITTER: the most dispatches `execute_static` will record into one command buffer
    /// before submitting it and opening the next (`0` = unlimited, never split).
    ///
    /// The GPU hang watchdog is armed per SUBMIT, so a forward pass recorded as one command buffer
    /// is one indivisible watchdog job. On a 2-CU integrated part that job is ~2.05 s of real GPU
    /// work and the device kills it at ~2.06 s — a margin so thin it was a coin flip, which is the
    /// `ring gfx_0.0.0 timeout` -> `VK_ERROR_DEVICE_LOST` this exists to prevent. Splitting the
    /// SAME work across N command buffers divides the per-job duration by N without removing any
    /// work: the segments still run back-to-back on the queue (`finish_nowait`, no host sync), the
    /// watchdog just gets N short jobs to watch instead of one long one.
    ///
    /// Seeded from `infr_core::initial_submit_dispatch_cap` (unlimited on discrete — a dGPU
    /// forward is tens of ms and must not pay for barriers it does not need) and then RE-TUNED
    /// from measurement after every forward (`infr_core::submit_cap_from_measurement`), so the
    /// bound tracks whatever the device actually is rather than a table of magic numbers.
    submit_dispatch_cap: AtomicUsize,
    /// Memory type index for DIRECT-TO-VRAM weight writes (Resizable BAR), or `None` when the
    /// device doesn't expose one. See [`probe_rebar_type`]: it is a
    /// `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` type sitting on the device's LARGEST
    /// device-local heap — i.e. the whole of VRAM is host-visible, not just a 256 MiB BAR window.
    rebar_type: Option<u32>,
    /// UNIFIED-MEMORY parts only (`None` on every discrete GPU): the host-visible memory type on
    /// the non-device-local heap that `GpuOnly` allocations SPILL into once the device-local heap
    /// is full. See [`probe_uma_overflow_type`] for why counting that heap in the budget is not
    /// enough on its own — the bytes have to be able to land there too.
    uma_overflow_type: Option<u32>,
    /// Bytes this process has placed on the UMA OVERFLOW heap (`uma_overflow_type`). Counted apart
    /// from `device_used`, which stays "bytes on the DEVICE-LOCAL heaps" — the spill decision is
    /// exactly "would this push the device-local heap past its declared size", so it must not see
    /// the bytes already diverted. Always 0 on a discrete GPU.
    uma_spilled: AtomicU64,
    /// Whether the CURRENT weight load writes straight into VRAM through `rebar_type`. Decided
    /// once per load in `weight_progress_scope` (needs `rebar_type` AND enough room on its heap
    /// for the model), so a ReBAR-less box — or a model too big for the host-visible heap —
    /// cleanly takes the staging-ring path instead. Only meaningful while a weight scope is open.
    weights_direct: AtomicBool,
    /// Reused staging ring for the NON-direct weight path (see [`StagingRing`]). Built lazily on
    /// the first staged weight upload of a load and torn down with the weight scope.
    staging_ring: Mutex<Option<StagingRing>>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanShared {
    /// Wait for every in-flight staging copy and tear the ring down. Called when the weight-load
    /// scope ends, so all weights are fully resident before any forward is recorded — this is the
    /// synchronization point that replaced the old per-tensor `queue_wait_idle`.
    fn drain_staging_ring(&self) {
        let Some(mut ring) = self.staging_ring.lock().unwrap().take() else {
            return;
        };
        let pending: Vec<vk::Fence> = (0..RING_SLOTS)
            .filter(|&i| ring.busy[i])
            .map(|i| ring.fences[i])
            .collect();
        unsafe {
            if !pending.is_empty() {
                let _ = self.device.wait_for_fences(&pending, true, u64::MAX);
            }
            let pool = *self.cmd_pool.lock().unwrap();
            self.device.free_command_buffers(pool, &ring.cmds);
            for f in ring.fences.drain(..) {
                self.device.destroy_fence(f, None);
            }
        }
        // `ring.bufs` drop here → the staging slots are freed.
    }
}

// ash Instances/Devices/handles are Send+Sync per the Vulkan spec when
// accessed through our Mutexes.
unsafe impl Send for VulkanShared {}
unsafe impl Sync for VulkanShared {}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanShared {
    /// Debounced disk save of the pipeline cache — call after a NEW pipeline lands so long-lived
    /// processes (serve) persist without waiting for a clean Drop.
    pub(crate) fn persist_pipeline_cache(&self) {
        if let Some(pc) = &self.pcache {
            pc.maybe_save(&self.device, self.pipeline_cache);
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Drop for VulkanShared {
    fn drop(&mut self) {
        unsafe {
            // Also the pipeline cache's TRIPWIRE verdict (see `pcache.rs`): VK_ERROR_DEVICE_LOST is
            // STICKY — once the device is lost every call returns it — so this one drain doubles as
            // "did this run hang the GPU?", with no flag to thread through every submit site.
            let device_lost = matches!(
                self.device.device_wait_idle(),
                Err(vk::Result::ERROR_DEVICE_LOST)
            );
            // Destroy the cached linear kernel (pipeline/layouts/shader/pool) if built.
            if let Some(k) = self.linear_kernel.get() {
                crate::linear::destroy_linear_kernel(&self.device, k);
            }
            if let Ok(map) = self.kernels.lock() {
                for k in map.values() {
                    crate::ops::destroy_compute_kernel(&self.device, k);
                }
            }
            // Persist the pipeline cache (final save — the debounced mid-run saves may have
            // missed the tail) and destroy it. On a LOST device this discards the file instead of
            // saving it, and either way it disarms this process's tripwire marker.
            if let Some(pc) = &self.pcache {
                pc.finish(&self.device, self.pipeline_cache, device_lost);
            }
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
            // Destroy command pool.
            let pool = *self.cmd_pool.lock().unwrap();
            self.device.destroy_command_pool(pool, None);
            // Free the weight arena's device memory before destroying the device.
            if let Some(arena) = self.weight_arena.lock().unwrap().as_mut() {
                arena.destroy(&self.device);
            }
            // Drop the allocator *before* destroying the device.
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ── VkBuffer ──────────────────────────────────────────────────────────────────

/// How a `VkBuffer`'s device memory is owned.
enum Backing {
    /// A gpu-allocator sub-allocation — freed back to the allocator on drop (transient buffers,
    /// host-visible staging/readback, and weights when no arena is reserved).
    Pooled(ManuallyDrop<Allocation>),
    /// Bump-allocated from the [`WeightArena`]. The arena block owns the memory; on drop the buffer
    /// only destroys its own handle (the block frees the memory when the arena drops).
    Arena,
    /// A DEDICATED `VkDeviceMemory` this buffer owns outright, allocated from the ReBAR memory type
    /// (`VulkanShared::rebar_type`) and PERSISTENTLY MAPPED: device-local VRAM that the host can
    /// write through directly. This is the fast weight-load path — `upload` memcpys the GGUF bytes
    /// straight into VRAM with no staging buffer, no `vkCmdCopyBuffer` and no queue stall. Freed
    /// (unmapped + `vkFreeMemory`) on drop.
    ///
    /// Allocated per weight tensor rather than from one big arena because
    /// `maxMemoryAllocationSize` is ~4 GiB on RADV — a single block for a 9 GiB (let alone
    /// 21.9 GiB) model is impossible, and a chunked arena would strand the tail of every block.
    /// Per-tensor `vkAllocateMemory` measured ~14 ms across a whole 443-tensor load, so there is
    /// nothing to win by pooling here.
    Vram {
        memory: vk::DeviceMemory,
        ptr: *mut u8,
        /// True when this is a UNIFIED-MEMORY SPILL (see `probe_uma_overflow_type`) rather than a
        /// ReBAR weight: the memory came from the non-device-local overflow heap, so it is charged
        /// to `VulkanShared::uma_spilled` instead of `device_used`. Keeping the two counters apart
        /// is what lets the spill decision ask "is the DEVICE-LOCAL heap full?" without the answer
        /// being polluted by the bytes it already spilled elsewhere.
        spilled: bool,
    },
    /// A logical BYTE RANGE of a [`BdaWeightArena`] block's single big `vk::Buffer` (`INFR_RESIDENT_BDA=1`
    /// resident weight sub-tensors — see [`VulkanBackend::bda_weight_alloc`]). Unlike every other
    /// variant, `VkBuffer::buffer` here is NOT this handle's own object: it is the block's buffer,
    /// shared byte-for-byte with every other sub-tensor carved from the same block, and with the
    /// block's own keepalive copy. The `Arc` is what keeps the block (and therefore its memory and
    /// buffer handle) alive for as long as any sub-tensor referencing it is alive; dropping this
    /// variant frees NOTHING — no `destroy_buffer`, no memory free — that happens exactly once, when
    /// the last `Arc<BdaBlockHandle>` clone (the arena's own, or the last live sub-tensor's) drops
    /// and `BdaBlockHandle::buf`'s ordinary `VkBuffer::drop` runs.
    BdaSub(Arc<BdaBlockHandle>),
}

struct VkBuffer {
    shared: Arc<VulkanShared>,
    buffer: vk::Buffer,
    backing: Backing,
    /// Logical buffer size (what the caller asked for and what `upload`/`fill_buf` touch).
    size: usize,
    /// Device-memory bytes actually committed for this buffer (`requirements.size`, i.e. `size`
    /// rounded up for alignment). Charged to / released from the VRAM budget guard's accounting
    /// by [`Backing::Vram`], which owns its `VkDeviceMemory` outright.
    mem_size: u64,
    location: MemoryLocation,
    /// Byte offset of this tensor's logical range within `buffer` — `0` for every buffer except a
    /// resident-BDA weight sub-tensor (`Backing::BdaSub`, see [`BdaWeightArena`]), where several
    /// `VkBuffer` handles share ONE big `vk::Buffer` (the arena block) and this field is what tells
    /// them apart. Every upload/download/fill site that touches `buffer` at a byte offset must add
    /// this in, and [`Buffer::device_addr`] is `block_base_addr + sub_offset`.
    sub_offset: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VkBuffer {
    /// Persistently-mapped host pointer for host-visible buffers — pooled host-visible allocations
    /// AND [`Backing::Vram`] weights (device-local VRAM the host can write through, via ReBAR).
    /// `None` for plain device-local or arena buffers, which are filled via a staging copy.
    /// Every "can I just memcpy into this?" decision (`upload`, `fill_buf`) keys off this.
    fn mapped_ptr(&self) -> Option<*mut u8> {
        match &self.backing {
            Backing::Pooled(a) => a.mapped_ptr().map(|p| p.as_ptr() as *mut u8),
            Backing::Arena => None,
            Backing::Vram { ptr, .. } => Some(*ptr),
            // Defensive, not currently reachable: `bda_weight_alloc`'s blocks are plain `GpuOnly`
            // dedicated allocations (never host-mapped), so `buf.mapped_ptr()` is `None` today. If a
            // future block ever WERE host-visible, offsetting by `sub_offset` here is what keeps
            // every "can I just memcpy?" call site (`upload`/`fill_buf`) correct without change.
            Backing::BdaSub(block) => block
                .buf
                .mapped_ptr()
                .map(|p| unsafe { p.add(self.sub_offset) }),
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Drop for VkBuffer {
    fn drop(&mut self) {
        unsafe {
            match &mut self.backing {
                Backing::Pooled(alloc) => {
                    let alloc = ManuallyDrop::take(alloc);
                    // Keep the budget guard's fallback accounting balanced (arena buffers don't own
                    // their memory — the arena block stays counted until the backend drops).
                    if self.location == MemoryLocation::GpuOnly {
                        self.shared
                            .device_used
                            .fetch_sub(alloc.size(), Ordering::Relaxed);
                    }
                    self.shared.allocator.lock().unwrap().free(alloc).ok();
                }
                // A dedicated VkDeviceMemory we own outright: either a ReBAR weight (device-local)
                // or a UMA spill (the overflow heap). Both are charged to the budget guard at
                // allocation (see `make_buf_ex`) — balance the counter each was charged to.
                Backing::Vram {
                    memory, spilled, ..
                } => {
                    let counter = if *spilled {
                        &self.shared.uma_spilled
                    } else {
                        &self.shared.device_used
                    };
                    counter.fetch_sub(self.mem_size, Ordering::Relaxed);
                    self.shared.device.unmap_memory(*memory);
                    self.shared.device.free_memory(*memory, None);
                }
                Backing::Arena => {}
                // Shares the block's `vk::Buffer` handle byte-for-byte with every other sub-tensor
                // and the block's own keepalive copy — this handle owns NEITHER the buffer object
                // NOR its memory, only an `Arc` clone. Dropping the `Arc` (below, implicitly, when
                // `self.backing` itself drops) is the whole of this handle's cleanup; the actual
                // `destroy_buffer`/memory-free happens once, inside `BdaBlockHandle::buf`'s own
                // `VkBuffer::drop`, when the last clone goes away.
                Backing::BdaSub(_) => {}
            }
            // Every OTHER variant owns `self.buffer` outright and must destroy it here. A `BdaSub`
            // handle's `buffer` is an alias of the block's — destroying it here would destroy the
            // block out from under every other sub-tensor still referencing it (and double-free when
            // the block's own `VkBuffer` later drops), so it is the one variant that skips this.
            if !matches!(self.backing, Backing::BdaSub(_)) {
                self.shared.device.destroy_buffer(self.buffer, None);
            }
        }
    }
}

unsafe impl Send for VkBuffer {}
unsafe impl Sync for VkBuffer {}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Buffer for VkBuffer {
    fn len_bytes(&self) -> usize {
        self.size
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device_addr(&self) -> Option<u64> {
        match &self.backing {
            Backing::BdaSub(block) => Some(block.base_addr + self.sub_offset as u64),
            _ => None,
        }
    }
}

// ── weight arena ────────────────────────────────────────────────────────────────

/// Buffer usage flags for every device buffer (must match across the arena probe and all
/// allocations so their memory-type bits / alignment agree).
const BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        // Any buffer may serve as vkCmdDispatchIndirect args (the split-K replay prologue writes
        // the partial pass's workgroup count GPU-side).
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
);

/// Overflow arena blocks (only allocated if the reserved block underflows the estimate) stay modest
/// so a tiny estimate miss can't waste a whole second model-sized block.
const ARENA_OVERFLOW_BLOCK: u64 = 64 * 1024 * 1024;

/// Find the memory type for DIRECT-TO-VRAM weight writes ("Resizable BAR") — OPT-IN, see below.
///
/// The target is a `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` memory type on the device's
/// LARGEST device-local heap: real VRAM the host can write straight through, so `upload` can
/// memcpy a weight in with no staging buffer, no `vkCmdCopyBuffer` and no fence. Requiring the
/// LARGEST device-local heap is what distinguishes true ReBAR from the legacy 256 MiB BAR window
/// (without ReBAR, RADV still exposes a `DEVICE_LOCAL | HOST_VISIBLE` type, but on a separate tiny
/// heap — allocating a 9 GiB model from it would exhaust it and silently spill into system RAM).
///
/// ## Why this is OFF by default: it is SLOWER than the staging ring. MEASURED, not assumed.
///
/// It is the obvious "one pass instead of two" optimization, and it loses. On a 7900 XTX
/// (PCIe 4.0 x16), Qwen3-14B-Q4_K_M (~9 GiB of weights), warm page cache:
///
/// | weight-upload path                        | `upload` self | effective  | total load |
/// |-------------------------------------------|---------------|------------|------------|
/// | direct-to-VRAM, single-threaded memcpy     | 1.02 s        |  8.8 GB/s  | 2.01 s     |
/// | direct-to-VRAM, parallel memcpy            | ~0.62 s       | ~14.5 GB/s | 1.57 s     |
/// | reused staging ring ([`StagingRing`])      | 0.50 s        | ~18 GB/s   | 1.42 s     |
///
/// The catch is WHERE the host's stores land. Writing to ReBAR VRAM puts every byte on the PCIe
/// bus AT MEMCPY TIME, through a write-combined mapping — one core saturates at ~8.8 GB/s, and even
/// all cores together can't beat the link. The staging ring's memcpy instead lands in ordinary
/// system RAM at full speed (~18 GB/s), and the PCIe crossing is done afterwards by the DMA
/// engine — which OVERLAPS with the host filling the next chunk. So the "extra" copy is free: it
/// buys you the ability to hide the bus behind the memcpy. Two overlapped passes beat one
/// serialized pass over a slower bus.
///
/// Kept because it is a genuine win on hardware where the host↔device link is NOT the bottleneck
/// (integrated/UMA GPUs, where "VRAM" is system RAM and the DMA is a pointless extra copy), and
/// because it is the natural path for a future load that wants to skip host staging entirely.
/// `INFR_REBAR=1` opts in; the default is the ring on every device.
fn probe_rebar_type(mp: &vk::PhysicalDeviceMemoryProperties) -> Option<u32> {
    if !std::env::var("INFR_REBAR").is_ok_and(|v| !v.is_empty() && v != "0") {
        return None;
    }
    let want = vk::MemoryPropertyFlags::DEVICE_LOCAL
        | vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT;
    // The heap holding the bulk of VRAM.
    let vram_heap = (0..mp.memory_heap_count)
        .filter(|&h| {
            mp.memory_heaps[h as usize]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .max_by_key(|&h| mp.memory_heaps[h as usize].size)?;
    (0..mp.memory_type_count).find(|&i| {
        let t = mp.memory_types[i as usize];
        t.property_flags.contains(want) && t.heap_index == vram_heap
    })
}

/// The size of the heap backing memory type `ty`.
fn heap_size_of(mp: &vk::PhysicalDeviceMemoryProperties, ty: u32) -> u64 {
    mp.memory_heaps[mp.memory_types[ty as usize].heap_index as usize].size
}

/// The UMA OVERFLOW memory type: a host-visible type on a NON-device-local heap. `None` on a
/// discrete GPU (never probed) and on any UMA part that doesn't expose one.
///
/// This is the other half of the unified-memory fix, and without it widening the budget is not
/// merely useless but actively harmful. `vram_info` budgets a UMA part against ALL heaps, but
/// gpu-allocator resolves `MemoryLocation::GpuOnly` to the FIRST DEVICE_LOCAL memory type and
/// never falls back — so every allocation lands on the device-local heap no matter how full it is.
/// RADV does not enforce the heap size (a 41 GiB run of 1 GiB allocations succeeded on a
/// "21.47 GiB" heap), so nothing errors; the kernel simply can no longer validate the buffer list
/// and the next SUBMIT dies with "Not enough memory for command submission" — a device-lost, i.e.
/// exactly the silent-degradation failure the guard exists to prevent, just moved later.
/// MEASURED on RAPHAEL_MENDOCINO with gemma-4-31B: weights + KV + activations cross the
/// device-local heap's 21.47 GiB and the guard sits there reporting 10.70 GiB "available" (which
/// is precisely heap 0's size — capacity nothing could reach) while the submit fails.
///
/// So the overflow must be PLACED, not just counted. On an APU heap 0 is the same DDR at the same
/// bandwidth as the synthetic device-local heap — the weights are read out of GTT either way
/// (`mem_info_gtt_used` accounts for them on both paths) — so spilling there costs no bandwidth.
/// It is only on a DISCRETE card that the non-device-local heap means "across PCIe", which is why
/// this is probed for UMA parts alone.
fn probe_uma_overflow_type(mp: &vk::PhysicalDeviceMemoryProperties) -> Option<u32> {
    let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    (0..mp.memory_type_count).find(|&i| {
        let t = mp.memory_types[i as usize];
        let heap = mp.memory_heaps[t.heap_index as usize];
        t.property_flags.contains(want)
            && !t
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            && !heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
    })
}

/// Free bytes on the DEVICE-LOCAL heaps alone — what the UMA spill decision keys off (unlike
/// [`vram_info`]'s UMA figure, which spans every heap). Live VK_EXT_memory_budget when present, so
/// a device-local heap another process has filled reads as full here too; otherwise the heap size
/// minus this process's tracked device-local bytes (`device_used`; spilled bytes are charged to
/// `uma_spilled` and correctly excluded).
fn device_local_room(s: &VulkanShared) -> u64 {
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceMemoryProperties2::default();
    if s.has_mem_budget {
        props2 = props2.push_next(&mut budget);
    }
    unsafe {
        s.instance
            .get_physical_device_memory_properties2(s.physical_device, &mut props2)
    };
    let mp = props2.memory_properties;
    let (mut size, mut avail) = (0u64, 0u64);
    for i in 0..mp.memory_heap_count as usize {
        if mp.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        {
            size += mp.memory_heaps[i].size;
            avail += budget.heap_budget[i]
                .saturating_sub(budget.heap_usage[i])
                .min(mp.memory_heaps[i].size);
        }
    }
    if s.has_mem_budget {
        avail
    } else {
        size.saturating_sub(s.device_used.load(Ordering::Relaxed))
    }
}

/// Human byte count for the budget guard's error, in the LARGEST unit that keeps a significant
/// digit. A fixed `{:.2} GiB` printed "0.00 GiB requested" for anything under ~5 MiB — a guard
/// error that reads as nonsense exactly when it fires on a small allocation (the last straw on a
/// budget the big tensors already filled).
fn fmt_bytes(b: u64) -> String {
    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    match b {
        b if b >= GIB => format!("{:.2} GiB", b as f64 / GIB as f64),
        b if b >= MIB => format!("{:.1} MiB", b as f64 / MIB as f64),
        b if b >= KIB => format!("{:.1} KiB", b as f64 / KIB as f64),
        b => format!("{b} B"),
    }
}

/// Copy `src` into a persistently-mapped destination, in PARALLEL for large buffers.
///
/// For a ReBAR weight the destination is write-combined VRAM across PCIe, where a single core
/// cannot saturate the link (measured ~8.8 GB/s single-threaded on a 7900 XTX / PCIe 4.0 x16).
/// Splitting the copy across cores lets several write-combine streams be in flight at once. Small
/// buffers copy inline — below the threshold the rayon fork/join costs more than it saves.
fn copy_to_mapped(src: &[u8], dst: *mut u8) {
    /// Below this, a plain memcpy beats paying for fork/join.
    const PAR_MIN: usize = 4 * 1024 * 1024;
    /// Chunk per task: big enough to amortize scheduling, small enough to spread over cores.
    const PAR_CHUNK: usize = 2 * 1024 * 1024;

    if src.len() < PAR_MIN {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    // `dst` is valid for `src.len()` bytes and the chunks are disjoint, so the per-chunk raw
    // writes never alias. `usize` is carried across the thread boundary because `*mut u8` is
    // not `Send`.
    let base = dst as usize;
    src.par_chunks(PAR_CHUNK).enumerate().for_each(|(i, c)| {
        let off = i * PAR_CHUNK;
        unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), (base + off) as *mut u8, c.len()) };
    });
}

/// A ring of REUSED, fixed-size staging buffers — the DEFAULT weight-upload path on every device
/// (it measured faster than writing straight into ReBAR VRAM; see [`probe_rebar_type`]).
///
/// The old path allocated a fresh DEDICATED staging buffer as large as each tensor, memcpy'd into
/// it, submitted a copy, then `vkQueueWaitIdle`'d and freed it — per tensor. That serialized the
/// host memcpy against the DMA and paid an allocate/submit/stall/free cycle 443 times.
///
/// Here the ring is allocated ONCE per load. Big tensors are chunked across slots, and each slot
/// carries its own command buffer + fence, so while slot N's DMA is in flight the host is already
/// memcpy'ing into slot N+1 — the copy engine and the CPU overlap instead of taking turns. A slot
/// is only waited on when it is reused (its fence), never after every tensor. That overlap is the
/// whole point: it hides the PCIe crossing behind a full-speed system-RAM memcpy.
struct StagingRing {
    /// Fixed-size host-visible staging slots (`RING_SLOTS` × `RING_SLOT_BYTES`).
    bufs: Vec<VkBuffer>,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    /// Whether slot `i` has work in flight that its fence must be waited on before reuse.
    busy: Vec<bool>,
    next: usize,
}

/// Staging-ring geometry: enough slots to keep the copy engine fed while the host fills the next.
const RING_SLOTS: usize = 4;
const RING_SLOT_BYTES: usize = 32 * 1024 * 1024;

/// One large device-local memory block the weight arena bump-allocates from.
struct ArenaBlock {
    memory: vk::DeviceMemory,
    size: u64,
    cursor: u64,
}

/// Pre-reserved VRAM for load-once weights: N big blocks sized to the model (each ≤
/// `maxMemoryAllocationSize` — a whole multi-GiB model can't be one `vkAllocateMemory`), bump-
/// allocated since weights are never individually freed. Reserving the whole model up front makes
/// it OWN its VRAM the instant the load starts and frees it in one shot.
/// MoE-ready: a future expert-streaming mode can hold a second arena/pool and evict experts into it
/// without disturbing the dense arena.
struct WeightArena {
    mem_type: u32,
    blocks: Vec<ArenaBlock>,
    /// Index of the block `bump` is currently filling. Weights bump sequentially and a single
    /// tensor never straddles a block boundary, so once a tensor doesn't fit the current block the
    /// cursor advances to the next PRE-RESERVED block (its tail is left as slack) — and only when
    /// every reserved block is exhausted does `bump` grow a fresh overflow block. Monotonic: it
    /// never revisits an earlier block. (The old single-`blocks.last()` bump stranded every
    /// pre-reserved block but the last, double-committing VRAM — see `reserve_weights`.)
    cur: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl WeightArena {
    /// Whether a `size`-at-`align` bump fits some ALREADY-COMMITTED block (the current one or a
    /// later pre-reserved one) WITHOUT growing a fresh overflow block — i.e. whether
    /// [`bump`](Self::bump) would commit new device memory. The budget guard checks this before
    /// bumping so only real new commitments are charged against the budget.
    fn fits(&self, size: u64, align: u64) -> bool {
        self.blocks[self.cur.min(self.blocks.len())..]
            .iter()
            .any(|b| {
                let off = b.cursor.div_ceil(align) * align;
                off + size <= b.size
            })
    }

    /// Bump-allocate `size` bytes at `align`. Walks the pre-reserved blocks from `cur` forward,
    /// placing the tensor in the first that fits (advancing `cur` past any it overruns); if none
    /// fit, grows a fresh overflow block (its bytes charged to `used` — the budget guard's fallback
    /// accounting). Returns the device memory + offset to bind a buffer to.
    fn bump(
        &mut self,
        device: &ash::Device,
        size: u64,
        align: u64,
        used: &AtomicU64,
    ) -> Result<(vk::DeviceMemory, u64)> {
        // Try the current block, then walk forward through the remaining pre-reserved blocks.
        while self.cur < self.blocks.len() {
            let b = &mut self.blocks[self.cur];
            let off = b.cursor.div_ceil(align) * align;
            if off + size <= b.size {
                b.cursor = off + size;
                return Ok((b.memory, off));
            }
            // Doesn't fit here. Advance to the next pre-reserved block if there is one; otherwise
            // drop out and grow an overflow block below.
            if self.cur + 1 < self.blocks.len() {
                self.cur += 1;
                continue;
            }
            break;
        }
        let bs = size.max(ARENA_OVERFLOW_BLOCK);
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(bs)
                    .memory_type_index(self.mem_type),
                None,
            )
        }
        .map_err(|e| be(format!("arena overflow allocate_memory({bs}): {e}")))?;
        used.fetch_add(bs, Ordering::Relaxed);
        self.blocks.push(ArenaBlock {
            memory,
            size: bs,
            cursor: size,
        });
        self.cur = self.blocks.len() - 1;
        Ok((memory, 0))
    }

    /// Free all blocks. Must be called before `destroy_device`.
    unsafe fn destroy(&mut self, device: &ash::Device) {
        for b in self.blocks.drain(..) {
            device.free_memory(b.memory, None);
        }
    }
}

// ── resident-BDA weight arena (INFR_RESIDENT_BDA) ──────────────────────────────
//
// Allocation-side plumbing for moving resident weights out of per-tensor SSBOs and into big BDA
// arena blocks — the same `bufferDeviceAddress` scheme the paged-MoE/dense-streaming kernels
// already read weights through (see `pager.rs`, `alloc_arena_bda`), just applied to the RESIDENT
// path instead of a paged/streamed one. Nothing in `adapter.rs`/`recorder.rs`'s dispatch logic
// reads a sub-tensor's `device_addr()` yet — that wiring is a following slice. Default OFF
// (`INFR_RESIDENT_BDA` unset): `make_alloc` never calls into any of this, so behavior is unchanged.

/// `INFR_RESIDENT_BDA=1` — opt in to sub-allocating resident weight tensors from
/// [`BdaWeightArena`] blocks instead of the plain per-tensor / [`WeightArena`] paths. Read once and
/// cached: flipping the env var mid-process would split one model load's tensors across two
/// allocation strategies, which nothing downstream (a sub-tensor's `Drop`/`device_addr`) expects to
/// handle. Default OFF — zero behavior change anywhere in this crate.
fn resident_bda_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("INFR_RESIDENT_BDA").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// Byte alignment for resident-BDA weight sub-tensors within a block. Sub-tensors share one buffer
/// object rather than getting their own `VkMemoryRequirements`, so this is a fixed constant rather
/// than a probed one — 256 comfortably covers every access width a weight-reading shader uses (the
/// widest today is a 16-byte vec4 load) with headroom to spare for future wider kernels.
const BDA_WEIGHT_ALIGN: u64 = 256;

/// Minimum size for a resident-BDA arena block (see [`VulkanBackend::bda_weight_alloc`]). Blocks
/// are opened ON DEMAND — one per call that outgrows the current block's remainder — so without a
/// floor a run of small tensors would each pay for a separate dedicated `vkAllocateMemory`. Reuses
/// [`ARENA_OVERFLOW_BLOCK`]'s size: the same "big enough to amortize, small enough not to waste"
/// reasoning applies unchanged.
const BDA_BLOCK_MIN: u64 = ARENA_OVERFLOW_BLOCK;

/// Keepalive + addressing info for one [`BdaWeightArena`] block: a single dedicated
/// `bufferDeviceAddress` buffer (exactly what [`VulkanBackend::alloc_arena_bda`] builds) that
/// resident weight tensors bump-allocate BYTE RANGES within, never separate buffer objects (see
/// [`Backing::BdaSub`]). Held behind an `Arc` so a sub-tensor's `VkBuffer` can keep the block (and
/// therefore its memory and buffer handle) alive without owning it: the block is destroyed by
/// `buf`'s own `Drop` exactly once, when the last `Arc` clone — the arena's own plus every live
/// sub-tensor's — goes away. Never stored directly on `VulkanShared` (see
/// `VulkanBackend::bda_weight_arena`'s doc for why: `buf` holds an `Arc<VulkanShared>` clone, so
/// that would form a reference cycle and leak the whole device, exactly the bug
/// `backend_drop_frees_device_after_moe_pager` guards against for `moe_pager`/`dense_pager`).
struct BdaBlockHandle {
    /// The whole block's buffer: a `force_dedicated`, `device_address` allocation
    /// (`Backing::Pooled`), built by `make_buf_ex` exactly like `alloc_arena_bda`'s paged-MoE arena.
    buf: VkBuffer,
    /// `vkGetBufferDeviceAddress(buf.buffer)` — this block's byte-0 device address. A sub-tensor's
    /// `Buffer::device_addr` is `base_addr + sub_offset`.
    base_addr: u64,
}

/// One [`BdaWeightArena`] block: the shared keepalive handle plus its live bump cursor. Split apart
/// from `BdaBlockHandle` so the cursor can be mutated (behind `VulkanBackend::bda_weight_arena`'s
/// `Mutex`) while sub-tensors hold their own `Arc` clone of the handle without needing `&mut`
/// through it.
struct BdaArenaBlock {
    handle: Arc<BdaBlockHandle>,
    /// This block's total capacity in bytes (`<= max_mem_alloc_size`).
    size: u64,
    /// Next free byte offset (pre-alignment). Monotonic — sub-tensors are never freed individually.
    cursor: u64,
}

/// `INFR_RESIDENT_BDA=1` weight sub-allocator (see [`resident_bda_enabled`]). Unlike [`WeightArena`]
/// (which pre-reserves the WHOLE model's footprint up front via `reserve_weights`), blocks here are
/// created ON DEMAND as `bda_weight_alloc` calls outgrow the current block — this slice is pure
/// allocation/upload plumbing and doesn't thread a loader's total through this path; an up-front
/// version can be layered on top of the same block/bump primitives later. Each block is capped at
/// `max_mem_alloc_size` (same reasoning as `WeightArena`'s block cap: a whole multi-GiB model can't
/// be one `vkAllocateMemory`); a tensor never straddles a block boundary — one that doesn't fit the
/// current block's remainder opens a fresh one.
#[derive(Default)]
struct BdaWeightArena {
    blocks: Vec<BdaArenaBlock>,
}

// ── VulkanBackend ─────────────────────────────────────────────────────────────

/// Vulkan device + allocator + pipeline cache.
pub struct VulkanBackend {
    // NOTE: `moe_pager` is declared before `shared` so the session's buffers are freed first on
    // drop (each holds its own `Arc<VulkanShared>` clone, so the device outlives them either way).
    /// Paged MoE expert cache (see `pager::MoePagerSession`) — `Some` only when the loaded model's
    /// expert banks don't fit VRAM and the seam's placement policy chose paging over the legacy
    /// host-visible split (see `infr-llama`'s `generate_dense_vulkan_session`). `None` is the
    /// overwhelming common case (fits resident) and costs nothing beyond one `Mutex` lock check
    /// per `Backend::moe_paged` call.
    ///
    /// Owned by the BACKEND handle, NOT `VulkanShared`: the session's arena/LUT/ring buffers each
    /// hold an `Arc<VulkanShared>` clone, so parking the session on `VulkanShared` formed an Arc
    /// CYCLE — the shared state (device, allocator, weight arena, the pager arenas themselves:
    /// ~23 GiB after a Scout load) never dropped until process exit, and every LATER model load
    /// in the same process hit the VRAM budget guard with "N GiB already in use" (the
    /// `cpu_backend` gpu_ test-suite flake; see `backend_drop_frees_device_after_moe_pager`).
    /// The session still lives exactly as long as a loaded paged model can be generated with:
    /// `infr-llama`'s sessions own the `VulkanBackend`, and a new backend is a new device whose
    /// buffers couldn't read the old session anyway.
    moe_pager: crate::pager::MoePagerCell,
    /// Dense layer-streaming cache (see `pager::DensePagerSession`) — `Some` only when the loaded
    /// DENSE model's per-layer weights don't fit VRAM and the seam's placement chose streaming.
    /// Same drop-ordering/ownership story as `moe_pager` (declared before `shared` so its
    /// arena/ring buffers free first; owned by the backend HANDLE, never `VulkanShared` — the Arc
    /// cycle lesson on `moe_pager`'s doc applies unchanged).
    dense_pager: crate::pager::DensePagerCell,
    /// `INFR_RESIDENT_BDA=1` resident-weight sub-allocator (see [`BdaWeightArena`],
    /// [`resident_bda_enabled`]) — `None` until the first resident-BDA weight alloc under the flag;
    /// `make_alloc` routes `BufferUsage::Weights` here instead of `make_buf` when it is set.
    ///
    /// Same drop-ordering/ownership story as `moe_pager`/`dense_pager` above and for the identical
    /// reason: each block's `BdaBlockHandle::buf` holds its own `Arc<VulkanShared>` clone, so
    /// parking this arena ON `VulkanShared` would form the same reference cycle that leaked the
    /// whole device for a paged-MoE session before `moe_pager` was moved off it — see
    /// `backend_drop_frees_device_after_moe_pager`.
    bda_weight_arena: Mutex<Option<BdaWeightArena>>,
    shared: Arc<VulkanShared>,
}

/// Device memory info for a backend's shared state — the body of [`VulkanBackend::vram`],
/// factored out so scopes that only hold the `Arc<VulkanShared>` (e.g. [`WeightProgress`]'s
/// post-load log) can read it too.
///
/// WHICH HEAPS COUNT — the whole point of this function, and the difference between refusing a
/// model the box could run and TDR-ing one it could not:
///
/// DISCRETE card — device-local heaps ONLY. The other heap is host RAM reachable over PCIe (GTT).
/// It is NOT capacity: RADV happily accepts a device-local allocation past the VRAM heap's size and
/// quietly spills the excess into GTT — MEASURED on a 7900 XTX, where a 41 GiB run of 1 GiB
/// device-local allocations succeeded and landed as `mem_info_vram_used` 23.08 GiB +
/// `mem_info_gtt_used` 18.01 GiB. Every byte on the GTT side is then read across PCIe at a fraction
/// of VRAM bandwidth. Counting it would turn a clean load error into a mysteriously slow model, so
/// the guard budgets device-local alone. THIS IS THE PRE-EXISTING BEHAVIOR AND MUST NOT CHANGE.
///
/// UNIFIED-MEMORY part (an APU — see [`Capabilities::unified_memory`]) — ALL heaps. There is no
/// VRAM here to spill out of; both heaps are the same DDR at the same bandwidth, and the driver's
/// split between them is bookkeeping, not physics. MEASURED on RADV RAPHAEL_MENDOCINO: it
/// advertises a 21.47 GiB "DEVICE_LOCAL" heap and a 10.73 GiB host-visible one, which sum to
/// EXACTLY `mem_info_vram_total` (2 GiB carveout) + `mem_info_gtt_total` (30.20 GiB) — RADV
/// synthesizes the device-local heap as 2/3 of (carveout + GTT). The same 41 GiB device-local probe
/// on that device landed as `mem_info_gtt_used` 30.01 GiB and `mem_info_vram_used` 1.03 GiB: the
/// "device-local" heap IS system RAM through the GART, and the 2 GiB carveout is not where the
/// weights go. So the honest capacity is the SUM of the heaps, and budgeting against the
/// device-local slice alone refuses models (gemma-4-31B UD-Q5_K_XL: 20.37 GiB of weights against a
/// 21.22 GiB budget) that fit the machine with room to spare.
///
/// Counting the overflow heap is only half of it — `probe_uma_overflow_type` is what lets bytes
/// actually LAND there once the device-local heap is full. Above the summed budget the failure mode
/// is the same on both classes (the driver oversubscribes and starts evicting), which is why the
/// guard exists at all — it just now guards the right number on each.
fn vram_info(s: &VulkanShared) -> VramInfo {
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceMemoryProperties2::default();
    if s.has_mem_budget {
        props2 = props2.push_next(&mut budget);
    }
    unsafe {
        s.instance
            .get_physical_device_memory_properties2(s.physical_device, &mut props2)
    };
    let mp = props2.memory_properties;
    let uma = s.caps.unified_memory;

    // Discrete: device-local heaps only. UMA: every heap (they are one pool of DDR). The live
    // VK_EXT_memory_budget figure is used on BOTH — it is what accounts for other processes, and
    // on a shared-memory part that matters more, not less: a second infr holding 21 GiB of the
    // same DDR is exactly the thing a UMA guard must see.
    let mut total = 0u64;
    let mut available = 0u64;
    for i in 0..mp.memory_heap_count as usize {
        let device_local = mp.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
        if uma || device_local {
            total += mp.memory_heaps[i].size;
            available += if s.has_mem_budget {
                // Live free = budget - usage (the budget is a CEILING, not free bytes). Clamped to
                // the heap size so a driver that reports usage past the heap (RADV on an APU, once
                // something has oversubscribed the synthetic split) can't hand back a bogus figure.
                budget.heap_budget[i]
                    .saturating_sub(budget.heap_usage[i])
                    .min(mp.memory_heaps[i].size)
            } else {
                mp.memory_heaps[i].size
            };
        }
    }
    VramInfo {
        total,
        available,
        live: s.has_mem_budget,
        uma,
    }
}

/// RAII scope for a weight-load progress bar (see [`VulkanBackend::weight_progress`]). While alive,
/// `BufferUsage::Weights` allocations advance the bar; on drop it finishes and clears it.
pub struct WeightProgress {
    shared: Arc<VulkanShared>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Drop for WeightProgress {
    fn drop(&mut self) {
        // Drain the staging ring FIRST: on the non-ReBAR path its copies are still in flight (we
        // fence per slot instead of stalling the queue per tensor), and the weights must be fully
        // resident before the loader records a forward. No-op on the direct-to-VRAM path, which
        // never builds a ring.
        self.shared.drain_staging_ring();
        self.shared.weights_direct.store(false, Ordering::Relaxed);
        if let Some(pb) = self.shared.weight_pb.lock().unwrap().take() {
            pb.finish_and_clear();
        }
        // Post-load memory-hygiene visibility (INFR_VRAM_LOG=1): the LIVE in-use figure right
        // after the LAST weight upload — the number the VRAM-audit residual math (in-use minus
        // weights+KV estimate) starts from. The upload staging that ran under this scope was
        // dedicated-allocated (see `Backend::upload`), so by this drop it has fully returned
        // its device memory; what remains is weights + already-allocated session buffers.
        if std::env::var("INFR_VRAM_LOG").is_ok() {
            let v = vram_info(&self.shared);
            eprintln!(
                "post-load vram in use: {:.2} GiB of {:.2} GiB ({})",
                v.total.saturating_sub(v.available) as f64 / (1u64 << 30) as f64,
                v.total as f64 / (1u64 << 30) as f64,
                if v.live { "live" } else { "tracked" },
            );
        }
    }
}

impl infr_core::backend::ProgressScope for WeightProgress {}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanBackend {
    /// `maxComputeSharedMemorySize` for the active device — the per-workgroup shared-memory budget
    /// the flash-attention tile height is sized against (cheap accessor; avoids cloning caps).
    pub fn max_shared_memory_bytes(&self) -> u32 {
        self.shared.caps.max_shared_memory_bytes
    }

    /// Borrowed capabilities — the kernel-tier fallback ladder's gate (`caps.f16_coopmat`,
    /// `caps.f16`, `caps.i8_dot`). Cheap: a reference, not the [`Backend::capabilities`] clone (which
    /// copies the `name: String`) — safe to call per-op inside the adapter's hot lowering loop.
    pub(crate) fn caps(&self) -> &Capabilities {
        &self.shared.caps
    }

    /// Initialize Vulkan: create instance, pick a GPU (prefer discrete), create a logical
    /// device + compute queue with the required extensions/features, set up the allocator.
    /// `Err` on Apple (Vulkan is unsupported there — use the Metal backend), `Ok(())` everywhere
    /// else. Split into two `cfg` bodies so the guard is a runtime `Result` the caller `?`s: that
    /// keeps the Vulkan body in [`new`](Self::new) compiling on macOS while never executing it.
    #[cfg(target_os = "macos")]
    fn reject_on_apple() -> Result<()> {
        Err(be(
            "Vulkan is not supported on Apple. Use the native Metal backend: it is the default on \
             macOS, or select it explicitly with `--dev metal` (or INFR_METAL=1). (The only Vulkan \
             on Apple is MoltenVK, which this backend deliberately does not target.)",
        ))
    }
    #[cfg(not(target_os = "macos"))]
    fn reject_on_apple() -> Result<()> {
        Ok(())
    }

    pub fn new() -> Result<Self> {
        // Apple: the Vulkan backend is DELIBERATELY unsupported (the only Vulkan on Apple is
        // MoltenVK, which lacks features this backend depends on — e.g. `bufferDeviceAddress` for
        // the paged MoE arena — and is slower than talking to Metal directly). infr ships a NATIVE
        // Metal backend for Apple GPUs. The `?` on a runtime `Result` here does NOT trip
        // `unreachable_code`, so the Vulkan body below still compiles clean on macOS (it is simply
        // never reached). See `reject_on_apple`.
        Self::reject_on_apple()?;

        // ── entry ──────────────────────────────────────────────────────────────
        let entry =
            unsafe { ash::Entry::load() }.map_err(|e| be(format!("ash::Entry::load: {e}")))?;

        // ── instance (Vulkan 1.3) ──────────────────────────────────────────────
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"infr")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"infr-vulkan")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        // Native Vulkan drivers only (Linux/Windows AMD/NVIDIA/Intel). The MoltenVK portability
        // opt-in that used to live here was removed with the Apple guard above — Apple is the only
        // place a portability driver is enumerated, and infr no longer runs Vulkan there.
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
        }
        .map_err(|e| be(format!("create_instance: {e}")))?;

        // ── physical device: `INFR_DEV` if set, else prefer discrete ──────────
        // `INFR_DEV=VulkanN` (set by the CLI's `--dev`) pins the Nth device in ENUMERATION order,
        // matching llama.cpp's `--dev VulkanN` naming so the two tools address the same GPU on a
        // multi-GPU box. Unset => the historical rule: first DISCRETE_GPU, else device 0.
        //
        // An out-of-range / unparseable INFR_DEV is a hard error, NOT a fallback: silently running
        // on a different GPU than the one asked for produces numbers that look plausible and are
        // wrong, which is far worse than refusing to start.
        let pdevices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| be(format!("enumerate_physical_devices: {e}")))?;
        if pdevices.is_empty() {
            return Err(be("no Vulkan physical devices"));
        }
        let physical_device = match std::env::var("INFR_DEV") {
            Ok(spec) => {
                let s = spec.trim();
                let idx_str = s
                    .strip_prefix("Vulkan")
                    .or_else(|| s.strip_prefix("vulkan"));
                let idx: usize = idx_str.unwrap_or(s).parse().map_err(|_| {
                    be(format!(
                        "INFR_DEV/--dev: expected `VulkanN` (e.g. Vulkan0, Vulkan1), got `{spec}`"
                    ))
                })?;
                *pdevices.get(idx).ok_or_else(|| {
                    let names: Vec<String> = pdevices
                        .iter()
                        .enumerate()
                        .map(|(i, &pd)| {
                            let p = unsafe { instance.get_physical_device_properties(pd) };
                            let n = unsafe { CStr::from_ptr(p.device_name.as_ptr()) }
                                .to_string_lossy()
                                .into_owned();
                            format!("Vulkan{i}={n}")
                        })
                        .collect();
                    be(format!(
                        "INFR_DEV/--dev `{spec}`: no such Vulkan device (this system has {}: {})",
                        pdevices.len(),
                        names.join(", ")
                    ))
                })?
            }
            Err(_) => pdevices
                .iter()
                .copied()
                .find(|&pd| {
                    let p = unsafe { instance.get_physical_device_properties(pd) };
                    p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .unwrap_or(pdevices[0]),
        };

        // ── compute queue family ───────────────────────────────────────────────
        // The first COMPUTE-capable family. On amdgpu this is the universal (graphics) family, so
        // the work lands on the `gfx` ring. Moving it to a compute-ONLY family (the `comp` rings)
        // was tried and is strictly WORSE on the surveyed integrated part: the same ~2 s forward
        // that is an intermittent device-lost on `gfx` is a DETERMINISTIC one on `comp` (measured
        // 4/4 runs), i.e. the compute ring's effective hang budget there is TIGHTER, not the 60 s
        // its module-parameter default advertises. The submit splitter is what actually bounds the
        // job (see `VulkanShared::submit_dispatch_cap`); the ring choice is not a lever.
        let qf_props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = qf_props
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|i| i as u32)
            .ok_or_else(|| be("no compute queue family found"))?;

        // ── probe device extensions ────────────────────────────────────────────
        let avail_exts = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(|e| be(format!("enumerate device extensions: {e}")))?;

        let has_ext = |name: &CStr| -> bool {
            avail_exts
                .iter()
                .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) == name })
        };

        let has_coop_matrix = has_ext(c"VK_KHR_cooperative_matrix");
        let has_16bit_storage = has_ext(c"VK_KHR_16bit_storage");
        let has_8bit_storage = has_ext(c"VK_KHR_8bit_storage");
        // Packed i8 (int8) dot (dp4a) — the decode i8 `mmv` accumulate. Promoted to core in Vulkan
        // 1.3; probed via the KHR ext for pre-1.3 drivers. Detection-only here (caps.i8_dot); the
        // adapter's i8-mmv gate consults it so a device without packed dot routes to the scalar
        // dequant GEMV instead of dispatching a dp4a kernel it can't run.
        let has_i8_dot_ext = has_ext(c"VK_KHR_shader_integer_dot_product");
        // f8 (== fp8, E4M3/E5M2) storage/convert support. ash 0.38 has no constant for the ext, so
        // match the raw name. Absent on RDNA3 → caps.f8 false.
        let has_f8_ext = has_ext(c"VK_EXT_shader_float8");
        // bf16 (bfloat16) storage/convert. ash 0.38 has no constant for the ext → match the raw
        // name. Absent on RDNA3 → caps.bf16 false; present on RDNA4/Navi44.
        let has_bf16_ext = has_ext(c"VK_KHR_shader_bfloat16");
        let has_subgroup_ext = has_ext(c"VK_KHR_shader_subgroup_extended_types");
        let has_mem_budget = has_ext(c"VK_EXT_memory_budget");
        // Lets every dispatch bind its buffers with one `cmd_push_descriptor_set` recorded
        // straight into the command buffer instead of `alloc_set` (pool allocate) +
        // `update_descriptor_sets` (a separate driver call) + `cmd_bind_descriptor_sets` per op —
        // measured as a real per-forward host-side cost at small-m shapes (many-op graphs where
        // GPU busy time is small, so the fixed per-dispatch descriptor churn is a bigger fraction
        // of wall time). Near-universally supported (desktop RADV/NVIDIA/Intel); the pooled path
        // stays as a fallback for drivers that lack it (e.g. some portability/MoltenVK builds).
        // INFR_NO_PUSH_DESC=1 forces the pooled-classic fallback even when the extension exists —
        // lets a RADV dev box exercise the code path a driver WITHOUT push descriptors takes
        // (field report: teardown validation findings on Intel Arc/ANV that RADV runs never
        // reproduce because the classic pools are never created here). Test/diagnosis knob only.
        let has_push_descriptor =
            has_ext(c"VK_KHR_push_descriptor") && std::env::var("INFR_NO_PUSH_DESC").is_err();

        // ── probe features (via VK 1.1 get_physical_device_features2) ─────────
        // Memory model and subgroup-size-control are probed rather than assumed: a portability
        // device (MoltenVK) may lack either, and enabling an unsupported feature fails
        // create_device outright.
        let mut f16_feat = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut memmodel_feat = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
        let mut sgsize_feat = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        let mut coopmat_feat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut intdot_feat = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        // Buffer-device-address: lets a shader read a buffer via a 64-bit `VkDeviceAddress`
        // (`GL_EXT_buffer_reference`), bypassing one SSBO binding's `maxStorageBufferRange` (~4 GiB
        // on RADV). infr's paged-MoE expert arena REQUIRES it — a per-role pool now spans as much
        // VRAM as the budget allows, addressed by a raw pointer. Core in Vulkan 1.2, so it is
        // hard-required below (not an opt-in ladder like coopmat).
        let mut bda_feat = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut feat2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f16_feat)
            .push_next(&mut memmodel_feat)
            .push_next(&mut sgsize_feat)
            .push_next(&mut coopmat_feat)
            .push_next(&mut intdot_feat)
            .push_next(&mut bda_feat);
        unsafe { instance.get_physical_device_features2(physical_device, &mut feat2) };
        // Core Vulkan 1.0 feature (no extension struct — `get_physical_device_features2` always
        // populates the chain's base `.features`). Several KV-cache dequant/attention shaders
        // (dequant_turbo_f16.comp, dequant_q8_f16.comp, attn_*.comp) declare SPIR-V's `Int16`
        // capability (16-bit integer arithmetic, e.g. GL_EXT_shader_explicit_arithmetic_types_int16
        // int16_t/uint16_t locals — distinct from `storageBuffer16BitAccess`, which only covers
        // 16-bit SSBO/UBO *storage*, not arithmetic): VUID-VkShaderModuleCreateInfo-pCode-08740
        // requires `shaderInt16` enabled on the DEVICE for that capability, same class of bug as the
        // `shaderIntegerDotProduct` one fixed below — detected via caps but never chained into
        // `device_ci`, so vkCreateShaderModule for those kernels violated the VUID under validation.
        let has_int16 = feat2.features.shader_int16 != 0;
        // Same 08740 class as `shaderInt16` above, for 64-bit integer arithmetic: the BDA arena
        // helper `native_arena_ref.glsl` (paged-MoE `-DPAGED` builds AND dense-streaming
        // `-DSTREAMED` builds) composes a 64-bit slot address from lo/hi u32 halves
        // (`uint64_t(hi) << 32 | uint64_t(lo)`, `GL_EXT_shader_explicit_arithmetic_types_int64`),
        // which emits SPIR-V's `Int64` capability — so `shaderInt64` MUST be enabled on the device
        // or vkCreateShaderModule for those kernels violates the VUID under validation. Core 1.0
        // feature; RADV/desktop support it universally, probed here for portability devices.
        let has_int64 = feat2.features.shader_int64 != 0;
        // Read AFTER the `feat2.features` access above: `feat2` holds a mutable borrow of every
        // pushed feature struct (incl. `bda_feat`) until its last use, so the pushed structs can
        // only be read once `feat2` itself is done being touched.
        let has_bda = bda_feat.buffer_device_address != 0;
        // Hard requirement, not a fallback: the paged-MoE arena is addressed by a 64-bit device
        // pointer, so a device that cannot hand out one has no 64-bit address space for infr to
        // use. bufferDeviceAddress is core in Vulkan 1.2 and this backend targets 1.3, so on any
        // real target this never fires — it is the clean guard for a driver/portability layer that
        // somehow omits it, failing at init with a clear message instead of miscompiling a shader.
        if !has_bda {
            let p = unsafe { instance.get_physical_device_properties(physical_device) };
            let name = unsafe { CStr::from_ptr(p.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let (major, minor, patch) = (
                vk::api_version_major(p.api_version),
                vk::api_version_minor(p.api_version),
                vk::api_version_patch(p.api_version),
            );
            return Err(be(format!(
                "this GPU/driver does not support bufferDeviceAddress (the 64-bit shader address \
                 space), which infr's paged-MoE arena requires — {name} / Vulkan \
                 {major}.{minor}.{patch}. bufferDeviceAddress is core in Vulkan 1.2; update the \
                 driver or use a device that exposes it."
            )));
        }
        let has_f16 = f16_feat.shader_float16 != 0;
        let has_memmodel = memmodel_feat.vulkan_memory_model != 0;
        let has_memmodel_dev = memmodel_feat.vulkan_memory_model_device_scope != 0;
        let has_sgsize = sgsize_feat.subgroup_size_control != 0;
        let has_full_sg = sgsize_feat.compute_full_subgroups != 0;
        // Packed i8 dot: ext advertised AND the feature bit set (same ext-AND-feature discipline as
        // coopmat). Detection-only — the current i8 mmv is DEFAULT-OFF at m=1 (scalar wins), and no
        // shader here uses the ext builtin yet, so we don't add it to the enabled feature chain; the
        // adapter's i8-mmv gate reads `caps.i8_dot` before ever dispatching a dp4a kernel.
        let has_i8_dot = has_i8_dot_ext && intdot_feat.shader_integer_dot_product != 0;
        // i8 (int8) shader storage/math — the same `shaderFloat16Int8` feature struct carries it.
        let has_int8 = f16_feat.shader_int8 != 0;
        // Extension presence alone doesn't guarantee the FEATURE bit (a driver may advertise
        // VK_KHR_cooperative_matrix with cooperativeMatrix=false — enabling it then fails
        // create_device, the same failure class #32 fixed for memmodel/sgsize). This is the
        // PREREQUISITE (unit exists + usable); which COMPONENT TYPES it accepts is a separate
        // enumeration below — the ext bit does NOT imply f16 support (the spec only promises a unit
        // exists, not that it does f16), so we don't assume it.
        let has_coop_ext_feat = has_coop_matrix && coopmat_feat.cooperative_matrix != 0;

        // Enumerate the device's cooperative-matrix configs ONCE — the AUTHORITATIVE source for
        // which component types AND tile dimensions the matrix unit accepts. Each config lists
        // m/n/k size + a/b/c/result types; the ext's presence alone tells us nothing about them.
        // Empty when the ext/feature is absent. Extract a Copy tuple — the returned structs borrow
        // the loader `cm`, so we can't let them outlive this block.
        type CoopmatConfig = (
            u32,
            u32,
            u32,
            vk::ComponentTypeKHR,
            vk::ComponentTypeKHR,
            vk::ComponentTypeKHR,
            vk::ComponentTypeKHR,
        );
        let coopmat_configs: Vec<CoopmatConfig> = if has_coop_ext_feat {
            let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
            unsafe { cm.get_physical_device_cooperative_matrix_properties(physical_device) }
                .map(|v| {
                    v.iter()
                        .map(|p| {
                            (
                                p.m_size,
                                p.n_size,
                                p.k_size,
                                p.a_type,
                                p.b_type,
                                p.c_type,
                                p.result_type,
                            )
                        })
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        // Diagnostic: dump every enumerated (M,N,K,aType,bType,cType,resultType) — the definitive
        // list of what the matrix unit accepts, for bringing up new HW (RDNA4 fp8, bf16, Intel's
        // 8/8/16 tiles) and sanity-checking the per-type/per-dim detection below.
        // `INFR_DEBUG_COOPMAT=1`.
        if std::env::var("INFR_DEBUG_COOPMAT").is_ok() {
            let raws: Vec<(u32, u32, u32, i32, i32, i32, i32)> = coopmat_configs
                .iter()
                .map(|&(m, n, k, a, b, c, r)| {
                    (m, n, k, a.as_raw(), b.as_raw(), c.as_raw(), r.as_raw())
                })
                .collect();
            eprintln!(
                "[infr] coopmat configs (M,N,K,aType_raw,bType_raw,cType_raw,resultType_raw): \
                 {raws:?}"
            );
        }
        // f16 coopmat: configs with f16 A AND B operands (accumulator/result f16 or f32), reduced
        // to ONE chosen (M,N,K) tile by `select_coopmat_shape`'s preference order: 16x16x16 first
        // (the shape every production coopmat shader is built for — every `coopmat<...,16,16,...>`
        // declaration across gemm_coopmat*/gemm_warp/native_gemm*/attn_*/deltanet_prep), then
        // 8x8x16 (Intel Arc/ANV XMX — ONLY under the `INFR_CM_8X8=1` opt-in, and only the
        // `native_gemm_warp` `_cm8` builds exist at that shape). Component types alone are NOT
        // sufficient — an Intel A770 (Mesa ANV) advertises f16×f16→f32 only at M=8,N=8,K=16;
        // creating our 16x16x16 pipeline on such a device silently fails
        // vkCreateComputePipelines (the segfault bug — the result wasn't checked, see
        // `create_compute_pipeline` below). Requiring the shape match here makes an unsupported
        // device fall back to the non-coopmat ladder instead of crashing. Derived from the
        // enumeration, not assumed from the ext bit. `has_coop_matrix` (any usable f16 shape)
        // keeps its downstream role (ext-enable, feature chain).
        let f16c = vk::ComponentTypeKHR::FLOAT16;
        let cm8_env = std::env::var("INFR_CM_8X8").is_ok();
        let coopmat_f16 = select_coopmat_shape(
            coopmat_configs
                .iter()
                .filter(|&&(_, _, _, a, b, _, _)| a == f16c && b == f16c)
                .map(|&(m, n, k, ..)| (m, n, k)),
            cm8_env,
        );
        // Extension-added ComponentTypeKHR raw values (ash 0.38 predates these variants, so match by
        // raw i32). CONFIRMED on RDNA4/Navi44 via INFR_DEBUG_COOPMAT — all CORE types are 0..=10
        // (FLOAT16=0/FLOAT32=1/SINT8=3/UINT8=7/…), these are the KHR-standard ext values:
        const CT_E4M3: i32 = 1_000_491_002; // VK_COMPONENT_TYPE_FLOAT_E4M3_KHR
        const CT_E5M2: i32 = 1_000_491_003; // VK_COMPONENT_TYPE_FLOAT_E5M2_KHR
        const CT_BF16: i32 = 1_000_141_000; // VK_COMPONENT_TYPE_BFLOAT16_KHR
        let is_f8 = |t: i32| t == CT_E4M3 || t == CT_E5M2;
        // f8 coopmat: configs with fp8 (E4M3/E5M2) A AND B operands, 16x16x16 ONLY (no f8 shader
        // exists at any other shape → `allow_8x8x16 = false`). Uses the KHR-standard fp8 raw
        // values (confirmed on RDNA4) rather than the older `>= 1e9` heuristic, which also
        // matched bf16 (`CT_BF16` is ext-range too) and would false-positive f8 on a bf16-only
        // unit. Also requires the float8 storage ext. NEVER Some on RDNA3 (enumerates no fp8
        // config).
        let coopmat_f8 = select_coopmat_shape(
            coopmat_configs
                .iter()
                .filter(|&&(_, _, _, a, b, _, _)| is_f8(a.as_raw()) && is_f8(b.as_raw()))
                .map(|&(m, n, k, ..)| (m, n, k)),
            false,
        )
        .filter(|_| has_f8_ext);
        // bf16 coopmat: BFLOAT16 A AND B operands, 16x16x16 only. Confirmed on RDNA4/Navi44
        // (bf16×bf16→bf16 and →f32); RDNA3 enumerates none. Same discipline as f8/f16 above.
        let coopmat_bf16 = select_coopmat_shape(
            coopmat_configs
                .iter()
                .filter(|&&(_, _, _, a, b, _, _)| a.as_raw() == CT_BF16 && b.as_raw() == CT_BF16)
                .map(|&(m, n, k, ..)| (m, n, k)),
            false,
        );
        // i8 coopmat: configs with SINT8 A AND B operands and a SINT32 result, 16x16x16 only (the
        // shape every int8 coopmat shader here uses) — same discipline as `coopmat_f16`'s shape
        // selection above. DETECTION ONLY (see the `coopmat_i8` doc comment on `Capabilities`):
        // the standalone `coopmat_int8_test` harness confirmed this exact config
        // (SINT8xSINT8->SINT32, subgroup-pinned 32, A RowMajor/B ColumnMajor) dispatches correctly
        // on this driver, but int8 coopmat hung an OLDER Mesa (commit ad82a77) despite enumerating
        // fine there too — so detection alone does NOT make this capability a safe default; the
        // adapter requires `INFR_I8_COOPMAT=1` in addition to `caps.i8_coopmat()` before ever
        // dispatching the kernel.
        let i8c = vk::ComponentTypeKHR::SINT8;
        let i32c = vk::ComponentTypeKHR::SINT32;
        let coopmat_i8 = select_coopmat_shape(
            coopmat_configs
                .iter()
                .filter(|&&(_, _, _, a, b, _, r)| a == i8c && b == i8c && r == i32c)
                .map(|&(m, n, k, ..)| (m, n, k)),
            false,
        );

        // ── force-disable capabilities for fallback-path testing on capable HW ──
        // These env knobs drop a DETECTED capability so the next kernel tier down is exercised on a
        // device that actually has the feature — otherwise the portability fallbacks are only
        // reachable on hardware we may not own. Applied before the ext list / feature chain so a
        // forced-off feature is genuinely NOT enabled on the device (a faithful simulation, not just
        // a caps flag flip). f16 is a coopmat prerequisite, so INFR_NO_F16 ⇒ NO coopmat too.
        let has_f16 = has_f16 && std::env::var("INFR_NO_F16").is_err();
        let coopmat_f16 = coopmat_f16
            .filter(|_| has_f16 && std::env::var("INFR_NO_COOPMAT").is_err())
            .filter(|&s| {
                // 8x8x16 additionally needs a pinnable subgroup size 16: the `_cm8` builds run
                // 128 threads = 8 warps × 16 lanes (XMX/DPAS is SIMD16-native) and reuse the
                // kernel's 8-warp index math, so a device that can't pin 16 can't run them.
                // (Checked here so the coopmat ext is never enabled for a shape we then refuse.
                // The physical-device subgroup range is queried again for `caps` below; this
                // early copy exists because that query currently runs after device creation.)
                if s != infr_core::COOPMAT_TILE_8 {
                    return true;
                }
                let mut sgp = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sgp);
                unsafe { instance.get_physical_device_properties2(physical_device, &mut p2) };
                has_sgsize && sgp.min_subgroup_size <= 16 && 16 <= sgp.max_subgroup_size
            });
        // `has_coop_matrix` = ANY usable f16 coopmat shape — drives the ext enable + feature
        // chain below. On a 16x16x16 device this is exactly the old boolean; on an 8x8x16-only
        // device it is false unless INFR_CM_8X8=1 selected the 8x8x16 shape above (default OFF:
        // the ext is then NOT enabled, byte-identical to the pre-shape-table behavior there).
        let has_coop_matrix = coopmat_f16.is_some();
        // f8 coopmat is a coopmat sub-tier, so dropping coopmat drops it too.
        let coopmat_f8 = coopmat_f8.filter(|_| has_coop_matrix);
        // bf16 coopmat: same coopmat sub-tier dependency (rides the coopmat device-feature enable).
        let coopmat_bf16 = coopmat_bf16.filter(|_| has_coop_matrix);
        // i8 coopmat rides the SAME device feature enable (coopmat_ci is only chained into
        // device_ci below when `has_coop_matrix`) — without it the extension isn't enabled on the
        // logical device even if int8 configs were enumerated, so this is a real dependency, not
        // just symmetry with coopmat_f8 above. INFR_NO_COOPMAT/INFR_NO_F16 drop it too.
        let coopmat_i8 = coopmat_i8.filter(|_| has_coop_matrix);
        let has_i8_dot = has_i8_dot && std::env::var("INFR_NO_I8DOT").is_err();
        // INFR_CM_8X8=1 outcome notice (once, at device init): the tester A/B knob must be loud
        // about whether it actually engaged — on RADV (16x16x16 enumerated) or any device without
        // an 8x8x16 f16 config it changes NOTHING, and the kernel set stays identical.
        if cm8_env {
            match coopmat_f16 {
                Some(infr_core::COOPMAT_TILE_8) => eprintln!(
                    "[infr] INFR_CM_8X8=1: 8x8x16 f16 coopmat selected — native_gemm_warp _cm8 \
                     prefill tier live (other coopmat families stay on their non-coopmat \
                     fallbacks)"
                ),
                Some(_) => eprintln!(
                    "[infr] INFR_CM_8X8=1 has no effect: device provides the default 16x16x16 \
                     f16 coopmat tile — kernel set unchanged"
                ),
                None => eprintln!(
                    "[infr] INFR_CM_8X8=1 has no effect: device enumerates no usable 8x8x16 f16 \
                     coopmat config (or coopmat is disabled) — kernel set unchanged"
                ),
            }
        }
        // Extend the INFR_DEBUG_COOPMAT dump with the CHOSEN shape per component type (the raw
        // enumeration is printed above, before selection).
        if std::env::var("INFR_DEBUG_COOPMAT").is_ok() {
            eprintln!(
                "[infr] coopmat chosen shapes (M,N,K): f16={coopmat_f16:?} bf16={coopmat_bf16:?} \
                 f8={coopmat_f8:?} i8={coopmat_i8:?}"
            );
        }

        // ── build extension name list (only available ones) ────────────────────
        let mut ext_ptrs: Vec<*const i8> = Vec::new();
        if has_coop_matrix {
            ext_ptrs.push(c"VK_KHR_cooperative_matrix".as_ptr());
        }
        if has_16bit_storage {
            ext_ptrs.push(c"VK_KHR_16bit_storage".as_ptr());
        }
        if has_8bit_storage {
            ext_ptrs.push(c"VK_KHR_8bit_storage".as_ptr());
        }
        if has_subgroup_ext {
            ext_ptrs.push(c"VK_KHR_shader_subgroup_extended_types".as_ptr());
        }
        if has_mem_budget {
            ext_ptrs.push(c"VK_EXT_memory_budget".as_ptr());
        }
        if has_push_descriptor {
            ext_ptrs.push(c"VK_KHR_push_descriptor".as_ptr());
        }
        // The int8 dp4a decode GEMVs (native_mmv.comp, native_mmv_mrow.comp, native_mmv_id_q4k.comp,
        // mul_mat_vec_q.comp's dotPacked builtins) compile to SPIR-V with the DotProduct /
        // DotProductInput4x8BitPacked capabilities, which VUID-VkShaderModuleCreateInfo-pCode-08740
        // requires `shaderIntegerDotProduct` to be enabled on the DEVICE for — not just detected.
        // This was previously probed into `caps.i8_dot` (detection-only, per an now-stale comment
        // claiming no shader used the builtin yet) but never actually enabled, so vkCreateShaderModule
        // for those kernels violated the VUID on any driver that validates it (reproduced on the
        // 7900 XTX under validation layers with an 8B model, which is wide enough to select the mmv
        // dp4a tier — the small model's shapes never hit it, hence the bug staying latent).
        if has_i8_dot {
            ext_ptrs.push(c"VK_KHR_shader_integer_dot_product".as_ptr());
        }
        // A portability (layered) device REQUIRES VK_KHR_portability_subset to be enabled when
        // it advertises it (Vulkan valid-usage rule); MoltenVK does.
        if has_ext(c"VK_KHR_portability_subset") {
            ext_ptrs.push(c"VK_KHR_portability_subset".as_ptr());
        }

        // ── logical device ─────────────────────────────────────────────────────
        let priorities = [1.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        // Feature chain — needed for cooperative-matrix kernels:
        //   shaderFloat16 (f16 math), 16-bit storage (f16 SSBOs), Vulkan memory model
        //   (required by coopmat), cooperativeMatrix itself.
        let mut shader_f16_ci = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(has_f16)
            .shader_int8(true);
        let mut storage16_ci = vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(has_16bit_storage);
        let mut storage8_ci = vk::PhysicalDevice8BitStorageFeatures::default()
            .storage_buffer8_bit_access(has_8bit_storage);
        let mut memmodel_ci = vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
            .vulkan_memory_model(has_memmodel)
            .vulkan_memory_model_device_scope(has_memmodel_dev);
        // Chained below only when `has_coop_matrix` (ext AND probed feature) — see the probe.
        let mut coopmat_ci =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        // Lets us pin the subgroup size to 32 (RDNA3 coopmat is wave32) for the tiled GEMM.
        let mut sgsize_ci = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
            .subgroup_size_control(has_sgsize)
            .compute_full_subgroups(has_full_sg);
        // Chained below only when `has_i8_dot` — see the ext_ptrs comment above.
        let mut intdot_ci = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
            .shader_integer_dot_product(true);
        // Buffer-device-address — hard-required above, so always enabled (the paged-MoE arena is
        // addressed by a `VkDeviceAddress`). Core in 1.2, promoted from VK_KHR_buffer_device_address,
        // so it needs no device extension on a 1.3 device — only the feature enable.
        let mut bda_ci =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);

        // Core 1.0 features (shaderInt16 — see the probe comment above): passed via
        // `enabled_features`, NOT a pNext-chained `PhysicalDeviceFeatures2` (the two are mutually
        // exclusive per the spec; this device_ci never chains `PhysicalDeviceFeatures2` itself, only
        // extension-specific feature structs, so `enabled_features` is the correct, conflict-free
        // slot for it).
        let core_features = vk::PhysicalDeviceFeatures::default()
            .shader_int16(has_int16)
            .shader_int64(has_int64);
        let mut device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_extension_names(&ext_ptrs)
            .enabled_features(&core_features)
            .push_next(&mut shader_f16_ci)
            .push_next(&mut storage16_ci)
            .push_next(&mut storage8_ci)
            .push_next(&mut memmodel_ci)
            .push_next(&mut sgsize_ci)
            .push_next(&mut bda_ci);
        if has_i8_dot {
            device_ci = device_ci.push_next(&mut intdot_ci);
        }
        if has_coop_matrix {
            device_ci = device_ci.push_next(&mut coopmat_ci);
        }

        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }
            .map_err(|e| be(format!("create_device: {e}")))?;

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // ── command pool ───────────────────────────────────────────────────────
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| be(format!("create_command_pool: {e}")))?;

        // ── capabilities ───────────────────────────────────────────────────────
        // Query base limits + the subgroup-size range together via properties2 (the coopmat GEMM
        // pins requiredSubgroupSize=32, so the fallback ladder needs to know whether 32 is in range).
        let mut sgsize_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        // Maintenance3 (core in Vulkan 1.1) carries `maxMemoryAllocationSize` — the weight arena
        // splits its up-front reservation into blocks no larger than this.
        let mut maint3_props = vk::PhysicalDeviceMaintenance3Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut sgsize_props)
            .push_next(&mut maint3_props);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
        let props = props2.properties;
        // 0 = not reported → fall back to the Vulkan-guaranteed floor (2^30 = 1 GiB).
        let max_mem_alloc_size = if maint3_props.max_memory_allocation_size == 0 {
            1 << 30
        } else {
            maint3_props.max_memory_allocation_size
        };
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        // (0,0) when subgroup-size-control is unsupported: can't pin any size — the adapter treats
        // that as "no 32-pin available" and uses the driver's default subgroup for the fallback.
        let (subgroup_min, subgroup_max) = if has_sgsize {
            (
                sgsize_props.min_subgroup_size,
                sgsize_props.max_subgroup_size,
            )
        } else {
            (0, 0)
        };

        // infr's Vulkan compute kernels are written for a PINNED subgroup size of 32 (RDNA3 wave32):
        // rmsnorm / softmax / quant_q8 / the coopmat GEMM / attention QK+PV / flash / DeltaNet all
        // dispatch via `kernel_sg(..., 32)`, which sets `requiredSubgroupSize=32` and FAILS pipeline
        // creation on any device that can't provide a size-32 subgroup (no `subgroup_size_control`,
        // or 32 outside `[minSubgroupSize, maxSubgroupSize]`). rmsnorm/softmax run on EVERY forward
        // and are NOT coopmat-gated, so gating only the coopmat caps wouldn't prevent the crash —
        // the whole backend needs 32. Refuse the Vulkan backend here (a clean Err, not a mid-forward
        // panic) so `gpu_available()`/the seam falls back to CPU. Every real target — RADV (32-64),
        // NVIDIA (32), Intel Arc (…-32) — provides 32; this only rejects exotic no-32 /
        // no-size-control devices (older/mobile/llvmpipe), which can't run these wave32 kernels
        // correctly anyway.
        if !(has_sgsize && subgroup_min <= 32 && 32 <= subgroup_max) {
            return Err(be(format!(
                "infr's Vulkan backend requires a pinnable subgroup size of 32 (wave32); this \
                 device's subgroup range is [{subgroup_min}, {subgroup_max}] and \
                 subgroup_size_control={has_sgsize} — falling back to another backend"
            )));
        }

        // ── sg_pref: pinned subgroup size for the decode GEMV/reduction family ────────────────
        // Vendor by vendorID, NOT by subgroup range (Xe2 SKUs report minSubgroupSize 8 or 16
        // depending on the part — size-sniffing would misclassify Battlemage).
        let vendor_intel = props.vendor_id == 0x8086;
        // Intel EUs are SIMD8/SIMD16: pinning the decode GEMV family at 32 makes ANV compile
        // SIMD32 shaders whose per-lane register budget starves those kernels (llama.cpp pins 16
        // for mul_mat_vec on Intel for exactly this). `max(16, subgroup_min)` keeps this
        // Battlemage-proof (min=8 SKUs still get 16, never 8 — the kernels' lane math is only
        // built for 16/32). Everything else (RADV 32-64, NVIDIA 32) keeps 32, so the default
        // kernel/pipeline set there is byte-identical to before this field existed.
        let sg_default = if vendor_intel && subgroup_min <= 16 {
            16u32.max(subgroup_min)
        } else {
            32
        };
        // INFR_SG=16|32: A/B override (Intel testers; inert on devices that can't pin the value).
        let sg_pref = match std::env::var("INFR_SG").ok().as_deref() {
            Some("16") => 16,
            Some("32") => 32,
            Some(other) => {
                return Err(be(format!(
                    "INFR_SG must be 16 or 32 (got {other:?}) — the decode GEMV family only has \
                     subgroup-16 and subgroup-32 builds"
                )))
            }
            None => sg_default,
        };
        // A 16 request/default is only usable where 16 is pinnable; otherwise CLEANLY fall back
        // to 32 (e.g. INFR_SG=16 on RADV wave32: subgroup_min == 32 → stays 32, path set
        // unchanged). 32 is always pinnable here (hard-required above).
        let sg_pref = if sg_pref == 16 && !(subgroup_min <= 16 && 16 <= subgroup_max) {
            eprintln!(
                "[infr] INFR_SG=16 requested but this device's subgroup range \
                 [{subgroup_min}, {subgroup_max}] cannot pin 16 — keeping 32"
            );
            32
        } else {
            sg_pref
        };

        // ── integrated GPU + compute-unit count ───────────────────────────────────────────────
        // An iGPU/APU is NOT just "a slow discrete card": it is forced onto the non-coopmat kernel
        // tier (RDNA2/Raphael enumerates no cooperative matrix at all) AND carries ~1/50th the
        // compute, so a prefill chunk sized for a 96-CU card becomes a single multi-SECOND command
        // buffer — past the ~10 s `gfx`-ring watchdog it is a GPU reset, not merely slow. Detect the
        // device class here and let the seam bound its per-submit work (`Capabilities::integrated`).
        let integrated = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;
        // Best-effort CU count (AMD only; 0 = unknown). `VK_AMD_shader_core_properties` is a
        // properties2 pNext, so it needs no device-extension ENABLE — only that the driver
        // advertises it, hence the `has_ext` guard (chaining an unsupported struct is UB).
        let compute_units = if has_ext(c"VK_AMD_shader_core_properties") {
            let mut sc = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sc);
            unsafe { instance.get_physical_device_properties2(physical_device, &mut p2) };
            sc.shader_engine_count
                * sc.shader_arrays_per_engine_count
                * sc.compute_units_per_shader_array
        } else {
            0
        };

        let caps = Capabilities {
            name: device_name,
            f16: has_f16,
            coopmat_f16,
            f8: has_f8_ext,
            coopmat_f8,
            i8: has_int8,
            i8_dot: has_i8_dot,
            coopmat_i8,
            bf16: has_bf16_ext,
            coopmat_bf16,
            subgroup_min,
            subgroup_max,
            sg_pref,
            vendor_intel,
            integrated,
            compute_units,
            buffer_device_address: has_bda,
            max_shared_memory_bytes: props.limits.max_compute_shared_memory_size,
            // An INTEGRATED_GPU has no VRAM to be separate FROM: its "device-local" heap is system
            // DDR reached through the GART (proven on RADV RAPHAEL_MENDOCINO — see `vram_info`,
            // which is the only consumer that matters today). A DISCRETE_GPU is never UMA, and
            // that is the class this must not perturb, so key off the device type exactly like
            // `integrated` above (llama.cpp's Vulkan backend sets its `uma` flag the same way).
            // Note this is a strictly WEAKER claim than `integrated`, which additionally means
            // "submits must stay under a TDR watchdog" — the two happen to coincide on Vulkan.
            unified_memory: integrated,
            // The seam adapter records the decode graph once and replays it (params-driven `_dyn`
            // kernels); the runner compiles the eligible qwen3 decode graph once.
            decode_replay: true,
            combined_gu: true,
            embed_gather: true,
            gpu_sample: true,
            argmax_rows: true,
            argmax_prob: true,
            // Fused per-head RMSNorm + SiLU gate multiply (qwen35 DeltaNet z-gate) — the
            // `rmsnorm_gate` kernel (rmsnorm.comp's -DGATE build). Collapses QkNorm→GatedAct's
            // read-after-write barrier into one dispatch. INFR_NO_GATED_RMSNORM forces the split
            // form for A/B.
            gated_rmsnorm: true,
            // Every KV write/read kernel maps position -> row modulo the cache's row capacity
            // (identity on full-context caches), so SWA layers may get window-sized ring caches.
            kv_swa_ring: true,
        };

        // Publish the device class BEFORE any caller can size a prefill chunk against it (the seam
        // reads this in `ubatch_rows`, which runs on the first session/KV allocation — strictly
        // after `VulkanBackend::new` returns).
        let _ = DEVICE_CLASS.set(DeviceClass {
            integrated: caps.integrated,
            compute_units: caps.compute_units,
        });

        // One-line device banner (stderr) — the first thing to check on a portability bug report:
        // which GPU was picked and which kernel tiers are live. `y`/`n` per capability + the
        // subgroup range + shared-mem budget. Printed on every `VulkanBackend::new()` (no
        // process-wide dedup): a single run constructing several backends on the same device (an
        // `infr bench` MTP rep loop; a CPU/Vulkan parity check) now genuinely means one construction
        // per printed line, not a duplicate — `DenseSeamChat`'s MTP chat path shares ONE backend
        // across `warmup()` + every turn (see `chat/vulkan.rs`'s `mtp_vk`), so an ordinary
        // `INFR_MTP=1` run prints exactly one banner again without needing this dedup.
        let yn = |b: bool| if b { "y" } else { "n" };
        eprintln!(
            "[infr] GPU: {} | f16:{} f16cm:{} bf16:{} bf16cm:{} f8:{} f8cm:{} i8:{} i8dot:{} i8cm:{} \
             subgroup:{}-{} sgp:{} shared:{}KB",
            caps.name,
            yn(caps.f16),
            yn(caps.f16_coopmat()),
            yn(caps.bf16),
            yn(caps.bf16_coopmat()),
            yn(caps.f8),
            yn(caps.f8_coopmat()),
            yn(caps.i8),
            yn(caps.i8_dot),
            yn(caps.i8_coopmat()),
            caps.subgroup_min,
            caps.subgroup_max,
            caps.sg_pref,
            caps.max_shared_memory_bytes / 1024,
        );
        // Submit splitter (see `VulkanShared::submit_dispatch_cap`): the initial, pre-measurement
        // cap. `INFR_SUBMIT_DISPATCHES` overrides it (`0` = never split) — the kill switch if this
        // ever misjudges a device.
        let submit_dispatch_cap = match std::env::var("INFR_SUBMIT_DISPATCHES") {
            Ok(v) => v.parse::<usize>().map_err(|e| {
                be(format!(
                    "INFR_SUBMIT_DISPATCHES: expected a dispatch count (0 = never split): {e}"
                ))
            })?,
            Err(_) => infr_core::initial_submit_dispatch_cap(caps.integrated),
        };
        // Integrated GPUs run a DIFFERENT shape of forward (smaller prefill chunk, and the whole
        // pass split across several submits so no single command buffer can trip the GPU's hang
        // watchdog), so say so out loud: it is the first thing to check when an iGPU run hangs or
        // prefills slowly. Silent on every discrete device (nothing changed there).
        if caps.integrated {
            eprintln!(
                "[infr] GPU: INTEGRATED (cu:{}) — prefill chunk {} rows, forward split every {} \
                 dispatches to stay under the GPU hang watchdog; INFR_UBATCH / \
                 INFR_SUBMIT_DISPATCHES override",
                if caps.compute_units > 0 {
                    caps.compute_units.to_string()
                } else {
                    "?".to_string()
                },
                infr_core::integrated_ubatch_rows(caps.compute_units),
                submit_dispatch_cap,
            );
        }
        // On a unified-memory part the budget guard counts EVERY heap, not just the device-local
        // one (see `vram_info`) — a materially different capacity, and the second thing to check
        // when an iGPU either loads a model you didn't expect to fit or starts swapping. Print the
        // number it will actually budget against. Silent on every discrete device.
        if caps.unified_memory {
            let mp = unsafe { instance.get_physical_device_memory_properties(physical_device) };
            let (mut all, mut dev_local) = (0u64, 0u64);
            for i in 0..mp.memory_heap_count as usize {
                all += mp.memory_heaps[i].size;
                if mp.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                {
                    dev_local += mp.memory_heaps[i].size;
                }
            }
            eprintln!(
                "[infr] GPU: UNIFIED MEMORY — budgeting against all {} heaps ({}), not the \
                 device-local slice alone ({}); this GPU's memory IS system RAM",
                mp.memory_heap_count,
                fmt_bytes(all),
                fmt_bytes(dev_local),
            );
        }

        // ── gpu-allocator ──────────────────────────────────────────────────────
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            // Every gpu-allocator allocation gets VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, which the
            // paged-MoE arena buffer (created with SHADER_DEVICE_ADDRESS usage) requires before it
            // can be bound. Harmless for all other buffers. `has_bda` is hard-required above.
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| be(format!("gpu_allocator::Allocator::new: {e}")))?;

        // ── on-disk pipeline cache (see `pcache.rs`) ───────────────────────────
        let pcache = crate::pcache::PcachePersist::new(&props);
        let initial = pcache.as_ref().and_then(|p| p.load()).unwrap_or_default();
        let mut pc_info = vk::PipelineCacheCreateInfo::default();
        if !initial.is_empty() {
            pc_info = pc_info.initial_data(&initial);
        }
        // A corrupt-but-well-enveloped blob can still fail creation: retry empty, never fatal.
        let pipeline_cache = unsafe { device.create_pipeline_cache(&pc_info, None) }
            .or_else(|_| unsafe {
                device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
            })
            .unwrap_or(vk::PipelineCache::null());

        // Built before `instance`/`device` move into `VulkanShared` below.
        let push_descriptor =
            has_push_descriptor.then(|| ash::khr::push_descriptor::Device::new(&instance, &device));

        // Direct-to-VRAM weight writes (Resizable BAR), if this device exposes the memory type.
        // Probed once here; whether a given LOAD actually uses it is decided in
        // `weight_progress_scope` (it also has to fit).
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let rebar_type = probe_rebar_type(&mem_props);
        // Probed for UMA parts ONLY — on a discrete card the non-device-local heap is host RAM
        // across PCIe and must never receive a GpuOnly buffer (see `probe_uma_overflow_type`).
        let uma_overflow_type = caps
            .unified_memory
            .then(|| probe_uma_overflow_type(&mem_props))
            .flatten();

        Ok(Self {
            moe_pager: Mutex::new(None),
            dense_pager: Mutex::new(None),
            bda_weight_arena: Mutex::new(None),
            shared: Arc::new(VulkanShared {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue,
                queue_family_index,
                cmd_pool: Mutex::new(cmd_pool),
                allocator: ManuallyDrop::new(Mutex::new(allocator)),
                caps,
                has_mem_budget,
                max_mem_alloc_size,
                push_descriptor,
                weight_arena: Mutex::new(None),
                linear_kernel: std::sync::OnceLock::new(),
                kernels: Mutex::new(HashMap::new()),
                pipeline_cache,
                pcache,
                weight_pb: Mutex::new(None),
                device_used: AtomicU64::new(0),
                submit_dispatch_cap: AtomicUsize::new(submit_dispatch_cap),
                rebar_type,
                uma_overflow_type,
                uma_spilled: AtomicU64::new(0),
                weights_direct: AtomicBool::new(false),
                staging_ring: Mutex::new(None),
            }),
        })
    }

    /// The submit splitter's current cap — see `VulkanShared::submit_dispatch_cap`. `0` =
    /// unlimited (record the whole forward into one command buffer, the discrete default).
    pub(crate) fn submit_dispatch_cap(&self) -> usize {
        self.shared.submit_dispatch_cap.load(Ordering::Relaxed)
    }

    /// Feed a completed forward's measurement back into the submit splitter: `elapsed` of wall
    /// across `dispatches` dispatches, measured with the queue drained at both ends. Re-tunes the
    /// cap so the NEXT forward's segments land inside `infr_core::SUBMIT_BUDGET_NS` on whatever
    /// device this actually is — the loop that makes the splitter hardware-agnostic instead of a
    /// table of per-GPU constants.
    ///
    /// The cap only ever RATCHETS DOWN within a process. A forward's wall time is a noisy sample
    /// (a cold pipeline compile, a busy host, a first-touch page fault all inflate it), and the
    /// asymmetry of the two mistakes is total: too small a cap costs a few extra submits, too
    /// large a cap costs a device-lost. So a slow sample tightens the bound and a fast one is
    /// simply ignored.
    pub(crate) fn observe_forward(&self, elapsed: std::time::Duration, dispatches: usize) {
        let ns = elapsed.as_nanos() as u64;
        let cur = self.submit_dispatch_cap();
        // A device that has never split (every discrete GPU) only starts splitting if a forward
        // actually lands in watchdog territory — see `infr_core::SUBMIT_DANGER_NS`. Without this,
        // the measured cap (finite, just very large) would eventually split a big enough graph on
        // a perfectly healthy dGPU, which is a tuned path this has no business touching.
        if cur == 0 && ns < infr_core::SUBMIT_DANGER_NS {
            return;
        }
        let want = infr_core::submit_cap_from_measurement(ns, dispatches);
        if want == 0 {
            return; // measurement says "no split needed"; never loosen an existing cap on it
        }
        let next = if cur == 0 { want } else { cur.min(want) };
        if next != cur {
            self.shared
                .submit_dispatch_cap
                .store(next, Ordering::Relaxed);
        }
    }

    /// Begin a "loading weights" progress bar covering `total_bytes` (pass `None` for an
    /// indeterminate byte spinner when the total isn't known up front). Every subsequent
    /// `BufferUsage::Weights` allocation advances it automatically — the ticking lives in `alloc`,
    /// so a model loader cannot forget it; it only has to open the scope once. The returned guard
    /// finishes and clears the bar on drop, so the bar's lifetime is the loader's scope.
    fn weight_progress_scope(&self, total_bytes: Option<u64>) -> WeightProgress {
        // Pick the weight-upload path for THIS load, from the device's actual memory properties.
        //
        // The DEFAULT is the reused, pipelined staging ring — it is both faster than direct-to-VRAM
        // writes and available on every device (see `probe_rebar_type` for the measurements).
        //
        // Direct-to-VRAM (ReBAR, opt-in via INFR_REBAR=1) additionally requires:
        //   * a DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT type on the main VRAM heap
        //     (`probe_rebar_type` — absent when ReBAR is off in the BIOS), and
        //   * enough room on that heap for the model. `total_bytes` is the loader's weight
        //     footprint; if it doesn't fit we take the ring rather than allocate until the heap
        //     gives out. Unknown total (`None`) → assume it fits; a mid-load failure still falls
        //     back gracefully (see `make_buf_ex`).
        let direct = match self.shared.rebar_type {
            Some(ty) => {
                let mp = unsafe {
                    self.shared
                        .instance
                        .get_physical_device_memory_properties(self.shared.physical_device)
                };
                total_bytes.is_none_or(|t| t <= heap_size_of(&mp, ty))
            }
            None => false,
        };
        self.shared.weights_direct.store(direct, Ordering::Relaxed);

        // ── VRAM UP FRONT: pre-reserve the whole resident weight set as a bump arena ────────────
        // Committing the footprint in a few big blocks (instead of dribbling one dedicated
        // VkDeviceMemory per tensor) makes the model OWN its VRAM the instant the load starts — no
        // window where another process can grab VRAM mid-load and no slow per-tensor climb in
        // `mem_info_vram_used`. Every subsequent `Weights` alloc sub-allocates from the arena
        // (`make_buf_ex`). Gated hard — reserve ONLY when:
        //   * total is known (an indeterminate load can't be sized), and
        //   * the model is FULLY RESIDENT: a paged-MoE (`moe_paged`) or dense-STREAMED
        //     (`dense_paged`) model DELIBERATELY keeps most weights OUT of VRAM, and `total` here
        //     counts the whole footprint — arena-ing it would reserve the entire model and OOM, or
        //     make streamed placeholders land in the arena and corrupt. The pager was already
        //     installed by `vulkan_moe_binder` before this scope opens, so these read true now, and
        //   * NOT the direct-to-VRAM (ReBAR) path: those weights get their own mapped dedicated
        //     allocations in `make_buf_ex` and never touch the arena — a reservation would just
        //     double-commit the VRAM, and
        //   * DISCRETE only: on a unified-memory part the device-local heap is a synthetic carveout
        //     smaller than total DDR, and the per-tensor path's `uma_overflow_type` spill is what
        //     places bytes on the other heap once it fills. A single big arena block can't spill
        //     mid-block, so a resident UMA model over the carveout would just fail the reserve and
        //     fall back anyway; reserving up front on shared DDR buys nothing (same memory, and the
        //     "another process grabs it" race doesn't bite a shared pool the same way). Leave UMA
        //     on the per-tensor spill path, which is already correct.
        // On ANY failure `reserve_weights` rolls back to no arena and the per-tensor path takes over
        // — the reservation is a pure optimization, never a load-blocker. `INFR_NO_WEIGHT_ARENA=1`
        // forces that per-tensor fallback (an escape hatch for a fragmented heap where one big
        // up-front block fails but many small allocs still succeed; also the A/B knob for the
        // up-front-commit measurement).
        let arena_off =
            std::env::var("INFR_NO_WEIGHT_ARENA").is_ok_and(|v| !v.is_empty() && v != "0");
        let arena_free = self.shared.weight_arena.lock().unwrap().is_none();
        if let Some(total) = total_bytes {
            if !arena_off
                && arena_free
                && !direct
                && !self.shared.caps.unified_memory
                && !self.moe_paged()
                && !self.dense_paged()
            {
                if let Err(e) = self.reserve_weights(total) {
                    eprintln!(
                        "[infr] weight arena reservation failed ({e}); \
                         falling back to per-tensor weight allocation"
                    );
                }
            }
        }

        let pb = infr_core::progress::bar(
            total_bytes,
            "loading weights",
            infr_core::progress::Unit::Bytes,
        );
        *self.shared.weight_pb.lock().unwrap() = Some(pb);
        WeightProgress {
            shared: self.shared.clone(),
        }
    }

    /// Install this model's paged-MoE session (see `pager::MoePagerSession`), sized but with no
    /// tensors registered yet — called BEFORE the seam's weight-load closure runs (see
    /// `pager::MoePagerLayout`'s doc for why the ordering matters: `Backend::moe_paged` must
    /// already read true by the time that closure's placeholder buffers are bound). Replaces any
    /// previous session (there is only ever one loaded model per process today).
    pub fn init_moe_pager(&self, layout: crate::pager::MoePagerLayout) -> Result<()> {
        let session = crate::pager::MoePagerSession::new(self, layout)?;
        *self.moe_pager.lock().unwrap() = Some(session);
        Ok(())
    }

    /// Register one paged layer's role tensor with the session `init_moe_pager` already installed
    /// — called from the seam's weight-load closure instead of uploading the tensor's full bytes.
    /// Panics if no session is installed (a caller bug: `init_moe_pager` must run first); errors
    /// if the layout has no pool matching the tensor's (role, per-expert bytes).
    pub fn register_paged_expert(
        &self,
        role: crate::pager::Role,
        buf_id: usize,
        source: crate::pager::ExpertSource,
    ) -> Result<()> {
        self.moe_pager
            .lock()
            .unwrap()
            .as_mut()
            .expect("register_paged_expert called before init_moe_pager")
            .register(role, buf_id, source)
    }

    /// `INFR_PAGER_STATS=1` reporting hook — a no-op when no paged model is loaded.
    pub fn print_moe_pager_stats(&self) {
        if let Some(s) = self.moe_pager.lock().unwrap().as_ref() {
            s.print_stats_if_enabled();
        }
    }

    /// Locked access to the paged-MoE session for the adapter's `execute_static` — `pub(crate)`
    /// (only `adapter.rs` reaches into this); see `pager.rs`'s module doc for why this lives
    /// outside the `Graph`/`Bindings` seam instead of a per-op flag.
    pub(crate) fn moe_pager(&self) -> &crate::pager::MoePagerCell {
        &self.moe_pager
    }

    /// Install this model's dense layer-streaming session (see `pager::DensePagerSession`) —
    /// `init_moe_pager`'s dense twin, same call-order contract (BEFORE the seam's weight-load
    /// closure binds the first placeholder, so `Backend::dense_paged` already reads true).
    pub fn init_dense_pager(&self, layout: crate::pager::DensePagerLayout) -> Result<()> {
        let session = crate::pager::DensePagerSession::new(self, layout)?;
        *self.dense_pager.lock().unwrap() = Some(session);
        Ok(())
    }

    /// Register one streamed dense block with the session `init_dense_pager` installed — called
    /// from the seam's weight-load closure instead of uploading the block's full bytes. Panics if
    /// no session is installed (a caller bug: `init_dense_pager` must run first).
    pub fn register_dense_stream(
        &self,
        pool: usize,
        buf_id: usize,
        source: crate::pager::DenseSource,
    ) -> Result<()> {
        self.dense_pager
            .lock()
            .unwrap()
            .as_mut()
            .expect("register_dense_stream called before init_dense_pager")
            .register(pool, buf_id, source)
    }

    /// `INFR_PAGER_STATS=1` reporting hook — a no-op when no dense-streamed model is loaded.
    pub fn print_dense_pager_stats(&self) {
        if let Some(s) = self.dense_pager.lock().unwrap().as_ref() {
            s.print_stats_if_enabled();
        }
    }

    /// [`Self::moe_pager`]'s dense twin — locked access for the adapter's `execute_static`.
    pub(crate) fn dense_pager(&self) -> &crate::pager::DensePagerCell {
        &self.dense_pager
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    /// Create a `vk::Buffer` + gpu-allocator sub-allocation of the requested size/location.
    /// Device-local VRAM: total heap size and currently-available bytes. `available` comes from
    /// VK_EXT_memory_budget (live, accounts for other processes + our own allocations) when the
    /// extension is present; otherwise it falls back to the total heap size (best effort).
    /// NOTE: the extension's `heapBudget` is a CEILING (how much this process may use in total),
    /// not free bytes — live free = `heapBudget - heapUsage`. Reporting the raw budget here once
    /// made `available` sit ~constant while we allocated GBs, which let the VRAM guard sail past
    /// a 53 GiB KV cache into VK_ERROR_DEVICE_LOST.
    pub fn vram(&self) -> VramInfo {
        vram_info(&self.shared)
    }

    /// Device-memory budget guard: hard-error BEFORE a device-local allocation of `want` bytes
    /// that would exceed the budget. Over-committing does not fail cleanly on GPUs — the driver
    /// accepts the allocation and then evicts, which on a discrete card means reading weights back
    /// across PCIe (measured: a 41 GiB device-local run on a 24 GiB 7900 XTX quietly placed 18 GiB
    /// in GTT) and can end in a device-lost (TDR) mid-inference. The only safe failure point is
    /// here, at allocation time (mirrors the Metal backend's working-set guard).
    ///
    /// The budget comes from [`vram_info`], which counts device-local heaps on a discrete card and
    /// EVERY heap on a unified-memory part (where they are one pool of DDR — see its doc). Uses the
    /// LIVE per-heap budget when VK_EXT_memory_budget is present (it accounts for other processes
    /// and everything we already hold); otherwise falls back to this backend's tracked bytes
    /// against the total heap. `GUARD_HEADROOM` absorbs allocation slop (alignment, gpu-allocator
    /// block rounding) and driver-internal allocations. `INFR_NO_VRAM_GUARD=1` disables the check
    /// (restoring the old fail-late behavior).
    ///
    /// Sub-MiB allocations skip the check (no per-tiny-alloc driver query; they cannot
    /// individually blow the budget and stay covered by the next large allocation's check).
    fn check_vram_budget(&self, want: u64) -> Result<()> {
        const CHECK_MIN: u64 = 1 << 20; // 1 MiB
        const GUARD_HEADROOM: u64 = 256 * 1024 * 1024;
        if want < CHECK_MIN || std::env::var("INFR_NO_VRAM_GUARD").is_ok() {
            return Ok(());
        }
        let v = self.vram();
        let used = if v.live {
            v.total.saturating_sub(v.available)
        } else {
            self.shared.device_used.load(Ordering::Relaxed)
        };
        let budget = v.total.saturating_sub(GUARD_HEADROOM);
        if used + want > budget {
            let pool = if v.uma {
                "unified memory (all heaps — this GPU shares system RAM)"
            } else {
                "device-local"
            };
            return Err(be(format!(
                "{} budget exceeded: {} requested + {} already in use ({}) > {} budget ({} {pool} \
                 minus 256 MiB headroom). Refusing to over-commit: exceeding it doesn't fail \
                 cleanly — the driver evicts (weights get read back over the bus) or the device is \
                 lost (TDR) mid-inference. Use a smaller context (INFR_CTX), a smaller/more- \
                 quantized model, close other GPU processes, or run on the CPU backend \
                 (INFR_CPU=1). INFR_NO_VRAM_GUARD=1 overrides at your own risk.",
                if v.uma { "Unified-memory" } else { "VRAM" },
                fmt_bytes(want),
                fmt_bytes(used),
                if v.live {
                    "live driver budget"
                } else {
                    "tracked by this process; no VK_EXT_memory_budget"
                },
                fmt_bytes(budget),
                fmt_bytes(v.total),
            )));
        }
        Ok(())
    }

    /// First device-local memory type compatible with `type_bits` (from a buffer's requirements).
    fn find_memory_type(&self, type_bits: u32, props: vk::MemoryPropertyFlags) -> Option<u32> {
        let mp = unsafe {
            self.shared
                .instance
                .get_physical_device_memory_properties(self.shared.physical_device)
        };
        (0..mp.memory_type_count).find(|&i| {
            (type_bits & (1 << i)) != 0
                && mp.memory_types[i as usize].property_flags.contains(props)
        })
    }

    /// Pre-reserve `total` bytes of device-local VRAM as a bump arena for load-once weights, so the
    /// whole model's weight memory is committed up front (a few big blocks, freed in one shot)
    /// instead of dribbled out per-tensor. Subsequent `BufferUsage::Weights` allocs sub-allocate
    /// from it. Call once after the footprint check, before uploading weights. On failure (e.g. no
    /// contiguous block available) rolls back every block it took and leaves NO arena → callers
    /// fall back to the per-tensor path (which no-ReBAR / tight-VRAM / UMA machines rely on).
    ///
    /// `total` cannot be one `vkAllocateMemory` for any real model — `maxMemoryAllocationSize` is
    /// ~4 GiB on RADV, and a 9 GiB (let alone 21.9 GiB) model exceeds it. So the reservation is
    /// split into N blocks each ≤ `max_mem_alloc_size`; the bump allocator sub-allocates a weight
    /// across the block boundary transparently (`WeightArena::bump` never straddles a block — it
    /// starts a fresh block when a tensor won't fit the current one, and grows overflow blocks if
    /// the estimate underflows).
    pub fn reserve_weights(&self, total: u64) -> Result<()> {
        self.reserve_weights_capped(total, self.shared.max_mem_alloc_size)
    }

    /// [`reserve_weights`](Self::reserve_weights) with an explicit per-block cap — the production
    /// path passes `max_mem_alloc_size`; tests pass a tiny cap to exercise the multi-block split
    /// without committing multiple GiB of real VRAM.
    fn reserve_weights_capped(&self, total: u64, block_cap: u64) -> Result<()> {
        // Probe a weight-shaped buffer for its memory-type bits + alignment (identical for every
        // weight buffer, since they all share BUFFER_USAGE).
        let probe = unsafe {
            self.shared.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(4096)
                    .usage(BUFFER_USAGE)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|e| be(format!("arena probe buffer: {e}")))?;
        let req = unsafe { self.shared.device.get_buffer_memory_requirements(probe) };
        unsafe { self.shared.device.destroy_buffer(probe, None) };

        let mem_type = self
            .find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .ok_or_else(|| be("no DEVICE_LOCAL memory type for weights"))?;
        let align = req.alignment.max(1);
        let total = total.next_multiple_of(align).max(align);
        // Budget-check the WHOLE reservation up front — the point of reserving is to own the entire
        // footprint, so refuse loudly here if it doesn't fit rather than after committing some
        // blocks (the caller then falls back to per-tensor, which fails the same way tensor by
        // tensor, but this keeps the up-front commitment honest).
        self.check_vram_budget(total)?;

        // Largest single allocation the device accepts, aligned down to the buffer alignment (an
        // alloc slightly under the cap is safest — the exact limit can fail on a fragmented heap).
        let block_cap = (block_cap / align * align).max(align);

        let mut blocks: Vec<ArenaBlock> = Vec::new();
        let mut remaining = total;
        while remaining > 0 {
            // Each block ≤ block_cap; the tail block is exactly the remainder.
            let bs = remaining.min(block_cap);
            let memory = match unsafe {
                self.shared.device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(bs)
                        .memory_type_index(mem_type),
                    None,
                )
            } {
                Ok(m) => m,
                Err(e) => {
                    // Roll back everything already committed and leave NO arena so the caller falls
                    // back to the per-tensor path (never a partial arena — that would silently cap
                    // the resident set at what happened to allocate).
                    for b in &blocks {
                        unsafe { self.shared.device.free_memory(b.memory, None) };
                    }
                    let taken: u64 = blocks.iter().map(|b| b.size).sum();
                    if taken > 0 {
                        self.shared.device_used.fetch_sub(taken, Ordering::Relaxed);
                    }
                    return Err(be(format!(
                        "reserve_weights block {bs} bytes (of {total} total, {} blocks placed): {e}",
                        blocks.len()
                    )));
                }
            };
            self.shared.device_used.fetch_add(bs, Ordering::Relaxed);
            blocks.push(ArenaBlock {
                memory,
                size: bs,
                cursor: 0,
            });
            remaining -= bs;
        }

        *self.shared.weight_arena.lock().unwrap() = Some(WeightArena {
            mem_type,
            blocks,
            cur: 0,
        });
        Ok(())
    }

    /// Bind `buffer` to a fresh, PERSISTENTLY MAPPED dedicated allocation of memory type `ty`. Two
    /// callers, distinguished by `spilled`: the ReBAR weight path (`false` — device-local VRAM the
    /// host can write through) and the UMA overflow spill (`true` — the non-device-local heap of a
    /// unified-memory part). See [`Backing::Vram`].
    ///
    /// The caller owns `buffer` and must destroy it if this returns `Err`. Budget-guarded and
    /// charged to `device_used` (or `uma_spilled` when `spilled`) like any other allocation.
    fn alloc_vram_mapped(
        &self,
        buffer: vk::Buffer,
        size: usize,
        requirements: &vk::MemoryRequirements,
        ty: u32,
        spilled: bool,
    ) -> Result<VkBuffer> {
        self.check_vram_budget(requirements.size)?;

        let memory = unsafe {
            self.shared.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(ty),
                None,
            )
        }
        .map_err(|e| be(format!("rebar allocate_memory({}): {e}", requirements.size)))?;

        let ptr = match unsafe {
            self.shared
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        } {
            Ok(p) => p as *mut u8,
            Err(e) => {
                unsafe { self.shared.device.free_memory(memory, None) };
                return Err(be(format!("rebar map_memory: {e}")));
            }
        };

        if let Err(e) = unsafe { self.shared.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.shared.device.unmap_memory(memory);
                self.shared.device.free_memory(memory, None);
            }
            return Err(be(format!("rebar bind_buffer_memory: {e}")));
        }

        // Charge the counter this allocation's heap belongs to (see `VulkanShared::uma_spilled`).
        if spilled {
            &self.shared.uma_spilled
        } else {
            &self.shared.device_used
        }
        .fetch_add(requirements.size, Ordering::Relaxed);

        Ok(VkBuffer {
            shared: Arc::clone(&self.shared),
            buffer,
            backing: Backing::Vram {
                memory,
                ptr,
                spilled,
            },
            // Logical size = what the caller asked for; `requirements.size` only rounds it up for
            // alignment, and `fill_buf`/`upload` must not touch past the logical extent.
            size,
            mem_size: requirements.size,
            location: MemoryLocation::GpuOnly,
            sub_offset: 0,
        })
    }

    fn make_buf(&self, size: usize, location: MemoryLocation, label: &str) -> Result<VkBuffer> {
        self.make_buf_ex(size, location, label, false, false)
    }

    /// Allocate the paged-MoE expert arena as a `bufferDeviceAddress` buffer and return it with its
    /// 64-bit `VkDeviceAddress`. Unlike a plain SSBO arena (capped at `maxStorageBufferRange`), the
    /// paged expert kernels read this through a `GL_EXT_buffer_reference` pointer, so it may be as
    /// large as VRAM allows. Always a dedicated GpuOnly allocation, budget-guarded like any weight;
    /// the `SHADER_DEVICE_ADDRESS` usage + the allocator's DEVICE_ADDRESS memory flag are what let
    /// `get_buffer_device_address` succeed.
    pub fn alloc_arena_bda(&self, bytes: usize) -> Result<(Box<dyn Buffer>, u64)> {
        let buf = self.make_buf_ex(bytes, MemoryLocation::GpuOnly, "moe-arena", true, true)?;
        let handle = buf.buffer;
        let addr = unsafe {
            self.shared
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(handle))
        };
        Ok((Box::new(buf) as Box<dyn Buffer>, addr))
    }

    /// Sub-allocate `size` bytes for a resident weight tensor from the `INFR_RESIDENT_BDA` arena
    /// (see [`BdaWeightArena`], [`resident_bda_enabled`]). Bump-allocates within the current block
    /// at [`BDA_WEIGHT_ALIGN`]; when the request doesn't fit the current block's remainder, opens a
    /// fresh dedicated block (never splitting a tensor across two blocks — the same rule
    /// `WeightArena::bump` follows for the plain SSBO arena). Returns a `VkBuffer` that shares the
    /// block's single `vk::Buffer` handle (`Backing::BdaSub`) with `sub_offset` set to this tensor's
    /// offset within it — see that variant's doc for why the handle must never be bound as a
    /// descriptor at its full range.
    fn bda_weight_alloc(&self, size: usize) -> Result<VkBuffer> {
        let want = (size as u64).max(1).next_multiple_of(BDA_WEIGHT_ALIGN);
        let mut guard = self.bda_weight_arena.lock().unwrap();
        let arena = guard.get_or_insert_with(BdaWeightArena::default);

        if let Some(b) = arena.blocks.last_mut() {
            let off = b.cursor.div_ceil(BDA_WEIGHT_ALIGN) * BDA_WEIGHT_ALIGN;
            if off + want <= b.size {
                b.cursor = off + want;
                return Ok(VkBuffer {
                    shared: Arc::clone(&self.shared),
                    buffer: b.handle.buf.buffer,
                    backing: Backing::BdaSub(Arc::clone(&b.handle)),
                    size,
                    mem_size: 0,
                    location: MemoryLocation::GpuOnly,
                    sub_offset: off as usize,
                });
            }
        }

        // The current block (if any) has no room: open a fresh dedicated block sized to the
        // request, floored at `BDA_BLOCK_MIN` and capped at `max_mem_alloc_size` — a whole
        // multi-GiB model can't be one `vkAllocateMemory`, same reasoning as `WeightArena`'s block
        // cap. A single tensor bigger than the cap can't be placed at all; report it rather than
        // silently truncating.
        let block_bytes = want.max(BDA_BLOCK_MIN).min(self.shared.max_mem_alloc_size);
        if want > block_bytes {
            return Err(be(format!(
                "resident-BDA weight tensor ({want} bytes) exceeds max_mem_alloc_size ({} bytes) \
                 — cannot fit in a single arena block",
                self.shared.max_mem_alloc_size
            )));
        }
        let block_buf = self.make_buf_ex(
            block_bytes as usize,
            MemoryLocation::GpuOnly,
            "resident-bda",
            true,
            true,
        )?;
        let buffer = block_buf.buffer;
        let base_addr = unsafe {
            self.shared
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        let handle = Arc::new(BdaBlockHandle {
            buf: block_buf,
            base_addr,
        });
        arena.blocks.push(BdaArenaBlock {
            handle: Arc::clone(&handle),
            size: block_bytes,
            cursor: want,
        });
        Ok(VkBuffer {
            shared: Arc::clone(&self.shared),
            buffer,
            backing: Backing::BdaSub(handle),
            size,
            mem_size: 0,
            location: MemoryLocation::GpuOnly,
            sub_offset: 0,
        })
    }

    /// [`make_buf`](Self::make_buf) with an explicit dedicated-allocation override. Post-load
    /// memory hygiene: `force_dedicated` bypasses gpu-allocator's general (sub-allocating)
    /// memory blocks entirely, so a TRANSIENT buffer frees its `VkDeviceMemory` fully on drop.
    /// Without it, sub-block transients grow general blocks the allocator then RETAINS: the
    /// vendored gpu-allocator (0.27) frees an emptied general block only while another general
    /// block exists in the same memory type (`active_general_blocks > 1` in its `free()`), and
    /// exposes no purge/trim API — so the last 64 MiB host-visible block (and a 256 MiB
    /// device-local one) would sit empty in the ReBAR heap for the whole session. Used by the
    /// weight-upload staging path below; never on a per-token path (a dedicated allocation costs
    /// a `vkAllocateMemory`, fine once per tensor at load, wrong per token).
    fn make_buf_ex(
        &self,
        size: usize,
        location: MemoryLocation,
        label: &str,
        force_dedicated: bool,
        device_address: bool,
    ) -> Result<VkBuffer> {
        // `device_address` (the paged-MoE arena only): add SHADER_DEVICE_ADDRESS so the buffer can
        // be handed to a shader as a 64-bit pointer. Its backing memory gets the matching
        // DEVICE_ADDRESS alloc flag from gpu-allocator (built with `buffer_device_address: true`),
        // so this buffer must take the gpu-allocator path below — never the weight bump arena or
        // the ReBAR mapped path, whose manual `allocate_memory` does not set that flag. The
        // "moe-arena" label steers it clear of both (they key on label == "weights").
        let usage = if device_address {
            vk::BufferUsageFlags::from_raw(
                BUFFER_USAGE.as_raw() | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
            )
        } else {
            BUFFER_USAGE
        };
        let buf_ci = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.shared.device.create_buffer(&buf_ci, None) }
            .map_err(|e| be(format!("create_buffer: {e}")))?;

        let requirements = unsafe { self.shared.device.get_buffer_memory_requirements(buffer) };

        // ── ReBAR fast path: a weight, during a load that chose direct-to-VRAM writes ──────────
        // Allocate this tensor's VRAM from the host-visible device-local type and map it, so
        // `upload` can memcpy the GGUF bytes straight in (no staging, no copy cmd, no stall).
        // The memory is device-local, so it is budget-guarded and accounted exactly like a GpuOnly
        // allocation — this changes WHERE weights are allocated from, not how much VRAM they take.
        if label == "weights" && self.shared.weights_direct.load(Ordering::Relaxed) {
            if let Some(ty) = self.shared.rebar_type {
                // Only if the buffer's requirements actually permit that memory type.
                if requirements.memory_type_bits & (1 << ty) != 0 {
                    match self.alloc_vram_mapped(buffer, size, &requirements, ty, false) {
                        Ok(b) => return Ok(b),
                        Err(e) => {
                            // Out of host-visible VRAM (or map failed): fall through to the normal
                            // allocator rather than failing the load — the staging path still works.
                            unsafe { self.shared.device.destroy_buffer(buffer, None) };
                            self.shared.weights_direct.store(false, Ordering::Relaxed);
                            eprintln!(
                                "[infr] direct-to-VRAM weight alloc failed ({e}); \
                                 falling back to the staging ring for the rest of this load"
                            );
                            return self.make_buf_ex(
                                size,
                                location,
                                label,
                                force_dedicated,
                                device_address,
                            );
                        }
                    }
                }
            }
        }

        // Load-once weights (label "weights") bind into the pre-reserved bump arena when one exists
        // — the whole model's VRAM is reserved up front (see `reserve_weights`). Everything else
        // (transient activations, host-visible staging/readback, and weights with no arena) uses the
        // gpu-allocator below.
        if label == "weights" {
            let mut arena = self.shared.weight_arena.lock().unwrap();
            if let Some(a) = arena.as_mut() {
                // A bump that fits the reserved block commits no NEW device memory (the block was
                // budget-checked at reserve time); an overflow block does — guard it first.
                if !a.fits(requirements.size, requirements.alignment) {
                    if let Err(e) =
                        self.check_vram_budget(requirements.size.max(ARENA_OVERFLOW_BLOCK))
                    {
                        unsafe { self.shared.device.destroy_buffer(buffer, None) };
                        return Err(e);
                    }
                }
                match a.bump(
                    &self.shared.device,
                    requirements.size,
                    requirements.alignment,
                    &self.shared.device_used,
                ) {
                    Ok((memory, offset)) => {
                        unsafe {
                            self.shared
                                .device
                                .bind_buffer_memory(buffer, memory, offset)
                        }
                        .map_err(|e| {
                            unsafe { self.shared.device.destroy_buffer(buffer, None) };
                            be(format!("arena bind_buffer_memory: {e}"))
                        })?;
                        return Ok(VkBuffer {
                            shared: Arc::clone(&self.shared),
                            buffer,
                            backing: Backing::Arena,
                            size,
                            mem_size: requirements.size,
                            location,
                            sub_offset: 0,
                        });
                    }
                    Err(e) => {
                        unsafe { self.shared.device.destroy_buffer(buffer, None) };
                        return Err(e);
                    }
                }
            }
        }

        // Large buffers (KV cache, big weights) get a DEDICATED exact-size VkDeviceMemory; otherwise
        // they sub-allocate into gpu-allocator's 256MB blocks and waste the remainder (e.g. 3×67MB
        // KV buffers per block leave ~55MB unused — ~0.7GB across a long-context KV cache). Small/
        // transient buffers stay sub-allocated (cheap, pooled).
        const DEDICATED_MIN: u64 = 32 * 1024 * 1024;
        let scheme = if force_dedicated || requirements.size >= DEDICATED_MIN {
            AllocationScheme::DedicatedBuffer(buffer)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };
        // Budget guard: fail fast, with a clear error, BEFORE committing device-local memory the
        // budget can't cover (host-visible staging/readback/host-weights are exempt — the guard
        // protects VRAM only).
        if location == MemoryLocation::GpuOnly {
            if let Err(e) = self.check_vram_budget(requirements.size) {
                unsafe { self.shared.device.destroy_buffer(buffer, None) };
                return Err(e);
            }
        }

        // ── UMA spill: device-local heap full, put this on the other heap ─────────────────────
        // UNIFIED-MEMORY PARTS ONLY (`uma_overflow_type` is `None` on every discrete GPU, so a
        // dGPU never even evaluates this — it falls straight through to gpu-allocator exactly as
        // before). Once the synthetic device-local heap is full, gpu-allocator would keep resolving
        // GpuOnly to it and RADV would keep saying yes, right up until the kernel can't validate
        // the buffer list and the SUBMIT dies. Place the overflow on the non-device-local heap
        // instead: same DDR, same bandwidth on an APU. See `probe_uma_overflow_type`.
        if location == MemoryLocation::GpuOnly {
            if let Some(ty) = self.shared.uma_overflow_type {
                // Leave the device-local heap a little slack rather than filling it to the last
                // byte: the driver makes its own internal allocations there (descriptor pools,
                // pipeline/shader memory, the command buffers themselves), and a heap with zero
                // room is how the "not enough memory for command submission" failure starts.
                const UMA_SPILL_MARGIN: u64 = 256 * 1024 * 1024;
                let dl_avail = device_local_room(&self.shared);
                let fits_device_local =
                    dl_avail >= requirements.size.saturating_add(UMA_SPILL_MARGIN);
                if !fits_device_local && requirements.memory_type_bits & (1 << ty) != 0 {
                    // Both heaps are out if this fails, so report it rather than retrying on the
                    // device-local heap we just established has no room (that path would recurse
                    // straight back to here, and RADV would accept the allocation anyway and hand
                    // the failure to the next submit as a device-lost — the exact thing this
                    // spill exists to prevent).
                    return match self.alloc_vram_mapped(buffer, size, &requirements, ty, true) {
                        Ok(b) => Ok(b),
                        Err(e) => {
                            // `alloc_vram_mapped` leaves the buffer to us on failure.
                            unsafe { self.shared.device.destroy_buffer(buffer, None) };
                            Err(be(format!(
                                "unified memory exhausted: {} for {label} did not fit the \
                                 device-local heap ({} free) and the overflow heap rejected it \
                                 too ({e})",
                                fmt_bytes(requirements.size),
                                fmt_bytes(dl_avail),
                            )))
                        }
                    };
                }
            }
        }

        let allocation = {
            let mut alloc = self.shared.allocator.lock().unwrap();
            alloc
                .allocate(&AllocationCreateDesc {
                    name: label,
                    requirements,
                    location,
                    linear: true,
                    allocation_scheme: scheme,
                })
                .map_err(|e| {
                    // Clean up the buffer we created if allocation fails.
                    unsafe { self.shared.device.destroy_buffer(buffer, None) };
                    be(format!("gpu_allocator::allocate: {e}"))
                })?
        };

        unsafe {
            self.shared
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .map_err(|e| be(format!("bind_buffer_memory: {e}")))?;

        // Charge the budget guard's fallback accounting (balanced by `VkBuffer::drop`).
        if location == MemoryLocation::GpuOnly {
            self.shared
                .device_used
                .fetch_add(allocation.size(), Ordering::Relaxed);
        }

        Ok(VkBuffer {
            shared: Arc::clone(&self.shared),
            buffer,
            backing: Backing::Pooled(ManuallyDrop::new(allocation)),
            size,
            mem_size: requirements.size,
            location,
            sub_offset: 0,
        })
    }

    /// Fill a buffer with the repeated byte `byte` (0x00 = zero-init, 0xFF = poison). Host-visible
    /// buffers are memset through the mapped pointer (no submit); device-local buffers use
    /// `vkCmdFillBuffer` via a one-shot submit. Every OTHER `VkBuffer` owns a distinct `vk::Buffer`
    /// handle addressing its region from offset 0 (plain `WeightArena` buffers included), so filling
    /// `[0, size)` of the handle is correct; a resident-BDA sub-tensor (`Backing::BdaSub`) instead
    /// shares its block's ONE big handle, so the fill must start at `buf.sub_offset` or it would
    /// clobber whatever tensor happens to sit at the block's byte 0.
    fn fill_buf(&self, buf: &VkBuffer, byte: u8) -> Result<()> {
        if let Some(ptr) = buf.mapped_ptr() {
            unsafe { std::ptr::write_bytes(ptr, byte, buf.size) };
        } else {
            let word = u32::from_ne_bytes([byte; 4]);
            let size = (buf.size / 4 * 4) as u64; // vkCmdFillBuffer requires a 4-byte multiple
            if size > 0 {
                let vkbuf = buf.buffer;
                let off = buf.sub_offset as u64;
                let shared = Arc::clone(&self.shared);
                self.one_shot(move |cmd| unsafe {
                    shared.device.cmd_fill_buffer(cmd, vkbuf, off, size, word);
                })?;
            }
        }
        Ok(())
    }

    /// Allocate `sizes.len()` buffers and zero-init them with (at most) ONE submit — the batched
    /// twin of [`Backend::alloc`], same calloc contract. `alloc`'s per-buffer `fill_buf` costs a
    /// one-shot submit + `queue_wait_idle` per device-local buffer; a graph execute's scratch set
    /// (~70 Internal tensors) paid ~2.5ms of pure submit overhead per call on a 7900 XTX. Here
    /// host-visible buffers are memset through their mapped pointer and every device-local fill is
    /// recorded into a single one-shot command buffer.
    pub(crate) fn alloc_zeroed_batch(
        &self,
        sizes: &[usize],
        usage: BufferUsage,
    ) -> Result<Vec<Box<dyn Buffer>>> {
        let bufs: Vec<VkBuffer> = sizes
            .iter()
            .map(|&b| self.make_alloc(b, usage))
            .collect::<Result<_>>()?;
        let mut dev: Vec<(vk::Buffer, u64, u64)> = Vec::new();
        for buf in &bufs {
            if let Some(ptr) = buf.mapped_ptr() {
                unsafe { std::ptr::write_bytes(ptr, 0u8, buf.size) };
            } else {
                let size = (buf.size / 4 * 4) as u64; // vkCmdFillBuffer requires a 4-byte multiple
                if size > 0 {
                    dev.push((buf.buffer, buf.sub_offset as u64, size));
                }
            }
        }
        if !dev.is_empty() {
            let shared = Arc::clone(&self.shared);
            self.one_shot(move |cmd| unsafe {
                for (b, off, size) in dev {
                    shared.device.cmd_fill_buffer(cmd, b, off, size, 0);
                }
            })?;
        }
        Ok(bufs
            .into_iter()
            .map(|b| Box::new(b) as Box<dyn Buffer>)
            .collect())
    }

    /// The shared body of `alloc`/`alloc_uninit`: pick the memory location + tick the weight-load
    /// progress bar. Zero/poison filling is applied by the callers.
    fn make_alloc(&self, bytes: usize, usage: BufferUsage) -> Result<VkBuffer> {
        // `INFR_RESIDENT_BDA=1` (default OFF): resident weight tensors sub-allocate from the BDA
        // arena instead of the ReBAR / `WeightArena` / gpu-allocator paths inside `make_buf` — see
        // `bda_weight_alloc`. Every other `BufferUsage` (and the flag OFF) is byte-for-byte the old
        // behavior below.
        if usage == BufferUsage::Weights && resident_bda_enabled() {
            let buf = self.bda_weight_alloc(bytes)?;
            if let Some(pb) = self.shared.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
            return Ok(buf);
        }
        let (location, label) = match usage {
            BufferUsage::Weights => (MemoryLocation::GpuOnly, "weights"),
            BufferUsage::Activations => (MemoryLocation::GpuOnly, "activations"),
            BufferUsage::Staging => (MemoryLocation::CpuToGpu, "staging"),
            BufferUsage::Readback => (MemoryLocation::GpuToCpu, "readback"),
            // GpuToCpu = HOST_VISIBLE|HOST_CACHED system RAM (never the ReBAR device-local
            // host-visible heap CpuToGpu prefers) — the point of the class is NOT living in VRAM.
            BufferUsage::HostWeights => (MemoryLocation::GpuToCpu, "host-weights"),
        };
        let buf = self.make_buf(bytes, location, label)?;
        // Advance the weight-load progress bar (if active) — the single funnel every weight upload
        // passes through, so no loader can forget to account for a tensor.
        if matches!(usage, BufferUsage::Weights | BufferUsage::HostWeights) {
            if let Some(pb) = self.shared.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
        }
        Ok(buf)
    }

    /// Copy `src` into device-local `dst_buf` through the REUSED staging ring (see [`StagingRing`])
    /// — the weight-load path on a device without ReBAR.
    ///
    /// The tensor is chunked across fixed-size slots. For each chunk we wait only on the fence of
    /// the slot we are about to REUSE (not on the queue as a whole), memcpy into it, and submit its
    /// copy. With `RING_SLOTS` slots in flight the host's memcpy for chunk N+1 overlaps the DMA of
    /// chunk N, instead of the old `queue_wait_idle`-after-every-tensor lockstep.
    ///
    /// Uploads are not awaited here; [`WeightProgress::drop`] drains the ring, which happens long
    /// before any forward is submitted.
    ///
    /// `dst_base` is added to every chunk's destination offset — `0` for an ordinary weight buffer
    /// (the tensor owns `dst_buf` outright, so its region starts at byte 0), or a resident-BDA
    /// sub-tensor's `sub_offset` when `dst_buf` is actually a whole arena BLOCK shared with other
    /// tensors (see [`Backing::BdaSub`]) — without it every such tensor would land at the block's
    /// byte 0 and overwrite whatever the previous tensor wrote there.
    fn upload_staged_ring(&self, dst_buf: vk::Buffer, dst_base: u64, src: &[u8]) -> Result<()> {
        let device = &self.shared.device;
        let mut guard = self.shared.staging_ring.lock().unwrap();
        if guard.is_none() {
            *guard = Some(self.make_staging_ring()?);
        }
        let ring = guard.as_mut().expect("just built");

        let mut off = 0usize;
        while off < src.len() {
            let n = (src.len() - off).min(RING_SLOT_BYTES);
            let i = ring.next;
            ring.next = (ring.next + 1) % RING_SLOTS;

            // Reuse of a slot is the ONLY place we block: wait for its previous copy to land.
            if ring.busy[i] {
                unsafe { device.wait_for_fences(&[ring.fences[i]], true, u64::MAX) }
                    .map_err(|e| be(format!("staging ring wait_for_fences: {e}")))?;
                ring.busy[i] = false;
            }
            unsafe { device.reset_fences(&[ring.fences[i]]) }
                .map_err(|e| be(format!("staging ring reset_fences: {e}")))?;

            let ptr = ring.bufs[i]
                .mapped_ptr()
                .ok_or_else(|| be("staging ring slot is not mapped"))?;
            copy_to_mapped(&src[off..off + n], ptr);

            let cmd = ring.cmds[i];
            unsafe {
                device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| be(format!("staging ring reset_command_buffer: {e}")))?;
                device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| be(format!("staging ring begin_command_buffer: {e}")))?;
                device.cmd_copy_buffer(
                    cmd,
                    ring.bufs[i].buffer,
                    dst_buf,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: dst_base + off as u64,
                        size: n as u64,
                    }],
                );
                device
                    .end_command_buffer(cmd)
                    .map_err(|e| be(format!("staging ring end_command_buffer: {e}")))?;

                let cmds = [cmd];
                let submit = vk::SubmitInfo::default().command_buffers(&cmds);
                device
                    .queue_submit(self.shared.queue, &[submit], ring.fences[i])
                    .map_err(|e| be(format!("staging ring queue_submit: {e}")))?;
            }
            ring.busy[i] = true;
            off += n;
        }
        Ok(())
    }

    /// Allocate the staging ring's slots, command buffers and fences — ONCE per load.
    fn make_staging_ring(&self) -> Result<StagingRing> {
        let device = &self.shared.device;
        let pool = *self.shared.cmd_pool.lock().unwrap();
        let cmds = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(RING_SLOTS as u32),
            )
        }
        .map_err(|e| be(format!("staging ring allocate_command_buffers: {e}")))?;

        let mut bufs = Vec::with_capacity(RING_SLOTS);
        let mut fences = Vec::with_capacity(RING_SLOTS);
        for _ in 0..RING_SLOTS {
            bufs.push(self.make_buf(
                RING_SLOT_BYTES,
                MemoryLocation::CpuToGpu,
                "upload_staging",
            )?);
            fences.push(
                unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                    .map_err(|e| be(format!("staging ring create_fence: {e}")))?,
            );
        }
        Ok(StagingRing {
            bufs,
            cmds,
            fences,
            busy: vec![false; RING_SLOTS],
            next: 0,
        })
    }

    /// Record a single command into a one-shot command buffer, submit it to the
    /// compute queue, and block until idle.
    ///
    /// The closure receives the command buffer handle to record into.
    /// All operations are serialised through the `cmd_pool` mutex.
    fn one_shot(&self, f: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let device = &self.shared.device;
        let pool = *self.shared.cmd_pool.lock().unwrap();

        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| be(format!("allocate_command_buffers: {e}")))?[0];

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|e| be(format!("begin_command_buffer: {e}")))?;

        f(cmd);

        unsafe { device.end_command_buffer(cmd) }
            .map_err(|e| be(format!("end_command_buffer: {e}")))?;

        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        unsafe { device.queue_submit(self.shared.queue, &[submit], vk::Fence::null()) }
            .map_err(|e| be(format!("queue_submit: {e}")))?;

        unsafe { device.queue_wait_idle(self.shared.queue) }
            .map_err(|e| be(format!("queue_wait_idle: {e}")))?;

        unsafe { device.free_command_buffers(pool, &cmds) };
        Ok(())
    }
}

// ── Backend impl ──────────────────────────────────────────────────────────────

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn capabilities(&self) -> Capabilities {
        self.shared.caps.clone()
    }

    fn weight_progress(
        &self,
        total_bytes: Option<u64>,
    ) -> Box<dyn infr_core::backend::ProgressScope> {
        Box::new(self.weight_progress_scope(total_bytes))
    }

    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // calloc contract: zero-init so recycled/uninitialized VRAM can't leak into a read-before-write.
        let buf = self.make_alloc(bytes, usage)?;
        self.fill_buf(&buf, 0x00)?;
        Ok(Box::new(buf))
    }

    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // Opt-out: skip the zero-fill (caller guarantees the full extent is written before any read).
        // Debug builds poison with 0xFF (= NaN as f32) so a misuse surfaces loudly in tests;
        // INFR_POISON_UNINIT=1 forces the poison in release too — for hunting layout-sensitive
        // read-before-write bugs whose output shifts with unrelated code changes.
        let buf = self.make_alloc(bytes, usage)?;
        #[cfg(debug_assertions)]
        self.fill_buf(&buf, 0xFF)?;
        #[cfg(not(debug_assertions))]
        if std::env::var("INFR_POISON_UNINIT").is_ok() {
            self.fill_buf(&buf, 0xFF)?;
        }
        Ok(Box::new(buf))
    }

    /// Copy `src` (host slice) into `dst` (device buffer).
    ///
    /// If `dst` is host-visible (`CpuToGpu`), writes directly through the
    /// Device-side prefix copy (`vkCmdCopyBuffer` region `[0, bytes)`) — no host bounce.
    fn copy_buffer(&self, src: &dyn Buffer, dst: &dyn Buffer, bytes: usize) -> Result<()> {
        // Safety: every buffer from this backend is a VkBuffer.
        let (s, d) = unsafe { (as_vk_buf(src), as_vk_buf(dst)) };
        let (sb, db) = (s.buffer, d.buffer);
        // one_shot's queue_wait_idle provides the ordering fence vs prior/following work.
        self.one_shot(move |cmd| unsafe {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: bytes as u64,
            };
            self.shared.device.cmd_copy_buffer(cmd, sb, db, &[region]);
        })
    }

    /// persistent mapped pointer.  Otherwise, creates a temporary staging buffer,
    /// writes there, then submits a `cmd_copy_buffer` to the compute queue.
    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        // Safety: every buffer from this backend is a VkBuffer.
        let vk_dst = unsafe { as_vk_buf(dst) };
        if src.len() > vk_dst.size {
            return Err(be(format!(
                "upload: {} bytes into a {}-byte buffer",
                src.len(),
                vk_dst.size
            )));
        }

        // ── Direct write: any PERSISTENTLY MAPPED destination ─────────────────────────────────
        // Host-visible staging/readback buffers as before, AND — the big one — ReBAR weights
        // (`Backing::Vram`): device-local VRAM the host writes straight through. One pass over the
        // bytes, no staging buffer, no `vkCmdCopyBuffer`, no queue stall. The memory is
        // HOST_COHERENT (see `probe_rebar_type`) so no explicit flush is needed, and the host
        // writes are made visible to the device by the implicit host-write domain operation that
        // `vkQueueSubmit` performs — every weight is written long before the first forward is
        // submitted.
        if let Some(ptr) = vk_dst.mapped_ptr() {
            copy_to_mapped(src, ptr);
            return Ok(());
        }

        // ── Staged write: device-local destination with no host mapping ───────────────────────
        // During a weight load this goes through the REUSED, pipelined staging ring; anywhere else
        // (and for a tensor larger than the ring's slot on a non-load path) it is a single
        // synchronous copy.
        if self.shared.weight_pb.lock().unwrap().is_some() {
            return self.upload_staged_ring(vk_dst.buffer, vk_dst.sub_offset as u64, src);
        }

        let staging = self.make_buf(src.len(), MemoryLocation::CpuToGpu, "upload_staging")?;
        let stg_ptr = staging
            .mapped_ptr()
            .ok_or_else(|| be("staging buffer is not mapped"))?;
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), stg_ptr, src.len()) };

        let stg_buf = staging.buffer;
        let dst_buf = vk_dst.buffer;
        // `dst.sub_offset` — 0 for every ordinary buffer; a resident-BDA sub-tensor's offset within
        // its block's shared `vk::Buffer` otherwise (see `Backing::BdaSub`).
        let dst_off = vk_dst.sub_offset as u64;
        let size = src.len() as u64;
        // Clone the Arc so the closure is independent of `self`.
        let shared = Arc::clone(&self.shared);
        self.one_shot(move |cmd| {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: dst_off,
                size,
            };
            unsafe {
                shared
                    .device
                    .cmd_copy_buffer(cmd, stg_buf, dst_buf, &[region])
            };
        })?;
        // `staging` is dropped here → frees vk::Buffer + gpu-allocator sub-allocation.
        Ok(())
    }

    /// Copy `src` (device buffer) into `dst` (host slice).
    ///
    /// If `src` is host-visible (persistently mapped — `Readback`/`GpuToCpu` OR `Staging`/CpuToGpu),
    /// reads STRAIGHT from the mapped pointer: zero submit/sync. Only a truly device-local
    /// (`GpuOnly`, unmapped) source copies via a temporary readback staging buffer + submit + wait.
    ///
    /// Covering CpuToGpu here matters on the hot decode loop: the record-once replay binds the
    /// device sampler's id output to a `Staging` buffer (`dec_ids_buf`, dual-purposed as the next
    /// iteration's on-device embed-gather input), and the per-token fallback / E2B path reads that
    /// id back every step. The old `GpuToCpu`-only check bounced it through a staging alloc +
    /// one_shot copy + `queue_wait_idle` PER TOKEN — the exact per-token full-sync cost `read_pos0`
    /// already dodges for `positions`. The mapped read carries the same contract the `GpuToCpu`
    /// path always had: the caller must have completed the GPU work that wrote `src` (every decode
    /// site does — `execute`/`replay` end in `queue_wait_idle`; the buffers are HOST_COHERENT so
    /// the write is visible with no explicit invalidate).
    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        // Safety: every buffer from this backend is a VkBuffer.
        let vk_src = unsafe { as_vk_buf(src) };

        if let Some(ptr) = vk_src.mapped_ptr() {
            // Host-visible (Readback or Staging): direct read from the persistently-mapped pointer.
            unsafe { std::ptr::copy_nonoverlapping(ptr as *const u8, dst.as_mut_ptr(), dst.len()) };
        } else {
            // Readback path: device-local → staging → host.
            let staging = self.make_buf(dst.len(), MemoryLocation::GpuToCpu, "download_staging")?;

            let src_buf = vk_src.buffer;
            let stg_buf = staging.buffer;
            // `src.sub_offset` — see `upload`'s `dst_off`; a resident-BDA sub-tensor reads from its
            // offset within the block's shared `vk::Buffer`, not from byte 0.
            let src_off = vk_src.sub_offset as u64;
            let size = dst.len() as u64;
            let shared = Arc::clone(&self.shared);
            self.one_shot(move |cmd| {
                let region = vk::BufferCopy {
                    src_offset: src_off,
                    dst_offset: 0,
                    size,
                };
                unsafe {
                    shared
                        .device
                        .cmd_copy_buffer(cmd, src_buf, stg_buf, &[region])
                };
            })?;

            // GPU→staging transfer is complete (queue_wait_idle returned).
            let ptr = staging
                .mapped_ptr()
                .ok_or_else(|| be("readback staging is not mapped"))?
                as *const u8;
            unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len()) };
            // `staging` dropped here.
        }
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        adapter::compile(graph)
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        adapter::execute(self, plan, bindings)
    }

    /// See `Backend::max_decode_chain`. A device that needs its FORWARD split into several
    /// submits (`submit_dispatch_cap` — every integrated part measured so far) cannot also afford
    /// to pack several decode steps into one: a decode graph is hundreds of dispatches, i.e.
    /// already at or past that cap on its own, so the honest bound there is a chain of ONE. A
    /// device that never splits (every discrete GPU) keeps the unbounded default, and the tuned
    /// chained-decode fast path is untouched.
    fn max_decode_chain(&self) -> usize {
        if self.submit_dispatch_cap() == 0 {
            usize::MAX
        } else {
            1
        }
    }

    fn execute_chain(
        &self,
        plan: &dyn Plan,
        bindings: &Bindings,
        n: usize,
    ) -> Result<Option<Vec<u32>>> {
        adapter::execute_chain(self, plan, bindings, n)
    }

    fn sync(&self) -> Result<()> {
        unsafe { self.shared.device.device_wait_idle() }
            .map_err(|e| be(format!("device_wait_idle: {e}")))
    }

    fn moe_paged(&self) -> bool {
        self.moe_pager.lock().unwrap().is_some()
    }

    fn dense_paged(&self) -> bool {
        self.dense_pager.lock().unwrap().is_some()
    }

    /// DiffusionGemma perf slice 3 (docs/DIFFUSIONGEMMA.md): one eager dispatch of
    /// `dg_eb_sample` + a synchronous wait (`Recorder::finish`) — this isn't part of a cached
    /// [`Plan`], it runs once per denoise step right after that step's forward `execute()`, on
    /// the same `logits` buffer the forward just wrote (still GPU-resident).
    fn eb_sample_reduce(
        &self,
        logits: &dyn Buffer,
        u: &dyn Buffer,
        rows: usize,
        dim: usize,
        temp_inv: f32,
        argmax_out: &dyn Buffer,
        entropy_out: &dyn Buffer,
        sampled_out: &dyn Buffer,
    ) -> Result<bool> {
        let rec = self.recorder()?;
        rec.dg_eb_sample(
            logits,
            u,
            argmax_out,
            entropy_out,
            sampled_out,
            rows,
            dim,
            temp_inv,
        );
        rec.finish()?;
        Ok(true)
    }
}

/// Pick ONE cooperative-matrix (M,N,K) tile for a component type from the device's enumerated
/// shape list, by preference order:
///
/// 1. [`infr_core::COOPMAT_TILE_16`] (16x16x16) — the shape EVERY production coopmat shader is
///    built for; a device that enumerates it (RADV/RDNA3+, NVIDIA, and reportedly some
///    Battlemage drivers) always gets it, regardless of `allow_8x8x16` — the env knob must never
///    move a device off the proven kernel set.
/// 2. [`infr_core::COOPMAT_TILE_8`] (8x8x16, Intel Arc/ANV XMX) — only when `allow_8x8x16`
///    (the `INFR_CM_8X8=1` opt-in; only `native_gemm_warp`'s `_cm8` builds exist at this shape,
///    and Alchemist coopmat is a llama.cpp-documented regression, so it stays default-OFF).
/// 3. `None` — no shape any kernel here is built for; the non-coopmat tiers take over.
///
/// Pure function of the enumerated list + the opt-in flag (no env reads) so the selection is
/// unit-testable with synthetic property lists.
fn select_coopmat_shape(
    shapes: impl IntoIterator<Item = (u32, u32, u32)>,
    allow_8x8x16: bool,
) -> Option<(u32, u32, u32)> {
    let mut has_8x8x16 = false;
    for s in shapes {
        if s == infr_core::COOPMAT_TILE_16 {
            return Some(infr_core::COOPMAT_TILE_16);
        }
        has_8x8x16 |= s == infr_core::COOPMAT_TILE_8;
    }
    (allow_8x8x16 && has_8x8x16).then_some(infr_core::COOPMAT_TILE_8)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::Backend;

    /// Shape selection over synthetic property lists (no GPU needed) — the caps-table core of the
    /// shape-aware coopmat gate. RADV-like (16x16x16 present, plus the other shapes RADV
    /// enumerates) must pick 16x16x16 regardless of the 8x8 opt-in; ANV-like (8x8x16 only) must
    /// stay dark by default and pick 8x8x16 only under the opt-in; empty (no coopmat) is None.
    #[test]
    fn coopmat_shape_selection() {
        let t16 = infr_core::COOPMAT_TILE_16;
        let t8 = infr_core::COOPMAT_TILE_8;
        // RADV-like: 16x16x16 f16 (RDNA3 WMMA). Opt-in must NOT move it off 16x16x16.
        let radv = [t16];
        assert_eq!(select_coopmat_shape(radv, false), Some(t16));
        assert_eq!(select_coopmat_shape(radv, true), Some(t16));
        // Device enumerating BOTH shapes: 16x16x16 preferred, opt-in irrelevant.
        let both = [t8, t16];
        assert_eq!(select_coopmat_shape(both, false), Some(t16));
        assert_eq!(select_coopmat_shape(both, true), Some(t16));
        // ANV-like (Intel Arc A770): 8x8x16 only — default OFF, opt-in selects it.
        let anv = [t8];
        assert_eq!(select_coopmat_shape(anv, false), None);
        assert_eq!(select_coopmat_shape(anv, true), Some(t8));
        // No configs (ext absent / feature off): None either way.
        assert_eq!(select_coopmat_shape([], false), None);
        assert_eq!(select_coopmat_shape([], true), None);
        // A shape no kernel is built for (e.g. a hypothetical 32x32x16): None even with opt-in.
        assert_eq!(select_coopmat_shape([(32, 32, 16)], true), None);
    }

    /// Weight arena: reserve a small arena, allocate several `Weights` buffers from it (forcing both
    /// the reserved block and at least one overflow block), and verify each round-trips bytes through
    /// the staging copy path — proving arena buffers bind to valid, distinct memory regions.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn weight_arena_roundtrip() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        // Reserve only 1 MB so the later allocations spill into an overflow block.
        be.reserve_weights(1024 * 1024).expect("reserve_weights");
        let sizes = [4096usize, 256 * 1024, 4 * 1024 * 1024]; // last forces an overflow block
        let mut bufs = Vec::new();
        for (bi, &sz) in sizes.iter().enumerate() {
            let data: Vec<u8> = (0..sz)
                .map(|i| (i as u8).wrapping_add(bi as u8 * 31))
                .collect();
            let buf = be
                .alloc(sz, BufferUsage::Weights)
                .expect("arena weight alloc");
            be.upload(buf.as_ref(), &data).expect("upload");
            let mut back = vec![0u8; sz];
            be.download(buf.as_ref(), &mut back).expect("download");
            assert_eq!(
                back, data,
                "arena buffer {bi} (size {sz}) round-trip mismatch"
            );
            bufs.push(buf);
        }
        // All three buffers coexist (distinct memory) — re-download the first and re-check.
        let mut back0 = vec![0u8; sizes[0]];
        be.download(bufs[0].as_ref(), &mut back0)
            .expect("re-download");
        assert_eq!(
            back0[1], 1u8,
            "first arena buffer corrupted by later allocs"
        );
    }

    /// Weight arena, MULTI-BLOCK reservation: reserve a `total` LARGER than one block (forcing the
    /// up-front split in `reserve_weights_capped` to place several blocks), assert `blocks.len() > 1`
    /// with distinct memory handles, then sub-allocate weight buffers that straddle the block
    /// boundary and round-trip bytes through each — proving cross-block sub-allocation is
    /// byte-correct. Uses a tiny per-block cap so the test commits only a few MiB, not GiB.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn weight_arena_multi_block() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        // Cap each block at 1 MiB and reserve 3.5 MiB → at least 4 blocks placed up front.
        const CAP: u64 = 1024 * 1024;
        const TOTAL: u64 = 7 * 512 * 1024; // 3.5 MiB
        be.reserve_weights_capped(TOTAL, CAP)
            .expect("reserve_weights_capped");
        {
            let arena = be.shared.weight_arena.lock().unwrap();
            let a = arena.as_ref().expect("arena reserved");
            assert!(
                a.blocks.len() > 1,
                "expected a multi-block reservation, got {} block(s)",
                a.blocks.len()
            );
            // Every block's device memory handle must be distinct.
            for i in 0..a.blocks.len() {
                for j in (i + 1)..a.blocks.len() {
                    assert_ne!(
                        a.blocks[i].memory, a.blocks[j].memory,
                        "arena blocks {i} and {j} share device memory"
                    );
                }
            }
            // The placed capacity must cover the whole request.
            let placed: u64 = a.blocks.iter().map(|b| b.size).sum();
            assert!(placed >= TOTAL, "placed {placed} < requested {TOTAL}");
        }
        // Sub-allocate several ~700 KiB weights: each is smaller than a 1 MiB block but the sequence
        // forces `bump` to start fresh blocks (a weight never straddles a block), walking across the
        // reserved blocks and into an overflow block. Round-trip each to prove the binding is valid.
        let sizes = [
            700 * 1024usize,
            700 * 1024,
            700 * 1024,
            700 * 1024,
            700 * 1024,
        ];
        let mut bufs = Vec::new();
        for (bi, &sz) in sizes.iter().enumerate() {
            let data: Vec<u8> = (0..sz)
                .map(|i| (i as u8).wrapping_add(bi as u8 * 17))
                .collect();
            let buf = be
                .alloc(sz, BufferUsage::Weights)
                .expect("arena weight alloc");
            be.upload(buf.as_ref(), &data).expect("upload");
            let mut back = vec![0u8; sz];
            be.download(buf.as_ref(), &mut back).expect("download");
            assert_eq!(
                back, data,
                "arena buffer {bi} (size {sz}) round-trip mismatch"
            );
            bufs.push(buf);
        }
        // All coexist in distinct regions — re-check the first after the later allocs.
        let mut back0 = vec![0u8; sizes[0]];
        be.download(bufs[0].as_ref(), &mut back0)
            .expect("re-download");
        assert_eq!(
            back0[3], 3u8,
            "first arena buffer corrupted by later allocs"
        );
    }

    /// Resident-BDA weight arena (`INFR_RESIDENT_BDA`): sub-allocate three odd-sized weight buffers
    /// directly from `bda_weight_alloc` (bypassing the env flag / `resident_bda_enabled`'s
    /// process-global `OnceLock` entirely, so this test can't race other tests that touch
    /// `BufferUsage::Weights` in the same process), and verify:
    ///   * every sub-tensor reports `Some(device_addr)`,
    ///   * addresses are 256-byte aligned and strictly increasing within the (single, since the
    ///     three sizes together are far under `BDA_BLOCK_MIN`) block,
    ///   * distinct byte patterns uploaded to each round-trip intact — proving the `sub_offset`
    ///     plumbing on both `upload` (staged one-shot path, no weight-load scope open) and
    ///     `download`,
    ///   * a plain `Activations` alloc through the ordinary `Backend::alloc` path still reports
    ///     `device_addr() == None` — the flag only ever affects `Weights` allocs routed through
    ///     `bda_weight_alloc`, never anything else.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn resident_bda_weight_arena_roundtrip() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let sizes = [1000usize, 4096, 300_000];
        let mut addrs = Vec::new();
        let mut bufs = Vec::new();
        for (bi, &sz) in sizes.iter().enumerate() {
            let data: Vec<u8> = (0..sz)
                .map(|i| (i as u8).wrapping_add(bi as u8 * 53))
                .collect();
            let buf = be.bda_weight_alloc(sz).expect("bda_weight_alloc");
            let addr = buf
                .device_addr()
                .expect("resident-BDA sub-tensor must report Some(device_addr)");
            assert_eq!(
                addr % 256,
                0,
                "sub-tensor {bi} device_addr {addr:#x} is not 256-byte aligned"
            );
            if let Some(&prev) = addrs.last() {
                assert!(
                    addr > prev,
                    "sub-tensor {bi} device_addr {addr:#x} did not increase past the previous \
                     tensor's {prev:#x}"
                );
            }
            addrs.push(addr);

            be.upload(&buf, &data).expect("upload");
            let mut back = vec![0u8; sz];
            be.download(&buf, &mut back).expect("download");
            assert_eq!(
                back, data,
                "sub-tensor {bi} (size {sz}) round-trip mismatch"
            );
            bufs.push(buf);
        }
        // All three coexist in distinct byte ranges of the same block — re-check the first after
        // the later uploads landed, proving they didn't overlap/clobber it.
        let mut back0 = vec![0u8; sizes[0]];
        be.download(&bufs[0], &mut back0).expect("re-download");
        assert_eq!(
            back0[1], 1u8,
            "first sub-tensor corrupted by later resident-BDA allocs"
        );

        // A plain Activations alloc is unaffected by any of this — never routed through
        // `bda_weight_alloc`, so it must report no device address.
        let act = be
            .alloc(64, BufferUsage::Activations)
            .expect("Activations alloc");
        assert!(
            act.device_addr().is_none(),
            "an ordinary Activations buffer must not report a device_addr"
        );
    }

    /// Dropping the backend must actually drop `VulkanShared` (device, allocator, weight arena —
    /// i.e. free the VRAM) even after a paged-MoE session was installed. The session's arena/LUT/
    /// ring buffers each hold an `Arc<VulkanShared>` clone, so parking the session ON
    /// `VulkanShared` formed an Arc cycle that leaked the whole device (~23 GiB after the Scout
    /// paged test) until process exit — every later model load in the same process then hit the
    /// VRAM budget guard with "N GiB already in use" (the cpu_backend gpu_ suite flake).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn backend_drop_frees_device_after_moe_pager() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        be.init_moe_pager(crate::pager::MoePagerLayout {
            n_blocks: 4,
            pools: vec![crate::pager::MoePoolSpec {
                role: crate::pager::Role::Gate,
                slot_bytes: 4096,
                n_slots: 2,
            }],
            ring_bytes: 1 << 20,
        })
        .expect("init_moe_pager");
        let weak = Arc::downgrade(&be.shared);
        drop(be);
        assert!(
            weak.upgrade().is_none(),
            "VulkanShared leaked after dropping the backend (Arc cycle via the paged-MoE session)"
        );
    }

    /// GPU f32 matmul correctness: compares `VulkanBackend::matmul_f32` against a CPU
    /// reference; asserts max relative error < 1e-3.
    ///
    /// Run with: `cargo test -p infr-vulkan -- --ignored --nocapture`
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn test_matmul_f32() {
        let backend = VulkanBackend::new().expect("VulkanBackend::new failed");
        let caps = backend.capabilities();
        println!("device: {}", caps.name);

        let (m, k, n) = (32usize, 32usize, 32usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

        // CPU reference
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c_ref[i * n + j] = sum;
            }
        }

        let c_gpu = backend
            .matmul_f32(&a, &b, m, k, n)
            .expect("matmul_f32 failed");

        let max_abs = c_gpu
            .iter()
            .zip(c_ref.iter())
            .map(|(g, r)| (*g - r).abs())
            .fold(0.0f32, f32::max);
        let max_ref = c_ref.iter().map(|r| r.abs()).fold(0.0f32, f32::max);
        let rel_err = if max_ref > 1e-6 {
            max_abs / max_ref
        } else {
            max_abs
        };

        println!("matmul {m}×{k}×{n}: max_rel_err = {rel_err:.2e}");
        assert!(rel_err < 1e-3, "matmul rel error too large: {rel_err:.2e}");
        println!("matmul GPU test PASS");
    }

    /// End-to-end roundtrip: init → alloc (device-local) → upload → download → assert.
    ///
    /// Marked `#[ignore]` so CI without a GPU passes; run manually with:
    /// ```text
    /// cargo test -p infr-vulkan -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn roundtrip_upload_download() {
        let backend = VulkanBackend::new().expect("VulkanBackend::new failed");

        let caps = backend.capabilities();
        println!("=== Capabilities ===\n{caps:#?}\n");

        const N: usize = 1024;
        // Pattern: bytes 0x00..0xFF repeating.
        let pattern: Vec<u8> = (0..N).map(|i| (i % 256) as u8).collect();

        // Alloc a device-local buffer (exercises the staging copy path).
        let buf = backend
            .alloc(N, BufferUsage::Weights)
            .expect("alloc Weights buffer");

        backend
            .upload(buf.as_ref(), &pattern)
            .expect("upload host→device");

        let mut got = vec![0u8; N];
        backend
            .download(buf.as_ref(), &mut got)
            .expect("download device→host");

        assert_eq!(pattern, got, "roundtrip data mismatch at 1024 bytes");

        backend.sync().expect("sync");

        println!("roundtrip OK — {N} bytes match");
    }
}

// qwen35 (Qwen3.5) SSM kernels: the GPU conv1d+SiLU and gated-DeltaNet recurrence must match the CPU
// reference. Self-skip without a GPU (so CI passes, runs locally with a device).
#[cfg(test)]
mod ssm_tests {
    use super::*;
    use infr_core::backend::{Buffer, BufferUsage};

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    fn softplus(x: f32) -> f32 {
        x.max(0.0) + (-x.abs()).exp().ln_1p()
    }
    fn det(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.137 + seed).sin()).collect()
    }
    fn dev(be: &VulkanBackend, data: &[f32]) -> Box<dyn Buffer> {
        let b = be
            .alloc((data.len() * 4).max(4), BufferUsage::Activations)
            .unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(data)).unwrap();
        b
    }
    fn read(be: &VulkanBackend, buf: &dyn Buffer, n: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; n * 4];
        be.download(buf, &mut bytes).unwrap();
        bytemuck::cast_slice(&bytes).to_vec()
    }
    fn maxerr(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn softcap_matches_cpu() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let (n, cap) = (100usize, 30.0f32);
        let x = det(n, 0.5);
        let out_cpu: Vec<f32> = x.iter().map(|&v| cap * (v / cap).tanh()).collect();
        let xb = dev(&be, &x);
        let ob = be.alloc(n * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.softcap(xb.as_ref(), ob.as_ref(), cap, n);
        rec.finish().unwrap();
        let out_gpu = read(&be, ob.as_ref(), n);
        let e = maxerr(&out_cpu, &out_gpu);
        assert!(e < 1e-4, "softcap err {e}");
    }

    #[test]
    fn deltanet_matches_cpu() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let (nv, nk, kd, vd) = (4usize, 2usize, 8usize, 8usize);
        let eps = 1e-6f32;
        let q = det(nk * kd, 0.1);
        let k = det(nk * kd, 0.7);
        let v = det(nv * vd, 1.3);
        let blog = det(nv, 2.0);
        let alpha = det(nv, 0.5);
        let acoef: Vec<f32> = (0..nv).map(|i| -(0.2 + 0.1 * i as f32)).collect();
        let dtbias = det(nv, -0.3);
        let state0 = det(nv * kd * vd, 0.05);

        // CPU reference (mirrors shaders/deltanet.comp + the qwen35 CPU mixer).
        let qscale = 1.0 / (kd as f32).sqrt();
        let mut s = state0.clone();
        let mut out_cpu = vec![0f32; nv * vd];
        for h in 0..nv {
            let khid = h % nk;
            let mut qh = q[khid * kd..khid * kd + kd].to_vec();
            let mut kh = k[khid * kd..khid * kd + kd].to_vec();
            let qn = (qh.iter().map(|x| x * x).sum::<f32>() + eps).sqrt();
            let kn = (kh.iter().map(|x| x * x).sum::<f32>() + eps).sqrt();
            for x in qh.iter_mut() {
                *x = *x / qn * qscale;
            }
            for x in kh.iter_mut() {
                *x /= kn;
            }
            let beta = sigmoid(blog[h]);
            let decay = (acoef[h] * softplus(alpha[h] + dtbias[h])).exp();
            let sb = h * kd * vd;
            for d in 0..vd {
                let mut kvv = 0.0;
                for kk in 0..kd {
                    let sv = s[sb + kk * vd + d] * decay;
                    s[sb + kk * vd + d] = sv;
                    kvv += kh[kk] * sv;
                }
                let delta = (v[h * vd + d] - kvv) * beta;
                let mut o = 0.0;
                for kk in 0..kd {
                    let sv = s[sb + kk * vd + d] + kh[kk] * delta;
                    s[sb + kk * vd + d] = sv;
                    o += qh[kk] * sv;
                }
                out_cpu[h * vd + d] = o;
            }
        }

        let (qb, kb, vb) = (dev(&be, &q), dev(&be, &k), dev(&be, &v));
        let (bb, ab) = (dev(&be, &blog), dev(&be, &alpha));
        let (acb, dtb) = (dev(&be, &acoef), dev(&be, &dtbias));
        let sbuf = dev(&be, &state0);
        let ob = be.alloc(nv * vd * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.deltanet(
            qb.as_ref(),
            kb.as_ref(),
            vb.as_ref(),
            bb.as_ref(),
            ab.as_ref(),
            acb.as_ref(),
            dtb.as_ref(),
            sbuf.as_ref(),
            ob.as_ref(),
            1, // rows: single-token bespoke path
            nv,
            nk,
            kd,
            vd,
            eps,
        );
        rec.finish().unwrap();
        let out_gpu = read(&be, ob.as_ref(), nv * vd);
        let s_gpu = read(&be, sbuf.as_ref(), nv * kd * vd);
        assert!(
            maxerr(&out_cpu, &out_gpu) < 1e-4,
            "deltanet out err {}",
            maxerr(&out_cpu, &out_gpu)
        );
        assert!(
            maxerr(&s, &s_gpu) < 1e-4,
            "deltanet state err {}",
            maxerr(&s, &s_gpu)
        );
    }

    #[test]
    fn conv1d_silu_matches_cpu() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let (cc, kconv) = (40usize, 4usize);
        let qkv = det(cc, 0.2);
        let w = det(cc * kconv, 1.1);
        let state0 = det((kconv - 1) * cc, 0.3);

        // CPU reference (mirrors shaders/conv1d_silu.comp).
        let mut st = state0.clone();
        let mut out_cpu = vec![0f32; cc];
        let km1 = kconv - 1;
        for ch in 0..cc {
            let mut acc = 0.0;
            for k in 0..km1 {
                acc += st[k * cc + ch] * w[ch * kconv + k];
            }
            acc += qkv[ch] * w[ch * kconv + km1];
            out_cpu[ch] = acc * sigmoid(acc);
            for k in 0..km1 - 1 {
                st[k * cc + ch] = st[(k + 1) * cc + ch];
            }
            st[(km1 - 1) * cc + ch] = qkv[ch];
        }

        let xb = dev(&be, &qkv);
        let wb = dev(&be, &w);
        let sbuf = dev(&be, &state0);
        let ob = be.alloc(cc * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.conv1d_silu(
            xb.as_ref(),
            wb.as_ref(),
            sbuf.as_ref(),
            ob.as_ref(),
            1, // rows: single-token bespoke path
            cc,
            kconv,
        );
        rec.finish().unwrap();
        let out_gpu = read(&be, ob.as_ref(), cc);
        let s_gpu = read(&be, sbuf.as_ref(), (kconv - 1) * cc);
        assert!(
            maxerr(&out_cpu, &out_gpu) < 1e-5,
            "conv out err {}",
            maxerr(&out_cpu, &out_gpu)
        );
        assert!(
            maxerr(&st, &s_gpu) < 1e-5,
            "conv state err {}",
            maxerr(&st, &s_gpu)
        );
    }
}
