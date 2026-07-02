//! Single-command-buffer forward recorder. Records many op dispatches (linear, rmsnorm,
//! rope, attention, silu_mul, add) into ONE command buffer with a conservative global
//! compute→compute barrier between each, then submits once on `finish()`. This collapses
//! the ~per-op submit+wait round-trips (the real bottleneck) to one per forward.
//!
//! Buffers are caller-owned `Box<dyn Buffer>` (from `Backend::alloc`); the recorder only
//! binds them. Reuses the cached kernels in `VulkanShared.kernels`.

use std::cell::RefCell;
use std::collections::HashSet;

use ash::vk;

use infr_core::{backend::Buffer, error::Result};

use super::ops::ComputeKernel;
use super::{as_vk_buf, be, VulkanBackend};

/// Output rows computed per workgroup by the subgroup decode GEMV (`mul_mat_vec_q.comp`); must match
/// the shader's `NUM_ROWS`.
const MMV_NUM_ROWS: u32 = 1;

/// Elements packed per u32 weight word for a given quant width (8 nibbles for q4, 4 bytes for q8).
/// `None` ⇒ the subgroup GEMV has no specialization for this width; caller uses the WGSL fallback.
fn mmv_epw(bits: u32) -> Option<usize> {
    match bits {
        4 => Some(8),
        8 => Some(4),
        _ => None,
    }
}

pub struct Recorder<'a> {
    be: &'a VulkanBackend,
    cmd: vk::CommandBuffer,
    pool: vk::DescriptorPool,
    /// Buffers written since the last barrier (for read-after-write / write-after-write detection).
    dirty_writes: RefCell<HashSet<vk::Buffer>>,
    /// Buffers read since the last barrier (for write-after-read detection).
    dirty_reads: RefCell<HashSet<vk::Buffer>>,
    /// Whether any un-barriered write was produced by a transfer (copy) rather than a shader.
    dirty_transfer: std::cell::Cell<bool>,
    barriers: RefCell<usize>,
    /// Debug knobs, read once (avoid env lookups in the per-dispatch hot path).
    no_barrier: bool,
    full_barrier: bool,
    prof: bool,
    /// Per-op GPU timestamp profiling (INFR_PROF2): a timestamp before each op + one at finish,
    /// attributed to per-op-type labels.
    prof2: bool,
    query_pool: vk::QueryPool,
    ts_labels: RefCell<Vec<&'static str>>,
    /// One-shot prof2 label override: the next `stamp()` uses this instead of its default, then
    /// clears it. Lets a caller attribute a generic op (e.g. a `linear` used for the vocab head vs a
    /// projection) to a distinct bucket without per-op API plumbing.
    next_label: std::cell::Cell<Option<&'static str>>,
    /// Record-once: when set, the command buffer is begun resubmittable (no ONE_TIME_SUBMIT) and
    /// `finish_record` hands back its cmd buffer + pool (a [`RecordedCmd`]) instead of submitting and
    /// freeing — so the caller can replay it across tokens.
    persistent: bool,
}

impl<'a> Recorder<'a> {
    pub(crate) fn new(backend: &'a VulkanBackend) -> Result<Self> {
        Self::new_inner(backend, false)
    }

    /// A recorder whose command buffer is resubmittable (no ONE_TIME_SUBMIT). `finish_record` returns
    /// a [`RecordedCmd`] the caller can replay instead of re-recording. Profiling is disabled on this
    /// path (the recorder is gone after recording, so per-replay timestamps can't be reported).
    pub(crate) fn new_persistent(backend: &'a VulkanBackend) -> Result<Self> {
        Self::new_inner(backend, true)
    }

    fn new_inner(backend: &'a VulkanBackend, persistent: bool) -> Result<Self> {
        let device = &backend.shared.device;
        let cmd_pool = *backend.shared.cmd_pool.lock().unwrap();
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| be(format!("alloc cmd buffer: {e}")))?[0];
        let begin_flags = if persistent {
            vk::CommandBufferUsageFlags::empty()
        } else {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        };
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(begin_flags),
            )
        }
        .map_err(|e| be(format!("begin cmd buffer: {e}")))?;

        // Big pool: one descriptor set per recorded op.
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 16384,
        }];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(4096)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|e| be(format!("create recorder pool: {e}")))?;

        const MAX_TS: u32 = 8192;
        // No per-op profiling on the persistent (replayed) path — the recorder is dropped after
        // recording, so it can't report timestamps for replays.
        let prof2 = std::env::var("INFR_PROF2").is_ok() && !persistent;
        let query_pool = if prof2 {
            let qp = unsafe {
                device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(MAX_TS),
                    None,
                )
            }
            .map_err(|e| be(format!("create query pool: {e}")))?;
            unsafe { device.cmd_reset_query_pool(cmd, qp, 0, MAX_TS) };
            qp
        } else {
            vk::QueryPool::null()
        };

        Ok(Self {
            be: backend,
            cmd,
            pool,
            dirty_writes: RefCell::new(HashSet::new()),
            dirty_reads: RefCell::new(HashSet::new()),
            dirty_transfer: std::cell::Cell::new(false),
            barriers: RefCell::new(0),
            no_barrier: std::env::var("INFR_NOBARRIER").is_ok(),
            full_barrier: std::env::var("INFR_FULLBARRIER").is_ok(),
            prof: std::env::var("INFR_PROF").is_ok(),
            prof2,
            query_pool,
            ts_labels: RefCell::new(Vec::new()),
            next_label: std::cell::Cell::new(None),
            persistent,
        })
    }

    /// Override the label of the NEXT profiled op (INFR_PROF2). Consumed once. No-op without prof2.
    pub fn label_next(&self, label: &'static str) {
        if self.prof2 {
            self.next_label.set(Some(label));
        }
    }

    /// Record a profiling timestamp (BOTTOM_OF_PIPE) tagged with an op label, if INFR_PROF2.
    fn stamp(&self, label: &'static str) {
        if !self.prof2 {
            return;
        }
        let label = self.next_label.take().unwrap_or(label);
        let idx = self.ts_labels.borrow().len() as u32;
        unsafe {
            self.be.shared.device.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                idx,
            );
        }
        self.ts_labels.borrow_mut().push(label);
    }

    /// Emit a global compute/transfer barrier only if `reads`/`writes` collide with work recorded
    /// since the last barrier (RAW / WAR / WAW). Independent dispatches then overlap on the GPU.
    /// `dst_transfer` = this op consumes via a transfer (copy) rather than a compute shader.
    /// Returns once any required barrier is recorded; updates the hazard-tracking sets.
    fn sync(&self, reads: &[vk::Buffer], writes: &[vk::Buffer], dst_transfer: bool) {
        if self.no_barrier {
            return;
        }
        let dw = self.dirty_writes.borrow();
        let dr = self.dirty_reads.borrow();
        let raw = self.full_barrier || reads.iter().any(|b| dw.contains(b));
        let waw = writes.iter().any(|b| dw.contains(b));
        let war = writes.iter().any(|b| dr.contains(b));
        drop(dw);
        drop(dr);
        if raw || waw || war {
            let mb = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::TRANSFER_WRITE,
                );
            // Only widen the (expensive) stage mask to TRANSFER when a copy is actually on the
            // producing or consuming side — most barriers are pure compute→compute.
            let mut src = vk::PipelineStageFlags::COMPUTE_SHADER;
            if self.dirty_transfer.get() {
                src |= vk::PipelineStageFlags::TRANSFER;
            }
            let mut dst = vk::PipelineStageFlags::COMPUTE_SHADER;
            if dst_transfer {
                dst |= vk::PipelineStageFlags::TRANSFER;
            }
            unsafe {
                self.be.shared.device.cmd_pipeline_barrier(
                    self.cmd,
                    src,
                    dst,
                    vk::DependencyFlags::empty(),
                    &[mb],
                    &[],
                    &[],
                );
            }
            self.dirty_writes.borrow_mut().clear();
            self.dirty_reads.borrow_mut().clear();
            self.dirty_transfer.set(false);
            *self.barriers.borrow_mut() += 1;
        }
        self.dirty_reads.borrow_mut().extend(reads.iter().copied());
        self.dirty_writes
            .borrow_mut()
            .extend(writes.iter().copied());
    }

    fn dispatch(
        &self,
        k: ComputeKernel,
        buffers: &[vk::Buffer],
        n_out: usize,
        push: &[u8],
        groups: u32,
    ) {
        self.dispatch3(k, buffers, n_out, push, groups, 1, 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch3(
        &self,
        k: ComputeKernel,
        buffers: &[vk::Buffer],
        n_out: usize,
        push: &[u8],
        gx: u32,
        gy: u32,
        gz: u32,
    ) {
        // The last `n_out` bound buffers are outputs; the rest are inputs. Inputs keep in-place
        // buffers (e.g. rope x==y) so a RAW from a prior op is still seen.
        let split = buffers.len() - n_out;
        let (reads, writes) = buffers.split_at(split);
        self.sync(reads, writes, false);
        let device = &self.be.shared.device;
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool)
                    .set_layouts(std::slice::from_ref(&k.ds_layout)),
            )
        }
        .expect("alloc descriptor set")[0];

        let infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|&buffer| vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        let ds_writes: Vec<vk::WriteDescriptorSet> = (0..buffers.len())
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i..i + 1])
            })
            .collect();
        unsafe { device.update_descriptor_sets(&ds_writes, &[]) };

        unsafe {
            device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline);
            device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                k.pipeline_layout,
                0,
                &[set],
                &[],
            );
            if k.push_size > 0 {
                device.cmd_push_constants(
                    self.cmd,
                    k.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            device.cmd_dispatch(self.cmd, gx, gy, gz);
        }
    }

    fn vkb(b: &dyn Buffer) -> vk::Buffer {
        unsafe { as_vk_buf(b) }.buffer
    }

    /// `y[rows,out] = x[rows,in] · Wᵀ` (W stored `[out,in]`).
    pub fn linear(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.stamp("lm_head");
        let k = self
            .be
            .kernel("linear_f16", crate::gemm::linear_f16_spv(), 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(w), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32, // one workgroup per output element (coalesced GEMV)
        );
    }

    /// Prefill projection GEMM: `c[m,n] = a[m,k] · Wᵀ` on the matrix cores (coopmat). `a` is f32;
    /// `wq` is the weight (f16 packed 2/u32 with bits=16, or quant idx with bits=4|8 + scales/mins).
    /// `c` MUST be allocated `ceil(m/64)*64` rows (the kernel writes padded rows as 0). `n%64==0`,
    /// `k%32==0`. For f16 weights pass any small non-empty buffer for scales/mins (unused).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_proj(
        &self,
        a: &dyn Buffer,
        wq: &dyn Buffer,
        scales: &dyn Buffer,
        mins: &dyn Buffer,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
        bits: u32,
        blk_shift: u32,
    ) {
        self.stamp("matmul_proj");
        // Warp tile (BM=64,BN=256, 256 threads / 8 warps — matches llama.cpp's AMD-RADV large
        // warptile; the extra warps hide W-dequant latency). Wins big for M≥768 (low/mid ctx:
        // 4k+21% 8k+19% 16k+5%); at very small M (32k chunk≈500) its wide N tile still loses to the
        // BN=64 tiled kernel, so gate on M. Also needs N%256.
        let warp = m >= 768 && n.is_multiple_of(256);
        let (name, spv, tiles_n) = if warp {
            ("gemm_proj_warp", crate::gemm::gemm_proj_warp_spv(), n / 256)
        } else {
            ("gemm_proj", crate::gemm::gemm_proj_spv(), n / 64)
        };
        let kern = self.be.kernel_sg(name, spv, 5, 20, 32);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&bits.to_ne_bytes());
        push[16..20].copy_from_slice(&blk_shift.to_ne_bytes());
        let groups = (m.div_ceil(64) * tiles_n) as u32; // both kernels use BM=64
        self.dispatch(
            kern,
            &[
                Self::vkb(a),
                Self::vkb(wq),
                Self::vkb(scales),
                Self::vkb(mins),
                Self::vkb(c),
            ],
            1,
            &push,
            groups,
        );
    }

    /// Native-block projection GEMM `c = a · Wᵀ` for prefill: raw GGUF blocks dequantized in-shader
    /// during coopmat tiled staging (decode-once per weight element, reused across the 64-row tile).
    /// `c` is allocated `ceil(m/64)*64` rows. Requires `n%64==0`, `k%32==0`.
    pub fn matmul_native(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer,
        w: &dyn Buffer,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.matmul_native_off(dtype, a, w, 0, c, m, k, n);
    }

    /// Native-block tiled coopmat GEMM reading the weight from element offset `w_base` — lets one
    /// stacked MoE expert tensor serve all experts (`w_base = expert_id * out_f * in_f`), so each
    /// expert weight is decoded ONCE and reused across the 64-row tile (vs the per-row GEMV re-read).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_native_off(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.stamp("matmul_proj");
        // Large-warptile variant (8-warp BM=64×BN=256): 4× the math per staged A-row / decoded
        // W-column vs the 64×64 tile, and the extra warps hide the dequant latency. Needs n%256,
        // k%32; only the hot formats are compiled — everything else stays on the 64×64 kernel.
        // INFR_NO_GEMM_WARP forces the 64×64 tile (A/B).
        let warp = if n.is_multiple_of(256)
            && k.is_multiple_of(32)
            && std::env::var("INFR_NO_GEMM_WARP").is_err()
        {
            crate::gemm::native_gemm_warp_build_spv(dtype)
        } else {
            None
        };
        let (name, spv) = match (warp, dtype) {
            (Some(spv), infr_core::DType::Q4K) => ("native_gemm_warp_q4k", spv),
            (Some(spv), infr_core::DType::Q6K) => ("native_gemm_warp_q6k", spv),
            (Some(spv), infr_core::DType::Q8_0) => ("native_gemm_warp_q8_0", spv),
            _ => (
                crate::linear::native_gemm_kernel_name(dtype),
                crate::gemm::native_gemm_build_spv(dtype).expect("native GEMM spv"),
            ),
        };
        let kern = self.be.kernel_sg(name, spv, 3, 16, 32);
        let groups_n = if warp.is_some() { n / 256 } else { n / 64 };
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * groups_n) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(w), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// Tiled Q4_K dp4a (mmq) GEMM for a stacked expert: `c = qa·W[w_base]ᵀ` using hardware int8
    /// dot-product (activations pre-quantized via `quant_q8`). `c` is `ceil(m/64)*64` rows. Faster
    /// than the coopmat-f16 `matmul_native` for u4 weights (the dense path's default for u4).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mmq_q4k(
        &self,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.stamp("matmul_proj");
        let kern = self.be.kernel(
            "native_gemm_mmq_q4k",
            crate::gemm::native_gemm_mmq_q4k_spv(),
            5,
            16,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * (n / 64)) as u32;
        self.dispatch(
            kern,
            &[
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(w),
                Self::vkb(c),
            ],
            1,
            &push,
            groups,
        );
    }

    /// Tiled Q6_K dp4a (mmq) GEMM for a stacked expert (the MoE down projection): `c = qa·W[w_base]ᵀ`
    /// using hardware int8 dot-product (activations pre-quantized via `quant_q8`; the per-block sum is
    /// unused — Q6_K is symmetric, no min). `c` is `ceil(m/64)*64` rows. Faster than the coopmat-f16
    /// `matmul_native` for u6 weights. Requires `n%64`, `k%16`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mmq_q6k(
        &self,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.stamp("matmul_proj");
        let kern = self.be.kernel(
            "native_gemm_mmq_q6k",
            crate::gemm::native_gemm_mmq_q6k_spv(),
            4,
            16,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * (n / 64)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(qa), Self::vkb(dact), Self::vkb(w), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// Integer (dp4a) u4 projection GEMM — the mmq path. Quantizes activations to int8 (Q8 per
    /// 32-block) via `quant_q8`, then runs the dp4a matmul keeping weights quantized (no per-GEMM
    /// dequant). Scratch (caller-allocated): `qa` = m*k bytes (int8), `dact`/`sact` = m*(k/32)*2
    /// bytes (f16). `c` = ceil(m/64)*64 rows f32. Requires u4 weights, `n%64==0`, `k%32==0`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_proj_mmq(
        &self,
        a: &dyn Buffer,
        wq: &dyn Buffer,
        scales: &dyn Buffer,
        mins: &dyn Buffer,
        c: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let nblk = k / 32;
        // pass 1: quantize activations to int8 (Q8, per 32-block) — one subgroup per (row, block)
        self.stamp("quant_q8");
        let kq = self
            .be
            .kernel_sg("quant_q8", crate::gemm::quant_q8_spv(), 4, 12, 32);
        let mut p1 = [0u8; 12];
        p1[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        p1[4..8].copy_from_slice(&(k as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&32u32.to_ne_bytes());
        self.dispatch(
            kq,
            &[
                Self::vkb(a),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
            ],
            3,
            &p1,
            (m * nblk) as u32,
        );
        // pass 2: integer dp4a matmul
        self.stamp("matmul_proj");
        let km = self
            .be
            .kernel_sg("gemm_proj_mmq", crate::gemm::gemm_proj_mmq_spv(), 7, 12, 32);
        let mut p2 = [0u8; 12];
        p2[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        self.dispatch3(
            km,
            &[
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(wq),
                Self::vkb(scales),
                Self::vkb(mins),
                Self::vkb(c),
            ],
            1,
            &p2,
            (n / 64) as u32,
            m.div_ceil(64) as u32,
            1,
        );
    }

    /// Quantized dequant GEMV `y = x·Wᵀ`. `bits` (4|8) and `blk_shift` (log2 scale-block) select
    /// the packed-weight layout.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_q(
        &self,
        quants: &dyn Buffer,
        scales: &dyn Buffer,
        mins: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
        bits: u32,
        blk_shift: u32,
    ) {
        self.stamp("lm_head");
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&bits.to_ne_bytes());
        push[16..20].copy_from_slice(&blk_shift.to_ne_bytes());
        let bufs = [
            Self::vkb(quants),
            Self::vkb(scales),
            Self::vkb(mins),
            Self::vkb(x),
            Self::vkb(y),
        ];
        if let Some(epw) = mmv_epw(bits) {
            if in_f.is_multiple_of(epw) {
                // subgroup GEMV: NUM_ROWS outputs/workgroup, one wave32 each (no shared reduction)
                let name = if bits == 4 {
                    "mul_mat_vec_q4"
                } else {
                    "mul_mat_vec_q8"
                };
                let k =
                    self.be
                        .kernel_sg(name, crate::gemm::mul_mat_vec_q_spv(bits, false), 5, 20, 32);
                let groups = (out_f as u32).div_ceil(MMV_NUM_ROWS);
                self.dispatch(k, &bufs, 1, &push, rows as u32 * groups);
                return;
            }
        }
        let k = self
            .be
            .kernel("linear_q", crate::gemm::linear_q_spv(), 5, 20);
        self.dispatch(k, &bufs, 1, &push, (rows * out_f) as u32);
    }

    /// Native-block dequant GEMV `y = x·Wᵀ`. Raw GGUF block bytes in `w` (padded
    /// to u32); format identified by `dtype`. Dispatch `rows*out_f` workgroups.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.linear_native_off(dtype, w, 0, x, y, rows, in_f, out_f);
    }

    /// Native-block dequant GEMV reading the weight from element offset `w_base` — lets one stacked
    /// MoE expert tensor serve all experts (`w_base = expert_id * out_f * in_f`).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_off(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        w_base: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.stamp("lm_head");
        let name = crate::linear::native_kernel_name(dtype, false);
        let spv = crate::gemm::native_build_spv(dtype, false).expect("native GEMV spv");
        let k = self.be.kernel(name, spv, 3, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(w), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// Native-block dequant GEMV with fused residual add: `y = residual + x·Wᵀ`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add_native(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        x: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.stamp("o_or_down");
        let name = crate::linear::native_kernel_name(dtype, true);
        let spv = crate::gemm::native_build_spv(dtype, true).expect("native GEMV spv");
        let k = self.be.kernel(name, spv, 4, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        // push[12..16] = w_base, 0 (residual native GEMV is not used for stacked experts).
        self.dispatch(
            k,
            &[
                Self::vkb(w),
                Self::vkb(x),
                Self::vkb(residual),
                Self::vkb(y),
            ],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// Quantized dequant GEMV with fused residual add: `y = residual + x·Wᵀ`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add_q(
        &self,
        quants: &dyn Buffer,
        scales: &dyn Buffer,
        mins: &dyn Buffer,
        x: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
        bits: u32,
        blk_shift: u32,
    ) {
        self.stamp("o_or_down");
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&bits.to_ne_bytes());
        push[16..20].copy_from_slice(&blk_shift.to_ne_bytes());
        let bufs = [
            Self::vkb(quants),
            Self::vkb(scales),
            Self::vkb(mins),
            Self::vkb(x),
            Self::vkb(residual),
            Self::vkb(y),
        ];
        if let Some(epw) = mmv_epw(bits) {
            if in_f.is_multiple_of(epw) {
                let name = if bits == 4 {
                    "mul_mat_vec_q4_res"
                } else {
                    "mul_mat_vec_q8_res"
                };
                let k =
                    self.be
                        .kernel_sg(name, crate::gemm::mul_mat_vec_q_spv(bits, true), 6, 20, 32);
                let groups = (out_f as u32).div_ceil(MMV_NUM_ROWS);
                self.dispatch(k, &bufs, 1, &push, rows as u32 * groups);
                return;
            }
        }
        let k = self
            .be
            .kernel("linear_res_q", crate::gemm::linear_res_q_spv(), 6, 20);
        self.dispatch(k, &bufs, 1, &push, (rows * out_f) as u32);
    }

    /// `y = residual + x·Wᵀ` (fused residual add). `residual` and `y` may be the same buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.stamp("o_or_down");
        let k = self
            .be
            .kernel("linear_res", crate::gemm::linear_res_spv(), 4, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(w),
                Self::vkb(x),
                Self::vkb(residual),
                Self::vkb(y),
            ],
            1,
            &push,
            (rows * out_f) as u32, // one workgroup per output element (coalesced GEMV)
        );
    }

    /// Fused attention input: `(q, k, v) = RoPE/identity(RMSNorm(hidden)·{Wq,Wk,Wv})`.
    /// `q`/`k` are RoPE'd, `v` raw. Requires `q_dim%64==0`, `kv_dim%64==0`, `hd` even, `ne<=8192`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_in(
        &self,
        hidden: &dyn Buffer,
        norm_w: &dyn Buffer,
        wq: &dyn Buffer,
        wk: &dyn Buffer,
        wv: &dyn Buffer,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        rows: usize,
        ne: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        pos: usize,
        eps: f32,
    ) {
        self.stamp("attn_in");
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        debug_assert_eq!(hd % 2, 0, "attn_in requires even hd (RoPE pairs)");
        let kern = self.be.kernel("attn_in", crate::gemm::attn_in_spv(), 8, 36);
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(q_dim as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(kv_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&theta.to_ne_bytes());
        push[28..32].copy_from_slice(&(pos as u32).to_ne_bytes());
        push[32..36].copy_from_slice(&eps.to_ne_bytes());
        let half = (q_dim + 2 * kv_dim) / 2;
        self.dispatch(
            kern,
            &[
                Self::vkb(hidden),
                Self::vkb(norm_w),
                Self::vkb(wq),
                Self::vkb(wk),
                Self::vkb(wv),
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(v),
            ],
            3,
            &push,
            (rows * half) as u32, // one workgroup per output pair (cooperative-over-K, coalesced)
        );
    }

    /// Record-once decode variant of `attn_in`: pos comes from the `params` SSBO (bound before the
    /// q/k/v outputs) so the buffer can be replayed across tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_in_dyn(
        &self,
        hidden: &dyn Buffer,
        norm_w: &dyn Buffer,
        wq: &dyn Buffer,
        wk: &dyn Buffer,
        wv: &dyn Buffer,
        params: &dyn Buffer,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        rows: usize,
        ne: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        eps: f32,
    ) {
        self.stamp("attn_in");
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let kern = self
            .be
            .kernel("attn_in_dyn", crate::gemm::attn_in_dyn_spv(), 9, 36);
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(q_dim as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(kv_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&theta.to_ne_bytes());
        // [28..32] pos: unused (from params)
        push[32..36].copy_from_slice(&eps.to_ne_bytes());
        let half = (q_dim + 2 * kv_dim) / 2;
        self.dispatch(
            kern,
            &[
                Self::vkb(hidden),
                Self::vkb(norm_w),
                Self::vkb(wq),
                Self::vkb(wk),
                Self::vkb(wv),
                Self::vkb(params),
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(v),
            ],
            3,
            &push,
            (rows * half) as u32,
        );
    }

    /// Quantized variant of `ffn_in` (Wgu = u8 quants + per-16-block f16 scale/min).
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_in_q(
        &self,
        hidden: &dyn Buffer,
        norm_w: &dyn Buffer,
        quants: &dyn Buffer,
        scales: &dyn Buffer,
        mins: &dyn Buffer,
        act: &dyn Buffer,
        rows: usize,
        ne: usize,
        nff: usize,
        eps: f32,
        bits: u32,
        blk_shift: u32,
    ) {
        self.stamp("ffn_in");
        let k = self
            .be
            .kernel("ffn_in_q", crate::gemm::ffn_in_q_spv(), 6, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nff as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&eps.to_ne_bytes());
        push[16..20].copy_from_slice(&bits.to_ne_bytes());
        push[20..24].copy_from_slice(&blk_shift.to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(hidden),
                Self::vkb(norm_w),
                Self::vkb(quants),
                Self::vkb(scales),
                Self::vkb(mins),
                Self::vkb(act),
            ],
            1,
            &push,
            (rows * nff) as u32,
        );
    }

    /// Fused RMSNorm + quant Q/K/V projection (Qwen3 decode): writes raw `qr`/`kr`/`vr`. Replaces
    /// rmsnorm + 3× `linear_q`. q/k/v carry their OWN `(bits, blk_shift)` (Q4_K_M mixes Q4_K/Q6_K).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_in_q(
        &self,
        hidden: &dyn Buffer,
        norm_w: &dyn Buffer,
        wq: (&dyn Buffer, &dyn Buffer, &dyn Buffer),
        wk: (&dyn Buffer, &dyn Buffer, &dyn Buffer),
        wv: (&dyn Buffer, &dyn Buffer, &dyn Buffer),
        qr: &dyn Buffer,
        kr: &dyn Buffer,
        vr: &dyn Buffer,
        rows: usize,
        ne: usize,
        q_dim: usize,
        kvrow: usize,
        eps: f32,
        qbb: (u32, u32),
        kbb: (u32, u32),
        vbb: (u32, u32),
    ) {
        self.stamp("attn_in_q");
        let k = self
            .be
            .kernel("attn_in_q", crate::gemm::attn_in_q_spv(), 14, 44);
        let mut push = [0u8; 44];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(q_dim as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(kvrow as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&eps.to_ne_bytes());
        push[20..24].copy_from_slice(&qbb.0.to_ne_bytes());
        push[24..28].copy_from_slice(&qbb.1.to_ne_bytes());
        push[28..32].copy_from_slice(&kbb.0.to_ne_bytes());
        push[32..36].copy_from_slice(&kbb.1.to_ne_bytes());
        push[36..40].copy_from_slice(&vbb.0.to_ne_bytes());
        push[40..44].copy_from_slice(&vbb.1.to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(hidden),
                Self::vkb(norm_w),
                Self::vkb(wq.0),
                Self::vkb(wq.1),
                Self::vkb(wq.2),
                Self::vkb(wk.0),
                Self::vkb(wk.1),
                Self::vkb(wk.2),
                Self::vkb(wv.0),
                Self::vkb(wv.1),
                Self::vkb(wv.2),
                Self::vkb(qr),
                Self::vkb(kr),
                Self::vkb(vr),
            ],
            3,
            &push,
            (rows * (q_dim + 2 * kvrow)) as u32,
        );
    }

    /// Fused FFN input: `act = SwiGLU(rmsnorm(hidden)·Wgu)`. Requires `nff % 64 == 0`, `ne <= 8192`.
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_in(
        &self,
        hidden: &dyn Buffer,
        norm_w: &dyn Buffer,
        wgu: &dyn Buffer,
        act: &dyn Buffer,
        rows: usize,
        ne: usize,
        nff: usize,
        eps: f32,
    ) {
        self.stamp("ffn_in");
        let k = self.be.kernel("ffn_in", crate::gemm::ffn_in_spv(), 4, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nff as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&eps.to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(hidden),
                Self::vkb(norm_w),
                Self::vkb(wgu),
                Self::vkb(act),
            ],
            1,
            &push,
            (rows * nff) as u32, // one workgroup per output (cooperative-over-K, coalesced)
        );
    }

    pub fn rmsnorm(
        &self,
        x: &dyn Buffer,
        w: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        self.stamp("rmsnorm");
        // 256-thread subgroup kernel (requiredSubgroupSize=32): more load/store parallelism and a
        // single barrier vs the 64-thread WGSL shared-tree. ~2.6× faster as a kernel; end-to-end
        // neutral here (decode is dispatch-latency-bound) but a win on slower/higher-latency GPUs.
        let k = self
            .be
            .kernel_sg("rmsnorm", crate::gemm::rmsnorm_spv(), 3, 12, 32);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(w), Self::vkb(y)],
            1,
            &push,
            rows as u32, // one workgroup per row (cooperative reduction)
        );
    }

    /// RoPE in place is allowed (`x` and `y` may be the same buffer). `pos_offset` shifts the
    /// absolute position of the first row (for cached decode).
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &dyn Buffer,
        y: &dyn Buffer,
        t: usize,
        n_heads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        pos_offset: usize,
    ) {
        self.stamp("rope");
        let k = self.be.kernel("rope", crate::gemm::rope_spv(), 2, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (t * n_heads) as u32,
        );
    }

    /// Cached attention: q `[q_len,nh,hd]` (abs positions `pos_offset..`) over the K/V cache
    /// `[kv_len,nkv,hd]`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        // Sliding-window attention: a query at abs pos `p` attends only keys `j > p - window`.
        // `0` = full causal (llama/qwen3 + gemma full-attention layers).
        window: usize,
        // QK scale: `> 0` overrides the default 1/√hd (gemma4 passes 1.0). `0.0` = default.
        scale: f32,
    ) {
        self.stamp("attention_kv");
        let kern = self
            .be
            .kernel("attention_kv", crate::gemm::attention_kv_spv(), 4, 32);
        let mut push = [0u8; 32];
        push[0..4].copy_from_slice(&(q_len as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&scale.to_ne_bytes());
        self.dispatch(
            kern,
            &[Self::vkb(q), Self::vkb(kc), Self::vkb(vc), Self::vkb(o)],
            1,
            &push,
            (q_len * nh) as u32,
        );
    }

    /// Non-FA prefill attention: clean coopmat QK → row softmax → coopmat PV (ollama's approach).
    /// `q`=[mpad,nh,hd] f16, `kc`/`vc`=[kv_len,nkv,hd] f16, `attn`=[mpad,nh*hd] f32 out, `s`=
    /// [nh,mpad,kv_pad] f16 scratch (mpad=ceil(n/64)*64, kv_pad=ceil(kv_len/64)*64). `pos_offset` is
    /// the absolute position of query row 0 (for causal masking).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_nonfa(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        attn: &dyn Buffer,
        s: &dyn Buffer,
        pv_part: &dyn Buffer,
        n: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        window: usize,
        scale: f32,
    ) {
        let mpad = (n.div_ceil(64) * 64) as u32;
        // kv padded to 256 (the 8-warp attn_qk's BN); extra cols are masked in softmax. Still %64 so
        // the 4-warp fallback + softmax + attn_pv are unaffected.
        let kv_pad = (kv_len.div_ceil(256) * 256) as u32;
        let hdu = hd as u32;
        // scale override: >0 uses the caller's value (gemma4 = 1.0, QK-norm controls magnitude);
        // 0 keeps the default 1/√hd (qwen3, gemma3).
        let scale = if scale > 0.0 {
            scale
        } else {
            1.0f32 / (hd as f32).sqrt()
        };

        // stage 1: S = scale·Q·Kᵀ. 8-warp/256-thread warptile (BN=256, matches ollama's mul_mm)
        // unless INFR_NO_QK_WARP forces the 4-warp/2×2 attn_qk.
        self.stamp("attn_qk");
        let qk_warp = std::env::var("INFR_NO_QK_WARP").is_err();
        let (qk_name, qk_spv, qk_bn) = if qk_warp {
            ("attn_qk_warp", crate::gemm::attn_qk_warp_spv(), 256u32)
        } else {
            ("attn_qk", crate::gemm::attn_qk_spv(), 64u32)
        };
        let kqk = self.be.kernel_sg(qk_name, qk_spv, 3, 24, 32);
        let mut p = [0u8; 24];
        p[0..4].copy_from_slice(&mpad.to_ne_bytes());
        p[4..8].copy_from_slice(&kv_pad.to_ne_bytes());
        p[8..12].copy_from_slice(&hdu.to_ne_bytes());
        p[12..16].copy_from_slice(&(nh as u32).to_ne_bytes());
        p[16..20].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p[20..24].copy_from_slice(&scale.to_ne_bytes());
        let qk_tiles = (mpad / 64) * (kv_pad / qk_bn);
        self.dispatch3(
            kqk,
            &[Self::vkb(q), Self::vkb(kc), Self::vkb(s)],
            1,
            &p,
            qk_tiles,
            1,
            nh as u32,
        );

        // stage 2: row softmax (causal + optional sliding window), in place S → P
        self.stamp("attn_softmax");
        let ksm = self
            .be
            .kernel("attn_softmax", crate::gemm::attn_softmax_spv(), 1, 20);
        let mut ps = [0u8; 20];
        ps[0..4].copy_from_slice(&mpad.to_ne_bytes());
        ps[4..8].copy_from_slice(&kv_pad.to_ne_bytes());
        ps[8..12].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        ps[12..16].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        ps[16..20].copy_from_slice(&(window as u32).to_ne_bytes());
        self.dispatch3(ksm, &[Self::vkb(s)], 1, &ps, mpad, 1, nh as u32);

        // stage 3: O = P·V  (one coopmat GEMM per head). Split-K when under-occupied: at high ctx
        // mpad is small so the base workgroup count ((mpad/64)*(hd/64)*nh) is far below the GPU's
        // capacity while each grinds a huge kv reduction → split the kv dim across gl_WorkGroupID.y
        // into partials, then sum them.
        self.stamp("attn_pv");
        let pv_base_wg = (mpad / 64) * (hdu / 64) * nh as u32;
        let n_splits = match std::env::var("INFR_PV_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(v) => v,
            None => {
                if pv_base_wg >= 1024 || kv_pad < 4096 {
                    1u32
                } else {
                    (2048 / pv_base_wg.max(1))
                        .clamp(2, 8)
                        .min(kv_pad / 2048)
                        .max(1)
                }
            }
        };
        let ksplit = kv_pad.div_ceil(n_splits).div_ceil(32) * 32;
        // 8-warp/256-thread PV warptile (BN=128=hd, matches ollama's mul_mm) when hd%128; else the
        // 4-warp/2×2 attn_pv (also handles hd<128, e.g. hd=64). INFR_NO_PV_WARP forces the 4-warp.
        let pv_warp = hdu.is_multiple_of(128) && std::env::var("INFR_NO_PV_WARP").is_err();
        let (pv_name, pv_spv, pv_bn) = if pv_warp {
            ("attn_pv_warp", crate::gemm::attn_pv_warp_spv(), 128u32)
        } else {
            ("attn_pv", crate::gemm::attn_pv_spv(), 64u32)
        };
        let kpv = self.be.kernel_sg(pv_name, pv_spv, 3, 28, 32);
        let mut pp = [0u8; 28];
        pp[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pp[4..8].copy_from_slice(&kv_pad.to_ne_bytes());
        pp[8..12].copy_from_slice(&hdu.to_ne_bytes());
        pp[12..16].copy_from_slice(&(nh as u32).to_ne_bytes());
        pp[16..20].copy_from_slice(&(nkv as u32).to_ne_bytes());
        pp[20..24].copy_from_slice(&n_splits.to_ne_bytes());
        pp[24..28].copy_from_slice(&ksplit.to_ne_bytes());
        let pv_tiles = (mpad / 64) * (hdu / pv_bn);
        let pv_out = if n_splits == 1 { attn } else { pv_part };
        self.dispatch3(
            kpv,
            &[Self::vkb(s), Self::vkb(vc), Self::vkb(pv_out)],
            1,
            &pp,
            pv_tiles,
            n_splits,
            nh as u32,
        );
        if n_splits > 1 {
            // sum the per-split partials → attn
            self.stamp("attn_pv");
            let total = mpad * nh as u32 * hdu;
            let kr = self
                .be
                .kernel("attn_pv_reduce", crate::gemm::attn_pv_reduce_spv(), 2, 8);
            let mut pr = [0u8; 8];
            pr[0..4].copy_from_slice(&total.to_ne_bytes());
            pr[4..8].copy_from_slice(&n_splits.to_ne_bytes());
            self.dispatch(
                kr,
                &[Self::vkb(pv_part), Self::vkb(attn)],
                1,
                &pr,
                total.div_ceil(256),
            );
        }
    }

    /// Flash-attention prefill: fused QK→softmax→PV, no materialized S buffer. `q`=[mpad,nh,hd] f16,
    /// `kc`/`vc`=[kv_len,nkv,hd] f16, `attn`=[mpad,nh*hd] f32 out. `pos_offset` = abs position of row
    /// 0 (causal). Split-K over kv for occupancy at high ctx (few q tiles): each (q-tile, head, split)
    /// emits an online-softmax partial into `po`/`pm`/`pl`, merged by attn_flash_combine. Scratch
    /// (caller, sized for ≤8 splits): `po` = 8·mpad·nh·hd f32, `pm`/`pl` = 8·mpad·nh f32. n_splits==1
    /// → single fused kernel writing `attn` directly (no scratch touched).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_flash(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        attn: &dyn Buffer,
        po: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        n: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
    ) {
        let mpad = (n.div_ceil(64) * 64) as u32;
        let base_wg = (mpad / 64) * nh as u32;
        let n_splits = match std::env::var("INFR_FLASH_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(v) => v,
            None => {
                if base_wg >= 1024 || kv_len < 4096 {
                    1u32
                } else {
                    (2048 / base_wg.max(1))
                        .clamp(2, 8)
                        .min((kv_len / 2048).max(1) as u32)
                }
            }
        };
        // hd=128 → register-blocked partial (always via partial+combine). Other hd → the
        // 4-subgroup path: single fused kernel for n_splits==1, else the scalar partial.
        //
        // The warp partial's shared scratch is bm*908 B (Ss+Ps+Os+mrow/lrow/corr, BN=64/HD=128).
        // Pick the largest tile the device's maxComputeSharedMemorySize allows: bm=64 → 58112 B (needs
        // 64 KB, e.g. RADV); bm=32 → 29056 B (fits NVIDIA 48 KB / MoltenVK 32 KB). The transformer
        // skips flash entirely when even bm=32 won't fit, so one of these always fits here.
        let shared_limit = self.be.max_shared_memory_bytes();
        let bm64_shared = 64 * crate::FLASH_SHARED_PER_ROW; // 58112 B
                                                            // INFR_FLASH_BM=32 forces the small (29056 B) tile even on a 64 KB device, so the bm=32
                                                            // shaders get numeric-parity coverage on any GPU (they otherwise only run on sub-64 KB ones).
        let force_bm32 = std::env::var("INFR_FLASH_BM").ok().as_deref() == Some("32");
        let bm: u32 = if !force_bm32 && shared_limit >= bm64_shared {
            64
        } else {
            32
        };
        // Each workgroup covers `bm` query rows → mpad/bm×nh groups (mpad is 64-aligned → /32 exact).
        let tile_wg = (mpad / bm) * nh as u32;
        // INFR_NO_FLASH_WARP routes to the non-warp partial. Both warp and non-warp paths have a
        // bm=32 build (29056 B) that fits sub-64 KB devices, so the knob is honored everywhere —
        // no longer forced back to warp on NVIDIA / MoltenVK.
        let warp = hd == 128 && std::env::var("INFR_NO_FLASH_WARP").is_err();
        if n_splits == 1 && !warp {
            self.stamp("attn_flash");
            let (fname, fspv): (&'static str, &[u32]) = if bm == 32 {
                ("attn_flash_bm32", crate::gemm::attn_flash_bm32_spv())
            } else {
                ("attn_flash", crate::gemm::attn_flash_spv())
            };
            let k = self.be.kernel_sg(fname, fspv, 4, 24, 32);
            let mut p = [0u8; 24];
            p[0..4].copy_from_slice(&mpad.to_ne_bytes());
            p[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
            p[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
            p[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
            p[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
            p[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
            self.dispatch(
                k,
                &[Self::vkb(q), Self::vkb(kc), Self::vkb(vc), Self::vkb(attn)],
                1,
                &p,
                tile_wg,
            );
            return;
        }
        // split-K partials
        self.stamp("attn_flash");
        let ksplit = (kv_len as u32).div_ceil(n_splits).div_ceil(64) * 64;
        // hd=128 → register-blocked warp partial; else the 4-subgroup partial. Each picks its
        // bm-sized shared build (warp/partial share the bm*908 B footprint) and covers tile_wg groups.
        let (pname, pspv): (&'static str, &[u32]) = match (warp, bm) {
            (true, 32) => (
                "attn_flash_warp_bm32",
                crate::gemm::attn_flash_warp_bm32_spv(),
            ),
            (true, _) => ("attn_flash_warp", crate::gemm::attn_flash_warp_spv()),
            (false, 32) => (
                "attn_flash_partial_bm32",
                crate::gemm::attn_flash_partial_bm32_spv(),
            ),
            (false, _) => ("attn_flash_partial", crate::gemm::attn_flash_partial_spv()),
        };
        let kp = self.be.kernel_sg(pname, pspv, 6, 32, 32);
        let mut pp = [0u8; 32];
        pp[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pp[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        pp[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        pp[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        pp[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        pp[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        pp[24..28].copy_from_slice(&n_splits.to_ne_bytes());
        pp[28..32].copy_from_slice(&ksplit.to_ne_bytes());
        self.dispatch3(
            kp,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(po),
                Self::vkb(pm),
                Self::vkb(pl),
            ],
            3,
            &pp,
            tile_wg,
            n_splits,
            1,
        );
        // combine → attn
        self.stamp("attn_flash");
        let kc2 = self.be.kernel_sg(
            "attn_flash_combine",
            crate::gemm::attn_flash_combine_spv(),
            4,
            16,
            32,
        );
        let mut pc2 = [0u8; 16];
        pc2[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pc2[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        pc2[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        pc2[12..16].copy_from_slice(&n_splits.to_ne_bytes());
        self.dispatch(
            kc2,
            &[Self::vkb(po), Self::vkb(pm), Self::vkb(pl), Self::vkb(attn)],
            1,
            &pc2,
            mpad * nh as u32,
        );
    }

    /// FlashAttention-2 register-O prefill (Br=128, per-thread register accumulator → no [Br][HD]
    /// shared O; 2× the query tile of the shared-Os flash → fewer q-tiles). hd MUST be 128 and the
    /// caller MUST allocate q/attn/po to mpad128 = ceil(n/128)*128 rows. Split-K → partials → combine.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_flash_reg(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        attn: &dyn Buffer,
        po: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        n: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
    ) {
        // mpad is 128-aligned → divisible by both BR tiles.
        let mpad = (n.div_ceil(128) * 128) as u32;
        // Register-O shared = BR*FLASH_REG_SHARED_PER_ROW: BR=128 → 58880 B (needs 64 KB); BR=64 →
        // 29440 B (NVIDIA 48 KB / MoltenVK 32 KB). Largest that fits; transformer skips reg if neither.
        let br: u32 = if self.be.max_shared_memory_bytes() >= 128 * crate::FLASH_REG_SHARED_PER_ROW
        {
            128
        } else {
            64
        };
        let base_wg = (mpad / br) * nh as u32;
        let n_splits = match std::env::var("INFR_FLASH_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(v) => v,
            None => {
                if base_wg >= 1024 || kv_len < 4096 {
                    1u32
                } else {
                    (2048 / base_wg.max(1))
                        .clamp(2, 8)
                        .min((kv_len / 2048).max(1) as u32)
                }
            }
        };
        let ksplit = (kv_len as u32).div_ceil(n_splits).div_ceil(64) * 64;
        self.stamp("attn_flash");
        let (rname, rspv): (&'static str, &[u32]) = if br == 64 {
            (
                "attn_flash_reg_br64",
                crate::gemm::attn_flash_reg_br64_spv(),
            )
        } else {
            ("attn_flash_reg", crate::gemm::attn_flash_reg_spv())
        };
        let kp = self.be.kernel_sg(rname, rspv, 6, 32, 32);
        let mut pp = [0u8; 32];
        pp[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pp[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        pp[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        pp[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        pp[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        pp[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        pp[24..28].copy_from_slice(&n_splits.to_ne_bytes());
        pp[28..32].copy_from_slice(&ksplit.to_ne_bytes());
        self.dispatch3(
            kp,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(po),
                Self::vkb(pm),
                Self::vkb(pl),
            ],
            3,
            &pp,
            base_wg,
            n_splits,
            1,
        );
        self.stamp("attn_flash");
        let kc2 = self.be.kernel_sg(
            "attn_flash_combine",
            crate::gemm::attn_flash_combine_spv(),
            4,
            16,
            32,
        );
        let mut pc2 = [0u8; 16];
        pc2[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pc2[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        pc2[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        pc2[12..16].copy_from_slice(&n_splits.to_ne_bytes());
        self.dispatch(
            kc2,
            &[Self::vkb(po), Self::vkb(pm), Self::vkb(pl), Self::vkb(attn)],
            1,
            &pc2,
            mpad * nh as u32,
        );
    }

    /// Cast-copy f32 `src[0..n]` → f16 `dst[off..off+n]` (write f32 activations into the f16 cache).
    pub fn store_f16(&self, src: &dyn Buffer, dst: &dyn Buffer, n: usize, off: usize) {
        self.stamp("store_f16");
        let k = self
            .be
            .kernel("store_f16", crate::gemm::store_f16_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Qwen3 QK-norm + RoPE over `x[rows, nheads, hd]` → `y` at rows `out_base..`. `nw` is the
    /// per-head [hd] norm weight. (q: out_base=0; k: out_base=pos so it lands in the cache.)
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope(
        &self,
        x: &dyn Buffer,
        nw: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        eps: f32,
        // gemma4 full-attention layers: per-pair RoPE frequency divisors (`Some`); `None` = normal
        // RoPE (qwen3 / gemma3 / gemma4 SWA layers).
        freq_factors: Option<&dyn Buffer>,
    ) {
        self.stamp("qk_norm_rope");
        let mut push = [0u8; 32];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(rope_pos as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(out_base as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        match freq_factors {
            Some(ff) => {
                let k =
                    self.be
                        .kernel("qk_norm_rope_ff", crate::gemm::qk_norm_rope_ff_spv(), 4, 32);
                self.dispatch(
                    k,
                    &[Self::vkb(x), Self::vkb(nw), Self::vkb(ff), Self::vkb(y)],
                    1,
                    &push,
                    (rows * nheads) as u32,
                );
            }
            None => {
                let k = self
                    .be
                    .kernel("qk_norm_rope", crate::gemm::qk_norm_rope_spv(), 3, 32);
                self.dispatch(
                    k,
                    &[Self::vkb(x), Self::vkb(nw), Self::vkb(y)],
                    1,
                    &push,
                    (rows * nheads) as u32,
                );
            }
        }
    }

    /// Flash-decoding attention (q_len==1): split the KV range into `n_chunks` chunks of `chunk`,
    /// compute per-chunk softmax partials in parallel (`pm`/`pl`/`pacc`), then combine into `o`.
    /// Parallelizes attention across `nh*n_chunks` workgroups so it stays fast at long context.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
    ) {
        // pass 1: per-chunk partials (subgroup-reduction QK; needs requiredSubgroupSize=32)
        self.stamp("attn_partial");
        let k1 = self
            .be
            .kernel_sg("attn_partial", crate::gemm::attn_partial_spv(), 6, 24, 32);
        let mut p1 = [0u8; 24];
        p1[0..4].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        p1[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        p1[16..20].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p1[20..24].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        self.dispatch(
            k1,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
            ],
            3,
            &p1,
            (nh * n_chunks) as u32,
        );
        // pass 2: combine — split each head's hd outputs across `ntile` workgroups for occupancy.
        self.stamp("attn_combine");
        let k2 = self
            .be
            .kernel("attn_combine", crate::gemm::attn_combine_spv(), 4, 16);
        let ntile = if hd.is_multiple_of(4) { 4u32 } else { 1u32 };
        let mut p2 = [0u8; 16];
        p2[0..4].copy_from_slice(&(nh as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(hd as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p2[12..16].copy_from_slice(&ntile.to_ne_bytes());
        self.dispatch(
            k2,
            &[Self::vkb(pm), Self::vkb(pl), Self::vkb(pacc), Self::vkb(o)],
            1,
            &p2,
            nh as u32 * ntile,
        );
    }

    // ---- Record-once decode variants (`_dyn`) ----
    // These read the per-token `pos`/`kv_len` from a host-updated `params` SSBO ([pos, kv_len] u32)
    // instead of push constants, so the decode command buffer can be recorded once and replayed every
    // token (only `params` + the embedding change). Used ONLY by the GPU-resident decode path; every
    // other caller keeps the push-constant kernels. `params` is inserted before the output(s) so the
    // recorder's reads|writes split stays output-last.

    /// QK-norm + RoPE, pos from `params`. `out_base_mul` = 0 for Q (write to a temp), 1 for K (write
    /// to the cache at row pos).
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_dyn(
        &self,
        x: &dyn Buffer,
        nw: &dyn Buffer,
        params: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        out_base_mul: usize,
        eps: f32,
    ) {
        self.stamp("qk_norm_rope");
        let k = self.be.kernel(
            "qk_norm_rope_dyn",
            crate::gemm::qk_norm_rope_dyn_spv(),
            4,
            32,
        );
        let mut push = [0u8; 32];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        // [20..24] rope_pos: unused (from params)
        push[24..28].copy_from_slice(&(out_base_mul as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(nw), Self::vkb(params), Self::vkb(y)],
            1,
            &push,
            (rows * nheads) as u32,
        );
    }

    /// Cast-copy f32 `src[0..n]` → f16 `dst[pos*n..]` (one KV row at position pos from `params`).
    pub fn store_f16_dyn(&self, src: &dyn Buffer, params: &dyn Buffer, dst: &dyn Buffer, n: usize) {
        self.stamp("store_f16");
        let k = self
            .be
            .kernel("store_f16_dyn", crate::gemm::store_f16_dyn_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        // [4..8] off: unused (computed as pos*n from params)
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(params), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Causal GQA over the KV cache (q_len==1), pos_offset from `params`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_dyn(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        params: &dyn Buffer,
        o: &dyn Buffer,
        q_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) {
        self.stamp("attention_kv");
        let kern = self.be.kernel(
            "attention_kv_dyn",
            crate::gemm::attention_kv_dyn_spv(),
            5,
            32,
        );
        let mut push = [0u8; 32];
        push[0..4].copy_from_slice(&(q_len as u32).to_ne_bytes());
        // [4..8] kv_len: unused
        push[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        // [20..24] pos_offset: unused (from params)
        // [24..28] window: 0 (record-once decode is gemma-disabled, so always full causal)
        // [28..32] scale: 0.0 → default 1/√hd (record-once decode is gemma-disabled)
        self.dispatch(
            kern,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(params),
                Self::vkb(o),
            ],
            1,
            &push,
            (q_len * nh) as u32,
        );
    }

    /// Split-K decode attention, kv_len from `params`. `chunk`/`n_chunks` stay push constants (they
    /// define the dispatch structure; the caller re-records when they change).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split_dyn(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        params: &dyn Buffer,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
    ) {
        self.stamp("attn_partial");
        let k1 = self.be.kernel_sg(
            "attn_partial_dyn",
            crate::gemm::attn_partial_dyn_spv(),
            7,
            24,
            32,
        );
        let mut p1 = [0u8; 24];
        // [0..4] kv_len: unused (from params)
        p1[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        p1[16..20].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p1[20..24].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        self.dispatch(
            k1,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(params),
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
            ],
            3,
            &p1,
            (nh * n_chunks) as u32,
        );
        // pass 2: combine (structure-only, unchanged from the push-constant path)
        self.stamp("attn_combine");
        let k2 = self
            .be
            .kernel("attn_combine", crate::gemm::attn_combine_spv(), 4, 16);
        let ntile = if hd.is_multiple_of(4) { 4u32 } else { 1u32 };
        let mut p2 = [0u8; 16];
        p2[0..4].copy_from_slice(&(nh as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(hd as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p2[12..16].copy_from_slice(&ntile.to_ne_bytes());
        self.dispatch(
            k2,
            &[Self::vkb(pm), Self::vkb(pl), Self::vkb(pacc), Self::vkb(o)],
            1,
            &p2,
            nh as u32 * ntile,
        );
    }

    /// Record a buffer→buffer copy of `bytes` from `src[src_offset..]` into `dst[dst_offset..]`.
    pub fn copy(
        &self,
        src: &dyn Buffer,
        src_offset: usize,
        dst: &dyn Buffer,
        dst_offset: usize,
        bytes: usize,
    ) {
        let device = &self.be.shared.device;
        self.sync(&[Self::vkb(src)], &[Self::vkb(dst)], true);
        self.dirty_transfer.set(true);
        unsafe {
            device.cmd_copy_buffer(
                self.cmd,
                Self::vkb(src),
                Self::vkb(dst),
                &[vk::BufferCopy {
                    src_offset: src_offset as u64,
                    dst_offset: dst_offset as u64,
                    size: bytes as u64,
                }],
            );
        }
    }

    pub fn attention(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        o: &dyn Buffer,
        t: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) {
        self.stamp("attention");
        let kern = self
            .be
            .kernel("attention", crate::gemm::attention_spv(), 4, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[Self::vkb(q), Self::vkb(k), Self::vkb(v), Self::vkb(o)],
            1,
            &push,
            (t * nh) as u32,
        );
    }

    /// Fused SwiGLU over a combined `gu` `[rows, 2*nff]` → `y` `[rows, nff]`.
    pub fn silu_mul_fused(&self, gu: &dyn Buffer, y: &dyn Buffer, rows: usize, nff: usize) {
        self.stamp("silu_mul");
        let k = self
            .be
            .kernel("silu_mul_fused", crate::gemm::silu_mul_fused_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nff as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(gu), Self::vkb(y)],
            1,
            &push,
            (rows * nff) as u32,
        );
    }

    /// Fused GeGLU (GELU tanh-approx gate) over a combined `gu` `[rows, 2*nff]` → `y` `[rows, nff]`.
    /// Same layout/dispatch as [`silu_mul_fused`]; gemma uses GELU instead of SiLU.
    pub fn gelu_mul_fused(&self, gu: &dyn Buffer, y: &dyn Buffer, rows: usize, nff: usize) {
        self.stamp("gelu_mul");
        let k = self
            .be
            .kernel("gelu_mul_fused", crate::gemm::gelu_mul_fused_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nff as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(gu), Self::vkb(y)],
            1,
            &push,
            (rows * nff) as u32,
        );
    }

    pub fn silu_mul(&self, gate: &dyn Buffer, up: &dyn Buffer, y: &dyn Buffer, n: usize) {
        self.stamp("silu_mul");
        let k = self
            .be
            .kernel("silu_mul", crate::gemm::silu_mul_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(gate), Self::vkb(up), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Gated-DeltaNet recurrence, one token (Qwen3-Next SSM). The persistent `state` buffer
    /// `[nv*kd*vd]` is updated in place; `out` is `[nv*vd]`. One workgroup per value head; the
    /// `nk` q/k heads are tiled up to `nv`. See shaders/deltanet.comp.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        blog: &dyn Buffer,
        alpha: &dyn Buffer,
        acoef: &dyn Buffer,
        dtbias: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) {
        self.stamp("deltanet");
        // The shader caches each column block's state [kd, 32] in shared memory (`ss[128*32]`), so kd
        // must be ≤ 128. Qwen3-Next uses kd=128; assert so a larger head_k_dim fails loudly instead of
        // corrupting LDS.
        debug_assert!(
            kd <= 128,
            "deltanet shared-state block assumes kd ≤ 128, got {kd}"
        );
        let kern = self
            .be
            .kernel("deltanet", crate::gemm::deltanet_spv(), 9, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nk as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(kd as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(vd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&eps.to_ne_bytes());
        push[24..28].copy_from_slice(&(1.0f32 / (kd as f32).sqrt()).to_ne_bytes());
        // One workgroup per (value head, block of 32 state columns); local_size_x=32.
        let n_blk = vd.div_ceil(32);
        self.dispatch(
            kern,
            &[
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(v),
                Self::vkb(blog),
                Self::vkb(alpha),
                Self::vkb(acoef),
                Self::vkb(dtbias),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &push,
            (nv * n_blk) as u32,
        );
    }

    /// CHUNKED gated-DeltaNet prefill (chunkwise delta rule, C=32): per 32-token chunk the
    /// recurrence collapses to dense matmuls + one unit-lower-triangular solve, so the state is
    /// traversed rows/32 times instead of `rows`. Same signature/bindings as `deltanet`; math
    /// validated against the sequential form in tests/chunked_delta_math.rs. Use for rows ≥ 32
    /// (decode keeps the sequential kernel). See shaders/deltanet_chunked.comp.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_chunked(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        blog: &dyn Buffer,
        alpha: &dyn Buffer,
        acoef: &dyn Buffer,
        dtbias: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) {
        self.stamp("deltanet");
        debug_assert!(
            kd <= 128,
            "deltanet_chunked LDS chunk tiles assume kd ≤ 128, got {kd}"
        );
        let kern = self.be.kernel(
            "deltanet_chunked",
            crate::gemm::deltanet_chunked_spv(),
            9,
            28,
        );
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nk as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(kd as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(vd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&eps.to_ne_bytes());
        push[24..28].copy_from_slice(&(1.0f32 / (kd as f32).sqrt()).to_ne_bytes());
        // One workgroup per (value head, block of 32 state columns); local_size_x=256.
        // (COLS=16 was tried for occupancy and REGRESSED 2670→3668µs — the per-block duplicated
        // A/Wq dots dominate; that's what the split prep+scan variant hoists out.)
        let n_blk = vd.div_ceil(32);
        self.dispatch(
            kern,
            &[
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(v),
                Self::vkb(blog),
                Self::vkb(alpha),
                Self::vkb(acoef),
                Self::vkb(dtbias),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &push,
            (nv * n_blk) as u32,
        );
    }

    /// Chunked gated-DeltaNet prefill, SPLIT variant (prep + gates + scan): the chunk-parallel
    /// work (q/k normalization, intra-chunk D=K̂K̂ᵀ / Dq=Q̂K̂ᵀ dot matrices, gates) is hoisted into
    /// two fully-parallel passes, so the sequential scan pass does ONLY state-coupled work — which
    /// parallelizes cleanly over small column blocks (COLS=16 → nv·(vd/16) workgroups). The
    /// monolithic `deltanet_chunked` duplicated that shared work per column block (~37% of it).
    /// Scratch (caller-alloc'd, alloc_uninit-safe — every read slot is written by prep/gates):
    /// kn/qn [rows·nk·kd] f32, dk/dq [nchunk·nk·C·C] f32, betag/gg [nchunk·nv·C] f32, C=32.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_chunked_split(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        blog: &dyn Buffer,
        alpha: &dyn Buffer,
        acoef: &dyn Buffer,
        dtbias: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        kn: &dyn Buffer,
        qn: &dyn Buffer,
        dk: &dyn Buffer,
        dq: &dyn Buffer,
        betag: &dyn Buffer,
        gg: &dyn Buffer,
        rows: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) {
        debug_assert!(kd <= 128, "deltanet split assumes kd ≤ 128, got {kd}");
        let nchunk = rows.div_ceil(32);
        self.stamp("deltanet");
        // pass 1: prep — (chunk, k-head) grid
        let kp = self
            .be
            .kernel("deltanet_prep", crate::gemm::deltanet_prep_spv(), 6, 20);
        let mut p1 = [0u8; 20];
        p1[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p1[4..8].copy_from_slice(&(nk as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(kd as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&eps.to_ne_bytes());
        p1[16..20].copy_from_slice(&(1.0f32 / (kd as f32).sqrt()).to_ne_bytes());
        self.dispatch(
            kp,
            &[
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(kn),
                Self::vkb(qn),
                Self::vkb(dk),
                Self::vkb(dq),
            ],
            4, // kn, qn, dk, dq
            &p1,
            (nchunk * nk) as u32,
        );
        // pass 2: gates — (chunk, value-head) grid
        self.stamp("deltanet");
        let kg = self
            .be
            .kernel("deltanet_gates", crate::gemm::deltanet_gates_spv(), 6, 8);
        let mut p2 = [0u8; 8];
        p2[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        self.dispatch(
            kg,
            &[
                Self::vkb(blog),
                Self::vkb(alpha),
                Self::vkb(acoef),
                Self::vkb(dtbias),
                Self::vkb(betag),
                Self::vkb(gg),
            ],
            2, // betag, gg
            &p2,
            (nchunk * nv) as u32,
        );
        // pass 3: scan — (value head, column block) grid, sequential over chunks inside
        self.stamp("deltanet");
        let ks = self
            .be
            .kernel_sg("deltanet_scan", crate::gemm::deltanet_scan_spv(), 9, 20, 32);
        let mut p3 = [0u8; 20];
        p3[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p3[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        p3[8..12].copy_from_slice(&(nk as u32).to_ne_bytes());
        p3[12..16].copy_from_slice(&(kd as u32).to_ne_bytes());
        p3[16..20].copy_from_slice(&(vd as u32).to_ne_bytes());
        let n_blk = vd.div_ceil(8); // COLS=8, keep in sync with deltanet_scan.comp
        self.dispatch(
            ks,
            &[
                Self::vkb(v),
                Self::vkb(kn),
                Self::vkb(qn),
                Self::vkb(dk),
                Self::vkb(dq),
                Self::vkb(betag),
                Self::vkb(gg),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &p3,
            (nv * n_blk) as u32,
        );
    }

    /// Causal depthwise conv1d + SiLU, one token (Qwen3-Next SSM input conv). The per-channel history
    /// `state` `[(kconv-1)*cc]` is updated in place; `out` is `[cc]`. See shaders/conv1d_silu.comp.
    pub fn conv1d_silu(
        &self,
        qkv: &dyn Buffer,
        w: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        cc: usize,
        kconv: usize,
    ) {
        self.stamp("conv1d_silu");
        let kern = self
            .be
            .kernel("conv1d_silu", crate::gemm::conv1d_silu_spv(), 4, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(cc as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(kconv as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[
                Self::vkb(qkv),
                Self::vkb(w),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &push,
            (cc as u32).div_ceil(256),
        );
    }

    /// BATCH depthwise conv1d + SiLU (rows ≥ kconv-1): pass 1 computes ALL rows·cc outputs in
    /// parallel from the virtual sequence [state ‖ qkv] (the sequential kernel walked the rows
    /// one by one, shuffling the history each token); pass 2 rebuilds the history from the last
    /// kconv-1 input rows. The recorder's hazard tracking orders pass 2 after pass 1 (pass 1
    /// reads the old state pass 2 overwrites). See shaders/conv1d_silu_par.comp / conv1d_shift.comp.
    pub fn conv1d_silu_batch(
        &self,
        qkv: &dyn Buffer,
        w: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        cc: usize,
        kconv: usize,
    ) {
        debug_assert!(rows >= kconv - 1, "conv1d_silu_batch needs rows ≥ kconv-1");
        self.stamp("conv1d_silu");
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(cc as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(kconv as u32).to_ne_bytes());
        let k1 = self
            .be
            .kernel("conv1d_silu_par", crate::gemm::conv1d_silu_par_spv(), 4, 12);
        self.dispatch(
            k1,
            &[
                Self::vkb(qkv),
                Self::vkb(w),
                Self::vkb(state),
                Self::vkb(out),
            ],
            1, // out only (state is read-only here)
            &push,
            ((rows * cc) as u32).div_ceil(256),
        );
        self.stamp("conv1d_shift");
        let k2 = self
            .be
            .kernel("conv1d_shift", crate::gemm::conv1d_shift_spv(), 2, 12);
        self.dispatch(
            k2,
            &[Self::vkb(qkv), Self::vkb(state)],
            1, // state out
            &push,
            (((kconv - 1) * cc) as u32).div_ceil(256),
        );
    }

    /// Batched strided row copy in ONE dispatch (word granularity): `rows` slices of `nw` u32
    /// words, `dst[dst_off + r*dst_stride + ..nw] = src[src_off + r*src_stride + ..nw]` (all in
    /// words). Replaces the per-row copy-command loop for Op::CopyStrided — at rows=512 that was
    /// 512 vkCmdCopyBuffer + hazard checks per split op, dwarfing the bytes moved.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_strided(
        &self,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        nw: usize,
        src_off: usize,
        src_stride: usize,
        dst_off: usize,
        dst_stride: usize,
    ) {
        self.stamp("copy");
        let kern = self
            .be
            .kernel("copy_strided", crate::gemm::copy_strided_spv(), 2, 24);
        let mut push = [0u8; 24];
        for (i, v) in [rows, nw, src_off, src_stride, dst_off, dst_stride]
            .iter()
            .enumerate()
        {
            push[i * 4..i * 4 + 4].copy_from_slice(&(*v as u32).to_ne_bytes());
        }
        self.dispatch(
            kern,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            ((rows * nw) as u32).div_ceil(256),
        );
    }

    /// Elementwise sigmoid gate: `y[i] = a[i] * sigmoid(b[i])` (Qwen3-Next attention output gate).
    pub fn mul_sigmoid(&self, a: &dyn Buffer, b: &dyn Buffer, y: &dyn Buffer, n: usize) {
        self.stamp("mul_sigmoid");
        let kern = self
            .be
            .kernel("mul_sigmoid", crate::gemm::mul_sigmoid_spv(), 3, 4);
        let mut push = [0u8; 4];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(b), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// GeGLU with separate gate/up buffers: `y[i] = gelu(gate[i]) * up[up_off_bytes/4 + i]` (GELU
    /// tanh-approx). `up_off_bytes` lets a layer-major slice of a larger buffer be read in place
    /// (gemma4 per-layer-embd gate: `gelu(inp_gate·hidden) * inp_per_layer[il]`).
    pub fn gelu_mul_off(
        &self,
        gate: &dyn Buffer,
        up: &dyn Buffer,
        up_off_bytes: usize,
        y: &dyn Buffer,
        n: usize,
    ) {
        let k = self
            .be
            .kernel("gelu_mul", crate::gemm::gelu_mul_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&((up_off_bytes / 4) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(gate), Self::vkb(up), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Whether the id-indexed native GEMV (GPU-resident MoE routing) supports this expert format.
    pub fn native_id_supported(dtype: infr_core::DType) -> bool {
        crate::linear::native_id_kernel_name(dtype).is_some()
    }

    /// GPU MoE router top-k for `n_tokens` tokens (one workgroup per token): softmax-renormalized
    /// top-`n_used` over each token's `logits[n_expert]` → selected expert `ids` + `wts` (per token,
    /// `n_used` each), all in VRAM (no host round-trip). `scale` = routing scale.
    pub fn moe_topk(
        &self,
        logits: &dyn Buffer,
        ids: &dyn Buffer,
        wts: &dyn Buffer,
        n_tokens: usize,
        n_expert: usize,
        n_used: usize,
        scale: f32,
    ) {
        let k = self
            .be
            .kernel("moe_topk", crate::gemm::moe_topk_spv(), 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(n_expert as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&scale.to_ne_bytes());
        // ids is read-modify-write (exclusion scan); bind it as an output alongside wts.
        self.dispatch(
            k,
            &[Self::vkb(logits), Self::vkb(ids), Self::vkb(wts)],
            2,
            &push,
            n_tokens as u32,
        );
    }

    /// Greedy argmax over `n` logits → token id (u32) in `out_id[0]`. One workgroup; lets greedy
    /// decode read back a 4-byte token instead of the whole vocab logits.
    pub fn argmax(&self, logits: &dyn Buffer, out_id: &dyn Buffer, n: usize) {
        let k = self.be.kernel("argmax", crate::gemm::argmax_spv(), 2, 4);
        self.dispatch(
            k,
            &[Self::vkb(logits), Self::vkb(out_id)],
            1,
            &(n as u32).to_ne_bytes(),
            1,
        );
    }

    /// Largest `top_k` the GPU stochastic sampler handles; above this the caller samples on the host.
    pub const SAMPLE_KMAX: usize = crate::gemm::SAMPLE_KMAX;

    /// GPU stochastic sampling over `n` logits → token id in `out_id[0]`: temperature + top-k +
    /// top-p (nucleus) via a radix N-ary select, inverse-CDF sampled with the host-drawn uniform `u`.
    /// Requires `2 ≤ top_k ≤ SAMPLE_KMAX`. Only the token reads back — the vocab logits stay in VRAM.
    pub fn sample(
        &self,
        logits: &dyn Buffer,
        out_id: &dyn Buffer,
        n: usize,
        top_k: usize,
        temp: f32,
        top_p: f32,
        u: f32,
    ) {
        let k = self
            .be
            .kernel("moe_sample", crate::gemm::moe_sample_spv(), 2, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(top_k as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&temp.to_ne_bytes());
        push[12..16].copy_from_slice(&top_p.to_ne_bytes());
        push[16..20].copy_from_slice(&u.to_ne_bytes());
        self.dispatch(k, &[Self::vkb(logits), Self::vkb(out_id)], 1, &push, 1);
    }

    /// Zero a buffer's first `n` 4-byte elements (cmd_fill_buffer) — clears the bucket counters.
    pub fn zero(&self, buf: &dyn Buffer, n: usize) {
        self.sync(&[], &[Self::vkb(buf)], true);
        unsafe {
            self.be
                .shared
                .device
                .cmd_fill_buffer(self.cmd, Self::vkb(buf), 0, (n * 4) as u64, 0);
        }
    }

    /// MoE bucketing pass 1 (count): tally assignments per expert into `counts` (pre-zeroed).
    pub fn moe_bucket_count(&self, tok_ids: &dyn Buffer, counts: &dyn Buffer, n_pairs: usize) {
        let k = self.be.kernel(
            "moe_bucket_count",
            crate::gemm::moe_bucket_count_spv(),
            2,
            4,
        );
        self.dispatch(
            k,
            &[Self::vkb(tok_ids), Self::vkb(counts)],
            1,
            &(n_pairs as u32).to_ne_bytes(),
            (n_pairs as u32).div_ceil(64),
        );
    }

    /// MoE bucketing pass 2 (scan): exclusive prefix sum `counts → offsets`, and reset `fill` to 0.
    pub fn moe_bucket_scan(
        &self,
        counts: &dyn Buffer,
        offsets: &dyn Buffer,
        fill: &dyn Buffer,
        n_expert: usize,
    ) {
        let k = self
            .be
            .kernel("moe_bucket_scan", crate::gemm::moe_bucket_scan_spv(), 3, 4);
        self.dispatch(
            k,
            &[Self::vkb(counts), Self::vkb(offsets), Self::vkb(fill)],
            2,
            &(n_expert as u32).to_ne_bytes(),
            1,
        );
    }

    /// MoE bucketing pass 3 (scatter): group token rows + weights by expert into `bucket_rows` /
    /// `bucket_wts` (each expert's run starts at `offsets[e]`).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_bucket_scatter(
        &self,
        tok_ids: &dyn Buffer,
        tok_wts: &dyn Buffer,
        offsets: &dyn Buffer,
        fill: &dyn Buffer,
        bucket_rows: &dyn Buffer,
        bucket_wts: &dyn Buffer,
        n_pairs: usize,
        n_used: usize,
    ) {
        let k = self.be.kernel(
            "moe_bucket_scatter",
            crate::gemm::moe_bucket_scatter_spv(),
            6,
            8,
        );
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n_pairs as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_used as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(tok_ids),
                Self::vkb(tok_wts),
                Self::vkb(offsets),
                Self::vkb(fill),
                Self::vkb(bucket_rows),
                Self::vkb(bucket_wts),
            ],
            3,
            &push,
            (n_pairs as u32).div_ceil(64),
        );
    }

    /// Id-indexed native-block GEMV `y = x · W[ids[slot]]ᵀ` from a stacked expert tensor (element
    /// stride per expert). Lets GPU-resident MoE decode pick the expert from a GPU buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_id(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        ids: &dyn Buffer,
        slot: usize,
        stride: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.stamp("lm_head");
        let name = crate::linear::native_id_kernel_name(dtype).expect("native id kernel");
        let spv = crate::gemm::native_id_build_spv(dtype).expect("native id spv");
        let k = self.be.kernel(name, spv, 4, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(slot as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(stride as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(w), Self::vkb(x), Self::vkb(ids), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// Multi-slot id GEMV: all `n_used` experts in ONE dispatch → `y` is [n_used, out_f]. The experts
    /// run concurrently (no inter-expert barrier). `x_per_slot`: false → all slots read the same row
    /// `x` (gate/up); true → slot reads `x[slot*in_f..]` (down). Decode FFN fusion.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_id_multi(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        ids: &dyn Buffer,
        n_used: usize,
        stride: usize,
        x: &dyn Buffer,
        x_per_slot: bool,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        self.stamp("expert_ffn");
        let name = crate::linear::native_idm_kernel_name(dtype).expect("native idm kernel");
        let spv = crate::gemm::native_idm_build_spv(dtype).expect("native idm spv");
        let k = self.be.kernel(name, spv, 4, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(stride as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(x_per_slot as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(w), Self::vkb(x), Self::vkb(ids), Self::vkb(y)],
            1,
            &push,
            (n_used * out_f) as u32,
        );
    }

    /// Quantize f32 activations `a` [m,k] → int8 `qa` [m,k] + per-32-block f16 `dact`/`sact`
    /// ([m, k/32]) for the dp4a mmq matmul. (Pass 1 of mmq, reusable standalone.)
    pub fn quant_q8(
        &self,
        a: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        m: usize,
        k: usize,
    ) {
        self.stamp("quant_q8");
        let kq = self
            .be
            .kernel_sg("quant_q8", crate::gemm::quant_q8_spv(), 4, 12, 32);
        let mut p = [0u8; 12];
        p[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        p[4..8].copy_from_slice(&(k as u32).to_ne_bytes());
        p[8..12].copy_from_slice(&32u32.to_ne_bytes());
        self.dispatch(
            kq,
            &[
                Self::vkb(a),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
            ],
            3,
            &p,
            (m * (k / 32)) as u32,
        );
    }

    /// Multi-slot id-indexed Q4_K dp4a (mmq) GEMV: like `linear_native_id_multi` but using hardware
    /// int8 dot-product against pre-quantized activations (`qa`/`dact`/`sact` from `quant_q8`, shared
    /// across slots). Q4_K weights only. `y` is [n_used, out_f].
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv_id_multi_q4k(
        &self,
        w: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        ids: &dyn Buffer,
        n_used: usize,
        stride: usize,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        self.stamp("mmq_expert");
        let k = self.be.kernel(
            "native_mmv_id_q4k",
            crate::gemm::native_mmv_id_q4k_spv(),
            6,
            16,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(stride as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(w),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(ids),
                Self::vkb(y),
            ],
            1,
            &push,
            (n_used * out_f) as u32,
        );
    }

    /// Weighted accumulate of all selected experts' down outputs into hidden:
    /// `hidden[i] += Σ_slot wts[slot] * down[slot*ne + i]`. Folds the per-expert axpys into one op.
    pub fn moe_accumulate(
        &self,
        down: &dyn Buffer,
        wts: &dyn Buffer,
        hidden: &dyn Buffer,
        ne: usize,
        n_used: usize,
    ) {
        let k = self
            .be
            .kernel("moe_accumulate", crate::gemm::moe_accumulate_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_used as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(down), Self::vkb(wts), Self::vkb(hidden)],
            1,
            &push,
            (ne as u32).div_ceil(64),
        );
    }

    /// `acc += wts[slot] * x` (indexed axpy) — the scale is read from a GPU buffer (the on-GPU router
    /// weights), so the weighted MoE expert accumulate needs no host scale.
    pub fn add_scaled_id(
        &self,
        x: &dyn Buffer,
        wts: &dyn Buffer,
        slot: usize,
        acc: &dyn Buffer,
        n: usize,
    ) {
        let k = self
            .be
            .kernel("add_scaled_id", crate::gemm::add_scaled_id_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(slot as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(wts), Self::vkb(acc)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// `acc += scale * x` (axpy), in place into `acc`. Accumulates weighted MoE expert outputs into
    /// the resident hidden state on the GPU (chained across experts via WAW barriers on `acc`).
    /// In-place elementwise scalar multiply: `y[i] *= scale` for `i < n` (gemma4 layer output scale).
    pub fn scale(&self, y: &dyn Buffer, scale: f32, n: usize) {
        self.stamp("scale");
        let k = self.be.kernel("scale", crate::gemm::scale_spv(), 1, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&scale.to_ne_bytes());
        self.dispatch(k, &[Self::vkb(y)], 1, &push, (n as u32).div_ceil(64));
    }

    /// Elementwise softcap `y[i] = cap·tanh(x[i]/cap)` (gemma final-logit / attn softcap). In-place
    /// safe — bind `x == y`.
    pub fn softcap(&self, x: &dyn Buffer, y: &dyn Buffer, cap: f32, n: usize) {
        self.stamp("softcap");
        let k = self.be.kernel("softcap", crate::gemm::softcap_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&cap.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    pub fn add_scaled(&self, x: &dyn Buffer, acc: &dyn Buffer, scale: f32, n: usize) {
        self.stamp("add");
        let k = self
            .be
            .kernel("add_scaled", crate::gemm::add_scaled_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&scale.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(acc)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Gather rows: `dst[j,:] = src[idx[j],:]` for j in 0..m, each row `ne` wide. Assembles an MoE
    /// expert's token batch from the resident activations.
    pub fn gather_rows(
        &self,
        src: &dyn Buffer,
        idx: &dyn Buffer,
        idx_base: usize,
        dst: &dyn Buffer,
        m: usize,
        ne: usize,
    ) {
        let k = self
            .be
            .kernel("gather_rows", crate::gemm::gather_rows_spv(), 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(idx_base as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(idx), Self::vkb(dst)],
            1,
            &push,
            ((m * ne) as u32).div_ceil(64),
        );
    }

    /// Weighted scatter-add: `dst[idx[j],:] += w[j] * y[j,:]` for j in 0..m, each row `ne` wide.
    /// Accumulates an MoE expert's weighted token outputs back into the resident hidden state
    /// (chained across experts via WAW barriers on `dst`).
    #[allow(clippy::too_many_arguments)]
    pub fn scatter_add_rows(
        &self,
        y: &dyn Buffer,
        idx: &dyn Buffer,
        w: &dyn Buffer,
        base: usize,
        dst: &dyn Buffer,
        m: usize,
        ne: usize,
    ) {
        let k = self.be.kernel(
            "scatter_add_rows",
            crate::gemm::scatter_add_rows_spv(),
            4,
            12,
        );
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(base as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(y), Self::vkb(idx), Self::vkb(w), Self::vkb(dst)],
            1,
            &push,
            ((m * ne) as u32).div_ceil(64),
        );
    }

    /// Elementwise add; in place allowed (`a` may equal `y`).
    pub fn add(&self, a: &dyn Buffer, b: &dyn Buffer, y: &dyn Buffer, n: usize) {
        self.stamp("add");
        let k = self.be.kernel("add", crate::gemm::add_spv(), 3, 4);
        self.dispatch(
            k,
            &[Self::vkb(a), Self::vkb(b), Self::vkb(y)],
            1,
            &(n as u32).to_ne_bytes(),
            (n as u32).div_ceil(64),
        );
    }

    /// End recording, submit once, wait, and release transient objects.
    pub fn finish(self) -> Result<()> {
        let device = &self.be.shared.device;
        if self.prof {
            eprintln!("[prof] barriers emitted = {}", self.barriers.borrow());
        }
        // Final timestamp so the last op has an interval to close.
        if self.prof2 {
            let idx = self.ts_labels.borrow().len() as u32;
            unsafe {
                device.cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    idx,
                );
            }
        }
        unsafe { device.end_command_buffer(self.cmd) }.map_err(|e| be(format!("end cmd: {e}")))?;
        let queue = self.be.shared.queue;
        unsafe {
            device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd))],
                    vk::Fence::null(),
                )
                .map_err(|e| be(format!("queue_submit: {e}")))?;
            device
                .queue_wait_idle(queue)
                .map_err(|e| be(format!("queue_wait_idle: {e}")))?;
            if self.prof2 {
                self.report_timestamps();
                device.destroy_query_pool(self.query_pool, None);
            }
            let cmd_pool = *self.be.shared.cmd_pool.lock().unwrap();
            device.free_command_buffers(cmd_pool, &[self.cmd]);
            device.destroy_descriptor_pool(self.pool, None);
        }
        Ok(())
    }

    /// End recording WITHOUT submitting, returning a [`RecordedCmd`] the caller can replay across
    /// tokens (skipping per-token re-recording). Only meaningful for a `new_persistent` recorder; the
    /// descriptor sets bind the (persistent) decode buffers, so replays stay valid as long as those
    /// buffers live.
    pub fn finish_record(self) -> Result<RecordedCmd> {
        if self.prof {
            eprintln!("[prof] barriers emitted = {}", self.barriers.borrow());
        }
        unsafe { self.be.shared.device.end_command_buffer(self.cmd) }
            .map_err(|e| be(format!("end cmd: {e}")))?;
        Ok(RecordedCmd {
            shared: std::sync::Arc::clone(&self.be.shared),
            cmd: self.cmd,
            pool: self.pool,
        })
    }

    /// Read back per-op timestamps and print GPU time aggregated by op label.
    fn report_timestamps(&self) {
        let labels = self.ts_labels.borrow();
        let n = labels.len();
        if n == 0 {
            return;
        }
        let mut ticks = vec![0u64; n + 1];
        unsafe {
            self.be
                .shared
                .device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .expect("get_query_pool_results");
        }
        let period = unsafe {
            self.be
                .shared
                .instance
                .get_physical_device_properties(self.be.shared.physical_device)
        }
        .limits
        .timestamp_period; // ns per tick
        use std::collections::BTreeMap;
        let mut by: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
        let mut total = 0f64;
        for i in 0..n {
            let us = (ticks[i + 1].wrapping_sub(ticks[i]) as f64) * period as f64 / 1000.0;
            let e = by.entry(labels[i]).or_insert((0.0, 0));
            e.0 += us;
            e.1 += 1;
            total += us;
        }
        eprintln!("[prof2] per-op GPU time (total {total:.0}us across {n} ops):");
        let mut rows: Vec<_> = by.into_iter().collect();
        rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        for (lbl, (us, cnt)) in rows {
            eprintln!(
                "[prof2]   {lbl:>14}  {us:8.0}us  ({cnt:3} ops, {:.1}us/op, {:.0}%)",
                us / cnt as f64,
                100.0 * us / total
            );
        }
    }
}

impl VulkanBackend {
    /// Start recording a single-submit forward.
    pub fn recorder(&self) -> Result<Recorder<'_>> {
        Recorder::new(self)
    }

    /// Start recording a resubmittable forward — `finish_record` returns a [`RecordedCmd`] to replay.
    pub fn recorder_persistent(&self) -> Result<Recorder<'_>> {
        Recorder::new_persistent(self)
    }
}

/// A pre-recorded, resubmittable command buffer (from [`Recorder::finish_record`]). Replaying it skips
/// per-token re-recording in the GPU-resident decode loop. Owns its command buffer + descriptor pool
/// (whose sets bind the persistent decode buffers); both are freed on drop after the GPU drains.
pub struct RecordedCmd {
    shared: std::sync::Arc<crate::VulkanShared>,
    cmd: vk::CommandBuffer,
    pool: vk::DescriptorPool,
}

impl RecordedCmd {
    /// Resubmit the recorded command buffer and wait for completion.
    pub fn replay(&self) -> Result<()> {
        let device = &self.shared.device;
        unsafe {
            device
                .queue_submit(
                    self.shared.queue,
                    &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd))],
                    vk::Fence::null(),
                )
                .map_err(|e| be(format!("replay submit: {e}")))?;
            device
                .queue_wait_idle(self.shared.queue)
                .map_err(|e| be(format!("replay wait: {e}")))?;
        }
        Ok(())
    }
}

impl Drop for RecordedCmd {
    fn drop(&mut self) {
        let device = &self.shared.device;
        unsafe {
            let _ = device.queue_wait_idle(self.shared.queue);
            let cmd_pool = *self.shared.cmd_pool.lock().unwrap();
            device.free_command_buffers(cmd_pool, &[self.cmd]);
            device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::{backend::BufferUsage, Backend};

    fn rmsnorm_cpu(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let scale = 1.0 / (ss + eps).sqrt();
        (0..n).map(|i| x[i] * scale * w[i]).collect()
    }
    fn dot(w: &[f32], row: usize, ne: usize, x: &[f32]) -> f32 {
        (0..ne).map(|k| w[row * ne + k] * x[k]).sum()
    }
    // ggml NORM-interleaved rope of a head-dim vector in place over the first rope_dim entries.
    fn rope_head(v: &mut [f32], hd: usize, rope_dim: usize, theta: f32, pos: usize) {
        for i in 0..rope_dim / 2 {
            let freq = theta.powf(-2.0 * i as f32 / rope_dim as f32);
            let ang = pos as f32 * freq;
            let (s, co) = (ang.sin(), ang.cos());
            let a = v[2 * i];
            let b = v[2 * i + 1];
            v[2 * i] = a * co - b * s;
            v[2 * i + 1] = a * s + b * co;
        }
        let _ = hd;
    }

    // round f32 → f16 → f32 (matches what the f16 q/k/v buffers store)
    fn r16(v: &[f32]) -> Vec<f32> {
        v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
    }
    // upload f32 values as an f16 buffer
    fn upf16(be: &VulkanBackend, v: &[f32]) -> Box<dyn Buffer> {
        let bits: Vec<u16> = v
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let b = be.alloc(bits.len() * 2, BufferUsage::Staging).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(&bits)).unwrap();
        b
    }

    // Reference attention. `window`>0 = sliding-window lower bound (gemma SWA); `scale_in`>0 overrides
    // the default 1/√hd (gemma4 = 1.0). Matches attention_kv.comp and attn_softmax.comp semantics.
    #[allow(clippy::too_many_arguments)]
    fn attn_kv_cpu(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        window: usize,
        scale_in: f32,
    ) -> Vec<f32> {
        let scale = if scale_in > 0.0 {
            scale_in
        } else {
            1.0 / (hd as f32).sqrt()
        };
        let mut o = vec![0f32; q_len * nh * hd];
        for ti in 0..q_len {
            let abs = pos_offset + ti;
            let lo = if window > 0 && abs + 1 > window {
                abs + 1 - window
            } else {
                0
            };
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                let qb = (ti * nh + h) * hd;
                let mut sc = vec![0f32; abs + 1 - lo]; // valid keys [lo, abs]
                let mut mx = f32::NEG_INFINITY;
                for (jj, scj) in sc.iter_mut().enumerate() {
                    let kb = ((lo + jj) * nkv + kvh) * hd;
                    let d: f32 = (0..hd).map(|x| q[qb + x] * k[kb + x]).sum();
                    *scj = d * scale;
                    mx = mx.max(*scj);
                }
                let mut l = 0f32;
                for s in &sc {
                    l += (s - mx).exp();
                }
                let ob = (ti * nh + h) * hd;
                for (jj, s) in sc.iter().enumerate() {
                    let p = (s - mx).exp() / l;
                    let vb = ((lo + jj) * nkv + kvh) * hd;
                    for x in 0..hd {
                        o[ob + x] += p * v[vb + x];
                    }
                }
            }
        }
        o
    }

    fn run_attn_kv(q_len: usize, kv_len: usize, nh: usize, nkv: usize, hd: usize) {
        let be = VulkanBackend::new().unwrap();
        let pos_offset = kv_len - q_len; // new tokens are the last q_len of the cache
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        // q/k/v are f16 on the GPU; round the reference inputs to f16 too so the test isolates the
        // attention math (not f16 rounding).
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let bq = upf16(&be, &q);
        let bk = upf16(&be, &k);
        let bv = upf16(&be, &v);
        let bo = be
            .alloc(q_len * nh * hd * 4, BufferUsage::Readback)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_kv(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
            0,   // full causal (no sliding window)
            0.0, // default 1/√hd scale
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; q_len * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, 0, 0.0);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attention_kv q_len={q_len} kv_len={kv_len} max_err={err:e}");
        assert!(err < 5e-3, "attention_kv mismatch: {err}");
    }

    fn run_attn_kv_split(kv_len: usize, nh: usize, nkv: usize, hd: usize) {
        let be = VulkanBackend::new().unwrap();
        let chunk = 512usize;
        let n_chunks = kv_len.div_ceil(chunk);
        let pos_offset = kv_len - 1; // decode: one new token at the end
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let bq = upf16(&be, &q);
        let bk = upf16(&be, &k);
        let bv = upf16(&be, &v);
        let bo = be.alloc(nh * hd * 4, BufferUsage::Readback).unwrap();
        let pm = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pl = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pacc = be
            .alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_kv_split(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            kv_len,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, 1, nh, nkv, hd, pos_offset, 0, 0.0);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_kv_split kv_len={kv_len} n_chunks={n_chunks} max_err={err:e}");
        assert!(err < 5e-3, "split mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_kv_split_matches_cpu() {
        run_attn_kv_split(600, 9, 3, 64); // 2 chunks
        run_attn_kv_split(2050, 9, 3, 64); // 5 chunks, partial last
        run_attn_kv_split(8000, 4, 2, 32); // 16 chunks
        run_attn_kv_split(830, 16, 2, 256); // qwen35 full-attn decode (hd=256 general path)
        run_attn_kv_split(2050, 16, 8, 256); // gemma SWA-shape decode (hd=256, GQA 16:8)
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_kv_decode_matches_cpu() {
        // decode: 1 new token over a cache; 2500 exercises multi-tile flash (>TILE=1024)
        run_attn_kv(1, 200, 9, 3, 64);
        run_attn_kv(1, 13, 4, 2, 32);
        run_attn_kv(1, 1, 4, 2, 32);
        run_attn_kv(1, 2500, 9, 3, 64);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_kv_prefill_matches_cpu() {
        run_attn_kv(17, 17, 9, 3, 64);
        run_attn_kv(40, 1500, 9, 3, 64); // multi-tile prefill (kv_len > TILE)
                                         // gemma4: SWA layers (hd=256, GQA 16:8) and full layers (hd=512, GQA 16:1).
        run_attn_kv(17, 17, 16, 8, 256);
        run_attn_kv(17, 17, 16, 1, 512);
        run_attn_kv(1, 200, 16, 1, 512); // gemma4 full-layer decode
    }

    // upload f32 values as an f32 buffer (qk_norm_rope reads x / nw / freq_factors as f32).
    fn upf32(be: &VulkanBackend, v: &[f32]) -> Box<dyn Buffer> {
        let b = be.alloc(v.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
        b
    }

    /// CPU reference for fused per-head QK-norm (RMSNorm over hd) + NEOX RoPE (optional freq_factors).
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope_cpu(
        x: &[f32],
        nw: &[f32],
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        eps: f32,
        ff: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut y = vec![0f32; (out_base + rows) * nheads * hd];
        let hf = rope_dim / 2;
        for r in 0..rows {
            for h in 0..nheads {
                let ib = (r * nheads + h) * hd;
                let ss: f32 = (0..hd).map(|i| x[ib + i] * x[ib + i]).sum::<f32>() / hd as f32;
                let scale = 1.0 / (ss + eps).sqrt();
                let ob = ((out_base + r) * nheads + h) * hd;
                for p in 0..hf {
                    let (i0, i1) = (p, p + hf);
                    let a = x[ib + i0] * scale * nw[i0];
                    let b = x[ib + i1] * scale * nw[i1];
                    let freq = theta.powf(-2.0 * p as f32 / rope_dim as f32);
                    let mut ang = (rope_pos + r) as f32 * freq;
                    if let Some(ff) = ff {
                        ang /= ff[p];
                    }
                    let (s, c) = (ang.sin(), ang.cos());
                    y[ob + i0] = a * c - b * s;
                    y[ob + i1] = a * s + b * c;
                }
            }
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn run_qk_norm_rope(
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        with_ff: bool,
    ) {
        let be = VulkanBackend::new().unwrap();
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let x = gen(rows * nheads * hd, 1);
        let nw: Vec<f32> = gen(hd, 2).iter().map(|v| v + 1.05).collect(); // ~gemma norm weights near 1
                                                                          // proportional-rope freq_factors: first quarter rotate (1.0), rest unrotated (1e30) — like gemma4
        let ff: Vec<f32> = (0..rope_dim / 2)
            .map(|p| if p < rope_dim / 4 { 1.0 } else { 1e30 })
            .collect();
        let bx = upf32(&be, &x);
        let bnw = upf32(&be, &nw);
        let bff = upf32(&be, &ff);
        let y_len = (out_base + rows) * nheads * hd;
        let by = be.alloc(y_len * 2, BufferUsage::Readback).unwrap(); // f16 out
        let rec = be.recorder().unwrap();
        rec.qk_norm_rope(
            bx.as_ref(),
            bnw.as_ref(),
            by.as_ref(),
            rows,
            nheads,
            hd,
            rope_dim,
            theta,
            rope_pos,
            out_base,
            1e-6,
            if with_ff { Some(bff.as_ref()) } else { None },
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; y_len * 2];
        be.download(by.as_ref(), &mut bytes).unwrap();
        let got: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let want = qk_norm_rope_cpu(
            &x,
            &nw,
            rows,
            nheads,
            hd,
            rope_dim,
            theta,
            rope_pos,
            out_base,
            1e-6,
            if with_ff { Some(&ff) } else { None },
        );
        // compare only the rows the kernel actually wrote (out_base..out_base+rows)
        let mut err = 0f32;
        for r in 0..rows {
            for h in 0..nheads {
                for i in 0..hd {
                    let idx = ((out_base + r) * nheads + h) * hd + i;
                    err = err.max((got[idx] - want[idx]).abs());
                }
            }
        }
        println!(
            "qk_norm_rope rows={rows} nheads={nheads} hd={hd} rope_dim={rope_dim} ff={with_ff} max_err={err:e}"
        );
        assert!(err < 1e-2, "qk_norm_rope mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn qk_norm_rope_matches_cpu() {
        run_qk_norm_rope(8, 4, 128, 128, 1e6, 0, 0, false); // qwen3 hd=128
        run_qk_norm_rope(8, 4, 256, 256, 1e4, 0, 0, false); // gemma3 hd=256
        run_qk_norm_rope(17, 16, 256, 256, 1e4, 5, 0, false); // gemma4 SWA Q (out_base=0)
        run_qk_norm_rope(17, 8, 256, 256, 1e4, 5, 5, false); // gemma4 SWA K (out_base=pos)
        run_qk_norm_rope(17, 16, 512, 512, 1e6, 5, 0, false); // gemma4 full Q
        run_qk_norm_rope(17, 1, 512, 512, 1e6, 5, 5, false); // gemma4 full K
        run_qk_norm_rope(17, 16, 512, 512, 1e6, 5, 0, true); // gemma4 full Q + freq_factors
        run_qk_norm_rope(17, 1, 512, 512, 1e6, 5, 5, true); // gemma4 full K + freq_factors
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_prefill_nonfa_matches_cpu() {
        // hd=128 (qwen3), full causal, default scale — the pre-existing regression cases.
        run_attn_prefill_nonfa(64, 64, 2, 1, 128, 0, 0.0);
        run_attn_prefill_nonfa(128, 200, 4, 2, 128, 0, 0.0);
        run_attn_prefill_nonfa(70, 70, 2, 2, 128, 0, 0.0);
        run_attn_prefill_nonfa(192, 500, 2, 1, 128, 0, 0.0);
        run_attn_prefill_nonfa(80, 300, 9, 3, 64, 0, 0.0);
        // gemma: hd=256 (SWA, GQA 16:8) and hd=512 (full, GQA 16:1), with the sliding-window mask
        // and the scale override (gemma4 = 1.0). These are the paths the new routing exercises.
        run_attn_prefill_nonfa(128, 300, 16, 8, 256, 100, 1.0); // gemma4 SWA (window)
        run_attn_prefill_nonfa(128, 300, 16, 1, 512, 0, 1.0); // gemma4 full (scale=1)
        run_attn_prefill_nonfa(128, 300, 16, 8, 256, 100, 0.0); // gemma3 SWA (window, 1/√hd)
        run_attn_prefill_nonfa(70, 400, 16, 8, 256, 64, 1.0); // SWA, non-64-aligned q
                                                              // force the split-K PV path (n_splits>1) and verify the partial-sum reduce is correct, incl.
                                                              // with a window (split reduce must respect the softmax mask) and hd=512.
        std::env::set_var("INFR_PV_SPLITS", "4");
        run_attn_prefill_nonfa(70, 300, 4, 2, 128, 0, 0.0);
        run_attn_prefill_nonfa(128, 500, 2, 1, 128, 0, 0.0);
        run_attn_prefill_nonfa(128, 3000, 16, 8, 256, 200, 1.0); // SWA, long kv, split-K
        run_attn_prefill_nonfa(128, 3000, 16, 1, 512, 0, 1.0); // full hd=512, long kv, split-K
        std::env::remove_var("INFR_PV_SPLITS");
    }

    fn run_attn_prefill_nonfa(
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        window: usize,
        scale: f32,
    ) {
        let be = VulkanBackend::new().unwrap();
        let pos_offset = kv_len - q_len;
        let mpad = q_len.div_ceil(64) * 64;
        let kv_pad = kv_len.div_ceil(256) * 256; // recorder pads kv to 256 (8-warp attn_qk BN)
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let mut qp = q.clone();
        qp.resize(mpad * nh * hd, 0.0);
        // K/V must cover the padded kv (the kernel reads padded rows; softmax masks them).
        let mut kp = k.clone();
        kp.resize((kv_pad + 64) * nkv * hd, 0.0);
        let mut vp = v.clone();
        vp.resize((kv_pad + 64) * nkv * hd, 0.0);
        let bq = upf16(&be, &qp);
        let bk = upf16(&be, &kp);
        let bv = upf16(&be, &vp);
        let bo = be.alloc(mpad * nh * hd * 4, BufferUsage::Readback).unwrap();
        let bs = be
            .alloc(nh * mpad * kv_pad * 2, BufferUsage::Activations)
            .unwrap();
        let bpv = be
            .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_prefill_nonfa(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            bs.as_ref(),
            bpv.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
            window,
            scale,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, window, scale);
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let err = got[..q_len * nh * hd]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_prefill_nonfa q_len={q_len} kv_len={kv_len} max_err={err:e}");
        assert!(err < 5e-3, "attn_prefill_nonfa mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_prefill_flash_matches_cpu() {
        for &(q, kv, nh, nkv, hd) in &[
            (64usize, 64usize, 2usize, 1usize, 128usize),
            (128, 200, 4, 2, 128),
            (70, 70, 2, 2, 128),
            (192, 500, 2, 1, 128),
            (80, 300, 9, 3, 64),
            (448, 2000, 16, 8, 128), // qwen3-shaped, multi-block kv
        ] {
            run_attn_prefill_flash(q, kv, nh, nkv, hd);
        }
        // force the split-K flash path (partial+combine) and verify the merge is correct
        std::env::set_var("INFR_FLASH_SPLITS", "4");
        run_attn_prefill_flash(64, 2000, 16, 8, 128);
        run_attn_prefill_flash(128, 500, 2, 1, 128);
        std::env::remove_var("INFR_FLASH_SPLITS");
        // Force the bm=32 tile (otherwise only selected on sub-64 KB-shared devices like NVIDIA /
        // MoltenVK) so the small shaders get numeric-parity coverage on any GPU: the fused kernel
        // (hd=64), the warp split-K partial+combine (hd=128), and the non-warp partial
        // (INFR_NO_FLASH_WARP). Without this, a 64 KB device only ever exercises the bm=64 build.
        std::env::set_var("INFR_FLASH_BM", "32");
        run_attn_prefill_flash(80, 300, 9, 3, 64); // fused attn_flash_bm32
        run_attn_prefill_flash(128, 200, 4, 2, 128); // warp partial+combine (bm32)
        run_attn_prefill_flash(448, 2000, 16, 8, 128); // warp, multi-block kv
        std::env::set_var("INFR_NO_FLASH_WARP", "1");
        run_attn_prefill_flash(128, 500, 2, 1, 128); // non-warp attn_flash_partial_bm32
        std::env::remove_var("INFR_NO_FLASH_WARP");
        std::env::remove_var("INFR_FLASH_BM");
    }

    fn run_attn_prefill_flash(q_len: usize, kv_len: usize, nh: usize, nkv: usize, hd: usize) {
        let be = VulkanBackend::new().unwrap();
        let pos_offset = kv_len - q_len;
        let mpad = q_len.div_ceil(64) * 64;
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let mut qp = q.clone();
        qp.resize(mpad * nh * hd, 0.0);
        let mut kp = k.clone();
        kp.resize((kv_len + 64) * nkv * hd, 0.0);
        let mut vp = v.clone();
        vp.resize((kv_len + 64) * nkv * hd, 0.0);
        let bq = upf16(&be, &qp);
        let bk = upf16(&be, &kp);
        let bv = upf16(&be, &vp);
        let bo = be.alloc(mpad * nh * hd * 4, BufferUsage::Readback).unwrap();
        let po = be
            .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let pmb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let plb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_prefill_flash(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            po.as_ref(),
            pmb.as_ref(),
            plb.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, 0, 0.0);
        let err = got[..q_len * nh * hd]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_prefill_flash q_len={q_len} kv_len={kv_len} max_err={err:e}");
        assert!(err < 5e-3, "attn_prefill_flash mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_prefill_flash_reg_matches_cpu() {
        for &(q, kv, nh, nkv) in &[
            (128usize, 128usize, 4usize, 2usize),
            (128, 300, 2, 1),
            (200, 600, 8, 4),
            (448, 2000, 16, 8), // qwen3-shaped
            (100, 100, 2, 2),   // q<128 → padded tile
        ] {
            run_attn_flash_reg(q, kv, nh, nkv, 128);
        }
        std::env::set_var("INFR_FLASH_SPLITS", "4");
        run_attn_flash_reg(128, 2000, 16, 8, 128);
        run_attn_flash_reg(200, 600, 2, 1, 128);
        std::env::remove_var("INFR_FLASH_SPLITS");
    }

    fn run_attn_flash_reg(q_len: usize, kv_len: usize, nh: usize, nkv: usize, hd: usize) {
        let be = VulkanBackend::new().unwrap();
        let pos_offset = kv_len - q_len;
        let mpad = q_len.div_ceil(128) * 128;
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let mut qp = q.clone();
        qp.resize(mpad * nh * hd, 0.0);
        let mut kp = k.clone();
        kp.resize((kv_len + 128) * nkv * hd, 0.0);
        let mut vp = v.clone();
        vp.resize((kv_len + 128) * nkv * hd, 0.0);
        let bq = upf16(&be, &qp);
        let bk = upf16(&be, &kp);
        let bv = upf16(&be, &vp);
        let bo = be.alloc(mpad * nh * hd * 4, BufferUsage::Readback).unwrap();
        let po = be
            .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let pmb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let plb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_prefill_flash_reg(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            po.as_ref(),
            pmb.as_ref(),
            plb.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, 0, 0.0);
        let err = got[..q_len * nh * hd]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_flash_reg q_len={q_len} kv_len={kv_len} nh={nh} max_err={err:e}");
        assert!(err < 5e-3, "attn_flash_reg mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn matmul_proj_mmq_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (70usize, 64usize, 128usize); // m not %64 → padding; k=2 blocks of 32; n%64
        let nblk = k / 32;
        let mpad = m.div_ceil(64) * 64;
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();
        // u4 weights: q in 0..15, per-32-block f16 scale/min
        let qv: Vec<u32> = (0..n * k).map(|i| (i * 7 % 16) as u32).collect();
        let scales: Vec<u16> = (0..n * k / 32)
            .map(|b| half::f16::from_f32(0.015 + (b % 5) as f32 * 0.002).to_bits())
            .collect();
        let mins: Vec<u16> = (0..n * k / 32)
            .map(|b| half::f16::from_f32(-0.12 + (b % 3) as f32 * 0.03).to_bits())
            .collect();
        // pack u4: 8 nibbles per u32, weight g = col*k + kk
        let mut packed = vec![0u32; n * k / 8];
        for (g, &q) in qv.iter().enumerate() {
            packed[g / 8] |= q << (4 * (g % 8) as u32);
        }
        let dq = |g: usize| {
            half::f16::from_bits(scales[g / 32]).to_f32() * qv[g] as f32
                + half::f16::from_bits(mins[g / 32]).to_f32()
        };

        let upf = |v: &[f32]| {
            let b = be.alloc(v.len() * 4, BufferUsage::Staging).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let ba = upf(&a);
        let bwq = be
            .upload_weight_bytes(bytemuck::cast_slice(&packed))
            .unwrap();
        let bs = be
            .upload_weight_bytes(bytemuck::cast_slice(&scales))
            .unwrap();
        let bmn = be.upload_weight_bytes(bytemuck::cast_slice(&mins)).unwrap();
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        // scratch
        let qa = be.alloc(mpad * k, BufferUsage::Activations).unwrap();
        let dact = be.alloc(mpad * nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(mpad * nblk * 2, BufferUsage::Activations).unwrap();

        let rec = be.recorder().unwrap();
        rec.matmul_proj_mmq(
            ba.as_ref(),
            bwq.as_ref(),
            bs.as_ref(),
            bmn.as_ref(),
            bc.as_ref(),
            qa.as_ref(),
            dact.as_ref(),
            sact.as_ref(),
            m,
            k,
            n,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * n * 4];
        be.download(bc.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let mut e = 0f32;
        for r in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|x| a[r * k + x] * dq(col * k + x)).sum();
                e = e.max((got[r * n + col] - want).abs());
            }
        }
        println!("matmul_proj_mmq max_err={e:e}");
        assert!(e < 2e-2, "matmul_proj_mmq mismatch: {e}"); // int8 activation quant tolerance
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn native_gemm_mmq_q6k_matches_cpu() {
        // Q6_K dp4a GEMM vs a CPU reference: build a synthetic Q6_K weight (q6 ∈ 0..63, i8 sub-scale
        // per 16, f16 super-scale per 256), pack it into the 210-byte block layout, and check the
        // int8-dot GEMM matches the dequantized matmul. k must be a multiple of 256 (a Q6_K superblock)
        // so each column's k packs into whole blocks — exactly the real (down-proj) layout.
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (70usize, 256usize, 128usize); // m not %64 → padding; k = 1 superblock; n%64
        let nblk = k / 32;
        let mpad = m.div_ceil(64) * 64;
        let nsb = k / 256; // superblocks per column
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();
        // q6 quants, per-16 i8 scales, per-256 f16 super-scale (weight g = col*k + kk)
        let q6: Vec<u32> = (0..n * k).map(|i| (i * 13 % 64) as u32).collect();
        let sc: Vec<i8> = (0..n * k / 16)
            .map(|b| ((b % 11) as i32 - 5) as i8)
            .collect();
        let d: Vec<half::f16> = (0..n * nsb)
            .map(|b| half::f16::from_f32(0.008 + (b % 5) as f32 * 0.001))
            .collect();
        // pack into 210-byte Q6_K blocks: [ql[128]][qh[64]][i8 scales[16]][f16 d]
        let mut blk = vec![0u8; n * nsb * 210];
        let sc_index = |g: usize| -> usize {
            // global i8-scale index for weight g, mirroring the shader's sc_idx within the superblock
            let p = g % 256;
            let (hf, ph) = (p / 128, p % 128);
            let (og, l) = (ph / 32, ph % 32);
            (g / 256) * 16 + hf * 8 + l / 16 + 2 * og
        };
        for (col, _) in (0..n).map(|c| (c, ())) {
            for sbk in 0..nsb {
                let base = (col * nsb + sbk) * 210;
                for p in 0..256usize {
                    let g = col * k + sbk * 256 + p;
                    let q = q6[g];
                    let (hf, ph) = (p / 128, p % 128);
                    let (og, l) = (ph / 32, ph % 32);
                    let lo = hf * 64;
                    let qh_byte = base + 128 + hf * 32 + l;
                    match og {
                        0 => blk[base + lo + l] |= (q & 0xF) as u8,
                        1 => blk[base + lo + l + 32] |= (q & 0xF) as u8,
                        2 => blk[base + lo + l] |= ((q & 0xF) << 4) as u8,
                        _ => blk[base + lo + l + 32] |= ((q & 0xF) << 4) as u8,
                    }
                    blk[qh_byte] |= (((q >> 4) & 3) << (2 * og)) as u8;
                }
                // i8 scales at +192, f16 d at +208
                for s in 0..16 {
                    blk[base + 192 + s] = sc[(col * nsb + sbk) * 16 + s] as u8;
                }
                let db = d[col * nsb + sbk].to_bits().to_le_bytes();
                blk[base + 208] = db[0];
                blk[base + 209] = db[1];
            }
        }
        let dq = |g: usize| -> f32 {
            d[g / 256].to_f32() * sc[sc_index(g)] as f32 * (q6[g] as f32 - 32.0)
        };

        let upf = |v: &[f32]| {
            let b = be.alloc(v.len() * 4, BufferUsage::Staging).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let ba = upf(&a);
        let bw = be.upload_weight_bytes(&blk).unwrap();
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        let qa = be.alloc(mpad * k, BufferUsage::Activations).unwrap();
        let dact = be.alloc(mpad * nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(mpad * nblk * 2, BufferUsage::Activations).unwrap();

        let rec = be.recorder().unwrap();
        rec.quant_q8(ba.as_ref(), qa.as_ref(), dact.as_ref(), sact.as_ref(), m, k);
        rec.matmul_mmq_q6k(
            qa.as_ref(),
            dact.as_ref(),
            bw.as_ref(),
            0,
            bc.as_ref(),
            m,
            k,
            n,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * n * 4];
        be.download(bc.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let mut e = 0f32;
        for r in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|x| a[r * k + x] * dq(col * k + x)).sum();
                e = e.max((got[r * n + col] - want).abs());
            }
        }
        println!("native_gemm_mmq_q6k max_err={e:e}");
        assert!(e < 2e-2, "native_gemm_mmq_q6k mismatch: {e}"); // int8 activation quant tolerance
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn matmul_proj_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (800usize, 64usize, 256usize); // m≥768 & not %64 → warp path + padding; n%256
        let mpad = m.div_ceil(64) * 64;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        // weight W[N,K] (row-major [out,in]), f16-rounded
        let w: Vec<f32> = (0..n * k)
            .map(|i| half::f16::from_f32(((i * 13 % 23) as f32 - 11.0) * 0.02).to_f32())
            .collect();
        let upf = |v: &[f32]| {
            let b = be.alloc(v.len() * 4, BufferUsage::Staging).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let ba = upf(&a);
        let dummy = be.alloc(4, BufferUsage::Activations).unwrap();
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        let cpu = |label: &str, c: &[f32]| {
            let mut e = 0f32;
            for r in 0..m {
                for col in 0..n {
                    let want: f32 = (0..k).map(|x| a[r * k + x] * w[col * k + x]).sum();
                    e = e.max((c[r * n + col] - want).abs());
                }
            }
            println!("matmul_proj {label} max_err={e:e}");
            assert!(e < 5e-3, "matmul_proj {label} mismatch: {e}");
        };

        // --- f16 weights (bits=16) ---
        let wf16: Vec<u16> = w
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let bw = be.upload_weight_bytes(bytemuck::cast_slice(&wf16)).unwrap();
        let rec = be.recorder().unwrap();
        rec.matmul_proj(
            ba.as_ref(),
            bw.as_ref(),
            dummy.as_ref(),
            dummy.as_ref(),
            bc.as_ref(),
            m,
            k,
            n,
            16,
            0,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * n * 4];
        be.download(bc.as_ref(), &mut bytes).unwrap();
        cpu("f16", bytemuck::cast_slice(&bytes));

        // --- quant weights (bits=8, per-16 scale/min) ---
        let blk = 16usize;
        let mut qu = vec![0u32; n * k / 4];
        let scales: Vec<u16> = (0..n * k / blk)
            .map(|_b| half::f16::from_f32(0.02).to_bits())
            .collect();
        let mins: Vec<u16> = (0..n * k / blk)
            .map(|_b| half::f16::from_f32(-1.5).to_bits())
            .collect();
        // choose u8 so that scale*u8+min == f16-rounded w (approx): u8 = round((w-min)/scale)
        let mut wq_ref = vec![0f32; n * k];
        for g in 0..n * k {
            let s = half::f16::from_bits(scales[g / blk]).to_f32();
            let mn = half::f16::from_bits(mins[g / blk]).to_f32();
            let q = (((w[g] - mn) / s).round().clamp(0.0, 255.0)) as u8;
            qu[g / 4] |= (q as u32) << (8 * (g % 4));
            wq_ref[g] = s * q as f32 + mn;
        }
        let bwq = be.upload_weight_bytes(bytemuck::cast_slice(&qu)).unwrap();
        let bs = be
            .upload_weight_bytes(bytemuck::cast_slice(&scales))
            .unwrap();
        let bm = be.upload_weight_bytes(bytemuck::cast_slice(&mins)).unwrap();
        let bc2 = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.matmul_proj(
            ba.as_ref(),
            bwq.as_ref(),
            bs.as_ref(),
            bm.as_ref(),
            bc2.as_ref(),
            m,
            k,
            n,
            8,
            4,
        );
        rec.finish().unwrap();
        let mut bytes2 = vec![0u8; mpad * n * 4];
        be.download(bc2.as_ref(), &mut bytes2).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes2);
        let mut e = 0f32;
        for r in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|x| a[r * k + x] * wq_ref[col * k + x]).sum();
                e = e.max((got[r * n + col] - want).abs());
            }
        }
        println!("matmul_proj quant max_err={e:e}");
        assert!(e < 5e-3, "matmul_proj quant mismatch: {e}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_q_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (rows, in_f, out_f) = (2usize, 128usize, 5usize); // in_f % 16 == 0
        let numel = in_f * out_f;
        // unified quant: u8 quants + per-16-block f16 scale/min; dequant = scale*q + min
        let qu8: Vec<u8> = (0..numel).map(|i| (i * 7 % 64) as u8).collect();
        let scales: Vec<u16> = (0..numel / 16)
            .map(|b| half::f16::from_f32(0.01 + (b % 5) as f32 * 0.003).to_bits())
            .collect();
        let mins: Vec<u16> = (0..numel / 16)
            .map(|b| half::f16::from_f32(-0.2 + (b % 3) as f32 * 0.05).to_bits())
            .collect();
        let mut quants = vec![0u32; numel / 4];
        for (g, &q) in qu8.iter().enumerate() {
            quants[g / 4] |= (q as u32) << (8 * (g % 4));
        }
        let x: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();

        let bq = be
            .upload_weight_bytes(bytemuck::cast_slice(&quants))
            .unwrap();
        let bs = be
            .upload_weight_bytes(bytemuck::cast_slice(&scales))
            .unwrap();
        let bm = be.upload_weight_bytes(bytemuck::cast_slice(&mins)).unwrap();
        let upx = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(upx.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let by = be.alloc(rows * out_f * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_q(
            bq.as_ref(),
            bs.as_ref(),
            bm.as_ref(),
            upx.as_ref(),
            by.as_ref(),
            rows,
            in_f,
            out_f,
            8,
            4,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; rows * out_f * 4];
        be.download(by.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);

        let dq = |g: usize| {
            half::f16::from_bits(scales[g / 16]).to_f32() * qu8[g] as f32
                + half::f16::from_bits(mins[g / 16]).to_f32()
        };
        let mut maxe = 0f32;
        for r in 0..rows {
            for o in 0..out_f {
                let want: f32 = (0..in_f).map(|i| dq(o * in_f + i) * x[r * in_f + i]).sum();
                maxe = maxe.max((got[r * out_f + o] - want).abs());
            }
        }
        println!("linear_q max_err = {maxe:e}");
        assert!(maxe < 1e-3, "linear_q mismatch: {maxe}");
    }

    // Covers the subgroup GEMV across BOTH quant widths and the residual variant at realistic
    // projection sizes (the original linear_q test only hit q8 / no-residual / tiny dims).
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn mul_mat_vec_q_all_variants() {
        let be = VulkanBackend::new().unwrap();
        // (4,5)=Q4_0/Q4_1/Q4K  (8,4)=Q6K  (8,5)=Q5_0/Q5_1/Q8_0  (4,4)=Q2K/Q3K
        for &(bits, blk_shift) in &[(4u32, 5u32), (8u32, 4u32), (8u32, 5u32), (4u32, 4u32)] {
            for &res in &[false, true] {
                let (rows, in_f, out_f) = (1usize, 1024usize, 1024usize);
                let numel = in_f * out_f;
                let block = 1usize << blk_shift;
                let nblk = numel / block;
                let qmax = if bits == 4 { 16usize } else { 256usize };
                let qv: Vec<u32> = (0..numel).map(|i| (i * 7 % qmax) as u32).collect();
                let scales: Vec<u16> = (0..nblk)
                    .map(|b| half::f16::from_f32(0.01 + (b % 5) as f32 * 0.003).to_bits())
                    .collect();
                let mins: Vec<u16> = (0..nblk)
                    .map(|b| half::f16::from_f32(-0.2 + (b % 3) as f32 * 0.05).to_bits())
                    .collect();
                // pack quants: q4 = 8 nibbles/u32, q8 = 4 bytes/u32
                let per = if bits == 4 { 8usize } else { 4usize };
                let shift = if bits == 4 { 4u32 } else { 8u32 };
                let mut quants = vec![0u32; numel / per];
                for (g, &q) in qv.iter().enumerate() {
                    quants[g / per] |= q << (shift * (g % per) as u32);
                }
                let x: Vec<f32> = (0..rows * in_f)
                    .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
                    .collect();
                let r: Vec<f32> = (0..rows * out_f).map(|i| (i % 7) as f32 * 0.1).collect();

                let bq = be
                    .upload_weight_bytes(bytemuck::cast_slice(&quants))
                    .unwrap();
                let bs = be
                    .upload_weight_bytes(bytemuck::cast_slice(&scales))
                    .unwrap();
                let bm = be.upload_weight_bytes(bytemuck::cast_slice(&mins)).unwrap();
                let upx = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
                be.upload(upx.as_ref(), bytemuck::cast_slice(&x)).unwrap();
                let upr = be.alloc(r.len() * 4, BufferUsage::Staging).unwrap();
                be.upload(upr.as_ref(), bytemuck::cast_slice(&r)).unwrap();
                let by = be.alloc(rows * out_f * 4, BufferUsage::Readback).unwrap();
                let rec = be.recorder().unwrap();
                if res {
                    rec.linear_add_q(
                        bq.as_ref(),
                        bs.as_ref(),
                        bm.as_ref(),
                        upx.as_ref(),
                        upr.as_ref(),
                        by.as_ref(),
                        rows,
                        in_f,
                        out_f,
                        bits,
                        blk_shift,
                    );
                } else {
                    rec.linear_q(
                        bq.as_ref(),
                        bs.as_ref(),
                        bm.as_ref(),
                        upx.as_ref(),
                        by.as_ref(),
                        rows,
                        in_f,
                        out_f,
                        bits,
                        blk_shift,
                    );
                }
                rec.finish().unwrap();
                let mut bytes = vec![0u8; rows * out_f * 4];
                be.download(by.as_ref(), &mut bytes).unwrap();
                let got: &[f32] = bytemuck::cast_slice(&bytes);

                let dq = |g: usize| {
                    half::f16::from_bits(scales[g / block]).to_f32() * qv[g] as f32
                        + half::f16::from_bits(mins[g / block]).to_f32()
                };
                let mut maxe = 0f32;
                for ri in 0..rows {
                    for o in 0..out_f {
                        let mut want: f32 =
                            (0..in_f).map(|i| dq(o * in_f + i) * x[ri * in_f + i]).sum();
                        if res {
                            want += r[ri * out_f + o];
                        }
                        maxe = maxe.max((got[ri * out_f + o] - want).abs());
                    }
                }
                println!("mul_mat_vec_q bits={bits} res={res} max_err={maxe:e}");
                assert!(maxe < 5e-3, "bits={bits} res={res} mismatch: {maxe}");
            }
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attn_in_q_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (ne, q_dim, kvrow, eps) = (128usize, 64usize, 32usize, 1e-5f32);
        let mkw = |salt: usize, out: usize| {
            let numel = out * ne;
            let qu8: Vec<u8> = (0..numel).map(|i| ((i * 7 + salt) % 64) as u8).collect();
            let scales: Vec<u16> = (0..numel / 16)
                .map(|b| half::f16::from_f32(0.01 + ((b + salt) % 5) as f32 * 0.003).to_bits())
                .collect();
            let mins: Vec<u16> = (0..numel / 16)
                .map(|b| half::f16::from_f32(-0.2 + ((b + salt) % 3) as f32 * 0.05).to_bits())
                .collect();
            let mut quants = vec![0u32; numel / 4];
            for (g, &q) in qu8.iter().enumerate() {
                quants[g / 4] |= (q as u32) << (8 * (g % 4));
            }
            (qu8, scales, mins, quants)
        };
        let rows = 3usize; // exercise rows>1 (short-prompt prefill path)
        let (q8, qs, qm, qq) = mkw(1, q_dim);
        let (k8, ks, km, kq) = mkw(2, kvrow);
        let (v8, vs, vm, vq) = mkw(3, kvrow);
        let hidden: Vec<f32> = (0..rows * ne)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.06)
            .collect();
        let nw: Vec<f32> = (0..ne).map(|i| 1.0 + (i % 7) as f32 * 0.02).collect();

        let up = |b: &[u8]| be.upload_weight_bytes(b).unwrap();
        let bh = be.alloc(rows * ne * 4, BufferUsage::Staging).unwrap();
        be.upload(bh.as_ref(), bytemuck::cast_slice(&hidden))
            .unwrap();
        let bnw = be.alloc(ne * 4, BufferUsage::Staging).unwrap();
        be.upload(bnw.as_ref(), bytemuck::cast_slice(&nw)).unwrap();
        let (bqq, bqs, bqm) = (
            up(bytemuck::cast_slice(&qq)),
            up(bytemuck::cast_slice(&qs)),
            up(bytemuck::cast_slice(&qm)),
        );
        let (bkq, bks, bkm) = (
            up(bytemuck::cast_slice(&kq)),
            up(bytemuck::cast_slice(&ks)),
            up(bytemuck::cast_slice(&km)),
        );
        let (bvq, bvs, bvm) = (
            up(bytemuck::cast_slice(&vq)),
            up(bytemuck::cast_slice(&vs)),
            up(bytemuck::cast_slice(&vm)),
        );
        let bqr = be.alloc(rows * q_dim * 4, BufferUsage::Readback).unwrap();
        let bkr = be.alloc(rows * kvrow * 4, BufferUsage::Readback).unwrap();
        let bvr = be.alloc(rows * kvrow * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.attn_in_q(
            bh.as_ref(),
            bnw.as_ref(),
            (bqq.as_ref(), bqs.as_ref(), bqm.as_ref()),
            (bkq.as_ref(), bks.as_ref(), bkm.as_ref()),
            (bvq.as_ref(), bvs.as_ref(), bvm.as_ref()),
            bqr.as_ref(),
            bkr.as_ref(),
            bvr.as_ref(),
            rows,
            ne,
            q_dim,
            kvrow,
            eps,
            (8, 4),
            (8, 4),
            (8, 4),
        );
        rec.finish().unwrap();
        let rd = |b: &dyn Buffer, n: usize| {
            let mut by = vec![0u8; n * 4];
            be.download(b, &mut by).unwrap();
            bytemuck::cast_slice::<u8, f32>(&by).to_vec()
        };
        let (gq, gk, gv) = (
            rd(bqr.as_ref(), rows * q_dim),
            rd(bkr.as_ref(), rows * kvrow),
            rd(bvr.as_ref(), rows * kvrow),
        );
        let dq = |q8: &[u8], s: &[u16], m: &[u16], g: usize| {
            half::f16::from_bits(s[g / 16]).to_f32() * q8[g] as f32
                + half::f16::from_bits(m[g / 16]).to_f32()
        };
        let check = |got: &[f32], q8: &[u8], s: &[u16], m: &[u16], out: usize, tag: &str| {
            let mut maxe = 0f32;
            for r in 0..rows {
                let ms: f32 = hidden[r * ne..(r + 1) * ne]
                    .iter()
                    .map(|h| h * h)
                    .sum::<f32>()
                    / ne as f32;
                let scale = 1.0 / (ms + eps).sqrt();
                for o in 0..out {
                    let want: f32 = scale
                        * (0..ne)
                            .map(|i| hidden[r * ne + i] * nw[i] * dq(q8, s, m, o * ne + i))
                            .sum::<f32>();
                    maxe = maxe.max((got[r * out + o] - want).abs());
                }
            }
            println!("attn_in_q {tag} max_err = {maxe:e}");
            assert!(maxe < 1e-3, "attn_in_q {tag} mismatch: {maxe}");
        };
        check(&gq, &q8, &qs, &qm, q_dim, "q");
        check(&gk, &k8, &ks, &km, kvrow, "k");
        check(&gv, &v8, &vs, &vm, kvrow, "v");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn ffn_in_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (ne, nff, eps) = (128usize, 256usize, 1e-5f32);
        let hidden: Vec<f32> = (0..ne)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.05)
            .collect();
        let nw: Vec<f32> = (0..ne).map(|i| 1.0 + (i % 5) as f32 * 0.01).collect();
        // f16-rounded weights so the test checks kernel logic, not f16 precision.
        let wgu: Vec<f32> = (0..2 * nff * ne)
            .map(|i| half::f16::from_f32(((i * 31 % 97) as f32 - 48.0) * 0.002).to_f32())
            .collect();

        let up = |v: &[f32], u| {
            let b = be.alloc(v.len() * 4, u).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let bh = up(&hidden, BufferUsage::Staging);
        let bn = up(&nw, BufferUsage::Staging);
        let bw = be.upload_weight_f16(&wgu).unwrap();
        let act = be.alloc(nff * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.ffn_in(
            bh.as_ref(),
            bn.as_ref(),
            bw.as_ref(),
            act.as_ref(),
            1,
            ne,
            nff,
            eps,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; nff * 4];
        be.download(act.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);

        let norm = rmsnorm_cpu(&hidden, &nw, eps);
        let mut max_err = 0f32;
        for (f, &got_f) in got.iter().take(nff).enumerate() {
            let g = dot(&wgu, f, ne, &norm);
            let u = dot(&wgu, nff + f, ne, &norm);
            let want = (g / (1.0 + (-g).exp())) * u;
            max_err = max_err.max((got_f - want).abs());
        }
        println!("ffn_in max_err = {max_err:e}");
        assert!(max_err < 1e-4, "ffn_in mismatch: {max_err}");
    }

    fn run_attn_in(rows: usize) {
        let be = VulkanBackend::new().unwrap();
        let (ne, nh, nkv, hd) = (128usize, 4usize, 2usize, 32usize);
        let (rope_dim, theta, pos, eps) = (32usize, 10000f32, 3usize, 1e-5f32);
        let (q_dim, kv_dim, ctx) = (nh * hd, nkv * hd, 16usize);
        let hidden: Vec<f32> = (0..rows * ne)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.04)
            .collect();
        let nw: Vec<f32> = (0..ne).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
        // f16-rounded weights so the test checks kernel logic, not f16 precision.
        let mkw = |r: usize, salt: usize| -> Vec<f32> {
            (0..r * ne)
                .map(|i| {
                    half::f16::from_f32((((i + salt) * 17 % 89) as f32 - 44.0) * 0.003).to_f32()
                })
                .collect()
        };
        let wq = mkw(q_dim, 1);
        let wk = mkw(kv_dim, 2);
        let wv = mkw(kv_dim, 3);

        let up = |v: &[f32], u| {
            let b = be.alloc(v.len() * 4, u).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let bh = up(&hidden, BufferUsage::Staging);
        let bn = up(&nw, BufferUsage::Staging);
        let bwq = be.upload_weight_f16(&wq).unwrap();
        let bwk = be.upload_weight_f16(&wk).unwrap();
        let bwv = be.upload_weight_f16(&wv).unwrap();
        // q + K/V cache are f16
        let bq = be.alloc(rows * q_dim * 2, BufferUsage::Readback).unwrap();
        let bkc = be.alloc(ctx * kv_dim * 2, BufferUsage::Readback).unwrap();
        let bvc = be.alloc(ctx * kv_dim * 2, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.attn_in(
            bh.as_ref(),
            bn.as_ref(),
            bwq.as_ref(),
            bwk.as_ref(),
            bwv.as_ref(),
            bq.as_ref(),
            bkc.as_ref(),
            bvc.as_ref(),
            rows,
            ne,
            nh,
            nkv,
            hd,
            rope_dim,
            theta,
            pos,
            eps,
        );
        rec.finish().unwrap();
        let rd = |b: &dyn Buffer, n: usize| {
            let mut bytes = vec![0u8; n * 2];
            be.download(b, &mut bytes).unwrap();
            bytemuck::cast_slice::<u8, u16>(&bytes)
                .iter()
                .map(|&h| half::f16::from_bits(h).to_f32())
                .collect::<Vec<f32>>()
        };
        let gq = rd(bq.as_ref(), rows * q_dim);
        let gk = rd(bkc.as_ref(), ctx * kv_dim);
        let gv = rd(bvc.as_ref(), ctx * kv_dim);

        let mut maxe = 0f32;
        for r in 0..rows {
            let norm = rmsnorm_cpu(&hidden[r * ne..(r + 1) * ne], &nw, eps);
            let abs = pos + r; // per-row absolute position (the prefill correctness check)
            let mut wq_r = vec![0f32; q_dim];
            for (c, v) in wq_r.iter_mut().enumerate() {
                *v = dot(&wq, c, ne, &norm);
            }
            for h in 0..nh {
                rope_head(&mut wq_r[h * hd..(h + 1) * hd], hd, rope_dim, theta, abs);
            }
            let mut wk_r = vec![0f32; kv_dim];
            for (c, v) in wk_r.iter_mut().enumerate() {
                *v = dot(&wk, c, ne, &norm);
            }
            for h in 0..nkv {
                rope_head(&mut wk_r[h * hd..(h + 1) * hd], hd, rope_dim, theta, abs);
            }
            for c in 0..q_dim {
                maxe = maxe.max((gq[r * q_dim + c] - wq_r[c]).abs());
            }
            for c in 0..kv_dim {
                maxe = maxe.max((gk[abs * kv_dim + c] - wk_r[c]).abs());
                maxe = maxe.max((gv[abs * kv_dim + c] - dot(&wv, c, ne, &norm)).abs());
            }
        }
        println!("attn_in rows={rows} max_err={maxe:e}");
        assert!(maxe < 5e-3, "attn_in mismatch rows={rows}: {maxe}"); // f16 q/k/v output
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attn_in_matches_cpu() {
        run_attn_in(1); // decode
        run_attn_in(3); // prefill — exercises per-row RoPE position
    }
}
