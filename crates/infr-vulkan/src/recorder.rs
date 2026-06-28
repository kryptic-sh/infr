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
use super::{as_vk_buf, be, ops, VulkanBackend};

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
}

impl<'a> Recorder<'a> {
    pub(crate) fn new(backend: &'a VulkanBackend) -> Result<Self> {
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
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
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
        let prof2 = std::env::var("INFR_PROF2").is_ok();
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
        })
    }

    /// Record a profiling timestamp (BOTTOM_OF_PIPE) tagged with an op label, if INFR_PROF2.
    fn stamp(&self, label: &'static str) {
        if !self.prof2 {
            return;
        }
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
            .kernel_spv("linear_f16", crate::gemm::linear_f16_spv(), 3, 12);
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
        let kern = self.be.kernel_spv_sg(name, spv, 5, 20, 32);
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
            .kernel_spv_sg("quant_q8", crate::gemm::quant_q8_spv(), 4, 12, 32);
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
        let km =
            self.be
                .kernel_spv_sg("gemm_proj_mmq", crate::gemm::gemm_proj_mmq_spv(), 7, 12, 32);
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
                let k = self.be.kernel_spv_sg(
                    name,
                    crate::gemm::mul_mat_vec_q_spv(bits, false),
                    5,
                    20,
                    32,
                );
                let groups = (out_f as u32).div_ceil(MMV_NUM_ROWS);
                self.dispatch(k, &bufs, 1, &push, rows as u32 * groups);
                return;
            }
        }
        let k = self
            .be
            .kernel_spv("linear_q", crate::gemm::linear_q_spv(), 5, 20);
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
        self.stamp("lm_head");
        let name = crate::linear::native_kernel_name(dtype, false);
        let wgsl = crate::linear::native_gemv_wgsl(dtype, false);
        let k = self.be.kernel(name, &wgsl, 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
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
        let wgsl = crate::linear::native_gemv_wgsl(dtype, true);
        let k = self.be.kernel(name, &wgsl, 4, 12);
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
                let k = self.be.kernel_spv_sg(
                    name,
                    crate::gemm::mul_mat_vec_q_spv(bits, true),
                    6,
                    20,
                    32,
                );
                let groups = (out_f as u32).div_ceil(MMV_NUM_ROWS);
                self.dispatch(k, &bufs, 1, &push, rows as u32 * groups);
                return;
            }
        }
        let k = self
            .be
            .kernel_spv("linear_res_q", crate::gemm::linear_res_q_spv(), 6, 20);
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
            .kernel_spv("linear_res", crate::gemm::linear_res_spv(), 4, 12);
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
        let kern = self.be.kernel("attn_in", ops::ATTN_IN_WGSL, 8, 36);
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
        let k = self.be.kernel("ffn_in_q", ops::FFN_IN_Q_WGSL, 6, 24);
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
        let k = self.be.kernel("attn_in_q", ops::ATTN_IN_Q_WGSL, 14, 44);
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
        let k = self.be.kernel("ffn_in", ops::FFN_IN_WGSL, 4, 16);
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
            .kernel_spv_sg("rmsnorm", crate::gemm::rmsnorm_spv(), 3, 12, 32);
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
        let k = self.be.kernel_spv("rope", crate::gemm::rope_spv(), 2, 24);
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
    ) {
        self.stamp("attention_kv");
        let kern = self
            .be
            .kernel("attention_kv", ops::ATTENTION_KV_WGSL, 4, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(q_len as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
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
    ) {
        let mpad = (n.div_ceil(64) * 64) as u32;
        // kv padded to 256 (the 8-warp attn_qk's BN); extra cols are masked in softmax. Still %64 so
        // the 4-warp fallback + softmax + attn_pv are unaffected.
        let kv_pad = (kv_len.div_ceil(256) * 256) as u32;
        let hdu = hd as u32;
        let scale = 1.0f32 / (hd as f32).sqrt();

        // stage 1: S = scale·Q·Kᵀ. 8-warp/256-thread warptile (BN=256, matches ollama's mul_mm)
        // unless INFR_NO_QK_WARP forces the 4-warp/2×2 attn_qk.
        self.stamp("attn_qk");
        let qk_warp = std::env::var("INFR_NO_QK_WARP").is_err();
        let (qk_name, qk_spv, qk_bn) = if qk_warp {
            ("attn_qk_warp", crate::gemm::attn_qk_warp_spv(), 256u32)
        } else {
            ("attn_qk", crate::gemm::attn_qk_spv(), 64u32)
        };
        let kqk = self.be.kernel_spv_sg(qk_name, qk_spv, 3, 24, 32);
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

        // stage 2: row softmax (causal), in place S → P
        self.stamp("attn_softmax");
        let ksm = self
            .be
            .kernel_spv("attn_softmax", crate::gemm::attn_softmax_spv(), 1, 16);
        let mut ps = [0u8; 16];
        ps[0..4].copy_from_slice(&mpad.to_ne_bytes());
        ps[4..8].copy_from_slice(&kv_pad.to_ne_bytes());
        ps[8..12].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        ps[12..16].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
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
        let kpv = self.be.kernel_spv_sg(pv_name, pv_spv, 3, 28, 32);
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
                .kernel_spv("attn_pv_reduce", crate::gemm::attn_pv_reduce_spv(), 2, 8);
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
        // hd=128 → 8-warp register-blocked partial (always via partial+combine). Other hd → the
        // 4-subgroup path: single fused kernel for n_splits==1, else the scalar partial.
        let warp = hd == 128 && std::env::var("INFR_NO_FLASH_WARP").is_err();
        if n_splits == 1 && !warp {
            self.stamp("attn_flash");
            let k = self
                .be
                .kernel_spv_sg("attn_flash", crate::gemm::attn_flash_spv(), 4, 24, 32);
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
                base_wg,
            );
            return;
        }
        // split-K partials
        self.stamp("attn_flash");
        let ksplit = (kv_len as u32).div_ceil(n_splits).div_ceil(64) * 64;
        let (pname, pspv) = if warp {
            ("attn_flash_warp", crate::gemm::attn_flash_warp_spv())
        } else {
            ("attn_flash_partial", crate::gemm::attn_flash_partial_spv())
        };
        let kp = self.be.kernel_spv_sg(pname, pspv, 6, 32, 32);
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
        // combine → attn
        self.stamp("attn_flash");
        let kc2 = self.be.kernel_spv_sg(
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
        let mpad = (n.div_ceil(128) * 128) as u32; // Br=128
        let base_wg = (mpad / 128) * nh as u32;
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
        let kp = self.be.kernel_spv_sg(
            "attn_flash_reg",
            crate::gemm::attn_flash_reg_spv(),
            6,
            32,
            32,
        );
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
        let kc2 = self.be.kernel_spv_sg(
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
            .kernel_spv("store_f16", crate::gemm::store_f16_spv(), 2, 8);
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
    ) {
        self.stamp("qk_norm_rope");
        let k = self
            .be
            .kernel("qk_norm_rope", ops::QK_NORM_ROPE_WGSL, 3, 32);
        let mut push = [0u8; 32];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(rope_pos as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(out_base as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(nw), Self::vkb(y)],
            1,
            &push,
            (rows * nheads) as u32,
        );
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
            .kernel_spv_sg("attn_partial", crate::gemm::attn_partial_spv(), 6, 24, 32);
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
            .kernel_spv("attn_combine", crate::gemm::attn_combine_spv(), 4, 16);
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
            .kernel_spv("attention", crate::gemm::attention_spv(), 4, 16);
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
        let k = self
            .be
            .kernel_spv("silu_mul_fused", crate::gemm::silu_mul_fused_spv(), 2, 8);
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
        let k = self
            .be
            .kernel_spv("silu_mul", crate::gemm::silu_mul_spv(), 3, 4);
        self.dispatch(
            k,
            &[Self::vkb(gate), Self::vkb(up), Self::vkb(y)],
            1,
            &(n as u32).to_ne_bytes(),
            (n as u32).div_ceil(64),
        );
    }

    /// Elementwise add; in place allowed (`a` may equal `y`).
    pub fn add(&self, a: &dyn Buffer, b: &dyn Buffer, y: &dyn Buffer, n: usize) {
        let k = self.be.kernel_spv("add", crate::gemm::add_spv(), 3, 4);
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

    fn attn_kv_cpu(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (hd as f32).sqrt();
        let mut o = vec![0f32; q_len * nh * hd];
        for ti in 0..q_len {
            let abs = pos_offset + ti;
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                let qb = (ti * nh + h) * hd;
                let mut sc = vec![0f32; abs + 1];
                let mut mx = f32::NEG_INFINITY;
                for (j, scj) in sc.iter_mut().enumerate() {
                    let kb = (j * nkv + kvh) * hd;
                    let d: f32 = (0..hd).map(|x| q[qb + x] * k[kb + x]).sum();
                    *scj = d * scale;
                    mx = mx.max(*scj);
                }
                let mut l = 0f32;
                for s in &sc {
                    l += (s - mx).exp();
                }
                let ob = (ti * nh + h) * hd;
                for (j, s) in sc.iter().enumerate() {
                    let p = (s - mx).exp() / l;
                    let vb = (j * nkv + kvh) * hd;
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
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; q_len * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset);
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
        let want = attn_kv_cpu(&q, &k, &v, 1, nh, nkv, hd, pos_offset);
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
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_prefill_nonfa_matches_cpu() {
        run_attn_prefill_nonfa(64, 64, 2, 1, 128);
        run_attn_prefill_nonfa(128, 200, 4, 2, 128);
        run_attn_prefill_nonfa(70, 70, 2, 2, 128);
        run_attn_prefill_nonfa(192, 500, 2, 1, 128);
        run_attn_prefill_nonfa(80, 300, 9, 3, 64);
        // force the split-K PV path (n_splits>1) and verify the partial-sum reduce is correct
        std::env::set_var("INFR_PV_SPLITS", "4");
        run_attn_prefill_nonfa(70, 300, 4, 2, 128);
        run_attn_prefill_nonfa(128, 500, 2, 1, 128);
        std::env::remove_var("INFR_PV_SPLITS");
    }

    fn run_attn_prefill_nonfa(q_len: usize, kv_len: usize, nh: usize, nkv: usize, hd: usize) {
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
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset);
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
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset);
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
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset);
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
