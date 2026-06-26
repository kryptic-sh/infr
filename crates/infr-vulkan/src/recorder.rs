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
            device.cmd_dispatch(self.cmd, groups, 1, 1);
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
            .kernel("linear_f16", crate::linear::LINEAR_F16_WGSL, 3, 12);
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
            .kernel("linear_res", crate::linear::LINEAR_RES_WGSL, 4, 12);
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
        let k = self.be.kernel("rmsnorm", ops::RMSNORM_WGSL, 3, 12);
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
        let k = self.be.kernel("rope", ops::ROPE_WGSL, 2, 24);
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
        // pass 1: per-chunk partials
        self.stamp("attn_partial");
        let k1 = self
            .be
            .kernel("attn_partial", ops::ATTN_PARTIAL_WGSL, 6, 24);
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
        // pass 2: combine
        self.stamp("attn_combine");
        let k2 = self
            .be
            .kernel("attn_combine", ops::ATTN_COMBINE_WGSL, 4, 12);
        let mut p2 = [0u8; 12];
        p2[0..4].copy_from_slice(&(nh as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(hd as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        self.dispatch(
            k2,
            &[Self::vkb(pm), Self::vkb(pl), Self::vkb(pacc), Self::vkb(o)],
            1,
            &p2,
            nh as u32,
        );
    }

    /// Record a buffer→buffer copy of `bytes` from `src[0..]` into `dst[dst_offset..]`.
    /// Used to append new K/V rows into the persistent cache.
    pub fn copy(&self, src: &dyn Buffer, dst: &dyn Buffer, dst_offset: usize, bytes: usize) {
        let device = &self.be.shared.device;
        self.sync(&[Self::vkb(src)], &[Self::vkb(dst)], true);
        self.dirty_transfer.set(true);
        unsafe {
            device.cmd_copy_buffer(
                self.cmd,
                Self::vkb(src),
                Self::vkb(dst),
                &[vk::BufferCopy {
                    src_offset: 0,
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
        let kern = self.be.kernel("attention", ops::ATTENTION_WGSL, 4, 16);
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
            .kernel("silu_mul_fused", ops::SILU_MUL_FUSED_WGSL, 2, 8);
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
        let k = self.be.kernel("silu_mul", ops::SILU_MUL_WGSL, 3, 4);
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
        let k = self.be.kernel("add", ops::ADD_WGSL, 3, 4);
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
        let q = gen(q_len * nh * hd, 1);
        let k = gen(kv_len * nkv * hd, 2);
        let v = gen(kv_len * nkv * hd, 3);
        let up = |val: &[f32]| {
            let b = be.alloc(val.len() * 4, BufferUsage::Staging).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(val)).unwrap();
            b
        };
        let bq = up(&q);
        let bk = up(&k);
        let bv = up(&v);
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
        assert!(err < 1e-4, "attention_kv mismatch: {err}");
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
        let q = gen(nh * hd, 1);
        let k = gen(kv_len * nkv * hd, 2);
        let v = gen(kv_len * nkv * hd, 3);
        let up = |val: &[f32]| {
            let b = be.alloc(val.len() * 4, BufferUsage::Staging).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(val)).unwrap();
            b
        };
        let bq = up(&q);
        let bk = up(&k);
        let bv = up(&v);
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
        assert!(err < 1e-4, "split mismatch: {err}");
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
        for f in 0..nff {
            let g = dot(&wgu, f, ne, &norm);
            let u = dot(&wgu, nff + f, ne, &norm);
            let want = (g / (1.0 + (-g).exp())) * u;
            max_err = max_err.max((got[f] - want).abs());
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
        let bq = be.alloc(rows * q_dim * 4, BufferUsage::Readback).unwrap();
        let bkc = be.alloc(ctx * kv_dim * 4, BufferUsage::Readback).unwrap();
        let bvc = be.alloc(ctx * kv_dim * 4, BufferUsage::Readback).unwrap();
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
            let mut bytes = vec![0u8; n * 4];
            be.download(b, &mut bytes).unwrap();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        };
        let gq = rd(bq.as_ref(), rows * q_dim);
        let gk = rd(bkc.as_ref(), ctx * kv_dim);
        let gv = rd(bvc.as_ref(), ctx * kv_dim);

        let mut maxe = 0f32;
        for r in 0..rows {
            let norm = rmsnorm_cpu(&hidden[r * ne..(r + 1) * ne], &nw, eps);
            let abs = pos + r; // per-row absolute position (the prefill correctness check)
            let mut wq_r = vec![0f32; q_dim];
            for c in 0..q_dim {
                wq_r[c] = dot(&wq, c, ne, &norm);
            }
            for h in 0..nh {
                rope_head(&mut wq_r[h * hd..(h + 1) * hd], hd, rope_dim, theta, abs);
            }
            let mut wk_r = vec![0f32; kv_dim];
            for c in 0..kv_dim {
                wk_r[c] = dot(&wk, c, ne, &norm);
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
        assert!(maxe < 1e-4, "attn_in mismatch rows={rows}: {maxe}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attn_in_matches_cpu() {
        run_attn_in(1); // decode
        run_attn_in(3); // prefill — exercises per-row RoPE position
    }
}
