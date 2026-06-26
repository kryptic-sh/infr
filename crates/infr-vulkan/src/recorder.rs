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
        })
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
        writes: &[vk::Buffer],
        push: &[u8],
        groups: u32,
    ) {
        // Inputs are every bound buffer except the output (always last). Do NOT drop in-place
        // buffers (e.g. rope x==y): they are genuinely read and may carry a RAW from a prior op.
        let reads = &buffers[..buffers.len() - 1];
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
        let k = self.be.kernel("linear", crate::linear::LINEAR_WGSL, 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(w), Self::vkb(x), Self::vkb(y)],
            &[Self::vkb(y)],
            &push,
            ((rows * out_f) as u32).div_ceil(64),
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
            &[Self::vkb(y)],
            &push,
            ((rows * out_f) as u32).div_ceil(64),
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
        let k = self.be.kernel("rmsnorm", ops::RMSNORM_WGSL, 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(w), Self::vkb(y)],
            &[Self::vkb(y)],
            &push,
            (rows as u32).div_ceil(64),
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
            &[Self::vkb(y)],
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
            &[Self::vkb(o)],
            &push,
            (q_len * nh) as u32,
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
        let kern = self.be.kernel("attention", ops::ATTENTION_WGSL, 4, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[Self::vkb(q), Self::vkb(k), Self::vkb(v), Self::vkb(o)],
            &[Self::vkb(o)],
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
            &[Self::vkb(y)],
            &push,
            (rows * nff) as u32,
        );
    }

    pub fn silu_mul(&self, gate: &dyn Buffer, up: &dyn Buffer, y: &dyn Buffer, n: usize) {
        let k = self.be.kernel("silu_mul", ops::SILU_MUL_WGSL, 3, 4);
        self.dispatch(
            k,
            &[Self::vkb(gate), Self::vkb(up), Self::vkb(y)],
            &[Self::vkb(y)],
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
            &[Self::vkb(y)],
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
            let cmd_pool = *self.be.shared.cmd_pool.lock().unwrap();
            device.free_command_buffers(cmd_pool, &[self.cmd]);
            device.destroy_descriptor_pool(self.pool, None);
        }
        Ok(())
    }
}

impl VulkanBackend {
    /// Start recording a single-submit forward.
    pub fn recorder(&self) -> Result<Recorder<'_>> {
        Recorder::new(self)
    }
}
