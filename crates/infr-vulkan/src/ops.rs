//! GPU compute kernels for the transformer ops the Llama forward needs (RMSNorm, RoPE, …),
//! plus a generic cached-kernel runner. Each op is validated against the host reference in
//! `infr-llama` (the working oracle).
//!
//! These eager runners (one submit each) are for validation; the single-command-buffer
//! resident forward (real speedup) reuses the same kernels via record APIs (added next).

use std::ffi::CStr;

use ash::vk;

use infr_core::{
    backend::{Buffer, BufferUsage},
    error::Result,
    Backend,
};

use super::{as_vk_buf, be, VulkanBackend};

/// Cached, reusable compute objects for one kernel. All fields are Vulkan handles (Copy).
#[derive(Clone, Copy)]
pub(crate) struct ComputeKernel {
    pub shader: vk::ShaderModule,
    pub ds_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub desc_pool: vk::DescriptorPool,
    pub n_buf: usize,
    pub push_size: u32,
}

fn compile_wgsl(src: &str) -> Vec<u32> {
    use naga::back::spv;
    use naga::front::wgsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let module = wgsl::parse_str(src).expect("WGSL parse");
    let info = Validator::new(
        ValidationFlags::all(),
        Capabilities::IMMEDIATES | Capabilities::SHADER_FLOAT16,
    )
    .validate(&module)
    .expect("WGSL validate");
    spv::write_vec(
        &module,
        &info,
        &spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        },
        None,
    )
    .expect("SPIR-V write")
}

pub(crate) fn make_compute_kernel(
    device: &ash::Device,
    wgsl: &str,
    n_buf: usize,
    push_size: u32,
) -> ComputeKernel {
    make_compute_kernel_from_spv(device, &compile_wgsl(wgsl), n_buf, push_size, None)
}

pub(crate) fn make_compute_kernel_from_spv(
    device: &ash::Device,
    spv: &[u32],
    n_buf: usize,
    push_size: u32,
    required_sg: Option<u32>,
) -> ComputeKernel {
    let shader = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None)
    }
    .expect("shader module");

    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_buf)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let ds_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .expect("ds layout");

    let mut plinfo =
        vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&ds_layout));
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_size);
    if push_size > 0 {
        plinfo = plinfo.push_constant_ranges(std::slice::from_ref(&push_range));
    }
    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&plinfo, None) }.expect("pl layout");

    let entry = CStr::from_bytes_with_nul(b"main\0").unwrap();
    let mut req_sz =
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(0);
    let mut stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(entry);
    if let Some(sz) = required_sg {
        req_sz = req_sz.required_subgroup_size(sz);
        stage = stage
            .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
            .push_next(&mut req_sz);
    }
    let pipeline = unsafe {
        device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)],
                None,
            )
            .expect("pipeline")[0]
    };

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: n_buf as u32,
    }];
    let desc_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
    }
    .expect("desc pool");

    ComputeKernel {
        shader,
        ds_layout,
        pipeline_layout,
        pipeline,
        desc_pool,
        n_buf,
        push_size,
    }
}

pub(crate) fn destroy_compute_kernel(device: &ash::Device, k: &ComputeKernel) {
    unsafe {
        device.destroy_descriptor_pool(k.desc_pool, None);
        device.destroy_pipeline(k.pipeline, None);
        device.destroy_pipeline_layout(k.pipeline_layout, None);
        device.destroy_descriptor_set_layout(k.ds_layout, None);
        device.destroy_shader_module(k.shader, None);
    }
}

impl VulkanBackend {
    /// Fetch-or-build a named kernel; returns a Copy of its handles.
    pub(crate) fn kernel(
        &self,
        name: &'static str,
        wgsl: &str,
        n_buf: usize,
        push_size: u32,
    ) -> ComputeKernel {
        let mut map = self.shared.kernels.lock().unwrap();
        *map.entry(name)
            .or_insert_with(|| make_compute_kernel(&self.shared.device, wgsl, n_buf, push_size))
    }

    /// Like `kernel`, but from precompiled SPIR-V (for GLSL-compiled coopmat shaders).
    pub(crate) fn kernel_spv(
        &self,
        name: &'static str,
        spv: &[u32],
        n_buf: usize,
        push_size: u32,
    ) -> ComputeKernel {
        let mut map = self.shared.kernels.lock().unwrap();
        *map.entry(name).or_insert_with(|| {
            make_compute_kernel_from_spv(&self.shared.device, spv, n_buf, push_size, None)
        })
    }

    /// Like `kernel_spv`, but pins the pipeline's subgroup size (coopmat needs wave32 on RDNA3).
    pub(crate) fn kernel_spv_sg(
        &self,
        name: &'static str,
        spv: &[u32],
        n_buf: usize,
        push_size: u32,
        sg_size: u32,
    ) -> ComputeKernel {
        let mut map = self.shared.kernels.lock().unwrap();
        *map.entry(name).or_insert_with(|| {
            make_compute_kernel_from_spv(&self.shared.device, spv, n_buf, push_size, Some(sg_size))
        })
    }

    /// Eagerly run a kernel: bind `inputs` (read) then one output buffer (read_write), push
    /// `push` bytes, dispatch `groups` workgroups in x. Host-visible buffers → memcpy I/O.
    fn run_kernel(
        &self,
        k: ComputeKernel,
        inputs: &[&[f32]],
        out_len: usize,
        push: &[u8],
        groups: u32,
    ) -> Result<Vec<f32>> {
        assert_eq!(inputs.len() + 1, k.n_buf);
        assert_eq!(push.len() as u32, k.push_size);
        let device = self.shared.device.clone();

        unsafe {
            device
                .reset_descriptor_pool(k.desc_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| be(format!("reset pool: {e}")))?;
        }
        let set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(k.desc_pool)
                        .set_layouts(std::slice::from_ref(&k.ds_layout)),
                )
                .map_err(|e| be(format!("alloc set: {e}")))?[0]
        };

        // input buffers (host-visible) + output buffer (readback)
        let mut bufs = Vec::with_capacity(k.n_buf);
        for inp in inputs {
            let bytes: &[u8] = bytemuck::cast_slice(inp);
            let b = self.alloc(bytes.len().max(4), BufferUsage::Staging)?;
            self.upload(b.as_ref(), bytes)?;
            bufs.push(b);
        }
        let out = self.alloc((out_len * 4).max(4), BufferUsage::Readback)?;

        let vk_bufs: Vec<vk::Buffer> = bufs
            .iter()
            .map(|b| unsafe { as_vk_buf(b.as_ref()) }.buffer)
            .chain(std::iter::once(unsafe { as_vk_buf(out.as_ref()) }.buffer))
            .collect();
        let infos: Vec<vk::DescriptorBufferInfo> = vk_bufs
            .iter()
            .map(|&buffer| vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = (0..k.n_buf)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i..i + 1])
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let push_vec = push.to_vec();
        let shared = std::sync::Arc::clone(&self.shared);
        self.one_shot(move |cmd| unsafe {
            shared
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline);
            shared.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                k.pipeline_layout,
                0,
                &[set],
                &[],
            );
            if k.push_size > 0 {
                shared.device.cmd_push_constants(
                    cmd,
                    k.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push_vec,
                );
            }
            shared.device.cmd_dispatch(cmd, groups, 1, 1);
        })?;

        let mut out_bytes = vec![0u8; out_len * 4];
        self.download(out.as_ref(), &mut out_bytes)?;
        Ok(bytemuck::cast_slice(&out_bytes).to_vec())
    }

    /// Eager GEMV against a PERSISTENT weight buffer (binding 0): `y[rows,out] = x[rows,in] · Wᵀ`.
    /// Unlike `run_kernel`, the weight is not re-uploaded — only the activation x (staging) and y
    /// (readback). `groups` is the workgroup count (cooperative-over-K kernels dispatch `rows*out`).
    /// Shared by the f16 / bf16 eager linears; the f32 `linear` keeps its own (thread-per-output)
    /// path. See module note: one submit per call, for the hybrid (CPU-interleaved) decode path.
    fn linear_wbuf(
        &self,
        k: ComputeKernel,
        groups: u32,
        w_buf: &dyn Buffer,
        x: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>> {
        assert_eq!(x.len(), rows * in_f, "x must be rows*in");
        let device = self.shared.device.clone();
        unsafe {
            device
                .reset_descriptor_pool(k.desc_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| be(format!("reset pool: {e}")))?;
        }
        let set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(k.desc_pool)
                        .set_layouts(std::slice::from_ref(&k.ds_layout)),
                )
                .map_err(|e| be(format!("alloc set: {e}")))?[0]
        };
        let x_bytes: &[u8] = bytemuck::cast_slice(x);
        let buf_x = self.alloc(x_bytes.len().max(4), BufferUsage::Staging)?;
        let buf_y = self.alloc((rows * out_f * 4).max(4), BufferUsage::Readback)?;
        self.upload(buf_x.as_ref(), x_bytes)?;

        let vk_bufs = [
            unsafe { as_vk_buf(w_buf) }.buffer,
            unsafe { as_vk_buf(buf_x.as_ref()) }.buffer,
            unsafe { as_vk_buf(buf_y.as_ref()) }.buffer,
        ];
        let infos: Vec<vk::DescriptorBufferInfo> = vk_bufs
            .iter()
            .map(|&buffer| vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i..i + 1])
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        let shared = std::sync::Arc::clone(&self.shared);
        self.one_shot(move |cmd| unsafe {
            shared
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline);
            shared.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                k.pipeline_layout,
                0,
                &[set],
                &[],
            );
            shared.device.cmd_push_constants(
                cmd,
                k.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            shared.device.cmd_dispatch(cmd, groups, 1, 1);
        })?;
        let mut y = vec![0u8; rows * out_f * 4];
        self.download(buf_y.as_ref(), &mut y)?;
        Ok(bytemuck::cast_slice(&y).to_vec())
    }

    /// Eager f16-weight GEMV: `w_buf` holds `W[out,in]` as f16 (see [`Self::upload_weight_f16`]).
    pub fn linear_f16(
        &self,
        w_buf: &dyn Buffer,
        x: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>> {
        let k = self.kernel("linear_f16_eager", crate::linear::LINEAR_F16_WGSL, 3, 12);
        self.linear_wbuf(k, (rows * out_f) as u32, w_buf, x, rows, in_f, out_f)
    }

    /// Eager bf16-weight GEMV: `w_buf` holds `W[out,in]` as bf16 (see [`Self::upload_weight_bf16`]).
    pub fn linear_bf16(
        &self,
        w_buf: &dyn Buffer,
        x: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>> {
        let k = self.kernel("linear_bf16_eager", crate::linear::LINEAR_BF16_WGSL, 3, 12);
        self.linear_wbuf(k, (rows * out_f) as u32, w_buf, x, rows, in_f, out_f)
    }

    /// RMSNorm over rows: `y[r,i] = x[r,i] / sqrt(mean(x[r]^2)+eps) * w[i]`.
    pub fn rmsnorm(
        &self,
        x: &[f32],
        w: &[f32],
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let k = self.kernel("rmsnorm", RMSNORM_WGSL, 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        self.run_kernel(k, &[x, w], rows * dim, &push, rows as u32) // one workgroup per row
    }

    /// RoPE (ggml NORM, interleaved pairs) over `x` laid out `[t, n_heads, hd]`.
    pub fn rope(
        &self,
        x: &[f32],
        t: usize,
        n_heads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
    ) -> Result<Vec<f32>> {
        let k = self.kernel("rope", ROPE_WGSL, 2, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&0u32.to_ne_bytes()); // pos_offset
        self.run_kernel(k, &[x], t * n_heads * hd, &push, (t * n_heads) as u32)
    }

    /// SwiGLU activation: `y[i] = silu(gate[i]) * up[i]`.
    pub fn silu_mul(&self, gate: &[f32], up: &[f32], n: usize) -> Result<Vec<f32>> {
        let k = self.kernel("silu_mul", SILU_MUL_WGSL, 3, 4);
        self.run_kernel(
            k,
            &[gate, up],
            n,
            &(n as u32).to_ne_bytes(),
            (n as u32).div_ceil(64),
        )
    }

    /// Elementwise add: `y[i] = a[i] + b[i]`.
    pub fn add(&self, a: &[f32], b: &[f32], n: usize) -> Result<Vec<f32>> {
        let k = self.kernel("add", ADD_WGSL, 3, 4);
        self.run_kernel(
            k,
            &[a, b],
            n,
            &(n as u32).to_ne_bytes(),
            (n as u32).div_ceil(64),
        )
    }

    /// Causal GQA attention (online softmax). q `[t, nh, hd]`, k/v `[t, nkv, hd]` → `[t, nh, hd]`.
    /// Requires `hd <= 128`.
    pub fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        t: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) -> Result<Vec<f32>> {
        assert!(hd <= 128, "attention kernel supports hd<=128");
        let kern = self.kernel("attention", ATTENTION_WGSL, 4, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        self.run_kernel(kern, &[q, k, v], t * nh * hd, &push, (t * nh) as u32)
    }
}

// ONE workgroup per row; its 64 threads cooperatively reduce the sum-of-squares (coalesced),
// then write the normalized row. Dispatch `rows` workgroups.
pub(crate) const RMSNORM_WGSL: &str = r#"
struct PC { rows: u32, dim: u32, eps: f32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let r = wid.x;
    let t = lid.x;
    let base = r * pc.dim;
    var pss: f32 = 0.0;
    for (var i: u32 = t; i < pc.dim; i = i + 64u) { let v = x[base + i]; pss = pss + v * v; }
    red[t] = pss;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if t < stride { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let scale = inverseSqrt(red[0] / f32(pc.dim) + pc.eps);
    for (var i: u32 = t; i < pc.dim; i = i + 64u) { y[base + i] = x[base + i] * scale * w[i]; }
}
"#;

pub(crate) const ROPE_WGSL: &str = r#"
struct PC { t: u32, nheads: u32, hd: u32, rope_dim: u32, theta: f32, pos_offset: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= pc.t * pc.nheads { return; }
    let pos = pc.pos_offset + idx / pc.nheads;
    let base = idx * pc.hd;
    for (var i: u32 = 0u; i < pc.hd; i = i + 1u) { y[base + i] = x[base + i]; }
    let half = pc.rope_dim / 2u;
    for (var i: u32 = 0u; i < half; i = i + 1u) {
        let freq = pow(pc.theta, -2.0 * f32(i) / f32(pc.rope_dim));
        let ang = f32(pos) * freq;
        let s = sin(ang);
        let co = cos(ang);
        let a = x[base + 2u * i];
        let b = x[base + 2u * i + 1u];
        y[base + 2u * i] = a * co - b * s;
        y[base + 2u * i + 1u] = a * s + b * co;
    }
}
"#;

pub(crate) const SILU_MUL_WGSL: &str = r#"
struct PC { n: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       up: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    let g = gate[i];
    y[i] = (g / (1.0 + exp(-g))) * up[i];
}
"#;

pub(crate) const ADD_WGSL: &str = r#"
struct PC { n: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    y[i] = a[i] + b[i];
}
"#;

/// Fused SwiGLU over a combined gate||up buffer `gu` `[rows, 2*nff]`:
/// `y[r,i] = silu(gu[r,i]) * gu[r, nff+i]`.
pub(crate) const SILU_MUL_FUSED_WGSL: &str = r#"
struct PC { rows: u32, nff: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       gu: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= pc.rows * pc.nff { return; }
    let r = idx / pc.nff;
    let i = idx % pc.nff;
    let base = r * 2u * pc.nff;
    let g = gu[base + i];
    y[idx] = (g / (1.0 + exp(-g))) * gu[base + pc.nff + i];
}
"#;

/// Fused FFN input: `act = SwiGLU(rmsnorm(hidden)·Wgu)`. ONE workgroup per output `act[f]`; its
/// 64 threads stride the K (ne) dimension so the weight row streams contiguously in lockstep
/// (coalesced — far better memory throughput than thread-per-output's 64 divergent streams).
/// The RMS scale is per-row and factored out: each thread accumulates `Σ hidden·nw·w` and
/// `Σ hidden²` over its K-slice, a tree-reduce sums them, then scale is applied once. Dispatch
/// `rows*nff` workgroups.
pub(crate) const FFN_IN_WGSL: &str = r#"
enable f16;
struct PC { rows: u32, ne: u32, nff: u32, eps: f32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       hidden: array<f32>; // [rows, ne]
@group(0) @binding(1) var<storage, read>       nw: array<f32>;     // [ne] rmsnorm weight
@group(0) @binding(2) var<storage, read>       wgu: array<f16>;    // [2*nff, ne] gate||up (f16)
@group(0) @binding(3) var<storage, read_write> act: array<f32>;    // [rows, nff]

var<workgroup> r_ss: array<f32, 64>;
var<workgroup> r_g: array<f32, 64>;
var<workgroup> r_u: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let unit = wid.x;              // = r * nff + f
    let f = unit % pc.nff;
    let r = unit / pc.nff;
    let rbase = r * pc.ne;
    let gbase = f * pc.ne;
    let ubase = (pc.nff + f) * pc.ne;

    var pss: f32 = 0.0;
    var pg: f32 = 0.0;
    var pu: f32 = 0.0;
    for (var k: u32 = t; k < pc.ne; k = k + 64u) {
        let hv = hidden[rbase + k];
        pss = pss + hv * hv;
        let hn = hv * nw[k];       // hidden * norm weight (RMS scale applied after the reduction)
        pg = pg + hn * f32(wgu[gbase + k]);
        pu = pu + hn * f32(wgu[ubase + k]);
    }
    r_ss[t] = pss;
    r_g[t] = pg;
    r_u[t] = pu;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if stride == 0u { break; }
        if t < stride {
            r_ss[t] = r_ss[t] + r_ss[t + stride];
            r_g[t] = r_g[t] + r_g[t + stride];
            r_u[t] = r_u[t] + r_u[t + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if t == 0u {
        let scale = inverseSqrt(r_ss[0] / f32(pc.ne) + pc.eps);
        let gate = r_g[0] * scale;
        let up = r_u[0] * scale;
        act[unit] = (gate / (1.0 + exp(-gate))) * up;
    }
}
"#;

/// Unified-quant variant of `FFN_IN_WGSL`: `wgu` repacked (u8 quants + per-16-block f16 scale/min).
/// Same cooperative-over-K structure + RMS-fold.
pub(crate) const FFN_IN_Q_WGSL: &str = r#"
enable f16;
struct PC { rows: u32, ne: u32, nff: u32, eps: f32, bits: u32, blk_shift: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       hidden: array<f32>;
@group(0) @binding(1) var<storage, read>       nw: array<f32>;
@group(0) @binding(2) var<storage, read>       quants: array<u32>; // [2*nff, ne]
@group(0) @binding(3) var<storage, read>       scales: array<f16>;
@group(0) @binding(4) var<storage, read>       mins: array<f16>;
@group(0) @binding(5) var<storage, read_write> act: array<f32>;

var<workgroup> r_ss: array<f32, 64>;
var<workgroup> r_g: array<f32, 64>;
var<workgroup> r_u: array<f32, 64>;

fn dq(g: u32) -> f32 {
    var q: f32;
    if pc.bits == 4u {
        q = f32((quants[g >> 3u] >> ((g & 7u) * 4u)) & 0xFu);
    } else {
        q = f32((quants[g >> 2u] >> ((g & 3u) * 8u)) & 0xFFu);
    }
    let blk = g >> pc.blk_shift;
    return f32(scales[blk]) * q + f32(mins[blk]);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let unit = wid.x;
    let f = unit % pc.nff;
    let r = unit / pc.nff;
    let rbase = r * pc.ne;
    let gbase = f * pc.ne;
    let ubase = (pc.nff + f) * pc.ne;
    var pss: f32 = 0.0;
    var pg: f32 = 0.0;
    var pu: f32 = 0.0;
    for (var k: u32 = t; k < pc.ne; k = k + 64u) {
        let hv = hidden[rbase + k];
        pss = pss + hv * hv;
        let hn = hv * nw[k];
        pg = pg + hn * dq(gbase + k);
        pu = pu + hn * dq(ubase + k);
    }
    r_ss[t] = pss; r_g[t] = pg; r_u[t] = pu;
    workgroupBarrier();
    var stride = 32u;
    loop { if stride == 0u { break; }
        if t < stride {
            r_ss[t] = r_ss[t] + r_ss[t + stride];
            r_g[t] = r_g[t] + r_g[t + stride];
            r_u[t] = r_u[t] + r_u[t + stride];
        }
        workgroupBarrier(); stride = stride / 2u; }
    if t == 0u {
        let scale = inverseSqrt(r_ss[0] / f32(pc.ne) + pc.eps);
        let gate = r_g[0] * scale;
        let up = r_u[0] * scale;
        act[unit] = (gate / (1.0 + exp(-gate))) * up;
    }
}
"#;

/// Fused RMSNorm + quant Q/K/V projection (Qwen3 decode): one workgroup per output column of the
/// `[q | k | v]` layout, folding RMSNorm into the dot (sum-of-squares + projection in one pass; the
/// per-row RMS scale factors out after the reduction — same trick as `ffn_in_q`). Writes the RAW
/// projections `qr`/`kr`/`vr` (QK-norm + RoPE follow). Replaces rmsnorm + 3× `linear_q` (4 dispatches
/// → 1). q/k/v share `bits`/`blk_shift` (same quant). Dispatch `rows * (q_dim + 2*kvrow)`.
pub(crate) const ATTN_IN_Q_WGSL: &str = r#"
enable f16;
// q/k/v can have DIFFERENT quant (Q4_K_M mixes Q4_K and Q6_K), so bits/blk_shift are per-region.
struct PC { rows: u32, ne: u32, q_dim: u32, kvrow: u32, eps: f32,
            qbits: u32, qblk: u32, kbits: u32, kblk: u32, vbits: u32, vblk: u32 }
var<immediate> pc: PC;
@group(0) @binding(0)  var<storage, read>       hidden: array<f32>;
@group(0) @binding(1)  var<storage, read>       nw: array<f32>;
@group(0) @binding(2)  var<storage, read>       wq: array<u32>;
@group(0) @binding(3)  var<storage, read>       sq: array<f16>;
@group(0) @binding(4)  var<storage, read>       mq: array<f16>;
@group(0) @binding(5)  var<storage, read>       wk: array<u32>;
@group(0) @binding(6)  var<storage, read>       sk: array<f16>;
@group(0) @binding(7)  var<storage, read>       mk: array<f16>;
@group(0) @binding(8)  var<storage, read>       wv: array<u32>;
@group(0) @binding(9)  var<storage, read>       sv: array<f16>;
@group(0) @binding(10) var<storage, read>       mv: array<f16>;
@group(0) @binding(11) var<storage, read_write> qr: array<f32>;  // [rows, q_dim]
@group(0) @binding(12) var<storage, read_write> kr: array<f32>;  // [rows, kvrow]
@group(0) @binding(13) var<storage, read_write> vr: array<f32>;  // [rows, kvrow]

var<workgroup> r_ss: array<f32, 64>;
var<workgroup> r_d: array<f32, 64>;

fn dqw(region: u32, g: u32) -> f32 {
    var q: u32;
    var s: f32;
    var m: f32;
    if region == 0u {
        if pc.qbits == 4u { q = (wq[g >> 3u] >> ((g & 7u) * 4u)) & 0xFu; } else { q = (wq[g >> 2u] >> ((g & 3u) * 8u)) & 0xFFu; }
        s = f32(sq[g >> pc.qblk]); m = f32(mq[g >> pc.qblk]);
    } else if region == 1u {
        if pc.kbits == 4u { q = (wk[g >> 3u] >> ((g & 7u) * 4u)) & 0xFu; } else { q = (wk[g >> 2u] >> ((g & 3u) * 8u)) & 0xFFu; }
        s = f32(sk[g >> pc.kblk]); m = f32(mk[g >> pc.kblk]);
    } else {
        if pc.vbits == 4u { q = (wv[g >> 3u] >> ((g & 7u) * 4u)) & 0xFu; } else { q = (wv[g >> 2u] >> ((g & 3u) * 8u)) & 0xFFu; }
        s = f32(sv[g >> pc.vblk]); m = f32(mv[g >> pc.vblk]);
    }
    return s * f32(q) + m;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let total = pc.q_dim + 2u * pc.kvrow;
    let unit = wid.x;
    let o = unit % total;
    let r = unit / total;
    // region 0=q,1=k,2=v and the weight-row index within that region
    var region: u32; var local: u32;
    if o < pc.q_dim { region = 0u; local = o; }
    else if o < pc.q_dim + pc.kvrow { region = 1u; local = o - pc.q_dim; }
    else { region = 2u; local = o - pc.q_dim - pc.kvrow; }
    let rbase = r * pc.ne;
    let gbase = local * pc.ne;
    var pss: f32 = 0.0;
    var pd: f32 = 0.0;
    for (var k: u32 = t; k < pc.ne; k = k + 64u) {
        let hv = hidden[rbase + k];
        pss = pss + hv * hv;
        pd = pd + hv * nw[k] * dqw(region, gbase + k);
    }
    r_ss[t] = pss; r_d[t] = pd;
    workgroupBarrier();
    var stride = 32u;
    loop { if stride == 0u { break; }
        if t < stride { r_ss[t] = r_ss[t] + r_ss[t + stride]; r_d[t] = r_d[t] + r_d[t + stride]; }
        workgroupBarrier(); stride = stride / 2u; }
    if t == 0u {
        let scale = inverseSqrt(r_ss[0] / f32(pc.ne) + pc.eps);
        let val = r_d[0] * scale;
        if region == 0u { qr[r * pc.q_dim + local] = val; }
        else if region == 1u { kr[r * pc.kvrow + local] = val; }
        else { vr[r * pc.kvrow + local] = val; }
    }
}
"#;

/// Fused attention input: RMSNorm(hidden) → Q/K/V projections → RoPE on Q,K, writing K/V into the
/// cache. ONE workgroup per output *pair* (columns 2p, 2p+1) of the packed `[q | k | v]` layout for
/// a row — its 64 threads stream the two weight rows over K in lockstep (coalesced), reduce, then
/// thread 0 applies RoPE (the pair coupling stays in-workgroup) and writes both. The RMS scale is
/// per-row and factored out after the reduction. Pairs align to RoPE (2i,2i+1) and stay within one
/// region since q_dim/kv_dim are multiples of hd (even). Dispatch `rows * (q_dim+2*kv_dim)/2`.
pub(crate) const ATTN_IN_WGSL: &str = r#"
enable f16;
struct PC { rows: u32, ne: u32, q_dim: u32, kv_dim: u32, hd: u32, rope_dim: u32, theta: f32, pos: u32, eps: f32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       hidden: array<f32>; // [rows, ne]
@group(0) @binding(1) var<storage, read>       nw: array<f32>;     // [ne]
@group(0) @binding(2) var<storage, read>       wq: array<f16>;     // [q_dim, ne] f16
@group(0) @binding(3) var<storage, read>       wk: array<f16>;     // [kv_dim, ne] f16
@group(0) @binding(4) var<storage, read>       wv: array<f16>;     // [kv_dim, ne] f16
@group(0) @binding(5) var<storage, read_write> q: array<f16>;      // [rows, q_dim] f16
@group(0) @binding(6) var<storage, read_write> kout: array<f16>;   // KV cache [ctx, kv_dim] f16
@group(0) @binding(7) var<storage, read_write> vout: array<f16>;   // KV cache [ctx, kv_dim] f16

var<workgroup> r_ss: array<f32, 64>;
var<workgroup> r_a: array<f32, 64>; // even column partial
var<workgroup> r_b: array<f32, 64>; // odd column partial

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let d = pc.q_dim + 2u * pc.kv_dim;
    let half = d / 2u;
    let unit = wid.x;            // = r * half + p
    let p = unit % half;
    let r = unit / half;
    let col = 2u * p;            // even column of this pair, in [0, d)
    let rbase = r * pc.ne;

    // weight base for the even/odd rows of this pair (region-dependent)
    var wbase_a: u32; var wbase_b: u32; var region: u32;
    if col < pc.q_dim {
        region = 0u; wbase_a = col * pc.ne;
    } else if col < pc.q_dim + pc.kv_dim {
        region = 1u; wbase_a = (col - pc.q_dim) * pc.ne;
    } else {
        region = 2u; wbase_a = (col - pc.q_dim - pc.kv_dim) * pc.ne;
    }
    wbase_b = wbase_a + pc.ne;

    var pss: f32 = 0.0; var pa: f32 = 0.0; var pb: f32 = 0.0;
    for (var k: u32 = t; k < pc.ne; k = k + 64u) {
        let hv = hidden[rbase + k];
        pss = pss + hv * hv;
        let hn = hv * nw[k];
        if region == 0u {
            pa = pa + hn * f32(wq[wbase_a + k]); pb = pb + hn * f32(wq[wbase_b + k]);
        } else if region == 1u {
            pa = pa + hn * f32(wk[wbase_a + k]); pb = pb + hn * f32(wk[wbase_b + k]);
        } else {
            pa = pa + hn * f32(wv[wbase_a + k]); pb = pb + hn * f32(wv[wbase_b + k]);
        }
    }
    r_ss[t] = pss; r_a[t] = pa; r_b[t] = pb;
    workgroupBarrier();
    var stride = 32u;
    loop { if stride == 0u { break; }
        if t < stride {
            r_ss[t] = r_ss[t] + r_ss[t + stride];
            r_a[t] = r_a[t] + r_a[t + stride];
            r_b[t] = r_b[t] + r_b[t + stride];
        }
        workgroupBarrier(); stride = stride / 2u; }

    if t == 0u {
        let scale = inverseSqrt(r_ss[0] / f32(pc.ne) + pc.eps);
        var a = r_a[0] * scale;
        var b = r_b[0] * scale;
        if region == 2u {
            let vc = col - pc.q_dim - pc.kv_dim;
            vout[(pc.pos + r) * pc.kv_dim + vc] = f16(a);
            vout[(pc.pos + r) * pc.kv_dim + vc + 1u] = f16(b);
        } else {
            // RoPE the (a,b) pair at within-head index ih (even)
            let local = select(col - pc.q_dim, col, region == 0u);
            let ih = local % pc.hd;
            var oa = a; var ob = b;
            if ih < pc.rope_dim {
                let freq = pow(pc.theta, -2.0 * f32(ih / 2u) / f32(pc.rope_dim));
                let ang = f32(pc.pos + r) * freq;
                let s = sin(ang); let co = cos(ang);
                oa = a * co - b * s;
                ob = a * s + b * co;
            }
            if region == 0u {
                q[r * pc.q_dim + col] = f16(oa);
                q[r * pc.q_dim + col + 1u] = f16(ob);
            } else {
                kout[(pc.pos + r) * pc.kv_dim + local] = f16(oa);
                kout[(pc.pos + r) * pc.kv_dim + local + 1u] = f16(ob);
            }
        }
    }
}
"#;

pub(crate) const ATTENTION_WGSL: &str = r#"
struct PC { t: u32, nh: u32, nkv: u32, hd: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= pc.t * pc.nh { return; }
    let ti = idx / pc.nh;
    let h = idx % pc.nh;
    let group = pc.nh / pc.nkv;
    let kvh = h / group;
    let hd = pc.hd;
    let scale = 1.0 / sqrt(f32(hd));
    let qbase = (ti * pc.nh + h) * hd;

    var acc: array<f32, 128>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { acc[d] = 0.0; }
    var m: f32 = -3.0e38;
    var l: f32 = 0.0;

    for (var j: u32 = 0u; j <= ti; j = j + 1u) {
        let kbase = (j * pc.nkv + kvh) * hd;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) { dot = dot + q[qbase + d] * k[kbase + d]; }
        let s = dot * scale;
        if s > m {
            let corr = exp(m - s);
            l = l * corr;
            for (var d: u32 = 0u; d < hd; d = d + 1u) { acc[d] = acc[d] * corr; }
            m = s;
        }
        let p = exp(s - m);
        l = l + p;
        let vbase = (j * pc.nkv + kvh) * hd;
        for (var d: u32 = 0u; d < hd; d = d + 1u) { acc[d] = acc[d] + p * v[vbase + d]; }
    }
    let obase = (ti * pc.nh + h) * hd;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { o[obase + d] = acc[d] / l; }
}
"#;

/// Cached GQA attention, TILED flash (one workgroup per (query token, head)). Streams the KV in
/// tiles of `TILE` so shared memory is O(TILE), not O(kv_len) — works for ANY context length
/// (bounded only by the KV-cache VRAM, not a fixed shared array). Per tile: scores into shared
/// (parallel over j, scalar max reduce), exponentiate (scalar sum reduce), then accumulate the
/// V-weighted sum parallel over the head dim (thread d owns d=t and d=t+64), online-rescaling the
/// running accumulator by exp(m_old - m_new). `hd<=128`. Used for prefill and short decode; long
/// single-token decode uses the split-K path instead. (Replaces the version that capped at 8192.)
pub(crate) const ATTENTION_KV_WGSL: &str = r#"
enable f16;
struct PC { q_len: u32, kv_len: u32, nh: u32, nkv: u32, hd: u32, pos_offset: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       q: array<f16>;  // q/k/v are f16
@group(0) @binding(1) var<storage, read>       k: array<f16>;
@group(0) @binding(2) var<storage, read>       v: array<f16>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;

const TILE: u32 = 1024u;
var<workgroup> q_sh: array<f32, 128>;
var<workgroup> sc: array<f32, 1024>; // one KV tile of scores/probabilities
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let unit = wid.x;            // one (token, head) per workgroup
    if unit >= pc.q_len * pc.nh { return; }
    let ti = unit / pc.nh;
    let h = unit % pc.nh;
    let kvlen = pc.pos_offset + ti + 1u; // causal: attend to positions [0, abs_pos]
    let kvh = h / (pc.nh / pc.nkv);
    let hd = pc.hd;
    let scale = 1.0 / sqrt(f32(hd));

    let qbase = (ti * pc.nh + h) * hd;
    for (var d: u32 = t; d < hd; d = d + 64u) { q_sh[d] = f32(q[qbase + d]); }
    workgroupBarrier();

    // running softmax state; acc held in registers (thread t owns output dims t and t+64)
    var m: f32 = -3.0e38;
    var l: f32 = 0.0;
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;

    var ts: u32 = 0u;
    loop {
        if ts >= kvlen { break; }
        var te = ts + TILE;
        if te > kvlen { te = kvlen; }

        // scores for this tile → sc[0..te-ts], local max
        var lmax: f32 = -3.0e38;
        for (var j: u32 = ts + t; j < te; j = j + 64u) {
            let kbase = (j * pc.nkv + kvh) * hd;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < hd; d = d + 1u) { dot = dot + q_sh[d] * f32(k[kbase + d]); }
            let s = dot * scale;
            sc[j - ts] = s;
            lmax = max(lmax, s);
        }
        red[t] = lmax;
        workgroupBarrier();
        var stride = 32u;
        loop { if stride == 0u { break; }
            if t < stride { red[t] = max(red[t], red[t + stride]); }
            workgroupBarrier(); stride = stride / 2u; }
        let mnew = max(m, red[0]);
        workgroupBarrier();

        // exp(s - mnew) → sc, local sum
        var lsum: f32 = 0.0;
        for (var j: u32 = ts + t; j < te; j = j + 64u) {
            let p = exp(sc[j - ts] - mnew);
            sc[j - ts] = p;
            lsum = lsum + p;
        }
        red[t] = lsum;
        workgroupBarrier();
        stride = 32u;
        loop { if stride == 0u { break; }
            if t < stride { red[t] = red[t] + red[t + stride]; }
            workgroupBarrier(); stride = stride / 2u; }
        let tsum = red[0];

        // online-rescale accumulator and add this tile's V contribution (parallel over d)
        let corr = exp(m - mnew);
        l = l * corr + tsum;
        m = mnew;
        if t < hd {
            var a: f32 = 0.0;
            for (var j: u32 = ts; j < te; j = j + 1u) { a = a + sc[j - ts] * f32(v[(j * pc.nkv + kvh) * hd + t]); }
            acc0 = acc0 * corr + a;
        }
        if t + 64u < hd {
            var a: f32 = 0.0;
            for (var j: u32 = ts; j < te; j = j + 1u) { a = a + sc[j - ts] * f32(v[(j * pc.nkv + kvh) * hd + t + 64u]); }
            acc1 = acc1 * corr + a;
        }
        workgroupBarrier(); // sc reused next tile
        ts = ts + TILE;
    }

    let obase = (ti * pc.nh + h) * hd;
    let inv = 1.0 / l;
    if t < hd { o[obase + t] = acc0 * inv; }
    if t + 64u < hd { o[obase + t + 64u] = acc1 * inv; }
}
"#;

/// Cast-copy f32 → f16 into `dst` at element offset `off` (for writing f32 activations into the
/// f16 KV cache). One thread per element.
pub(crate) const STORE_F16_WGSL: &str = r#"
enable f16;
struct PC { n: u32, off: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f16>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= pc.n { return; }
    dst[pc.off + gid.x] = f16(src[gid.x]);
}
"#;

/// Qwen3 QK-norm + RoPE: ONE workgroup per (token, head). RMS-normalizes the head's `hd`-vector
/// (weight `nw`) then applies ggml-NORM-interleaved RoPE. Reads raw projections `x[rows,nheads,hd]`,
/// writes to `y` at row `out_base + r` (q→0, k→cache pos). `hd<=128`.
pub(crate) const QK_NORM_ROPE_WGSL: &str = r#"
enable f16;
struct PC { rows: u32, nheads: u32, hd: u32, rope_dim: u32, theta: f32, rope_pos: u32, out_base: u32, eps: f32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       x: array<f32>;  // [rows, nheads, hd] raw projections
@group(0) @binding(1) var<storage, read>       nw: array<f32>; // [hd]
@group(0) @binding(2) var<storage, read_write> y: array<f16>;  // q buffer / KV cache (f16)

var<workgroup> xs: array<f32, 128>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let unit = wid.x;
    let h = unit % pc.nheads;
    let r = unit / pc.nheads;
    let in_base = (r * pc.nheads + h) * pc.hd;

    var pss: f32 = 0.0;
    for (var i: u32 = t; i < pc.hd; i = i + 64u) { let v = x[in_base + i]; xs[i] = v; pss = pss + v * v; }
    red[t] = pss;
    workgroupBarrier();
    var stride = 32u;
    loop { if stride == 0u { break; }
        if t < stride { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier(); stride = stride / 2u; }
    let scale = inverseSqrt(red[0] / f32(pc.hd) + pc.eps);

    let abs = pc.rope_pos + r;
    let out_base_idx = ((pc.out_base + r) * pc.nheads + h) * pc.hd;
    // NEOX-style RoPE (Qwen): pair element p with p + rope_dim/2. (Assumes rope_dim == hd, true for
    // Qwen3.) RMSNorm weight is applied per element before rotation.
    let half = pc.rope_dim / 2u;
    for (var p: u32 = t; p < half; p = p + 64u) {
        let i0 = p;
        let i1 = p + half;
        let a = xs[i0] * scale * nw[i0];
        let b = xs[i1] * scale * nw[i1];
        let freq = pow(pc.theta, -2.0 * f32(p) / f32(pc.rope_dim));
        let ang = f32(abs) * freq;
        let s = sin(ang);
        let co = cos(ang);
        y[out_base_idx + i0] = f16(a * co - b * s);
        y[out_base_idx + i1] = f16(a * s + b * co);
    }
}
"#;

/// Flash-decoding pass 1 (split-K): ONE workgroup per (head, KV-chunk). Computes that chunk's
/// softmax partial — max `m`, sum `l`, and un-normalized weighted-V `acc[hd]` (relative to `m`) —
/// for the single decode query. Many chunks → many workgroups → the GPU stays busy at long
/// context (the non-split kernel used only `nh` workgroups). Combine merges them. `hd<=128`,
/// `chunk<=1024`. q is `[nh, hd]` (q_len==1).
// Flash-decoding pass 1 (split-K) is now a GLSL subgroup-reduction kernel: shaders/attn_partial.comp
// (the old thread-per-key WGSL version had uncoalesced K reads that dominated long-context decode).

/// Flash-decoding pass 2 (combine): ONE workgroup per head. Merges the `n_chunks` partials via the
/// online-softmax rule (`M=max mₖ; l=Σ lₖ·e^{mₖ−M}; acc=Σ accₖ·e^{mₖ−M}; o=acc/l`), parallel over
/// the head dim. `n_chunks<=64`.
pub(crate) const ATTN_COMBINE_WGSL: &str = r#"
struct PC { nh: u32, hd: u32, n_chunks: u32, ntile: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       pm: array<f32>;   // [nh, n_chunks]
@group(0) @binding(1) var<storage, read>       pl: array<f32>;   // [nh, n_chunks]
@group(0) @binding(2) var<storage, read>       pacc: array<f32>; // [nh, n_chunks, hd]
@group(0) @binding(3) var<storage, read_write> o: array<f32>;    // [nh, hd]

var<workgroup> wexp: array<f32, 512>; // exp(pm[c]-max) per chunk, precomputed once and reused

// One workgroup per (head, hd-tile): nh workgroups starved a 96-CU GPU (combine was latency-bound),
// so split each head's hd outputs across `ntile` workgroups for occupancy. m/l are cheap to recompute
// per tile (a scan over n_chunks), which avoids any cross-workgroup reduction.
@compute @workgroup_size(32, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let h = wid.x / pc.ntile;
    let tile = wid.x % pc.ntile;
    let hd = pc.hd;
    let hdt = hd / pc.ntile;
    let d0 = tile * hdt;
    let t = lid.x;
    let base = h * pc.n_chunks;
    var mm: f32 = -3.0e38;
    for (var c: u32 = 0u; c < pc.n_chunks; c = c + 1u) { mm = max(mm, pm[base + c]); }
    for (var c: u32 = t; c < pc.n_chunks; c = c + 32u) { wexp[c] = exp(pm[base + c] - mm); }
    workgroupBarrier();
    var l: f32 = 0.0;
    for (var c: u32 = 0u; c < pc.n_chunks; c = c + 1u) { l = l + pl[base + c] * wexp[c]; }
    let inv = 1.0 / l;
    for (var d: u32 = d0 + t; d < d0 + hdt; d = d + 32u) {
        var acc: f32 = 0.0;
        for (var c: u32 = 0u; c < pc.n_chunks; c = c + 1u) {
            acc = acc + pacc[(base + c) * hd + d] * wexp[c];
        }
        o[h * hd + d] = acc * inv;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn rmsnorm_matches_host() {
        let be = VulkanBackend::new().unwrap();
        let (rows, dim, eps) = (3usize, 8usize, 1e-5f32);
        let x: Vec<f32> = (0..rows * dim).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + i as f32 * 0.05).collect();
        let got = be.rmsnorm(&x, &w, rows, dim, eps).unwrap();
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let scale = 1.0 / (ms + eps).sqrt();
            for i in 0..dim {
                let want = row[i] * scale * w[i];
                assert!(
                    (got[r * dim + i] - want).abs() < 1e-4,
                    "{} vs {}",
                    got[r * dim + i],
                    want
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn rope_matches_host() {
        let be = VulkanBackend::new().unwrap();
        let (t, nh, hd, rd, theta) = (4usize, 2usize, 8usize, 8usize, 10000.0f32);
        let x: Vec<f32> = (0..t * nh * hd).map(|i| (i as f32) * 0.03).collect();
        let got = be.rope(&x, t, nh, hd, rd, theta).unwrap();
        // host reference (ggml NORM interleaved)
        let mut want = x.clone();
        for pos in 0..t {
            for h in 0..nh {
                let base = (pos * nh + h) * hd;
                for i in 0..rd / 2 {
                    let freq = (theta as f64).powf(-2.0 * i as f64 / rd as f64) as f32;
                    let ang = pos as f32 * freq;
                    let (s, co) = ang.sin_cos();
                    let a = x[base + 2 * i];
                    let b = x[base + 2 * i + 1];
                    want[base + 2 * i] = a * co - b * s;
                    want[base + 2 * i + 1] = a * s + b * co;
                }
            }
        }
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "{g} vs {w}");
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn silu_mul_matches_host() {
        let be = VulkanBackend::new().unwrap();
        let n = 100usize;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 2.0).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let got = be.silu_mul(&gate, &up, n).unwrap();
        for i in 0..n {
            let g = gate[i];
            let want = (g / (1.0 + (-g).exp())) * up[i];
            assert!((got[i] - want).abs() < 1e-4, "{} vs {}", got[i], want);
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn add_matches_host() {
        let be = VulkanBackend::new().unwrap();
        let n = 100usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let got = be.add(&a, &b, n).unwrap();
        for i in 0..n {
            assert!((got[i] - (a[i] + b[i])).abs() < 1e-4);
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_matches_host() {
        let be = VulkanBackend::new().unwrap();
        let (t, nh, nkv, hd) = (5usize, 4usize, 2usize, 8usize);
        let q: Vec<f32> = (0..t * nh * hd).map(|i| (i as f32 * 0.013).sin()).collect();
        let k: Vec<f32> = (0..t * nkv * hd)
            .map(|i| (i as f32 * 0.017).cos())
            .collect();
        let v: Vec<f32> = (0..t * nkv * hd).map(|i| (i as f32 * 0.011)).collect();
        let got = be.attention(&q, &k, &v, t, nh, nkv, hd).unwrap();

        // host reference: causal GQA with standard softmax
        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        let mut want = vec![0f32; t * nh * hd];
        for ti in 0..t {
            for h in 0..nh {
                let kvh = h / group;
                let qv = &q[(ti * nh + h) * hd..(ti * nh + h) * hd + hd];
                let mut scores = vec![0f32; ti + 1];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=ti {
                    let kv = &k[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                    let mut d = 0f32;
                    for x in 0..hd {
                        d += qv[x] * kv[x];
                    }
                    d *= scale;
                    scores[j] = d;
                    mx = mx.max(d);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let ob = (ti * nh + h) * hd;
                for j in 0..=ti {
                    let p = scores[j] / sum;
                    let vv = &v[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                    for x in 0..hd {
                        want[ob + x] += p * vv[x];
                    }
                }
            }
        }
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "{g} vs {w}");
        }
    }
}
