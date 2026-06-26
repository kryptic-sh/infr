//! GPU compute kernels for the transformer ops the Llama forward needs (RMSNorm, RoPE, …),
//! plus a generic cached-kernel runner. Each op is validated against the host reference in
//! `infr-llama` (the working oracle).
//!
//! These eager runners (one submit each) are for validation; the single-command-buffer
//! resident forward (real speedup) reuses the same kernels via record APIs (added next).

use std::ffi::CStr;

use ash::vk;

use infr_core::{backend::BufferUsage, error::Result, Backend};

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
    let info = Validator::new(ValidationFlags::all(), Capabilities::IMMEDIATES)
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
    let spv = compile_wgsl(wgsl);
    let shader = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
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
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(entry);
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
        self.run_kernel(k, &[x, w], rows * dim, &push, (rows as u32).div_ceil(64))
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
        let k = self.kernel("rope", ROPE_WGSL, 2, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
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

pub(crate) const RMSNORM_WGSL: &str = r#"
struct PC { rows: u32, dim: u32, eps: f32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= pc.rows { return; }
    let base = r * pc.dim;
    var ss: f32 = 0.0;
    for (var i: u32 = 0u; i < pc.dim; i = i + 1u) { let v = x[base + i]; ss = ss + v * v; }
    let scale = 1.0 / sqrt(ss / f32(pc.dim) + pc.eps);
    for (var i: u32 = 0u; i < pc.dim; i = i + 1u) { y[base + i] = x[base + i] * scale * w[i]; }
}
"#;

pub(crate) const ROPE_WGSL: &str = r#"
struct PC { t: u32, nheads: u32, hd: u32, rope_dim: u32, theta: f32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= pc.t * pc.nheads { return; }
    let pos = idx / pc.nheads;
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
