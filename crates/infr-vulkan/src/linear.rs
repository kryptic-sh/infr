//! Persistent-weight linear layer: `y = W · x` where `W` is stored `[out, in]` row-major
//! (the GGUF layout: data index `o*in + i`). The weight buffer is uploaded once
//! (`upload_weight`) and reused; the compute pipeline is built once (cached in
//! `VulkanShared.linear_kernel`) and reused across all calls — only the (small) activation
//! buffers are created per call.
//!
//! WGSL → SPIR-V via naga, same pattern as `matmul.rs`.

use std::ffi::CStr;
use std::sync::OnceLock;

use ash::vk;

use infr_core::{
    backend::{Buffer, BufferUsage},
    error::Result,
    Backend,
};

use super::{as_vk_buf, be, VulkanBackend};

pub(crate) const LINEAR_WGSL: &str = r#"
struct PushConstants { rows: u32, in_f: u32, out_f: u32 }
var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read>       w_buf: array<f32>; // [out, in]  (w[o*in+i])
@group(0) @binding(1) var<storage, read>       x_buf: array<f32>; // [rows, in]
@group(0) @binding(2) var<storage, read_write> y_buf: array<f32>; // [rows, out]

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = pc.rows * pc.out_f;
    if idx >= total { return; }
    let r = idx / pc.out_f;
    let o = idx % pc.out_f;
    let wbase = o * pc.in_f;
    let xbase = r * pc.in_f;
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < pc.in_f; i = i + 1u) {
        acc = acc + w_buf[wbase + i] * x_buf[xbase + i];
    }
    y_buf[r * pc.out_f + o] = acc;
}
"#;

/// Like `LINEAR_WGSL` but adds a residual: `y = residual + x·Wᵀ`. `r_buf` and `y_buf` may alias
/// (in-place residual): each invocation reads and writes only index `idx`, so it is safe.
pub(crate) const LINEAR_RES_WGSL: &str = r#"
struct PushConstants { rows: u32, in_f: u32, out_f: u32 }
var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read>       w_buf: array<f32>; // [out, in]
@group(0) @binding(1) var<storage, read>       x_buf: array<f32>; // [rows, in]
@group(0) @binding(2) var<storage, read>       r_buf: array<f32>; // [rows, out] residual
@group(0) @binding(3) var<storage, read_write> y_buf: array<f32>; // [rows, out]

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = pc.rows * pc.out_f;
    if idx >= total { return; }
    let r = idx / pc.out_f;
    let o = idx % pc.out_f;
    let wbase = o * pc.in_f;
    let xbase = r * pc.in_f;
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < pc.in_f; i = i + 1u) {
        acc = acc + w_buf[wbase + i] * x_buf[xbase + i];
    }
    y_buf[idx] = r_buf[idx] + acc;
}
"#;

static LINEAR_SPV: OnceLock<Vec<u32>> = OnceLock::new();

fn linear_spv() -> &'static [u32] {
    LINEAR_SPV.get_or_init(|| {
        use naga::back::spv;
        use naga::front::wgsl;
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let module = wgsl::parse_str(LINEAR_WGSL).expect("linear WGSL parse");
        let info = Validator::new(ValidationFlags::all(), Capabilities::IMMEDIATES)
            .validate(&module)
            .expect("linear WGSL validate");
        spv::write_vec(
            &module,
            &info,
            &spv::Options {
                lang_version: (1, 3),
                ..Default::default()
            },
            None,
        )
        .expect("linear SPIR-V write")
    })
}

/// Cached, reusable compute objects for the linear kernel (built once per device).
pub(crate) struct LinearKernel {
    pub shader: vk::ShaderModule,
    pub ds_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub desc_pool: vk::DescriptorPool,
}

pub(crate) fn create_linear_kernel(device: &ash::Device) -> LinearKernel {
    let spv = linear_spv();
    let shader = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None)
    }
    .expect("create linear shader module");

    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
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
    .expect("create linear ds layout");

    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(12);
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&ds_layout))
                .push_constant_ranges(std::slice::from_ref(&push_range)),
            None,
        )
    }
    .expect("create linear pipeline layout");

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
            .expect("create linear pipeline")[0]
    };

    // Pool holds one set; we reset + reallocate it each call (single-stream gen).
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 3,
    }];
    let desc_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
    }
    .expect("create linear desc pool");

    LinearKernel {
        shader,
        ds_layout,
        pipeline_layout,
        pipeline,
        desc_pool,
    }
}

pub(crate) fn destroy_linear_kernel(device: &ash::Device, k: &LinearKernel) {
    unsafe {
        device.destroy_descriptor_pool(k.desc_pool, None);
        device.destroy_pipeline(k.pipeline, None);
        device.destroy_pipeline_layout(k.pipeline_layout, None);
        device.destroy_descriptor_set_layout(k.ds_layout, None);
        device.destroy_shader_module(k.shader, None);
    }
}

impl VulkanBackend {
    fn linear_kernel(&self) -> &LinearKernel {
        self.shared
            .linear_kernel
            .get_or_init(|| create_linear_kernel(&self.shared.device))
    }

    /// Upload an `[out, in]` f32 weight to a persistent device buffer.
    pub fn upload_weight(&self, data: &[f32]) -> Result<Box<dyn Buffer>> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let buf = self.alloc(bytes.len(), BufferUsage::Weights)?;
        self.upload(buf.as_ref(), bytes)?;
        Ok(buf)
    }

    /// Compute `y[rows, out] = x[rows, in] · Wᵀ` where `w_buf` holds `W[out, in]`.
    /// Reuses the cached pipeline; only the per-call x/y buffers + descriptor set are fresh.
    pub fn linear(
        &self,
        w_buf: &dyn Buffer,
        x: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>> {
        assert_eq!(x.len(), rows * in_f, "x must be rows*in");
        let device = self.shared.device.clone();
        let k = self.linear_kernel();

        // fresh descriptor set from the cached pool
        unsafe {
            device
                .reset_descriptor_pool(k.desc_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| be(format!("reset_descriptor_pool: {e}")))?;
        }
        let desc_set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(k.desc_pool)
                        .set_layouts(std::slice::from_ref(&k.ds_layout)),
                )
                .map_err(|e| be(format!("allocate_descriptor_sets: {e}")))?[0]
        };

        // Host-visible activation buffers: upload/download become direct memcpy (no extra
        // submit+wait), leaving the dispatch as the only GPU round-trip in this call.
        let x_bytes: &[u8] = bytemuck::cast_slice(x);
        let buf_x = self.alloc(x_bytes.len(), BufferUsage::Staging)?;
        let buf_y = self.alloc(rows * out_f * 4, BufferUsage::Readback)?;
        self.upload(buf_x.as_ref(), x_bytes)?;

        let vk_w = unsafe { as_vk_buf(w_buf) }.buffer;
        let vk_x = unsafe { as_vk_buf(buf_x.as_ref()) }.buffer;
        let vk_y = unsafe { as_vk_buf(buf_y.as_ref()) }.buffer;

        let infos = [
            vk::DescriptorBufferInfo {
                buffer: vk_w,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vk_x,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vk_y,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
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

        let groups = ((rows * out_f) as u32).div_ceil(64);
        let shared = std::sync::Arc::clone(&self.shared);
        let (pipeline, pipeline_layout) = (k.pipeline, k.pipeline_layout);
        self.one_shot(move |cmd| unsafe {
            let barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(vk_x)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            shared.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
            shared
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            shared.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            shared.device.cmd_push_constants(
                cmd,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            shared.device.cmd_dispatch(cmd, groups, 1, 1);
        })?;

        let mut y_bytes = vec![0u8; rows * out_f * 4];
        self.download(buf_y.as_ref(), &mut y_bytes)?;
        Ok(bytemuck::cast_slice(&y_bytes).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (rows, in_f, out_f) = (3usize, 5usize, 4usize);
        let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32) * 0.01).collect();
        let x: Vec<f32> = (0..rows * in_f).map(|i| (i as f32) * 0.02).collect();
        let wbuf = be.upload_weight(&w).unwrap();
        // run twice to exercise the cached pipeline path
        let _ = be.linear(wbuf.as_ref(), &x, rows, in_f, out_f).unwrap();
        let y = be.linear(wbuf.as_ref(), &x, rows, in_f, out_f).unwrap();
        let mut want = vec![0.0f32; rows * out_f];
        for r in 0..rows {
            for o in 0..out_f {
                let mut acc = 0.0;
                for i in 0..in_f {
                    acc += x[r * in_f + i] * w[o * in_f + i];
                }
                want[r * out_f + o] = acc;
            }
        }
        for (g, w) in y.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-3, "{g} vs {w}");
        }
    }
}
