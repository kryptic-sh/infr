//! GPU compute kernels for the transformer ops the Llama forward needs (RMSNorm, RoPE, …),
//! plus a generic cached-kernel runner. Each op is validated against the host reference in
//! `infr-llama` (the working oracle).
//!
//! These eager runners (one submit each) are for validation; the single-command-buffer
//! resident forward (real speedup) reuses the same kernels via record APIs (added next).

use ash::vk;

use infr_core::{backend::BufferUsage, error::Result, Backend};

use super::{as_vk_buf, be, VulkanBackend};

/// Cached, reusable compute objects for one kernel. All fields are Vulkan handles (Copy).
#[derive(Clone, Copy)]
pub(crate) struct ComputeKernel {
    /// The cache key passed to [`VulkanBackend::kernel`] — the INFR_PROF_OPS auto-label: every
    /// recorder dispatch stamps its timestamp with this name (no manual stamp calls).
    pub name: &'static str,
    pub shader: vk::ShaderModule,
    pub ds_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub desc_pool: vk::DescriptorPool,
    pub n_buf: usize,
    pub push_size: u32,
    /// Hash of the SPIR-V this kernel was built from. The cache is keyed on `name` alone, so this
    /// lets [`VulkanBackend::kernel`]/[`VulkanBackend::kernel_sg`] debug-assert on a cache hit that
    /// the request's shader/layout matches the built one — a name reused with a different
    /// `spv`/`n_buf`/`push_size` would otherwise silently return a mismatched pipeline.
    pub spv_hash: u64,
    /// The subgroup pin the pipeline was created with, so a `DISPATCH_BASE` twin
    /// ([`VulkanBackend::dispatch_base_pipeline`]) is created the same way.
    pub pin: Option<SubgroupPin>,
    /// The shader declares `gl_NumWorkGroups` (`spirv::reads_num_workgroups`): its grid may then
    /// be cut into ranges only under a convention the op states (`work::Split::WideGrid`).
    pub reads_num_workgroups: bool,
}

/// Cheap non-cryptographic hash of a kernel's SPIR-V words — only used to catch a `name` reused
/// with a different shader on a cache hit (see [`ComputeKernel::spv_hash`]). Computed once per
/// build; the request side is only hashed inside the `debug_assert!`, so release builds skip it.
fn hash_spv(spv: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    spv.hash(&mut h);
    h.finish()
}

/// `constant_id` of `W64` in `shaders/wave64.glsl`.
pub(crate) const WAVE64_SPEC_ID: u32 = 1000;

/// The subgroup width the wave64 path pins every pipeline to.
const WAVE64_WIDTH: u32 = 64;

/// How a pipeline's subgroup is pinned: `requiredSubgroupSize`, and whether it is created with
/// `REQUIRE_FULL_SUBGROUPS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SubgroupPin {
    pub size: u32,
    pub full: bool,
}

/// The [`SubgroupPin`] for a kernel that asked for `required_sg`, or `None` to leave the width to
/// the driver.
///
/// On the wave64 path every pipeline pins [`WAVE64_WIDTH`], whatever it asked for, and asks for
/// full subgroups wherever the spec allows it (a literal `local_size_x` that is a multiple of 64,
/// VUID-02759): there every subgroup holds 64 invocations, which `wave64.glsl` splits into two
/// 32-lane halves. A workgroup of at most 32 invocations cannot ask for it; `wave64.glsl` runs such
/// a workgroup as one subgroup, which `VulkanBackend::new_with` checks once per device
/// (`caps::check_wave64_layout`).
pub(crate) fn subgroup_pin(
    required_sg: Option<u32>,
    wave64: bool,
    local_size: Option<[u32; 3]>,
) -> Option<SubgroupPin> {
    if wave64 {
        return Some(SubgroupPin {
            size: WAVE64_WIDTH,
            full: local_size.is_some_and(|l| l[0] % WAVE64_WIDTH == 0),
        });
    }
    required_sg.map(|size| SubgroupPin { size, full: true })
}

/// Build one kernel's Vulkan objects, returning an error instead of aborting the process.
///
/// The kernel cache (`VulkanBackend::kernel_inner`) turns an error into a panic, after releasing
/// its lock. The fallible surface exists for capability PROBES: a probe dispatched to find out
/// whether a device can honour a tier must be able to answer "no" (a driver that returns a null
/// pipeline for an unsupported coopmat config is exactly the case being probed for), and a panic
/// there would turn a refused tier into a crashed process.
#[cfg_attr(infr_profile, infr_prof::instrument)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_make_compute_kernel(
    device: &ash::Device,
    pcache: vk::PipelineCache,
    name: &'static str,
    spv: &[u32],
    n_buf: usize,
    push_size: u32,
    required_sg: Option<u32>,
    push_descriptor: bool,
    limits: crate::caps::KernelLimits,
    rte_f16: bool,
    wave64: bool,
) -> Result<ComputeKernel> {
    // The device's real `maxPushConstantsSize` — Vulkan only GUARANTEES 128 bytes, and every push
    // block here was sized against that floor by inspection. Checked before the layout is built so
    // an oversize block names itself instead of surfacing as a driver-side VUID.
    crate::caps::check_push_constant_size(name, push_size, limits.push_constants).map_err(be)?;
    // Same for the workgroup size baked into the module.
    crate::caps::check_kernel_workgroup(name, spv, &limits).map_err(be)?;
    // `spv` itself stays the kernel's identity (`spv_hash` below): the patch is a function of the
    // module and the device, so the same `spv` always builds the same pipeline here.
    let rte_patched = rte_f16.then(|| crate::spirv::with_rte_f16(spv)).flatten();
    let code = rte_patched.as_deref().unwrap_or(spv);
    let shader = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)
    }
    .map_err(|e| {
        be(format!(
            "create_shader_module failed for kernel {name:?}: {e} — recoverable driver/alloc \
             failure (e.g. OUT_OF_DEVICE_MEMORY) during kernel compile"
        ))
    })?;

    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_buf)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    // PUSH_DESCRIPTOR_KHR: this set is bound via `cmd_push_descriptor_set` (recorder.rs /
    // ops.rs `run_kernel`), never `vkAllocateDescriptorSets` — the two are mutually exclusive
    // per the Vulkan spec, so the flag must match how every call site below binds it.
    let mut ds_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    if push_descriptor {
        ds_ci = ds_ci.flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
    }
    let ds_layout = match unsafe { device.create_descriptor_set_layout(&ds_ci, None) } {
        Ok(l) => l,
        Err(e) => {
            unsafe { device.destroy_shader_module(shader, None) };
            return Err(be(format!(
                "create_descriptor_set_layout failed for kernel {name:?}: {e} — recoverable \
                 driver/alloc failure (e.g. OUT_OF_DEVICE_MEMORY) during kernel compile"
            )));
        }
    };

    let mut plinfo =
        vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&ds_layout));
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_size);
    if push_size > 0 {
        plinfo = plinfo.push_constant_ranges(std::slice::from_ref(&push_range));
    }
    let pipeline_layout = match unsafe { device.create_pipeline_layout(&plinfo, None) } {
        Ok(l) => l,
        Err(e) => {
            unsafe {
                device.destroy_descriptor_set_layout(ds_layout, None);
                device.destroy_shader_module(shader, None);
            }
            return Err(be(format!(
                "create_pipeline_layout failed for kernel {name:?}: {e} — recoverable driver/alloc \
                 failure (e.g. OUT_OF_DEVICE_MEMORY) during kernel compile"
            )));
        }
    };

    let pin = subgroup_pin(required_sg, wave64, crate::spirv::local_size(spv));
    // Everything built so far, for the error paths below (nothing else may return early past here
    // without releasing them).
    let undo = |device: &ash::Device| unsafe {
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(ds_layout, None);
        device.destroy_shader_module(shader, None);
    };
    let pipeline = match create_pipeline(
        device,
        pcache,
        shader,
        pipeline_layout,
        pin,
        wave64,
        vk::PipelineCreateFlags::empty(),
    ) {
        Ok(p) => p,
        Err(e) => {
            undo(device);
            return Err(be(format!(
                "create_compute_pipelines failed for kernel {name:?}: {e} — a device whose \
                 coopmat/subgroup config doesn't match this kernel should have been filtered out \
                 by capability detection before reaching here"
            )));
        }
    };
    // A driver can return VK_SUCCESS with a VK_NULL_HANDLE pipeline (observed as the root cause of
    // a segfault on Intel/Mesa ANV: a coopmat pipeline whose tile size the device doesn't support).
    // Binding/dispatching that handle later is the actual crash — fail loudly HERE instead, with
    // the offending kernel named, rather than deref garbage downstream.
    if pipeline == vk::Pipeline::null() {
        undo(device);
        return Err(be(format!(
            "create_compute_pipelines returned VK_SUCCESS with a null pipeline handle for kernel \
             {name:?} — likely an unsupported coopmat/subgroup config that slipped past capability \
             detection"
        )));
    }

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: n_buf as u32,
    }];
    let desc_pool = match unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
    } {
        Ok(p) => p,
        Err(e) => {
            unsafe { device.destroy_pipeline(pipeline, None) };
            undo(device);
            return Err(be(format!(
                "create_descriptor_pool failed for kernel {name:?}: {e} — recoverable driver/alloc \
                 failure (e.g. OUT_OF_DEVICE_MEMORY) during kernel compile"
            )));
        }
    };

    Ok(ComputeKernel {
        name,
        shader,
        ds_layout,
        pipeline_layout,
        pipeline,
        desc_pool,
        n_buf,
        push_size,
        spv_hash: hash_spv(spv),
        pin,
        reads_num_workgroups: crate::spirv::reads_num_workgroups(spv),
    })
}

/// Create one compute pipeline of `shader` under `layout`: the wave64 specialization when
/// `wave64`, the subgroup `pin`, and `flags`. Shared by a kernel's own pipeline and its
/// `DISPATCH_BASE` twin, so the two can only differ by `flags`.
fn create_pipeline(
    device: &ash::Device,
    pcache: vk::PipelineCache,
    shader: vk::ShaderModule,
    layout: vk::PipelineLayout,
    pin: Option<SubgroupPin>,
    wave64: bool,
    flags: vk::PipelineCreateFlags,
) -> std::result::Result<vk::Pipeline, vk::Result> {
    let entry = c"main";
    let mut req_sz =
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(0);
    let mut stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(entry);
    let w64_entry = vk::SpecializationMapEntry::default()
        .constant_id(WAVE64_SPEC_ID)
        .offset(0)
        .size(std::mem::size_of::<vk::Bool32>());
    let w64_data = vk::TRUE.to_ne_bytes();
    let w64_spec = vk::SpecializationInfo::default()
        .map_entries(std::slice::from_ref(&w64_entry))
        .data(&w64_data);
    if wave64 {
        stage = stage.specialization_info(&w64_spec);
    }
    if let Some(pin) = pin {
        req_sz = req_sz.required_subgroup_size(pin.size);
        if pin.full {
            stage = stage.flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
        }
        stage = stage.push_next(&mut req_sz);
    }
    unsafe {
        device.create_compute_pipelines(
            pcache, // disk-persisted device cache (see pcache.rs); null = caching off
            &[vk::ComputePipelineCreateInfo::default()
                .flags(flags)
                .stage(stage)
                .layout(layout)],
            None,
        )
    }
    .map(|p| p[0])
    .map_err(|(_, e)| e)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn destroy_compute_kernel(device: &ash::Device, k: &ComputeKernel) {
    unsafe {
        device.destroy_descriptor_pool(k.desc_pool, None);
        device.destroy_pipeline(k.pipeline, None);
        device.destroy_pipeline_layout(k.pipeline_layout, None);
        device.destroy_descriptor_set_layout(k.ds_layout, None);
        device.destroy_shader_module(k.shader, None);
    }
}

/// Set-0 descriptor binding for the one-shot eager runners (`run_kernel` and
/// gemm.rs's eager/bench GEMMs). A [`ComputeKernel`]'s set layout carries PUSH_DESCRIPTOR_KHR
/// whenever the device has `VK_KHR_push_descriptor`, and such a layout must never be passed to
/// `vkAllocateDescriptorSets` (VUID-VkDescriptorSetAllocateInfo-pSetLayouts-00308) — so this
/// pushes the set inside the command buffer in that case and only falls back to a pooled classic
/// set (allocated up front from the kernel's pool) when the extension, and thus the flag, is
/// absent. Build with [`VulkanBackend::eager_bind`]; record with [`EagerBind::bind`].
pub(crate) struct EagerBind {
    /// `Some` iff the device lacks push descriptors (the classic-bind fallback).
    pooled_set: Option<vk::DescriptorSet>,
    /// One STORAGE_BUFFER info per binding 0..n_buf; owned here so the write structs built in
    /// [`Self::writes`] stay valid for the synchronous `one_shot` submit.
    infos: Vec<vk::DescriptorBufferInfo>,
}

impl EagerBind {
    fn writes(&self) -> Vec<vk::WriteDescriptorSet<'_>> {
        self.infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                let w = vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info));
                match self.pooled_set {
                    Some(set) => w.dst_set(set),
                    None => w, // push descriptors: dst_set is ignored by the spec
                }
            })
            .collect()
    }

    /// Record the set-0 binding into the one-shot command buffer (push or classic, per
    /// construction).
    ///
    /// # Safety
    /// `cmd` must be in the recording state and `layout` must be `k.pipeline_layout` of the
    /// kernel this binding was built for.
    pub(crate) unsafe fn bind(
        &self,
        shared: &super::VulkanShared,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
    ) {
        match (&shared.push_descriptor, self.pooled_set) {
            (Some(pd), _) => pd.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &self.writes(),
            ),
            (None, Some(set)) => shared.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            ),
            (None, None) => unreachable!("pooled_set is Some whenever push_descriptor is None"),
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanBackend {
    /// Fetch-or-build a named kernel from precompiled SPIR-V (build-compiled GLSL → `.spv`).
    pub(crate) fn kernel(
        &self,
        name: &'static str,
        spv: &[u32],
        n_buf: usize,
        push_size: u32,
    ) -> ComputeKernel {
        self.kernel_inner(name, spv, n_buf, push_size, None)
    }

    /// Like `kernel`, but pins the pipeline's subgroup size (coopmat needs wave32 on RDNA3).
    /// A size of `0` means "no subgroup requirement" (treated like [`Self::kernel`]) — a caller
    /// that passes 0 by mistake would otherwise build an invalid `RequiredSubgroupSizeCreateInfo`.
    pub(crate) fn kernel_sg(
        &self,
        name: &'static str,
        spv: &[u32],
        n_buf: usize,
        push_size: u32,
        sg_size: u32,
    ) -> ComputeKernel {
        self.kernel_inner(
            name,
            spv,
            n_buf,
            push_size,
            (sg_size > 0).then_some(sg_size),
        )
    }

    /// Shared fetch-or-build for [`Self::kernel`]/[`Self::kernel_sg`]. The build runs UNDER the
    /// cache lock so exactly one thread compiles a given `name` on concurrent first use — the old
    /// lock→get→unlock→compile→lock→insert had two racing threads both build the full
    /// shader+pipeline+pool and the second `insert` overwrite (and leak for process life) the
    /// first's Vulkan objects.
    ///
    /// A build that fails still panics (this surface is infallible until B98 threads a `Result`
    /// through), but only after the lock is released: a panic while holding it would poison the
    /// cache, and every later kernel request in the process would panic on the poison instead of
    /// building.
    fn kernel_inner(
        &self,
        name: &'static str,
        spv: &[u32],
        n_buf: usize,
        push_size: u32,
        required_sg: Option<u32>,
    ) -> ComputeKernel {
        let mut map = self.shared.kernels.lock().unwrap();
        let cached = map.get(name).copied();
        let built = cached.is_none();
        let k = match cached {
            Some(k) => k,
            None => match try_make_compute_kernel(
                &self.shared.device,
                self.shared.pipeline_cache,
                name,
                spv,
                n_buf,
                push_size,
                required_sg,
                self.shared.push_descriptor.is_some(),
                self.kernel_limits(),
                self.shared
                    .rte_f16
                    .load(std::sync::atomic::Ordering::Relaxed),
                self.shared.wave64,
            ) {
                Ok(k) => *map.entry(name).or_insert(k),
                Err(e) => {
                    drop(map);
                    panic!(
                        "{e} — kernel build does not yet thread Result to this call site (see \
                         docs/backlog.md B98)"
                    )
                }
            },
        };
        drop(map);
        if built {
            // One line per pipeline this backend creates, with what it was created with, so two
            // builds' kernel sets can be compared (`RUST_LOG=infr_vulkan::kernels=debug`).
            tracing::debug!(
                target: "infr_vulkan::kernels",
                "kernel built: {name} sg={} n_buf={n_buf} push={push_size}",
                required_sg.map_or_else(|| "-".to_string(), |s| s.to_string()),
            );
            self.shared.persist_pipeline_cache(); // new pipeline -> debounced disk save
        } else {
            // Cache HIT: the cache is keyed on `name` alone, so a name reused with a different
            // shader/layout would silently hand back a mismatched pipeline (VUID/UB). Fail loudly
            // in debug instead. The spv hash is only computed on this hit path in debug builds.
            debug_assert_eq!(
                k.n_buf, n_buf,
                "kernel {name:?} cached with n_buf={} but re-requested with n_buf={n_buf}",
                k.n_buf
            );
            debug_assert_eq!(
                k.push_size, push_size,
                "kernel {name:?} cached with push_size={} but re-requested with push_size={push_size}",
                k.push_size
            );
            debug_assert_eq!(
                k.spv_hash,
                hash_spv(spv),
                "kernel {name:?} cached from a different SPIR-V than this request — a name reused \
                 across two distinct shaders returns a mismatched pipeline"
            );
        }
        k
    }

    /// The twin of `k`'s pipeline created with `VK_PIPELINE_CREATE_DISPATCH_BASE_BIT`, which
    /// `vkCmdDispatchBase` with a non-zero base requires: the same shader module, layout,
    /// specialization and subgroup pin, so it runs the same code. Built on first use and cached
    /// by kernel name; only a dispatch the work bound cuts into ranges ever asks for one, so a
    /// device that never cuts creates none and its kernels' own pipelines are untouched.
    pub(crate) fn dispatch_base_pipeline(&self, k: &ComputeKernel) -> vk::Pipeline {
        let mut map = self.shared.base_pipelines.lock().unwrap();
        if let Some(&p) = map.get(k.name) {
            return p;
        }
        let built = create_pipeline(
            &self.shared.device,
            self.shared.pipeline_cache,
            k.shader,
            k.pipeline_layout,
            k.pin,
            self.shared.wave64,
            vk::PipelineCreateFlags::DISPATCH_BASE,
        );
        match built {
            Ok(p) if p != vk::Pipeline::null() => {
                map.insert(k.name, p);
                drop(map);
                self.shared.persist_pipeline_cache();
                p
            }
            other => {
                drop(map);
                panic!(
                    "creating the DISPATCH_BASE twin of kernel {:?} failed ({other:?}) although its own pipeline was created from the same module",
                    k.name
                )
            }
        }
    }

    /// Build the [`EagerBind`] for one one-shot dispatch of `k`: with push descriptors the
    /// writes are recorded later by [`EagerBind::bind`]; without, a classic set is allocated
    /// from `k.desc_pool` (reset first — the pool holds one set) and written here.
    pub(crate) fn eager_bind(&self, k: &ComputeKernel, bufs: &[vk::Buffer]) -> Result<EagerBind> {
        assert_eq!(bufs.len(), k.n_buf);
        let device = &self.shared.device;
        let pooled_set = if self.shared.push_descriptor.is_none() {
            unsafe {
                device
                    .reset_descriptor_pool(k.desc_pool, vk::DescriptorPoolResetFlags::empty())
                    .map_err(|e| be(format!("reset pool: {e}")))?;
            }
            Some(unsafe {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(k.desc_pool)
                            .set_layouts(std::slice::from_ref(&k.ds_layout)),
                    )
                    .map_err(|e| be(format!("alloc set: {e}")))?[0]
            })
        } else {
            None
        };
        let bind = EagerBind {
            pooled_set,
            infos: bufs
                .iter()
                .map(|&buffer| vk::DescriptorBufferInfo {
                    buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                })
                .collect(),
        };
        if bind.pooled_set.is_some() {
            unsafe { device.update_descriptor_sets(&bind.writes(), &[]) };
        }
        Ok(bind)
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
            .map(|b| b.as_ref())
            .chain(std::iter::once(out.as_ref()))
            .map(|b| as_vk_buf(b).map(|v| v.buffer))
            .collect::<Result<Vec<_>>>()?;
        let binding = self.eager_bind(&k, &vk_bufs)?;

        #[cfg(test)]
        crate::recorder::note_test_grid(k.name, [groups, 1, 1]);
        let shared = &self.shared;
        self.one_shot(|cmd| unsafe {
            shared
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline);
            binding.bind(shared, cmd, k.pipeline_layout);
            if k.push_size > 0 {
                shared.device.cmd_push_constants(
                    cmd,
                    k.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
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
        let k = self.kernel_sg("rmsnorm", crate::gemm::rmsnorm_spv(), 3, 12, 32);
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
        // push_size 28 must match rope.comp's full PC block (…, pos_offset, out_base) AND the
        // recorder's `kernel("rope", …, 28)` — the kernel cache is shared by name, so a smaller
        // range here would create a pipeline layout that doesn't contain the shader's block
        // (VUID-VkComputePipelineCreateInfo-layout-10069).
        let k = self.kernel("rope", crate::gemm::rope_spv(), 2, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&0u32.to_ne_bytes()); // pos_offset
        push[24..28].copy_from_slice(&0u32.to_ne_bytes()); // out_base (eager: write at row 0)
        let groups = (t * n_heads).div_ceil(crate::recorder::ROPE_WG) as u32;
        self.run_kernel(k, &[x], t * n_heads * hd, &push, groups)
    }

    /// SwiGLU activation: `y[i] = silu(gate[i]) * up[i]`.
    pub fn silu_mul(&self, gate: &[f32], up: &[f32], n: usize) -> Result<Vec<f32>> {
        let k = self.kernel("silu_mul", crate::gemm::silu_mul_spv(), 3, 36);
        // Full params: the trailing `do_clamp`/`limit` pair stays zero (this eager helper has no
        // SwiGLU-clamp caller), but must be PUSHED or the shader reads stale words.
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        self.run_kernel(k, &[gate, up], n, &push, (n as u32).div_ceil(64))
    }

    /// Elementwise add: `y[i] = a[i] + b[i]`.
    pub fn add(&self, a: &[f32], b: &[f32], n: usize) -> Result<Vec<f32>> {
        let k = self.kernel("add", crate::gemm::add_spv(), 3, 4);
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
        let kern = self.kernel("attention", crate::gemm::attention_spv(), 4, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        let groups = (t * nh).div_ceil(crate::recorder::ATTENTION_WG) as u32;
        self.run_kernel(kern, &[q, k, v], t * nh * hd, &push, groups)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Host RoPE over `x` `[t, nh, hd]` as `rope.comp` computes it: token `p` at position
    /// `pos0 + p`, the first `rd` lanes rotated (interleaved NORM pairs, or NEOX split halves), the
    /// angle divided by `ff[pair]` when given and its sine negated for the backward build, and
    /// lanes `rd..hd` passed through.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn host_rope(
        x: &[f32],
        t: usize,
        nh: usize,
        hd: usize,
        rd: usize,
        theta: f32,
        pos0: usize,
        ff: Option<&[f32]>,
        neox: bool,
        backward: bool,
    ) -> Vec<f32> {
        let mut want = x.to_vec();
        let hf = rd / 2;
        for p in 0..t {
            for h in 0..nh {
                let base = (p * nh + h) * hd;
                for i in 0..hf {
                    let freq = (theta as f64).powf(-2.0 * i as f64 / rd as f64) as f32;
                    let mut ang = (pos0 + p) as f32 * freq;
                    if let Some(ff) = ff {
                        ang /= ff[i];
                    }
                    let (mut s, co) = ang.sin_cos();
                    if backward {
                        s = -s;
                    }
                    let (i0, i1) = if neox {
                        (i, i + hf)
                    } else {
                        (2 * i, 2 * i + 1)
                    };
                    let (a, b) = (x[base + i0], x[base + i1]);
                    want[base + i0] = a * co - b * s;
                    want[base + i1] = a * s + b * co;
                }
            }
        }
        want
    }

    /// Host causal GQA attention with a standard softmax: q `[t, nh, hd]`, k/v `[t, nkv, hd]` →
    /// `[t, nh, hd]`, as `attention.comp` computes it with an online one.
    pub(crate) fn host_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        t: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) -> Vec<f32> {
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
        want
    }

    #[test]
    fn wave64_pins_every_pipeline_to_64_and_full_only_where_allowed() {
        let wg = |x| Some([x, 1, 1]);
        // Wave32 devices: exactly what the kernel asked for, always with full subgroups.
        assert_eq!(subgroup_pin(None, false, wg(64)), None);
        assert_eq!(
            subgroup_pin(Some(32), false, wg(32)),
            Some(SubgroupPin {
                size: 32,
                full: true
            })
        );
        assert_eq!(
            subgroup_pin(Some(16), false, wg(128)),
            Some(SubgroupPin {
                size: 16,
                full: true
            })
        );
        // Wave64: 64 whatever was asked, including nothing; full only for a multiple of 64.
        for asked in [None, Some(16), Some(32)] {
            for (x, full) in [(32, false), (64, true), (96, false), (256, true)] {
                assert_eq!(
                    subgroup_pin(asked, true, wg(x)),
                    Some(SubgroupPin { size: 64, full }),
                    "asked {asked:?}, local_size_x {x}"
                );
            }
            assert_eq!(
                subgroup_pin(asked, true, None),
                Some(SubgroupPin {
                    size: 64,
                    full: false
                }),
                "an unreadable workgroup size must not claim full subgroups"
            );
        }
    }

    /// Every built module that touches subgroups is covered by `shaders/wave64.glsl` (it carries
    /// the `W64` specialization constant), is a cooperative-matrix kernel (never dispatched on the
    /// wave64 path), or is on the short list of kernels whose subgroup arithmetic holds at ANY
    /// width. A covered module must also be one the header's two modes fit — a workgroup of at most
    /// 32 invocations, or `local_size_x` a multiple of 64 so the host can ask for full subgroups —
    /// and use only operations the header remaps. In a module with more than 32 invocations every
    /// plain reduction must come with its cluster-32 twin of the same opcode (the macro emits both;
    /// a lone plain one, say from a header included before `wave64.glsl`, spans both halves of a
    /// 64-lane subgroup on a wave64 device).
    #[test]
    fn subgroup_kernels_are_wave64_safe() {
        use std::collections::HashMap;
        // SPIR-V enumerants (unified spec 1.6).
        const OP_CAPABILITY: u32 = 17;
        const CAPABILITY_COOPMAT: u32 = 6022; // CooperativeMatrixKHR
        const OP_CONSTANT: u32 = 43;
        const OP_DECORATE: u32 = 71;
        const DECORATION_SPEC_ID: u32 = 1;
        const DECORATION_BUILTIN: u32 = 11;
        const OP_ELECT: u32 = 333;
        const OP_FADD: u32 = 350;
        const OP_UMIN: u32 = 354;
        const OP_ARITH_FIRST: u32 = 349; // OpGroupNonUniformIAdd
        const OP_ARITH_LAST: u32 = 364; // OpGroupNonUniformLogicalXor
        const OP_QUAD_LAST: u32 = 366; // OpGroupNonUniformQuadSwap
        const OP_ROTATE: u32 = 4431; // OpGroupNonUniformRotateKHR
        const REDUCE: u32 = 0;
        const INCLUSIVE_SCAN: u32 = 1;
        const CLUSTERED_REDUCE: u32 = 3;
        // SubgroupSize, NumSubgroups, SubgroupId, SubgroupLocalInvocationId.
        const SUBGROUP_BUILTINS: [u32; 4] = [36, 38, 40, 41];
        // SubgroupEq/Ge/Gt/Le/LtMask.
        const LANE_MASKS: std::ops::RangeInclusive<u32> = 4416..=4420;
        // The workgroup `wave64.glsl` runs as one subgroup rather than as 32-lane halves.
        const WHOLE_MAX: u32 = 32;
        const HOW_TO_ADD: &str = "remap it in shaders/wave64.glsl the way subgroupAdd is \
                                  (a cluster-32 twin under VW_HALVES), then allow its opcode here";
        // Kernels written for any subgroup width, read for it: `moe_topk.comp` reduces with
        // subgroupMax/Min, picks a lane with subgroupElect and walks `gl_NumSubgroups` partials,
        // none of which assumes 32 lanes; `wave64_probe.comp` reads the real subgroups on purpose.
        const WIDTH_AGNOSTIC: &[&str] = &["moe_topk", "wave64_probe"];

        let (mut covered, mut agnostic, mut coopmat) = (0, 0, 0);
        for entry in std::fs::read_dir(env!("OUT_DIR")).expect("OUT_DIR") {
            let path = entry.expect("dir entry").path();
            if path.extension().is_none_or(|e| e != "spv") {
                continue;
            }
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let spv = crate::gemm::spv_words(&std::fs::read(&path).expect("read spv"));
            let mut consts = HashMap::new();
            let (mut w64, mut uses, mut cm, mut bad) = (false, false, false, Vec::new());
            let (mut elects, mut fadd_scans) = (0usize, 0usize);
            // Per opcode: plain reductions, and cluster-size ids of the clustered ones.
            let mut plain: HashMap<u32, usize> = HashMap::new();
            let mut clustered: Vec<(u32, u32)> = Vec::new();
            let mut at = 5usize;
            while at < spv.len() {
                let wc = (spv[at] >> 16) as usize;
                assert!(wc > 0 && at + wc <= spv.len(), "{name}: malformed SPIR-V");
                let (op, ins) = (spv[at] & 0xffff, &spv[at..at + wc]);
                match op {
                    OP_CAPABILITY => cm |= ins[1] == CAPABILITY_COOPMAT,
                    OP_CONSTANT if wc == 4 => {
                        consts.insert(ins[2], ins[3]);
                    }
                    OP_DECORATE if wc == 4 && ins[2] == DECORATION_SPEC_ID => {
                        w64 |= ins[3] == WAVE64_SPEC_ID;
                    }
                    OP_DECORATE if wc == 4 && ins[2] == DECORATION_BUILTIN => {
                        if SUBGROUP_BUILTINS.contains(&ins[3]) {
                            uses = true;
                        } else if LANE_MASKS.contains(&ins[3]) {
                            uses = true;
                            bad.push(format!("lane-mask builtin {}", ins[3]));
                        }
                    }
                    OP_ELECT => {
                        uses = true;
                        elects += 1;
                    }
                    OP_ARITH_FIRST..=OP_ARITH_LAST => {
                        uses = true;
                        match ins[4] {
                            REDUCE => *plain.entry(op).or_default() += 1,
                            INCLUSIVE_SCAN if op == OP_FADD => fadd_scans += 1,
                            CLUSTERED_REDUCE => clustered.push((op, ins[6])),
                            g => bad.push(format!(
                                "opcode {op} with group operation {g}: {HOW_TO_ADD}"
                            )),
                        }
                    }
                    o if (OP_ELECT..=OP_QUAD_LAST).contains(&o) || o == OP_ROTATE => {
                        uses = true;
                        bad.push(format!("opcode {o}: {HOW_TO_ADD}"));
                    }
                    _ => {}
                }
                at += wc;
            }
            if !uses {
                continue;
            }
            if cm {
                // Cooperative-matrix kernels never run on the wave64 path: the backend drops every
                // coopmat tier there (`!wave64` on `coopmat_f16` in `VulkanBackend::new_with`),
                // because a coopmat fragment spans the whole 64-lane subgroup.
                coopmat += 1;
                continue;
            }
            if !w64 {
                assert!(
                    WIDTH_AGNOSTIC.iter().any(|k| name.starts_with(k)),
                    "{name}: uses subgroups without including wave64.glsl (and is not one of the \
                     width-agnostic kernels) — on a wave64 device its subgroups are 64 lanes"
                );
                agnostic += 1;
                continue;
            }
            let local = crate::spirv::local_size(&spv)
                .unwrap_or_else(|| panic!("{name}: no readable LocalSize"));
            let invocations: u32 = local.iter().product();
            assert!(
                invocations <= WHOLE_MAX || local[0].is_multiple_of(64),
                "{name}: workgroup {local:?} is neither at most {WHOLE_MAX} invocations nor 64-aligned \
                 in x, so wave64.glsl can neither run it as one subgroup nor ask for full ones"
            );
            let mut twins: HashMap<u32, usize> = HashMap::new();
            for (op, id) in clustered {
                match consts.get(&id) {
                    Some(32) => *twins.entry(op).or_default() += 1,
                    Some(&n) if n < 32 && 32 % n == 0 => {}
                    n => bad.push(format!(
                        "opcode {op} clustered at {n:?}, which does not divide 32"
                    )),
                }
            }
            if invocations > WHOLE_MAX {
                // `vw_half_elect` is the elect's twin: a cluster-32 UMin.
                *plain.entry(OP_UMIN).or_default() += elects;
                for (op, &n) in &plain {
                    let t = twins.get(op).copied().unwrap_or(0);
                    if n > t {
                        bad.push(format!(
                            "opcode {op}: {n} plain reduction(s) (elects counted as UMin) but {t} \
                             cluster-32 twin(s) — one spans both halves; {HOW_TO_ADD}"
                        ));
                    }
                }
                // `vw_half_inclusive_add` scans twice beside its plain twin.
                if fadd_scans % 3 != 0 {
                    bad.push(format!(
                        "{fadd_scans} float inclusive scans, not three per subgroupInclusiveAdd"
                    ));
                }
            }
            assert!(
                bad.is_empty(),
                "{name}: includes wave64.glsl but is not wave64-safe: {bad:?}"
            );
            covered += 1;
        }
        assert!(
            covered > 0 && agnostic > 0 && coopmat > 0,
            "matched {covered} wave64.glsl modules, {agnostic} width-agnostic and {coopmat} \
             coopmat ones — the check ran over nothing"
        );
    }

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
        // More (token, head) pairs than one workgroup holds, the last workgroup ragged, so a grid
        // sized short of them leaves rows un-roped.
        let (t, nh, hd, rd, theta) = (13usize, 11usize, 8usize, 8usize, 10000.0f32);
        let x: Vec<f32> = (0..t * nh * hd).map(|i| (i as f32) * 0.03).collect();
        let got = be.rope(&x, t, nh, hd, rd, theta).unwrap();
        // host reference (ggml NORM interleaved)
        let want = host_rope(&x, t, nh, hd, rd, theta, 0, None, false, false);
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
        // More (token, head) pairs than one workgroup holds, the last workgroup ragged, so a grid
        // sized short of them leaves rows unwritten (the readback buffer starts zeroed, which no
        // row of this output is).
        let (t, nh, nkv, hd) = (13usize, 12usize, 4usize, 8usize);
        let q: Vec<f32> = (0..t * nh * hd).map(|i| (i as f32 * 0.013).sin()).collect();
        let k: Vec<f32> = (0..t * nkv * hd)
            .map(|i| (i as f32 * 0.017).cos())
            .collect();
        let v: Vec<f32> = (0..t * nkv * hd).map(|i| 0.1 + i as f32 * 0.011).collect();
        let (got, grids) = crate::recorder::capture_test_grids(|| {
            be.attention(&q, &k, &v, t, nh, nkv, hd).unwrap()
        });
        let want = host_attention(&q, &k, &v, t, nh, nkv, hd);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "{g} vs {w}");
        }
        // One thread per (token, head) pair: the grid is the pairs in `ATTENTION_WG`-wide groups.
        let groups = (t * nh).div_ceil(crate::recorder::ATTENTION_WG) as u32;
        assert_eq!(grids, [("attention", [groups, 1, 1])]);
    }
}
