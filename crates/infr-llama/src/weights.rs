//! Weight-footprint accounting: how many VRAM bytes a model's weights will occupy once resident
//! (dense vs MoE-expert), mirroring the native-quant/f16 upload policy used by the production
//! agnostic-seam loaders.
use infr_core::WeightSource;
use infr_gguf::Gguf;

/// VRAM the model's weights will occupy once resident, split dense vs MoE-expert. Experts are
/// tracked separately so a future expert-streaming / partial-offload mode can budget them apart
/// from the always-resident dense weights — for a dense model `expert` is 0.
#[derive(Clone, Copy, Debug)]
pub struct WeightFootprint {
    /// Always-resident weights: projections, embeddings, norms.
    pub dense: u64,
    /// MoE expert weights (GGUF `*_exps` stacked tensors). 0 for dense models.
    pub expert: u64,
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl WeightFootprint {
    /// All-resident footprint: dense + every expert kept in VRAM.
    pub fn total(&self) -> u64 {
        self.dense + self.expert
    }

    /// Footprint if experts are STREAMED through an `n_slots`-slot pool of `stride`-byte slots
    /// (a VRAM slot pool) instead of all kept resident: `dense + n_slots·stride`, bounded
    /// regardless of the model's expert count. The MoE loader picks all-resident ([`total`]) when it
    /// fits VRAM, else reserves this and streams. (`stride` = one expert's max packed weight bytes.)
    pub fn streaming_total(&self, n_slots: usize, stride: usize) -> u64 {
        self.dense + n_slots as u64 * stride as u64
    }
}

/// Resident VRAM bytes for one tensor, mirroring [`upload_wt`]'s path so the estimate matches what
/// actually gets allocated: native raw blocks (padded to u32) for every quant format, else f16
/// (float/norms dequanted to half).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn tensor_resident_bytes(dtype: infr_core::DType, numel: usize, nbytes: usize) -> u64 {
    if infr_vulkan::linear::native_dense_supported(dtype) {
        ((nbytes + 3) & !3) as u64 // raw blocks, padded to u32 alignment
    } else {
        (numel * 2) as u64 // f16
    }
}

/// Sum the resident weight footprint across all tensors (MoE-aware). Enumerating every tensor means
/// stacked expert tensors are counted in full, so this is correct for MoE the moment the arch is
/// supported. `token_embd` is excluded (it lives in host RAM for the CPU embedding gather) unless
/// the lm head is tied to it (no `output.weight`), where an f16 copy is uploaded to VRAM.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn weight_footprint(g: &Gguf) -> WeightFootprint {
    let has_output = g.tensors().iter().any(|t| t.name == "output.weight");
    let mut dense = 0u64;
    let mut expert = 0u64;
    for t in g.tensors() {
        let numel: usize = t.shape.iter().product();
        if t.name == "token_embd.weight" {
            if !has_output {
                dense += (numel * 2) as u64; // tied lm head, uploaded as f16
            }
            continue;
        }
        let bytes = tensor_resident_bytes(t.dtype, numel, t.nbytes);
        if t.name.contains("_exps") {
            expert += bytes;
        } else {
            dense += bytes;
        }
    }
    WeightFootprint { dense, expert }
}
