//! GPU stochastic sampler (`Recorder::sample_topk` / `Op::Sample`) vs the host reference: same
//! logits + same uniform draw must pick the same token. The reference below replicates
//! `infr-llama::sampling::sample_logits` with the draw factored out (this crate can't depend on
//! infr-llama); keep the two in sync.
//! Run: cargo test -p infr-vulkan --test sample_topk -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

fn host_sample(logits: &[f32], k: usize, temp: f32, top_p: f32, u: f32) -> u32 {
    let cmp = |a: &usize, b: &usize| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(k - 1, cmp);
    idx.truncate(k);
    idx.sort_unstable_by(cmp);
    let maxl = logits[idx[0]];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - maxl) / temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let mut cum = 0.0;
    let mut cutoff = probs.len();
    for (j, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cutoff = j + 1;
            break;
        }
    }
    let total: f32 = probs[..cutoff].iter().sum();
    let r = u * total;
    let mut acc = 0.0;
    for j in 0..cutoff {
        acc += probs[j];
        if r <= acc {
            return idx[j] as u32;
        }
    }
    idx[cutoff - 1] as u32
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn sample_topk_matches_host() {
    let be = VulkanBackend::new().unwrap();
    let n = 151_936usize; // qwen3 vocab
                          // Deterministic pseudo-random logits (distinct values — exact ties would legitimately
                          // diverge in sort order between the two implementations).
    let mut state = 0x2545F4914F6CDD1Du64;
    let logits: Vec<f32> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0
        })
        .collect();
    let lb = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    be.upload(lb.as_ref(), bytemuck::cast_slice(&logits))
        .unwrap();
    let (top_k, temp, top_p) = (20usize, 0.6f32, 0.95f32);
    let cand = be
        .alloc(2 * 256 * top_k * 4, BufferUsage::Activations)
        .unwrap();
    let ub = be.alloc(4, BufferUsage::Staging).unwrap();
    let ob = be.alloc(4, BufferUsage::Readback).unwrap();
    for i in 0..32 {
        let u = (i as f32 + 0.5) / 32.0;
        be.upload(ub.as_ref(), &u.to_le_bytes()).unwrap();
        let rec = be.recorder().unwrap();
        rec.sample_topk(
            lb.as_ref(),
            cand.as_ref(),
            ub.as_ref(),
            ob.as_ref(),
            n,
            top_k,
            temp,
            top_p,
        );
        rec.finish().unwrap();
        let mut idb = [0u8; 4];
        be.download(ob.as_ref(), &mut idb).unwrap();
        let gpu = u32::from_le_bytes(idb);
        let host = host_sample(&logits, top_k, temp, top_p, u);
        assert_eq!(gpu, host, "u={u}: gpu {gpu} != host {host}");
    }
}

/// One GPU draw over `logits` with uniform `u`.
fn gpu_sample(
    be: &VulkanBackend,
    logits: &[f32],
    top_k: usize,
    temp: f32,
    top_p: f32,
    u: f32,
) -> u32 {
    let n = logits.len();
    let lb = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    be.upload(lb.as_ref(), bytemuck::cast_slice(logits))
        .unwrap();
    let cand = be
        .alloc(2 * 256 * top_k * 4, BufferUsage::Activations)
        .unwrap();
    let ub = be.alloc(4, BufferUsage::Staging).unwrap();
    let ob = be.alloc(4, BufferUsage::Readback).unwrap();
    be.upload(ub.as_ref(), &u.to_le_bytes()).unwrap();
    let rec = be.recorder().unwrap();
    rec.sample_topk(
        lb.as_ref(),
        cand.as_ref(),
        ub.as_ref(),
        ob.as_ref(),
        n,
        top_k,
        temp,
        top_p,
    );
    rec.finish().unwrap();
    let mut idb = [0u8; 4];
    be.download(ob.as_ref(), &mut idb).unwrap();
    u32::from_le_bytes(idb)
}

/// Exact ties at the k-th value must not evict a strictly greater one. Stage 1 gives each of its
/// 256 workgroups a contiguous slice; here the first slice holds 63 equal values and, at its
/// end, the one clear maximum, so far more than `top_k` values reach the k-th key. The maximum
/// carries essentially all the probability at this temperature, so every draw must return it.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn sample_topk_keeps_the_maximum_past_exact_ties() {
    let be = VulkanBackend::new().unwrap();
    let slice = 64usize;
    let mut logits = vec![-10.0f32; 256 * slice];
    logits[..slice - 1].fill(1.0);
    let top = slice - 1;
    logits[top] = 5.0;
    for i in 0..8 {
        let u = (i as f32 + 0.5) / 8.0;
        let got = gpu_sample(&be, &logits, 4, 0.05, 1.0, u);
        assert_eq!(got, top as u32, "u={u}: the maximum was evicted by ties");
    }
}

/// Among exact ties the LOWEST indices are kept and ranked first, so the draw is a function of
/// the logits and `u` alone. All logits equal: the top 4 are tokens 0..4 in that order, each with
/// a quarter of the mass.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn sample_topk_breaks_exact_ties_by_lowest_index() {
    let be = VulkanBackend::new().unwrap();
    let logits = vec![0.0f32; 151_936];
    for (j, u) in [0.1f32, 0.35, 0.6, 0.85].into_iter().enumerate() {
        assert_eq!(gpu_sample(&be, &logits, 4, 1.0, 1.0, u), j as u32, "u={u}");
    }
}
