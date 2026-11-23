//! Decode split-K attention against a host reference: the partial pass and `attn_combine.comp`
//! together must give the softmax attention a plain f64 loop gives, at every chunk count.
//!
//! `attn_decode_parity` compares two builds of the partial pass with each other, and both feed
//! the same combine, so an error in the combine cancels there. This file is what holds the
//! combine itself: it loads the partials in batches of 8 chunks with a scalar tail, so the chunk
//! counts below sit on both sides of every batch edge, and the replay cases leave workgroups past
//! the live key range whose partials the combine must not read into the result.
//!
//! Run: `cargo test --release -p infr-vulkan --test attn_split_host -- --ignored`
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_vulkan::VulkanBackend;

const NH: usize = 4;
const NKV: usize = 2;
const HD: usize = 128;

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    }
}

/// `n` values in [-1, 1) rounded to f16, as the bytes the GPU reads and the values they hold.
fn f16_data(n: usize, seed: u64) -> (Vec<u8>, Vec<f64>) {
    let mut r = Rng(seed | 1);
    let mut bytes = Vec::with_capacity(n * 2);
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        let h = half::f16::from_f32(r.next_f32());
        bytes.extend_from_slice(&h.to_bits().to_le_bytes());
        vals.push(f64::from(h.to_f32()));
    }
    (bytes, vals)
}

/// Causal attention of one decode row over `kv_len` keys, in f64. Q is `[NH, HD]`, the caches
/// `[kv_len, NKV, HD]`, head `h` reads KV head `h / (NH / NKV)`, the scale is `1/sqrt(HD)`.
fn host_attention(q: &[f64], k: &[f64], v: &[f64], kv_len: usize) -> Vec<f64> {
    let scale = 1.0 / (HD as f64).sqrt();
    let mut out = vec![0f64; NH * HD];
    for h in 0..NH {
        let kvh = h / (NH / NKV);
        let row = |c: &[f64], j: usize| -> Vec<f64> {
            let at = (j * NKV + kvh) * HD;
            c[at..at + HD].to_vec()
        };
        let scores: Vec<f64> = (0..kv_len)
            .map(|j| {
                let kj = row(k, j);
                (0..HD).map(|d| q[h * HD + d] * kj[d]).sum::<f64>() * scale
            })
            .collect();
        let top = scores.iter().cloned().fold(f64::MIN, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
        let total: f64 = weights.iter().sum();
        for (j, w) in weights.iter().enumerate() {
            let vj = row(v, j);
            for d in 0..HD {
                out[h * HD + d] += w / total * vj[d];
            }
        }
    }
    out
}

fn assert_close(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len());
    let mut worst = 0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(*g) - w).abs();
        worst = worst.max(err);
        assert!(
            err <= 2e-6 + 1e-5 * w.abs(),
            "{what}: element {i} (head {}, dim {}) is {g}, the host reference {w}",
            i / HD,
            i % HD
        );
    }
    println!("{what}: within {worst:.2e} of the host reference");
}

/// One case on both recording paths. `chunk` is the static path's chunk and the replay path's
/// baked floor; `stride` is the partials scratch's chunk count.
fn check(be: &VulkanBackend, kv_len: usize, chunk: usize) {
    let stride = kv_len.div_ceil(chunk);
    let elems = kv_len * NKV * HD;
    let (q_bytes, q) = f16_data(NH * HD, 101);
    let (k_bytes, k) = f16_data(elems, 11);
    let (v_bytes, v) = f16_data(elems, 23);
    let want = host_attention(&q, &k, &v, kv_len);

    let upload = |bytes: &[u8], usage| -> Box<dyn Buffer> {
        let b = be.alloc(bytes.len(), usage).unwrap();
        be.upload(b.as_ref(), bytes).unwrap();
        b
    };
    let qb = upload(&q_bytes, BufferUsage::Activations);
    let kb = upload(&k_bytes, BufferUsage::KvCache);
    let vb = upload(&v_bytes, BufferUsage::KvCache);
    let ka = kb.device_addr().expect("KvCache K device address");
    let va = vb.device_addr().expect("KvCache V device address");
    let scratch = |n: usize| be.alloc(n * 4, BufferUsage::Activations).unwrap();
    let (pm, pl, pacc) = (
        scratch(NH * stride),
        scratch(NH * stride),
        scratch(NH * stride * HD),
    );
    let read = |b: &dyn Buffer| -> Vec<f32> {
        let mut out = vec![0f32; NH * HD];
        be.download(b, bytemuck::cast_slice_mut(&mut out)).unwrap();
        out
    };
    let pos = kv_len - 1;

    let o_stat = scratch(NH * HD);
    let rec = be.recorder().unwrap();
    rec.attention_kv_split_at(
        qb.as_ref(),
        kb.as_ref(),
        vb.as_ref(),
        ka,
        va,
        o_stat.as_ref(),
        pm.as_ref(),
        pl.as_ref(),
        pacc.as_ref(),
        1, // rows
        pos,
        kv_len,
        NH,
        NKV,
        HD,
        chunk,
        stride,
        0.0,   // the kernel's 1/sqrt(hd)
        0,     // no window
        None,  // canvas_lo
        false, // k f16
        false, // v f16
        0,     // cap
        false, // batched
        None,  // kv_ml
        None,  // pages
    );
    rec.finish().unwrap();
    assert_close(
        &read(o_stat.as_ref()),
        &want,
        &format!("static kv {kv_len} chunk {chunk} ({stride} chunks)"),
    );

    let params = upload(
        bytemuck::cast_slice(&[pos as u32, kv_len as u32]),
        BufferUsage::Activations,
    );
    let args = be.alloc(16, BufferUsage::Activations).unwrap();
    let o_rep = scratch(NH * HD);
    let rec = be.recorder().unwrap();
    rec.attn_live_prologue(params.as_ref(), args.as_ref(), NH, chunk, 0);
    rec.attention_kv_split_dynac_at(
        qb.as_ref(),
        kb.as_ref(),
        vb.as_ref(),
        ka,
        va,
        o_rep.as_ref(),
        pm.as_ref(),
        pl.as_ref(),
        pacc.as_ref(),
        params.as_ref(),
        args.as_ref(),
        NH,
        NKV,
        HD,
        chunk,
        stride,
        0.0,
        0,
        false, // q8
        0,     // cap
        0,     // page_rows
    );
    rec.finish().unwrap();
    // The replay kernels re-derive the chunk from the live span, never below the baked floor.
    let live = chunk.max((kv_len / 32).clamp(1, 512));
    assert_close(
        &read(o_rep.as_ref()),
        &want,
        &format!(
            "replay kv {kv_len} floor {chunk} ({} live chunks of {live}, stride {stride})",
            kv_len.div_ceil(live)
        ),
    );
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn split_attention_matches_the_host_at_every_chunk_count() {
    let Some(be) = infr_testkit::gpu_or_skip(VulkanBackend::new()) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // Chunk counts on both sides of the combine's batches of 8, each with a ragged last chunk.
    for chunks in [1usize, 2, 7, 8, 9, 15, 16, 17, 24, 31, 33] {
        check(&be, chunks * 16 - 3, 16);
    }
    // A span whose live chunk outgrows the baked floor: the scratch stride (63) is wider than the
    // live count (33), so the combine has to take the live count from the prologue.
    check(&be, 1000, 16);
    // The floor every device ran before the decode floor became a parameter.
    for chunks in [2usize, 8, 9, 17] {
        check(&be, chunks * 64 - 5, 64);
    }
}
