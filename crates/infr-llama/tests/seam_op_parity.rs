//! Per-op parity probe: run a one-op agnostic Graph on the CPU reference backend and on Vulkan,
//! compare outputs. Isolates which qwen35-specific op diverges on the Vulkan seam (the whole-model
//! seam garbles; this pinpoints the culprit). Run with:
//!   cargo test -p infr-llama --release --test seam_op_parity -- --include-ignored --nocapture
use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::TensorDesc;
use infr_core::{DType, TensorId};

fn f32d(n: usize) -> TensorDesc {
    TensorDesc::new(vec![n], DType::F32)
}

/// Run `build` (returns the graph + the ordered (handle, data) inputs + the output handle+len) on
/// `be`, returning the downloaded output.
fn run(
    be: &dyn Backend,
    g: &Graph,
    inputs: &[(TensorId, &[f32])],
    weights: &[(TensorId, &[f32])],
    out: TensorId,
    out_len: usize,
) -> Vec<f32> {
    let plan = be.compile(g).expect("compile");
    // Alloc + upload all inputs/weights first (owned), then bind from the Vec so the bound refs
    // outlive `execute`.
    let mut keep: Vec<(TensorId, Box<dyn infr_core::backend::Buffer>)> = Vec::new();
    for (id, data) in inputs {
        let buf = be
            .alloc(data.len() * 4, BufferUsage::Activations)
            .expect("alloc in");
        be.upload(buf.as_ref(), bytemuck::cast_slice(data)).unwrap();
        keep.push((*id, buf));
    }
    for (id, data) in weights {
        let buf = be
            .alloc(data.len() * 4, BufferUsage::Weights)
            .expect("alloc w");
        be.upload(buf.as_ref(), bytemuck::cast_slice(data)).unwrap();
        keep.push((*id, buf));
    }
    let obuf = be
        .alloc(out_len * 4, BufferUsage::Readback)
        .expect("alloc out");
    let mut b = Bindings::new();
    for (id, buf) in &keep {
        b.bind(*id, buf.as_ref());
    }
    b.bind(out, obuf.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; out_len];
    be.download(obuf.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

fn gpu() -> Option<infr_vulkan::VulkanBackend> {
    infr_vulkan::VulkanBackend::new().ok()
}

/// Does an in-place-mutated recurrent state Input PERSIST across `execute` calls? (Decode runs one
/// token per execute, carrying conv/SSM state in the bound buffer.) Runs Conv1dSilu twice reusing the
/// same state buffer on each backend; the 2nd output must match — if Vulkan doesn't persist the
/// in-place state write, its 2nd token diverges (the whole-model seam garble).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn state_persists_across_executes() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (cc, kernel) = (32usize, 4usize);
    let build = || {
        let mut g = Graph::new();
        let x = g.input(f32d(cc));
        let w = g.weight(f32d(cc * kernel));
        let state = g.input(f32d((kernel - 1) * cc));
        let dst = g.output(f32d(cc));
        g.push(Op::Conv1dSilu {
            x,
            weight: w,
            state,
            dst,
            rows: 1,
            channels: cc as u32,
            kernel: kernel as u32,
        });
        (g, x, w, state, dst)
    };
    let wi = gen(cc * kernel, 7);
    let x1 = gen(cc, 10);
    let x2 = gen(cc, 11);
    // Second-token output when the SAME state buffer is reused across two executes.
    let second = |be: &dyn Backend| -> Vec<f32> {
        let (g, x, w, state, dst) = build();
        let plan = be.compile(&g).unwrap();
        let sbuf = be
            .alloc((kernel - 1) * cc * 4, BufferUsage::Activations)
            .unwrap(); // zeroed
        let wbuf = be.alloc(cc * kernel * 4, BufferUsage::Weights).unwrap();
        be.upload(wbuf.as_ref(), bytemuck::cast_slice(&wi)).unwrap();
        let xbuf = be.alloc(cc * 4, BufferUsage::Activations).unwrap();
        let obuf = be.alloc(cc * 4, BufferUsage::Readback).unwrap();
        let run1 = |xin: &[f32]| {
            be.upload(xbuf.as_ref(), bytemuck::cast_slice(xin)).unwrap();
            let mut b = Bindings::new();
            b.bind(x, xbuf.as_ref());
            b.bind(w, wbuf.as_ref());
            b.bind(state, sbuf.as_ref());
            b.bind(dst, obuf.as_ref());
            be.execute(plan.as_ref(), &b).unwrap();
        };
        run1(&x1);
        run1(&x2);
        let mut o = vec![0f32; cc];
        be.download(obuf.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = second(&cpu);
    let v = second(&vk);
    println!("state-persist 2nd-token max_err={:e}", maxerr(&c, &v));
    assert!(
        maxerr(&c, &v) < 1e-3,
        "recurrent state does NOT persist across executes on Vulkan"
    );
}

fn maxerr(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn gen(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
        .collect()
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn copystrided_parity() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // convout[rows, cc=q|k|v] → split q (first key_dim) with per-row stride cc.
    let (rows, key_dim, nv_vd) = (3usize, 8usize, 6usize);
    let cc = 2 * key_dim + nv_vd;
    let mut g = Graph::new();
    let src = g.input(f32d(rows * cc));
    let dq = g.output(f32d(rows * key_dim));
    g.push(Op::CopyStrided {
        src,
        src_off: key_dim as u32, // k slice
        src_stride: cc as u32,
        dst: dq,
        dst_off: 0,
        dst_stride: key_dim as u32,
        rows: rows as u32,
        n: key_dim as u32,
    });
    let input = gen(rows * cc, 1);
    let c = run(&cpu, &g, &[(src, &input)], &[], dq, rows * key_dim);
    let v = run(&vk, &g, &[(src, &input)], &[], dq, rows * key_dim);
    println!(
        "CopyStrided max_err={:e}\n cpu={:?}\n vk ={:?}",
        maxerr(&c, &v),
        c,
        v
    );
    assert!(maxerr(&c, &v) < 1e-5, "CopyStrided diverges");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn gated_sigmoid_parity() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nff) = (2usize, 16usize);
    let mut g = Graph::new();
    let gate = g.input(f32d(rows * nff));
    let up = g.input(f32d(rows * nff));
    let dst = g.output(f32d(rows * nff));
    g.push(Op::GatedAct {
        gate,
        up,
        dst,
        rows: rows as u32,
        nff: nff as u32,
        act: Activation::Sigmoid,
        up_off: 0,
    });
    let gi = gen(rows * nff, 2);
    let ui = gen(rows * nff, 3);
    let c = run(&cpu, &g, &[(gate, &gi), (up, &ui)], &[], dst, rows * nff);
    let v = run(&vk, &g, &[(gate, &gi), (up, &ui)], &[], dst, rows * nff);
    println!("GatedAct(sigmoid) max_err={:e}", maxerr(&c, &v));
    assert!(maxerr(&c, &v) < 1e-3, "GatedAct sigmoid diverges");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn qknorm_parity() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // per-head rmsnorm over head_dim (qwen35 ssm_norm, applied to the DeltaNet output).
    let (rows, n_head, head_dim) = (2usize, 4usize, 16usize);
    let mut g = Graph::new();
    let x = g.input(f32d(rows * n_head * head_dim));
    let w = g.weight(f32d(head_dim));
    let dst = g.output(f32d(rows * n_head * head_dim));
    g.push(Op::QkNorm {
        x,
        weight: w,
        dst,
        rows: rows as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        eps: 1e-6,
    });
    let xi = gen(rows * n_head * head_dim, 4);
    let wi = gen(head_dim, 5).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let c = run(
        &cpu,
        &g,
        &[(x, &xi)],
        &[(w, &wi)],
        dst,
        rows * n_head * head_dim,
    );
    let v = run(
        &vk,
        &g,
        &[(x, &xi)],
        &[(w, &wi)],
        dst,
        rows * n_head * head_dim,
    );
    println!("QkNorm max_err={:e}", maxerr(&c, &v));
    assert!(maxerr(&c, &v) < 1e-3, "QkNorm diverges");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn qknormrope_parity_qwen35_dims() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // qwen35 attention: head_dim=256, PARTIAL rope (rope_dim=64), batched rows>1.
    let (rows, nh, hd, rope_dim) = (15usize, 4usize, 256usize, 64usize);
    let mut g = Graph::new();
    let x = g.input(f32d(rows * nh * hd));
    let w = g.weight(f32d(hd));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let dst = g.output(f32d(rows * nh * hd));
    g.push(Op::QkNormRope {
        x,
        weight: w,
        positions: pos,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rope_dim as u32,
        theta: 1e7,
        eps: 1e-6,
        freq_factors: None,
    });
    let xi = gen(rows * nh * hd, 4);
    let wi = gen(hd, 5).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let posv: Vec<i32> = (0..rows as i32).collect();
    // positions are I32; upload the raw bytes as if f32 (same 4-byte width) via a tiny inline run.
    let run256 = |be: &dyn Backend| -> Vec<f32> {
        let plan = be.compile(&g).unwrap();
        let xb = be.alloc(xi.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(xb.as_ref(), bytemuck::cast_slice(&xi)).unwrap();
        let wb = be.alloc(wi.len() * 4, BufferUsage::Weights).unwrap();
        be.upload(wb.as_ref(), bytemuck::cast_slice(&wi)).unwrap();
        let pb = be.alloc(posv.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(pb.as_ref(), bytemuck::cast_slice(&posv)).unwrap();
        let ob = be.alloc(xi.len() * 4, BufferUsage::Readback).unwrap();
        let mut b = Bindings::new();
        b.bind(x, xb.as_ref());
        b.bind(w, wb.as_ref());
        b.bind(pos, pb.as_ref());
        b.bind(dst, ob.as_ref());
        be.execute(plan.as_ref(), &b).unwrap();
        let mut o = vec![0f32; xi.len()];
        be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = run256(&cpu);
    let v = run256(&vk);
    let nan = v.iter().any(|x| x.is_nan());
    println!(
        "QkNormRope(qwen35 hd=256,rope=64) max_err={:e} vulkan_nan={nan}",
        maxerr(&c, &v)
    );
    // NOTE: qk_norm_rope writes f16 into `dst`; declaring `dst` f32 above reads f16-packed bytes as
    // f32 → nominal max_err is huge (expected). The DECISIVE test is `qknormrope_attn_chain` below,
    // which chains QkNormRope→Attention exactly as the seam does (f16 producer→consumer, f32 out).
    let _ = (nan, c, v);
}

/// The REAL qwen35 attention handshake: QkNormRope (writes f16 q) → Attention (reads f16 q, f16 KV
/// cache, writes f32 o). Reproduces the exact producer→consumer dtype flow at qwen35 dims (hd=256,
/// PARTIAL rope=64, GQA nh=4/nkv=2, BATCHED rows>1). The dense seam never exercises attention_kv at
/// rows>1 (hd=128 → flash) and the bespoke qwen35 only runs it at rows=1, so batched attention_kv is
/// untested. Output is f32 → clean CPU-vs-Vulkan comparison. Localizes the seam NaN to this pair.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn qknormrope_attn_chain() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nh, nkv, hd, rope_dim) = (15usize, 4usize, 2usize, 256usize, 64usize);
    let kv_len = rows; // pos=0, causal: query ti attends keys [0, ti]
    let mut g = Graph::new();
    let x = g.input(f32d(rows * nh * hd));
    let qw = g.weight(f32d(hd));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let kc = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
    let qa = g.internal(f32d(rows * nh * hd));
    let dst = g.output(f32d(rows * nh * hd));
    g.push(Op::QkNormRope {
        x,
        weight: qw,
        positions: pos,
        dst: qa,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rope_dim as u32,
        theta: 1e7,
        eps: 1e-6,
        freq_factors: None,
    });
    g.push(Op::Attention {
        q: qa,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: AttnMask::Causal,
        pos: 0,
    });
    let xi = gen(rows * nh * hd, 4);
    let wi = gen(hd, 5).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let posv: Vec<i32> = (0..rows as i32).collect();
    // f16 KV cache (as the seam's WriteKv produces).
    let f16b = |vals: &[f32]| -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect()
    };
    let kf = f16b(&gen(kv_len * nkv * hd, 8));
    let vf = f16b(&gen(kv_len * nkv * hd, 9));
    let runner = |be: &dyn Backend| -> Vec<f32> {
        let plan = be.compile(&g).unwrap();
        let xb = be.alloc(xi.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(xb.as_ref(), bytemuck::cast_slice(&xi)).unwrap();
        let wb = be.alloc(wi.len() * 4, BufferUsage::Weights).unwrap();
        be.upload(wb.as_ref(), bytemuck::cast_slice(&wi)).unwrap();
        let pb = be.alloc(posv.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(pb.as_ref(), bytemuck::cast_slice(&posv)).unwrap();
        let kb = be.alloc(kf.len(), BufferUsage::Activations).unwrap();
        be.upload(kb.as_ref(), &kf).unwrap();
        let vb = be.alloc(vf.len(), BufferUsage::Activations).unwrap();
        be.upload(vb.as_ref(), &vf).unwrap();
        let ob = be.alloc(xi.len() * 4, BufferUsage::Readback).unwrap();
        let mut b = Bindings::new();
        b.bind(x, xb.as_ref());
        b.bind(qw, wb.as_ref());
        b.bind(pos, pb.as_ref());
        b.bind(kc, kb.as_ref());
        b.bind(vc, vb.as_ref());
        b.bind(dst, ob.as_ref());
        be.execute(plan.as_ref(), &b).unwrap();
        let mut o = vec![0f32; xi.len()];
        be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = runner(&cpu);
    let v = runner(&vk);
    let nan = v.iter().any(|x| x.is_nan());
    println!(
        "QkNormRope→Attention(qwen35) max_err={:e} vulkan_nan={nan}",
        maxerr(&c, &v)
    );
    assert!(!nan && maxerr(&c, &v) < 5e-2, "qwen35 attn chain diverges");
}

/// FULL qwen35 attention core in ONE graph/command buffer: QkNormRope(K)→WriteKv (fused peephole,
/// f16 cache write at rows>1) + WriteKv(V) + Attention — all reading/writing the SAME kc/vc cache
/// buffers within a single execute. This is what the seam does but the isolated chain above does
/// NOT: it tests (a) the fused K-QkNormRope→cache write at rows>1 and (b) the WriteKv→Attention
/// read-after-write ordering inside one command buffer. If THIS diverges, the bug is the in-buffer
/// KV write→read handshake (barrier) or the fused K path at batched rows.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn qwen35_attn_core_writekv() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nh, nkv, hd, rope_dim) = (15usize, 4usize, 2usize, 256usize, 64usize);
    let kv_len = rows;
    let mut g = Graph::new();
    let qx = g.input(f32d(rows * nh * hd));
    let kx = g.input(f32d(rows * nkv * hd));
    let vx = g.input(f32d(rows * nkv * hd));
    let qw = g.weight(f32d(hd));
    let kw = g.weight(f32d(hd));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let qa = g.internal(f32d(rows * nh * hd));
    // K-norm output is an F16 scratch → the Vulkan peephole fuses QkNormRope+WriteKv into a direct
    // cache write. (An F32 `ka` here reproduces the seam bug: f16 written into f32, then store_f16
    // reads it as f32 → garbage cache.)
    let ka = g.internal(TensorDesc::new(vec![rows * nkv * hd], DType::F16));
    let kc = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
    let dst = g.output(f32d(rows * nh * hd));
    let qknr = |x, weight, dst, n_head| Op::QkNormRope {
        x,
        weight,
        positions: pos,
        dst,
        rows: rows as u32,
        n_head,
        head_dim: hd as u32,
        rope_dim: rope_dim as u32,
        theta: 1e7,
        eps: 1e-6,
        freq_factors: None,
    };
    g.push(qknr(qx, qw, qa, nh as u32));
    g.push(qknr(kx, kw, ka, nkv as u32)); // fused with the next WriteKv by the peephole
    g.push(Op::WriteKv {
        src: ka,
        cache: kc,
        rows: rows as u32,
        row_stride: (nkv * hd) as u32,
        pos: 0,
    });
    g.push(Op::WriteKv {
        src: vx,
        cache: vc,
        rows: rows as u32,
        row_stride: (nkv * hd) as u32,
        pos: 0,
    });
    g.push(Op::Attention {
        q: qa,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: AttnMask::Causal,
        pos: 0,
    });
    let qi = gen(rows * nh * hd, 4);
    let ki = gen(rows * nkv * hd, 8);
    let vi = gen(rows * nkv * hd, 9);
    let qwi = gen(hd, 5).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let kwi = gen(hd, 6).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let posv: Vec<i32> = (0..rows as i32).collect();
    let out_len = rows * nh * hd;
    let cache_bytes = kv_len * nkv * hd * 2;
    let runner = |be: &dyn Backend| -> Vec<f32> {
        let plan = be.compile(&g).unwrap();
        let up = |data: &[f32], usage| {
            let b = be.alloc(data.len() * 4, usage).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(data)).unwrap();
            b
        };
        let qb = up(&qi, BufferUsage::Activations);
        let kb = up(&ki, BufferUsage::Activations);
        let vb = up(&vi, BufferUsage::Activations);
        let qwb = up(&qwi, BufferUsage::Weights);
        let kwb = up(&kwi, BufferUsage::Weights);
        let pbuf = be.alloc(posv.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(pbuf.as_ref(), bytemuck::cast_slice(&posv))
            .unwrap();
        let kcb = be.alloc(cache_bytes, BufferUsage::Activations).unwrap(); // zeroed
        let vcb = be.alloc(cache_bytes, BufferUsage::Activations).unwrap();
        let ob = be.alloc(out_len * 4, BufferUsage::Readback).unwrap();
        let mut b = Bindings::new();
        b.bind(qx, qb.as_ref());
        b.bind(kx, kb.as_ref());
        b.bind(vx, vb.as_ref());
        b.bind(qw, qwb.as_ref());
        b.bind(kw, kwb.as_ref());
        b.bind(pos, pbuf.as_ref());
        b.bind(kc, kcb.as_ref());
        b.bind(vc, vcb.as_ref());
        b.bind(dst, ob.as_ref());
        be.execute(plan.as_ref(), &b).unwrap();
        let mut o = vec![0f32; out_len];
        be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = runner(&cpu);
    let v = runner(&vk);
    let nan = v.iter().any(|x| x.is_nan());
    println!(
        "qwen35 attn-core(WriteKv) max_err={:e} vulkan_nan={nan}",
        maxerr(&c, &v)
    );
    assert!(
        !nan && maxerr(&c, &v) < 5e-2,
        "qwen35 attn core (WriteKv) diverges"
    );
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn conv1d_silu_parity() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, cc, kernel) = (4usize, 32usize, 4usize);
    let mut g = Graph::new();
    let x = g.input(f32d(rows * cc));
    let w = g.weight(f32d(cc * kernel));
    let state = g.input(f32d((kernel - 1) * cc)); // zeroed history (calloc)
    let dst = g.output(f32d(rows * cc));
    g.push(Op::Conv1dSilu {
        x,
        weight: w,
        state,
        dst,
        rows: rows as u32,
        channels: cc as u32,
        kernel: kernel as u32,
    });
    let xi = gen(rows * cc, 6);
    let wi = gen(cc * kernel, 7);
    let st = vec![0f32; (kernel - 1) * cc];
    let c = run(
        &cpu,
        &g,
        &[(x, &xi), (state, &st)],
        &[(w, &wi)],
        dst,
        rows * cc,
    );
    let v = run(
        &vk,
        &g,
        &[(x, &xi), (state, &st)],
        &[(w, &wi)],
        dst,
        rows * cc,
    );
    println!("Conv1dSilu max_err={:e}", maxerr(&c, &v));
    assert!(maxerr(&c, &v) < 1e-3, "Conv1dSilu diverges");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn deltanet_chunked_parity() {
    // rows ≥ 32 routes to the CHUNKED delta-rule kernel (deltanet_chunked.comp): qwen35-like dims,
    // GQA tiling, a NONZERO initial state (exercises the cross-chunk carry) and a partial last
    // chunk (130 = 4×32 + 2). The CPU oracle is the sequential recurrence.
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nv, nk, kd, vd) = (130usize, 8usize, 4usize, 128usize, 128usize);
    let mut g = Graph::new();
    let q = g.input(f32d(rows * nk * kd));
    let k = g.input(f32d(rows * nk * kd));
    let v = g.input(f32d(rows * nv * vd));
    let b = g.input(f32d(rows * nv));
    let a = g.input(f32d(rows * nv));
    let a_coef = g.weight(f32d(nv));
    let dt_bias = g.weight(f32d(nv));
    let state = g.input(f32d(nv * kd * vd));
    let dst = g.output(f32d(rows * nv * vd));
    g.push(Op::DeltaNet {
        q,
        k,
        v,
        b,
        a,
        a_coef,
        dt_bias,
        state,
        dst,
        rows: rows as u32,
        n_vhead: nv as u32,
        n_khead: nk as u32,
        head_k: kd as u32,
        head_v: vd as u32,
        eps: 1e-6,
    });
    let (qi, ki, vi) = (
        gen(rows * nk * kd, 1),
        gen(rows * nk * kd, 2),
        gen(rows * nv * vd, 3),
    );
    let (bi, ai) = (gen(rows * nv, 4), gen(rows * nv, 5));
    // a_coef must be negative (log-decay scale); gen() is symmetric, so force sign.
    let aci: Vec<f32> = gen(nv, 8).iter().map(|x| -x.abs() - 0.1).collect();
    let dti = gen(nv, 9);
    let st = gen(nv * kd * vd, 10);
    let ins = [
        (q, &qi[..]),
        (k, &ki[..]),
        (v, &vi[..]),
        (b, &bi[..]),
        (a, &ai[..]),
        (state, &st[..]),
    ];
    let ws = [(a_coef, &aci[..]), (dt_bias, &dti[..])];
    let c = run(&cpu, &g, &ins, &ws, dst, rows * nv * vd);
    let vv = run(&vk, &g, &ins, &ws, dst, rows * nv * vd);
    let e = maxerr(&c, &vv);
    println!("DeltaNet-chunked rows={rows} max_err={e:e}");
    assert!(
        e < 1e-3,
        "chunked DeltaNet diverges from the sequential oracle"
    );
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn deltanet_parity() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nv, nk, kd, vd) = (4usize, 4usize, 2usize, 16usize, 16usize);
    let mut g = Graph::new();
    let q = g.input(f32d(rows * nk * kd));
    let k = g.input(f32d(rows * nk * kd));
    let v = g.input(f32d(rows * nv * vd));
    let b = g.input(f32d(rows * nv));
    let a = g.input(f32d(rows * nv));
    let a_coef = g.weight(f32d(nv));
    let dt_bias = g.weight(f32d(nv));
    let state = g.input(f32d(nv * kd * vd)); // zeroed
    let dst = g.output(f32d(rows * nv * vd));
    g.push(Op::DeltaNet {
        q,
        k,
        v,
        b,
        a,
        a_coef,
        dt_bias,
        state,
        dst,
        rows: rows as u32,
        n_vhead: nv as u32,
        n_khead: nk as u32,
        head_k: kd as u32,
        head_v: vd as u32,
        eps: 1e-6,
    });
    let (qi, ki, vi) = (
        gen(rows * nk * kd, 1),
        gen(rows * nk * kd, 2),
        gen(rows * nv * vd, 3),
    );
    let (bi, ai) = (gen(rows * nv, 4), gen(rows * nv, 5));
    let (aci, dti) = (gen(nv, 8), gen(nv, 9));
    let st = vec![0f32; nv * kd * vd];
    let ins = [
        (q, &qi[..]),
        (k, &ki[..]),
        (v, &vi[..]),
        (b, &bi[..]),
        (a, &ai[..]),
        (state, &st[..]),
    ];
    let ws = [(a_coef, &aci[..]), (dt_bias, &dti[..])];
    let c = run(&cpu, &g, &ins, &ws, dst, rows * nv * vd);
    let vv = run(&vk, &g, &ins, &ws, dst, rows * nv * vd);
    println!(
        "DeltaNet max_err={:e}\n cpu={:?}\n vk ={:?}",
        maxerr(&c, &vv),
        c,
        vv
    );
    assert!(maxerr(&c, &vv) < 1e-2, "DeltaNet diverges");
}
