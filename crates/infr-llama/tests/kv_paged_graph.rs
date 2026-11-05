//! A graph whose KV caches describe a whole page pool (not one slot's rows) must attend the same
//! bits as the same rows laid out contiguously. This is the seam's descriptor shape under
//! `kv.paged`: the cache input is pool-sized, carries a [`PageMap`], and the slot's rows sit at
//! `row_base` inside it.
//!
//! Run: `cargo test -p infr-llama --test kv_paged_graph`

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{AttnMask, Graph, Op};
use infr_core::tensor::{PageMap, TensorDesc};
use infr_core::{DType, TensorId};
use infr_vulkan::paged::PAGE_ROWS;

const NH: usize = 4;
const NKV: usize = 2;
const HD: usize = 64;
/// One prefill chunk, well inside the slot's first logical page.
const ROWS: usize = 256;
/// The slot under test lives in pool block 1, so a store that forgets the row base is visible.
const ROW_BASE: usize = PAGE_ROWS;

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    }
}

fn gen(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| r.next_f32()).collect()
}

fn f32d(n: usize) -> TensorDesc {
    TensorDesc::new(vec![n], DType::F32)
}

fn f16d(n: usize) -> TensorDesc {
    TensorDesc::new(vec![n], DType::F16)
}

/// The attention core over caches of `cache_elems` elements per side. `paged` is `(row_base,
/// table_len)` for a cache the slot reaches through a page table, `None` for a contiguous one.
struct Built {
    g: Graph,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    kc: TensorId,
    vc: TensorId,
    table: Option<TensorId>,
    out: TensorId,
}

fn attention_graph(cache_elems: usize, paged: Option<(usize, usize)>) -> Built {
    let mut g = Graph::new();
    let q = g.input(f16d(ROWS * NH * HD));
    let k = g.input(f32d(ROWS * NKV * HD));
    let v = g.input(f32d(ROWS * NKV * HD));
    let table = paged.map(|(_, len)| g.input(TensorDesc::new(vec![len], DType::I32)));
    let mut kc_desc = f16d(cache_elems);
    let mut vc_desc = f16d(cache_elems);
    if let (Some(t), Some((row_base, _))) = (table, paged) {
        kc_desc.pages = Some(PageMap {
            table: t,
            page_rows: PAGE_ROWS as u32,
            row_base,
        });
        vc_desc.pages = Some(PageMap {
            table: t,
            page_rows: PAGE_ROWS as u32,
            row_base,
        });
    }
    let kc = g.input(kc_desc);
    let vc = g.input(vc_desc);
    let out = g.output(f32d(ROWS * NH * HD));
    g.push(Op::WriteKv {
        src: k,
        cache: kc,
        rows: ROWS as u32,
        row_stride: (NKV * HD) as u32,
        pos: 0,
    });
    g.push(Op::WriteKv {
        src: v,
        cache: vc,
        rows: ROWS as u32,
        row_stride: (NKV * HD) as u32,
        pos: 0,
    });
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst: out,
        rows: ROWS as u32,
        kv_len: ROWS as u32,
        n_head: NH as u32,
        n_kv: NKV as u32,
        head_dim: HD as u32,
        scale: 1.0 / (HD as f32).sqrt(),
        mask: AttnMask::Causal,
        pos: 0,
        sinks: None,
        key_bias: None,
    });
    Built {
        g,
        q,
        k,
        v,
        kc,
        vc,
        table,
        out,
    }
}

fn table_bytes(pages: usize) -> Vec<u8> {
    // Every logical page maps to pool page 1 — block 1's only page.
    (0..=pages)
        .flat_map(|_| 1u32.to_le_bytes())
        .collect::<Vec<u8>>()
}

#[allow(clippy::too_many_arguments)]
fn run(
    be: &infr_vulkan::VulkanBackend,
    built: &Built,
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    cache_bytes: usize,
    table: Option<&[u8]>,
    out_len: usize,
) -> Vec<f32> {
    let plan = be.compile(&built.g).expect("compile");
    let f16 = |data: &[f32]| {
        let bytes = data
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
            .collect::<Vec<u8>>();
        let b = be.alloc(bytes.len(), BufferUsage::Activations).unwrap();
        be.upload(b.as_ref(), &bytes).unwrap();
        b
    };
    let f32b = |data: &[f32]| {
        let b = be.alloc(data.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(data)).unwrap();
        b
    };
    let qb = f16(q_data);
    let kb = f32b(k_data);
    let vb = f32b(v_data);
    let kc = be.alloc(cache_bytes, BufferUsage::KvCache).unwrap();
    let vc = be.alloc(cache_bytes, BufferUsage::KvCache).unwrap();
    let ob = be.alloc(out_len * 4, BufferUsage::Readback).unwrap();
    let tb = table.map(|bytes| {
        let t = be.alloc(bytes.len(), BufferUsage::KvCache).unwrap();
        be.upload(t.as_ref(), bytes).unwrap();
        t
    });
    let mut b = Bindings::new();
    b.bind(built.q, qb.as_ref());
    b.bind(built.k, kb.as_ref());
    b.bind(built.v, vb.as_ref());
    b.bind(built.kc, kc.as_ref());
    b.bind(built.vc, vc.as_ref());
    b.bind(built.out, ob.as_ref());
    if let (Some(t), Some(id)) = (tb.as_ref(), built.table) {
        b.bind(id, t.as_ref());
    }
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; out_len];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

#[test]
fn a_paged_cache_attends_the_same_bits_as_a_contiguous_one() {
    let be = match infr_vulkan::VulkanBackend::new_with(std::sync::Arc::new(
        infr_core::config::Config::default(),
    )) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("skipping: no Vulkan device ({e})");
            return;
        }
    };
    let row = NKV * HD;
    let q_data = gen(ROWS * NH * HD, 11);
    let k_data = gen(ROWS * row, 12);
    let v_data = gen(ROWS * row, 13);
    let out_len = ROWS * NH * HD;

    // Contiguous: the slot's rows are the whole cache.
    let lin_elems = ROWS * row;
    let lin = attention_graph(lin_elems, None);
    let want = run(
        &be,
        &lin,
        &q_data,
        &k_data,
        &v_data,
        lin_elems * 2,
        None,
        out_len,
    );

    // Paged: the pool holds two blocks of one page, the slot's rows start at ROW_BASE, and its
    // single logical page maps onto block 1's page.
    let pps = ROW_BASE.div_ceil(PAGE_ROWS);
    let pool_rows = 2 * pps * PAGE_ROWS;
    let pool_elems = pool_rows * row;
    let pg = attention_graph(pool_elems, Some((ROW_BASE, pps + 1)));
    let table = table_bytes(pps);
    let got = run(
        &be,
        &pg,
        &q_data,
        &k_data,
        &v_data,
        pool_elems * 2,
        Some(&table),
        out_len,
    );

    assert!(
        want.iter().any(|x| *x != 0.0),
        "the output must not be all zero"
    );
    // The two legs can run different prefill tiers (the paged one is split-K only), so the f32
    // accumulation order differs; a wrong page would move values by far more than this.
    let max_err = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-4,
        "the paged graph diverges: max_err={max_err:e}"
    );
}

/// The decode shape: one query row at the end of `DEPTH` cached rows, all of them written by the
/// same execute. A paged slot reads the whole span through its table from `row_base`.
const DEPTH: usize = 200;

fn decode_graph(cache_elems: usize, paged: Option<(usize, usize)>) -> (Built, usize) {
    let kv_len = DEPTH + 1;
    let mut g = Graph::new();
    let q = g.input(f16d(NH * HD));
    let k = g.input(f32d(kv_len * NKV * HD));
    let v = g.input(f32d(kv_len * NKV * HD));
    let table = paged.map(|(_, len)| g.input(TensorDesc::new(vec![len], DType::I32)));
    let mut kc_desc = f16d(cache_elems);
    let mut vc_desc = f16d(cache_elems);
    if let (Some(t), Some((row_base, _))) = (table, paged) {
        kc_desc.pages = Some(PageMap {
            table: t,
            page_rows: PAGE_ROWS as u32,
            row_base,
        });
        vc_desc.pages = Some(PageMap {
            table: t,
            page_rows: PAGE_ROWS as u32,
            row_base,
        });
    }
    let kc = g.input(kc_desc);
    let vc = g.input(vc_desc);
    let out = g.output(f32d(NH * HD));
    g.push(Op::WriteKv {
        src: k,
        cache: kc,
        rows: kv_len as u32,
        row_stride: (NKV * HD) as u32,
        pos: 0,
    });
    g.push(Op::WriteKv {
        src: v,
        cache: vc,
        rows: kv_len as u32,
        row_stride: (NKV * HD) as u32,
        pos: 0,
    });
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst: out,
        rows: 1,
        kv_len: kv_len as u32,
        n_head: NH as u32,
        n_kv: NKV as u32,
        head_dim: HD as u32,
        scale: 1.0 / (HD as f32).sqrt(),
        mask: AttnMask::Causal,
        pos: DEPTH as u32,
        sinks: None,
        key_bias: None,
    });
    (
        Built {
            g,
            q,
            k,
            v,
            kc,
            vc,
            table,
            out,
        },
        kv_len,
    )
}

#[test]
fn a_paged_decode_over_a_written_span_matches_a_contiguous_one() {
    let be = match infr_vulkan::VulkanBackend::new_with(std::sync::Arc::new(
        infr_core::config::Config::default(),
    )) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("skipping: no Vulkan device ({e})");
            return;
        }
    };
    let row = NKV * HD;
    let (lin, kv_len) = decode_graph((DEPTH + 1) * NKV * HD, None);
    let k_data = gen(kv_len * row, 21);
    let v_data = gen(kv_len * row, 22);
    let q_data = gen(NH * HD, 23);
    let want = run(
        &be,
        &lin,
        &q_data,
        &k_data,
        &v_data,
        kv_len * row * 2,
        None,
        NH * HD,
    );

    let pps = ROW_BASE.div_ceil(PAGE_ROWS);
    let pool_elems = 2 * pps * PAGE_ROWS * row;
    let (pg, _) = decode_graph(pool_elems, Some((ROW_BASE, pps + 1)));
    let table = table_bytes(pps);
    let got = run(
        &be,
        &pg,
        &q_data,
        &k_data,
        &v_data,
        pool_elems * 2,
        Some(&table),
        NH * HD,
    );

    assert!(
        want.iter().any(|x| *x != 0.0),
        "the output must not be all zero"
    );
    let max_err = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-4,
        "the paged decode diverges: max_err={max_err:e}"
    );
}
