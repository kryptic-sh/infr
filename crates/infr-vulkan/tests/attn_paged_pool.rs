//! Forward PagedAttention: two slots that share a prefix read the same physical
//! KV pages, and each one's attention output is bit-identical to the same rows
//! laid out contiguously.
//!
//! Slot A owns three pages; slot B forks A's mapping and maps one page of its
//! own. B's logical rows 0..1024 are A's pages 0 and 1, so a write to B must not
//! touch them — the pool's refcounts are what make that a check rather than a
//! promise.
//!
//! Run: `cargo test --release -p infr-vulkan --test attn_paged_pool`

use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::config::Config;
use infr_vulkan::paged::{PagePool, SlotPages, PAGE_ROWS};
use infr_vulkan::VulkanBackend;
use std::sync::Arc;
use std::time::Instant;

const NH: usize = 8;
const NKV: usize = 2;
const HD: usize = 128;
const CHUNK: usize = 512;

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    }
}

fn f16_data(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        out.extend_from_slice(&half::f16::from_f32(r.next_f32()).to_bits().to_le_bytes());
    }
    out
}

#[derive(Clone, Copy)]
enum Layout<'a> {
    /// Rows in logical order, one buffer.
    Linear(&'a dyn Buffer),
    /// Rows through a page table.
    Paged(&'a dyn Buffer, &'a dyn Buffer, u32),
}

struct Out {
    bits: Vec<u32>,
    us: f64,
}

#[allow(clippy::too_many_arguments)]
fn run(
    be: &VulkanBackend,
    q: &dyn Buffer,
    k: Layout<'_>,
    v: Layout<'_>,
    kv_len: usize,
    chunk: usize,
    reps: usize,
) -> Out {
    let n_chunks = kv_len.div_ceil(chunk);
    let pm = be
        .alloc(NH * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(NH * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(NH * n_chunks * HD * 4, BufferUsage::Activations)
        .unwrap();
    let out_bytes = NH * HD * 4;
    let out = be.alloc(out_bytes, BufferUsage::Activations).unwrap();
    let (kb, ka) = match k {
        Layout::Linear(b) => (b as &dyn Buffer, b.device_addr().unwrap()),
        Layout::Paged(b, t, rows) => {
            assert_eq!(rows as usize, PAGE_ROWS);
            (b, t.device_addr().unwrap())
        }
    };
    let (vb, va) = match v {
        Layout::Linear(b) => (b as &dyn Buffer, b.device_addr().unwrap()),
        Layout::Paged(b, _, _) => (b, b.device_addr().unwrap()),
    };
    let pages = match k {
        Layout::Linear(_) => None,
        Layout::Paged(_, t, rows) => Some((t.device_addr().unwrap(), rows)),
    };
    let _ = ka;
    let _ = va;

    let rec = be.recorder().unwrap();
    let start = Instant::now();
    for _ in 0..reps {
        rec.attention_kv_split_at(
            q,
            kb,
            vb,
            kb.device_addr().unwrap(),
            vb.device_addr().unwrap(),
            out.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            1,
            kv_len - 1,
            kv_len,
            NH,
            NKV,
            HD,
            chunk,
            n_chunks,
            0.0,
            0,
            None,
            false,
            false,
            0,
            false,
            None,
            pages,
        );
    }
    rec.finish().unwrap();
    let mut bytes = vec![0u8; out_bytes];
    be.download(out.as_ref(), &mut bytes).unwrap();
    let us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
    Out {
        bits: bytemuck::cast_slice::<u8, u32>(&bytes).to_vec(),
        us,
    }
}

/// The record-once replay leg: same dispatch, but the page table's address rides the params
/// SSBO's words 2-3 (the tape is recorded once, so it cannot be a push constant) and only the
/// page size is pushed. Must produce the same bits as the static paged leg.
fn run_replay(
    be: &VulkanBackend,
    q: &dyn Buffer,
    k: Layout<'_>,
    v: Layout<'_>,
    kv_len: usize,
    chunk: usize,
) -> Vec<u32> {
    let n_chunks = kv_len.div_ceil(chunk);
    let pm = be
        .alloc(NH * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(NH * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(NH * n_chunks * HD * 4, BufferUsage::Activations)
        .unwrap();
    let args = be.alloc(16, BufferUsage::Activations).unwrap();
    let out_bytes = NH * HD * 4;
    let out = be.alloc(out_bytes, BufferUsage::Activations).unwrap();
    let (kb, vb, table, page_rows) = match (k, v) {
        (Layout::Paged(kb, t, rows), Layout::Paged(vb, _, _)) => {
            (kb, vb, Some(t.device_addr().unwrap()), rows)
        }
        (Layout::Linear(kb), Layout::Linear(vb)) => (kb, vb, None, 0),
        _ => unreachable!("replay leg takes both sides paged or both linear"),
    };
    let params = be.alloc(24, BufferUsage::Activations).unwrap();
    let mut pb = [0u8; 24];
    pb[0..4].copy_from_slice(&((kv_len - 1) as u32).to_le_bytes());
    pb[4..8].copy_from_slice(&(kv_len as u32).to_le_bytes());
    if let Some(t) = table {
        pb[8..12].copy_from_slice(&(t as u32).to_le_bytes());
        pb[12..16].copy_from_slice(&((t >> 32) as u32).to_le_bytes());
    }
    be.upload(params.as_ref(), &pb).unwrap();

    let rec = be.recorder().unwrap();
    rec.attn_live_prologue(params.as_ref(), args.as_ref(), NH, chunk, 0);
    rec.attention_kv_split_dynac_at(
        q,
        kb,
        vb,
        kb.device_addr().unwrap(),
        vb.device_addr().unwrap(),
        out.as_ref(),
        pm.as_ref(),
        pl.as_ref(),
        pacc.as_ref(),
        params.as_ref(),
        args.as_ref(),
        NH,
        NKV,
        HD,
        chunk,
        n_chunks,
        0.0,
        0,
        false,
        0,
        page_rows as usize,
    );
    rec.finish().unwrap();
    let mut bytes = vec![0u8; out_bytes];
    be.download(out.as_ref(), &mut bytes).unwrap();
    bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
}

#[test]
fn two_slots_share_a_prefix_and_stay_bit_identical() {
    let be = match VulkanBackend::new_with(Arc::new(Config::default())) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("skipping: no Vulkan device ({e})");
            return;
        }
    };
    let page_elems = PAGE_ROWS * NKV * HD;
    // The pool buffer is sized for the long-context leg below; the sharing leg
    // only hands out four of these pages.
    let pages = 16usize;
    let mut pool = PagePool::new(6);

    // Slot A: three pages. Slot B: shares A's first two, then one of its own.
    let mut a = SlotPages::new();
    let mut b = SlotPages::new();
    for _ in 0..3 {
        assert!(a.grow(&mut pool));
    }
    b.fork(&mut pool, &a);
    b.truncate(&mut pool, 2);
    assert!(b.grow(&mut pool));
    let pa = a.table();
    assert_eq!(pool.refs(pa[0]), 2, "B must be sharing A's first page");
    assert_eq!(pool.refs(pa[2]), 1, "B must not be sharing A's tail page");
    assert_eq!(pool.in_use(), 4);

    // Physical contents, one distinct stream per page.
    let kb = be
        .alloc(pages * page_elems * 2, BufferUsage::KvCache)
        .unwrap();
    let vb = be
        .alloc(pages * page_elems * 2, BufferUsage::KvCache)
        .unwrap();
    let mut kbytes = Vec::with_capacity(pages * page_elems * 2);
    let mut vbytes = Vec::with_capacity(pages * page_elems * 2);
    for p in 0..pages {
        kbytes.extend_from_slice(&f16_data(page_elems, 3000 + p as u64));
        vbytes.extend_from_slice(&f16_data(page_elems, 5000 + p as u64));
    }
    be.upload(kb.as_ref(), &kbytes).unwrap();
    be.upload(vb.as_ref(), &vbytes).unwrap();

    // The same rows, contiguous, keyed by the logical page order of each slot.
    let linear = |order: &[u32]| {
        let mut k = Vec::with_capacity(order.len() * page_elems * 2);
        let mut v = Vec::with_capacity(order.len() * page_elems * 2);
        for &p in order {
            let o = p as usize * page_elems * 2;
            k.extend_from_slice(&kbytes[o..o + page_elems * 2]);
            v.extend_from_slice(&vbytes[o..o + page_elems * 2]);
        }
        let kb = be.alloc(k.len(), BufferUsage::KvCache).unwrap();
        let vb = be.alloc(v.len(), BufferUsage::KvCache).unwrap();
        be.upload(kb.as_ref(), &k).unwrap();
        be.upload(vb.as_ref(), &v).unwrap();
        (kb, vb)
    };

    let q = be.alloc(NH * HD * 2, BufferUsage::Activations).unwrap();
    be.upload(q.as_ref(), &f16_data(NH * HD, 777)).unwrap();

    let tables: Vec<Box<dyn Buffer>> =
        [(0..3).collect::<Vec<_>>(), (0..2).chain(Some(3)).collect()]
            .iter()
            .map(|order: &Vec<u32>| {
                let buf = be.alloc(order.len() * 4 * 4, BufferUsage::KvCache).unwrap();
                let mut t: Vec<u32> = order.clone();
                t.push(*t.last().unwrap());
                let mut bytes = Vec::with_capacity(t.len() * 4);
                for x in &t {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                be.upload(buf.as_ref(), &bytes).unwrap();
                buf
            })
            .collect();

    let kv_len = PAGE_ROWS * 3;
    for (i, (order, table)) in [(vec![0u32, 1, 2], &tables[0]), (vec![0, 1, 3], &tables[1])]
        .into_iter()
        .enumerate()
    {
        let (lk, lv) = linear(&order);
        let want = run(
            &be,
            q.as_ref(),
            Layout::Linear(lk.as_ref()),
            Layout::Linear(lv.as_ref()),
            kv_len,
            CHUNK,
            1,
        );
        let got = run(
            &be,
            q.as_ref(),
            Layout::Paged(kb.as_ref(), table.as_ref(), PAGE_ROWS as u32),
            Layout::Paged(vb.as_ref(), table.as_ref(), PAGE_ROWS as u32),
            kv_len,
            CHUNK,
            1,
        );
        assert_eq!(
            got.bits, want.bits,
            "slot {i} must read the same bits through pages"
        );
        assert!(walk_nonzero(&want.bits) > 0, "slot {i} output is all zero");
        let rep = run_replay(
            &be,
            q.as_ref(),
            Layout::Paged(kb.as_ref(), table.as_ref(), PAGE_ROWS as u32),
            Layout::Paged(vb.as_ref(), table.as_ref(), PAGE_ROWS as u32),
            kv_len,
            CHUNK,
        );
        assert_eq!(
            rep, got.bits,
            "slot {i}: the replay tape must read the same pages through params words 2-3"
        );
    }

    // Cost of the indirection at a longer context: 16 pages, 8192 rows.
    let long = pages;
    let mut c = SlotPages::new();
    let mut pool2 = PagePool::new(long);
    for _ in 0..long {
        assert!(c.grow(&mut pool2));
    }
    let t = be.alloc((long + 1) * 4, BufferUsage::KvCache).unwrap();
    let mut bytes = Vec::with_capacity((long + 1) * 4);
    for x in &c.table() {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    be.upload(t.as_ref(), &bytes).unwrap();
    let long_len = PAGE_ROWS * long;
    let mut lin_us = Vec::new();
    let mut pg_us = Vec::new();
    for paged in [false, true, true, false] {
        let out = if paged {
            run(
                &be,
                q.as_ref(),
                Layout::Paged(kb.as_ref(), t.as_ref(), PAGE_ROWS as u32),
                Layout::Paged(vb.as_ref(), t.as_ref(), PAGE_ROWS as u32),
                long_len,
                CHUNK,
                40,
            )
        } else {
            run(
                &be,
                q.as_ref(),
                // Same physical pages, addressed as one contiguous run, so the
                // only difference between the legs is the indirection itself.
                Layout::Linear(kb.as_ref()),
                Layout::Linear(vb.as_ref()),
                long_len,
                CHUNK,
                40,
            )
        };
        if paged {
            pg_us.push(out.us);
        } else {
            lin_us.push(out.us);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!(
        "attn_decode kv=8192: contiguous {:.1} us/dispatch, paged {:.1} us/dispatch ({:+.1}%)",
        mean(&lin_us),
        mean(&pg_us),
        (mean(&pg_us) / mean(&lin_us) - 1.0) * 100.0
    );
}

fn walk_nonzero(bits: &[u32]) -> usize {
    bits.iter().filter(|b| **b & 0x7fff_ffff != 0).count()
}

/// A write lands at a page's byte offset inside the pool, and the same rows read back
/// through the table bit-for-bit — the store side of the pool, minus the kernel that
/// computes the address.
#[test]
fn a_write_at_a_page_offset_reads_back_through_the_table() {
    let be = match VulkanBackend::new_with(Arc::new(Config::default())) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("skipping: no Vulkan device ({e})");
            return;
        }
    };
    let page_elems = PAGE_ROWS * NKV * HD;
    let bytes = page_elems * 2;
    let k_data = f16_data(page_elems, 9001);
    let v_data = f16_data(page_elems, 9002);

    let kb = be.alloc(bytes * 2, BufferUsage::KvCache).unwrap();
    let vb = be.alloc(bytes * 2, BufferUsage::KvCache).unwrap();
    be.upload_at(kb.as_ref(), bytes, &k_data).unwrap();
    be.upload_at(vb.as_ref(), bytes, &v_data).unwrap();

    let mut mirrored = vec![0u8; bytes];
    be.download_at(kb.as_ref(), bytes, &mut mirrored).unwrap();
    assert_eq!(mirrored, k_data, "the write must land at the page offset");

    let lk = be.alloc(bytes, BufferUsage::KvCache).unwrap();
    let lv = be.alloc(bytes, BufferUsage::KvCache).unwrap();
    be.upload(lk.as_ref(), &k_data).unwrap();
    be.upload(lv.as_ref(), &v_data).unwrap();

    let table = be.alloc(8, BufferUsage::KvCache).unwrap();
    be.upload(
        table.as_ref(),
        &[1u32.to_le_bytes(), 1u32.to_le_bytes()].concat(),
    )
    .unwrap();

    let q = be.alloc(NH * HD * 2, BufferUsage::Activations).unwrap();
    be.upload(q.as_ref(), &f16_data(NH * HD, 4242)).unwrap();

    let want = run(
        &be,
        q.as_ref(),
        Layout::Linear(lk.as_ref()),
        Layout::Linear(lv.as_ref()),
        PAGE_ROWS,
        CHUNK,
        1,
    );
    let got = run(
        &be,
        q.as_ref(),
        Layout::Paged(kb.as_ref(), table.as_ref(), PAGE_ROWS as u32),
        Layout::Paged(vb.as_ref(), table.as_ref(), PAGE_ROWS as u32),
        PAGE_ROWS,
        CHUNK,
        1,
    );
    assert_eq!(got.bits, want.bits);
    assert!(walk_nonzero(&want.bits) > 0);
}
