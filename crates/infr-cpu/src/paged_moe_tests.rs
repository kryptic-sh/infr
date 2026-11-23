//! `Op::MoeFfn` over expert banks paged per expert ([`paged::PagedWeights::register_experts`]),
//! held against the same banks held whole: the bytes it computes with, which experts it reads,
//! and how the expert pool behaves under a prefill sweep.

use super::test_util::det_quant_rows;
use super::*;
use infr_core::blockio::{BlockDesc, BlockIo};
use infr_core::graph::{MoeGating, WeightReuse};
use infr_core::tensor::TensorDesc;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::time::Duration;

/// A model file held in memory. Counts its reads and records the most it ever had in flight at
/// once; each read sleeps `delay` first, so that reads issued together overlap.
struct MemIo {
    bytes: Vec<u8>,
    delay: Duration,
    reads: AtomicUsize,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

impl MemIo {
    fn new(bytes: Vec<u8>, delay: Duration) -> Self {
        Self {
            bytes,
            delay,
            reads: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
        }
    }
    fn reads(&self) -> usize {
        self.reads.load(SeqCst)
    }
}

impl BlockIo for MemIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> infr_core::error::Result<()> {
        let now = self.in_flight.fetch_add(1, SeqCst) + 1;
        self.max_in_flight.fetch_max(now, SeqCst);
        std::thread::sleep(self.delay);
        let mut at = 0;
        for e in &desc.extents {
            let o = e.offset as usize;
            dst[at..at + e.len].copy_from_slice(&self.bytes[o..o + e.len]);
            at += e.len;
        }
        self.in_flight.fetch_sub(1, SeqCst);
        self.reads.fetch_add(1, SeqCst);
        Ok(())
    }
}

/// `n` values in `[-0.5, 0.5)` from a xorshift stream seeded by `seed`.
fn rng(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        })
        .collect()
}

/// `rows` weight rows of `in_f` inputs in `dt`, deterministic in `seed`.
fn weight_bytes(dt: DType, rows: usize, in_f: usize, seed: u64) -> Vec<u8> {
    let quant = |blk_elems: usize, blk_bytes: usize, d_offs: &[usize]| {
        det_quant_rows(rows, in_f / blk_elems, blk_bytes, d_offs, seed)
    };
    match dt {
        DType::F32 => bytemuck::cast_slice(&rng(seed, rows * in_f)).to_vec(),
        DType::Q8_0 => quant(32, 34, &[0]),
        DType::Q4K => quant(256, 144, &[0, 2]),
        DType::Q2K => quant(256, 84, &[80, 82]),
        other => panic!("no fixture for {other:?} banks"),
    }
}

/// A stack of `Op::MoeFfn` layers, each feeding the next.
struct Moe {
    dt: DType,
    /// One `[n_expert, 2*nff, ne]` gate_up bank per layer instead of separate gate and up banks.
    fused: bool,
    layers: usize,
    ne: usize,
    nff: usize,
    n_expert: usize,
    n_used: usize,
    rows: usize,
}

/// A [`Moe`] as a graph, with its banks' bytes in binding order (layer by layer: gate or gate_up,
/// then up when split, then down) and its routers.
struct Built {
    g: Graph,
    x: TensorId,
    y: TensorId,
    routers: Vec<(TensorId, Vec<f32>)>,
    /// Each bank's graph handle, layer and bytes, in binding order.
    banks: Vec<(TensorId, usize, Vec<u8>)>,
}

impl Moe {
    fn build(&self) -> Built {
        let (ne, nff, n_expert) = (self.ne, self.nff, self.n_expert);
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![self.rows, ne], DType::F32));
        let y = g.output(TensorDesc::new(vec![self.rows, ne], DType::F32));
        let (mut routers, mut banks) = (Vec::new(), Vec::new());
        let mut cur = x;
        for l in 0..self.layers {
            let seed = 100 * l as u64;
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::F32));
            routers.push((ri, rng(seed + 1, n_expert * ne)));
            let mut bank = |rows: usize, in_f: usize, s: u64| {
                let id = g.weight(TensorDesc::new(vec![n_expert, rows, in_f], self.dt));
                banks.push((
                    id,
                    l,
                    weight_bytes(self.dt, n_expert * rows, in_f, seed + s),
                ));
                id
            };
            let (gate, up) = if self.fused {
                let gu = bank(2 * nff, ne, 2);
                (gu, gu)
            } else {
                (bank(nff, ne, 2), bank(nff, ne, 3))
            };
            let down = bank(ne, nff, 4);
            let dst = if l + 1 == self.layers {
                y
            } else {
                g.internal(TensorDesc::new(vec![self.rows, ne], DType::F32))
            };
            g.push(Op::MoeFfn {
                x: cur,
                router_x: cur,
                router: ri,
                gate_exps: gate,
                up_exps: up,
                down_exps: down,
                down_scale: None,
                fused_gate_up: self.fused,
                dst,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: self.n_used as u32,
                n_ff_exp: nff as u32,
                scale: 1.0,
                act: Activation::Silu,
                gating: MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            cur = dst;
        }
        Built {
            g,
            x,
            y,
            routers,
            banks,
        }
    }
}

impl Built {
    /// The bytes one op pins at most: every expert of each of one layer's banks. The expert pools'
    /// combined floor, by `plan_pools`' rule.
    fn floor_bytes(&self) -> usize {
        self.banks
            .iter()
            .filter(|(_, l, _)| *l == 0)
            .map(|(_, _, b)| b.len())
            .sum()
    }

    /// Each bank as the planner sees it.
    fn pageable(&self, n_expert: usize, n_used: usize) -> Vec<paged::Pageable> {
        self.banks
            .iter()
            .map(|(_, layer, b)| paged::Pageable {
                nbytes: b.len(),
                reads: paged::Reads::Experts {
                    n_expert,
                    n_used,
                    layer: *layer,
                },
            })
            .collect()
    }

    /// Run the graph on `x` with the banks bound to `banks` (binding order).
    fn run(&self, be: &CpuBackend, banks: &[&dyn Buffer], x: &[f32]) -> Vec<f32> {
        let upload = |v: &[f32], usage| {
            let b = be.alloc(v.len() * 4, usage).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let xb = upload(x, BufferUsage::Activations);
        let rbs: Vec<_> = self
            .routers
            .iter()
            .map(|(_, r)| upload(r, BufferUsage::Weights))
            .collect();
        let yb = be.alloc(x.len() * 4, BufferUsage::Activations).unwrap();
        let mut bind = Bindings::new();
        bind.bind(self.x, xb.as_ref());
        for ((ri, _), rb) in self.routers.iter().zip(&rbs) {
            bind.bind(*ri, rb.as_ref());
        }
        for ((id, _, _), b) in self.banks.iter().zip(banks) {
            bind.bind(*id, *b);
        }
        bind.bind(self.y, yb.as_ref());
        let plan = be.compile(&self.g).unwrap();
        be.execute(plan.as_ref(), &bind).unwrap();
        let mut bytes = vec![0u8; x.len() * 4];
        be.download(yb.as_ref(), &mut bytes).unwrap();
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// The banks held whole, as uploaded buffers.
    fn whole(&self, be: &CpuBackend) -> Vec<Box<dyn Buffer>> {
        self.banks
            .iter()
            .map(|(_, _, b)| {
                let buf = be.alloc(b.len(), BufferUsage::Weights).unwrap();
                be.upload(buf.as_ref(), b).unwrap();
                buf
            })
            .collect()
    }

    /// The banks laid end to end in a [`MemIo`] file and registered per expert in a store planned
    /// for `budget`.
    fn paged(
        &self,
        be: &CpuBackend,
        n_expert: usize,
        n_used: usize,
        budget: usize,
        delay: Duration,
    ) -> (Arc<MemIo>, Arc<paged::PagedWeights>, Vec<Box<dyn Buffer>>) {
        let file: Vec<u8> = self.banks.iter().flat_map(|(_, _, b)| b.clone()).collect();
        let io = Arc::new(MemIo::new(file, delay));
        let plans = paged::plan_pools(budget, &self.pageable(n_expert, n_used));
        assert!(
            plans.iter().all(|p| p.experts) && !plans.is_empty(),
            "no expert pool at {budget} bytes: {plans:?}"
        );
        let store = Arc::new(paged::PagedWeights::new(&plans, io.clone(), None).unwrap());
        let mut at = 0u64;
        let bufs = self
            .banks
            .iter()
            .map(|(_, _, b)| {
                let bank = store
                    .register_experts(&[(at, b.len())], n_expert)
                    .unwrap()
                    .expect("an expert pool for every bank");
                at += b.len() as u64;
                be.paged_experts_weight(store.clone(), bank, b.len())
            })
            .collect();
        (io, store, bufs)
    }
}

fn refs(bufs: &[Box<dyn Buffer>]) -> Vec<&dyn Buffer> {
    bufs.iter().map(|b| b.as_ref()).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// **Paged per expert, with the pool evicting, the op computes bit for bit what it computes over
/// the banks held whole.** Two chained layers, 48 rows routing top-4 of 32 experts (nearly every
/// expert of every bank each pass), a pool at exactly the expert floor and at 1.4 times it, four
/// passes on different inputs and one repeated: each pass evicts experts the previous one read and
/// reads them back. Every dtype the CPU's expert kernels take a distinct route for — f32 (raw
/// activations), Q8_0 (32-blocks), Q4_K (super-blocks, and with a fused gate_up bank the VNNI
/// repack cache where the CPU has it), and Q2_K, the dtype of DeepSeek V4-Flash's banks — in the
/// split gate/up/down layout and the fused gate_up one. The fused layout also puts gate_up and down
/// in different slice sizes, so two expert pools share one op.
#[test]
fn paged_experts_compute_what_whole_banks_compute_while_evicting() {
    let (n_expert, n_used) = (32usize, 4usize);
    let cases = [
        (DType::F32, false),
        (DType::Q8_0, false),
        (DType::Q4K, false),
        (DType::Q2K, false),
        (DType::Q4K, true),
        (DType::Q2K, true),
    ];
    for (dt, fused) in cases {
        let moe = Moe {
            dt,
            fused,
            layers: 2,
            ne: 512,
            nff: 256,
            n_expert,
            n_used,
            rows: 48,
        };
        let built = moe.build();
        assert!(
            built
                .banks
                .iter()
                .all(|(_, _, b)| b.len() >= paged::MIN_PAGED_BYTES),
            "{dt:?}: every bank must be pageable"
        );
        let be = CpuBackend::new();
        let whole = built.whole(&be);
        for mult in [1.0f64, 1.4] {
            let budget = (built.floor_bytes() as f64 * mult) as usize;
            let (io, store, paged) = built.paged(&be, n_expert, n_used, budget, Duration::ZERO);
            for seed in [1u64, 2, 3, 1] {
                let x = rng(1000 + seed, moe.rows * moe.ne);
                let want = built.run(&be, &refs(&whole), &x);
                assert!(want.iter().all(|v| v.is_finite()), "{dt:?}: non-finite");
                let got = built.run(&be, &refs(&paged), &x);
                assert!(
                    bits(&got) == bits(&want),
                    "{dt:?} fused={fused} at {mult}x the floor, pass {seed}: paged != whole"
                );
            }
            let evictions: u64 = store
                .pool_stats()
                .iter()
                .map(|p| p.stats.pager.evictions)
                .sum();
            assert!(
                evictions > 0,
                "{dt:?} fused={fused} at {mult}x: nothing was evicted ({} reads)",
                io.reads()
            );
        }
    }
}

/// `Op::MoeFfn` over expert banks paged per expert reads ONLY the routed experts, all of them in
/// one concurrent wave, and computes exactly what it computes over the same banks held whole.
///
/// The router sends every row to experts 3 and 11 (their rows of the router are the only non-zero
/// ones, and the input is positive), so the op must read six blocks — two experts of each of three
/// banks — out of the forty-eight the banks hold. The wave runs on threads of its own, so it is
/// concurrent whatever the compute pool's size (`RAYON_NUM_THREADS=1` included).
#[test]
fn moe_reads_only_the_routed_experts_of_a_bank_paged_per_expert() {
    let (n_expert, n_used) = (16usize, 2usize);
    let moe = Moe {
        dt: DType::F32,
        fused: false,
        layers: 1,
        ne: 256,
        nff: 128,
        n_expert,
        n_used,
        rows: 3,
    };
    let mut built = moe.build();
    let routed = [3usize, 11];
    let router = &mut built.routers[0].1;
    router.fill(0.0);
    router[routed[0] * moe.ne..(routed[0] + 1) * moe.ne].fill(0.02);
    router[routed[1] * moe.ne..(routed[1] + 1) * moe.ne].fill(0.01);
    let x: Vec<f32> = (0..moe.rows * moe.ne)
        .map(|i| 0.5 + 0.25 * (i as f32).cos())
        .collect();
    let be = CpuBackend::new();
    let want = built.run(&be, &refs(&built.whole(&be)), &x);
    let (io, _store, paged) = built.paged(&be, n_expert, n_used, 1 << 30, Duration::from_millis(5));
    assert_eq!(built.run(&be, &refs(&paged), &x), want, "paged != whole");
    assert_eq!(
        io.reads(),
        routed.len() * 3,
        "the op read experts its routing did not pick"
    );
    assert!(
        io.max_in_flight.load(SeqCst) > 1,
        "the routed experts were read one after another"
    );
    // A second pass finds them resident.
    assert_eq!(built.run(&be, &refs(&paged), &x), want);
    assert_eq!(
        io.reads(),
        routed.len() * 3,
        "a resident expert was re-read"
    );
}

/// **A prefill sweep keeps its resident set.** Six layers whose 128-row chunks route top-4 of 32
/// experts — every expert of every bank, layer after layer, like a long prompt's chunks — with an
/// expert pool of 40% of all expert blocks. Each chunk after the first must find the experts the
/// pool kept resident.
///
/// Measured with these counters (230 slots of 576 expert blocks, 576 pins a chunk): with
/// every expert inserted most-recently-used, as before `paged::expert_insert` existed, the four
/// chunks hit `[0, 0, 0, 0]` and read 2277 blocks — least-recently-used evicts, layer after layer,
/// exactly the experts the next chunk asks for first. With the sweep inserted at the cold end they
/// hit `[0, 137, 134, 134]` and read 1872. The kept set is smaller than the pool because each op
/// holds a floor's worth of pins, which the cold end has to walk past. Decode-shaped ops still
/// insert by recency (the next test).
#[test]
fn a_prefill_sweep_keeps_its_resident_set() {
    let (n_expert, n_used, layers) = (32usize, 4usize, 6usize);
    let moe = Moe {
        dt: DType::F32,
        fused: false,
        layers,
        ne: 128,
        nff: 128,
        n_expert,
        n_used,
        rows: 128,
    };
    let built = moe.build();
    let all: usize = built.banks.iter().map(|(_, _, b)| b.len()).sum();
    let budget = all * 2 / 5;
    assert!(budget >= built.floor_bytes(), "the pool must hold one op");
    let be = CpuBackend::new();
    let (io, store, paged) = built.paged(&be, n_expert, n_used, budget, Duration::ZERO);
    let pins = layers * 3 * n_expert;
    let mut hits_per_chunk = Vec::new();
    let mut last_hits = 0;
    for chunk in 0..4u64 {
        built.run(&be, &refs(&paged), &rng(7 + chunk, moe.rows * moe.ne));
        let hits: u64 = store.pool_stats().iter().map(|p| p.stats.pager.hits).sum();
        hits_per_chunk.push(hits - last_hits);
        last_hits = hits;
    }
    let slots = store.pool_stats()[0].n_slots;
    for (c, &h) in hits_per_chunk.iter().enumerate().skip(1) {
        assert!(
            h as usize * 5 >= pins,
            "chunk {c} hit {h} of {pins} pins: the sweep flushed the resident set (pool of \
             {slots} slots, hits per chunk {hits_per_chunk:?}, {} reads)",
            io.reads()
        );
    }
}

/// The prompt [`layer_major`] and [`chunk_major`] run: `CHUNKS` chunks of `rows` rows each.
const CHUNKS: u64 = 8;

/// A layer-major prefill of `built`'s stack over `paged`: every chunk through layer 0, then every
/// chunk through layer 1, and so on, one single-layer graph per (layer, chunk) whose
/// `weight_reuse` is `hint(last chunk of the layer)`. Returns each chunk's output.
fn layer_major(
    built: &Built,
    layer: &mut Built,
    be: &CpuBackend,
    paged: &[Box<dyn Buffer>],
    hint: fn(bool) -> WeightReuse,
) -> Vec<Vec<f32>> {
    let per_layer = built.banks.len() / built.routers.len();
    let n = built.g.desc(built.x).numel();
    let mut acts: Vec<Vec<f32>> = (0..CHUNKS).map(|c| rng(50 + c, n)).collect();
    for (l, (_, router)) in built.routers.iter().enumerate() {
        layer.routers[0].1 = router.clone();
        let banks = refs(&paged[l * per_layer..(l + 1) * per_layer]);
        for (c, x) in acts.iter_mut().enumerate() {
            layer.g.weight_reuse = hint(c as u64 + 1 == CHUNKS);
            *x = layer.run(be, &banks, x);
        }
    }
    acts
}

/// The same prompt chunk-major: the whole stack per chunk, as an ordinary forward pass.
fn chunk_major(built: &Built, be: &CpuBackend, paged: &[Box<dyn Buffer>]) -> Vec<Vec<f32>> {
    let rows_ne = built.g.desc(built.x).numel();
    (0..CHUNKS)
        .map(|c| built.run(be, &refs(paged), &rng(50 + c, rows_ne)))
        .collect()
}

/// The layer-major hints a prefill gives: [`WeightReuse::Again`] while more chunks of the layer
/// follow, [`WeightReuse::Done`] on its last.
fn prefill_hints(last: bool) -> WeightReuse {
    if last {
        WeightReuse::Done
    } else {
        WeightReuse::Again
    }
}

/// **A layer-major prefill reads each expert it routes to once, with a pool of one layer's
/// experts.** Six layers of top-4-of-32 routing, eight chunks of eight rows: one chunk routes to
/// part of a layer's experts, the eight together to nearly all of them. With the pool at the
/// expert floor (one layer's experts, a sixth of all blocks), layer-major with the prefill's hints
/// (`Again`, `Done` on a layer's last chunk) reads exactly as many blocks as with a pool that
/// holds every block, where nothing is read twice — and computes the chunk-major outputs bit for
/// bit. The same order with the ordinary forward's insert policy (`Sweep`: a sweep's chunk goes to
/// the cold end) reads experts again within a layer, since a chunk's misses evict the experts the
/// chunk before brought in; chunk-major reads most of the model once per chunk. Measured
/// 2026-11-17, of 576 blocks: a pool of every block reads 546 (the blocks routed to); one layer's
/// pool reads 546 layer-major, 889 layer-major with sweep hints and 2177 chunk-major.
#[test]
fn layer_major_chunks_read_each_routed_expert_once() {
    let moe = Moe {
        dt: DType::F32,
        fused: false,
        layers: 6,
        ne: 128,
        nff: 128,
        n_expert: 32,
        n_used: 4,
        rows: 8,
    };
    let built = moe.build();
    let all: usize = built.banks.iter().map(|(_, _, b)| b.len()).sum();
    let floor = built.floor_bytes();
    let run = |budget: usize, order: &str| {
        let be = CpuBackend::new();
        let (io, _store, paged) =
            built.paged(&be, moe.n_expert, moe.n_used, budget, Duration::ZERO);
        let mut layer = Moe { layers: 1, ..moe }.build();
        let out = match order {
            "layer-major" => layer_major(&built, &mut layer, &be, &paged, prefill_hints),
            "layer-major, sweep hints" => {
                layer_major(&built, &mut layer, &be, &paged, |_| WeightReuse::Sweep)
            }
            _ => chunk_major(&built, &be, &paged),
        };
        (io.reads(), out)
    };
    let (once, want) = run(all, "layer-major");
    let (lm, lm_out) = run(floor, "layer-major");
    let (swept, _) = run(floor, "layer-major, sweep hints");
    let (cm, cm_out) = run(floor, "chunk-major");
    let blocks = 6 * 3 * moe.n_expert;
    let counts = format!(
        "{blocks} expert blocks; reads with a pool of all of them {once}, with one layer's: \
         layer-major {lm}, layer-major with sweep hints {swept}, chunk-major {cm}"
    );
    assert!(once <= blocks, "{counts}");
    assert_eq!(
        lm, once,
        "a layer-major prefill read an expert twice: {counts}"
    );
    assert!(swept > lm, "the hints changed nothing: {counts}");
    assert!(
        cm > 2 * lm,
        "chunk-major did not re-read the model: {counts}"
    );
    for (c, (a, b)) in lm_out.iter().zip(&cm_out).enumerate() {
        assert_eq!(
            bits(a),
            bits(b),
            "chunk {c}: layer-major computed other bits"
        );
    }
    for (c, (a, b)) in lm_out.iter().zip(&want).enumerate() {
        assert_eq!(
            bits(a),
            bits(b),
            "chunk {c}: the pool size changed the bits"
        );
    }
}

/// **A layer-major prefill lets each finished layer go first.** The last chunk of a layer runs
/// with [`WeightReuse::Done`], which moves the layer's experts to the cold end, so the next
/// layer's misses evict a layer this prefill no longer needs before anything older. With a pool
/// of two layers' experts, a second prefill of the same prompt then still finds the experts of
/// the first layer (kept, since every later layer's misses took the layer just finished) and of
/// the last one: it reads fewer blocks than when every chunk is hinted `Again`, which leaves plain
/// recency to evict the oldest layer each time and so leaves only the last two layers resident,
/// both evicted again by the time a second prefill could use them. Measured 2026-11-17: 546 reads
/// each first prefill, then 354 with demotion against 546 without.
#[test]
fn a_finished_layer_is_evicted_before_older_residents() {
    let moe = Moe {
        dt: DType::F32,
        fused: false,
        layers: 6,
        ne: 128,
        nff: 128,
        n_expert: 32,
        n_used: 4,
        rows: 8,
    };
    let built = moe.build();
    let run = |hint: fn(bool) -> WeightReuse| {
        let be = CpuBackend::new();
        let budget = 2 * built.floor_bytes();
        let (io, _store, paged) =
            built.paged(&be, moe.n_expert, moe.n_used, budget, Duration::ZERO);
        let mut layer = Moe { layers: 1, ..moe }.build();
        layer_major(&built, &mut layer, &be, &paged, hint);
        let first = io.reads();
        layer_major(&built, &mut layer, &be, &paged, hint);
        (first, io.reads() - first)
    };
    let (first, demoted) = run(prefill_hints);
    let (first_lru, kept_by_recency) = run(|_| WeightReuse::Again);
    let counts = format!(
        "first prefill {first} reads ({first_lru} without demotion); second prefill {demoted} \
         reads with the finished layers demoted, {kept_by_recency} without"
    );
    assert_eq!(first, first_lru, "{counts}");
    assert!(demoted < kept_by_recency, "{counts}");
}

/// **The decode after a layer-major prefill starts warm.** A layer's last chunk keeps the experts
/// its last row routed to out of the demotion (`paged::done_keep`), so that row run again alone
/// through the whole stack — it routes exactly as it did in its chunk, every op computing a row
/// on its own — finds every expert resident: no read. With a pool of three layers' experts.
#[test]
fn the_last_rows_experts_survive_a_layer_major_prefill() {
    let moe = Moe {
        dt: DType::F32,
        fused: false,
        layers: 6,
        ne: 128,
        nff: 128,
        n_expert: 32,
        n_used: 4,
        rows: 8,
    };
    let built = moe.build();
    let be = CpuBackend::new();
    let budget = 3 * built.floor_bytes();
    let (io, _store, paged) = built.paged(&be, moe.n_expert, moe.n_used, budget, Duration::ZERO);
    let mut layer = Moe { layers: 1, ..moe }.build();
    let out = layer_major(&built, &mut layer, &be, &paged, prefill_hints);
    let reads = io.reads();
    let one = Moe { rows: 1, ..moe }.build();
    let n = built.g.desc(built.x).numel();
    let last_row = &rng(50 + CHUNKS - 1, n)[n - moe.ne..];
    let y = one.run(&be, &refs(&paged), last_row);
    assert_eq!(
        bits(&y),
        bits(&out[CHUNKS as usize - 1][n - moe.ne..]),
        "the row alone computed other bits"
    );
    assert_eq!(
        io.reads(),
        reads,
        "the last row's experts were evicted by the prefill"
    );
}

/// **`host_streams` answers for the weights it is handed, not for the backend.** Two models on
/// one backend: the one whose pool holds every block does not stream, the one whose pool holds a
/// third does, whichever was bound last; weights with no paged buffer do not stream.
#[test]
fn host_streams_answers_for_the_model_asked_about() {
    let moe = Moe {
        dt: DType::F32,
        fused: false,
        layers: 3,
        ne: 128,
        nff: 128,
        n_expert: 32,
        n_used: 4,
        rows: 8,
    };
    let built = moe.build();
    let all: usize = built.banks.iter().map(|(_, _, b)| b.len()).sum();
    let be = CpuBackend::new();
    let (_, _, small) = built.paged(&be, 32, 4, built.floor_bytes(), Duration::ZERO);
    let (_, _, whole) = built.paged(&be, 32, 4, all, Duration::ZERO);
    let streams = |bufs: &[Box<dyn Buffer>]| be.host_streams(&mut bufs.iter().map(|b| b.as_ref()));
    assert!(
        streams(&small),
        "a pool of one layer's experts does not stream"
    );
    assert!(!streams(&whole), "a pool holding every block streams");
    assert!(
        streams(&small),
        "the model bound last answered for the other"
    );
    assert!(!streams(&built.whole(&be)), "banks held whole stream");
}

/// The insert rule itself: decode-shaped ops recency, sweeps the cold end.
#[test]
fn a_decode_step_inserts_by_recency_and_a_sweep_at_the_cold_end() {
    use infr_core::pager::Insert;
    assert_eq!(
        paged::expert_insert(6, 256, WeightReuse::Sweep),
        Insert::Mru
    );
    assert_eq!(
        paged::expert_insert(127, 256, WeightReuse::Sweep),
        Insert::Mru
    );
    assert_eq!(
        paged::expert_insert(128, 256, WeightReuse::Sweep),
        Insert::Cold
    );
    assert_eq!(
        paged::expert_insert(256, 256, WeightReuse::Sweep),
        Insert::Cold
    );
}

/// A per-expert bank is readable only through `Op::MoeFfn`: any other op naming one is refused
/// with an error rather than reading the whole bank, which paging it by expert exists to avoid.
#[test]
fn only_moe_ffn_may_read_a_bank_paged_per_expert() {
    let (n_expert, eb) = (4usize, paged::MIN_PAGED_BYTES / 4);
    let io = Arc::new(MemIo::new(vec![0u8; n_expert * eb], Duration::ZERO));
    let bank = paged::Pageable {
        nbytes: n_expert * eb,
        reads: paged::Reads::Experts {
            n_expert,
            n_used: 1,
            layer: 0,
        },
    };
    let store =
        Arc::new(paged::PagedWeights::new(&paged::plan_pools(1 << 30, &[bank]), io, None).unwrap());
    let b = store
        .register_experts(&[(0, n_expert * eb)], n_expert)
        .unwrap()
        .expect("an expert pool");
    let be = CpuBackend::new();
    let wb = be.paged_experts_weight(store.clone(), b, n_expert * eb);
    let n = n_expert * eb / 4;
    let mut g = Graph::new();
    let xi = g.input(TensorDesc::new(vec![1, n], DType::F32));
    let wi = g.weight(TensorDesc::new(vec![n], DType::F32));
    let yi = g.output(TensorDesc::new(vec![1, n], DType::F32));
    g.push(Op::RmsNorm {
        x: xi,
        weight: wi,
        dst: yi,
        rows: 1,
        dim: n as u32,
        eps: 1e-6,
    });
    let xb = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    let yb = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    let mut bind = Bindings::new();
    bind.bind(xi, xb.as_ref());
    bind.bind(wi, wb.as_ref());
    bind.bind(yi, yb.as_ref());
    let plan = be.compile(&g).unwrap();
    let err = be.execute(plan.as_ref(), &bind).expect_err("must refuse");
    assert!(err.to_string().contains("Op::MoeFfn"), "{err}");
}

/// **A panic inside a kernel the pool runs fails the forward promptly, and the backend computes
/// the next one correctly.** A bank held whole read at twice its expert stride slices past the
/// bank's end inside the gate GEMM's tasks — the fault that once left a test process idle until it
/// was killed. The forward must fail with that slice's own message, and the same backend, banks and
/// pool must then compute what they computed before the fault.
#[test]
fn a_kernel_panic_fails_the_forward_and_the_next_one_computes() {
    super::test_util::within(|| {
        let moe = Moe {
            dt: DType::Q8_0,
            fused: false,
            layers: 1,
            ne: 512,
            nff: 256,
            n_expert: 32,
            n_used: 4,
            rows: 48,
        };
        let built = moe.build();
        let be = CpuBackend::new();
        let whole = built.whole(&be);
        let x = rng(7, moe.rows * moe.ne);
        let want = built.run(&be, &refs(&whole), &x);

        be.double_whole_bank_stride.store(true, SeqCst);
        let msg = super::test_util::panic_of(|| {
            built.run(&be, &refs(&whole), &x);
        })
        .expect("the out-of-bounds expert slice did not fail the forward");
        assert!(
            msg.contains("out of range for slice"),
            "not the slice's own panic: {msg}"
        );
        be.double_whole_bank_stride.store(false, SeqCst);

        let got = built.run(&be, &refs(&whole), &x);
        assert!(
            bits(&got) == bits(&want),
            "the forward after the panic computed differently"
        );
    })
}

/// One owned buffer bound to two banks of one `Op::MoeFfn` is refused by name. The op holds every
/// bank's lock at once, so the second lock on the same mutex would hang the forward with no
/// message — the shape of the one hang the 2026-11-16 investigation reproduced.
#[test]
fn moe_ffn_refuses_one_owned_buffer_bound_to_two_banks() {
    super::test_util::within(|| {
        let moe = Moe {
            dt: DType::Q8_0,
            fused: false,
            layers: 1,
            ne: 512,
            nff: 256,
            n_expert: 8,
            n_used: 2,
            rows: 4,
        };
        let built = moe.build();
        let be = CpuBackend::new();
        let whole = built.whole(&be);
        let x = rng(9, moe.rows * moe.ne);
        // Binding order is gate, up, down: the gate bank stands in for the up bank too.
        let banks = [whole[0].as_ref(), whole[0].as_ref(), whole[2].as_ref()];
        let msg = super::test_util::panic_of(|| {
            built.run(&be, &banks, &x);
        })
        .expect("the doubly bound bank was not refused");
        assert!(
            msg.contains("Op::MoeFfn: gate_exps and up_exps are bound to the same owned buffer"),
            "{msg}"
        );
    })
}
