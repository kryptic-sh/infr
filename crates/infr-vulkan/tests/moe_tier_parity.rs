//! Proves the MoE paging session's tier below VRAM is consulted exactly when the VRAM arena
//! misses, that every expert staged through it lands in VRAM byte-for-byte, and that a miss whose
//! bytes cannot be produced leaves nothing behind.
//!
//! `MoePagerSession::stage_role` resolves each routed expert against the VRAM arena first. A hit
//! needs nothing from the tier below; only a miss reads the expert's bytes from it. Consulting the
//! host tier on a hit as well is invisible in the output — the bytes are discarded — but it costs
//! a disk read whenever the host arena does not hold that expert, may evict a host block to make
//! room for bytes nobody uses, and makes the host tier's counters and recency mirror VRAM traffic.
//! So the observable is the host tier's own counters, compared against the VRAM arena's misses,
//! on a sequence where VRAM hits experts the (smaller) host arena has already evicted.
//!
//! Three legs stage the identical sequence and differ only in where a miss reads from: the bank
//! in memory (`ExpertBytes::Mmap`, the reference), a caching host tier (`HostPager::new`, the
//! discrete-GPU shape) and an arena-less one (`HostPager::stream_only`, the unified-memory shape).
//! All three must reach the same VRAM residency decisions and stage the same bytes. Content is
//! checked the way the paged kernels read it: through the layer's frozen LUT window in the tape,
//! then the arena slot it names.
//!
//! Run: `cargo test -p infr-vulkan --test moe_tier_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};
use infr_core::error::{Error, Result};
use infr_core::hostpager::{HostPager, HostPagerStats};
use infr_core::pager::{PagerStats, NOT_RESIDENT};
use infr_vulkan::pager::{
    buffer_identity, ExpertBytes, ExpertSource, MoePagerLayout, MoePagerSession, MoePoolSpec, Role,
};
use infr_vulkan::VulkanBackend;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

const N_EXPERT: usize = 8;
/// One expert's bytes — a whole number of u32 words, as the arena requires.
const STRIDE: usize = 256;
/// VRAM slots: fewer than the experts, so the sequence both hits and misses.
const VRAM_SLOTS: usize = 4;
/// Host slots: fewer than VRAM's, so VRAM goes on hitting experts the host arena has evicted —
/// the case where consulting the host tier on a hit costs a read.
const DRAM_SLOTS: usize = 2;
/// Ring bytes: two slots per half, so a batch with more misses than that must rotate.
const RING_BYTES: usize = 4 * STRIDE;

/// Routed batches, one `(layer, role)` residency resolution each: `(expert ids, scan)`. `scan`
/// is the prefill full-set sweep's cold-end insert; otherwise a decode touch, MRU. No batch names
/// more distinct experts than VRAM holds, as the session requires.
///
/// The tier below sees only VRAM's misses, so it holds the most recently MISSED experts, and a
/// VRAM miss finds one there only when VRAM evicted it soon after admitting it. The middle of the
/// sequence builds that case on purpose: expert 4 misses, VRAM hits on 1–3 make 4 its coldest,
/// expert 5's miss evicts it — and the host arena, which saw neither those hits nor anything but
/// the misses on 4 and 5, still holds it for the routed touch after.
const BATCHES: &[(&[u32], bool)] = &[
    (&[0, 1, 2, 3], false),
    // All VRAM hits — and experts 0 and 1 are no longer in the two-slot host arena.
    (&[0, 1, 2, 3], false),
    (&[4], false),
    (&[1, 2, 3], false),
    (&[5], false),
    (&[4], false),
    (&[6, 7], true),
    (&[5, 5, 0], false),
    (&[6, 1], true),
];

/// No block is set to fail.
const NO_FAULT: u32 = u32::MAX;

/// The "model file": the bank in memory, read block by block, failing the read of one chosen
/// block while `fail_on` names it. `HostPager`'s own `reads` counter is the observable, so this
/// needs no counting of its own.
struct BankIo {
    bank: Vec<u8>,
    fail_on: AtomicU32,
}

impl BlockIo for BankIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        if self.fail_on.load(Ordering::SeqCst) == desc.id {
            return Err(Error::backend(format!(
                "injected read failure on {}",
                desc.id
            )));
        }
        let mut off = 0;
        for e in &desc.extents {
            let at = e.offset as usize;
            dst[off..off + e.len].copy_from_slice(&self.bank[at..at + e.len]);
            off += e.len;
        }
        Ok(())
    }
}

/// Distinct bytes per expert (across every layer), so a slot holding the wrong expert is visible.
fn synth_bank(layers: usize) -> Vec<u8> {
    (0..layers * N_EXPERT * STRIDE)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ (i / STRIDE).wrapping_mul(40503)) >> 7;
            h as u8
        })
        .collect()
}

/// A host tier over `bank`, one block per expert of every layer under its global id. Returns the
/// tier and its I/O, whose `fail_on` the caller may set.
fn host_tier(bank: &[u8], caches: bool) -> (Arc<HostPager>, Arc<BankIo>) {
    let io = Arc::new(BankIo {
        bank: bank.to_vec(),
        fail_on: AtomicU32::new(NO_FAULT),
    });
    let host = if caches {
        HostPager::new(DRAM_SLOTS, STRIDE, io.clone()).unwrap()
    } else {
        HostPager::stream_only(STRIDE, io.clone()).unwrap()
    };
    for b in 0..bank.len() / STRIDE {
        host.register(BlockDesc {
            id: b as u32,
            extents: vec![BlockExtent {
                offset: (b * STRIDE) as u64,
                len: STRIDE,
            }],
        })
        .unwrap();
    }
    (Arc::new(host), io)
}

/// One session over `layers` paged layers of one role, plus the caller-side ring and tape state
/// the adapter keeps.
struct Fixture<'a> {
    be: &'a VulkanBackend,
    sess: MoePagerSession,
    /// Real placeholder buffers, so each `buf_id` is the same kind of identity production uses.
    _placeholders: Vec<Box<dyn Buffer>>,
    buf_ids: Vec<usize>,
    bank: Vec<u8>,
    half: usize,
    cursor: usize,
    rotations: usize,
    tape_cursor: usize,
}

impl<'a> Fixture<'a> {
    fn new(
        be: &'a VulkanBackend,
        bank: &[u8],
        host: Option<Arc<HostPager>>,
        bytes: impl Fn(usize) -> ExpertBytes,
    ) -> Self {
        let layers = bank.len() / (N_EXPERT * STRIDE);
        let mut sess = MoePagerSession::new(
            be,
            MoePagerLayout {
                n_blocks: layers * N_EXPERT,
                pools: vec![MoePoolSpec {
                    role: Role::Gate,
                    slot_bytes: STRIDE,
                    n_slots: VRAM_SLOTS,
                    host,
                }],
                ring_bytes: RING_BYTES,
            },
        )
        .unwrap();
        let placeholders: Vec<_> = (0..layers)
            .map(|_| be.alloc_uninit(4, BufferUsage::Weights).unwrap())
            .collect();
        let buf_ids: Vec<usize> = placeholders
            .iter()
            .map(|p| buffer_identity(p.as_ref()))
            .collect();
        for (l, &buf_id) in buf_ids.iter().enumerate() {
            sess.register(
                Role::Gate,
                buf_id,
                ExpertSource {
                    bytes: bytes(l),
                    stride_bytes: STRIDE,
                    layer_base: (l * N_EXPERT) as u32,
                },
                N_EXPERT,
            )
            .unwrap();
        }
        Self {
            be,
            sess,
            _placeholders: placeholders,
            buf_ids,
            bank: bank.to_vec(),
            half: 0,
            cursor: 0,
            rotations: 0,
            tape_cursor: 0,
        }
    }

    /// Stage `ids` of `layer` as one batch, rotating the ring whenever its half fills, as the
    /// adapter does. Returns the ring bytes consumed — one slot per miss, nothing for a hit.
    ///
    /// Each recording is submitted before its half is reused: `stage_role`'s ring-lifetime
    /// contract. The adapter pipelines the two halves against a fence instead; these tests are
    /// about which bytes land, not about overlapping them with compute.
    fn stage(&mut self, layer: usize, ids: &[u32], scan: bool) -> Result<usize> {
        let buf_id = self.buf_ids[layer];
        self.sess.begin_batch(buf_id).unwrap();
        let (mut start, mut consumed) = (0, 0);
        while start < ids.len() {
            let rec = self.be.recorder().unwrap();
            let half_base = self.half * self.sess.ring_half_bytes();
            let before = self.cursor;
            let staged = self.sess.stage_role(
                &rec,
                half_base,
                &mut self.cursor,
                buf_id,
                &ids[start..],
                scan,
            );
            consumed += self.cursor - before;
            rec.finish().unwrap();
            start += staged?;
            if start < ids.len() {
                self.half ^= 1;
                self.cursor = 0;
                self.rotations += 1;
            }
        }
        Ok(consumed)
    }

    /// `layer`'s LUT window as the paged kernels would read it: frozen into the tape now.
    fn lut(&mut self, layer: usize) -> Vec<u32> {
        let buf_id = self.buf_ids[layer];
        let window = self
            .sess
            .lut_window(&mut self.tape_cursor, buf_id, N_EXPERT)
            .unwrap() as usize;
        let mut tape = vec![0u32; self.tape_cursor];
        self.be
            .download(self.sess.tape(), bytemuck::cast_slice_mut(&mut tape))
            .unwrap();
        tape[window..window + N_EXPERT].to_vec()
    }

    /// Assert every expert in `ids` of `layer` is served, through the LUT window, from a slot
    /// holding exactly its own bytes.
    fn check(&mut self, layer: usize, ids: &[u32], what: &str) {
        let lut = self.lut(layer);
        let mut arena = vec![0u8; VRAM_SLOTS * STRIDE];
        self.be
            .download(self.sess.arena(self.buf_ids[layer]).unwrap(), &mut arena)
            .unwrap();
        for &e in ids {
            let slot = lut[e as usize];
            assert_ne!(slot, NOT_RESIDENT, "{what}: expert {e} is not resident");
            let (slot, g) = (slot as usize, layer * N_EXPERT + e as usize);
            assert!(
                arena[slot * STRIDE..(slot + 1) * STRIDE]
                    == self.bank[g * STRIDE..(g + 1) * STRIDE],
                "{what}: slot {slot} does not hold expert {e}'s bytes"
            );
        }
    }

    fn stats(&self) -> (PagerStats, Option<HostPagerStats>) {
        let (_, vram, host) = self.sess.pool_stats().remove(0);
        (vram, host)
    }
}

/// The in-memory bank as one mmap view per layer.
fn mmap_bytes(bank: &[u8]) -> impl Fn(usize) -> ExpertBytes + '_ {
    move |l| {
        let layer = bank[l * N_EXPERT * STRIDE..(l + 1) * N_EXPERT * STRIDE].to_vec();
        ExpertBytes::Mmap(Arc::new(layer))
    }
}

/// What one leg of the routed sequence observed.
struct Leg {
    vram: PagerStats,
    host: Option<HostPagerStats>,
    rotations: usize,
}

/// Stage every batch of [`BATCHES`] through a fresh session and check, after each, that every
/// expert it named is served from a slot holding exactly that expert's bytes.
///
/// `before_batch` runs before each batch with its index and the previous batch's LUT window — the
/// hook the caching leg uses to show the case under test actually arises.
fn run_leg(
    be: &VulkanBackend,
    bank: &[u8],
    host: Option<Arc<HostPager>>,
    bytes: impl Fn(usize) -> ExpertBytes,
    mut before_batch: impl FnMut(usize, &[u32]),
) -> Leg {
    let mut fx = Fixture::new(be, bank, host, bytes);
    let mut lut = vec![NOT_RESIDENT; N_EXPERT];
    for (b, &(ids, scan)) in BATCHES.iter().enumerate() {
        before_batch(b, &lut);
        fx.stage(0, ids, scan).unwrap();
        fx.check(0, ids, &format!("batch {b}"));
        lut = fx.lut(0);
    }
    let (vram, host) = fx.stats();
    Leg {
        vram,
        host,
        rotations: fx.rotations,
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn the_host_tier_is_consulted_only_on_a_vram_miss() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let bank = synth_bank(1);
    for e in 1..N_EXPERT {
        assert_ne!(
            bank[..STRIDE],
            bank[e * STRIDE..(e + 1) * STRIDE],
            "experts 0 and {e} are identical — a mis-served slot would be invisible"
        );
    }

    // ── Leg 1: the bank in memory — the reference schedule ─────────────────────────────────────
    let mmap = run_leg(&be, &bank, None, mmap_bytes(&bank), |_, _| {});

    // ── Leg 2: a caching host tier smaller than VRAM ───────────────────────────────────────────
    let (host, _) = host_tier(&bank, true);
    let mut vram_only_hits = 0usize;
    let cached = run_leg(
        &be,
        &bank,
        Some(host.clone()),
        |_| ExpertBytes::Host,
        |b, lut| {
            // Count the experts this batch will hit in VRAM that the host arena does NOT hold —
            // the case a hit used to pay a read for. Only a batch resident in full at its start
            // is counted: nothing in it can miss, so nothing can evict a later id mid-batch and
            // turn a counted hit into a miss. `try_pin` moves no counter, so the probe leaves the
            // stats below untouched.
            let ids = BATCHES[b].0;
            if ids.iter().all(|&e| lut[e as usize] != NOT_RESIDENT) {
                vram_only_hits += ids.iter().filter(|&&e| host.try_pin(e).is_none()).count();
            }
        },
    );

    // ── Leg 3: the arena-less tier a unified-memory device gets ────────────────────────────────
    let stream = run_leg(
        &be,
        &bank,
        Some(host_tier(&bank, false).0),
        |_| ExpertBytes::Host,
        |_, _| {},
    );

    // Same VRAM decisions on every leg: where a miss reads from must not change what is resident.
    for (name, leg) in [("caching", &cached), ("arena-less", &stream)] {
        assert_eq!(
            (leg.vram.hits, leg.vram.misses, leg.vram.evictions),
            (mmap.vram.hits, mmap.vram.misses, mmap.vram.evictions),
            "the {name} host tier changed the VRAM schedule"
        );
        assert_eq!(
            leg.rotations, mmap.rotations,
            "{name}: ring rotations differ"
        );
    }
    let vram = mmap.vram;
    assert!(vram.hits > 0, "no VRAM hit — the case under test never ran");
    assert!(
        vram.misses > 0 && vram.evictions > 0,
        "no VRAM eviction pressure"
    );
    assert!(mmap.rotations > 0, "the ring never rotated");
    assert!(
        vram_only_hits > 0,
        "VRAM never hit an expert the host arena had evicted — the sequence tests nothing"
    );

    // The caching tier: consulted exactly on the VRAM misses, each consultation a pin (a host hit
    // or a read-and-admit), never a stream.
    let h = cached.host.expect("leg 2 has a host tier");
    let consulted = h.pager.hits + h.pager.misses + h.streamed;
    assert_eq!(
        consulted, vram.misses,
        "the host tier was consulted {consulted} times for {} VRAM misses",
        vram.misses
    );
    assert_eq!(h.streamed, 0, "the MoE host tier pins; it never streams");
    assert_eq!(
        h.reads, h.pager.misses,
        "every host miss is exactly one read"
    );
    assert!(
        h.pager.hits > 0,
        "no VRAM miss was absorbed by the host arena"
    );
    assert!(h.pager.evictions > 0, "the host arena never evicted");

    // The arena-less tier: every VRAM miss is one read straight into the ring, nothing more.
    let s = stream.host.expect("leg 3 has a host tier");
    assert_eq!(
        (s.reads, s.streamed),
        (vram.misses, vram.misses),
        "the arena-less tier must read exactly once per VRAM miss"
    );
    assert_eq!(
        (s.pager.hits, s.pager.misses),
        (0, 0),
        "an arena-less tier caches nothing"
    );

    eprintln!(
        "vram: {} hits / {} misses / {} evictions; dram: {} hits / {} misses / {} evictions, \
         {} reads; {vram_only_hits} VRAM hits on experts the host arena did not hold; {} ring \
         rotations",
        vram.hits,
        vram.misses,
        vram.evictions,
        h.pager.hits,
        h.pager.misses,
        h.pager.evictions,
        h.reads,
        mmap.rotations,
    );
}

/// A miss whose bytes cannot be read must leave nothing behind: not the block (its slot holds the
/// previous occupant's bytes, never its own), not a LUT entry pointing at that slot. The staging
/// call fails, and the next staging of the same expert — once the read works — is a clean miss
/// that puts the right bytes in.
///
/// Run on both host-tier shapes, because their miss paths read differently (a pin into the host
/// arena, a positioned read straight into the ring). The recording the failed call was given is
/// SUBMITTED here, not discarded: a caller that does not route the error into
/// `forget_unsubmitted` must still find a consistent cache, which is what makes the rollback the
/// pager's job rather than every caller's.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_failed_read_leaves_nothing_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let bank = synth_bank(1);
    for caches in [true, false] {
        let what = if caches { "caching" } else { "arena-less" };
        let (host, io) = host_tier(&bank, caches);
        let mut fx = Fixture::new(&be, &bank, Some(host), |_| ExpertBytes::Host);

        // Fill VRAM, so the failing miss has to evict: expert 0 is the least recently used.
        fx.stage(0, &[0, 1, 2, 3], false).unwrap();
        let before = fx.lut(0);
        let victim_slot = before[0];

        io.fail_on.store(4, Ordering::SeqCst);
        let err = fx
            .stage(0, &[1, 4], false)
            .expect_err("a failed read must fail the staging call");
        assert!(
            err.to_string().contains("injected read failure on 4"),
            "{what}: unexpected error: {err}"
        );

        let lut = fx.lut(0);
        assert_eq!(lut[4], NOT_RESIDENT, "{what}: the failed expert is mapped");
        assert_eq!(
            lut[0], NOT_RESIDENT,
            "{what}: the evicted victim is still mapped to a slot it lost"
        );
        assert!(
            !lut.contains(&victim_slot),
            "{what}: an entry still maps to slot {victim_slot}, which the failed miss was given \
             and now holds nobody's bytes: {lut:?}"
        );
        // The counters say what happened, not what was kept: the failed touch was a miss, and it
        // displaced expert 0 to make room.
        let (vram, _) = fx.stats();
        assert_eq!(
            (vram.hits, vram.misses, vram.evictions),
            (1, 5, 1),
            "{what}: counters after the failed miss"
        );

        // Fault cleared: expert 4 is a miss again (a full slot of ring), not a hit on a slot that
        // never received its bytes, and lands right. The survivors are untouched.
        io.fail_on.store(NO_FAULT, Ordering::SeqCst);
        assert_eq!(
            fx.stage(0, &[4], false).unwrap(),
            STRIDE,
            "{what}: the expert whose read failed was treated as resident"
        );
        fx.check(0, &[1, 2, 3, 4], what);
    }
}

/// Repeats inside one batch, which the decode split path passes through as the router wrote them
/// (rows above one can name an expert twice). The first occurrence misses and the repeat hits —
/// one host consultation — including when the first occurrence was staged just before the ring
/// half filled (the repeat needs no room, so it does not force a rotation) and when a repeated
/// miss arrives after one (staged in the fresh half, then hit).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_repeated_expert_is_staged_once() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let bank = synth_bank(1);
    let (host, _) = host_tier(&bank, true);
    let mut fx = Fixture::new(&be, &bank, Some(host), |_| ExpertBytes::Host);

    // 0 and 1 miss and fill the half; 0 again hits with the half full; 2 misses into the next
    // half after a rotation; 2 again hits.
    let consumed = fx.stage(0, &[0, 1, 0, 2, 2], false).unwrap();
    assert_eq!(consumed, 3 * STRIDE, "three misses, two hits");
    assert_eq!(fx.rotations, 1, "only the third miss needed a fresh half");
    let (vram, host) = fx.stats();
    let h = host.expect("a host tier");
    assert_eq!((vram.hits, vram.misses), (2, 3));
    assert_eq!(
        h.pager.hits + h.pager.misses + h.streamed,
        3,
        "the host tier must be consulted once per distinct missing expert"
    );
    fx.check(0, &[0, 1, 2], "repeats");
}

/// An expert id past the layer's own experts is refused before anything changes. On the
/// host-backed path `layer_base + id` is a valid block of the NEXT layer, so without the check the
/// call would read and cache that layer's expert under the wrong layer's request.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn an_out_of_range_expert_is_refused_before_any_state_changes() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let bank = synth_bank(2);
    let (host, _) = host_tier(&bank, true);
    let mut legs: Vec<(&str, Fixture)> = vec![
        ("mmap", Fixture::new(&be, &bank, None, mmap_bytes(&bank))),
        (
            "host",
            Fixture::new(&be, &bank, Some(host), |_| ExpertBytes::Host),
        ),
    ];
    for (what, fx) in &mut legs {
        let err = fx
            .stage(0, &[N_EXPERT as u32], false)
            .expect_err("an out-of-range expert id must be refused");
        assert!(
            err.to_string().contains("out of range"),
            "{what}: unexpected error: {err}"
        );
        let (vram, host) = fx.stats();
        assert_eq!(
            (vram.hits, vram.misses),
            (0, 0),
            "{what}: the VRAM pager was touched"
        );
        if let Some(h) = host {
            assert_eq!(h.reads, 0, "{what}: the host tier read a block");
        }
        assert!(
            fx.lut(1).iter().all(|&s| s == NOT_RESIDENT),
            "{what}: the next layer's expert was staged"
        );
    }
}
