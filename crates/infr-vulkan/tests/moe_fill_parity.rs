//! Proves that the MoE paging session's background fills (`MoePagerSession::request_fills`,
//! `stage_ready_fills`) put every expert's own bytes into the arena, from every tier a miss reads
//! from, each read once — and that the reader stops once nothing is left to fill.
//!
//! A pool with a slot for every registered expert fills the experts no token routed to, so that a
//! decode layer becomes wholly resident and skips its router readback. A fill that copied the
//! wrong bytes would decode garbage only for the tokens that later route to it, so the arena is
//! compared byte for byte with the bank here, through the LUT windows the paged kernels read.
//!
//! The loop below plays the adapter's part on a synthetic two-layer model, one forward per round:
//! each layer that is not wholly resident admits what the reader has ready into a recording that
//! is submitted and waited (the adapter's `sync_stream`), then asks for its own absent experts.
//!
//! Run: `cargo test -p infr-vulkan --test moe_fill_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};
use infr_core::error::{Error, Result};
use infr_core::hostpager::HostPager;
use infr_core::pager::NOT_RESIDENT;
use infr_vulkan::pager::{
    buffer_identity, ExpertBytes, ExpertSource, MoePagerLayout, MoePagerSession, MoePoolSpec, Role,
};
use infr_vulkan::VulkanBackend;
use std::sync::Arc;

const N_EXPERT: usize = 8;
const LAYERS: usize = 2;
/// One expert's bytes — a whole number of u32 words, as the arena requires.
const STRIDE: usize = 256;
/// Experts a token routes to before any fill: each layer's first two.
const ROUTED: &[u32] = &[0, 1];
/// Experts one half of the staging ring holds: the most one staging call can stage.
const RING_HALF_EXPERTS: usize = 2;
/// Rounds in a row without a new fill before the test calls the reader stuck. Each round sleeps
/// a little, so this is generous for a reader copying a few kilobytes from memory.
const STALL_ROUNDS: usize = 1000;

/// The "model file": the bank in memory, read block by block.
struct BankIo(Vec<u8>);

impl BlockIo for BankIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        let mut off = 0;
        for e in &desc.extents {
            let at = e.offset as usize;
            dst[off..off + e.len].copy_from_slice(&self.0[at..at + e.len]);
            off += e.len;
        }
        Ok(())
    }
}

/// Distinct bytes per expert (across every layer), so a slot holding the wrong expert is visible.
fn synth_bank() -> Vec<u8> {
    (0..LAYERS * N_EXPERT * STRIDE)
        .map(|i| ((i.wrapping_mul(2654435761) ^ (i / STRIDE).wrapping_mul(40503)) >> 7) as u8)
        .collect()
}

/// [`BankIo`], except that every read of block `bad` fails — a bad sector, a truncated file.
struct FailingIo {
    bank: BankIo,
    bad: u32,
}

impl BlockIo for FailingIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        if desc.id == self.bad {
            return Err(Error::backend(format!(
                "injected read error on block {}",
                desc.id
            )));
        }
        self.bank.read_block(desc, dst)
    }
}

/// A host tier over `bank` holding one block per expert under its global id: a caching one with
/// `slots` slots, or an arena-less one.
fn host_tier(bank: &[u8], slots: Option<usize>) -> Arc<HostPager> {
    host_tier_over(Arc::new(BankIo(bank.to_vec())), bank.len() / STRIDE, slots)
}

/// [`host_tier`] reading its `n_blocks` blocks through `io`.
fn host_tier_over(io: Arc<dyn BlockIo>, n_blocks: usize, slots: Option<usize>) -> Arc<HostPager> {
    let host = match slots {
        Some(n) => HostPager::new(n, STRIDE, io).unwrap(),
        None => HostPager::stream_only(STRIDE, io).unwrap(),
    };
    for b in 0..n_blocks {
        host.register(BlockDesc {
            id: b as u32,
            extents: vec![BlockExtent {
                offset: (b * STRIDE) as u64,
                len: STRIDE,
            }],
        })
        .unwrap();
    }
    Arc::new(host)
}

/// A session with one pool of `n_slots` holding [`LAYERS`] layers, each read from `bytes(layer)`.
/// Returns it with its placeholder buffers (kept alive for their identities).
fn session(
    be: &VulkanBackend,
    host: Option<Arc<HostPager>>,
    bytes: impl Fn(usize) -> ExpertBytes,
    n_slots: usize,
) -> (MoePagerSession, Vec<Box<dyn Buffer>>, Vec<usize>) {
    let mut sess = MoePagerSession::new(
        be,
        MoePagerLayout {
            n_blocks: LAYERS * N_EXPERT,
            pools: vec![MoePoolSpec {
                role: Role::Gate,
                slot_bytes: STRIDE,
                n_slots,
                host,
            }],
            ring_bytes: 2 * RING_HALF_EXPERTS * STRIDE,
        },
    )
    .unwrap();
    let placeholders: Vec<_> = (0..LAYERS)
        .map(|_| be.alloc_uninit(4, BufferUsage::Weights).unwrap())
        .collect();
    let ids: Vec<usize> = placeholders
        .iter()
        .map(|p| buffer_identity(p.as_ref()))
        .collect();
    for (l, &id) in ids.iter().enumerate() {
        sess.register(
            Role::Gate,
            id,
            ExpertSource {
                bytes: bytes(l),
                stride_bytes: STRIDE,
                layer_base: (l * N_EXPERT) as u32,
            },
            N_EXPERT,
        )
        .unwrap();
    }
    (sess, placeholders, ids)
}

/// Stage `ids` of the layer `buf_id` names as routed misses, as a decode token would.
fn route(be: &VulkanBackend, sess: &mut MoePagerSession, buf_id: usize, ids: &[u32]) {
    sess.begin_batch(buf_id).unwrap();
    let rec = be.recorder().unwrap();
    let mut cursor = 0;
    assert_eq!(
        sess.stage_role(&rec, 0, &mut cursor, buf_id, ids, false)
            .unwrap(),
        ids.len()
    );
    rec.finish().unwrap();
}

/// Every expert of every layer but the global ids in `skip`, read through a freshly frozen LUT
/// window, holds exactly its own bytes from `bank`.
fn check_every_expert(
    be: &VulkanBackend,
    sess: &mut MoePagerSession,
    ids: &[usize],
    bank: &[u8],
    skip: &[usize],
    what: &str,
) {
    let mut tape_cursor = 0;
    let windows: Vec<usize> = ids
        .iter()
        .map(|&id| sess.lut_window(&mut tape_cursor, id, N_EXPERT).unwrap() as usize)
        .collect();
    let mut tape = vec![0u32; tape_cursor];
    be.download(sess.tape(), bytemuck::cast_slice_mut(&mut tape))
        .unwrap();
    let mut arena = vec![0u8; sess.arena(ids[0]).unwrap().len_bytes()];
    be.download(sess.arena(ids[0]).unwrap(), &mut arena)
        .unwrap();
    for (l, &w) in windows.iter().enumerate() {
        for e in (0..N_EXPERT).filter(|e| !skip.contains(&(l * N_EXPERT + e))) {
            let slot = tape[w + e];
            assert_ne!(
                slot, NOT_RESIDENT,
                "{what}: layer {l} expert {e} is not resident"
            );
            let (slot, g) = (slot as usize, l * N_EXPERT + e);
            assert!(
                arena[slot * STRIDE..(slot + 1) * STRIDE] == bank[g * STRIDE..(g + 1) * STRIDE],
                "{what}: layer {l} expert {e}'s slot {slot} does not hold its bytes"
            );
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn fills_land_every_experts_own_bytes_from_every_tier() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let bank = synth_bank();
    let every = LAYERS * N_EXPERT;
    type Bytes<'a> = Box<dyn Fn(usize) -> ExpertBytes + 'a>;
    let mmap: Bytes = Box::new(|l| {
        ExpertBytes::Mmap(Arc::new(
            bank[l * N_EXPERT * STRIDE..(l + 1) * N_EXPERT * STRIDE].to_vec(),
        ))
    });
    let legs: Vec<(&str, Option<Arc<HostPager>>, Bytes)> = vec![
        ("mmap", None, mmap),
        // Two host slots: the routed experts pin through them, and the fills must neither pin
        // nor take them.
        (
            "caching host tier",
            Some(host_tier(&bank, Some(2))),
            Box::new(|_| ExpertBytes::Host),
        ),
        (
            "arena-less host tier",
            Some(host_tier(&bank, None)),
            Box::new(|_| ExpertBytes::Host),
        ),
    ];
    for (what, host, bytes) in legs {
        let (mut sess, _placeholders, ids) = session(&be, host, bytes, every);
        for &id in &ids {
            assert!(sess.holds_every_expert(id) && sess.fills_layer(&[id]));
            route(&be, &mut sess, id, ROUTED);
        }
        let mut rounds = 0;
        let mut stalled = 0;
        let mut fills = sess.fills();
        while ids.iter().any(|&id| !sess.all_resident(id, N_EXPERT)) {
            rounds += 1;
            for &id in &ids {
                if sess.all_resident(id, N_EXPERT) {
                    continue;
                }
                let rec = be.recorder().unwrap();
                sess.stage_ready_fills(&rec).unwrap();
                rec.finish().unwrap();
                // Every recording so far has completed: `finish` waits.
                sess.fills_copied(u64::MAX);
                sess.request_fills(&be, &[id]).unwrap();
            }
            drop(sess.end_forward());
            if sess.fills() > fills {
                (fills, stalled) = (sess.fills(), 0);
            } else {
                stalled += 1;
                assert!(
                    stalled < STALL_ROUNDS,
                    "{what}: no fill for {stalled} rounds with experts still absent ({fills} \
                     fills after {rounds} rounds)"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        check_every_expert(&be, &mut sess, &ids, &bank, &[], what);
        let (_, vram, below) = sess.pool_stats().remove(0);
        assert_eq!(
            sess.fills() as usize,
            every - LAYERS * ROUTED.len(),
            "{what}: every expert no token routed to is one fill"
        );
        if let Some(h) = below {
            assert_eq!(
                h.reads, every as u64,
                "{what}: the tier below read each expert once (its routed pin or its fill)"
            );
        }
        // A forward in which no layer asked for anything stops the reader.
        assert!(sess.fill_reader_active(), "{what}: the reader never ran");
        drop(sess.end_forward());
        assert!(
            !sess.fill_reader_active(),
            "{what}: the reader outlived the last expert it could fill"
        );
        eprintln!("{what}: {fills} fills in {rounds} rounds, {vram:?}");
    }
}

/// A pool one slot short of its registered experts never fills: a fill there could take the slot
/// a routed expert later needs, so the request is refused.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_pool_that_cannot_hold_every_expert_does_not_fill() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let bank = synth_bank();
    let (mut sess, _placeholders, ids) = session(
        &be,
        None,
        |l| {
            ExpertBytes::Mmap(Arc::new(
                bank[l * N_EXPERT * STRIDE..(l + 1) * N_EXPERT * STRIDE].to_vec(),
            ))
        },
        LAYERS * N_EXPERT - 1,
    );
    assert!(!sess.holds_every_expert(ids[0]) && !sess.fills_layer(&ids[..1]));
    let err = sess
        .request_fills(&be, &ids[..1])
        .expect_err("a fill in a pool that cannot hold every expert");
    assert!(err.to_string().contains("cannot hold"), "{err}");
    assert!(!sess.fill_reader_active());
}

/// A background read that fails never fails the forward that admits it, which needs no fill:
/// filling turns off for the session, its staging slots are freed, and the forwards go on as they
/// would without fills. A token that routes to the unreadable expert still fails, through the
/// demand path, and one that routes to a readable absent expert still gets it.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_failed_fill_read_turns_fills_off_without_failing_the_forward() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let bank = synth_bank();
    // Layer 0's expert after the routed ones: the first fill the reader is asked for.
    let bad = ROUTED.len();
    let io = Arc::new(FailingIo {
        bank: BankIo(bank.clone()),
        bad: bad as u32,
    });
    let host = host_tier_over(io, LAYERS * N_EXPERT, None);
    let (mut sess, _placeholders, ids) =
        session(&be, Some(host), |_| ExpertBytes::Host, LAYERS * N_EXPERT);
    route(&be, &mut sess, ids[0], ROUTED);
    let mut rounds = 0;
    while sess.fills_layer(&ids[..1]) {
        rounds += 1;
        assert!(
            rounds < STALL_ROUNDS,
            "filling never turned off ({} fills)",
            sess.fills()
        );
        let rec = be.recorder().unwrap();
        let staged = sess.stage_ready_fills(&rec);
        rec.finish().unwrap();
        sess.fills_copied(u64::MAX);
        if let Err(e) = staged {
            panic!("a failed fill read failed the forward: {e}");
        }
        sess.request_fills(&be, &ids[..1]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(
        !sess.fills_layer(&ids[1..]),
        "fills are off for every layer"
    );
    assert!(!sess.fill_reader_active());
    assert_eq!(sess.fills_pending(), 0);
    assert_eq!(sess.fill_slots_held(), 0, "a staging slot was left held");
    // Off is off: asking again reads nothing and fails nothing.
    sess.request_fills(&be, &ids[..1]).unwrap();
    let rec = be.recorder().unwrap();
    assert_eq!(sess.stage_ready_fills(&rec).unwrap(), 0);
    rec.finish().unwrap();
    assert!(!sess.fill_reader_active() && sess.fills_pending() == 0);
    // Demand still works: every readable expert of both layers, routed, holds its own bytes.
    // Layer 1 never asked for a fill, so each of its experts is a demand read.
    for &id in &ids {
        let local: Vec<u32> = (0..N_EXPERT as u32)
            .filter(|&e| !(id == ids[0] && e as usize == bad))
            .collect();
        // As many as one ring half holds per call, which `route` checks were all staged.
        for chunk in local.chunks(RING_HALF_EXPERTS) {
            route(&be, &mut sess, id, chunk);
        }
    }
    check_every_expert(&be, &mut sess, &ids, &bank, &[bad], "after a failed fill");
    // ...and the unreadable one still fails where a token needs it.
    sess.begin_batch(ids[0]).unwrap();
    let rec = be.recorder().unwrap();
    let mut cursor = 0;
    let err = sess
        .stage_role(&rec, 0, &mut cursor, ids[0], &[bad as u32], false)
        .expect_err("a routed expert that cannot be read");
    rec.discard().unwrap();
    assert!(err.to_string().contains("injected read error"), "{err}");
}

/// A layer's bank whose reads, while the gate is closed, wait until it opens: the reader blocks
/// on its first job, so the jobs behind it stay queued for as long as a test needs.
struct Gate {
    bytes: Vec<u8>,
    /// (closed, a read is waiting at it).
    state: std::sync::Mutex<(bool, bool)>,
    cv: std::sync::Condvar,
}

impl Gate {
    /// How long a read waits at the gate, or the test for a read to get there, before giving up —
    /// so a failed test cannot leave the reader blocked forever (its owner joins it on drop).
    const PATIENCE: std::time::Duration = std::time::Duration::from_secs(10);

    fn set_closed(&self, closed: bool) {
        self.state.lock().unwrap().0 = closed;
        self.cv.notify_all();
    }

    /// Wait until a read is blocked at the closed gate, or fail.
    fn wait_for_reader(&self) {
        let (st, _) = self
            .cv
            .wait_timeout_while(self.state.lock().unwrap(), Self::PATIENCE, |st| !st.1)
            .unwrap();
        assert!(st.1, "the reader never reached the gate");
    }
}

impl AsRef<[u8]> for Gate {
    fn as_ref(&self) -> &[u8] {
        let mut st = self.state.lock().unwrap();
        if st.0 {
            st.1 = true;
            self.cv.notify_all();
            st = self
                .cv
                .wait_timeout_while(st, Self::PATIENCE, |st| st.0)
                .unwrap()
                .0;
            st.1 = false;
        }
        &self.bytes
    }
}

/// A demand miss in any layer takes back every fill the reader has not started (demand reads come
/// first), and the layer's cursor moves back to the first one taken back: the next request asks
/// for exactly those experts again, rather than waiting until the one in progress has landed.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_demand_miss_takes_back_queued_fills_and_rewinds_the_layer() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let bank = synth_bank();
    let gate = Arc::new(Gate {
        bytes: bank[..N_EXPERT * STRIDE].to_vec(),
        state: std::sync::Mutex::new((false, false)),
        cv: std::sync::Condvar::new(),
    });
    let layer0 = Arc::clone(&gate);
    let (mut sess, _placeholders, ids) = session(
        &be,
        None,
        |l| {
            if l == 0 {
                ExpertBytes::Mmap(layer0.clone())
            } else {
                ExpertBytes::Mmap(Arc::new(
                    bank[l * N_EXPERT * STRIDE..(l + 1) * N_EXPERT * STRIDE].to_vec(),
                ))
            }
        },
        LAYERS * N_EXPERT,
    );
    for &id in &ids {
        route(&be, &mut sess, id, ROUTED);
    }
    let absent = N_EXPERT - ROUTED.len();
    gate.set_closed(true);
    // The first request only notes the demand count; the second asks for every absent expert.
    sess.request_fills(&be, &ids[..1]).unwrap();
    sess.request_fills(&be, &ids[..1]).unwrap();
    assert_eq!(sess.fills_pending(), absent);
    gate.wait_for_reader();
    // A token of layer 1 misses: layer 0's next request takes back all but the read in progress.
    route(&be, &mut sess, ids[1], &[ROUTED.len() as u32]);
    sess.request_fills(&be, &ids[..1]).unwrap();
    assert_eq!(
        sess.fills_pending(),
        1,
        "the queued fills were not taken back"
    );
    // No miss since: the request after asks for the taken-back experts again.
    sess.request_fills(&be, &ids[..1]).unwrap();
    assert_eq!(
        sess.fills_pending(),
        absent,
        "the layer's cursor did not move back to the experts taken back"
    );
    gate.set_closed(false);
    let mut rounds = 0;
    while !sess.all_resident(ids[0], N_EXPERT) {
        rounds += 1;
        assert!(rounds < STALL_ROUNDS, "layer 0 never filled");
        let rec = be.recorder().unwrap();
        sess.stage_ready_fills(&rec).unwrap();
        rec.finish().unwrap();
        sess.fills_copied(u64::MAX);
        sess.request_fills(&be, &ids[..1]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        sess.fills() as usize,
        absent,
        "each absent expert filled once"
    );
    // Layer 1 holds only what was routed to it.
    let layer1_absent: Vec<usize> = (N_EXPERT + ROUTED.len() + 1..LAYERS * N_EXPERT).collect();
    check_every_expert(
        &be,
        &mut sess,
        &ids,
        &bank,
        &layer1_absent,
        "after a rewind",
    );
    drop(sess.end_forward());
}
