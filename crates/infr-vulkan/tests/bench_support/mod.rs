//! Test-local submission bounds, not a watchdog guarantee: the first dispatch cannot be preempted.
use std::sync::Arc;
use std::time::{Duration, Instant};

use infr_core::backend::Backend;
use infr_core::config::{Config, ConfigLayer};
use infr_vulkan::{Recorder, VulkanBackend};

const BATCH_BUDGET: Duration = Duration::from_millis(50);
const MAX_DISPATCHES: usize = 16;
const MAX_OPERATIONS: usize = 16;

pub fn backend() -> VulkanBackend {
    initialize(false).expect("benchmark requires a Vulkan device")
}

pub fn optional_backend() -> Option<VulkanBackend> {
    initialize(true)
}

fn unavailable(error: &infr_core::Error, explicit: bool) -> bool {
    !explicit
        && matches!(error, infr_core::Error::Backend(message)
            if message.starts_with("ash::Entry::load:")
                || message == "no Vulkan physical devices"
                || message == "create_instance: Unable to find a Vulkan driver"
                || message.starts_with("Vulkan is not supported on Apple."))
}

fn initialize(optional: bool) -> Option<VulkanBackend> {
    let cfg =
        Config::load_from_layers(&[ConfigLayer::env().expect("benchmark environment config")]);
    let selected = cfg.device.dev.clone();
    let be = match VulkanBackend::new_with(Arc::new(cfg)) {
        Ok(be) => be,
        Err(error) if optional && unavailable(&error, selected.is_some()) => {
            eprintln!("skip: {error}");
            return None;
        }
        Err(error) => panic!("benchmark Vulkan device initialization: {error}"),
    };
    let caps = be.capabilities();
    println!(
        "device={selected:?} name={} profile={} f16_coopmat={} i8_dot={} batch_budget={BATCH_BUDGET:?} dispatch_cap={MAX_DISPATCHES}",
        caps.name,
        if caps.integrated {
            "integrated-small"
        } else {
            "discrete"
        },
        caps.f16_coopmat(),
        caps.i8_dot,
    );
    Some(be)
}

fn batch_operations(
    integrated: bool,
    remaining: usize,
    dispatches_per_op: usize,
    elapsed_per_op: Duration,
    budget: Duration,
) -> usize {
    if remaining == 0 || dispatches_per_op == 0 || dispatches_per_op > MAX_DISPATCHES {
        return 0;
    }
    let time_cap =
        (budget.as_nanos() / elapsed_per_op.as_nanos().max(1)).min(MAX_OPERATIONS as u128) as usize;
    remaining
        .min(if integrated { 1 } else { MAX_OPERATIONS })
        .min(MAX_DISPATCHES / dispatches_per_op)
        .min(time_cap.max(1))
}

#[derive(Debug)]
pub struct Batch {
    pub operations: usize,
    pub dispatches: usize,
    pub elapsed: Duration,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Complete,
    InsufficientSamples,
    Stopped { remaining: usize },
}

#[derive(Debug)]
pub struct Timing {
    pub cold: Batch,
    pub batches: Vec<Batch>,
    pub outcome: Outcome,
}

impl Timing {
    pub fn mean_us(&self) -> Option<f64> {
        if self.outcome != Outcome::Complete {
            return None;
        }
        let operations: usize = self.batches.iter().map(|b| b.operations).sum();
        if operations == 0 {
            return None;
        }
        Some(
            self.batches
                .iter()
                .map(|b| b.elapsed.as_secs_f64())
                .sum::<f64>()
                * 1e6
                / operations as f64,
        )
    }
}

fn submit(
    be: &VulkanBackend,
    label: &str,
    operations: usize,
    op: &impl Fn(&Recorder),
) -> Result<Batch, String> {
    let start = Instant::now();
    let rec = be.recorder().map_err(|e| e.to_string())?;
    for _ in 0..operations {
        op(&rec);
        let dispatches = rec.dispatches();
        if dispatches > MAX_DISPATCHES {
            rec.discard().map_err(|e| e.to_string())?;
            return Err(format!(
                "{label}: discarded {dispatches} dispatches before submission (cap {MAX_DISPATCHES})"
            ));
        }
    }
    let dispatches = rec.dispatches();
    if dispatches == 0 {
        rec.discard().map_err(|e| e.to_string())?;
        return Err(format!("{label}: operation recorded no dispatches"));
    }
    rec.finish().map_err(|e| e.to_string())?;
    let elapsed = start.elapsed();
    println!(
        "{label}: operations={operations} dispatches={dispatches} record+submit+wait={elapsed:?}"
    );
    Ok(Batch {
        operations,
        dispatches,
        elapsed,
    })
}

pub fn time(
    be: &VulkanBackend,
    label: &str,
    repetitions: usize,
    op: impl Fn(&Recorder),
) -> Result<Timing, String> {
    let cold = submit(be, &format!("{label} cold"), 1, &op)?;
    let timing = measure(
        cold,
        be.capabilities().integrated,
        repetitions,
        |operations| submit(be, label, operations, &op),
    )?;
    println!(
        "{label} summary: cold_operations={} cold_dispatches={} cold_record+submit+wait={:?} operations={} dispatches={} record+submit+wait={:?} outcome={:?} mean_us={:?} kernels={:?}",
        timing.cold.operations,
        timing.cold.dispatches,
        timing.cold.elapsed,
        timing.batches.iter().map(|b| b.operations).sum::<usize>(),
        timing.batches.iter().map(|b| b.dispatches).sum::<usize>(),
        timing.batches.iter().map(|b| b.elapsed).sum::<Duration>(),
        timing.outcome,
        timing.mean_us(),
        be.built_kernel_names()
    );
    Ok(timing)
}

fn measure(
    cold: Batch,
    integrated: bool,
    repetitions: usize,
    mut submit_batch: impl FnMut(usize) -> Result<Batch, String>,
) -> Result<Timing, String> {
    let mut remaining = repetitions;
    let mut elapsed_per_op = cold.elapsed;
    let mut dispatches_per_op = cold.dispatches;
    let mut batches = Vec::new();
    let mut excessive = cold.elapsed > BATCH_BUDGET;
    while remaining > 0 && !excessive {
        let operations = batch_operations(
            integrated,
            remaining,
            dispatches_per_op,
            elapsed_per_op,
            BATCH_BUDGET,
        );
        let batch = submit_batch(operations)?;
        remaining -= operations;
        dispatches_per_op = batch.dispatches.div_ceil(operations);
        elapsed_per_op = batch.elapsed / operations as u32;
        excessive = batch.elapsed > BATCH_BUDGET;
        batches.push(batch);
    }
    let outcome = if excessive {
        Outcome::Stopped { remaining }
    } else if batches.is_empty() {
        Outcome::InsufficientSamples
    } else {
        Outcome::Complete
    };
    Ok(Timing {
        cold,
        batches,
        outcome,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::backend::BufferUsage;

    #[test]
    fn budget_boundaries() {
        let unit = Duration::from_millis(1);
        assert_eq!(batch_operations(true, 50, 2, unit, BATCH_BUDGET), 1);
        assert_eq!(batch_operations(false, 50, 2, unit, BATCH_BUDGET), 8);
        assert_eq!(
            batch_operations(false, 50, MAX_DISPATCHES, unit, BATCH_BUDGET),
            1
        );
        assert_eq!(
            batch_operations(false, 50, MAX_DISPATCHES + 1, unit, BATCH_BUDGET),
            0
        );
        assert_eq!(batch_operations(false, 0, 1, unit, BATCH_BUDGET), 0);
        assert_eq!(batch_operations(false, 50, 0, unit, BATCH_BUDGET), 0);
        assert_eq!(
            batch_operations(false, 50, 1, Duration::ZERO, BATCH_BUDGET),
            MAX_OPERATIONS
        );
        assert_eq!(batch_operations(false, 50, 1, unit, Duration::ZERO), 1);
        assert_eq!(
            batch_operations(false, 50, 1, unit, Duration::from_nanos(1)),
            1
        );
        assert_eq!(batch_operations(false, 50, 1, unit, unit * 3), 3);
        let mut remaining = 19;
        let mut batches = Vec::new();
        while remaining > 0 {
            let n = batch_operations(false, remaining, 2, unit, BATCH_BUDGET);
            batches.push(n);
            remaining -= n;
        }
        assert_eq!(batches, [8, 8, 3]);
    }

    #[test]
    fn optional_backend_only_skips_unrequested_absence() {
        for message in [
            "ash::Entry::load: library not found",
            "no Vulkan physical devices",
            "create_instance: Unable to find a Vulkan driver",
            "Vulkan is not supported on Apple. Use the native Metal backend:",
        ] {
            let error = infr_core::Error::Backend(message.into());
            assert!(unavailable(&error, false));
            assert!(!unavailable(&error, true));
        }
        for message in [
            "invalid INFR_SG",
            "INFR_DEV: no such Vulkan device",
            "create_device: ERROR_DEVICE_LOST",
        ] {
            assert!(!unavailable(
                &infr_core::Error::Backend(message.into()),
                false
            ));
        }
    }

    #[test]
    fn deterministic_partitions_and_stop_outcomes() {
        let batch = |operations, millis| Batch {
            operations,
            dispatches: operations,
            elapsed: Duration::from_millis(millis),
        };
        for (integrated, expected) in [(false, vec![2, 1]), (true, vec![1, 1, 1])] {
            let timing =
                measure(batch(1, 20), integrated, 3, |n| Ok(batch(n, n as u64 * 20))).unwrap();
            assert_eq!(
                timing
                    .batches
                    .iter()
                    .map(|b| b.operations)
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(timing.outcome, Outcome::Complete);
            assert_eq!(timing.mean_us(), Some(20_000.0));
        }
        let cold_stop = measure(batch(1, 51), false, 3, |_| {
            panic!("must stop before submission")
        })
        .unwrap();
        assert!(cold_stop.batches.is_empty());
        assert_eq!(cold_stop.outcome, Outcome::Stopped { remaining: 3 });
        assert_eq!(cold_stop.mean_us(), None);
        for repetitions in [2, 3] {
            let mut calls = 0;
            let stopped = measure(batch(1, 20), false, repetitions, |n| {
                calls += 1;
                assert_eq!(calls, 1, "must not submit after excessive batch");
                Ok(batch(n, 51))
            })
            .unwrap();
            assert_eq!(calls, 1);
            assert_eq!(stopped.batches[0].operations, 2);
            assert_eq!(
                stopped.outcome,
                Outcome::Stopped {
                    remaining: repetitions - 2
                }
            );
            assert_eq!(stopped.mean_us(), None);
        }
        let empty = measure(batch(1, 1), false, 0, |_| panic!("no samples requested")).unwrap();
        assert_eq!(empty.outcome, Outcome::InsufficientSamples);
        assert_eq!(empty.mean_us(), None);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU (bounded helper regression)"]
    fn bounded_gpu_batches_and_discard() {
        let be = backend();
        let w = be.alloc(16, BufferUsage::Weights).unwrap();
        let x = be.alloc(16, BufferUsage::Activations).unwrap();
        let y = be.alloc(4, BufferUsage::Activations).unwrap();
        be.upload(w.as_ref(), bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]))
            .unwrap();
        be.upload(x.as_ref(), bytemuck::cast_slice(&[2.0f32, 3.0, 4.0, 5.0]))
            .unwrap();
        let op = |rec: &Recorder| rec.linear_f32(w.as_ref(), x.as_ref(), y.as_ref(), 1, 4, 1);
        // Prime compilation separately; cold compilation still obeys the actual-dispatch cap.
        time(&be, "helper prime", 0, op).unwrap();
        let timing = time(&be, "helper regression", 3, op).unwrap();
        assert_eq!(timing.cold.dispatches, 1);
        let executed = timing.batches.iter().map(|b| b.operations).sum::<usize>();
        assert!(executed <= 3);
        for batch in std::iter::once(&timing.cold).chain(&timing.batches) {
            assert!((1..=MAX_DISPATCHES).contains(&batch.dispatches));
            assert_eq!(batch.dispatches, batch.operations);
            if be.capabilities().integrated {
                assert_eq!(batch.operations, 1);
            }
        }
        match timing.outcome {
            Outcome::Complete => {
                assert_eq!(executed, 3);
                assert!(timing.mean_us().unwrap().is_finite());
            }
            Outcome::Stopped { remaining } => {
                assert_eq!(executed + remaining, 3);
                assert_eq!(timing.mean_us(), None);
            }
            Outcome::InsufficientSamples => panic!("requested samples must complete or stop"),
        }
        let mut out = [0u8; 4];
        be.download(y.as_ref(), &mut out).unwrap();
        assert_eq!(f32::from_ne_bytes(out), 40.0);
        be.upload(y.as_ref(), &(-7.0f32).to_ne_bytes()).unwrap();
        let error = time(&be, "helper over-cap", 1, |rec| {
            for _ in 0..=MAX_DISPATCHES {
                op(rec);
            }
        })
        .unwrap_err();
        assert!(
            error.contains("discarded 17 dispatches before submission"),
            "{error}"
        );
        be.download(y.as_ref(), &mut out).unwrap();
        assert_eq!(
            f32::from_ne_bytes(out),
            -7.0,
            "over-cap recording must never execute"
        );
    }
}
