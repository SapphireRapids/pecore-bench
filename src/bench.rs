//! Timing harness and the shared result data model.

use std::time::Instant;

use crate::cache::{self, CacheBufs};
use crate::kern::{Features, DEP_UNROLL};
use crate::sys::{self, CpuSet, PinMode};

/// One measured kernel on one core.
#[derive(Clone, Debug)]
pub struct KernelResult {
    pub id: String,
    pub group: String,
    pub name: String,
    pub kind: String,
    pub ops_per_iter: u64,
    pub ns_per_iter: f64,
    /// Core cycles per loop iteration. Derived from the add-chain reference
    /// measured *immediately after* this kernel, so slow clock drift during the
    /// sweep cannot distort it. RDTSC cannot be used for this: it ticks at the
    /// invariant TSC rate, not at the core clock, so it would penalise the
    /// higher-boosting P-cores.
    pub cyc_per_iter: f64,
    /// Core clock implied by that adjacent reference, GHz.
    pub freq_ghz: f64,
}

impl KernelResult {
    /// Core cycles per retired operation.
    pub fn cyc_per_op(&self) -> f64 {
        self.cyc_per_iter / self.ops_per_iter as f64
    }
}

/// One point of a cache curve.
#[derive(Clone, Debug)]
pub struct CachePoint {
    /// Working set size in bytes.
    pub bytes: usize,
    /// ns per dependent load (latency curve) or GB/s (bandwidth curve).
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct CoreResult {
    pub lpi: u8,
    pub cpu_set_id: u32,
    pub core_index: u8,
    pub llc_index: u8,
    pub numa: u8,
    pub efficiency: u8,
    pub is_p: bool,
    pub class_name: String,
    /// Core clock estimated on this core, GHz.
    pub freq_ghz: f64,
    /// Mean core clock observed by the per-kernel references, i.e. the clock the
    /// reported cycle counts are actually scaled by.
    pub freq_ghz_load: f64,
    pub tsc_ghz: f64,
    pub pin: PinMode,
    /// `GetCurrentProcessorNumberEx` read back after pinning: the core this
    /// actually ran on. Must equal `lpi` or the row is not trustworthy.
    pub proc_no: u32,
    pub apic_id: u32,
    pub features: Features,
    pub kernels: Vec<KernelResult>,
    pub cache_latency: Vec<CachePoint>,
    pub cache_bw: Vec<CachePoint>,
}

impl CoreResult {
    /// True when the pin was verified: the OS reports the thread running on
    /// exactly the logical processor we asked for.
    pub fn pin_verified(&self) -> bool {
        self.pin != PinMode::None && self.proc_no == self.lpi as u32
    }
}

/// Best-of-N wall time for one kernel call, in ns per loop iteration.
///
/// Minimum rather than mean/median: an interrupt or a frequency dip can only
/// ever make a pass slower, so the fastest pass is the one closest to the
/// hardware's true rate.
pub fn best_ns_per_iter(f: unsafe fn(u64) -> u64, iters: u64, passes: usize, warmup: usize) -> f64 {
    let mut best = f64::MAX;
    for i in 0..(warmup + passes) {
        let t0 = Instant::now();
        let r = unsafe { f(iters) };
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(r);
        if i >= warmup && dt < best {
            best = dt;
        }
    }
    best * 1e9 / iters as f64
}

/// Estimate the core clock by timing a dependent `add` chain, which retires at
/// exactly one instruction per cycle on every x86 core. This is the only honest
/// way to convert ns into core cycles on a hybrid part, because P-cores and
/// E-cores run at different frequencies and RDTSC is frequency-invariant.
///
/// The chain is unrolled, so the reported per-iteration time covers
/// `DEP_UNROLL` adds; divide by that to get the add rate, whose inverse is the
/// clock.
pub fn estimate_freq_ghz(add_dep: unsafe fn(u64) -> u64) -> f64 {
    let ns_per_iter = best_ns_per_iter(add_dep, 5_000_000, 9, 3);
    DEP_UNROLL as f64 / ns_per_iter
}

/// Run the whole compute kernel table on the calling (already pinned) thread.
///
/// Every kernel is followed immediately by a short add-chain reference on the
/// same core, and cycles are computed as `ns_kernel / ns_reference`. A single
/// calibration taken once before the sweep is not good enough here: a P-core
/// sheds several hundred MHz of boost over the ~2 s the sweep takes, so a
/// kernel measured late would be scaled by a clock that no longer applies.
pub fn run_kernels(features: &Features, quick: bool) -> Vec<KernelResult> {
    let add_dep = crate::kern::gpr_add_dep;
    let mut out = Vec::new();
    for k in crate::kern::table() {
        if !features.has(k.feat) {
            continue;
        }
        let (passes, warmup) = if quick { (3, 1) } else { (7, 2) };
        let ns = best_ns_per_iter(k.f, k.iters, passes, warmup);
        // Adjacent reference: short enough to be cheap, long enough to be stable.
        // One reference iteration covers DEP_UNROLL adds, so its time stands for
        // DEP_UNROLL core cycles.
        let ns_ref = best_ns_per_iter(add_dep, 2_000_000, 3, 1);
        out.push(KernelResult {
            id: k.id.to_string(),
            group: k.group.to_string(),
            name: k.name.to_string(),
            kind: k.kind.to_string(),
            ops_per_iter: k.ops_per_iter,
            ns_per_iter: ns,
            cyc_per_iter: ns * DEP_UNROLL as f64 / ns_ref,
            freq_ghz: DEP_UNROLL as f64 / ns_ref,
        });
    }
    out
}

/// Everything the benchmark knows about one logical processor.
///
/// Must be called from a dedicated thread: it pins the calling thread and
/// never restores it.
pub fn run_core(set: &CpuSet, is_p: bool, caches: &CacheBufs, quick: bool) -> CoreResult {
    sys::raise_priority();
    let pin = sys::pin_self(set);
    // Give the scheduler a moment to actually move us onto the target LP.
    std::thread::sleep(std::time::Duration::from_millis(3));
    let proc_no = sys::current_processor_number();
    let apic_id = sys::cpuid::apic_id();
    let features = Features::detect();
    let tsc = sys::tsc_hz();

    // Warm the core up so the clock has settled before anything is timed.
    for _ in 0..2 {
        std::hint::black_box(unsafe { kern_warm() });
    }
    let freq = estimate_freq_ghz(crate::kern::gpr_add_dep);
    let kernels = run_kernels(&features, quick);
    let freq_load = if kernels.is_empty() {
        freq
    } else {
        kernels.iter().map(|k| k.freq_ghz).sum::<f64>() / kernels.len() as f64
    };

    let (lat_passes, bw_passes) = if quick { (3, 3) } else { (5, 4) };
    let mut cache_latency = Vec::with_capacity(caches.sizes.len());
    let mut cache_bw = Vec::with_capacity(caches.sizes.len());
    for (i, &bytes) in caches.sizes.iter().enumerate() {
        let perm = &caches.bufs[i];
        cache_latency.push(CachePoint {
            bytes,
            value: cache::latency_ns(perm, lat_passes),
        });
        cache_bw.push(CachePoint {
            bytes,
            value: cache::read_bw_gbs(perm, bw_passes),
        });
    }

    CoreResult {
        lpi: set.lpi,
        cpu_set_id: set.id,
        core_index: set.core_index,
        llc_index: set.llc_index,
        numa: set.numa,
        efficiency: set.efficiency,
        is_p,
        class_name: if is_p { "P" } else { "E" }.to_string(),
        freq_ghz: freq,
        freq_ghz_load: freq_load,
        tsc_ghz: tsc,
        pin,
        proc_no,
        apic_id,
        features,
        kernels,
        cache_latency,
        cache_bw,
    }
}

#[inline(never)]
unsafe fn kern_warm() -> u64 {
    // A short dependent add chain, long enough to spin the core up to boost
    // but short enough not to matter for the total runtime.
    let n: u64 = 4_000_000;
    let out: u64;
    core::arch::asm!(
        "xor {out}, {out}",
        "2:",
        "add {out}, 1",
        "sub {n}, 1",
        "jnz 2b",
        out = out(reg) out,
        n = inout(reg) n => _,
        options(nostack),
    );
    out
}
