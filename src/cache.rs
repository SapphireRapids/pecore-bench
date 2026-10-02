//! Cache hierarchy probing.
//!
//! Latency: a dependent pointer chase through a *randomly permuted* linked list.
//! The permutation is what makes this a cache test rather than a prefetcher
//! test -- a sequential or strided walk would be consumed by the hardware
//! prefetchers and would report L1 speed no matter how big the array is.
//! Sattolo's algorithm is used so the list is one single cycle of length n:
//! chasing from any index visits every element exactly once per traversal.
//!
//! Bandwidth: a linear AVX2 read of the same buffers with eight independent
//! accumulators, reported in GB/s.

use std::hint::black_box;

/// Working-set sizes in bytes. Clustered around the interesting boundaries:
/// 32/48 KiB (L1d), 2 MiB (P-core L2), 4 MiB (E-core L2), 24-36 MiB (L3).
pub const SIZES: &[usize] = &[
    1 << 10, 2 << 10, 4 << 10, 8 << 10, 16 << 10, 24 << 10, 32 << 10, 48 << 10, 64 << 10,
    96 << 10, 128 << 10, 192 << 10, 256 << 10, 384 << 10, 512 << 10, 768 << 10,
    1 << 20, 3 << 19, 2 << 20, 3 << 20, 4 << 20, 6 << 20, 8 << 20, 12 << 20, 16 << 20,
    24 << 20, 32 << 20, 48 << 20, 64 << 20,
];

pub struct CacheBufs {
    pub sizes: Vec<usize>,
    pub bufs: Vec<Vec<u64>>,
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// One buffer per size; each buffer is used both as the permuted list (latency)
/// and as a flat array (bandwidth), which halves the footprint.
pub fn build(sizes: &[usize]) -> CacheBufs {
    let mut bufs = Vec::with_capacity(sizes.len());
    for (i, &bytes) in sizes.iter().enumerate() {
        let n = (bytes / 8).max(64);
        let mut idx: Vec<u64> = (0..n as u64).collect();
        let mut st = 0x9E3779B97F4A7C15u64 ^ (i as u64).wrapping_mul(0x2545F4914F6CDD1D);
        // Sattolo: single n-cycle, so the chase touches every line in the set.
        for k in (1..n).rev() {
            let j = (xorshift(&mut st) as usize) % k;
            idx.swap(k, j);
        }
        bufs.push(idx);
    }
    CacheBufs {
        sizes: sizes.to_vec(),
        bufs,
    }
}

#[inline(never)]
fn chase(perm: &[u64], start: usize, iters: usize) -> usize {
    let mut p = start;
    for _ in 0..iters {
        // SAFETY: p is always an index produced by a previous element of perm.
        p = unsafe { *perm.get_unchecked(p) } as usize;
    }
    p
}

fn time_chase(perm: &[u64], iters: usize) -> f64 {
    let t0 = std::time::Instant::now();
    let r = chase(perm, 0, iters);
    let dt = t0.elapsed().as_secs_f64();
    black_box(r);
    dt
}

/// ns per dependent load, best of `passes`. The iteration count is calibrated
/// per size so every pass lasts ~4 ms regardless of which cache level is hit.
pub fn latency_ns(perm: &[u64], passes: usize) -> f64 {
    let probe = 20_000usize;
    let t = time_chase(perm, probe);
    let want = 4e-3_f64;
    let mut iters = (want / t.max(1e-9) * probe as f64) as usize;
    iters = iters.clamp(probe, 4_000_000);
    let mut best = f64::MAX;
    for _ in 0..passes {
        let dt = time_chase(perm, iters);
        if dt < best {
            best = dt;
        }
    }
    best * 1e9 / iters as f64
}

/// Read the whole buffer once with eight independent 256-bit accumulators, so
/// the loop is bound by load throughput rather than by one dependency chain.
///
/// The loads are inline asm on purpose. Written with `_mm256_loadu_si256`, LTO
/// proved the function read-only, hoisted the call out of the repeat loop in
/// `read_bw_gbs`, and the benchmark reported its own loop overhead as
/// bandwidth: a flat 0.21 ns per pass at every working-set size, from 1 KiB to
/// 2 MiB. An asm block without `readonly`/`pure` is opaque, so nothing can move
/// it.
///
/// `n` must be a multiple of 32 elements; the caller rounds down.
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn bw_read_once(p: *const u64, n: usize) -> u64 {
    let out: u64;
    core::arch::asm!(
        "vpxor ymm0, ymm0, ymm0",
        "vpxor ymm1, ymm1, ymm1",
        "vpxor ymm2, ymm2, ymm2",
        "vpxor ymm3, ymm3, ymm3",
        "vpxor ymm4, ymm4, ymm4",
        "vpxor ymm5, ymm5, ymm5",
        "vpxor ymm6, ymm6, ymm6",
        "vpxor ymm7, ymm7, ymm7",
        "xor {i}, {i}",
        "2:",
        "vpaddq ymm0, ymm0, [{p} + {i}*8]",
        "vpaddq ymm1, ymm1, [{p} + {i}*8 + 32]",
        "vpaddq ymm2, ymm2, [{p} + {i}*8 + 64]",
        "vpaddq ymm3, ymm3, [{p} + {i}*8 + 96]",
        "vpaddq ymm4, ymm4, [{p} + {i}*8 + 128]",
        "vpaddq ymm5, ymm5, [{p} + {i}*8 + 160]",
        "vpaddq ymm6, ymm6, [{p} + {i}*8 + 192]",
        "vpaddq ymm7, ymm7, [{p} + {i}*8 + 224]",
        "add {i}, 32",
        "cmp {i}, {n}",
        "jb 2b",
        "vpaddq ymm0, ymm0, ymm1",
        "vpaddq ymm2, ymm2, ymm3",
        "vpaddq ymm4, ymm4, ymm5",
        "vpaddq ymm6, ymm6, ymm7",
        "vpaddq ymm0, ymm0, ymm2",
        "vpaddq ymm4, ymm4, ymm6",
        "vpaddq ymm0, ymm0, ymm4",
        "vmovq {out}, xmm0",
        p = in(reg) p,
        n = in(reg) n,
        i = out(reg) _,
        out = out(reg) out,
        out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
        out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
        options(nostack),
    );
    out
}

/// SSE2 fallback for the bandwidth probe, for x86-64 CPUs without AVX2.
///
/// SSE2 is part of the x86-64 baseline, so this path runs on every machine the
/// binary can start on. Four 128-bit loads per iteration instead of eight
/// 256-bit ones; the point is not to be fast but to be *available* -- without
/// it a pre-AVX2 CPU would take #UD on the first bandwidth pass.
#[inline(never)]
unsafe fn bw_read_once_sse2(p: *const u64, n: usize) -> u64 {
    let out: u64;
    core::arch::asm!(
        "pxor xmm0, xmm0",
        "pxor xmm1, xmm1",
        "pxor xmm2, xmm2",
        "pxor xmm3, xmm3",
        "xor {i}, {i}",
        "2:",
        "paddq xmm0, [{p} + {i}*8]",
        "paddq xmm1, [{p} + {i}*8 + 16]",
        "paddq xmm2, [{p} + {i}*8 + 32]",
        "paddq xmm3, [{p} + {i}*8 + 48]",
        "add {i}, 16",
        "cmp {i}, {n}",
        "jb 2b",
        "paddq xmm0, xmm1",
        "paddq xmm2, xmm3",
        "paddq xmm0, xmm2",
        "movq {out}, xmm0",
        p = in(reg) p,
        n = in(reg) n,
        i = out(reg) _,
        out = out(reg) out,
        out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
        options(nostack),
    );
    out
}

/// Read bandwidth in GB/s (10^9 bytes/s), best of `passes`.
pub fn read_bw_gbs(buf: &[u64], passes: usize) -> f64 {
    // 16-element (128 B) granularity on the fallback, 32-element (256 B) on AVX2.
    let avx2 = std::arch::is_x86_feature_detected!("avx2");
    let read_once: unsafe fn(*const u64, usize) -> u64 = if avx2 {
        bw_read_once
    } else {
        bw_read_once_sse2
    };
    let n = buf.len() & !31;
    let bytes = n * 8;
    if n == 0 {
        return f64::NAN;
    }
    let p = buf.as_ptr();
    // Fault the pages in and get the code hot before timing anything: the first
    // read of a 64 MiB buffer is a page-fault storm and would size the repeat
    // loop for work that never happens again.
    for _ in 0..2 {
        black_box(unsafe { read_once(p, n) });
    }
    // Calibrate over 64 reads so the probe is far above the clock's resolution,
    // then aim for ~8 ms per pass.
    let t0 = std::time::Instant::now();
    for _ in 0..64 {
        black_box(unsafe { read_once(p, n) });
    }
    let t = t0.elapsed().as_secs_f64() / 64.0;
    let iters = ((8e-3 / t.max(1e-9)) as usize).clamp(1, 4_000_000);
    let mut best = f64::MAX;
    for _ in 0..passes {
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            black_box(unsafe { read_once(p, n) });
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    if !(best > 0.0) {
        return f64::NAN;
    }
    // Each pass reads the whole buffer `iters` times.
    (bytes * iters) as f64 / best / 1e9
}
