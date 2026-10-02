//! Hand-written kernels. Everything is inline `asm!` on purpose: the goal is to
//! measure *one specific instruction* under a *known* dependency structure, not
//! whatever LLVM decides to emit for a Rust loop.
//!
//! Two shapes per instruction:
//!   *_dep -- one accumulator, so every instruction waits for the previous one.
//!            Result is the instruction's latency (cycles).
//!   *_tp  -- eight independent accumulators, enough to keep a 4-5 cycle
//!            latency busy. Result is the instruction's throughput (ops/cycle).
//!
//! The dependent chains are unrolled `DEP_UNROLL` times per loop iteration. A
//! three-instruction loop (`op`, `sub`, `jnz`) turned out to be measurably
//! sensitive to where the linker happened to place it: the same kernel built
//! into two different binaries differed by up to 60% on the E-cores, because a
//! loop that straddles a 32-byte boundary stops being served from the decoded
//! stream buffer. Amortising the loop overhead over several instructions makes
//! the per-op number a property of the instruction instead of the build.
//!
//! AVX-VNNI encoding trap (this machine, verified the hard way): LLVM's
//! assembler knows only the EVEX form of the `vpdpbusd` mnemonic, and EVEX is
//! #UD on any chip without AVX-512. On this Raptor Lake ES the mnemonic form
//! traps even though the instruction itself runs perfectly well on P-cores and
//! E-cores alike, so the VNNI kernels emit the VEX bytes directly (see
//! `vpdpbusd!`). The flat intrinsic `_mm256_dpbusd_epi32` is likewise the
//! AVX512VL form; `_mm256_dpbusd_avx_epi32` is the 256-bit VEX one.
//!
//! Immediate-operand trap (this machine, verified): a dependent chain of
//! `add r64, imm` measures at ALU *throughput* on the P-cores — four of them
//! cost the same as the bare `sub`/`jnz` loop, about 1 cycle, and eight cost
//! about 1.5, i.e. exactly the rename width. The same chain written
//! `add r64, r64` measures 1 cycle per add on both core types, which is the
//! textbook latency. `imul r64, r64, imm` is unaffected (3 cycles per op).
//! So the add chain that calibrates the clock must take its operand from a
//! register, never from an immediate.

/// Instructions per loop iteration in every `*_dep` kernel.
pub const DEP_UNROLL: u64 = 4;

/// Body of a dependent chain: the same instruction `DEP_UNROLL` times, each one
/// reading the accumulator the previous one wrote.
macro_rules! dep_body {
    ($insn:literal) => {
        concat!(
            $insn,
            "\n",
            $insn,
            "\n",
            $insn,
            "\n",
            $insn
        )
    };
}

use core::arch::x86_64::{
    __m128d, __m256i, _mm256_add_epi64, _mm256_extract_epi64, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_set1_pd, _mm256_setzero_si256, _mm_add_pd, _mm_cvtsd_f64,
    _mm_set1_pd,
};

// ---------------------------------------------------------------- GPR kernels

macro_rules! gpr_chain {
    ($(#[$attr:meta])* $dep:ident, $tp:ident, $kval:literal, $insn:literal, $tp_body:literal) => {
        $(#[$attr])*
        #[inline(never)]
        pub unsafe fn $dep(n: u64) -> u64 {
            let out: u64;
            let k: u64 = $kval;
            core::arch::asm!(
                "xor {out}, {out}",
                "2:",
                dep_body!($insn),
                "sub {n}, 1",
                "jnz 2b",
                out = out(reg) out,
                k = in(reg) k,
                n = inout(reg) n => _,
                options(nostack),
            );
            out
        }

        $(#[$attr])*
        #[inline(never)]
        pub unsafe fn $tp(n: u64) -> u64 {
            let o0: u64; let o1: u64; let o2: u64; let o3: u64;
            let o4: u64; let o5: u64; let o6: u64; let o7: u64;
            core::arch::asm!(
                "xor rax, rax\n xor rcx, rcx\n xor rdx, rdx\n xor r14, r14",
                "xor rsi, rsi\n xor rdi, rdi\n xor r12, r12\n xor r13, r13",
                "2:",
                $tp_body,
                "sub r9, 1",
                "jnz 2b",
                out("rax") o0, out("rcx") o1, out("rdx") o2, out("r14") o3,
                out("rsi") o4, out("rdi") o5, out("r12") o6, out("r13") o7,
                in("r8") 1u64,
                inout("r9") n => _,
                options(nostack),
            );
            o0 ^ o1 ^ o2 ^ o3 ^ o4 ^ o5 ^ o6 ^ o7
        }
    };
}

gpr_chain!(
    gpr_add_dep, gpr_add_tp, 1,
    "add {out}, {k}",
    "add rax, r8\n add rcx, r8\n add rdx, r8\n add r14, r8\n add rsi, r8\n add rdi, r8\n add r12, r8\n add r13, r8"
);

gpr_chain!(
    gpr_imul_dep, gpr_imul_tp, 3,
    "imul {out}, {k}",
    "imul rax, rax, 3\n imul rcx, rcx, 3\n imul rdx, rdx, 3\n imul r14, r14, 3\n imul rsi, rsi, 3\n imul rdi, rdi, 3\n imul r12, r12, 3\n imul r13, r13, 3"
);

// ------------------------------------------------------- scalar FP kernels

macro_rules! xmm_chain {
    ($(#[$attr:meta])* $dep:ident, $tp:ident, $insn:literal, $tp_body:literal, $c:expr) => {
        $(#[$attr])*
        #[inline(never)]
        pub unsafe fn $dep(n: u64) -> u64 {
            let c1 = $c;
            let c2 = $c;
            let out: __m128d;
            core::arch::asm!(
                "vxorps {out}, {out}, {out}",
                "2:",
                dep_body!($insn),
                "sub {n}, 1",
                "jnz 2b",
                out = out(xmm_reg) out,
                in("xmm8") c1,
                in("xmm9") c2,
                n = inout(reg) n => _,
                options(nostack),
            );
            _mm_cvtsd_f64(out).to_bits()
        }

        $(#[$attr])*
        #[inline(never)]
        pub unsafe fn $tp(n: u64) -> u64 {
            let c1 = $c;
            let c2 = $c;
            let o0: __m128d; let o1: __m128d; let o2: __m128d; let o3: __m128d;
            let o4: __m128d; let o5: __m128d; let o6: __m128d; let o7: __m128d;
            core::arch::asm!(
                "vxorps xmm0, xmm0, xmm0\n vxorps xmm1, xmm1, xmm1\n vxorps xmm2, xmm2, xmm2\n vxorps xmm3, xmm3, xmm3",
                "vxorps xmm4, xmm4, xmm4\n vxorps xmm5, xmm5, xmm5\n vxorps xmm6, xmm6, xmm6\n vxorps xmm7, xmm7, xmm7",
                "2:",
                $tp_body,
                "sub r9, 1",
                "jnz 2b",
                out("xmm0") o0, out("xmm1") o1, out("xmm2") o2, out("xmm3") o3,
                out("xmm4") o4, out("xmm5") o5, out("xmm6") o6, out("xmm7") o7,
                in("xmm8") c1,
                in("xmm9") c2,
                inout("r9") n => _,
                options(nostack),
            );
            let s = _mm_add_pd(
                _mm_add_pd(o0, o1),
                _mm_add_pd(_mm_add_pd(o2, o3), _mm_add_pd(_mm_add_pd(o4, o5), _mm_add_pd(o6, o7))),
            );
            _mm_cvtsd_f64(s).to_bits()
        }
    };
}

xmm_chain!(
    #[target_feature(enable = "avx")]
    fp_addsd_dep, fp_addsd_tp,
    "addsd {out}, xmm8",
    "addsd xmm0, xmm8\n addsd xmm1, xmm8\n addsd xmm2, xmm8\n addsd xmm3, xmm8\n addsd xmm4, xmm8\n addsd xmm5, xmm8\n addsd xmm6, xmm8\n addsd xmm7, xmm8",
    _mm_set1_pd(1.0)
);

xmm_chain!(
    #[target_feature(enable = "avx")]
    fp_mulsd_dep, fp_mulsd_tp,
    "mulsd {out}, xmm8",
    "mulsd xmm0, xmm8\n mulsd xmm1, xmm8\n mulsd xmm2, xmm8\n mulsd xmm3, xmm8\n mulsd xmm4, xmm8\n mulsd xmm5, xmm8\n mulsd xmm6, xmm8\n mulsd xmm7, xmm8",
    _mm_set1_pd(1.0)
);

xmm_chain!(
    #[target_feature(enable = "avx,fma")]
    fp_fma_dep, fp_fma_tp,
    "vfmadd231sd {out}, xmm8, xmm9",
    "vfmadd231sd xmm0, xmm8, xmm9\n vfmadd231sd xmm1, xmm8, xmm9\n vfmadd231sd xmm2, xmm8, xmm9\n vfmadd231sd xmm3, xmm8, xmm9\n vfmadd231sd xmm4, xmm8, xmm9\n vfmadd231sd xmm5, xmm8, xmm9\n vfmadd231sd xmm6, xmm8, xmm9\n vfmadd231sd xmm7, xmm8, xmm9",
    _mm_set1_pd(1.0)
);

// ------------------------------------------------------------- YMM kernels

macro_rules! ymm_chain {
    ($(#[$attr:meta])* $dep:ident, $tp:ident, $insn:literal, $tp_body:literal, $c1:expr, $c2:expr) => {
        $(#[$attr])*
        #[inline(never)]
        pub unsafe fn $dep(n: u64) -> u64 {
            let c1 = $c1;
            let c2 = $c2;
            let out: __m256i;
            core::arch::asm!(
                "vpxor {out}, {out}, {out}",
                "2:",
                dep_body!($insn),
                "sub {n}, 1",
                "jnz 2b",
                out = out(ymm_reg) out,
                in("ymm8") c1,
                in("ymm9") c2,
                n = inout(reg) n => _,
                options(nostack),
            );
            _mm256_extract_epi64::<0>(out) as u64
        }

        $(#[$attr])*
        #[inline(never)]
        pub unsafe fn $tp(n: u64) -> u64 {
            let c1 = $c1;
            let c2 = $c2;
            let o0: __m256i; let o1: __m256i; let o2: __m256i; let o3: __m256i;
            let o4: __m256i; let o5: __m256i; let o6: __m256i; let o7: __m256i;
            core::arch::asm!(
                "vpxor ymm0, ymm0, ymm0\n vpxor ymm1, ymm1, ymm1\n vpxor ymm2, ymm2, ymm2\n vpxor ymm3, ymm3, ymm3",
                "vpxor ymm4, ymm4, ymm4\n vpxor ymm5, ymm5, ymm5\n vpxor ymm6, ymm6, ymm6\n vpxor ymm7, ymm7, ymm7",
                "2:",
                $tp_body,
                "sub r9, 1",
                "jnz 2b",
                out("ymm0") o0, out("ymm1") o1, out("ymm2") o2, out("ymm3") o3,
                out("ymm4") o4, out("ymm5") o5, out("ymm6") o6, out("ymm7") o7,
                in("ymm8") c1,
                in("ymm9") c2,
                inout("r9") n => _,
                options(nostack),
            );
            let s = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_add_epi64(o0, o1), _mm256_add_epi64(o2, o3)),
                _mm256_add_epi64(_mm256_add_epi64(o4, o5), _mm256_add_epi64(o6, o7)),
            );
            _mm256_extract_epi64::<0>(s) as u64
        }
    };
}

ymm_chain!(
    #[target_feature(enable = "avx2")]
    avx2_paddd_dep, avx2_paddd_tp,
    "vpaddd {out}, {out}, ymm8",
    "vpaddd ymm0, ymm0, ymm8\n vpaddd ymm1, ymm1, ymm8\n vpaddd ymm2, ymm2, ymm8\n vpaddd ymm3, ymm3, ymm8\n vpaddd ymm4, ymm4, ymm8\n vpaddd ymm5, ymm5, ymm8\n vpaddd ymm6, ymm6, ymm8\n vpaddd ymm7, ymm7, ymm8",
    _mm256_set1_epi32(1),
    _mm256_set1_epi32(1)
);

ymm_chain!(
    #[target_feature(enable = "avx2")]
    avx2_pmulld_dep, avx2_pmulld_tp,
    "vpmulld {out}, {out}, ymm8",
    "vpmulld ymm0, ymm0, ymm8\n vpmulld ymm1, ymm1, ymm8\n vpmulld ymm2, ymm2, ymm8\n vpmulld ymm3, ymm3, ymm8\n vpmulld ymm4, ymm4, ymm8\n vpmulld ymm5, ymm5, ymm8\n vpmulld ymm6, ymm6, ymm8\n vpmulld ymm7, ymm7, ymm8",
    _mm256_set1_epi32(3),
    _mm256_set1_epi32(1)
);

ymm_chain!(
    #[target_feature(enable = "avx2")]
    avx2_fma_dep, avx2_fma_tp,
    "vfmadd231pd {out}, ymm8, ymm9",
    "vfmadd231pd ymm0, ymm8, ymm9\n vfmadd231pd ymm1, ymm8, ymm9\n vfmadd231pd ymm2, ymm8, ymm9\n vfmadd231pd ymm3, ymm8, ymm9\n vfmadd231pd ymm4, ymm8, ymm9\n vfmadd231pd ymm5, ymm8, ymm9\n vfmadd231pd ymm6, ymm8, ymm9\n vfmadd231pd ymm7, ymm8, ymm9",
    _mm256_set1_pd(1.0),
    _mm256_set1_pd(1.0)
);

/// `vpdpbusd ymm{d}, ymm8, ymm9` as raw VEX bytes.
///
/// Emitted as bytes because LLVM's assembler only produces the EVEX encoding of
/// this mnemonic, which is #UD without AVX-512 — see the module doc. Layout:
/// VEX.256.66.0F38.W0 50 /r, so `C4` + `C2` (R=1,X=1,B=0,map=0F38) + `3D`
/// (W=0, vvvv=~ymm8, L=1, pp=66) + opcode `50` + ModRM with reg = the
/// destination and rm = ymm9.
macro_rules! vpdpbusd {
    (0) => { ".byte 0xC4, 0xC2, 0x3D, 0x50, 0xC1" };
    (1) => { ".byte 0xC4, 0xC2, 0x3D, 0x50, 0xC9" };
    (2) => { ".byte 0xC4, 0xC2, 0x3D, 0x50, 0xD1" };
    (3) => { ".byte 0xC4, 0xC2, 0x3D, 0x50, 0xD9" };
    (4) => { ".byte 0xC4, 0xC2, 0x3D, 0x50, 0xE1" };
    (5) => { ".byte 0xC4, 0xC2, 0x3D, 0x50, 0xE9" };
    (6) => { ".byte 0xC4, 0xC2, 0x3D, 0x50, 0xF1" };
    (7) => { ".byte 0xC4, 0xC2, 0x3D, 0x50, 0xF9" };
}

/// The dependent chain: `vpdpbusd ymm0, ymm8, ymm9` DEP_UNROLL times, each one
/// reading the accumulator the previous one wrote.
macro_rules! vnni_dep_body {
    () => {
        concat!(
            vpdpbusd!(0),
            "\n",
            vpdpbusd!(0),
            "\n",
            vpdpbusd!(0),
            "\n",
            vpdpbusd!(0)
        )
    };
}

/// Latency. Same shape as the other `*_dep` kernels: one accumulator.
#[target_feature(enable = "avx2")]
#[inline(never)]
pub unsafe fn vnni_dpbusd_dep(n: u64) -> u64 {
    let c1 = _mm256_set1_epi32(0x04040404); // u8 lanes
    let c2 = _mm256_set1_epi32(0x03030303); // i8 lanes
    let out: u64;
    core::arch::asm!(
        "vpxor ymm0, ymm0, ymm0",
        "2:",
        vnni_dep_body!(),
        "sub {n}, 1",
        "jnz 2b",
        "vmovq {out}, xmm0",
        out = out(reg) out,
        in("ymm8") c1,
        in("ymm9") c2,
        n = inout(reg) n => _,
        out("ymm0") _,
        options(nostack),
    );
    out
}

/// Throughput. Eight independent accumulators, enough to keep a 4-5 cycle
/// latency busy.
#[target_feature(enable = "avx2")]
#[inline(never)]
pub unsafe fn vnni_dpbusd_tp(n: u64) -> u64 {
    let c1 = _mm256_set1_epi32(0x04040404);
    let c2 = _mm256_set1_epi32(0x03030303);
    let o0: __m256i; let o1: __m256i; let o2: __m256i; let o3: __m256i;
    let o4: __m256i; let o5: __m256i; let o6: __m256i; let o7: __m256i;
    core::arch::asm!(
        "vpxor ymm0, ymm0, ymm0\n vpxor ymm1, ymm1, ymm1\n vpxor ymm2, ymm2, ymm2\n vpxor ymm3, ymm3, ymm3",
        "vpxor ymm4, ymm4, ymm4\n vpxor ymm5, ymm5, ymm5\n vpxor ymm6, ymm6, ymm6\n vpxor ymm7, ymm7, ymm7",
        "2:",
        concat!(
            vpdpbusd!(0), "\n", vpdpbusd!(1), "\n", vpdpbusd!(2), "\n", vpdpbusd!(3), "\n",
            vpdpbusd!(4), "\n", vpdpbusd!(5), "\n", vpdpbusd!(6), "\n", vpdpbusd!(7)
        ),
        "sub r9, 1",
        "jnz 2b",
        out("ymm0") o0, out("ymm1") o1, out("ymm2") o2, out("ymm3") o3,
        out("ymm4") o4, out("ymm5") o5, out("ymm6") o6, out("ymm7") o7,
        in("ymm8") c1,
        in("ymm9") c2,
        inout("r9") n => _,
        options(nostack),
    );
    let s = _mm256_add_epi64(
        _mm256_add_epi64(_mm256_add_epi64(o0, o1), _mm256_add_epi64(o2, o3)),
        _mm256_add_epi64(_mm256_add_epi64(o4, o5), _mm256_add_epi64(o6, o7)),
    );
    _mm256_extract_epi64::<0>(s) as u64
}

/// The pre-VNNI int8 recipe: `vpmaddubsw` + `vpmaddwd` + `vpaddd` does the same
/// 32-MAC dot product in three instructions instead of one `vpdpbusd`. Counted
/// in instructions per iteration so it is directly comparable with the VNNI row.
#[target_feature(enable = "avx2")]
#[inline(never)]
pub unsafe fn avx2_int8_recipe_tp(n: u64) -> u64 {
    let b = _mm256_set1_epi32(0x04040404); // u8 lanes
    let a = _mm256_set1_epi32(0x03030303); // i8 lanes
    let ones = _mm256_set1_epi16(1);
    let zero = _mm256_setzero_si256();
    let o0: __m256i; let o1: __m256i; let o2: __m256i; let o3: __m256i;
    let o4: __m256i; let o5: __m256i; let o6: __m256i; let o7: __m256i;
    core::arch::asm!(
        "2:",
        "vpmaddubsw ymm0, ymm8, ymm9\n vpmaddwd ymm0, ymm0, ymm10\n vpaddd ymm0, ymm0, ymm11",
        "vpmaddubsw ymm1, ymm8, ymm9\n vpmaddwd ymm1, ymm1, ymm10\n vpaddd ymm1, ymm1, ymm11",
        "vpmaddubsw ymm2, ymm8, ymm9\n vpmaddwd ymm2, ymm2, ymm10\n vpaddd ymm2, ymm2, ymm11",
        "vpmaddubsw ymm3, ymm8, ymm9\n vpmaddwd ymm3, ymm3, ymm10\n vpaddd ymm3, ymm3, ymm11",
        "vpmaddubsw ymm4, ymm8, ymm9\n vpmaddwd ymm4, ymm4, ymm10\n vpaddd ymm4, ymm4, ymm11",
        "vpmaddubsw ymm5, ymm8, ymm9\n vpmaddwd ymm5, ymm5, ymm10\n vpaddd ymm5, ymm5, ymm11",
        "vpmaddubsw ymm6, ymm8, ymm9\n vpmaddwd ymm6, ymm6, ymm10\n vpaddd ymm6, ymm6, ymm11",
        "vpmaddubsw ymm7, ymm8, ymm9\n vpmaddwd ymm7, ymm7, ymm10\n vpaddd ymm7, ymm7, ymm11",
        "sub r9, 1",
        "jnz 2b",
        out("ymm0") o0, out("ymm1") o1, out("ymm2") o2, out("ymm3") o3,
        out("ymm4") o4, out("ymm5") o5, out("ymm6") o6, out("ymm7") o7,
        in("ymm8") b, in("ymm9") a, in("ymm10") ones, in("ymm11") zero,
        inout("r9") n => _,
        options(nostack),
    );
    let s = _mm256_add_epi64(
        _mm256_add_epi64(_mm256_add_epi64(o0, o1), _mm256_add_epi64(o2, o3)),
        _mm256_add_epi64(_mm256_add_epi64(o4, o5), _mm256_add_epi64(o6, o7)),
    );
    _mm256_extract_epi64::<0>(s) as u64
}

// --------------------------------------------------------------- ZMM kernels
//
// AVX-512 kernels, one per subset (F / BW / DQ / VL / VNNI). None of them has
// ever been executed on the machine this was developed on: its AVX-512 is fused
// off, `Features::detect()` reports every subset absent here, and the report
// prints these rows as "不支持". They exist so the same binary carried to a
// machine that has AVX-512 measures it, each gated on its own feature.
//
// Encoding is left to LLVM's assembler with the feature enabled on the
// function, which is correct by construction. The EVEX-only trap that forced
// hand-written VEX bytes for the 256-bit `vpdpbusd` does not apply here: a zmm
// operand has no VEX form at all, so the EVEX encoding is the only one that
// exists. What *is* hand-written is where the constants live: they are
// materialised inside the asm from an immediate and broadcast to zmm, so these
// kernels keep the plain `fn(u64) -> u64` shape of the table and need no
// AVX-512 intrinsics.

/// One dependent chain and one 8-chain throughput kernel over 512-bit vectors.
/// `$setup` must materialise the two constant operands (zmm8/zmm9) using the
/// `{c}` scratch operand, `$dep_insn` is the chained instruction, `$tp_body`
/// eight independent copies of it on zmm0..zmm7.
macro_rules! zmm_chain {
    ($(#[$attr:meta])* $dep:ident, $tp:ident, $setup:literal, $dep_insn:literal, $tp_body:literal) => {
        $(#[$attr])*
        #[inline(never)]
        pub unsafe fn $dep(n: u64) -> u64 {
            let out: u64;
            core::arch::asm!(
                $setup,
                "vpxord zmm0, zmm0, zmm0",
                "2:",
                dep_body!($dep_insn),
                "sub {n}, 1",
                "jnz 2b",
                "vmovq {out}, xmm0",
                out = out(reg) out,
                c = out(reg) _,
                n = inout(reg) n => _,
                out("zmm0") _, out("zmm8") _, out("zmm9") _,
                options(nostack),
            );
            out
        }

        $(#[$attr])*
        #[inline(never)]
        pub unsafe fn $tp(n: u64) -> u64 {
            let out: u64;
            core::arch::asm!(
                $setup,
                "vpxord zmm0, zmm0, zmm0\n vpxord zmm1, zmm1, zmm1\n vpxord zmm2, zmm2, zmm2\n vpxord zmm3, zmm3, zmm3",
                "vpxord zmm4, zmm4, zmm4\n vpxord zmm5, zmm5, zmm5\n vpxord zmm6, zmm6, zmm6\n vpxord zmm7, zmm7, zmm7",
                "2:",
                $tp_body,
                "sub {n}, 1",
                "jnz 2b",
                "vpaddq zmm0, zmm0, zmm1\n vpaddq zmm2, zmm2, zmm3\n vpaddq zmm4, zmm4, zmm5\n vpaddq zmm6, zmm6, zmm7",
                "vpaddq zmm0, zmm0, zmm2\n vpaddq zmm4, zmm4, zmm6\n vpaddq zmm0, zmm0, zmm4",
                "vmovq {out}, xmm0",
                out = out(reg) out,
                c = out(reg) _,
                n = inout(reg) n => _,
                out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
                out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
                out("zmm8") _, out("zmm9") _,
                options(nostack),
            );
            out
        }
    };
}

zmm_chain!(
    #[target_feature(enable = "avx512f")]
    zmm_paddd_dep, zmm_paddd_tp,
    "mov {c:e}, 1\n vmovd xmm8, {c:e}\n vpbroadcastd zmm8, xmm8\n vmovd xmm9, {c:e}\n vpbroadcastd zmm9, xmm9",
    "vpaddd zmm0, zmm0, zmm8",
    "vpaddd zmm0, zmm0, zmm8\n vpaddd zmm1, zmm1, zmm8\n vpaddd zmm2, zmm2, zmm8\n vpaddd zmm3, zmm3, zmm8\n vpaddd zmm4, zmm4, zmm8\n vpaddd zmm5, zmm5, zmm8\n vpaddd zmm6, zmm6, zmm8\n vpaddd zmm7, zmm7, zmm8"
);

zmm_chain!(
    #[target_feature(enable = "avx512f")]
    zmm_pmulld_dep, zmm_pmulld_tp,
    "mov {c:e}, 1\n vmovd xmm8, {c:e}\n vpbroadcastd zmm8, xmm8\n vmovd xmm9, {c:e}\n vpbroadcastd zmm9, xmm9",
    "vpmulld zmm0, zmm0, zmm8",
    "vpmulld zmm0, zmm0, zmm8\n vpmulld zmm1, zmm1, zmm8\n vpmulld zmm2, zmm2, zmm8\n vpmulld zmm3, zmm3, zmm8\n vpmulld zmm4, zmm4, zmm8\n vpmulld zmm5, zmm5, zmm8\n vpmulld zmm6, zmm6, zmm8\n vpmulld zmm7, zmm7, zmm8"
);

zmm_chain!(
    #[target_feature(enable = "avx512f")]
    zmm_fma_dep, zmm_fma_tp,
    "mov {c}, 0x3FF0000000000000\n vmovq xmm8, {c}\n vpbroadcastq zmm8, xmm8\n vmovq xmm9, {c}\n vpbroadcastq zmm9, xmm9",
    "vfmadd231pd zmm0, zmm8, zmm9",
    "vfmadd231pd zmm0, zmm8, zmm9\n vfmadd231pd zmm1, zmm8, zmm9\n vfmadd231pd zmm2, zmm8, zmm9\n vfmadd231pd zmm3, zmm8, zmm9\n vfmadd231pd zmm4, zmm8, zmm9\n vfmadd231pd zmm5, zmm8, zmm9\n vfmadd231pd zmm6, zmm8, zmm9\n vfmadd231pd zmm7, zmm8, zmm9"
);

zmm_chain!(
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512dq")]
    zmm_mullq_dep, zmm_mullq_tp,
    "mov {c}, 3\n vmovq xmm8, {c}\n vpbroadcastq zmm8, xmm8\n vmovq xmm9, {c}\n vpbroadcastq zmm9, xmm9",
    "vpmullq zmm0, zmm0, zmm8",
    "vpmullq zmm0, zmm0, zmm8\n vpmullq zmm1, zmm1, zmm8\n vpmullq zmm2, zmm2, zmm8\n vpmullq zmm3, zmm3, zmm8\n vpmullq zmm4, zmm4, zmm8\n vpmullq zmm5, zmm5, zmm8\n vpmullq zmm6, zmm6, zmm8\n vpmullq zmm7, zmm7, zmm8"
);

zmm_chain!(
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512bw")]
    zmm_addb_dep, zmm_addb_tp,
    "mov {c:e}, 1\n vmovd xmm8, {c:e}\n vpbroadcastb zmm8, xmm8\n vmovd xmm9, {c:e}\n vpbroadcastb zmm9, xmm9",
    "vpaddb zmm0, zmm0, zmm8",
    "vpaddb zmm0, zmm0, zmm8\n vpaddb zmm1, zmm1, zmm8\n vpaddb zmm2, zmm2, zmm8\n vpaddb zmm3, zmm3, zmm8\n vpaddb zmm4, zmm4, zmm8\n vpaddb zmm5, zmm5, zmm8\n vpaddb zmm6, zmm6, zmm8\n vpaddb zmm7, zmm7, zmm8"
);

zmm_chain!(
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512vnni")]
    zmm_vnni_dpbusd_dep, zmm_vnni_dpbusd_tp,
    "mov {c:e}, 1\n vmovd xmm8, {c:e}\n vpbroadcastd zmm8, xmm8\n vmovd xmm9, {c:e}\n vpbroadcastd zmm9, xmm9",
    "vpdpbusd zmm0, zmm8, zmm9",
    "vpdpbusd zmm0, zmm8, zmm9\n vpdpbusd zmm1, zmm8, zmm9\n vpdpbusd zmm2, zmm8, zmm9\n vpdpbusd zmm3, zmm8, zmm9\n vpdpbusd zmm4, zmm8, zmm9\n vpdpbusd zmm5, zmm8, zmm9\n vpdpbusd zmm6, zmm8, zmm9\n vpdpbusd zmm7, zmm8, zmm9"
);

/// 256-bit EVEX: `vpternlogd ymm` has no VEX form, so this is the kernel that
/// proves AVX-512VL is live — it encodes as EVEX.256 and traps outright on a
/// machine without AVX-512.
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vl")]
#[inline(never)]
pub unsafe fn vl_ternlog_dep(n: u64) -> u64 {
    let out: u64;
    core::arch::asm!(
        "mov {c:e}, 1",
        "vmovd xmm8, {c:e}\n vpbroadcastd ymm8, xmm8",
        "mov {c:e}, 3",
        "vmovd xmm9, {c:e}\n vpbroadcastd ymm9, xmm9",
        "vpxor ymm0, ymm0, ymm0",
        "2:",
        dep_body!("vpternlogd ymm0, ymm8, ymm9, 0x96"),
        "sub {n}, 1",
        "jnz 2b",
        "vmovq {out}, xmm0",
        out = out(reg) out,
        c = out(reg) _,
        n = inout(reg) n => _,
        out("ymm0") _, out("ymm8") _, out("ymm9") _,
        options(nostack),
    );
    out
}

// ------------------------------------------------------------ kernel table

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Feat {
    /// Baseline x86-64.
    Base,
    Avx,
    Fma,
    Avx2,
    AvxVnni,
    Avx512F,
    Avx512BW,
    Avx512DQ,
    Avx512VL,
    Avx512Vnni,
}

impl Feat {
    /// Short name for "skipped because this CPU lacks it" messages.
    pub fn label(self) -> &'static str {
        match self {
            Feat::Base => "x86-64",
            Feat::Avx => "AVX",
            Feat::Fma => "FMA",
            Feat::Avx2 => "AVX2",
            Feat::AvxVnni => "AVX-VNNI",
            Feat::Avx512F => "AVX-512F",
            Feat::Avx512BW => "AVX-512BW",
            Feat::Avx512DQ => "AVX-512DQ",
            Feat::Avx512VL => "AVX-512VL",
            Feat::Avx512Vnni => "AVX-512VNNI",
        }
    }
}

#[derive(Clone, Copy)]
pub struct Kernel {
    pub id: &'static str,
    pub group: &'static str,
    pub name: &'static str,
    /// "延迟" = one dependent chain; "吞吐" = 8 independent chains.
    pub kind: &'static str,
    /// Instructions executed per loop iteration.
    pub ops_per_iter: u64,
    pub feat: Feat,
    pub iters: u64,
    pub f: unsafe fn(u64) -> u64,
}

pub fn table() -> Vec<Kernel> {
    use Feat::*;
    vec![
        Kernel { id: "int_add_lat",    group: "整数 (GPR)",      name: "add r64 依赖链 x4",        kind: "延迟", ops_per_iter: DEP_UNROLL,  feat: Base,   iters: 5_000_000, f: gpr_add_dep },
        Kernel { id: "int_add_tp",     group: "整数 (GPR)",      name: "add r64 x8 独立链",     kind: "吞吐", ops_per_iter: 8,  feat: Base,   iters: 5_000_000,  f: gpr_add_tp },
        Kernel { id: "int_imul_lat",   group: "整数 (GPR)",      name: "imul r64 依赖链 x4",       kind: "延迟", ops_per_iter: DEP_UNROLL,  feat: Base,   iters: 2_000_000,  f: gpr_imul_dep },
        Kernel { id: "int_imul_tp",    group: "整数 (GPR)",      name: "imul r64 x8 独立链",    kind: "吞吐", ops_per_iter: 8,  feat: Base,   iters: 5_000_000,  f: gpr_imul_tp },
        Kernel { id: "fp_addsd_lat",   group: "浮点标量 (SSE)",  name: "addsd 依赖链 x4",          kind: "延迟", ops_per_iter: DEP_UNROLL,  feat: Avx,    iters: 2_000_000,  f: fp_addsd_dep },
        Kernel { id: "fp_mulsd_lat",   group: "浮点标量 (SSE)",  name: "mulsd 依赖链 x4",          kind: "延迟", ops_per_iter: DEP_UNROLL,  feat: Avx,    iters: 2_000_000,  f: fp_mulsd_dep },
        Kernel { id: "fp_mulsd_tp",    group: "浮点标量 (SSE)",  name: "mulsd x8 独立链",       kind: "吞吐", ops_per_iter: 8,  feat: Avx,    iters: 5_000_000,  f: fp_mulsd_tp },
        Kernel { id: "fp_addsd_tp",    group: "浮点标量 (SSE)",  name: "addsd x8 独立链",       kind: "吞吐", ops_per_iter: 8,  feat: Avx,    iters: 5_000_000,  f: fp_addsd_tp },
        Kernel { id: "fp_fma_lat",     group: "浮点标量 (FMA)",  name: "vfmadd231sd 依赖链 x4",    kind: "延迟", ops_per_iter: DEP_UNROLL,  feat: Fma,    iters: 2_000_000,  f: fp_fma_dep },
        Kernel { id: "fp_fma_tp",      group: "浮点标量 (FMA)",  name: "vfmadd231sd x8 独立链", kind: "吞吐", ops_per_iter: 8,  feat: Fma,    iters: 5_000_000,  f: fp_fma_tp },
        Kernel { id: "avx2_paddd_lat", group: "AVX2 整数",       name: "vpaddd ymm 依赖链 x4",     kind: "延迟", ops_per_iter: DEP_UNROLL,  feat: Avx2,   iters: 5_000_000, f: avx2_paddd_dep },
        Kernel { id: "avx2_paddd_tp",  group: "AVX2 整数",       name: "vpaddd ymm x8 独立链",  kind: "吞吐", ops_per_iter: 8,  feat: Avx2,   iters: 5_000_000,  f: avx2_paddd_tp },
        Kernel { id: "avx2_pmulld_lat",group: "AVX2 整数",       name: "vpmulld ymm 依赖链 x4",    kind: "延迟", ops_per_iter: DEP_UNROLL,  feat: Avx2,   iters: 1_000_000,  f: avx2_pmulld_dep },
        Kernel { id: "avx2_pmulld_tp", group: "AVX2 整数",       name: "vpmulld ymm x8 独立链", kind: "吞吐", ops_per_iter: 8,  feat: Avx2,   iters: 5_000_000,  f: avx2_pmulld_tp },
        Kernel { id: "avx2_fma_lat",   group: "AVX2 浮点",       name: "vfmadd231pd ymm 依赖链 x4",kind: "延迟", ops_per_iter: DEP_UNROLL,  feat: Avx2,   iters: 2_000_000,  f: avx2_fma_dep },
        Kernel { id: "avx2_fma_tp",    group: "AVX2 浮点",       name: "vfmadd231pd ymm x8",    kind: "吞吐", ops_per_iter: 8,  feat: Avx2,   iters: 5_000_000,  f: avx2_fma_tp },
        Kernel { id: "avx2_int8_recipe_tp", group: "AVX-VNNI",   name: "老配方 vpmaddubsw+vpmaddwd+vpaddd x8", kind: "吞吐", ops_per_iter: 24, feat: Avx2, iters: 3_000_000, f: avx2_int8_recipe_tp },
        Kernel { id: "vnni_dpbusd_lat",group: "AVX-VNNI",        name: "vpdpbusd ymm 依赖链 x4",   kind: "延迟", ops_per_iter: DEP_UNROLL,  feat: AvxVnni,iters: 2_000_000,  f: vnni_dpbusd_dep },
        Kernel { id: "vnni_dpbusd_tp", group: "AVX-VNNI",        name: "vpdpbusd ymm x8 独立链",kind: "吞吐", ops_per_iter: 8,  feat: AvxVnni,iters: 5_000_000,  f: vnni_dpbusd_tp },
        // AVX-512 (each gated on its own subset; skipped entirely on CPUs without it)
        Kernel { id: "avx512_paddd_lat",  group: "AVX-512 整数",  name: "vpaddd zmm 依赖链 x4",     kind: "延迟", ops_per_iter: DEP_UNROLL, feat: Avx512F,   iters: 2_000_000, f: zmm_paddd_dep },
        Kernel { id: "avx512_paddd_tp",   group: "AVX-512 整数",  name: "vpaddd zmm x8 独立链",     kind: "吞吐", ops_per_iter: 8, feat: Avx512F,   iters: 3_000_000, f: zmm_paddd_tp },
        Kernel { id: "avx512_pmulld_lat", group: "AVX-512 整数",  name: "vpmulld zmm 依赖链 x4",    kind: "延迟", ops_per_iter: DEP_UNROLL, feat: Avx512F,   iters: 1_000_000, f: zmm_pmulld_dep },
        Kernel { id: "avx512_pmulld_tp",  group: "AVX-512 整数",  name: "vpmulld zmm x8 独立链",    kind: "吞吐", ops_per_iter: 8, feat: Avx512F,   iters: 3_000_000, f: zmm_pmulld_tp },
        Kernel { id: "avx512_fma_lat",    group: "AVX-512 浮点",  name: "vfmadd231pd zmm 依赖链 x4",kind: "延迟", ops_per_iter: DEP_UNROLL, feat: Avx512F,   iters: 2_000_000, f: zmm_fma_dep },
        Kernel { id: "avx512_fma_tp",     group: "AVX-512 浮点",  name: "vfmadd231pd zmm x8 独立链",kind: "吞吐", ops_per_iter: 8, feat: Avx512F,   iters: 3_000_000, f: zmm_fma_tp },
        Kernel { id: "avx512_mullq_lat",  group: "AVX-512 DQ",    name: "vpmullq zmm 依赖链 x4",    kind: "延迟", ops_per_iter: DEP_UNROLL, feat: Avx512DQ,  iters: 1_000_000, f: zmm_mullq_dep },
        Kernel { id: "avx512_mullq_tp",   group: "AVX-512 DQ",    name: "vpmullq zmm x8 独立链",    kind: "吞吐", ops_per_iter: 8, feat: Avx512DQ,  iters: 3_000_000, f: zmm_mullq_tp },
        Kernel { id: "avx512_addb_dep",   group: "AVX-512 BW",    name: "vpaddb zmm 依赖链 x4",     kind: "延迟", ops_per_iter: DEP_UNROLL, feat: Avx512BW,  iters: 2_000_000, f: zmm_addb_dep },
        Kernel { id: "avx512_addb_tp",    group: "AVX-512 BW",    name: "vpaddb zmm x8 独立链",     kind: "吞吐", ops_per_iter: 8, feat: Avx512BW,  iters: 3_000_000, f: zmm_addb_tp },
        Kernel { id: "avx512_vl_ternlog_lat", group: "AVX-512 VL", name: "vpternlogd ymm 依赖链 x4", kind: "延迟", ops_per_iter: DEP_UNROLL, feat: Avx512VL,  iters: 2_000_000, f: vl_ternlog_dep },
        Kernel { id: "avx512_vnni_dpbusd_lat", group: "AVX-512 VNNI", name: "vpdpbusd zmm 依赖链 x4", kind: "延迟", ops_per_iter: DEP_UNROLL, feat: Avx512Vnni, iters: 2_000_000, f: zmm_vnni_dpbusd_dep },
        Kernel { id: "avx512_vnni_dpbusd_tp",  group: "AVX-512 VNNI", name: "vpdpbusd zmm x8 独立链", kind: "吞吐", ops_per_iter: 8, feat: Avx512Vnni, iters: 3_000_000, f: zmm_vnni_dpbusd_tp },
    ]
}

/// Per-core feature read. Runs on whatever core the caller is pinned to.
///
/// Deliberately built on `is_x86_feature_detected!` rather than a hand-rolled
/// CPUID bit test. A hand-rolled read of CPUID.(EAX=07H,ECX=0):EBX[bit 4] —
/// which is HLE, not AVX_VNNI — reported AVX-VNNI as absent on this machine and
/// silently dropped both VNNI kernels from the report, even though the
/// instruction executes on every core here. The standard library reads the bit
/// that actually means it. CPUID is package-wide, so every core reports the
/// same answer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Features {
    pub avx: bool,
    pub fma: bool,
    pub avx2: bool,
    pub avxvnni: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512dq: bool,
    pub avx512vl: bool,
    pub avx512vnni: bool,
}

impl Features {
    pub fn detect() -> Features {
        Features {
            avx: std::arch::is_x86_feature_detected!("avx"),
            fma: std::arch::is_x86_feature_detected!("fma"),
            avx2: std::arch::is_x86_feature_detected!("avx2"),
            avxvnni: std::arch::is_x86_feature_detected!("avxvnni"),
            avx512f: std::arch::is_x86_feature_detected!("avx512f"),
            avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
            avx512dq: std::arch::is_x86_feature_detected!("avx512dq"),
            avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
            avx512vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
        }
    }

    pub fn has(&self, f: Feat) -> bool {
        match f {
            Feat::Base => true,
            Feat::Avx => self.avx,
            Feat::Fma => self.fma,
            Feat::Avx2 => self.avx2,
            Feat::AvxVnni => self.avxvnni,
            Feat::Avx512F => self.avx512f,
            Feat::Avx512BW => self.avx512bw,
            Feat::Avx512DQ => self.avx512dq,
            Feat::Avx512VL => self.avx512vl,
            Feat::Avx512Vnni => self.avx512vnni,
        }
    }

    /// One line for the console banner and the report header.
    pub fn summary(&self) -> String {
        let k = |b: bool| if b { "+" } else { "-" };
        format!(
            "AVX {}  FMA {}  AVX2 {}  AVX-VNNI {}  |  AVX-512F {}  BW {}  DQ {}  VL {}  VNNI {}",
            k(self.avx),
            k(self.fma),
            k(self.avx2),
            k(self.avxvnni),
            k(self.avx512f),
            k(self.avx512bw),
            k(self.avx512dq),
            k(self.avx512vl),
            k(self.avx512vnni),
        )
    }
}
