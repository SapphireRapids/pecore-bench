//! Win32 plumbing: CPU-set enumeration, per-core thread pinning, CPUID, TSC.
//!
//! `GetSystemCpuSetInformation` is the only supported way to learn *which*
//! logical processors are P-cores and which are E-cores. It is used for
//! classification only; the actual pinning is done with `SetThreadGroupAffinity`,
//! because on this machine `GetCurrentProcessorNumberEx` disagrees with the CPU
//! set a thread is confined to (verified: pinning to CPU set 275 reported
//! processor 4 and 5 in two runs, with clock estimates that contradicted the
//! reported number). Affinity masks are the one mechanism whose effect can be
//! read back unambiguously.

use std::ffi::c_void;

type Bool = i32;
type Handle = *mut c_void;
type Ulong = u32;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GroupAffinity {
    pub mask: usize,
    pub group: u16,
    pub reserved: [u16; 3],
}

#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentThread() -> Handle;
    fn GetCurrentProcess() -> Handle;
    fn SetThreadPriority(hThread: Handle, nPriority: i32) -> Bool;
    fn SetThreadAffinityMask(hThread: Handle, dwThreadAffinityMask: usize) -> usize;
    fn SetThreadGroupAffinity(
        hThread: Handle,
        GroupAffinity: *const GroupAffinity,
        PreviousGroupAffinity: *mut GroupAffinity,
    ) -> Bool;
    fn GetSystemCpuSetInformation(
        Information: *mut u8,
        BufferLength: Ulong,
        ReturnedLength: *mut Ulong,
        Process: Handle,
        Flags: Ulong,
    ) -> Bool;
    fn SetThreadSelectedCpuSets(
        Thread: Handle,
        CpuSetIds: *const Ulong,
        CpuSetIdCount: Ulong,
    ) -> Bool;
    fn GetStdHandle(nStdHandle: Ulong) -> Handle;
    fn GetConsoleMode(hConsoleHandle: Handle, lpMode: *mut Ulong) -> Bool;
}

#[link(name = "shell32")]
extern "system" {
    fn ShellExecuteW(
        hwnd: Handle,
        lpOperation: *const u16,
        lpFile: *const u16,
        lpParameters: *const u16,
        lpDirectory: *const u16,
        nShowCmd: i32,
    ) -> Handle;
}

/// True when standard input is a real console, i.e. the program was started by
/// double-clicking or from a shell without redirection. A redirected stdin
/// fails `GetConsoleMode`, which is how the interactive menu stays out of the
/// way of scripted runs.
pub fn stdin_is_console() -> bool {
    unsafe {
        const STD_INPUT_HANDLE: Ulong = -10i32 as Ulong;
        let h = GetStdHandle(STD_INPUT_HANDLE);
        if h.is_null() || h as isize == -1 {
            return false;
        }
        let mut mode: Ulong = 0;
        GetConsoleMode(h, &mut mode) != 0
    }
}

/// Hand a file (here: the generated HTML report) to the user's default browser
/// via `ShellExecuteW`. No effect and `false` on failure, so the caller can
/// print the path for manual opening.
pub fn open_in_default_browser(path: &std::path::Path) -> bool {
    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let op: Vec<u16> = "open".encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        const SW_SHOWNORMAL: i32 = 1;
        let r = ShellExecuteW(
            std::ptr::null_mut(),
            op.as_ptr(),
            wide.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            SW_SHOWNORMAL,
        );
        // ShellExecuteW returns a value <= 32 on failure, per its contract.
        r as isize > 32
    }
}

/// Block until the user presses Enter. Used to keep the console window alive
/// when the program was started by double-click, so the printed summary does
/// not vanish with the process.
pub fn pause_for_enter(prompt: &str) {
    use std::io::Write;
    print!("{}", prompt);
    std::io::stdout().flush().ok();
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).ok();
}

use std::os::windows::ffi::OsStrExt;

/// Mirror of the first 24 bytes of `SYSTEM_CPU_SET_INFORMATION` (winbase.h).
/// The on-wire `Size` is 32 on this build; the extra 8 bytes are the trailing
/// `Reserved` ULONG plus alignment, and the walk honours `size` rather than
/// assuming it. Field offsets were verified against a labelled byte dump:
/// `20|00 00 00 00|00 01 00 00|00 00 00 00|00 00 01 01|...` decodes to
/// Size=32, Type=CpuSet, Tag=256, Group=0, Lpi=0, Core=0 -- the only layout
/// under which entries 0 and 1 differ (the published Id@12/Group@16 layout
/// makes both entries read Lpi=1, Core=1).
#[repr(C)]
#[derive(Clone, Copy)]
struct RawCpuSet {
    size: Ulong,
    kind: Ulong,
    tag: Ulong,
    group: u16,
    logical_processor_index: u8,
    core_index: u8,
    last_level_cache_index: u8,
    numa_node_index: u8,
    efficiency_class: u8,
    parked: u8,
    allocated: u8,
    allocated_to_target_process: u8,
    real_time: u8,
}

#[derive(Clone, Debug)]
pub struct CpuSet {
    /// The `Id`/`Tag` value `SetThreadSelectedCpuSets` wants.
    pub id: u32,
    pub group: u16,
    /// Group-relative logical processor number (== affinity bit index).
    pub lpi: u8,
    /// Physical core index within the group (shared by SMT siblings).
    pub core_index: u8,
    pub llc_index: u8,
    pub numa: u8,
    /// Raw `EfficiencyClass` byte. Its polarity is vendor- and
    /// silicon-dependent: on this Raptor Lake ES the E-cores report 0 and the
    /// P-cores report 1, i.e. the opposite of what the name suggests. Reported
    /// for the record; classification uses SMT sharing instead.
    pub efficiency: u8,
    /// True when Windows has parked this logical processor. A parked LP still
    /// accepts a pinned thread, but it is worth showing in the topology table
    /// because it explains an unexpectedly low clock.
    pub parked: bool,
}

/// All CPU sets this process is allowed to run on, in enumeration order.
pub fn enumerate() -> Vec<CpuSet> {
    unsafe {
        let mut needed: Ulong = 0;
        GetSystemCpuSetInformation(
            std::ptr::null_mut(),
            0,
            &mut needed,
            GetCurrentProcess(),
            0,
        );
        if needed == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; needed as usize];
        let mut again = needed;
        let ok = GetSystemCpuSetInformation(
            buf.as_mut_ptr(),
            needed,
            &mut again,
            GetCurrentProcess(),
            0,
        );
        if ok == 0 || again == 0 {
            return Vec::new();
        }
        let end = again as usize;
        let mut out = Vec::new();
        let mut off = 0usize;
        while off + 4 <= end {
            let raw = &*(buf.as_ptr().add(off) as *const RawCpuSet);
            let sz = raw.size as usize;
            if sz < 24 || off + sz > end {
                break;
            }
            out.push(CpuSet {
                id: raw.tag,
                group: raw.group,
                lpi: raw.logical_processor_index,
                core_index: raw.core_index,
                llc_index: raw.last_level_cache_index,
                numa: raw.numa_node_index,
                efficiency: raw.efficiency_class,
                parked: raw.parked != 0,
            });
            off += sz;
        }
        out
    }
}

/// Which logical processors are P-cores, indexed like the input slice.
///
/// On every shipping hybrid part only the P-cores have a second hyperthread,
/// so a `core_index` claimed by two logical processors marks a P-core and a
/// `core_index` claimed once marks an E-core. If no core index is shared at
/// all -- AMD, or a hybrid part booted with SMT off -- there is nothing to
/// separate and everything is treated as a P-core, which is the honest answer
/// for a non-hybrid machine.
pub fn classify_p_cores(sets: &[CpuSet]) -> Vec<bool> {
    use std::collections::HashMap;
    let mut shares: HashMap<u8, usize> = HashMap::new();
    for s in sets {
        *shares.entry(s.core_index).or_insert(0) += 1;
    }
    let any_smt = shares.values().any(|&n| n > 1);
    sets.iter()
        .map(|s| !any_smt || shares[&s.core_index] > 1)
        .collect()
}

/// How the worker thread ended up pinned.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PinMode {
    /// `SetThreadGroupAffinity` -- effect readable back via
    /// `GetCurrentProcessorNumberEx`, so the measurement is attributable.
    GroupAffinity,
    /// Legacy `SetThreadAffinityMask` fallback.
    AffinityMask,
    /// `SetThreadSelectedCpuSets`, used only past 64 logical processors.
    CpuSet,
    None,
}

/// Pin the *calling* thread to one logical processor. Returns the mode used.
pub fn pin_self(set: &CpuSet) -> PinMode {
    unsafe {
        if set.lpi < 64 {
            let ga = GroupAffinity {
                mask: 1usize << set.lpi,
                group: set.group,
                reserved: [0; 3],
            };
            let mut prev = GroupAffinity::default();
            if SetThreadGroupAffinity(GetCurrentThread(), &ga, &mut prev) != 0 {
                return PinMode::GroupAffinity;
            }
            if SetThreadAffinityMask(GetCurrentThread(), 1usize << set.lpi) != 0 {
                return PinMode::AffinityMask;
            }
        }
        // Past 64 logical processors a single 64-bit mask cannot express the
        // target, so fall back to CPU sets and accept that the processor number
        // read back may not identify the core.
        let id = set.id;
        if SetThreadSelectedCpuSets(GetCurrentThread(), &id, 1) != 0 {
            return PinMode::CpuSet;
        }
        PinMode::None
    }
}

/// `GetCurrentProcessorNumber` for the calling thread. On a pinned thread this
/// is the ground truth for "which core did this run on".
pub fn current_processor_number() -> u32 {
    unsafe {
        #[repr(C)]
        struct Pn {
            group: u16,
            number: u8,
            reserved: u8,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcessorNumberEx(proc: *mut Pn);
        }
        let mut pn = Pn {
            group: 0,
            number: 0,
            reserved: 0,
        };
        GetCurrentProcessorNumberEx(&mut pn);
        pn.number as u32
    }
}

pub fn raise_priority() {
    // THREAD_PRIORITY_HIGHEST: enough to stop the scheduler from parking us on
    // an idle core, not so much that we starve the desktop.
    unsafe {
        SetThreadPriority(GetCurrentThread(), 2);
    }
}

/// `__cpuid` executes on the *current* core, so calling this from a pinned
/// worker gives a genuinely per-core feature read.
pub mod cpuid {
    use std::arch::x86_64::__cpuid_count;

    pub fn brand_string() -> String {
        let mut words = [0u32; 12];
        for i in 0..3usize {
            let r = __cpuid_count(0x8000_0002u32 + i as u32, 0);
            words[i * 4] = r.eax;
            words[i * 4 + 1] = r.ebx;
            words[i * 4 + 2] = r.ecx;
            words[i * 4 + 3] = r.edx;
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        String::from_utf8_lossy(&bytes)
            .trim_end_matches('\0')
            .trim()
            .to_string()
    }

    pub fn family_model_stepping() -> (u32, u32, u32) {
        let r = __cpuid_count(1, 0);
        let stepping = r.eax & 0xf;
        let mut model = (r.eax >> 4) & 0xf;
        let mut family = (r.eax >> 8) & 0xf;
        if family == 0xf || family == 0x6 {
            model += ((r.eax >> 16) & 0xf) << 4;
        }
        if family == 0xf {
            family += (r.eax >> 20) & 0xff;
        }
        (family, model, stepping)
    }

    /// Initial APIC ID: the hardware's own name for the core we are sitting on.
    /// Two different logical processors sharing an APIC ID are SMT siblings of
    /// one physical core, which makes this an independent check on the pinning.
    pub fn apic_id() -> u32 {
(__cpuid_count(1, 0).ebx >> 24) & 0xff
    }

    /// Raw cross-check for `Features::detect()`: the CPUID words that carry the
    /// AVX-512 bits, plus XCR0 (whether the OS enabled the AVX-512 register
    /// state). `is_x86_feature_detected!` stays the authority — reading these
    /// bits by hand is exactly the mistake that once dropped AVX-VNNI from the
    /// report (EBX bit 4 is HLE, not VNNI) — but when the two disagree it is
    /// worth seeing why. Only bits whose meaning is unambiguous are labelled.
    pub fn feature_dump() -> String {
        let l7 = __cpuid_count(7, 0);
        let l1 = __cpuid_count(1, 0);
        let mut s = String::new();
        s.push_str(&format!(
            "  CPUID.(EAX=07H,ECX=0): EBX={:#010x}  ECX={:#010x}  EDX={:#010x}\n",
            l7.ebx, l7.ecx, l7.edx
        ));
        for (b, name) in [
            (5, "AVX2"),
            (16, "AVX512F"),
            (17, "AVX512DQ"),
            (21, "AVX512IFMA"),
            (30, "AVX512BW"),
            (31, "AVX512VL"),
        ] {
            s.push_str(&format!(
                "    EBX[{:<2}] {:<12} = {}\n",
                b,
                name,
                (l7.ebx >> b) & 1
            ));
        }
        s.push_str(&format!(
            "    ECX[11] {:<12} = {}\n",
            "AVX512VNNI",
            (l7.ecx >> 11) & 1
        ));
        let osxsave = (l1.ecx >> 27) & 1 == 1;
        s.push_str(&format!("  CPUID.(EAX=01H):  OSXSAVE = {}\n", osxsave as u8));
        if osxsave {
            let xcr0 = unsafe { core::arch::x86_64::_xgetbv(0) };
            s.push_str(&format!(
                "  XCR0 = {:#x}  (opmask={} ZMM_Hi256={} Hi16_ZMM={})\n",
                xcr0,
                (xcr0 >> 5) & 1,
                (xcr0 >> 6) & 1,
                (xcr0 >> 7) & 1
            ));
        }
        s
    }

}

/// Invariant TSC rate in GHz, measured against the wall clock.
///
/// RDTSC ticks at a fixed rate regardless of core frequency, so this is *not*
/// the core clock and is not used to derive any core-cycle number in this
/// program -- every timing goes through `Instant`. On the machine this was
/// developed on the TSC reads ~1.80 GHz and stays there under load, while the
/// P-cores boost past 4.9 GHz; that is a property of the silicon's TSC
/// reference, not of this measurement, which is why the value is only reported
/// and never trusted.
pub fn tsc_hz() -> f64 {
    use std::time::Instant;
    let mut best = 0f64;
    for _ in 0..3 {
        let t0 = Instant::now();
        let c0 = unsafe { core::arch::x86_64::_rdtsc() };
        let mut spins = 0u64;
        while t0.elapsed().as_secs_f64() < 0.05 {
            std::hint::spin_loop();
            spins += 1;
        }
        let c1 = unsafe { core::arch::x86_64::_rdtsc() };
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(spins);
        // Largest of three: the shortest window is the one least likely to
        // have been stretched by a scheduling hiccup.
        let hz = (c1 - c0) as f64 / dt;
        if hz > best {
            best = hz;
        }
    }
    best / 1e9
}
