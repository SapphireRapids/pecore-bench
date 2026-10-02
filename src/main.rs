//! pecore-bench -- measure P-cores and E-cores separately on a Windows 11
//! hybrid CPU: integer / scalar FP / AVX2 / AVX-VNNI throughput and latency,
//! plus cache latency and bandwidth curves with the capacity at which each
//! level stops fitting.

mod bench;
mod cache;
mod kern;
mod report;
mod sys;

use std::path::PathBuf;

struct Args {
    quick: bool,
    only_lpi: Option<Vec<u8>>,
    class: Option<char>,
    list: bool,
    features: bool,
    ask: bool,
    yes: bool,
    open: Option<bool>,
    out: Option<PathBuf>,
    cache_max_mib: usize,
    no_cache: bool,
}

fn usage() -> &'static str {
    "pecore-bench [选项]\n\
     \n\
     直接双击运行（不带参数）会先显示 CPU 与指令集探测，再询问是否开始测试；\n\
     测完自动用默认浏览器打开报告。\n\
     \n\
     \x20   --quick, -q        快速模式，采样轮数减半（精度略降）\n\
     \x20   --core, -c <lpi,…> 只测指定逻辑处理器，如 --core 0,12\n\
     \x20   --class <p|e>      只测 P 核或只测 E 核\n\
     \x20   --list, -l         只列拓扑，不测量\n\
     \x20   --features         只打印指令集探测（含 CPUID 原始位）后退出\n\
     \x20   --yes, -y          不询问，直接开始（脚本用）\n\
     \x20   --ask              即使标准输入不是控制台也询问\n\
     \x20   --open             测完用默认浏览器打开报告\n\
     \x20   --no-open          测完不打开浏览器\n\
     \x20   --cache-max <MiB>  缓存曲线最大工作集，默认 64\n\
     \x20   --no-cache         跳过缓存曲线，只测计算内核\n\
     \x20   --out, -o <目录>   报告输出目录，默认 <exe 所在目录>\\pecore-report\n\
     \x20   --help, -h         显示本帮助"
}

/// Where the report goes when `--out` was not given: next to the executable,
/// not next to the current directory. A double-clicked exe and a shortcut with
/// a foreign working directory then behave identically.
fn default_out_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("pecore-report")))
        .unwrap_or_else(|| PathBuf::from("pecore-report"))
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        quick: false,
        only_lpi: None,
        class: None,
        list: false,
        features: false,
        ask: false,
        yes: false,
        open: None,
        out: None,
        cache_max_mib: 64,
        no_cache: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--quick" | "-q" => a.quick = true,
            "--list" | "-l" => a.list = true,
            "--features" => a.features = true,
            "--ask" => a.ask = true,
            "--yes" | "-y" => a.yes = true,
            "--open" => a.open = Some(true),
            "--no-open" => a.open = Some(false),
            "--no-cache" => a.no_cache = true,
            "--help" | "-h" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            "--core" | "-c" => {
                let v = it.next().ok_or("--core 需要一个参数")?;
                let mut lpis = Vec::new();
                for part in v.split(',') {
                    let n: u8 = part
                        .trim()
                        .parse()
                        .map_err(|_| format!("无法解析逻辑处理器号: {}", part))?;
                    lpis.push(n);
                }
                a.only_lpi = Some(lpis);
            }
            "--class" => {
                let v = it.next().ok_or("--class 需要一个参数")?;
                let c = v.chars().next().ok_or("--class 需要一个参数")?;
                match c.to_ascii_lowercase() {
                    'p' | 'e' => a.class = Some(c.to_ascii_lowercase()),
                    _ => return Err("--class 只接受 p 或 e".into()),
                }
            }
            "--cache-max" => {
                let v = it.next().ok_or("--cache-max 需要一个参数")?;
                a.cache_max_mib = v
                    .trim()
                    .parse()
                    .map_err(|_| format!("无法解析容量: {}", v))?;
            }
            "--out" | "-o" => {
                a.out = Some(PathBuf::from(it.next().ok_or("--out 需要一个参数")?));
            }
            other => return Err(format!("未知参数: {}（用 --help 看用法）", other)),
        }
    }
    Ok(a)
}

/// `--features` / menu item 4: what this CPU supports and which kernels that
/// means. The std detector is the verdict; the raw CPUID dump is printed only
/// so a disagreement with it can be diagnosed rather than puzzled over.
fn print_features() {
    let f = kern::Features::detect();
    let brand = sys::cpuid::brand_string();
    let (family, model, stepping) = sys::cpuid::family_model_stepping();
    println!(
        "CPU: {}  (family {} model {} stepping {})",
        brand, family, model, stepping
    );
    println!("指令集（std::arch::is_x86_feature_detected!，权威判定）:");
    for (n, v) in [
        ("AVX", f.avx),
        ("FMA", f.fma),
        ("AVX2", f.avx2),
        ("AVX-VNNI", f.avxvnni),
        ("AVX-512F", f.avx512f),
        ("AVX-512BW", f.avx512bw),
        ("AVX-512DQ", f.avx512dq),
        ("AVX-512VL", f.avx512vl),
        ("AVX-512VNNI", f.avx512vnni),
    ] {
        println!("  {:<12} {}", n, if v { "有" } else { "无" });
    }
    println!();
    println!("CPUID 原始位（交叉核对用）:");
    print!("{}", sys::cpuid::feature_dump());
    println!();
    let tbl = kern::table();
    let mut skip = Vec::new();
    for k in &tbl {
        if !f.has(k.feat) {
            skip.push(k.name);
        }
    }
    println!("内核表共 {} 条，本机将运行 {} 条。", tbl.len(), tbl.len() - skip.len());
    if !skip.is_empty() {
        println!("因 CPU 不支持而跳过 {} 条：", skip.len());
        for n in skip {
            println!("  - {}", n);
        }
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("错误: {}", e);
            eprintln!("{}", usage());
            std::process::exit(2);
        }
    };
    if args.features {
        print_features();
        return;
    }
    let any_args = std::env::args().len() > 1;

    let sets = sys::enumerate();
    if sets.is_empty() {
        eprintln!("枚举不到任何 CPU set，无法继续。");
        if sys::stdin_is_console() {
            sys::pause_for_enter("按回车键退出...");
        }
        std::process::exit(1);
    }
    let is_p = sys::classify_p_cores(&sets);
    let mut order: Vec<usize> = (0..sets.len()).collect();
    order.sort_by_key(|&i| sets[i].lpi);

    let p_count = order.iter().filter(|&&i| is_p[i]).count();
    let e_count = order.len() - p_count;
    let brand = sys::cpuid::brand_string();
    let (family, model, stepping) = sys::cpuid::family_model_stepping();
    let tsc = sys::tsc_hz();
    let features = kern::Features::detect();

    println!("pecore-bench — P 核 / E 核分核性能测试");
    println!("CPU: {}", brand);
    println!(
        "     family {} model {} stepping {}   RDTSC 读数 {:.3} GHz（不变时钟，不参与计时）",
        family, model, stepping, tsc
    );
    println!("指令集: {}", features.summary());
    println!(
        "拓扑: {} 逻辑处理器 = {} P 核 + {} E 核",
        order.len(),
        p_count,
        e_count
    );
    println!();
    println!(
        "{:<5} {:<6} {:<8} {:<8} {:<7} {:<9} {}",
        "LPI", "类别", "物理核", "LLC", "Eclass", "SMT兄弟", "驻留"
    );
    for &i in &order {
        let s = &sets[i];
        let sibling = sets
            .iter()
            .enumerate()
            .find(|(j, o)| *j != i && o.core_index == s.core_index && o.group == s.group)
            .map(|(_, o)| o.lpi)
            .unwrap_or(255);
        println!(
            "{:<5} {:<6} {:<8} {:<8} {:<7} {:<9} {}",
            s.lpi,
            if is_p[i] { "P" } else { "E" },
            s.core_index,
            s.llc_index,
            s.efficiency,
            if sibling == 255 {
                "-".to_string()
            } else {
                format!("LP {}", sibling)
            },
            if s.parked { "已驻留" } else { "活动中" },
        );
    }

    // Interactive when the program was started by double-click (no arguments,
    // stdin is a real console) or when --ask forces it. Every scripted run
    // keeps its old behaviour: no prompt, no browser.
    let interactive = !args.yes && (args.ask || (!any_args && sys::stdin_is_console()));
    let mut quick = args.quick;
    let mut list_only = args.list;
    if interactive {
        println!();
        println!("请选择：");
        println!("  1) 开始完整测试（20 核约 80–120 秒，期间请勿干扰机器）");
        println!("  2) 开始快速测试（采样减半，精度略降）");
        println!("  3) 只列拓扑，不测试");
        println!("  4) 只看指令集探测");
        println!("  q) 退出");
        print!("输入编号后回车 [1]: ");
        use std::io::Write;
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        match line.trim() {
            "2" => quick = true,
            "3" => list_only = true,
            "4" => {
                println!();
                print_features();
                sys::pause_for_enter("\n按回车键退出...");
                return;
            }
            "q" | "Q" => return,
            _ => {}
        }
        println!();
    }

    if list_only {
        if interactive {
            sys::pause_for_enter("按回车键退出...");
        }
        return;
    }

    let do_open = args.open.unwrap_or(interactive);

    // Which logical processors to measure.
    let selected: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&i| {
            if let Some(lpis) = &args.only_lpi {
                lpis.contains(&sets[i].lpi)
            } else if let Some(c) = args.class {
                (c == 'p') == is_p[i]
            } else {
                true
            }
        })
        .collect();
    if selected.is_empty() {
        eprintln!("筛选条件没有匹配到任何逻辑处理器。");
        std::process::exit(1);
    }

    let sizes: Vec<usize> = if args.no_cache {        Vec::new()
    } else {
        cache::SIZES
            .iter()
            .copied()
            .filter(|&b| b <= args.cache_max_mib << 20)
            .collect()
    };
    let caches = if sizes.is_empty() {
        cache::CacheBufs {
            sizes: Vec::new(),
            bufs: Vec::new(),
        }
    } else {
        let t0 = std::time::Instant::now();
        let b = cache::build(&sizes);
        println!(
            "\n缓存测试缓冲 {:.0} MiB，建置换表 {:.1}s",
            sizes.iter().sum::<usize>() as f64 / (1 << 20) as f64,
            t0.elapsed().as_secs_f64()
        );
        b
    };

    let mut results = Vec::new();
    let total = selected.len();
    for (n, &i) in selected.iter().enumerate() {
        let set = sets[i].clone();
        let is_p_core = is_p[i];
        print!(
            "\r测量 {}/{}  LP {:<3} ({}) ...",
            n + 1,
            total,
            set.lpi,
            if is_p_core { "P" } else { "E" }
        );
        use std::io::Write;
        std::io::stdout().flush().ok();
        let t0 = std::time::Instant::now();
        // A dedicated thread per core: run_core pins its caller and never
        // restores it. Scoped so the thread can borrow the shared cache
        // buffers, and joined before the next core starts so the cores cannot
        // perturb each other.
        let r = std::thread::scope(|s| {
            let h = std::thread::Builder::new()
                .stack_size(1 << 20)
                .spawn_scoped(s, || bench::run_core(&set, is_p_core, &caches, quick))
                .expect("无法创建工作线程");
            match h.join() {
                Ok(r) => r,
                Err(e) => std::panic::resume_unwind(e),
            }
        });
        let pin_note = if r.pin_verified() {
            "已确认".to_string()
        } else {
            format!("异常(ran_on {})", r.proc_no)
        };
        println!(
            "\r  LP {:<3} {} 核  {:.3} GHz  钉核{}  APIC {:<3} {:.1}s",
            r.lpi,
            r.class_name,
            r.freq_ghz,
            pin_note,
            r.apic_id,
            t0.elapsed().as_secs_f64()
        );
        results.push(r);
    }

    let info = report::MachineInfo {
        brand,
        family,
        model,
        stepping,
        logical: order.len(),
        p_cores: p_count,
        e_cores: e_count,
        tsc_ghz: tsc,
        quick,
    };

    report::print_summary(&results, &info);

    let out_dir = args.out.clone().unwrap_or_else(default_out_dir);
    std::fs::create_dir_all(&out_dir).ok();
    let csv = out_dir.join("pecore-bench.csv");
    let html = out_dir.join("pecore-bench.html");
    report::write_csv(&csv, &results).expect("写 CSV 失败");
    report::write_html(&html, &info, &results).expect("写 HTML 失败");
    println!("报告: {}", csv.display());
    println!("      {}", html.display());

    if do_open {
        if sys::open_in_default_browser(&html) {
            println!("\n已用默认浏览器打开报告。");
        } else {
            println!("\n（自动打开浏览器失败，请手动打开上面的 HTML 文件）");
        }
    }
    if interactive {
        sys::pause_for_enter("\n按回车键退出...");
    }
}
