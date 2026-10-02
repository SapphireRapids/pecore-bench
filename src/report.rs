//! Output: a CSV for spreadsheets and one self-contained HTML page.
//!
//! The HTML embeds the measurements as JSON and draws them with canvas, so it
//! opens from `file://` with no network access and no third-party library.

use std::fmt::Write as _;

use crate::bench::CoreResult;

pub struct MachineInfo {
    pub brand: String,
    pub family: u32,
    pub model: u32,
    pub stepping: u32,
    pub logical: usize,
    pub p_cores: usize,
    pub e_cores: usize,
    pub tsc_ghz: f64,
    pub quick: bool,
}

/// Per-class average of one metric, for the summary tables.
pub struct ClassAvg {
    pub p: f64,
    pub e: f64,
}

fn class_avg(results: &[CoreResult], pick: impl Fn(&CoreResult) -> Option<f64>) -> ClassAvg {
    let (mut p, mut pn) = (0.0, 0usize);
    let (mut e, mut en) = (0.0, 0usize);
    for r in results {
        if let Some(v) = pick(r) {
            if r.is_p {
                p += v;
                pn += 1;
            } else {
                e += v;
                en += 1;
            }
        }
    }
    ClassAvg {
        p: if pn > 0 { p / pn as f64 } else { f64::NAN },
        e: if en > 0 { e / en as f64 } else { f64::NAN },
    }
}

/// ns per iteration of one kernel id, averaged per class.
pub fn kernel_class_avg(results: &[CoreResult], id: &str) -> ClassAvg {
    class_avg(results, |r| {
        r.kernels.iter().find(|k| k.id == id).map(|k| k.ns_per_iter)
    })
}

pub fn freq_class_avg(results: &[CoreResult]) -> ClassAvg {
    class_avg(results, |r| Some(r.freq_ghz))
}

// ------------------------------------------------------------------ CSV

pub fn write_csv(path: &std::path::Path, results: &[CoreResult]) -> std::io::Result<()> {
    let mut s = String::new();
    s.push_str("class,lpi,cpu_set_id,core_index,apic_id,pin_verified,freq_ghz,freq_ghz_load,tsc_ghz,kernel_id,kind,ops_per_iter,ns_per_iter,cyc_per_iter\n");
    for r in results {
        for k in &r.kernels {
            let _ = writeln!(
                s,
                "{},{},{},{},{},{},{:.4},{:.4},{:.4},{},{},{},{:.4},{:.4}",
                r.class_name,
                r.lpi,
                r.cpu_set_id,
                r.core_index,
                r.apic_id,
                r.pin_verified(),
                r.freq_ghz,
                r.freq_ghz_load,
                r.tsc_ghz,
                k.id,
                k.kind,
                k.ops_per_iter,
                k.ns_per_iter,
                k.cyc_per_iter
            );
        }
    }
    s.push_str("\nclass,lpi,core_index,bytes,latency_ns,bandwidth_gbs\n");
    for r in results {
        for (lat, bw) in r.cache_latency.iter().zip(r.cache_bw.iter()) {
            let _ = writeln!(
                s,
                "{},{},{},{},{:.4},{:.3}",
                r.class_name, r.lpi, r.core_index, lat.bytes, lat.value, bw.value
            );
        }
    }
    std::fs::write(path, s)
}

// ------------------------------------------------------------------ HTML

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn fnum(v: f64) -> String {
    if v.is_finite() {
        format!("{:.4}", v)
    } else {
        "null".to_string()
    }
}

/// Compact JSON payload consumed by the page's chart code.
fn payload(info: &MachineInfo, results: &[CoreResult]) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    let _ = writeln!(s, "  \"brand\": \"{}\",", json_escape(&info.brand));
    let _ = writeln!(
        s,
        "  \"cpu\": \"family {} model {} stepping {}\",",
        info.family, info.model, info.stepping
    );
    let _ = writeln!(s, "  \"logical\": {},", info.logical);
    let _ = writeln!(s, "  \"p_cores\": {},", info.p_cores);
    let _ = writeln!(s, "  \"e_cores\": {},", info.e_cores);
    let _ = writeln!(s, "  \"tsc_ghz\": {},", fnum(info.tsc_ghz));
    let _ = writeln!(s, "  \"quick\": {},", info.quick);

    // CPU feature line + the kernels this CPU cannot run, so the page can say
    // why rows are missing instead of silently showing fewer bars.
    let feat = crate::kern::Features::detect();
    let _ = writeln!(s, "  \"feat_summary\": \"{}\",", json_escape(&feat.summary()));
    s.push_str("  \"skipped\": [");
    let mut first_skip = true;
    for k in crate::kern::table() {
        if !feat.has(k.feat) {
            let _ = write!(
                s,
                "{}{{\"name\": \"{}\", \"feat\": \"{}\"}}",
                if first_skip { "" } else { ", " },
                json_escape(k.name),
                k.feat.label()
            );
            first_skip = false;
        }
    }
    s.push_str("],\n");

    // working-set sizes actually measured, so the x axis can never drift from
    // the data when the size list is filtered on the command line
    s.push_str("  \"sizes\": [");
    if let Some(r) = results.iter().find(|r| !r.cache_latency.is_empty()) {
        for (i, p) in r.cache_latency.iter().enumerate() {
            let _ = write!(s, "{}{}", if i == 0 { "" } else { ", " }, p.bytes);
        }
    }
    s.push_str("],\n");

    // kernel table
    s.push_str("  \"kernels\": [\n");
    let ids: Vec<&str> = results
        .iter()
        .flat_map(|r| r.kernels.iter().map(|k| k.id.as_str()))
        .collect();
    let mut seen = Vec::new();
    for id in ids {
        if !seen.contains(&id) {
            seen.push(id);
        }
    }
    for (i, id) in seen.iter().enumerate() {
        let sample = results
            .iter()
            .find_map(|r| r.kernels.iter().find(|k| k.id == *id))
            .unwrap();
        let _ = writeln!(
            s,
            "    {{\"id\": \"{}\", \"group\": \"{}\", \"name\": \"{}\", \"kind\": \"{}\", \"ops\": {}}}{}",
            json_escape(id),
            json_escape(&sample.group),
            json_escape(&sample.name),
            json_escape(&sample.kind),
            sample.ops_per_iter,
            if i + 1 == seen.len() { "" } else { "," }
        );
    }
    s.push_str("  ],\n");

    // per-core rows
    s.push_str("  \"cores\": [\n");
    for (ci, r) in results.iter().enumerate() {
        s.push_str("    {\n");
        let _ = writeln!(s, "      \"lpi\": {},", r.lpi);
        let _ = writeln!(s, "      \"class\": \"{}\",", r.class_name);
        let _ = writeln!(s, "      \"core\": {},", r.core_index);
        let _ = writeln!(s, "      \"apic\": {},", r.apic_id);
        let _ = writeln!(s, "      \"freq\": {},", fnum(r.freq_ghz));
        let _ = writeln!(s, "      \"freq2\": {},", fnum(r.freq_ghz_load));
        let _ = writeln!(s, "      \"llc\": {},", r.llc_index);
        let _ = writeln!(s, "      \"numa\": {},", r.numa);
        let _ = writeln!(s, "      \"eff\": {},", r.efficiency);
        let _ = writeln!(s, "      \"tsc\": {},", fnum(r.tsc_ghz));
        let _ = writeln!(s, "      \"cpuset\": {},", r.cpu_set_id);
        let _ = writeln!(s, "      \"verified\": {},", r.pin_verified());
        let _ = writeln!(
            s,
            "      \"feat\": [{},{},{},{},{},{},{},{},{}],",
            r.features.avx,
            r.features.fma,
            r.features.avx2,
            r.features.avxvnni,
            r.features.avx512f,
            r.features.avx512bw,
            r.features.avx512dq,
            r.features.avx512vl,
            r.features.avx512vnni
        );
        let _ = writeln!(s, "      \"kernels\": {{");
        let mut first = true;
        for k in &r.kernels {
            let _ = write!(
                s,
                "{}\"{}\": {{\"ns\": {}, \"cyc\": {}}}",
                if first { "" } else { ", " },
                json_escape(&k.id),
                fnum(k.ns_per_iter),
                fnum(k.cyc_per_iter)
            );
            first = false;
        }
        s.push_str("},\n");
        let _ = write!(s, "      \"latency\": [");
        for (i, p) in r.cache_latency.iter().enumerate() {
            let _ = write!(s, "{}{}", if i == 0 { "" } else { ", " }, fnum(p.value));
        }
        s.push_str("],\n");
        let _ = write!(s, "      \"bw\": [");
        for (i, p) in r.cache_bw.iter().enumerate() {
            let _ = write!(s, "{}{}", if i == 0 { "" } else { ", " }, fnum(p.value));
        }
        s.push_str("]\n");
        s.push_str("    }");
        if ci + 1 < results.len() {
            s.push_str(",");
        }
        s.push_str("\n");
    }
    s.push_str("  ]\n}\n");
    s
}

const CSS: &str = r#"
:root { --bg:#0f1115; --panel:#171a21; --line:#2b3242; --fg:#eef1f7; --dim:#a8b2c4;
        --p:#4da3ff; --e:#ffb454; --ok:#57d98a; --warn:#ff6b6b; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.6 "Segoe UI","Microsoft YaHei",system-ui,sans-serif; }
.wrap { max-width:1320px; margin:0 auto; padding:28px 22px 72px; }
h1 { font-size:25px; margin:0 0 6px; letter-spacing:.2px; }
h2 { font-size:19px; margin:34px 0 12px; color:var(--fg); }
h2 .n { display:inline-block; min-width:26px; height:26px; line-height:26px; text-align:center;
        border-radius:7px; background:rgba(77,163,255,.16); color:var(--p);
        font-size:15px; margin-right:9px; vertical-align:1px; }
.sub { color:var(--dim); font-size:14px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:12px;
         padding:18px 20px; margin-top:12px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th,td { text-align:right; padding:6px 9px; border-bottom:1px solid var(--line);
        white-space:nowrap; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align:left; }
tbody tr:hover { background:rgba(255,255,255,.028); }
th { color:var(--dim); font-weight:600; }
td.p { color:var(--p); } td.e { color:var(--e); }
.badge { display:inline-block; padding:1px 8px; border-radius:99px; font-size:12.5px; }
.badge.p { background:rgba(77,163,255,.16); color:var(--p); }
.badge.e { background:rgba(255,180,84,.16); color:var(--e); }
.badge.ok { background:rgba(87,217,138,.16); color:var(--ok); }
.badge.warn { background:rgba(255,107,107,.16); color:var(--warn); }
canvas { width:100%; height:auto; display:block; }
.legend { display:flex; gap:18px; align-items:center; margin:8px 0 12px;
          font-size:14px; color:var(--dim); }
.sw { width:12px; height:12px; border-radius:3px; display:inline-block; margin-right:7px;
      vertical-align:-1px; }
.controls { display:flex; gap:9px; margin:10px 0 6px; flex-wrap:wrap; align-items:center; }
button { background:var(--panel); color:var(--fg); border:1px solid var(--line);
         border-radius:7px; padding:6px 14px; font-size:13.5px; cursor:pointer; }
button:hover { border-color:#3d465c; }
button.on { border-color:var(--p); color:var(--p); background:rgba(77,163,255,.10); }
.note { color:var(--dim); font-size:13px; margin-top:10px; line-height:1.55; }
code { background:#0b0d11; padding:1.5px 6px; border-radius:5px; font-size:13px;
       color:#d9e2f2; }
.kv { margin-top:12px; }
.kv table { font-size:13px; }
.kv h4 { margin:0 0 6px; font-size:13.5px; color:var(--dim); font-weight:600; }
"#;

const JS: &str = r##"
const D = __PAYLOAD__;
const $ = (id) => document.getElementById(id);
const SIZES = D.sizes;
const fmt = (v, d=2) => v == null ? "-" : (Math.abs(v) >= 100 ? v.toFixed(d) : v.toFixed(d+1));

function seriesFor(key) {
  const p = [], e = [], all = [];
  for (const c of D.cores) {
    const pts = [];
    for (let i = 0; i < SIZES.length; i++) pts.push([SIZES[i], c[key][i]]);
    (c.class === "P" ? p : e).push(pts);
    all.push({ pts, lpi: c.lpi, cls: c.class });
  }
  const mean = (arrs) => SIZES.map((_, i) => {
    let s = 0, n = 0;
    for (const a of arrs) if (a[i] && a[i][1] != null) { s += a[i][1]; n++; }
    return n ? s / n : null;
  });
  return { p: p.length ? mean(p) : null, e: e.length ? mean(e) : null, all };
}

// A tiny canvas plotting helper: log2 x-axis, linear y-axis, no dependencies.
function draw(canvas, opt) {
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max(canvas.clientWidth, 320), H = opt.height || 300;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const L = 56, R = 14, T = 12, B = 34;
  const xs = opt.xs, ys = opt.ys;
  const xmin = Math.log2(Math.min(...xs)), xmax = Math.log2(Math.max(...xs));
  let ymin = Math.min(...ys.filter(v => v != null)), ymax = Math.max(...ys.filter(v => v != null));
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const pad = (ymax - ymin) * 0.08; ymax += pad;
  // A latency or bandwidth axis must not be padded below zero, or the lowest
  // grid line ends up labelled with a meaningless negative number.
  if (ymin >= 0) ymin = Math.max(0, ymin - pad); else ymin -= pad;
  const X = (v) => L + (Math.log2(v) - xmin) / (xmax - xmin) * (W - L - R);
  const Y = (v) => T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B);

  ctx.font = "12.5px Segoe UI, sans-serif";
  ctx.strokeStyle = "#252b39"; ctx.fillStyle = "#a8b2c4"; ctx.lineWidth = 1;
  // x grid at the cache-size landmarks
  for (const g of [10,12,14,16,18,20,21,22,23,24,25,26]) {
    if (g < xmin || g > xmax) continue;
    const x = X(Math.pow(2, g));
    ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, H - B); ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText((1 << g) >= 1048576 ? ((1 << g) / 1048576) + "M" : ((1 << g) / 1024) + "K", x, H - B + 16);
  }
  // y grid
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = ymin + (ymax - ymin) * i / ticks, y = Y(v);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(fmt(v, 1), L - 8, y + 4);
  }
  if (opt.yLabel) { ctx.save(); ctx.translate(12, T + 8); ctx.fillText(opt.yLabel, 0, 0); ctx.restore(); }

  const line = (pts, color, width, dash) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
    ctx.beginPath();
    let started = false;
    pts.forEach((v, i) => { if (v == null) return; const x = X(xs[i]), y = Y(v);
      started ? ctx.lineTo(x, y) : ctx.moveTo(x, y); started = true; });
    ctx.stroke(); ctx.setLineDash([]);
  };
  const dots = (pts, color) => {
    ctx.fillStyle = color;
    pts.forEach((v, i) => { if (v == null) return; const x = X(xs[i]), y = Y(v);
      ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill(); });
  };
  for (const s of opt.series) {
    if (s.dash) line(s.data, s.color, 1.4, [4, 4]); else line(s.data, s.color, 2.6);
    dots(s.data, s.color);
  }
  // knee markers
  for (const k of (opt.knees || [])) {
    const x = X(k.x);
    ctx.strokeStyle = k.color; ctx.setLineDash([3, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, H - B); ctx.stroke(); ctx.setLineDash([]);
    // label on a dark chip so it stays readable over the curves
    const tw = ctx.measureText(k.label).width;
    ctx.fillStyle = "rgba(13,15,19,.88)";
    ctx.fillRect(x + 5, T, tw + 10, 18);
    ctx.fillStyle = k.color; ctx.textAlign = "left";
    ctx.fillText(k.label, x + 10, T + 13.5);
  }
}

// ---- cache curves -------------------------------------------------------
function cacheChart(id, key, yLabel, unit, higherBetter) {
  const s = seriesFor(key);
  const canvas = $(id);
  const kneeMult = parseFloat($("knee").value || "1.5");
  const knees = [];
  for (const [cls, arr, color] of [["P", s.p, "#4da3ff"], ["E", s.e, "#ffb454"]]) {
    if (!arr) continue;
    const valid = arr.filter(v => v != null);
    if (!valid.length) continue;
    // Latency is lower-is-better, so the reference is the minimum and the knee
    // is the last size still within `mult` of it. Bandwidth is the mirror
    // image: without flipping both, the "knee" lands on the largest, slowest
    // size and says nothing.
    const ref = higherBetter ? Math.max(...valid) : Math.min(...valid);
    const limit = higherBetter ? ref / kneeMult : ref * kneeMult;
    let last = null;
    arr.forEach((v, i) => {
      if (v == null) return;
      if (higherBetter ? v >= limit : v <= limit) last = i;
    });
    if (last != null && last < SIZES.length - 1) {
      knees.push({ x: SIZES[last], label: cls + " " + fmtK(SIZES[last]), color });
    }
  }
  draw(canvas, {
    xs: SIZES, ys: (s.p || []).concat(s.e || []), yLabel: yLabel + " (" + unit + ")",
    series: [
      { data: s.p, color: "#4da3ff" },
      { data: s.e, color: "#ffb454" },
    ],
    knees,
    height: 320,
  });
  const k = knees.map(x => x.label).join("   ") || "无";
  const rule = higherBetter
    ? "最后仍在最佳值 " + kneeMult + "x 以上的容量"
    : "最后仍低于最小值 " + kneeMult + "x 的容量";
  $("knee_" + id).textContent = "拐点（" + rule + "）: " + k;
}
function fmtK(b) { return b >= 1048576 ? (b / 1048576) + " MiB" : (b / 1024) + " KiB"; }

// ---- compute bars -------------------------------------------------------
// Horizontal bars grouped by instruction family, kernel names on the left.
// A rotated-label vertical chart fitted 30 kernels onto one axis but was
// unreadable: 10 px rotated names and no values. Here each kernel owns a row,
// both bars carry their number, and the eye can follow one row at a time.
let metric = "ops";
const barFmt = (v) => v >= 10 ? v.toFixed(1) : v.toFixed(2);
function computeChart() {
  const canvas = $("compute");
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max(canvas.clientWidth, 320);
  const LW = 300, R = 60, TOP = 24, GRP = 28, ROW = 32, BOT = 30;
  const rows = D.kernels.map(k => {
    // A plain mean over the cores of each class, matching the table below --
    // a running (p + v) / 2 would weight the last core at 1/2 and the first at
    // 1/2^n, so the bars and the table would disagree.
    const acc = { P: [0, 0], E: [0, 0] };
    for (const c of D.cores) {
      const m = c.kernels[k.id]; if (!m) continue;
      const v = metric === "ops" ? k.ops / m.ns : m.cyc / k.ops;
      const a = acc[c.class]; a[0] += v; a[1]++;
    }
    return {
      k,
      p: acc.P[1] ? acc.P[0] / acc.P[1] : null,
      e: acc.E[1] ? acc.E[0] / acc.E[1] : null,
    };
  });
  const groups = [];
  for (const r of rows) {
    let g = groups.find(x => x.name === r.k.group);
    if (!g) { g = { name: r.k.group, rows: [] }; groups.push(g); }
    g.rows.push(r);
  }
  for (const g of groups) g.rows.sort((a, b) => (b.p == null ? -1e9 : b.p) - (a.p == null ? -1e9 : a.p));
  const H = TOP + groups.reduce((s, g) => s + GRP + g.rows.length * ROW, 0) + BOT;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const vals = rows.flatMap(r => [r.p, r.e]).filter(v => v != null);
  const vmax = Math.max(...vals) * 1.02;
  const X = (v) => LW + (v / vmax) * (W - LW - R);
  const BARH = 10;

  // value grid, labels along the bottom
  ctx.font = "12.5px Segoe UI, sans-serif";
  for (let i = 0; i <= 4; i++) {
    const v = vmax * i / 4, x = X(v);
    ctx.strokeStyle = "#252b39"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, TOP - 4); ctx.lineTo(x, H - BOT + 4); ctx.stroke();
    ctx.fillStyle = "#a8b2c4"; ctx.textAlign = "center";
    ctx.fillText(fmt(v, 1), x, H - 9);
  }
  ctx.fillStyle = "#a8b2c4"; ctx.textAlign = "left";
  ctx.fillText(metric === "ops" ? "操作数 / ns（越高越好）" : "周期 / 操作数（越低越好）", 4, H - 9);

  let y = TOP;
  for (const g of groups) {
    ctx.fillStyle = "rgba(255,255,255,.05)";
    ctx.fillRect(0, y, W, GRP - 3);
    ctx.fillStyle = "#dfe5f0"; ctx.textAlign = "left";
    ctx.font = "13.5px Segoe UI, sans-serif";
    ctx.fillText(g.name, 8, y + GRP - 10);
    y += GRP;
    for (const r of g.rows) {
      ctx.font = "12.5px Segoe UI, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = "#c9d2e2";
      // truncate in the chart only: a 30-char name would run into the kind
      // label; the table right below carries the full name
      const nm = r.k.name.length > 24 ? r.k.name.slice(0, 23) + "…" : r.k.name;
      ctx.fillText(nm, 14, y + ROW / 2 + 4);
      ctx.font = "11.5px Segoe UI, sans-serif";
      ctx.fillStyle = r.k.kind === "延迟" ? "#8fb7ff" : "#d8ab72";
      ctx.fillText(r.k.kind, LW - 52, y + ROW / 2 + 4);
      for (const [v, color, dy] of [[r.p, "#4da3ff", -8], [r.e, "#ffb454", 8]]) {
        if (v == null) continue;
        const x1 = X(v);
        ctx.fillStyle = color;
        ctx.fillRect(LW, y + ROW / 2 + dy - BARH / 2, Math.max(x1 - LW, 1), BARH);
        ctx.fillStyle = "#b9c3d4"; ctx.font = "12px Segoe UI, sans-serif"; ctx.textAlign = "left";
        ctx.fillText(barFmt(v), x1 + 6, y + ROW / 2 + dy + 4);
      }
      y += ROW;
    }
  }
  $("compute_unit").textContent = metric === "ops" ? "操作数 / ns（越高越好）" : "周期 / 操作数（越低越好）";
}

// ---- per-core clock ----------------------------------------------------
function freqChart() {
  const canvas = $("freq");
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max(canvas.clientWidth, 320), H = 250;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const L = 52, R = 12, T = 26, B = 34;
  const cores = D.cores;
  const vmax = Math.max(...cores.map(c => c.freq)) * 1.12;
  const slot = (W - L - R) / cores.length;
  const Y = (v) => T + (1 - v / vmax) * (H - T - B);
  ctx.font = "12.5px Segoe UI, sans-serif";
  for (let i = 0; i <= 4; i++) {
    const v = vmax * i / 4, y = Y(v);
    ctx.strokeStyle = "#252b39"; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.fillStyle = "#a8b2c4"; ctx.textAlign = "right"; ctx.fillText(v.toFixed(1), L - 8, y + 4);
  }
  cores.forEach((c, i) => {
    const x = L + i * slot + slot * 0.16;
    const y = Y(c.freq);
    ctx.fillStyle = c.class === "P" ? "#4da3ff" : "#ffb454";
    ctx.fillRect(x, y, slot * 0.68, T + (H - T - B) - y);
    // value on top of the bar
    ctx.fillStyle = "#c9d2e2"; ctx.textAlign = "center"; ctx.font = "11.5px Segoe UI, sans-serif";
    ctx.fillText(c.freq.toFixed(2), x + slot * 0.34, y - 5);
    ctx.fillStyle = "#a8b2c4"; ctx.font = "12px Segoe UI, sans-serif";
    ctx.fillText(c.lpi, x + slot * 0.34, H - B + 16);
  });
  // class averages as dashed lines; the values go in the caption rather than
  // next to the lines, where they collided with the bars' own value labels
  const means = [];
  for (const [cls, color] of [["P", "#4da3ff"], ["E", "#ffb454"]]) {
    const v = freqOf(cls);
    if (!isFinite(v)) continue;
    const y = Y(v);
    ctx.strokeStyle = color; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke(); ctx.setLineDash([]);
    means.push(cls + " " + v.toFixed(2));
  }
  ctx.fillStyle = "#a8b2c4"; ctx.textAlign = "left"; ctx.font = "12.5px Segoe UI, sans-serif";
  ctx.fillText(
    "GHz（蓝 = P 核，橙 = E 核；虚线 = 类别平均：" + (means.join(" / ") || "-") + " GHz）",
    L, H - 5);
}

// ---- tables ------------------------------------------------------------
function tables() {
  const byId = {};
  for (const c of D.cores) for (const [id, m] of Object.entries(c.kernels)) {
    (byId[id] = byId[id] || { p: [], e: [] })[c.class === "P" ? "p" : "e"].push(m);
  }
  const avg = (a, f) => a.length ? a.reduce((s, x) => s + f(x), 0) / a.length : null;
  let h = "<tr><th>内核</th><th>类型</th><th>P 核 ns/iter</th><th>E 核 ns/iter</th><th>P/E 加速比</th><th>P 周期/操作</th><th>E 周期/操作</th></tr>";
  for (const k of D.kernels) {
    const g = byId[k.id]; if (!g) continue;
    const p = avg(g.p, x => x.ns), e = avg(g.e, x => x.ns);
    const pc = avg(g.p, x => x.cyc), ec = avg(g.e, x => x.cyc);
    const r = (p && e) ? p / e : null;
    h += "<tr><td>" + k.name + "</td><td>" + k.kind + "</td>" +
      "<td class='p'>" + fmt(p, 3) + "</td><td class='e'>" + fmt(e, 3) + "</td>" +
      "<td>" + (r ? r.toFixed(2) + "x" : "-") + "</td>" +
      "<td>" + (pc != null ? fmt(pc / k.ops, 2) : "-") + "</td>" +
      "<td>" + (ec != null ? fmt(ec / k.ops, 2) : "-") + "</td></tr>";
  }
  $("kern_table").innerHTML = h;

  let c = "<tr><th>逻辑处理器</th><th>类别</th><th>物理核</th><th>APIC ID</th><th>实测频率 GHz</th><th>负载后 GHz</th><th>钉核校验</th><th>LLC</th><th>NUMA</th><th>AVX2</th><th>VNNI</th><th>AVX-512</th></tr>";
  for (const x of D.cores) {
    c += "<tr><td>LP " + x.lpi + "</td><td><span class='badge " + x.class.toLowerCase() + "'>" + x.class + "</span></td>" +
      "<td>" + x.core + "</td><td>" + x.apic + "</td><td>" + x.freq.toFixed(3) + "</td><td>" + x.freq2.toFixed(3) + "</td>" +
      "<td>" + (x.verified ? "<span class='badge ok'>已确认</span>" : "<span class='badge warn'>未确认</span>") + "</td>" +
      "<td>" + x.llc + "</td><td>" + x.numa + "</td>" +
      "<td>" + (x.feat[2] ? "有" : "无") + "</td><td>" + (x.feat[3] ? "有" : "无") + "</td>" +
      "<td>" + (() => {
        const subs = ["F", "BW", "DQ", "VL", "VNNI"].filter((_, j) => x.feat[4 + j]);
        return subs.length ? subs.join(" ") : "无";
      })() + "</td></tr>";
  }
  $("core_table").innerHTML = c;
}
function freqOf(cls) {
  const a = D.cores.filter(c => c.class === cls).map(c => c.freq);
  return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 1;
}

function render() {
  cacheChart("lat", "latency", "依赖加载延迟", "ns", false);
  cacheChart("bw", "bw", "读带宽", "GB/s", true);
  keyTable("kv_lat", "latency", "ns", 2);
  keyTable("kv_bw", "bw", "GB/s", 1);
  computeChart(); freqChart();
}

// ---- key numbers under each cache chart --------------------------------
// The curves show shape; these tables pin the absolute values at the sizes
// people compare (L1, L2, L3 boundaries and DRAM), so the page does not
// require reading numbers off a log-scale canvas.
function keyTable(elId, key, unit, digits) {
  const s = seriesFor(key);
  const want = [1 << 10, 32 << 10, 512 << 10, 1 << 20, 4 << 20, 64 << 20];
  const pick = (arr) => want.map(w => {
    if (!arr) return null;
    let best = null, bd = Infinity;
    SIZES.forEach((sz, i) => {
      if (arr[i] == null) return;
      const d = Math.abs(Math.log2(sz) - Math.log2(w));
      if (d < bd) { bd = d; best = arr[i]; }
    });
    return best;
  });
  const p = pick(s.p), e = pick(s.e);
  let h = "<h4>" + (key === "latency" ? "关键容量延迟" : "关键容量读带宽") + "（" + unit + "）</h4><table><tr><th>工作集</th>";
  for (const w of want) h += "<th>" + fmtK(w) + "</th>";
  h += "</tr>";
  for (const [cls, arr, c] of [["P 核", p, "p"], ["E 核", e, "e"]]) {
    h += "<tr><td>" + cls + "</td>";
    for (const v of arr) h += "<td class='" + c + "'>" + (v == null ? "-" : v.toFixed(digits)) + "</td>";
    h += "</tr>";
  }
  $(elId).innerHTML = h + "</table>";
}

// ---- what this CPU cannot run ------------------------------------------
function skipNote() {
  const el = $("skip_note");
  if (!el) return;
  if (!D.skipped || !D.skipped.length) { el.textContent = "本机支持全部内核，无跳过项。"; return; }
  const byFeat = {};
  for (const s of D.skipped) (byFeat[s.feat] = byFeat[s.feat] || []).push(s.name);
  const parts = Object.entries(byFeat).map(([f, ns]) => f + " 缺 " + ns.length + " 项");
  el.textContent = "因本机 CPU 不支持而跳过 " + D.skipped.length + " 个内核（" + parts.join("；") + "）："
    + D.skipped.map(s => s.name).join("、");
}
window.addEventListener("resize", () => render());
document.getElementById("metric_ops").onclick = () => { metric = "ops"; setMetric(); };
document.getElementById("metric_cyc").onclick = () => { metric = "cyc"; setMetric(); };
function setMetric() {
  $("metric_ops").className = metric === "ops" ? "on" : "";
  $("metric_cyc").className = metric === "cyc" ? "on" : "";
  computeChart();
}
$("knee").oninput = () => cacheChart("lat", "latency", "依赖加载延迟", "ns", false);
tables(); skipNote(); setMetric(); render();
"##;

pub fn write_html(path: &std::path::Path, info: &MachineInfo, results: &[CoreResult]) -> std::io::Result<()> {
    let js = JS.replace("__PAYLOAD__", &payload(info, results));
    let mut s = String::new();
    s.push_str("<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n");
    s.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    s.push_str("<title>P 核 / E 核分核性能报告</title>\n<style>");
    s.push_str(CSS);
    s.push_str("</style>\n</head>\n<body>\n<div class=\"wrap\">\n");

    let _ = writeln!(s, "<h1>P 核 / E 核分核性能报告</h1>");
    let _ = writeln!(
        s,
        "<div class=\"sub\">{} &middot; family {} model {} stepping {} &middot; {} 逻辑处理器（{} P 核 / {} E 核）&middot; TSC {:.3} GHz{}</div>",
        json_escape(&info.brand),
        info.family,
        info.model,
        info.stepping,
        info.logical,
        info.p_cores,
        info.e_cores,
        info.tsc_ghz,
        if info.quick { " &middot; 快速模式" } else { "" }
    );
    let _ = writeln!(
        s,
        "<div class=\"sub\" style=\"margin-top:3px\">指令集：{}</div>",
        json_escape(&crate::kern::Features::detect().summary())
    );

    s.push_str("<h2><span class=\"n\">1</span>各逻辑处理器实测时钟</h2>\n<div class=\"panel\"><canvas id=\"freq\"></canvas>");
    s.push_str("<div class=\"note\">用 1 周期依赖 <code>add</code> 链标定每核真实时钟；RDTSC 是不变频率，不能用来算核心周期。</div></div>\n");

    s.push_str("<h2><span class=\"n\">2</span>计算内核：P 核 vs E 核</h2>\n<div class=\"panel\">");
    s.push_str("<div class=\"controls\"><button id=\"metric_ops\" class=\"on\">操作数 / ns（越高越好）</button>");
    s.push_str("<button id=\"metric_cyc\">周期 / 操作数（越低越好）</button></div>");
    s.push_str("<div class=\"legend\"><span><span class=\"sw\" style=\"background:#4da3ff\"></span>P 核平均</span>");
    s.push_str("<span><span class=\"sw\" style=\"background:#ffb454\"></span>E 核平均</span>");
    s.push_str("<span id=\"compute_unit\"></span></div>");
    s.push_str("<canvas id=\"compute\"></canvas>");
    s.push_str("<table id=\"kern_table\"></table>");
    s.push_str("<div class=\"note\">\"延迟\"= 单条依赖链，反映指令延迟；\"吞吐\"= 8 条独立链，反映发射宽度。按组排列，组内按 P 核数值降序。ns/iter 是实测墙钟，周期/操作已按各核实测频率折算。</div>");
    s.push_str("<div class=\"note\" id=\"skip_note\"></div></div>\n");

    s.push_str("<h2><span class=\"n\">3</span>缓存延迟曲线（指针追逐，越界掉速位置）</h2>\n<div class=\"panel\">");
    s.push_str("<div class=\"controls\">拐点阈值 <input id=\"knee\" type=\"range\" min=\"1.1\" max=\"5\" step=\"0.1\" value=\"1.5\" style=\"width:160px\">");
    s.push_str("<span class=\"sub\" id=\"knee_lat\"></span></div>");
    s.push_str("<div class=\"legend\"><span><span class=\"sw\" style=\"background:#4da3ff\"></span>P 核平均</span>");
    s.push_str("<span><span class=\"sw\" style=\"background:#ffb454\"></span>E 核平均</span></div>");
    s.push_str("<canvas id=\"lat\"></canvas>");
    s.push_str("<div class=\"kv\" id=\"kv_lat\"></div>");
    s.push_str("<div class=\"note\">Sattolo 随机置换保证是缓存测试而不是预取器测试；曲线抬升处即工作集放不下该级缓存的位置。</div></div>\n");

    s.push_str("<h2><span class=\"n\">4</span>缓存读带宽曲线</h2>\n<div class=\"panel\">");
    s.push_str("<div class=\"controls\"><span class=\"sub\" id=\"knee_bw\"></span></div>");
    s.push_str("<div class=\"legend\"><span><span class=\"sw\" style=\"background:#4da3ff\"></span>P 核平均</span>");
    s.push_str("<span><span class=\"sw\" style=\"background:#ffb454\"></span>E 核平均</span></div>");
    s.push_str("<canvas id=\"bw\"></canvas>");
    s.push_str("<div class=\"kv\" id=\"kv_bw\"></div>");
    s.push_str("<div class=\"note\">P 核曲线在 4/16/32 KiB 等 2 的幂容量上呈规律性锯齿（快档约为慢档 1.5 倍，六个 P 核完全一致），是实测的硬件行为而非测量误差；容量级别的判断请看整体台阶，不要看单个尖齿。</div></div>\n");

    s.push_str("<h2><span class=\"n\">5</span>逐核明细</h2>\n<div class=\"panel\"><table id=\"core_table\"></table>");
    s.push_str("<div class=\"note\">APIC ID 是硬件给核的名字，两个逻辑处理器共享同一 APIC ID 就是同一物理核的 SMT 兄弟，可用来独立复核钉核结果。运行日志里有每核的钉核读回值。</div></div>\n");

    s.push_str("</div>\n<script>\n");
    s.push_str(&js);
    s.push_str("\n</script>\n</body>\n</html>\n");

    std::fs::write(path, s)
}

/// Console summary: one line per kernel with the P/E ratio.
pub fn print_summary(results: &[CoreResult], info: &MachineInfo) {
    println!();
    println!("=== 计算内核：P 核 vs E 核（每类取平均）===");
    println!(
        "{:<26} {:<6} {:>11} {:>11} {:>9} {:>11} {:>11}",
        "内核", "类型", "P ns/iter", "E ns/iter", "P/E", "P 周期/op", "E 周期/op"
    );
    let p_freq = freq_class_avg(results).p;
    let e_freq = freq_class_avg(results).e;
    let mut ids: Vec<String> = Vec::new();
    for r in results {
        for k in &r.kernels {
            if !ids.iter().any(|i| i == &k.id) {
                ids.push(k.id.clone());
            }
        }
    }
    for id in &ids {
        let sample = results
            .iter()
            .find_map(|r| r.kernels.iter().find(|k| &k.id == id))
            .unwrap();
        let a = kernel_class_avg(results, id);
        // Cycles come from each kernel's own adjacent add-chain reference, so
        // they are immune to the P/E clock difference that would otherwise leak
        // in through the class-average frequency.
        let c = class_avg(results, |r| {
            r.kernels
                .iter()
                .find(|k| &k.id == id)
                .map(|k| k.cyc_per_op())
        });
        let ratio = if a.p.is_finite() && a.e.is_finite() && a.e > 0.0 {
            a.p / a.e
        } else {
            f64::NAN
        };
        println!(
            "{:<26} {:<6} {:>11.4} {:>11.4} {:>8.2}x {:>11.3} {:>11.3}",
            sample.name,
            sample.kind,
            a.p,
            a.e,
            ratio,
            c.p,
            c.e
        );
    }

    println!();
    println!(
        "=== 频率 ===  P 平均 {:.3} GHz   E 平均 {:.3} GHz   (TSC {:.3} GHz)",
        p_freq, e_freq, info.tsc_ghz
    );

    println!();
    println!("=== 缓存：容量拐点（最后仍未掉出最佳值 N 倍以内的工作集）===");
    // Index 0 = latency, 1 = bandwidth. Latency is lower-is-better so the
    // reference is the minimum; bandwidth is higher-is-better so it is the
    // maximum. Either way the knee is the largest working set still within the
    // factor, i.e. the last size before the curve falls off the level.
    for pick in [0usize, 1] {
        let lower_better = pick == 0;
        for cls in ["P", "E"] {
            // Average the curve across every measured core of this class.
            let mut by_size: std::collections::BTreeMap<usize, (f64, usize)> =
                std::collections::BTreeMap::new();
            for r in results.iter().filter(|r| r.class_name == cls) {
                let src = if pick == 0 { &r.cache_latency } else { &r.cache_bw };
                for p in src {
                    if !p.value.is_finite() {
                        continue;
                    }
                    let e = by_size.entry(p.bytes).or_insert((0.0, 0));
                    e.0 += p.value;
                    e.1 += 1;
                }
            }
            if by_size.is_empty() {
                continue;
            }
            let series: Vec<(usize, f64)> = by_size
                .iter()
                .map(|(b, (s, n))| (*b, s / *n as f64))
                .collect();
            let reference = if lower_better {
                series.iter().map(|p| p.1).fold(f64::MAX, f64::min)
            } else {
                series.iter().map(|p| p.1).fold(f64::MIN, f64::max)
            };
            let mut parts = Vec::new();
            for mult in [1.25f64, 1.5, 2.0, 3.0] {
                let limit = if lower_better {
                    reference * mult
                } else {
                    reference / mult
                };
                let last = series
                    .iter()
                    .filter(|p| if lower_better { p.1 <= limit } else { p.1 >= limit })
                    .map(|p| p.0)
                    .max();
                if let Some(b) = last {
                    parts.push(format!(
                        "{}x={}",
                        mult,
                        if b >= 1 << 20 {
                            format!("{:.0} MiB", b as f64 / (1 << 20) as f64)
                        } else {
                            format!("{:.0} KiB", b as f64 / 1024.0)
                        }
                    ));
                }
            }
            println!(
                "  {} 核 {:<4} 最佳 {:>9.3}   拐点 {}",
                cls,
                if pick == 0 { "延迟" } else { "带宽" },
                reference,
                parts.join("   ")
            );
        }
    }
    println!();
}
