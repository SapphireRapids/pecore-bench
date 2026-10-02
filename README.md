# pecore-bench

在 Windows 11 混合架构 CPU 上**把 P 核和 E 核分开测**：整数 / 标量浮点 / AVX2 / AVX-VNNI / AVX-512 的延迟与吞吐，以及缓存延迟和读带宽曲线，并指出工作集从哪一容量开始掉速。

零第三方依赖，只用 Rust 标准库（CRT 也静态链接），编译产物是**一个自包含的 exe**：拷到任何 Windows x86-64 机器上双击即可，不需要装任何运行时。所有内核都是手写 inline `asm!`，因为目的是测**某一条指令在已知依赖结构下**的表现，而不是 LLVM 愿意为某个 Rust 循环生成什么。

## 构建与运行

```bash
cargo build --release
```

产物 `target/release/pecore-bench.exe` 可直接拷走。两种用法：

**双击运行**（推荐给只想看结果的人）：先打印 CPU 型号、拓扑和指令集探测，然后给出菜单：

```
  1) 开始完整测试（20 核约 80–120 秒，期间请勿干扰机器）
  2) 开始快速测试（采样减半，精度略降）
  3) 只列拓扑，不测试
  4) 只看指令集探测
  q) 退出
```

选 1 或 2 后逐核测量，结束后**自动用默认浏览器打开报告**，并等按回车再关窗口（留在 exe 同目录的 `pecore-report\` 下）。

**命令行**（脚本用）：

```bash
./target/release/pecore-bench.exe                # 全量，20 核约 80 秒
./target/release/pecore-bench.exe --list         # 只看拓扑，不测
./target/release/pecore-bench.exe --features     # 只看指令集探测（含 CPUID 原始位）后退出
./target/release/pecore-bench.exe -y             # 不询问直接跑（不带参数就是全量）
./target/release/pecore-bench.exe -y --open      # 跑完自动开浏览器
./target/release/pecore-bench.exe --quick        # 减少采样轮数
./target/release/pecore-bench.exe --class p      # 只测 P 核
./target/release/pecore-bench.exe --core 0,12    # 只测指定逻辑处理器
./target/release/pecore-bench.exe --no-cache     # 跳过缓存曲线
./target/release/pecore-bench.exe --cache-max 8  # 缓存曲线最大 8 MiB
./target/release/pecore-bench.exe --out 报告目录
```

带参数时不会弹菜单、也不会自动开浏览器（`--open` 除外），交互只在“无参数 + 标准输入是控制台”时出现。

跑之前最好让机器空闲一两分钟。连续跑会让核温上去、频率下移，绝对 ns 数会变（每操作周期数不变，见下）。

## 输出

| 文件 | 内容 |
|---|---|
| `pecore-bench.csv` | 逐核原始数据：拓扑、频率、32 个内核的 ns/iter 与周期数、29 个容量的延迟与带宽 |
| `pecore-bench.html` | 自包含报告，无外部请求：逐核时钟图、分组的横向内核对比图、缓存延迟/带宽曲线（log2 横轴，带可调拐点阈值）、关键容量数值表、逐核明细表 |

默认输出到 **exe 所在目录**下的 `pecore-report\`（不是当前目录），这样双击和从别处调用行为一致；`--out` 可覆盖。HTML 直接用浏览器打开即可，图表是 canvas 现画的。

仓库里放了一份**样例报告**（[`sample-report/pecore-bench.html`](sample-report/pecore-bench.html)，配同目录 CSV），就是下面"这台机器的结果"那一轮的真实输出，下载下来直接双击就能看，不用先编译。

## 方法

**钉核。** 用 `SetThreadGroupAffinity` 把线程按位掩码钉到某一个逻辑处理器，然后必须回读校验：`GetCurrentProcessorNumberEx()` 返回的号要等于钉的 LPI。只用 CPU Sets API 的 `SetThreadSelectedCpuSets` 是不够的——在这台机器上它报告的执行号和实际执行号不一致（实测钉到 cpu set 275 时两次运行分别报 processor 4 和 5，时钟估计也跟着矛盾）。亲和性掩码是唯一效果能被明确读回的机制。每个核一个独立线程，测完 join 再测下一个，核与核之间不互相干扰。

**P/E 分类。** 用 SMT 共享关系判：同一个物理核被两个以上逻辑处理器认领的就是 P 核，独占的就是 E 核。不用 `EfficiencyClass`，因为它在这片 Raptor Lake ES 上极性是反的（E 核报 0、P 核报 1）。

**时钟标定。** RDTSC 是不变频率（这台机器 1.805 GHz），测不出核心在跑多快。用一条依赖 `add r64, r64` 链标定——这条链在所有 x86 上都是每 add 1 周期。每个内核测完后立刻在旁边再跑一次参考链，用 `参考链 ns / 4` 推出当前核心周期，再把内核的 ns 折算成周期。这样做出来的周期数与核心频率无关，换机器、换温度都稳定。

**指令内核。** 每条指令出两个形状：`*_dep` 是单累加器依赖链，测指令延迟；`*_tp` 是 8 条独立链，够把 4-5 周期的延迟填满，测发射宽度。依赖链每次循环展开 4 条——三指令的小循环对链接器把它放在哪里非常敏感（同一个内核编进两个不同二进制，在 E 核上能差 60%，因为跨 32 字节边界的循环不再由 decoded stream buffer 供给），把循环开销摊到多条指令上，每操作数才是指令自己的属性。

**指令集门控。** 32 个内核按 CPU 能力自动取舍：CPU 不支持的内核不测、不显示为“0”，而是在报告里列出“因本机 CPU 不支持而跳过 N 项”。判定用 `std::arch::is_x86_feature_detected!`（`--features` 可以单独跑一遍看结果，并附 CPUID 原始位供交叉核对）。AVX-512 覆盖 F / BW / DQ / VL / VNNI 五个子集，各子集独立门控。

**缓存。** 延迟用 Sattolo 算法生成的单循环随机置换做指针追逐——必须随机化，否则测到的是硬件预取器而不是缓存。带宽用 8 个独立累加器的 AVX2 读，**没有 AVX2 的机器自动退到 SSE2 回退路径**（128 位、4 累加器），保证在任何 x86-64 机器上都不会因指令非法而崩溃。容量从 1 KiB 扫到 64 MiB，29 个点。

## 这台机器的结果

Raptor Lake ES（`Genuine Intel(R) 0000`，family 6 model 186 stepping 2），6 P 核 + 8 E 核，20 个逻辑处理器。**AVX-512 被熔断**：报告里 13 个 AVX-512 内核全部标记为“不支持”，这不是检测失误——三个独立信号一致：`is_x86_feature_detected!` 全部为假、CPUID.(EAX=07H):EBX[16..31] 的 AVX-512 位全为 0、XCR0=0x7（OS 只开了 SSE+AVX 状态，没有 opmask/ZMM）。所以 AVX-512 内核只在支持它的机器上运行，本机无法实测。

时钟（空闲后首轮）：P 平均 4.81 GHz，E 平均 3.79 GHz。连续负载后会不断下探——这台 ES 片在四轮连续全量后落到 P 4.09 / E 2.62 GHz——**每操作周期数不变**（见下），所以周期数的结论与频率无关，只有绝对 ns 和带宽会跟着时钟走。

每操作周期数（随报告一同交付的那一轮；跨多轮对比：吞吐项稳定在 3% 以内，延迟项有 10~20% 跳动，但 P 核与 E 核的大小关系每一轮都一致）：

| 内核 | P | E |
|---|---|---|
| `add r64` 延迟 | 1.02 | 1.00 |
| `imul r64` 延迟 | 3.10 | 5.01 |
| `addsd` 延迟 | 2.02 | 3.00 |
| `mulsd` 延迟 | 4.03 | 4.00 |
| `vfmadd231sd` 延迟 | 4.05 | 6.00 |
| `vpaddd ymm` 延迟 | 1.02 | 1.10 |
| `vpmulld ymm` 延迟 | 10.32 | 4.00 |
| `vfmadd231pd ymm` 延迟 | 4.95 | 6.00 |
| 老配方 `vpmaddubsw+vpmaddwd+vpaddd` 吞吐 | 0.38 | 1.34 |
| `vpdpbusd ymm` 延迟 | 5.10 | 4.00 |
| `vpdpbusd ymm` 吞吐 | 0.65 | 2.01 |

几处值得注意的：

- `vpmulld ymm` 的延迟在 P 核上是 10.3 周期、E 核只有 4.0——P 核这条指令慢 2.5 倍，但吞吐反过来是 E 核慢近 3 倍。延迟和吞吐要分开看。
- `vpdpbusd` 延迟 E 核（4.00）比 P 核（5.10）还低一点，但吞吐 P 核是 E 核的 3 倍（0.65 vs 2.01 周期/操作）。
- 同样做 32 个 int8 MAC：一条 `vpdpbusd` 在老配方的三条指令面前，P 核快 1.7 倍、E 核快 1.5 倍。

缓存（拐点取"最后仍落在最佳值 N 倍以内的容量"）：

| | 延迟 1.25x | 带宽 1.25x | 带宽 3x |
|---|---|---|---|
| P 核 | 48 KiB | 32 KiB | 2 MiB |
| E 核 | 32 KiB | 2 MiB | 4 MiB |

即 P 核 L1d 48 KiB、L2 2 MiB，E 核 L1d 32 KiB、L2 约 2 MiB。缓存各级的**拐点容量**在多轮之间非常稳定，上表的值每轮都一样；**绝对值**则会随当时频率漂移，所以给区间：

| 指标 | P 核 | E 核 |
|---|---|---|
| L1 延迟 | 1.04~1.21 ns | 1.06~1.53 ns |
| DRAM 延迟（64 MiB） | 20~24 ns | 26~27 ns |
| L1 读带宽 | 211~238 GB/s | 84~121 GB/s |
| DRAM 读带宽 | 27 GB/s 上下 | 11 GB/s 上下 |

带宽差得多是因为它随核心频率线性变化：E 核在 3.79 GHz 时 L1 约 121 GB/s，被连续负载压到 2.6 GHz 后就只剩 84 GB/s 左右——同一个内核、同一个二进制。所以引用绝对带宽时务必看报告里的逐核时钟；**拐点容量与周期数不受影响**。

P 核带宽曲线在 4/16/32 KiB 等 2 的幂容量上呈规律性锯齿，快档约为慢档 1.5 倍，六个 P 核完全一致——是实测的硬件行为（疑似 L1 组冲突），不是测量误差。判断容量级别要看整体台阶，不要看单个齿。

## 踩过的坑

这些都已经在代码注释里记了，这里汇总一遍，因为每一条都曾经让报告出错：

1. **依赖 `add r64, imm` 链测不出延迟。** 在这片 Raptor Cove P 核上，`add rax, 1` 的依赖链跑的是 ALU *吞吐*：4 条 add 和裸 `sub/jnz` 循环一样快（约 1 周期），8 条约 1.5 周期，正好是重命名宽度。改成寄存器操作数 `add rax, r8` 才是教科书上的每 add 1 周期。标定时钟的尺子必须用寄存器操作数。
2. **`lto = true` 会把只读函数提出循环。** LLVM 能证明一个 intrinsics 写的只读函数是 `readonly`，于是把它提出重复循环，bench 测到的是自己的循环开销（表现为每个容量都是 0.21 ns/次的平坦直线）。inline `asm!` 不标 `readonly`/`pure`，编译器搬不动，带宽内核因此整体改成手写 asm。
3. **LLVM 只给 `vpdpbusd` 发 EVEX 编码。** EVEX 在没有 AVX-512 的片上是 #UD。这片 Raptor Lake ES 上写助记符直接 SIGILL，尽管指令本身在 P 核和 E 核上都能正常执行。VNNI 内核因此直接发 VEX 字节（`vpdpbusd!` 宏）。
4. **CPUID 取错位。** 手读 `CPUID.(EAX=07H,ECX=0):EBX[bit 4]` 当 AVX-VNNI，那一位其实是 HLE，于是两个 VNNI 内核被静默跳过、报告里少两组数据。特性检测改用 `is_x86_feature_detected!`。
5. **拐点方向。** 延迟是越低越好、带宽是越高越好，两边的拐点逻辑是镜像的。用同一套"低于最小值 N 倍"去算带宽，拐点会永远落在最大的那个容量上。
6. **周期数要按参考链的展开倍数缩放。** 参考链每次循环做 4 条 add，所以周期 = `ns × 4 / 参考链 ns`，漏掉这个 4 会让周期列整体小 4 倍。
7. **HTML 里的 JS 必须真渲染一次。** 用 node 模拟 `seriesFor`/`knee`/表格的数据路径全对，但页面在浏览器里一张图都没画：`tables()` 末尾多打了一个 `}`，让 `const D = …` 之后整段脚本变成 SyntaxError，而且 node 只读生成的字符串、不执行它。把 `<script>` 抽出来跑 `node --check` + 在浏览器里回读 canvas 像素，才发现。
8. **少一个元素会级联炸掉后面的图。** `cacheChart()` 结尾要写 `knee_bw` 的文案，但第 4 节的面板里根本没这个 `<span>`；于是 `render()` 画完带宽图就抛异常，后面的 `freqChart()` 再也没执行——只有 `compute` 因为 `setMetric()` 也调它才幸免。另外容器宽度为 0 时 `clientWidth` 返回 0，canvas 会被设成 0 宽、什么都画不出，所以三处都加了 `Math.max(clientWidth, 320)` 兜底。
9. **AVX-512 内核无法本地执行，验证靠反汇编。** 编码正确性不能靠“编译过了”推断，所以用 `dumpbin /disasm` 反汇编产物，逐条核对每条 AVX-512 指令真的生成了 EVEX 编码（`62 …` 前缀、L'L=512 位、操作数顺序）：`vpdpbusd zmm0,zmm8,zmm9` = `62 D2 3D 48 50 C1`、`vpmulld`/`vpmullq`/`vpaddb`/`vfmadd231pd` 同理。`vpternlogd ymm` 编成 EVEX.256（`62 D3 3D 28 25 C1 96`），它在没有 AVX-512 的机器上是 #UD——这正是它作为 VL 子集探针的意义。zmm 内核用 `#[target_feature(enable = "avx512f")]` 等特性门控加运行时检测，双保险。

## 代码结构

| 文件 | 作用 |
|---|---|
| `src/sys.rs` | Win32：CPU set 枚举、亲和性钉核与回读校验、CPUID（含特性原始位导出）、TSC、控制台检测、调用默认浏览器 |
| `src/kern.rs` | 32 个手写 asm 内核（含 13 个 AVX-512）+ 特性检测 + 内核表 |
| `src/bench.rs` | 计时框架、频率标定、逐核编排 |
| `src/cache.rs` | 指针追逐延迟 + AVX2 读带宽（带 SSE2 回退） |
| `src/report.rs` | 控制台汇总、CSV、自包含 HTML（分组横向条形图 + 缓存曲线 + 数值表） |
| `src/main.rs` | CLI、交互菜单、自动开浏览器 |
| `.cargo/config.toml` | 静态链接 CRT，产出可拷走的单文件 exe |
