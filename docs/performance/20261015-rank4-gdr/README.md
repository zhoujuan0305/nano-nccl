# 双机四卡：最新 nano 对照 NCCL 2.30.7（共享 GPU）

## 发布记录与源码范围

这是独立 experiments 仓库 `nano-nccl/two-host-rdma-gdr/runs/run-20261015-001/` 的脱敏结果副本。
对应 GAP-ID：`GAP-LATEST-NCCL-RANK4-SHARED-001`。公开副本包含全部224条作业测量、
112对配对结果、28格统计、构建/二进制指纹与审核摘要；不包含原始机器日志或二进制。

**被测 nano 是本地 base commit `cf473a44b9bdc89a1dacb88ccfef612a3e4deba9`
加冻结的未提交诊断修改，不能用该 base commit 或当前 main 单独重建。**
本次文档提交没有合入这些实现修改，也没有把 U8/边界对齐设成 main 默认值。
`meta.yaml` 记录被测 kernel SHA256、NCCL 源码/库与 nccl-tests 指纹；
`binary-pins.csv` 保留两端各 channel 的实际二进制 SHA256。完整源码及构建冻结清单
留存在原实验记录中。NCCL 既有参考构建缺完整编译日志，不能声称逐字节可重现。

此处 `verification.json` 的历史字段名 `*_sm_*` 实际存储
`nvidia-smi utilization.gpu` 离散采样，不是细粒度 SM occupancy。


## 结果摘要

完成 channels1/2/3/4 ×7尺寸 ×4对AB/BA，共 **224个正式作业**，全部退出0、wrong0。28格中，按两臂各四个作业均值的中位数比较，nano在27格更快；25格四对全部更快。

28格中位数耗时比的等权几何平均 `NCCL/nano` 为 **1.1024×**；全部112个同轮配对比值的几何平均为 1.1155×。这些是本矩阵指标，不代表实际训练负载的加速比例。

## 测量条件

- 双机每台physical GPU0/1，MPI ranks0/1在 `<host-a>`、ranks2/3在 `<host-b>`。两端数据口100Gb/s、MTU1500，本机NVLink P2P，跨机发送/接收GDR；每channel实际Ring经日志核验。原有获准vLLM workers按PID、UID、start_ticks、bootID冻结身份，持续共存；未结束其他任务。
- nano：最新开发worktree，U8、16B channel边界对齐、向量主体/标量尾部、原Simple同步、shared progress、WRITE+CTS、selective CQE。288 logical/256 worker threads、32MiB FIFO/channel，CUDA12.8/SM86；热路径插桩OFF。**默认仍是U1/边界对齐OFF，本轮测试的是诊断配置。**
- NCCL2.30.7参考现有库，明确Ring/Simple、NTHREADS256（实测9warps）、32MiB/channel、GDR。两端库与tests SHA相同；库/源码/版本及DSO均核验。正式NCCL profiler显式none；没有使用NCU/nsys。
- 双方float/sum/OOP AllReduce、w5/n20。nano query/max，nccl-tests `-g1 -z2 -a3 -m1`，每GPU一个MPI进程。每格4对顺序AB/BA/AB/BA；第2、第4轮反转cell遍历顺序。全部慢样本保留。
- 参考NCCL源码、库与tests pins见[meta.yaml](meta.yaml)、[binary-pins.csv](binary-pins.csv)。参考构建缺完整编译日志，既有cubin列含SM80等而未列SM86；本轮在SM86设备运行，不能声称可从完整构建命令逐字节重现。

## 全部尺寸与 channel

每格为 **nano / NCCL 的中位耗时（μs）**；每臂中位数来自四个独立作业的20轮均值。

| 消息 | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| 1MiB | 194.44 / 229.12 | 163.47 / 199.26 | 156.94 / 187.21 | 154.98 / 195.28 |
| 1MiB+4B | 201.08 / 227.79 | 164.37 / 195.81 | 157.03 / 187.29 | 157.22 / 188.59 |
| 9MiB | 1416.87 / 1532.55 | 1262.10 / 1442.60 | 1246.85 / 1463.53 | 1241.38 / 1429.62 |
| 12MiB | 1881.48 / 2067.82 | 1679.88 / 1836.99 | 1654.19 / 1798.06 | 1664.79 / 1787.08 |
| 16MiB | 2513.69 / 2676.43 | 2227.34 / 2408.14 | 2215.30 / 2305.43 | 2204.70 / 2277.51 |
| 16MiB+4B | 2661.34 / 2619.15 | 2243.06 / 2396.92 | 2211.47 / 2350.96 | 2214.41 / 2274.05 |
| 32MiB | 5040.16 / 5184.35 | 4476.26 / 4577.50 | 4497.89 / 4604.35 | 4468.45 / 4579.76 |

## 同轮配对统计

每格为 `100 × (1 − median(nano_us / nccl_us))`，正值表示nano少耗时；括号是nano更快的配对数/4。该统计不能与两臂独立中位数之比混用。

| 消息 | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| 1MiB | +15.31% (4/4) | +17.40% (4/4) | +17.06% (4/4) | +20.19% (4/4) |
| 1MiB+4B | +11.04% (3/4) | +16.00% (4/4) | +15.56% (4/4) | +16.52% (4/4) |
| 9MiB | +7.11% (4/4) | +12.99% (4/4) | +14.80% (4/4) | +13.11% (4/4) |
| 12MiB | +9.01% (4/4) | +8.15% (4/4) | +7.26% (4/4) | +4.46% (3/4) |
| 16MiB | +5.90% (4/4) | +7.17% (4/4) | +4.01% (4/4) | +3.34% (4/4) |
| 16MiB+4B | -1.78% (1/4) | +6.27% (4/4) | +5.99% (4/4) | +2.69% (4/4) |
| 32MiB | +2.62% (4/4) | +2.26% (4/4) | +2.38% (4/4) | +2.51% (4/4) |

## 跨作业波动（保留全部样本）

作为次要统计，各臂四次作业均值再取算术平均时，nano在26/28格更快；该均值耗时比等权几何平均为1.1175×。以下按双方较大CV降序列出前五格，完整四次值和所有格统计均在summary.csv。CV来自四个作业均值，不是逐collective长尾指标。

| 配置 | nano范围 μs | NCCL范围 μs | nano / NCCL CV | nano / NCCL均值 μs |
|---|---:|---:|---:|---:|
| C2/1MiB+4B | 162.76–167.83 | 190.98–286.58 | 1.5% / 21.3% | 164.83 / 217.29 |
| C4/1MiB+4B | 155.49–159.40 | 183.06–271.45 | 1.0% / 20.4% | 157.33 / 207.93 |
| C2/9MiB | 1251.37–1281.27 | 1362.46–2045.11 | 1.0% / 20.2% | 1264.21 / 1573.19 |
| C1/1MiB+4B | 199.65–279.88 | 225.78–242.12 | 18.0% / 3.3% | 220.42 / 230.87 |
| C4/12MiB | 1654.29–2311.95 | 1721.91–1822.41 | 17.8% / 2.8% | 1823.95 / 1779.62 |

## 背景进程与验证

28个metadata格、1232个COLL描述符均验证四rank、count、OOP/IP、Ring/Simple、目标channel、9warps。native GDR correctness及各channel带尾/32MiB短测通过。五阶段228项/阶段源代码、二进制及helper SHA核验一致，10份双机DSO快照无缺失依赖。

共核验2116次参与GPU遥测，recovery均None、显存total-used余量≥1GiB；获准worker身份核验通过（允许原进程退出，出现/缺失次数见verification.json）。每次前后采样不能排除采样间隙中的短暂负载。

显存执行注：本次runner门槛采用total-used余量，NVML实际memory.free另在初末核验，四张卡均为8718–8990MiB且初末相同；两种字段分开保存在verification.json。

| 主机/GPU | 采样GPU利用率范围 | 非零采样 / 总采样 |
|---|---:|---:|
| host-a/GPU0 | 0–0% | 0 / 529 |
| host-a/GPU1 | 0–0% | 0 / 529 |
| host-b/GPU0 | 0–0% | 0 / 529 |
| host-b/GPU1 | 0–0% | 0 / 529 |

## 解释范围

按两臂中位数，nano较慢的格子：C1/16MiB+4B，慢1.61%。各格的因果归因未闭合，保持Unknown；正负样本均保留。

未参与的GPU2/3上也有计算任务，全程离散采样最高GPU利用率约80–84%。参与卡背景采样为0%不代表CPU、内存、PCIe或网络资源完全隔离，也不能排除采样间隙内的活动。这里的利用率是nvidia-smi的utilization.gpu，不是细粒度SM occupancy。

源码另确认C1下nano的16MiB+4B会触发第二个仅含一个有效float的outer loop，NCCL仍在一次outer loop内；详见[SOURCE_NOTES.md](SOURCE_NOTES.md)。这是结构差异，计时差异的因果贡献仍为Unknown。

本轮回答指定四卡、既有vLLM背景和受控Ring/Simple/GDR配置下的相对性能。它不单独分离U8、channel对齐、尾部向量化或协议差异的收益；实现差异的因果贡献仍需独立A/B。不能将本轮与旧双机8rank、10月10日不同GPU映射或未完成样本混合。

四个作业均值只能描述跨作业波动，不能推断逐collective p99或证明长尾消失。共享GPU/主机/网络背景未完全隔离；本轮也不构成完整dtype/op/collective scope及正式3% nano基线回归验收。

## 证据

- [summary.csv](summary.csv)：中位数、配对下降、全部样本、范围、四作业均值CV。
- [paired-results.csv](paired-results.csv)：112对逐轮原始对照。
- [measurements.csv](measurements.csv)：全部224条有效作业数据。
- [verification.json](verification.json)：去除机器标识后的审核摘要。
- 完整配置/命令、日志、环境与进程身份、metadata、SHA/DSO及独立审核脚本保留在独立 experiments 仓库的本地原始记录中；这些原始文件未包含在此公开摘要中。
