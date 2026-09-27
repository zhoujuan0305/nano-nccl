# nano-nccl 项目长期记忆

更新基准：上游 `e31149b`（2026-09-25）。本文件记录可提交的架构事实、验证状态和研究结论；不得写真实主机名、IP、物理网卡名、MAC、GPU UUID、用户绝对路径、凭据。环境差异和原始证据应使用脱敏占位符，并存放在实验记录中。

## 项目范围

nano-nccl 聚焦 Ring + Simple 的 GPU collective，当前包含 out-of-place AllReduce、ReduceScatter、AllGather，以及按 edge 选择的 SHM、P2P、Socket、host-pinned RDMA 和 GDR。GDR 是可选 NET 内存放置方式；此处的 GDR 是 host proxy 驱动 verbs 的 GPUDirect RDMA，不是 GPU 发起的 IBGDA。

## e31149b 中的 GDR 实现路径

- `NANO_NCCL_RDMA_GDR` 未设置或设为 `0`、`false`、`off` 时使用 host-pinned FIFO；设为 `1`、`true`、`on` 时显式请求 device FIFO。其他值报错。显式 GDR 请求失败时抛错，不静默退回 host-pinned。
- `src/transport/rdma/rdma_gdr.{h,cc}` 实现 `RdmaRegisteredMemory` RAII 注册对象、host/device MR 注册与释放，以及 GPU 接收可见性处理。device MR 注册先尝试 CUDA DMA-BUF 导出和 `ibv_reg_dmabuf_mr`，再尝试 `ibv_reg_mr`（例如 peer-memory 路径）；本轮增加可选逐 FIFO 注册机制诊断，详见下文。
- `src/collective/all_reduce/communicator.cu` 在 RDMA edge 上按 placement 分配 host-pinned 或 CUDA device FIFO。控制 FIFO 和 payload 长度元数据仍由 host-mapped 内存承载；RDMA edge 使用 RC QP。非 RDMA edge 继续按原规则走本地 P2P/SHM，因此一次跨机运行的聚合 transport 常为 `mixed`。
- SEND/RECV 与 WRITE+CTS 都有实现。`NANO_NCCL_RDMA_USE_WRITE=1` 选择 WRITE+CTS；host proxy 仍负责提交 verbs 操作。GDR WRITE 的接收端在 CQE 后、推进接收可见性之前，按设备属性调用 `cudaDeviceFlushGPUDirectRDMAWrites`；若设备要求 flush 而 CUDA 不支持 host flush，则初始化失败。
- `tests/rdma_gdr.cu` 是注册/能力 smoke test；`tests/rdma_gdr_static.py` 等是静态结构检查。两者不能单独证明多 rank collective 的端到端正确性或性能。

## 已记录的验证状态与真实缺口

`performance.md` 在 `e31149b` 记录过 2 hosts × 4 GPUs 的 GDR AllReduce 60-cell dtype/op/size 矩阵：每个实现每个 cell 取 5 次完整矩阵重复的中位数，nano/NCCL busbw 几何平均比为 1.101，最低 cell 为 0.937，无 cell 低于 0.90。Nano 请求 GDR，NCCL 使用 `NCCL_NET_GDR_LEVEL=SYS`，报告称 NCCL 日志确认跨机 edge 为 `/GDRDMA`。该矩阵记录的两台主机均为 Linux 5.15、驱动 580.82.07，不能直接作为本轮 Linux 6.8、驱动 580.173.02 的第二台机器上 NCCL GDR 成功的证据。

这份历史矩阵尚未形成已接受的 GDR 基线，因此 formal 3% baseline-regression gate 尚未裁定；低于 NCCL 的 cell 仍为 `Unknown`，不能宣称性能验收通过。`performance.md` 同时记录 packed FP16/BF16 `max`/`min` 的 single-NaN propagation failure 在 GDR=0 与 GDR=1 均存在，普通输入矩阵不构成完整 dtype/reduce-op 语义验收。

## 2026-09-25 本轮开发与验证

基线为 `e31149b`，开发分支为 `gdr-complete`。本轮新增 C++/C 的 `edge_uses_gdr` 查询、benchmark 逐有向 edge 的 transport/`rdma_memory` 输出、RDMA peer 的 GDR 内存放置与 SEND/RECV/WRITE+CTS 协商、32 MiB 实际 FIFO 大小的注册 smoke，以及 `rdma-gdr` 双机长消息正确性测试。`edge_uses_gdr` 在非 RDMA edge 返回 false；在 RDMA edge 报告创建和注册 FIFO 时使用的内存放置。双方配置不一致会在 bootstrap 期间报错。

直接执行的验证：

- 两台机器分别完成 RDMA 构建和 12 项相关 CTest，均 12/12 PASS；参与设备的 32 MiB `nano_nccl_rdma_gdr` 注册 smoke PASS。后续加注册机制诊断后，两端重建成功；单设备 smoke 分别报告 `registration=dmabuf` 和 `registration=device-verbs`，后一名称仅表示设备地址经 `ibv_reg_mr` 成功，不推断内核具体机制。
- 双机 2 rank 和 4 rank 的 `rdma-gdr` 均分别以 SEND/RECV 与 WRITE+CTS 运行并 PASS。每种配置验证 float、FP16、BF16 的 AllReduce、ReduceScatter、AllGather（`sum` 规约），每项 5,242,883 元素、重复两轮。两端使用同一源码，Open MPI 4.1.2；双机进程经管理网络启动，RDMA 数据经独立数据口。不同主机的 CUDA toolkit 版本分别为 13.0 与 12.8。
- 双机 4 rank benchmark 的 host-pinned 256 KiB 与 GDR 256 KiB、1/4/16/20 MiB float `sum` 试跑均报 0 wrong。逐 edge 输出明确显示同机 P2P 与跨机 RDMA 的 `host-pinned`/`gdr` 放置。20 MiB GDR `-w 5 -n 20` 的 busbw 约 11.29 GB/s；这只是试跑，不构成性能基线。
- 交换机仅启用参与测试的两个数据端口及对应 L2/ARP 项；双向绑定数据口 ping 均 3/3 成功。脚本六项单测 PASS，现场幂等重复应用成功。测试结束后交换机服务仍保留，恢复步骤见 `scripts/README_tofino_gdr_l2.md`。

上述 2/4 rank 双机 PASS 发生在注册机制诊断的小改动前。诊断改动不改变注册调用或回退选择，但最新版源码的双机复测未通过：第二台机器的 GPU0、GPU1 在 communicator 初始化多次注册 FIFO 时返回 `Bad address`，尽管各自的单次 32 MiB smoke PASS；同配置下 GPU1 的 host-pinned RDMA 测试退出码为 0。其 GPU2、GPU3 的独立 CUDA smoke 后来均返回 `no CUDA-capable device is detected`。8 rank 构建完成，较早的一次 8 rank 长测试出现结果 mismatch 和 CUDA IPC teardown 错误；不能仅凭 GPU3 异常推断 mismatch 的根因。未重置设备，也未停止其他用户的进程。最新版 GDR 代码的端到端验收需在设备环境稳定后重跑。

维护者后来提供了第二台机器从当天零点开始的内核 NVRM 日志，修正了最初“无法判断故障是否与本轮测试相关”的结论：当日首条 NVRM/Xid 记录在 19:10:59+08，GPU1 报 `Xid 119`（等待 GSP RPC 超时 45 秒），进程名为 `nano_nccl_mpi_n`；19:12:30 的 `Xid 154` 将其恢复状态从 None 改为 GPU Reset Required。GPU0 在 19:17:17 报同类 `Xid 119`，进程同为 `nano_nccl_mpi_n`；19:18:48 的 `Xid 154` 同样要求 reset。两次 Xid dump 的 RPC event history 均包含 `MMU_FAULT_QUEUED`，但不足以定位 GPU 访存、RDMA/IOMMU 映射、驱动或硬件中的具体根因。我们的测试进程触发了驱动故障；与更早的 2/4 rank PASS 相比，后续的 `Bad address` 和 CUDA 初始化失败是在故障状态下观察到的，不能将其当成独立的软件回归证据。GPU0 上其他用户的进程仍存活，但其服务是否正常未验证。我们未 kill、reset 或重载驱动；第二台机器上的进一步 GPU 测试已暂停。

复核当日本地原始执行记录后，已把两次 Xid 对应到具体测试。19:10:11+08 发起 8 rank 双机 `build-gdr-r8-mpi/tests/nano_nccl_mpi_native_api rdma-gdr`，第二台机器的 PID `4126815` 在运行期间由 `pgrep` 捕获，随后于 19:10:59+08 出现在 GPU1 的首条 Xid 中；该轮最终报 `GDR all_reduce output mismatch` 与 CUDA IPC 清理错误。19:16:29+08 又发起 4 rank 双机 `build-gdr-r4-mpi/tests/nano_nccl_mpi_native_api rdma-gdr`，第二台机器的 PID `4131188` 在运行期间由 `pgrep` 捕获，随后于 19:17:17+08 出现在 GPU0 的首条 Xid 中。两次命令均设置 `NANO_NCCL_RDMA_GDR=1`、`NANO_NCCL_RDMA_USE_WRITE=0`，即请求 GPU FIFO 注册及 SEND/RECV 数据路径；后一次测试是在 GPU1 首次 Xid 之后发起的，当时没有及时取得内核日志确认故障。8 rank mismatch 与 CUDA IPC 错误出现在该轮异常退出时，不能据此断定它们先于或引发 Xid。两次超时的 `GSP_RM_CONTROL` 命令码 `0x20801702` 在 NVIDIA 公开驱动头文件中是 `NV2080_CTRL_CMD_MC_SERVICE_INTERRUPTS`（服务中断），不是 GPU 显存注册 API；进程和配置已经确认，但造成 GSP 停顿的原始访存或驱动原因仍未查明。

双机环境并不相同：第一台为 Linux 5.15、NVIDIA 580.82.07 Open Kernel Module、IOMMU 显式禁用，单次注册走 DMA-BUF；第二台为 Linux 6.8、NVIDIA 580.173.02 proprietary module，单次注册走设备地址的 verbs 回退路径。进一步只读检查确认第二台 GPU0/1 与本轮 RDMA 数据网卡的 IOMMU group `type` 均为 `DMA-FQ`，第一台对应设备没有 IOMMU group。`DMA-FQ` 表示默认 DMA domain，并不报告每笔 GPU/NIC peer transaction 的实际映射或路由；本轮第二台机器此前还通过了 2/4 rank 的 GDR SEND/RECV 与 WRITE+CTS 测试。因此不能仅凭 `DMA-FQ` 断定第二台 GDR 不可用，更不能将它认定为 `Xid 119` 的根因。NVIDIA GPUDirect RDMA 文档对 IOMMU 地址转换有兼容性限制，但是否影响本轮具体路径尚需同设备、同 HCA 的实测证据；不要仅凭这个 sysfs 类型修改共享主机的 IOMMU 配置。

第二台 `/usr/local/cuda/bin/` 已有 Nsight Systems 2024.6.2 与 Nsight Compute 2025.1.0，只是默认 PATH 没有这两个命令；故障前 GPU 性能计数器参数 `RmProfilingAdminOnly=1`，`kernel.perf_event_paranoid=4`。安装/PATH 和在线 sysctl 无须重启；共享机器优先保持管理员计数器限制。DMA-BUF GDR 需要 Open Kernel Module，`nvidia-peermem` 加载不等于 DMA-BUF；现有项目代码先试 DMA-BUF、失败后自动试 `ibv_reg_mr`，没有强制 DMA-BUF 的开关。第二台 NVIDIA 580 软件包有 hold，`apt-get -s install nvidia-driver-580-open` 因版本依赖冲突失败；不可在没有完整同版本包方案与回退方案时直接切换驱动 flavor。

2026-09-25 后续恢复尝试：第二台 GPU0 上其他用户的 vLLM 后来由维护者清理，GPU0/1 显存占用均降为 0 MiB，但 `GPU Recovery Action` 仍为 `Reset`。维护者先在 `nv-hostengine` 占用 GPU0/1 时执行 `nvidia-smi --gpu-reset -i 0,1`，两卡均返回 `Not Supported`；随后停止实际的 `snap.dcgm.nv-hostengine.service`，`fuser` 不再显示 GPU0/1 占用，重复命令仍返回相同结果。主机为裸机，Linux PCI sysfs 对四张卡均列出 `reset_method=flr bus`，因此不能简单归因于虚拟机限制、用户态占用或 PCI 层完全没有复位方法；这也不证明 NVIDIA 驱动当前能够执行复位。复位尝试时间窗口的内核日志在过滤掉既有重复 IRQ 警告后，无新增 NVRM/Xid/reset/AER 信息，尚无法判定具体拒绝条件。DCGM 服务重启后报 `CacheManager Init Failed. Error: -19` 并进入 failed；DCGM 的 -19 是 `DCGM_ST_RESET_REQUIRED`，属于故障未清除的后果。前述“只能整机重启”的表述过于确定：安排主机重启是比绕开驱动做 PCI/sysfs 复位更稳妥的恢复选择，但 `Not Supported` 本身不能证明其它复位机制在硬件上绝无可能。重启前需协调第二台机器的所有用户与任务；恢复后先验卡状态，再排查 GDR 根因，不能直接重跑压力测试。

第二台本机 `nvidia-smi --help` 还列出 `-r bus`：此前失败的 `--gpu-reset -i 0,1` 使用默认 FLR，因此 NVIDIA 接口的 Bus Reset 尚未尝试。只读 PCI 拓扑显示 GPU0、GPU1、GPU2、GPU3 各挂在不同的下游 bus 和上游桥；GPU0/1 的 sysfs `reset` 节点存在，`reset_method=flr bus`，但这些事实不能保证 NVIDIA Bus Reset 对已挂起的 GSP 成功，也不能保证驱动层面完全不影响其它 GPU。当前账号远程 `sudo -n` 需要密码，未执行任何额外复位。若维护窗口内核实目标 GPU 无占用，可以优先评估 NVIDIA 提供的 `sudo nvidia-smi -r bus -i 0,1`，而不是直接写 PCI sysfs 或卸载全机驱动；失败则安排协调重启。即使 Bus Reset 恢复显示，仍需单独查明 GDR 故障；不能从 `DMA-FQ` 一项直接决定修改 IOMMU 配置。

2026-09-26 第二台重启后，四张卡的 `gpu_recovery_action` 均为 `None`；只读复核发现 `nvidia_peermem` 未加载，GPU0/1 与 RDMA NIC 的 IOMMU group `type` 仍为 `DMA-FQ`，`RmProfilingAdminOnly=0`。这只是重启后的状态快照，不能反推故障前 `nvidia_peermem` 是否已加载，也不能证明此时 NCCL 或 nano 正在走真正的 GDR。维护者报告 NCCL GDR 可运行；若要把该报告用于当前主机的路径对照，需保留同设备、同 HCA、同次运行的 `/GDRDMA` 日志和正确性结果。

2026-09-26 使用现有 `nccl-tests` 完成了当前两台机器的 8 GPU NCCL GDR 对照。第二台加载 `nvidia_peermem` 后，每台各 4 GPU、RoCE 同一数据口、`NCCL_NET_GDR_LEVEL=SYS`，日志确认跨机 `NET/IB/.../GDRDMA`。最初第二台数据口 MTU 9000、第一台 1500，两次 8 GPU 试跑均在首个数据操作附近由第一台报告 `<hca>` QP `local access violation work queue error`，没有正确性结果；两次中一次禁用、一次允许 NCCL DMA-BUF，均失败。普通主机内存 `ib_write_bw -m 1024` 在同一数据口成功。维护者把第二台数据口临时调成 MTU 1500 后，两端 HCA `active_mtu` 都变成 1024；相同的 8 GPU GDR 小消息测试及 1/4/16 MiB 测试均退出 0，`Out of bounds values: 0`，无 NCCL WARN，最高 16 MiB 的 busbw 约 11.43 GB/s。两端 GPU `gpu_recovery_action` 仍均为 `None`。这些同次结果证明当前第二台可运行 NCCL GDR，并把 MTU 不一致确定为可复现失败的相关条件；它们还不足以证明 MTU 是单独根因。此轮设了 `NCCL_NET_GDR_READ=0`，日志只证明接收端 `/GDRDMA`，不能据此推断 NIC 从发送端 GPU 显存读取也已验证。随后在相同 MTU 下设 `NCCL_NET_GDR_READ=1`，8 GPU 256 KiB–1 MiB 测试也退出 0、0 wrong、0 WARN，日志确认发送和接收均为 `/GDRDMA`。

同日代码审查确认 `register_device()` 原先在 DMA-BUF 与设备地址 verbs 注册前，均未给 CUDA allocation 设置 `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`。本轮已在两条注册路径共同入口添加 `cuPointerSetAttribute` 及失败显式报错；另增可选 `NANO_NCCL_RDMA_DIAGNOSTICS=1`，逐 GDR FIFO 打印方向、edge、channel、注册方式和 MR key，失败 CQE 打印 `wr_id`、`qp_num`、`vendor_err`。`SYNC_MEMOPS` 是 CUDA GPUDirect RDMA 无旧式 P2P token 路径的内存操作顺序要求；遗漏是实际代码缺口，但现有日志不能证明其引发旧 `Xid 119`，普通 NCCL NET/IB 路径也没有对应的逐 allocation 调用，不能将 NCCL 与 nano 的差异简化为这一行。当前测试同时改变了重启后的设备状态、`nvidia_peermem` 加载情况、MTU 和此代码，因此缺少旧故障的单变量复现。

两端用相同补丁构建后，双机 2 rank GDR 256 KiB 与 8 rank GDR 256 KiB、1/4/16 MiB float `sum` AllReduce 均退出 0、0 wrong；8 rank 原生 `rdma-gdr` 完整用例在 SEND/RECV 和 WRITE+CTS 下均 PASS，覆盖 float/FP16/BF16 的 AllReduce、ReduceScatter、AllGather `sum`，每项 5,242,883 元素、两轮。诊断日志显示第一台的 GPU FIFO 注册为 `dmabuf`，第二台为 `device-verbs`；两端 GPU `gpu_recovery_action` 仍为 `None`。这些是有限正确性和短时稳定性验证，不构成完整 dtype/redop 或性能验收。旧故障的首条 Xid 位于第二台 GPU1，而其他用户 vLLM 当时占用 GPU0；GPU0 的 Xid 出现在 GPU1 故障后的第二轮测试。GPU0 的共享负载可能增加压力，但不能解释首条 GPU1 Xid，不能据此归因为 vLLM。实验记录见独立 experiments 仓库的 `nano-nccl/two-host-rdma-gdr/runs/2026-09-26-diagnosis/RESULTS.md`。第二台数据口的 MTU 1500 仍为临时调整，后续变更应保持两端一致并协调维护者。

同日对当前两台机器的 8 GPU GDR AllReduce 进行了初始性能采样：float/FP16/BF16 × sum/avg/max/min × 256 KiB、1/4/16/64 MiB 共 60 格，每格各模式及 NCCL 双向 GDR `-w 5 -n 20` 交错执行五轮并分别取耗时中位数；两组 nano（WRITE+CTS、SEND/RECV）各自配同轮 NCCL `NCCL_NET_GDR_READ=1` 对照。两组共 240 次进程组运行、1200 条尺寸样本，全部退出 0、`wrong=0`，每次后两端 GPU `gpu_recovery_action=None`。旧表 WRITE+CTS 的 nano/NCCL `busbw` 几何平均比为 1.032、最低格 0.303，60 格中 1 格低于 0.90；SEND/RECV 为 0.962、最低格 0.420，15 格低于 0.90。这些只是不同计时口径下的观测数值，不再作为 0.95/0.90 门槛判定。WRITE+CTS 的 float/max/256 KiB 五轮出现 151–4797 µs 的大幅波动，中位数 515 µs；没有因异常偏慢而删样。SEND/RECV 多个 16 MiB 格慢于旧 NCCL 对照。该矩阵与历史 `performance.md` 的环境和 NCCL GDR read 设置不同，正式 baseline 仍未冻结。完整 60 格、五轮原始数据及硬件/构建指纹见独立 experiments 仓库 `nano-nccl/two-host-rdma-gdr/runs/run-20260926-001/RESULTS.md`。

随后新增 `nano_nccl_collective_bench`，以相同两台机器、同一 `-w 5 -n 20`、五轮交错、NCCL 双向 GDR 对照，实测 ReduceScatter 的 60 格与 AllGather 的 15 格；两种 nano GDR 模式合计 300 次进程组运行、1500 条尺寸样本，全部退出 0、`wrong=0`，每次后两端 GPU 恢复状态均为 `None`。旧表 ReduceScatter WRITE+CTS 的 nano/NCCL 几何平均比 0.754、最低格 0.189、16/60 格低于 0.90；SEND/RECV 分别为 0.685、0.059、25/60。AllGather WRITE+CTS 分别为 0.623、0.189、6/15；SEND/RECV 为 0.455、0.064、11/15。由于 nano 逐轮同步取 rank 最大值，而 nccl-tests 默认将 20 轮异步入队后统一同步并取 rank 平均，这四组旧数值不能用来裁定 0.95/0.90 性能门槛。新 benchmark 还覆盖 8 rank 下 ReduceScatter float/FP16/BF16 的 sum/avg/max/min 与 AllGather 三种 dtype 的有限输入正确性；不涵盖 NaN 语义。小消息中的 nano 耗时有明显长尾，未删样。完整 75 格与五轮原始数据、源码和二进制指纹见独立 experiments 仓库 `nano-nccl/two-host-rdma-gdr/runs/run-20260926-002/RESULTS.md`。

计时口径复核与定向同口径对照见独立 experiments 仓库 `nano-nccl/two-host-rdma-gdr/runs/run-20260926-003/RESULTS.md`。NCCL-only 五轮对照表明，改用 `-z 2` 每轮同步会让其 256 KiB ReduceScatter 和 AllGather 耗时分别增加约 31% 和 23%；`-a 3` 的进程最大值聚合影响相对小。保持双向 GDR、Ring/Simple、4 channels 后，以 NCCL `-z 2 -a 3` 配对测三组代表性 dtype/op × 五个大小：WRITE+CTS 的三组 15 格全部达到或超过 NCCL（最低比约 1.01）；SEND/RECV 仍有 AllGather 256 KiB–16 MiB 和部分 ReduceScatter/AllReduce 格较慢。SEND 1 MiB AllGather 的诊断显示慢轮的 host enqueue 仍约 3–5 µs，send proxy 一直轮询，但 GPU 大部分时间在等待上游 `recv_tail`；verbs post、send CQ completion、recv CQE 后发布均远短于异常间隔。当前聚合 timeline 尚无法找出首个迟到的 hop，不能确定是哪个驱动、硬件或代码事件触发。加入代理线程 CPU 时钟诊断后 12 轮全部变慢，提示插桩可能扰动性能；已从源码撤回该临时插桩。此次同口径定向结果尚不能代替完整 dtype/op/size 矩阵。

2026-09-27 修复 benchmark 口径：两个 nano benchmark 增加 `--wait-mode query|sync`，默认 `query`；`query` 对每次 collective 使用 `cudaStreamQuery`、检查异步错误并在未完成时 `yield`，匹配当前 nccl-tests 的 `-z 2` 等待策略。可用 `sync` 保留旧式 `cudaStreamSynchronize` 作 A/B。比较命令固定 NCCL `-z 2 -a 3 -m 1`、双方 `-w 5 -n 20`，且双方均每 GPU 一个 MPI 进程（双机每机 4 rank，NCCL `-g 1`）；记录口径与 rank 布局，旧 `performance.md` 及 run-20260926-001/002 的比值继续标为历史不匹配，不补写为已通过门槛。新增 `scripts/run_gdr_matched.py` 为三种 collective 保存双机 GDR 的原始逐轮日志、同轮配对及中位数，并检查双方 GDR 路径。脚本级 1 MiB float smoke 共 6 次进程组运行，AllReduce、ReduceScatter、AllGather 双方均 `wrong=0`、两端恢复状态 `None`；其中 AllGather 一次 nano 样本为 3.6 ms 长尾。nano 同一 rank 布局下 `query`/`sync` 各 3 轮的耗时都有显著长尾，因此尚不能把短测差值归因于等待模式或通信实现。nccl-tests 的全尺寸预热遍历、每个 size 额外一次同步操作和 buffer offset 轮换仍与 nano 实现不同；两者计时/聚合与进程布局已匹配，不等于逐条指令完全相同。

随后在同一双机 8 GPU 配置运行首次完整新口径矩阵（现已被修正测量替代）：AllReduce 60 格、ReduceScatter 60 格、AllGather 15 格，每格 nano WRITE+CTS 和 NCCL 双向 GDR 各 5 次，共 270 次进程组运行、1350 条测量行。运行均完成，全部 `wrong=0`；全部原始日志确认 nano 两条跨机 edge 的 `rdma_memory=gdr`，以及 NCCL 两条跨机 edge 发送和接收双向 `/GDRDMA`。两端最终 8 张 GPU 的 `gpu_recovery_action=None`。每格分别取 5 次耗时中位数后计算 `NCCL time / nano time`，AllReduce、ReduceScatter、AllGather 几何平均比分别为 0.672、0.692、0.483，最低格分别 0.313、0.216、0.199；全部 135 格几何平均 0.656，68 格低于 0.90。按该次有偏测量计算，0.95 几何平均、0.90 单格门槛未通过；此门槛结论已撤回。64 MiB 的 27 格均不低于 0.90，但 256 KiB 的 27 格有 24 格低于 0.90。nano 小消息同配置的五轮数据多次呈快慢双模，慢样没有删除；当时尚未定位导致短消息测量偏高的事件。此结果不同于旧环境和旧口径的历史矩阵，也不是已接受的 nano GDR baseline，3% baseline-regression gate 尚无法裁定。完整 135 格、1350 条逐次数据、runner 与两端二进制指纹见独立 experiments 仓库 `nano-nccl/two-host-rdma-gdr/runs/run-20260927-001/RESULTS.md`。

2026-09-27 小消息慢轮定位：旧 nano benchmark 把 5 次预热全部放在 `MPI_Barrier` 前，barrier 返回后立即计时。双机 8 rank AllGather 256 KiB 的逐 rank 时间戳显示，同一主机有 1 个 rank 比其它 rank 晚开始约 7–9 ms，另外 7 个 rank 在计时迭代 0 各等待 7–9 ms；其余 19 次并没有同量级等待。20 次均值因此增加约 350–450 µs，再取 rank 最大值，形成旧矩阵的 450–600 µs 假性慢样。`query` 与 `sync` 均出现，不能归因于 `cudaStreamQuery`。受控对照中，在 barrier 后放一次不计时 collective 后，同机起跑差从 7207 µs 缩至 29 µs，首次等待峰值从 7291 µs 降到 98 µs，报告均时从 450.51 µs 降到 85.79 µs。正式把原有 5 次预热中的最后一次移到 barrier 后，总预热次数不变；AllReduce、ReduceScatter、AllGather benchmark 均完成此修复。旧 `run-20260927-001` 保留作诊断证据，不再用于性能门槛。

使用修正后的 benchmark 重新完成完整双机 8 GPU WRITE+CTS GDR / NCCL 双向 GDR 配对矩阵：135 格、每格双方 5 次、270 次进程组、1350 条逐次测量，`failures=0`、全部 `wrong=0`。两端 nano 构建均为 4 channel；NCCL `NCCL_ALGO=Ring`、`NCCL_PROTO=Simple`、`NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=4`，135 份 NCCL 标准输出全部出现 4 条 Ring channel，并在跨机收发方向显示 `/GDRDMA`；135 份 nano 标准输出均有两条 `rdma_memory=gdr` 跨机 edge。两端全部 8 张 GPU 测后 `gpu_recovery_action=None`。格内先取各自 5 次耗时中位数，`NCCL time / nano time` 的总体几何平均为 1.107、最低格 1.000；AllReduce、ReduceScatter、AllGather 分别为 1.077、1.127、1.146，三个 collective 的 0.95 几何平均与每格 0.90 的 NCCL 相对门槛均通过。仍有少数逐次长尾和另一个全 rank 稳态慢态；5 次中位数未被其主导，底层触发原因未定位。现有结果仅针对有限输入、WRITE+CTS；SEND/RECV 完整矩阵与 NaN 语义不在本次性能结论内。当前环境没有已接受的 nano GDR baseline，formal 3% baseline-regression gate 仍不可裁定。完整表与原始逐次数据见独立 experiments 仓库 `nano-nccl/two-host-rdma-gdr/runs/run-20260927-003/RESULTS.md`，逐 rank 因果对照见 `run-20260927-002/RESULTS.md`。

仍待完成：

1. 可选诊断现能逐 FIFO 报告方向、edge、channel 与注册方法；正常 benchmark 仍只报告 FIFO 放置。旧故障后的 `Bad address` 不能在恢复后的短时测试中复现，其先后因果仍待查。
2. 初始对照采用不同计时口径；首次完整同口径矩阵仍有 barrier 后起跑错位偏差，两者的门槛判定均已撤回。修正后的完整 WRITE+CTS 矩阵通过当前 NCCL 相对门槛，但少数逐次长尾及另一种全 rank 稳态慢态尚未定位。SEND/RECV 的完整同口径矩阵、已接受的 GDR nano baseline 与 formal 3% 回退 gate 均未完成。
3. 本轮跨机 8 rank 的 ReduceScatter 所有有限输入规约及 AllGather 三种 dtype 已通过五个尺寸的 benchmark 正确性检查；NaN 语义、其它 rank 的完整矩阵仍未验证。
4. 已知 packed FP16/BF16 `max`/`min` single-NaN 缺陷尚未修复，完整 scope contract 验收不能宣称通过。

后续试验仍应记录精确源码 commit、脱敏原始日志和构建环境；不得把上述试跑当作已接受的性能基线。

## 验收证据格式

每项结论按以下记录，区分直接事实、推断与未知：

```text
日期与记录 ID：
nano-nccl commit：
测试类型：静态 / 构建 / 注册 smoke / 单机 / 双机 correctness / 性能
拓扑与 rank placement：使用 <host-a>、<device> 等占位符
软件环境：CUDA、driver、NIC driver/firmware、MPI、verbs 版本
构建选项与相关环境变量：
执行命令与退出码：
GDR 选择及每条跨机 edge 的实际 placement：
注册能力：每个参与设备成功/失败；DMA-BUF、peer-memory 或未知
正确性矩阵：collective × dtype × redop × size；失败保留最小复现
性能矩阵：原始每次结果、汇总方式、对照顺序、异常样本及处理理由
原始日志/trace 的脱敏 artifact 相对路径与 SHA256：
结论：直接事实 / 推断 / 未知；keep / reject / pending
尚未通过的 gate 与下一步：
```

双机试验至少保存两端相同源码 commit 和二进制校验、rank 到 GPU 映射、接口/HCA 选择（以占位符记录）、GDR/NCCL GDR 日志、退出码与原始逐次数据。若一端 GDR 注册失败，保留完整错误并将该轮标为环境或能力失败；不能把 host-pinned fallback 的结果算作 GDR 结果。

## 文档注意

`AGENTS.md` 已把早期“eventually implement GPUDirect RDMA”的过时描述更新为现有 host-proxy GDR 路径。其较早的 Current Implementation Snapshot 是历史范围陈述；上述本轮证据优先用于判断 GDR 的当前覆盖状态。
