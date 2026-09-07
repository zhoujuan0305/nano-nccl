# nano-nccl

[English](README.md)

nano-nccl 是一个刻意收窄范围的 GPU collective 通信库，用可读的
`Ring` + `Simple` 实现 out-of-place AllReduce、ReduceScatter 和 AllGather。
它不是 NCCL 的 drop-in replacement，也不承诺兼容 NCCL API 或 ABI。

当前执行模型是 **1 个 OS 进程管理 1 个 GPU rank**：一个 communicator 内的每个
MPI rank 对应一个进程，并选择、独占一张 GPU。这与 PyTorch/nano-megatron 常用的
1 进程 1 GPU 模型一致，公共 collective descriptor 因此只接收一个本地
`send_buffer`、`recv_buffer` 和 `stream`。

## 已实现范围

- collective：AllReduce、ReduceScatter、AllGather
- 算法与协议：Ring only、Simple only
- dtype：`float`、FP16、BF16
- reduce op：`sum`、`avg`、`max`、`min`；AllGather 无 reduce op
- buffer：仅 out-of-place，不支持任意重叠或 in-place
- rank：构建期 `NANO_NCCL_NRANKS`，合同范围为 2、4、8
- 同机进程间 transport：SHM、CUDA IPC P2P，以及逐 Ring edge 的 `auto`
- 跨机 transport：Socket、host-pinned RDMA，以及可选 GDR

本次 1 进程 1 GPU 路径已在单台 4x RTX A6000（SM86）上验证：

- rank 2：显式 SHM、显式 CUDA IPC P2P 和 `auto`
- rank 4：显式 SHM、显式 CUDA IPC P2P 和 `auto`
- 三个 collective，`float`/FP16/BF16，全部适用的 reduce op
- AllReduce 与 ReduceScatter 的 NaN 传播
- 原生 C++ API 与 C ABI v2
- 注入 SHM allocation/registration 与 CUDA IPC allocation/open 失败后，均可成功重建
  communicator

CUDA IPC P2P 由实际 open/close probe 决定，因此即使没有 direct NVLink，只要 PCIe
P2P 可用也会被接受。rank 8 在具备对应硬件执行覆盖前不能描述为已验证支持。

历史 [performance.md](performance.md) 中的表格早于 1 进程 1 GPU 合同；除非明确
标注重新验证，否则不能作为当前架构的性能验收数据。

修改后的 rank 2 重复性 smoke（`float`/`sum`、`-w 5 -n 20`、连续 3 轮）在
256 KiB/1 MiB/4 MiB/16 MiB/64 MiB 上得到的 busbw 中位数分别为：P2P
7.60/19.39/26.62/32.62/34.36 GB/s，SHM
5.57/10.91/13.62/15.08/15.32 GB/s，且 `#wrong=0`。这些只是 nano-nccl 本机
smoke 数据，不是同轮 NCCL 对比或性能验收结果。

## 同机进程间 transport

transport 按有向 Ring edge 解析：

- `p2p`：receiver 在自己的 GPU 上分配 Simple FIFO/control region，通过
  `cudaIpcGetMemHandle` 导出，sender 使用 `cudaIpcOpenMemHandle` 映射。该 edge 的
  CUDA IPC probe 成功即可选择；NVLink 与 PCIe P2P 都有效。
- `shm`：receiver 使用 `shm_open` 创建 POSIX shared-memory FIFO/control region；
  相邻进程 `mmap` 同一 region，并用 `cudaHostRegisterMapped` 暴露给各自 GPU。
- `auto`：同机 edge 优先使用可用的 CUDA IPC P2P，否则使用 SHM；跨机 edge 使用
  Socket。不同 edge 可以得到 `mixed` plan。

显式请求 `p2p` 或 `shm` 时，任一 edge 不满足条件都会在 communicator 创建阶段
失败，不会回退；错误会指出有向 edge 和实际可用 backend。
`Communicator::edge_transport` 与 `nano_nccl_edge_transport` 可查询每条有向 edge 的
解析结果。Simple FIFO 的 step/credit 和 system-scope release/acquire ordering
保持 transport-neutral；SHM/CUDA IPC 只负责 region 的分配、交换、映射与生命周期。

## 构建

依赖：CUDA 12+、CMake 3.18+、libnuma-dev。communicator bootstrap、同机进程间
transport 和 benchmark 都需要 Open MPI 4.1.2 C ABI，因此正常运行构建应开启 MPI：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON \
  -DNANO_NCCL_NRANKS=4 \
  -DNANO_NCCL_CUDA_ARCH=86
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

`NANO_NCCL_NRANKS` 是 communicator 的全局 MPI 进程数，不是单个进程管理的 GPU
数量。进程既可以只看到一张 GPU 并使用本地 `cuda:0`，也可以看到多张 GPU，再通过
`CommunicatorConfig::device` 只选择其中一张。
communicator 创建阶段会拒绝两个同机 MPI rank 选择同一张物理 CUDA device。

主要构建产物：

- `build/src/libnano_nccl.a`：核心实现和公共 collective C ABI
- `build/src/libnano_nccl_mpi.a`：原生 MPI communicator factory
- `build/src/libnano_nccl_mpi_c.so`：供 Python/ctypes 等调用的 MPI C adapter
- `build/benchmarks/nano_nccl_all_reduce_bench`：1 进程 1 GPU AllReduce benchmark
- `build/tests/nano_nccl_mpi_native_api`：原生 API 三个 collective 的正确性矩阵
- `build/tests/nano_nccl_mpi_c_api`：C ABI v2 三个 collective 的 smoke matrix

常用 CMake 选项：

| 选项 | 默认值 | 说明 |
|---|---:|---|
| `NANO_NCCL_NRANKS` | 4 | communicator 的全局 rank/进程数 |
| `NANO_NCCL_NCHANNELS` | 4 | channel 数 |
| `NANO_NCCL_CUDA_ARCH` | 70 | CUDA compute capability；低于 70 的值会被拒绝 |
| `NANO_NCCL_BLOCK_THREADS` | 512 | 每个 block 的线程数 |
| `NANO_NCCL_FIFO_BUFF_BYTES` | 33554432 | 每 channel FIFO 字节数 |
| `NANO_NCCL_ENABLE_MPI` | `OFF` | 构建 MPI bootstrap、进程间 transport 和 benchmark |
| `NANO_NCCL_ENABLE_RDMA` | `OFF` | 构建 RC RDMA；要求 MPI 和 libibverbs |
| `NANO_NCCL_ENABLE_BENCH_PROFILING` | `OFF` | 为 benchmark 启用 NVTX/CUDA profiler；性能验收时保持关闭 |

`float` 和 FP16 需要 SM70+，因为 Simple counter 使用 system-scope ordering；
BF16 需要 SM80+。

## 运行

启动器可以让每个进程只看到自己的 GPU。下面的本机示例使用 Open MPI 提供的
local rank 设置 `CUDA_VISIBLE_DEVICES`，因此隐式使用 `--device 0`：

```bash
mpirun -np 4 --bind-to none bash -c '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  exec ./build/benchmarks/nano_nccl_all_reduce_bench \
    --algo ring_simple --transport auto --dtype float --redop sum \
    -b 262144 -e 67108864 -f 4 -w 5 -n 20
'
```

显式 SHM/P2P 正确性测试：

```bash
mpirun -np 4 --bind-to none bash -c '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  exec ./build/tests/nano_nccl_mpi_native_api shm
'

# 仅当构建 rank 的每条 Ring edge 都通过 CUDA IPC probe 时使用。
mpirun -np 2 --bind-to none bash -c '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  exec ./build/tests/nano_nccl_mpi_native_api p2p
'
```

`--redop` 接受 `sum`、`avg`、`max`、`min`。`avg` 是逐元素
`sum / nranks`。nano-nccl 对三种 dtype 的四种 reduce op 都保持 NaN 传播语义，
包括 packed FP16/BF16 的 `max`/`min`。

## 公共 C++ API

调用者先初始化 MPI，再用一个包含 `NANO_NCCL_NRANKS` 个进程的 communicator 创建
nano-nccl communicator。`CommunicatorConfig::device` 选择该进程拥有的唯一 GPU
rank；torchrun 风格不屏蔽 GPU 时传 `local_rank`，逐进程屏蔽后传 `0`。

```cpp
#include "nano_nccl/mpi.h"

MPI_Init(&argc, &argv);

nano_nccl::CommunicatorConfig config;
config.device = 0;
config.transport = nano_nccl::TransportKind::Auto;
auto communicator =
    nano_nccl::create_communicator_from_mpi(MPI_COMM_WORLD, config);

// send、recv 和 stream 都属于 config.device。
nano_nccl::AllReduceArgs args{
    send,
    recv,
    stream,
    count,
    nano_nccl::DType::Float,
    nano_nccl::RedOp::Sum,
};
communicator->all_reduce(args);  // 只入队，不同步 stream。
cudaStreamSynchronize(stream);
communicator->check_async_error();

communicator.reset();
MPI_Finalize();
```

三个独立 descriptor 的 count/layout 语义为：

| 操作 | count 字段 | 当前 rank 的输入 | 当前 rank 的输出 |
|---|---|---|---|
| `all_reduce` | `AllReduceArgs::count` | `count` | `count` |
| `reduce_scatter` | `ReduceScatterArgs::recv_count` | `recv_count * global_rank_count` | `recv_count` |
| `all_gather` | `AllGatherArgs::send_count` | `send_count` | `send_count * global_rank_count` |

buffer 和非空 CUDA stream 由调用者持有。输入与输出 byte range 不得相同或部分重叠。
communicator 必须在调用者释放 buffer/stream 和 `MPI_Finalize` 之前销毁。

## C ABI v2 与 nano-megatron adapter

`nano_nccl/nano_nccl.h` 提供 exception-safe C ABI；MPI factory 位于
`nano_nccl/mpi_c_api.h`，由 `libnano_nccl_mpi_c.so` 导出。ABI v2 将 collective
descriptor 从本地 rank 数组改成单个标量 pointer/stream，并移除了旧的进程内多 GPU
`nano_nccl_create_communicator` factory。

```c
#include "nano_nccl/mpi_c_api.h"

nano_nccl_mpi_subgroup_config_t config = {
    .color = dp_group_id,
    .key = dp_rank,
    .device = local_rank,  // 若逐进程设置 CUDA_VISIBLE_DEVICES，则改为 0。
    .transport = NANO_NCCL_TRANSPORT_AUTO,
};
nano_nccl_communicator_t* communicator = NULL;
nano_nccl_status_t status =
    nano_nccl_create_mpi_subgroup_communicator(&config, &communicator);

nano_nccl_all_reduce_args_t args = {
    .send_buffer = send,
    .recv_buffer = recv,
    .stream = stream,
    .count = count,
    .dtype = NANO_NCCL_DTYPE_FLOAT,
    .redop = NANO_NCCL_REDOP_SUM,
};
if (status == NANO_NCCL_STATUS_SUCCESS) {
    status = nano_nccl_all_reduce(communicator, &args);
}
if (status != NANO_NCCL_STATUS_SUCCESS) {
    fprintf(stderr, "%s: %s\n", nano_nccl_status_string(status),
            nano_nccl_get_last_error());
}
nano_nccl_destroy_communicator(communicator);
```

adapter 的 `color` 决定从 `MPI_COMM_WORLD` 切出的通信组，`key` 决定 Ring rank 顺序；
每个 subgroup 的大小必须等于构建期 `NANO_NCCL_NRANKS`。如果 embedding application
尚未初始化 MPI，adapter 会以 `MPI_THREAD_FUNNELED` 初始化，并在最后一个由 adapter
创建的 communicator 销毁后释放 MPI。

nano-megatron 的 ctypes adapter 需要同步到 ABI v2：期望版本改为 `2`，三个
collective args 的 buffer/stream 字段使用直接的 `c_void_p`，不再构造长度为 1 的
pointer 数组。ABI v1 consumer 不能加载 ABI v2 library。

## 跨机 Socket/RDMA

Socket listener 只支持 IPv4；每个进程设置
`NANO_NCCL_SOCKET_IFNAME=<interface>`。它没有 TLS、认证或自动重连，只能用于可信
私有网络。

RDMA 构建：

```bash
cmake -S . -B build-rdma -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON \
  -DNANO_NCCL_NRANKS=<global-rank-count> \
  -DNANO_NCCL_CUDA_ARCH=<cuda-arch>
cmake --build build-rdma -j$(nproc)
```

每个进程还需设置 `NANO_NCCL_RDMA_IFNAME=<rdma-interface>`；必要时设置
`NANO_NCCL_RDMA_GID_INDEX=<gid-index>`。默认是注册的 host-pinned FIFO 与 RC
SEND/RECV；`NANO_NCCL_RDMA_USE_WRITE=1` 选择 WRITE+CTS，
`NANO_NCCL_RDMA_GDR=1` 显式选择 GPU memory registration。GDR 能力不可用时会失败，
不会回退。`rdma` 对同机 edge 仍按 P2P/SHM auto 规则解析，对跨机 edge 使用 RDMA，
因此聚合结果通常是 `mixed`。

## 实现归属

- `src/transport/simple/` 负责 Simple FIFO layout、step ordering 和 slice geometry，
  并为每个 rank/channel 独立持久化 send 和 recv base step。
- `src/transport/shm/` 仅负责 mapped host FIFO storage 和 control。
- `src/transport/p2p/` 负责 CUDA IPC allocation、handle exchange、mapping 和
  capability check。
- `src/collective/all_reduce/ring_simple_geometry.h` 负责 Ring channel/rank work partitioning。

transport runtime lifecycle 与 orchestration 仍由 `Communicator::Impl` 管理。

## 当前限制

- 仅 Ring + Simple；无 Tree、CollNet、NVLS、LL、LL128。
- 仅 out-of-place；无 in-place、任意 overlap、broadcast、`double`、`int8`、`prod`。
- `nranks` 仍由一次构建固定，尚未切换到生成的 2/4/8 specialization registry。
- Socket/RDMA 的 1 进程 1 GPU 全矩阵和 NCCL-relative 性能需要重新验收。
- CUDA IPC teardown 使用有界的 peer-close acknowledgement；如果 peer 未在超时内关闭
  imported handle，exporter 会有意泄漏该 device region，而不是释放仍被 peer 映射的
  内存。异常进程失败恢复仍受 MPI runtime 约束。
