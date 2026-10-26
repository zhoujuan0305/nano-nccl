# nano-nccl

[English](README.md)

## 简介

nano-nccl 是一个轻量级 GPU 集合通信库：分析 NCCL 的高性能机制，提炼并实现其中关键的优化，以更精简的实现达到相近的通信性能。

核心实现包括 Ring/Simple、多 channel 分片流水、收取–归约–转发融合、向量化访存、低精度打包归约、NUMA 感知的内存放置，以及 P2P/GPUDirect RDMA。支持 float、FP16、BF16 的 out-of-place AllReduce、ReduceScatter 和 AllGather，归约操作支持 sum、avg、max、min。

## 性能

已公开的双机四卡对照中，相比 **NCCL 2.30.7**，nano-nccl 的**有效带宽几何平均为 1.10 倍，28 个测试组合中有 27 个中位延迟更低**，正确性检查全部通过。测试覆盖 1–4 channels 和七种消息大小，采用 RTX A6000、Ring/Simple float/sum AllReduce、节点内 NVLink P2P 和跨节点 100 Gb/s Ethernet GDR。

四 channel 的延迟结果（单位：µs）：

| 消息大小 | nano-nccl | NCCL |
|---|---:|---:|
| 1 MiB | 154.98 | 195.28 |
| 1 MiB + 4 B | 157.22 | 188.59 |
| 9 MiB | 1241.38 | 1429.62 |
| 12 MiB | 1664.79 | 1787.08 |
| 16 MiB | 2204.70 | 2277.51 |
| 16 MiB + 4 B | 2214.41 | 2274.05 |
| 32 MiB | 4468.45 | 4579.76 |

数据为四轮配对作业平均耗时的中位数，每轮使用 `-w 5 -n 20`。完整的 2026-10-15 矩阵、测量配置及集成验证见 [performance.md](performance.md#2026-10-15-two-host-four-rank-gdr)。

## 如何使用

### 构建

依赖 Linux、CUDA 12+、CMake 3.18+、libnuma 和 Open MPI 4.1.2。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON \
  -DNANO_NCCL_NRANKS=4 \
  -DNANO_NCCL_CUDA_ARCH=86
cmake --build build -j$(nproc)
```

将 `NANO_NCCL_NRANKS` 设为总 rank 数，`NANO_NCCL_CUDA_ARCH` 设为 GPU 架构。`float` 和 FP16 需要 SM70+；BF16 需要 SM80+。
使用 RDMA 时，额外启用 `-DNANO_NCCL_ENABLE_RDMA=ON` 并安装 libibverbs。

### 运行

单机 AllReduce：

```bash
mpirun -np 4 --bind-to none bash -c '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  exec ./build/benchmarks/nano_nccl_all_reduce_bench \
    --transport auto --dtype float --redop sum \
    -b 262144 -e 67108864 -f 4 -w 5 -n 20 --wait-mode query
'
```

双机各两卡时，两端使用相同的 MPI/RDMA 构建，并启用 GDR 和 WRITE+CTS：

```bash
export NANO_NCCL_SOCKET_IFNAME='<interface>'
export NANO_NCCL_RDMA_IFNAME='<rdma-interface>'
export NANO_NCCL_RDMA_GDR=1
export NANO_NCCL_RDMA_USE_WRITE=1
mpirun -np 4 --host '<host-a>:2,<host-b>:2' --bind-to none \
  -x NANO_NCCL_SOCKET_IFNAME -x NANO_NCCL_RDMA_IFNAME \
  -x NANO_NCCL_RDMA_GDR -x NANO_NCCL_RDMA_USE_WRITE \
  bash -c '
    export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
    exec ./build/benchmarks/nano_nccl_all_reduce_bench \
      --transport rdma --dtype float --redop sum \
      -b 262144 -e 67108864 -f 4 -w 5 -n 20 --wait-mode query
  '
```

为两端设置可互通的 bootstrap/RDMA 接口；RoCE 环境需要时设置 `NANO_NCCL_RDMA_GID_INDEX`。GDR 需要可用的 GPU 内存注册支持。bootstrap 连接仅用于可信网络。

### C++ API / C ABI

包含 [`nano_nccl/mpi.h`](include/nano_nccl/mpi.h)，链接 CMake target `nano_nccl_mpi`：

```cmake
target_link_libraries(example PRIVATE nano_nccl_mpi)
```

初始化 MPI 后，由所有 rank 调用以下函数。输入、输出和非空 stream 均属于可见 GPU 0，与上述启动命令一致。输入与输出缓冲区不能重叠。

```cpp
#include "nano_nccl/mpi.h"

#include <stdexcept>

void all_reduce_example(float* send, float* recv, std::size_t count,
                        cudaStream_t stream) {
    nano_nccl::CommunicatorConfig config;
    config.device = 0;
    config.transport = nano_nccl::TransportKind::Auto;
    auto comm = nano_nccl::create_communicator_from_mpi(MPI_COMM_WORLD, config);

    comm->all_reduce({send, recv, stream, count,
                      nano_nccl::DType::Float, nano_nccl::RedOp::Sum});
    const auto status = cudaStreamSynchronize(stream);
    comm->check_async_error();
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}
```

重复通信时只创建一次 communicator 并复用。collective 调用向指定 stream 入队；释放缓冲区、stream 或调用 `MPI_Finalize` 前先销毁 communicator。

| 方法 | 参数类型 | 每个 rank 的元素数 |
|---|---|---|
| `all_reduce` | `AllReduceArgs` | 输入、输出均为 `count` |
| `reduce_scatter` | `ReduceScatterArgs` | 输出为 `recv_count`，输入为 `recv_count * nranks` |
| `all_gather` | `AllGatherArgs` | 输入为 `send_count`，输出为 `send_count * nranks` |

版本化 C ABI v2 使用 [`nano_nccl.h`](include/nano_nccl/nano_nccl.h) 和 [`mpi_c_api.h`](include/nano_nccl/mpi_c_api.h)，由 `build/src/libnano_nccl_mpi_c.so` 提供。

## License

[MIT](LICENSE)。
