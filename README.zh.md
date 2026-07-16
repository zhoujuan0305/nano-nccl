# nano-nccl

[English](README.md)

面向单机多 GPU 的 All Reduce 通信库，目标是达到 NCCL `Ring` + `Simple` + 4 channels 的性能；可选 MPI/socket 路径用于多机正确性运行。

---

## 性能

下表给出 out-of-place `busbw` 绝对值（GB/s，`-w 5 -n 20`）。

### 单机：4× RTX A6000

CUDA 12.8，`CUDA_VISIBLE_DEVICES=0,1,2,3`，`--transport auto` 解析为 `mixed`。NCCL 使用 `Ring` + `Simple`、4 channels 和 `NCCL_BUFFSIZE=33554432`。各 size 单元格为 `nano-nccl / NCCL` 的 busbw（GB/s）。

| dtype | 256 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | nano/NCCL 几何平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float | 6.88 / 6.43 | 14.80 / 14.14 | 19.30 / 19.50 | 22.47 / 22.61 | 23.29 / 22.95 | 1.023 |
| fp16 | 6.84 / 6.33 | 14.43 / 14.29 | 19.32 / 19.70 | 22.44 / 22.73 | 23.29 / 23.17 | 1.012 |
| bf16 | 6.83 / 6.41 | 14.73 / 14.33 | 19.38 / 19.95 | 22.48 / 22.77 | 23.31 / 22.74 | 1.015 |

### 两机：2×4 RTX A6000，经 TCP socket

每台机器一个管理四张 GPU 的 MPI 进程，使用 `eno2`、CUDA 12.8、Open MPI 4.1.2。nano-nccl 使用 `--algo ring_simple --transport auto`，解析为 `mixed`（跨机边为 socket）。NCCL 2.25.1 使用 `Ring` + `Simple`、4 channels、`NCCL_SOCKET_IFNAME=eno2`、`NCCL_IB_DISABLE=1`。

| size | nano-nccl time (us) | nano busbw (GB/s) | NCCL time (us) | NCCL busbw (GB/s) | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4203.72 | 0.109130 | 3989.82 | 0.114981 | 0.949116 |
| 1 MiB | 16030.16 | 0.114472 | 16012.80 | 0.114596 | 0.998917 |
| 4 MiB | 66742.16 | 0.109976 | 63165.10 | 0.116204 | 0.946405 |
| 16 MiB | 275982.46 | 0.106384 | 252055.00 | 0.116483 | 0.913301 |
| 64 MiB | 1107119.12 | 0.106078 | 1019285.00 | 0.115219 | 0.920664 |

---

## 构建

依赖：CUDA 12+、CMake 3.18+、libnuma-dev。可选 MPI/socket 构建要求每台机器使用 Open MPI 4.1.2，且所有启动端必须使用相同的 Open MPI ABI。

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_NRANKS=<你的GPU数量> \
  -DNANO_NCCL_CUDA_ARCH=<你的CUDA算力>
make -j$(nproc)
```

例如，4 GPU RTX A6000 (sm_86) 系统：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=86
```

构建产物：

- `build/benchmarks/nano_nccl_all_reduce_bench` — 性能 + 正确性 benchmark
- `build/tests/nano_nccl_correctness` — 纯正确性测试
- `build/tests/nano_nccl_smoke` — 冒烟测试
- `build/tests/nano_nccl_public_api` — 公共 C++ API 覆盖测试
- `build/tests/nano_nccl_p2p_step_counters` — P2P step-counter 覆盖测试
- `build/tests/nano_nccl_p2p_topology` — P2P topology 覆盖测试
- `build/tests/nano_nccl_simple_protocol` — Simple protocol layout 覆盖测试
- `build-mpi/tests/nano_nccl_mpi_correctness` — MPI/socket 正确性测试（MPI 构建）
- `build-mpi/tests/nano_nccl_mpi_bootstrap` — MPI bootstrap 冒烟测试（MPI 构建）
- `build-mpi/tests/nano_nccl_socket_protocol` — socket framing 与 proxy 测试（MPI 构建）

启用 `BUILD_TESTING`（默认开启）时，`ctest --test-dir build
--output-on-failure` 还会运行 BF16 capability-validation 的静态回归检查。

### CMake 选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `NANO_NCCL_NRANKS` | 4 | GPU 数 |
| `NANO_NCCL_NCHANNELS` | 4 | channel 数 |
| `NANO_NCCL_CUDA_ARCH` | 61 | CUDA 算力（如 61 对应 sm_61，75 对应 Turing，86 对应 Ampere） |
| `NANO_NCCL_BLOCK_THREADS` | 512 | 每 block 线程数 |
| `NANO_NCCL_FIFO_BUFF_BYTES` | 33554432 | FIFO buffer 大小（字节，默认 32 MiB） |
| `NANO_NCCL_ENABLE_MPI` | `OFF` | 构建 MPI communicator bootstrap 与分布式 benchmark/test |
| `NANO_NCCL_SOCKET_TEST_FAULT_INJECTION` | `OFF` | 为 `nano_nccl_mpi_correctness` 构建独立的仅测试故障注入库；普通 MPI benchmark 永不包含该钩子 |

NUMA 拓扑在运行时从 `/sys/bus/pci/devices/*/numa_node` 自动检测，换机器不需要改源码。

### 可选 MPI/socket 构建

每台机器都要从同一个 commit、以全局 GPU 数构建。socket listener 仅支持 IPv4；`NANO_NCCL_SOCKET_IFNAME` 必须指定一个恰好解析为一个可用 IPv4 地址的接口。socket 连接没有 TLS、认证或自动重连，因此只能在可信私有网络中使用。

```bash
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_NRANKS=8 -DNANO_NCCL_CUDA_ARCH=86
cmake --build build-mpi -j$(nproc)
```

两机、每机四张 GPU 时，MPMD 的两个 app context 都要显式导出接口：

```bash
mpirun \
  -np 1 --host 192.168.104.246 -x NANO_NCCL_SOCKET_IFNAME=eno2 \
    ./build-mpi/tests/nano_nccl_mpi_correctness --dtype float \
  : -np 1 --host 192.168.104.247 -x NANO_NCCL_SOCKET_IFNAME=eno2 \
    ./build-mpi/tests/nano_nccl_mpi_correctness --dtype float
```

---

## 运行

```bash
# Benchmark（性能 + 正确性）
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype float --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20

# FP16 和 BF16 的正确性/性能运行（BF16 需要 SM80+）
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype fp16 --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype bf16 --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20

# 纯正确性测试
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_correctness

# 冒烟测试
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_smoke
```

## 公共 C++ API

`nano_nccl/communicator.h` 暴露 move-only 的 `Communicator`，用于一个进程管理
全部已配置的本机 GPU。device buffer 与 CUDA stream 由调用者持有。三个数组都必须
对每个 device 各有一个元素，且顺序与 `CommunicatorConfig::devices` 一致。

```cpp
#include "nano_nccl/communicator.h"

#include <memory>
#include <vector>

std::vector<int> devices{0, 1, 2, 3};
nano_nccl::CommunicatorConfig config{devices};
std::unique_ptr<nano_nccl::Communicator> communicator =
    nano_nccl::create_communicator(config);

std::vector<const void*> send_buffers(devices.size());
std::vector<void*> recv_buffers(devices.size());
std::vector<cudaStream_t> streams(devices.size());

// 在每张 device 上各分配一对 out-of-place 的 send/receive buffer 和一个 stream。
// send_buffers[i]、recv_buffers[i]、streams[i] 必须属于 devices[i]。
// ... cudaSetDevice(devices[i]), cudaMalloc, cudaStreamCreateWithFlags ...

constexpr std::size_t count = 1 << 20;  // 每个本地 rank 的元素数。
nano_nccl::CollectiveArgs args{
    send_buffers,
    recv_buffers,
    streams,
    count,
    nano_nccl::DType::Float,
    nano_nccl::RedOp::Sum,
};

communicator->all_reduce(args);  // 仅入队，不会同步。

for (std::size_t i = 0; i < devices.size(); ++i) {
    cudaSetDevice(devices[i]);
    cudaStreamSynchronize(streams[i]);
}
communicator->check_async_error();
```

单机 adapter 要求 `devices` 正好是可见 device 顺序
`{0, ..., NANO_NCCL_NRANKS - 1}`。MPI 构建时，`nano_nccl/mpi.h` 提供
`create_communicator_from_mpi(MPI_COMM_WORLD, config)` 以创建分布式 communicator。
`all_reduce` 为 out-of-place，支持 `float`、FP16、BF16 和 `sum`。
`reduce_scatter` 与 `all_gather` 已在公共 interface 中暴露，但当前会抛出
unsupported-operation 错误。

### 通信路径选择

单机时 `--transport` 接受 `auto`、`shm` 和 `p2p`。分布式 MPI communicator
接受 `auto`，跨进程 ring edge 会解析为 socket。

- `auto`（默认值）对每条 ring edge 独立选择：只有具备 direct NVLink 和双向 CUDA peer access 时才选择 P2P；其余 edge 使用 SHM。最终路径可能是 `shm`、`p2p` 或 `mixed`。
- `shm` 强制使用 mapped host memory 的 SHM FIFO 路径。
- `p2p` 要求每条 ring edge 都具备所需的双向 peer access。任一方向不可用时，初始化会在第一个不可用方向报错，不会回退。

P2P 是单机通信路径，需要每对完整配置环邻居之间的双向 CUDA peer access；它不是多机或网络通信路径。socket 使用可信、仅 IPv4 的 TCP 网络边界，不提供 TLS 或自动重连。

---

## 当前限制

当前仅支持：

- 单机多 GPU 性能路径（已验证 `CUDA_VISIBLE_DEVICES=0,1,2,3`）；可选 MPI/socket 多机 `all_reduce` 正确性路径
- `float`、FP16（`fp16`）和 BF16（`bf16`）类型；BF16 需要 SM80+
- `sum` 规约操作
- out-of-place
- SHM FIFO、device P2P FIFO 通信路径，以及跨进程 ring edge 的可选 MPI/socket；P2P 仅支持单机

尚未建立多机性能验收 gate。本项目不是通用 NCCL 替代品。

未来计划扩展：

- dtype：`double` / `int8`
- reduce op：`max` / `min` / `prod`
- rank 数：2 / 8 / 16（模板参数化，host 侧分发）
- collective：`all_gather` / `reduce_scatter` / `broadcast`
- 通信路径：网络

---

## 许可证

[MIT](LICENSE)
