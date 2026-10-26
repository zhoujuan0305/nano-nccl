# nano-nccl

[中文](README.zh.md)

## Overview

nano-nccl is a lightweight GPU collective communication library built by analyzing
NCCL's high-performance mechanisms and extracting the key optimizations needed
to achieve comparable performance.

It combines Ring/Simple, multi-channel slice pipelines, fused receive–reduce–send
kernels, vectorized memory access, packed reductions, NUMA-aware placement, and
P2P/GPUDirect RDMA. It provides out-of-place AllReduce, ReduceScatter, and AllGather
for float, FP16, and BF16, with sum, avg, max, and min reductions where applicable.

## Performance

The published two-host, four-GPU comparison against **NCCL 2.30.7** achieved
**1.10× geometric-mean effective bandwidth**, with lower median latency in
**all 20 cases** and zero correctness errors. The matrix covers 1–4 channels
and five message sizes, using Ring/Simple float/sum AllReduce, local NVLink P2P,
and cross-host GDR over 100 Gb/s Ethernet on RTX A6000 GPUs.

Four-channel results (latency in µs):

| Message | nano-nccl | NCCL |
|---|---:|---:|
| 1 MiB | 154.98 | 195.28 |
| 9 MiB | 1241.38 | 1429.62 |
| 12 MiB | 1664.79 | 1787.08 |
| 16 MiB | 2204.70 | 2277.51 |
| 32 MiB | 4468.45 | 4579.76 |

Values are medians of four paired job averages with `-w 5 -n 20`.
See [performance.md](performance.md) for the test environment and full matrix.

## Usage

### Build

Requires Linux, CUDA 12+, CMake 3.18+, libnuma, and Open MPI 4.1.2.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON \
  -DNANO_NCCL_NRANKS=4 \
  -DNANO_NCCL_CUDA_ARCH=86
cmake --build build -j$(nproc)
```

Set `NANO_NCCL_NRANKS` to the total rank count and `NANO_NCCL_CUDA_ARCH` to your GPU
architecture. `float` and FP16 require SM70+; BF16 requires SM80+.
For RDMA, add `-DNANO_NCCL_ENABLE_RDMA=ON` and install libibverbs.

### Run

Single-host AllReduce:

```bash
mpirun -np 4 --bind-to none bash -c '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  exec ./build/benchmarks/nano_nccl_all_reduce_bench \
    --transport auto --dtype float --redop sum \
    -b 262144 -e 67108864 -f 4 -w 5 -n 20 --wait-mode query
'
```

For two hosts with two GPUs each, build the same MPI/RDMA configuration on both
hosts and launch with GDR and WRITE+CTS enabled:

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

Use reachable bootstrap/RDMA interfaces on each host; set
`NANO_NCCL_RDMA_GID_INDEX` when needed by your RoCE setup. GDR requires a working
GPU-memory registration provider. Use bootstrap connections only on a trusted
network.

### C++ API / C ABI

Include [`nano_nccl/mpi.h`](include/nano_nccl/mpi.h) and link the CMake target
`nano_nccl_mpi`:

```cmake
target_link_libraries(example PRIVATE nano_nccl_mpi)
```

After initializing MPI, call this function on every rank. The input, output, and
non-null stream belong to visible GPU 0, as selected by the launch commands above.
Input and output buffers must not overlap.

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

Create the communicator once and reuse it for repeated collectives. Calls enqueue
work on the supplied stream. Destroy the communicator before releasing its buffers
or stream and before `MPI_Finalize`.

| Method | Descriptor | Count per rank |
|---|---|---|
| `all_reduce` | `AllReduceArgs` | `count` input and output elements |
| `reduce_scatter` | `ReduceScatterArgs` | `recv_count` output elements; input is `recv_count * nranks` |
| `all_gather` | `AllGatherArgs` | `send_count` input elements; output is `send_count * nranks` |

For the versioned C ABI v2, use
[`nano_nccl.h`](include/nano_nccl/nano_nccl.h) and
[`mpi_c_api.h`](include/nano_nccl/mpi_c_api.h), provided by
`build/src/libnano_nccl_mpi_c.so`.

## License

[MIT](LICENSE).
