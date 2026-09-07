# nano-nccl

[中文](README.zh.md)

nano-nccl is a deliberately narrow GPU collective communication library. It
implements out-of-place AllReduce, ReduceScatter, and AllGather with readable
`Ring` + `Simple` code. It is not a drop-in NCCL replacement and does not
promise NCCL API or ABI compatibility.

The current execution model is **one OS process per GPU rank**: every MPI rank
in a communicator selects and owns one GPU. This matches the usual
PyTorch/nano-megatron model, so each public collective descriptor contains one
local `send_buffer`, `recv_buffer`, and `stream`.

## Implemented scope

- Collectives: AllReduce, ReduceScatter, and AllGather
- Algorithm and protocol: Ring only, Simple only
- Dtypes: `float`, FP16, and BF16
- Reduce ops: `sum`, `avg`, `max`, and `min`; AllGather has no reduce op
- Buffers: out-of-place only; arbitrary overlap and in-place are unsupported
- Ranks: build-time `NANO_NCCL_NRANKS`; the contract set is 2, 4, and 8
- Same-host interprocess transports: SHM, CUDA IPC P2P, and per-edge `auto`
- Cross-host transports: Socket, host-pinned RDMA, and optional GDR

The one-process-per-GPU path has been validated on one 4x RTX A6000 (SM86)
host as follows:

- rank 2: explicit SHM, explicit CUDA IPC P2P, and `auto`
- rank 4: explicit SHM, explicit CUDA IPC P2P, and `auto`
- all three collectives, `float`/FP16/BF16, and every applicable reduce op
- NaN propagation for AllReduce and ReduceScatter
- the native C++ API and C ABI v2
- injected SHM allocation/registration and CUDA IPC allocation/open failures,
  each followed by successful communicator reconstruction

CUDA IPC P2P is selected from a real open/close probe, so a working PCIe P2P
edge is accepted even without direct NVLink. Rank 8 is not validated support
until it has matching hardware execution coverage.

The historical tables in [performance.md](performance.md) predate the
one-process-per-GPU contract. Unless a result is explicitly marked as
revalidated, it is not acceptance evidence for this architecture.

A post-change rank-2 repeatability smoke (`float`/`sum`, `-w 5 -n 20`, three
runs) produced median busbw of 7.60/19.39/26.62/32.62/34.36 GB/s for P2P and
5.57/10.91/13.62/15.08/15.32 GB/s for SHM at 256 KiB/1 MiB/4 MiB/16 MiB/64
MiB, with zero wrong values. These are nano-nccl-only smoke numbers, not a
same-round NCCL comparison or performance acceptance result.

## Same-host interprocess transports

Transport is resolved independently for every directed Ring edge:

- `p2p`: the receiver allocates a Simple FIFO/control region on its GPU and
  exports it with `cudaIpcGetMemHandle`; the sender maps it with
  `cudaIpcOpenMemHandle`. Selection requires a successful CUDA IPC probe for
  that edge; NVLink and PCIe P2P are both valid.
- `shm`: the receiver owns a POSIX shared-memory FIFO/control region created
  with `shm_open`; both neighboring processes `mmap` it and expose it to their
  GPUs with `cudaHostRegisterMapped`.
- `auto`: a same-host edge uses CUDA IPC P2P when available and SHM otherwise;
  a cross-host edge uses Socket. Different edges may resolve to a `mixed` plan.

An explicit `p2p` or `shm` request fails during communicator creation if any
edge cannot satisfy it, and the error names the directed edge and available
backend. `Communicator::edge_transport` and `nano_nccl_edge_transport` expose
the resolved backend of every directed edge. Simple step/credit and system-scope release/acquire
ordering remain transport-neutral; the SHM and CUDA IPC backends own only
region allocation, exchange, mapping, and lifetime.

## Build

Dependencies are CUDA 12+, CMake 3.18+, and libnuma-dev. Communicator
bootstrap, interprocess transports, and the benchmark require the Open MPI
4.1.2 C ABI, so a normal runnable build enables MPI:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON \
  -DNANO_NCCL_NRANKS=4 \
  -DNANO_NCCL_CUDA_ARCH=86
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

`NANO_NCCL_NRANKS` is the global MPI process count for a communicator, not the
number of GPUs owned by one process. A process may see one GPU and use local
`cuda:0`, or see multiple GPUs and select exactly one with
`CommunicatorConfig::device`. Communicator creation rejects two same-host MPI
ranks that select the same physical CUDA device.

Important build products:

- `build/src/libnano_nccl.a`: core implementation and collective C ABI
- `build/src/libnano_nccl_mpi.a`: native MPI communicator factory
- `build/src/libnano_nccl_mpi_c.so`: MPI C adapter for Python/ctypes consumers
- `build/benchmarks/nano_nccl_all_reduce_bench`: one-process-per-GPU benchmark
- `build/tests/nano_nccl_mpi_native_api`: native correctness matrix for all collectives
- `build/tests/nano_nccl_mpi_c_api`: C ABI v2 smoke matrix for all collectives

Common CMake options:

| Option | Default | Meaning |
|---|---:|---|
| `NANO_NCCL_NRANKS` | 4 | Global ranks/processes in one communicator |
| `NANO_NCCL_NCHANNELS` | 4 | Channel count |
| `NANO_NCCL_CUDA_ARCH` | 70 | CUDA compute capability; values below 70 are rejected |
| `NANO_NCCL_BLOCK_THREADS` | 512 | Threads per block |
| `NANO_NCCL_FIFO_BUFF_BYTES` | 33554432 | FIFO bytes per channel |
| `NANO_NCCL_ENABLE_MPI` | `OFF` | Build MPI bootstrap, interprocess transports, and benchmark |
| `NANO_NCCL_ENABLE_RDMA` | `OFF` | Build RC RDMA; requires MPI and libibverbs |
| `NANO_NCCL_ENABLE_BENCH_PROFILING` | `OFF` | Add NVTX/CUDA profiling to the benchmark; keep off for acceptance runs |

`float` and FP16 require SM70+ because Simple counters use system-scope
ordering. BF16 requires SM80+.

## Run

The launcher may expose only the process-local GPU. This same-host example
uses Open MPI's local rank to set `CUDA_VISIBLE_DEVICES`, making `--device 0`
implicit:

```bash
mpirun -np 4 --bind-to none bash -c '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  exec ./build/benchmarks/nano_nccl_all_reduce_bench \
    --algo ring_simple --transport auto --dtype float --redop sum \
    -b 262144 -e 67108864 -f 4 -w 5 -n 20
'
```

Explicit SHM and P2P correctness runs:

```bash
mpirun -np 4 --bind-to none bash -c '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  exec ./build/tests/nano_nccl_mpi_native_api shm
'

# Use only when the CUDA IPC probe succeeds for every Ring edge.
mpirun -np 2 --bind-to none bash -c '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  exec ./build/tests/nano_nccl_mpi_native_api p2p
'
```

`--redop` accepts `sum`, `avg`, `max`, and `min`. `avg` means element-wise
`sum / nranks`. nano-nccl propagates NaN for every reduce op and dtype,
including packed FP16/BF16 `max` and `min`.

## Native C++ API

The caller initializes MPI, then creates nano-nccl from an MPI communicator
containing exactly `NANO_NCCL_NRANKS` processes. `CommunicatorConfig::device`
selects the one GPU rank owned by this process. With torchrun-style unmasked
visibility, pass `local_rank`; with per-process masking, pass `0`.

```cpp
#include "nano_nccl/mpi.h"

MPI_Init(&argc, &argv);

nano_nccl::CommunicatorConfig config;
config.device = 0;
config.transport = nano_nccl::TransportKind::Auto;
auto communicator =
    nano_nccl::create_communicator_from_mpi(MPI_COMM_WORLD, config);

// send, recv, and stream belong to config.device.
nano_nccl::AllReduceArgs args{
    send,
    recv,
    stream,
    count,
    nano_nccl::DType::Float,
    nano_nccl::RedOp::Sum,
};
communicator->all_reduce(args);  // Enqueues work; does not synchronize.
cudaStreamSynchronize(stream);
communicator->check_async_error();

communicator.reset();
MPI_Finalize();
```

The three typed descriptors have these count and layout semantics:

| Operation | Count field | Local input | Local output |
|---|---|---|---|
| `all_reduce` | `AllReduceArgs::count` | `count` | `count` |
| `reduce_scatter` | `ReduceScatterArgs::recv_count` | `recv_count * global_rank_count` | `recv_count` |
| `all_gather` | `AllGatherArgs::send_count` | `send_count` | `send_count * global_rank_count` |

The caller owns both buffers and the non-null CUDA stream. Input and output
byte ranges must not be identical or partially overlap. Destroy the
communicator before releasing its buffers/stream and before `MPI_Finalize`.

## C ABI v2 and the nano-megatron adapter

`nano_nccl/nano_nccl.h` provides an exception-safe C ABI. Its MPI factory is in
`nano_nccl/mpi_c_api.h` and exported by `libnano_nccl_mpi_c.so`. ABI v2 changes
collective descriptors from local-rank arrays to one scalar pointer/stream and
removes the old in-process multi-GPU `nano_nccl_create_communicator` factory.

```c
#include "nano_nccl/mpi_c_api.h"

nano_nccl_mpi_subgroup_config_t config = {
    .color = dp_group_id,
    .key = dp_rank,
    .device = local_rank,  // Use 0 instead when CUDA_VISIBLE_DEVICES is masked.
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

The adapter's `color` selects a subgroup of `MPI_COMM_WORLD`; `key` determines
its Ring rank order. Every subgroup must contain the build-time
`NANO_NCCL_NRANKS` processes. If the embedding application has not initialized
MPI, the adapter initializes it with `MPI_THREAD_FUNNELED` and releases its
owned MPI state after the last adapter-created communicator is destroyed.

The nano-megatron ctypes adapter must move with ABI v2: expect version `2`, use
direct `c_void_p` fields for each collective buffer/stream, and stop building
length-one pointer arrays. An ABI v1 consumer cannot load an ABI v2 library.

## Cross-host Socket/RDMA

The Socket listener is IPv4-only. Each process sets
`NANO_NCCL_SOCKET_IFNAME=<interface>`. It has no TLS, authentication, or
automatic reconnect and must only be used on a trusted private network.

RDMA build:

```bash
cmake -S . -B build-rdma -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON \
  -DNANO_NCCL_NRANKS=<global-rank-count> \
  -DNANO_NCCL_CUDA_ARCH=<cuda-arch>
cmake --build build-rdma -j$(nproc)
```

Each process also sets `NANO_NCCL_RDMA_IFNAME=<rdma-interface>` and, when
needed, `NANO_NCCL_RDMA_GID_INDEX=<gid-index>`. The default data plane is a
registered host-pinned FIFO with RC SEND/RECV.
`NANO_NCCL_RDMA_USE_WRITE=1` selects WRITE+CTS, and
`NANO_NCCL_RDMA_GDR=1` explicitly selects GPU memory registration. Unsupported
GDR setup fails rather than falling back. `rdma` still resolves same-host edges
with the P2P/SHM auto rule and uses RDMA only for cross-host edges, so the
aggregate result is commonly `mixed`.

## Implementation ownership

- `src/transport/simple/` owns the Simple FIFO layout, step ordering, and slice geometry,
  including distinct send and recv base steps.
- `src/transport/shm/` owns mapped host FIFO storage and control only.
- `src/transport/p2p/` owns CUDA IPC allocation, handle exchange, mapping, and
  capability checks.
- `src/collective/all_reduce/ring_simple_geometry.h` owns Ring channel/rank work partitioning.

Transport runtime lifecycle and orchestration remain in `Communicator::Impl`.

## Current limitations

- Ring + Simple only; no Tree, CollNet, NVLS, LL, or LL128.
- Out-of-place only; no in-place, arbitrary overlap, broadcast, `double`,
  `int8`, or `prod`.
- `nranks` is still fixed by one build; the generated 2/4/8 specialization
  registry is not implemented yet.
- Socket/RDMA one-process-per-GPU matrices and NCCL-relative performance still
  require revalidation.
- CUDA IPC teardown uses a bounded peer-close acknowledgement. If a peer does
  not close its imported handle within the timeout, the exporter intentionally
  leaks that device region instead of freeing memory still mapped by the peer;
  abnormal process-failure recovery remains constrained by the MPI runtime.
