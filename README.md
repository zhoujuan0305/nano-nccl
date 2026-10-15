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
edge is accepted even without direct NVLink. Rank-8 cross-host GDR evidence
is limited to the recorded finite-input campaigns; it does not establish full
contract acceptance.

Historical sections in [performance.md](performance.md) predate the
one-process-per-GPU and matched-timing contracts unless explicitly marked as
revalidated. The 2026-10-15 result below is a scoped diagnostic comparison, not
full-scope performance acceptance.

A post-change rank-2 repeatability smoke (`float`/`sum`, `-w 5 -n 20`, three
runs) produced median busbw of 7.60/19.39/26.62/32.62/34.36 GB/s for P2P and
5.57/10.91/13.62/15.08/15.32 GB/s for SHM at 256 KiB/1 MiB/4 MiB/16 MiB/64
MiB, with zero wrong values. These are nano-nccl-only smoke numbers, not a
same-round NCCL comparison or performance acceptance result.

### GDR and channel performance updates

The current code includes host-proxy GDR, shared RDMA progress, vector-aligned
AllReduce channel partitions, and vector-body/scalar-tail copy and reduction.
Float4 workers use an eight-vector load/store batch by default
(`NANO_NCCL_FLOAT4_UNROLL=8`, with `1` available for comparison). The default
block has 288 threads: 256 workers and 32 control threads. These changes bring
the tested channel/vector implementation into `main`; no diagnostic build
flags are needed.

The 2026-10-15 matched `float`/`sum`/out-of-place Ring + Simple AllReduce
comparison against NCCL 2.30.7 used two hosts, physical GPUs 0 and 1 on each,
and four ranks. Channels 1–4, seven sizes, and four pairs per cell produced
224 jobs, all with `wrong=0`. Nano had lower median time in 27/28 cells and
won all four pairs in 25/28; the geometric mean of the 28 cell median
`NCCL time / nano time` ratios was 1.1024.

The measurements were collected from the development implementation before
integration, with U8, 16-byte-aligned channel partitions, 288 threads,
WRITE+CTS, and shared progress. These mechanisms are now part of the code.
GDR and WRITE+CTS still require the runtime opt-ins below. Both implementations
had slow samples, shared host/GPU resources remained in use, and the full
scope and 3% baseline-regression gate remain open. See
[performance.md](performance.md#2026-10-15-two-host-four-rank-gdr) for all
28 cells, the exact configuration, and the integration validation results.

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
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
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
| `NANO_NCCL_BLOCK_THREADS` | 288 | Threads per block; normally 256 workers + 32 control threads |
| `NANO_NCCL_FLOAT4_UNROLL` | 8 | Float4 load/store batch size; accepts 1 or 8 |
| `NANO_NCCL_FIFO_BUFF_BYTES` | 33554432 | FIFO bytes per channel |
| `NANO_NCCL_ENABLE_MPI` | `OFF` | Build MPI bootstrap, interprocess transports, and benchmark |
| `NANO_NCCL_ENABLE_RDMA` | `OFF` | Build RC RDMA; requires MPI and libibverbs |
| `NANO_NCCL_ENABLE_BENCH_PROFILING` | `OFF` | Add NVTX/CUDA profiling to the benchmark; keep off for acceptance runs |

`float` and FP16 require SM70+ because Simple counters use system-scope
ordering. BF16 requires SM80+.

AllReduce channel boundaries are aligned to 16 bytes; only the final channel
may retain a scalar remainder. Aligned slices process full vectors first and
only the remainder scalarly; misaligned addresses keep the scalar fallback.
Use `-DNANO_NCCL_NCHANNELS=1`, `2`, `3`, or `4` in separate build directories
to select the validated channel counts. The boundary change is specific to
AllReduce; ReduceScatter and AllGather keep their existing work partition.
Use a fresh build directory, or set `NANO_NCCL_BLOCK_THREADS=288` and
`NANO_NCCL_FLOAT4_UNROLL=8` explicitly when reusing an older CMake cache.

## Run

The launcher may expose only the process-local GPU. This same-host example
uses Open MPI's local rank to set `CUDA_VISIBLE_DEVICES`, making `--device 0`
implicit:

```bash
mpirun -np 4 --bind-to none bash -c '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  exec ./build/benchmarks/nano_nccl_all_reduce_bench \
    --algo ring_simple --transport auto --dtype float --redop sum \
    -b 262144 -e 67108864 -f 4 -w 5 -n 20 --wait-mode query
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

RDMA progress defaults to one communicator-owned host polling thread per
MPI process, multiplexing that communicator's send and receive proxies across
channels. It does not create one polling thread per direction/channel, and
threads are not shared across MPI processes. Set
`NANO_NCCL_RDMA_SHARED_PROGRESS=0` to compare with dedicated proxy threads.
The shared engine reports failures through the communicator's stable async
error and publishes the GPU abort flag before making that error visible.

GPU memory registration first enables `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`, then
tries DMA-BUF and, if unavailable, device-address `ibv_reg_mr` (requiring a
working peer-memory provider, such as `nvidia-peermem`). Receive visibility
flushes and selective CQEs are preserved. GDR is host-proxy RDMA, not
GPU-initiated IBGDA, and hardware/driver support is still required.

`Communicator::edge_uses_gdr(edge)` (and `nano_nccl_edge_uses_gdr`) reports
whether a directed RDMA edge uses a registered GPU FIFO. The benchmark prints
each directed edge with its transport and RDMA memory placement before the
measurement table. Peers reject mismatched GDR placement or RDMA data-plane
selection during bootstrap. `nano_nccl_rdma_gdr` probes a full 32 MiB FIFO
registration and prints `registration=dmabuf` or `registration=device-verbs`
on success; the latter identifies the verbs call, not a particular kernel
registration mechanism.

For an end-to-end GDR correctness run, build with `NANO_NCCL_NRANKS=2` and
launch one MPI rank on each host with the same binary and environment:

```bash
export NANO_NCCL_SOCKET_IFNAME=<interface>
export NANO_NCCL_RDMA_IFNAME=<rdma-interface>
export NANO_NCCL_RDMA_GDR=1
mpirun -np 2 --host <host-a>:1,<host-b>:1 --bind-to none \
  -x NANO_NCCL_SOCKET_IFNAME -x NANO_NCCL_RDMA_IFNAME \
  -x NANO_NCCL_RDMA_GDR \
  ./build-rdma/tests/nano_nccl_mpi_native_api rdma-gdr
```

The `rdma-gdr` mode checks GPU FIFO placement on cross-host edges and runs
long, repeated float/FP16/BF16 AllReduce, ReduceScatter, and AllGather
correctness cases. Run it once with the default SEND/RECV data plane and once
with `NANO_NCCL_RDMA_USE_WRITE=1` exported to all ranks for WRITE+CTS.

For the two-host, eight-GPU GDR performance comparison, see
`scripts/run_gdr_matched.py --help`. The runner takes hosts, MPI prefix, all
three collective binaries for both implementations, network devices, and an
output directory. It defaults to five alternating repetitions per case,
one MPI process per GPU, nano `--wait-mode query` with WRITE+CTS GDR, and
nccl-tests `-z 2 -a 3 -m 1 -g 1` with bidirectional GDR. It saves raw logs
and checks correctness and actual GDR placement. Earlier performance tables
used a different timing contract and process layout.

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
- Full-scope Socket/RDMA performance and the 3% regression check against an
  accepted nano-nccl baseline remain to be validated; the scoped GDR comparison
  above does not complete either gate.
- CUDA IPC teardown uses a bounded peer-close acknowledgement. If a peer does
  not close its imported handle within the timeout, the exporter intentionally
  leaks that device region instead of freeing memory still mapped by the peer;
  abnormal process-failure recovery remains constrained by the MPI runtime.
