# nano-nccl

[中文说明](README.zh.md)

A GPU collective communication library for single-host multi-GPU All Reduce, targeting NCCL `Ring` + `Simple` + 4 channels performance. Optional MPI/socket and MPI/RDMA paths support multi-host `all_reduce` runs; RDMA is host-pin only (no GPUDirect RDMA); default SEND/RECV, optional WRITE+CTS via `NANO_NCCL_RDMA_USE_WRITE=1`.

---

## Performance

[Detailed single-host performance results](performance.md) record the tested topology, environment, and point-by-point NCCL comparisons for in-process auto (P2P/SHM) plus two-host TCP socket and host-pinned RDMA (`float` / FP16 / BF16 × `sum` / `avg` / `max` / `min`).

---

## Build

Dependencies: CUDA 12+, CMake 3.18+, libnuma-dev. Distributed builds require Open MPI 4.1.2 on every host with the same MPI ABI. RDMA builds additionally require libibverbs-dev.

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_NRANKS=<your_gpu_count> \
  -DNANO_NCCL_CUDA_ARCH=<your_cuda_arch>
make -j$(nproc)
```

For example, for a 4-GPU RTX A6000 (sm_86) system:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=86
```

Build artifacts:

- `build/benchmarks/nano_nccl_all_reduce_bench` — perf + correctness benchmark
- `build/tests/nano_nccl_correctness` — correctness-only test
- `build/tests/nano_nccl_smoke` — smoke test
- `build/tests/nano_nccl_public_api` — public C++ API coverage
- `build/tests/nano_nccl_p2p_step_counters` — P2P step-counter coverage
- `build/tests/nano_nccl_p2p_topology` — P2P topology coverage
- `build/tests/nano_nccl_simple_protocol` — Simple protocol layout coverage
- `build-mpi/tests/nano_nccl_mpi_correctness` — MPI/socket correctness test (MPI build)
- `build-mpi/tests/nano_nccl_mpi_bootstrap` — MPI bootstrap smoke test (MPI build)
- `build-mpi/tests/nano_nccl_socket_protocol` — socket framing and proxy test (MPI build)
- `build-rdma/tests/nano_nccl_rdma_protocol` — RDMA protocol-layout test (MPI/RDMA build)
- `build-rdma/tests/nano_nccl_rdma_bootstrap` — local RC QP bootstrap smoke test (MPI/RDMA build)

When `BUILD_TESTING` is enabled (the default), `ctest --test-dir build
--output-on-failure` also runs the static BF16 capability-validation regression
and benchmark profiling static regressions.

### CMake options

| Option | Default | Description |
|---|---|---|
| `NANO_NCCL_NRANKS` | 4 | Number of GPU ranks |
| `NANO_NCCL_NCHANNELS` | 4 | Number of channels |
| `NANO_NCCL_CUDA_ARCH` | 70 | CUDA compute capability (e.g. 70 for Volta, 75 for Turing, 86 for Ampere); values below 70 are rejected |
| `NANO_NCCL_BLOCK_THREADS` | 512 | Threads per block |
| `NANO_NCCL_FIFO_BUFF_BYTES` | 33554432 | FIFO buffer size in bytes (32 MiB) |
| `NANO_NCCL_ENABLE_MPI` | `OFF` | Build the MPI communicator bootstrap and distributed benchmark/test |
| `NANO_NCCL_ENABLE_RDMA` | `OFF` | Build the RC send/receive RDMA transport; requires `NANO_NCCL_ENABLE_MPI=ON` and libibverbs |
| `NANO_NCCL_SOCKET_TEST_FAULT_INJECTION` | `OFF` | Build a separate test-only fault-injection library for `nano_nccl_mpi_correctness`; ordinary MPI benchmarks never include the hook |
| `NANO_NCCL_ENABLE_BENCH_PROFILING` | `OFF` | Compile NVTX/CUDA-profiler instrumentation into the all-reduce benchmark; keep `OFF` for reported bandwidth |

`float` and FP16 require SM70+ because Simple FIFO counters use system-scope
release/acquire operations. BF16 requires SM80+.

NUMA topology is detected at runtime by reading `/sys/bus/pci/devices/*/numa_node` — no source code changes needed when moving to a different machine.

### Optional MPI/socket build

Build the same commit on each host with the global GPU count. The socket listener is IPv4-only; set `NANO_NCCL_SOCKET_IFNAME` to an interface that resolves to exactly one usable IPv4 address. The socket connection has no TLS, authentication, or automatic reconnect, so use it only on a trusted private network.

```bash
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_NRANKS=8 -DNANO_NCCL_CUDA_ARCH=86
cmake --build build-mpi -j$(nproc)
```

### Optional MPI/RDMA build

Build the same commit on every host with the global GPU count (`RdmaPeerInfo` is
64 bytes; mismatched commits fail bootstrap). RDMA defaults to RC send/receive
over registered host-pinned FIFO memory; the host proxy multi-flights SEND/RECV
up to Simple FIFO slice depth with selective CQ signaling. Empty Simple trailing
slices do not take a network round-trip. Set `NANO_NCCL_RDMA_USE_WRITE=1` to
opt into the WRITE+CTS data plane (RC `WRITE_WITH_IMM` plus a host-pinned CTS
slot ring); unset/`0` keeps SEND/RECV. Both planes stay host-pin only — no
  GPUDirect RDMA. Compare against NCCL with `NCCL_NET_GDR_LEVEL=0`. SEND and
  WriteCts always post SGE from the registered mapped FIFO (`fifo_mr_`).
  After worker FIFO stores and a full-block sync, the publisher advances
  `send_tail` with `st.release.sys`; proxies load it with acquire (no host
  bounce path). Single-host 4-rank WRITE+CTS vs NCCL GDR=0 NET/IB
  (float/fp16/bf16 × sum/avg/max/min, 256 KiB–64 MiB) is in
  [performance.md](performance.md) with `#wrong=0` (dedicated progress
  default). The MPI binding uses the MPI C API,
  so an Open MPI installation need not ship the retired C++ binding library.

```bash
cmake -S . -B build-rdma -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON \
  -DNANO_NCCL_NRANKS=<global_gpu_count> -DNANO_NCCL_CUDA_ARCH=<cuda_arch>
cmake --build build-rdma -j$(nproc)
```

Set `NANO_NCCL_SOCKET_IFNAME=<interface>` and
`NANO_NCCL_RDMA_IFNAME=<rdma-interface>` in every MPI process. Set
`NANO_NCCL_RDMA_GID_INDEX=<gid-index>` when the default GID entry is not
routable. Use `--transport rdma`; cross-process ring edges use RDMA and local
edges resolve like `auto` (P2P when bidirectional NVLink peer access is
available, otherwise SHM), so the reported aggregate transport is normally
`mixed`. For fair NCCL comparisons use `NCCL_NET_GDR_LEVEL=0` (same host-pin
class). Bind Open MPI TCP/OOB to the bootstrap interface (`btl_tcp_if_include` /
`oob_tcp_if_include`) so MPI does not pick a non-routable NIC.

For a two-host, four-GPU-per-host correctness launch:

```bash
mpirun \
  -np 1 --host <host-a> -x NANO_NCCL_SOCKET_IFNAME=<interface> \
    ./build-mpi/tests/nano_nccl_mpi_correctness --dtype float \
  : -np 1 --host <host-b> -x NANO_NCCL_SOCKET_IFNAME=<interface> \
    ./build-mpi/tests/nano_nccl_mpi_correctness --dtype float
```

RDMA bench (one 4-GPU process per host; matching trees on every host):

```bash
mpirun --mca btl_tcp_if_include <interface> --mca oob_tcp_if_include <interface> \
  --host <host-a>:1 -np 1 \
  -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH \
  -x NANO_NCCL_SOCKET_IFNAME -x NANO_NCCL_RDMA_IFNAME -x NANO_NCCL_RDMA_GID_INDEX \
  ./build-rdma/benchmarks/nano_nccl_all_reduce_bench \
    --algo ring_simple --transport rdma --dtype float --redop sum \
    -b 262144 -e 67108864 -f 4 -w 5 -n 20 : \
  --host <host-b>:1 -np 1 \
  -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH \
  -x NANO_NCCL_SOCKET_IFNAME -x NANO_NCCL_RDMA_IFNAME -x NANO_NCCL_RDMA_GID_INDEX \
  ./build-rdma/benchmarks/nano_nccl_all_reduce_bench \
    --algo ring_simple --transport rdma --dtype float --redop sum \
    -b 262144 -e 67108864 -f 4 -w 5 -n 20
```

---

## Usage

```bash
# Benchmark (perf + correctness)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype float --redop sum --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20

# FP16 and BF16 correctness/performance runs (BF16 requires SM80+)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype fp16 --redop max --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype bf16 --redop avg --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20

# Correctness-only test
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_correctness

# Smoke test
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_smoke
```

`--redop` accepts `sum` (the default), `avg`, `max`, and `min`. `avg` is the
element-wise `sum / nranks`. `max` and `min` propagate NaN when either operand
is NaN. The selected reduction operation is compiled into the device kernel;
the rank count remains a runtime kernel parameter.

### Optional NVTX/CUDA profiling

Build a separate profiling binary; do not use this build for performance comparisons:

```bash
cmake -S . -B build-profile -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=86 \
  -DNANO_NCCL_ENABLE_BENCH_PROFILING=ON
cmake --build build-profile -j$(nproc)
nsys profile --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --output=bench-nvtx-profile \
  ./build-profile/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --transport auto --dtype float -b 262144 -e 262144 -f 2 -w 1 -n 2
nsys stats --report nvtx_pushpop_sum bench-nvtx-profile.nsys-rep
```

For every message size, the capture contains an outer `all_reduce size=<bytes>B` range and one `all_reduce size=<bytes>B iteration=<iteration>` range per measured iteration. Warmup is outside capture. CUDA 12.8 emits an NVTX 2 deprecation notice for `<nvToolsExt.h>`; it does not invalidate the capture.

## Public C++ API

`nano_nccl/communicator.h` exposes a move-only `Communicator` for one process
that owns all configured local GPUs. The caller owns the device buffers and
CUDA streams. Buffer and stream arrays must have one entry per device, in the
same order as `CommunicatorConfig::devices`.

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

// Allocate one out-of-place send/receive pair and one stream on each device.
// send_buffers[i], recv_buffers[i], and streams[i] must belong to devices[i].
// ... cudaSetDevice(devices[i]), cudaMalloc, cudaStreamCreateWithFlags ...

constexpr std::size_t count = 1 << 20;  // Elements per local rank.
nano_nccl::CollectiveArgs args{
    send_buffers,
    recv_buffers,
    streams,
    count,
    nano_nccl::DType::Float,
    nano_nccl::RedOp::Sum,
};

communicator->all_reduce(args);  // Enqueues work; it does not synchronize.

for (std::size_t i = 0; i < devices.size(); ++i) {
    cudaSetDevice(devices[i]);
    cudaStreamSynchronize(streams[i]);
}
communicator->check_async_error();
```

The single-host adapter requires `devices` to be the visible-device sequence
`{0, ..., NANO_NCCL_NRANKS - 1}`. In an MPI build, `nano_nccl/mpi.h` provides
`create_communicator_from_mpi(MPI_COMM_WORLD, config)` for a distributed
communicator. `all_reduce` is out-of-place and supports `float`, FP16, BF16,
and `sum`, `avg`, `max`, and `min`. `avg` is `sum / nranks`; `max` and `min`
propagate NaN. `reduce_scatter` and `all_gather` are present in the public interface
but throw an unsupported-operation error.

### Transport selection

`--transport` accepts `auto`, `shm`, and `p2p` on a single host. Distributed
MPI communicators accept `auto` and `rdma`; `auto` resolves cross-process ring
edges to socket, while explicit `rdma` selects RDMA for those edges only when
built with MPI/RDMA support.

- `auto` (the default) selects P2P independently for each ring edge only when
  it has a direct NVLink and CUDA peer access in both directions; other edges
  use SHM. The resulting transport is `shm`, `p2p`, or `mixed`.
- `shm` forces the mapped-host-memory SHM FIFO path.
- `p2p` requires that every ring edge has the required bidirectional peer
  access and fails during setup on the first unavailable direction.
- `rdma` requires `NANO_NCCL_ENABLE_MPI=ON` and `NANO_NCCL_ENABLE_RDMA=ON`.
  It uses multi-flight RC RDMA (up to Simple FIFO slice depth, selective CQ
  signal, empty-slice elision) for cross-process ring edges and resolves local
  edges like `auto` (P2P when bidirectional NVLink peer access is available,
  otherwise SHM). Default data plane is SEND/RECV; `NANO_NCCL_RDMA_USE_WRITE=1`
  selects WRITE+CTS (still host-pin; fair NCCL baseline is
  `NCCL_NET_GDR_LEVEL=0`). Both hosts must build the same commit (64-byte
  `RdmaPeerInfo`). SEND/WriteCts post from the registered mapped FIFO; after
  worker stores and block sync, the publisher uses `st.release.sys(send_tail)`
  (host acquire loads; no host bounce). Small Simple slices may post recv
  credit immediately after consuming the recv FIFO. Cross-process RDMA edges
  default to dedicated per-proxy host threads; set
  `NANO_NCCL_RDMA_SHARED_PROGRESS=1` for one shared progress thread. Single-host
  4-rank WRITE+CTS vs NCCL GDR=0 NET/IB is in [performance.md](performance.md).

P2P is a single-node transport. It requires CUDA peer access for the complete
configured ring; it is not a multi-node or network transport. Socket uses a
trusted, IPv4-only TCP network boundary and has no TLS or auto reconnect.

### Implementation ownership

- `src/transport/simple/` owns the Simple FIFO layout, step ordering, and slice geometry.
- `src/transport/shm/` owns mapped host FIFO storage and control only.
- `src/collective/all_reduce/ring_simple_geometry.h` owns Ring channel/rank work partitioning.

Persistent progress uses distinct send and recv base steps for every rank/channel.
The kernel initializes and persists each direction independently so elided empty
slices do not couple incoming and outgoing progress.

Transport runtime lifecycle and orchestration remain in `Communicator::Impl`;
this module split does not move them into `src/transport/simple/`. Make Simple
protocol changes in `src/transport/simple/` and Ring scheduling changes in
`ring_simple_geometry.h`.

---

## Limitations

Currently supports only:

- Single-node multi-GPU performance path (tested with `CUDA_VISIBLE_DEVICES=0,1,2,3`); published tables in [performance.md](performance.md) cover in-process auto, 4-rank socket, and 4-rank host-pinned RDMA
- `float` and FP16 (`fp16`) on SM70+, and BF16 (`bf16`) on SM80+
- `sum`, `avg`, `max`, and `min` reduce ops; `avg` is `sum / nranks`, and `max`/`min` propagate NaN
- out-of-place
- SHM FIFO and device P2P FIFO transports, plus optional MPI/socket or MPI/RDMA for cross-process ring edges; P2P is single-node only; RDMA is host-pin RC (no GPUDirect RDMA)

Published performance tables are single-host only. Socket is loopback-TCP; RDMA is measured against NCCL `NCCL_NET_GDR_LEVEL=0` with P2P/SHM disabled. This project is not a general NCCL replacement.

Future expansion plans:

- dtype: `double` / `int8`
- reduce op: `prod`
- rank count: 2 / 8 / 16 (runtime kernel parameter)
- collective: `all_gather` / `reduce_scatter` / `broadcast`

---

## License

[MIT](LICENSE)
