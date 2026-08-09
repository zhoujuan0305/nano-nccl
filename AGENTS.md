# AGENTS.md

## Project Positioning

This project is a GPU collective communication library targeting NCCL-equivalent All Reduce performance.

Current capabilities:

- Single-node multi-GPU performance path (tested with `CUDA_VISIBLE_DEVICES=0,1,2,3`; rank count configurable via CMake `NANO_NCCL_NRANKS`)
- Optional MPI/socket and MPI/RDMA multi-host, out-of-place `all_reduce` correctness paths; Open MPI 4.1.2 and the same MPI ABI are required on all hosts. MPI targets use the MPI C API and do not require Open MPI's retired C++ binding library.
- `float`, FP16, and BF16 dtypes with `sum`, `avg`, `max`, and `min` reduce ops, out-of-place; `avg` is `sum / nranks`, `max`/`min` propagate NaN, and BF16 requires SM80+
- Ring + Simple protocol, with SHM FIFO, device P2P FIFO, optional MPI/socket transport, and optional MPI/RDMA transport
- MPI/RDMA correctness is validated across `float`, FP16, and BF16; `sum`, `avg`, `max`, and `min`; and 256 KiB through 64 MiB. The BF16 device-capability validation is cached after its first successful use; the full A6000 single-host performance gate is pending revalidation (see Acceptance section below)

Future expansion axes:

- **dtype**: `float`/FP16/BF16 → `double`/`int8`
- **reduce op**: `sum`/`avg`/`max`/`min` → `prod`
- **rank count**: 4 → 2/8/16 (runtime parameter, no template specialization needed)
- **collective**: `all_reduce` → `all_gather`/`reduce_scatter`/`broadcast`
- **transport**: SHM FIFO/P2P FIFO → network

`all_gather` and `reduce_scatter` are unsupported. There is no multi-host performance gate. Socket has no TLS or automatic reconnect; use it only on a trusted network. RDMA defaults to RC send/receive over registered host-pinned FIFO memory; `NANO_NCCL_RDMA_USE_WRITE=1` enables WRITE+CTS. Neither path uses GPUDirect RDMA (compare with `NCCL_NET_GDR_LEVEL=0`); the host proxy multi-flights WQEs up to Simple FIFO slice depth. Both hosts must use the same commit (`RdmaPeerInfo` is 64 bytes). Do not claim general NCCL replacement capability until expansion is complete.

## Sensitive Information

Do not write hostnames, IP addresses, physical interface names, MAC addresses, GPU UUIDs, absolute user home paths, credentials, or tokens to files that can be committed to Git. Use placeholders such as `<host-a>`, `<interface>`, and `<path-to-nccl-lib>` in documentation, examples, tests, and logs.

## Directory Structure

```
nano-nccl/
├── CMakeLists.txt
├── AGENTS.md
├── include/nano_nccl/
│   ├── types.h                        # dtype/redop/config enums and structs
│   ├── traits.h                       # dtype traits (pack/unpack), redop traits
│   ├── communicator.h                 # public communicator and collective API
│   ├── mpi.h                          # optional MPI communicator factory
│   └── all_reduce.h                   # public benchmark API
├── src/
│   ├── CMakeLists.txt
│   ├── core/
│   │   ├── buffer.h / buffer.cu       # DeviceBuffer/MappedBuffer/RegisteredMappedBuffer
│   │   ├── numa.h / numa.cu           # NUMA mapping
│   │   └── stream.h                   # Stream/Event/GraphExec RAII wrappers
│   ├── transport/
│   │   ├── transport.h                # Transport interface (seam for network)
│   │   ├── simple_protocol.h          # Shared Simple protocol constants
│   │   ├── p2p/
│   │   │   ├── p2p_fifo.h / .cu       # Device FIFO storage
│   │   │   ├── p2p_step_counters.h/.cu # Device P2P step counters
│   │   │   └── p2p_topology.h / .cu   # Ring peer-access checks
│   │   ├── shm/
│   │       ├── shm_fifo.h / .cu       # SHM FIFO buffer management
│   │       └── shm_step.h             # step counter (wait/post)
│   │   ├── socket/
│   │       ├── socket_endpoint.h / .cc # IPv4 endpoint and HELLO exchange
│   │       ├── socket_protocol.h       # framed socket slice protocol
│   │       └── socket_proxy.h / .cc    # host send/receive proxy threads
│   │   └── rdma/
│   │       ├── rdma_endpoint.h / .cc   # HCA selection, PD/CQ allocation
│   │       ├── rdma_qp.h / .cc         # RC QP lifecycle
│   │       ├── rdma_protocol.h          # QP bootstrap peer information
│   │       └── rdma_proxy.h / .cc       # host send/receive proxy threads
│   ├── collective/
│   │   ├── collective.h               # Collective interface (seam for all_gather etc.)
│   │   └── all_reduce/
│   │       ├── communicator.cu         # communicator orchestration and transports
│   │       ├── communicator_internal.h  # factory and owned socket connections
│   │       ├── mpi_communicator.cc     # MPI bootstrap and socket connection exchange
│   │       ├── topology.h / .cc         # local/global rank and edge topology
│   │       └── ring_simple.h / .cu      # single-host benchmark implementation
│   └── kernels/
│       └── ring_simple_kernel.cuh     # device kernel template
├── benchmarks/
│   ├── CMakeLists.txt
│   └── all_reduce_bench.cu            # correctness + perf integrated benchmark
├── tests/
│   ├── CMakeLists.txt
│   ├── communicator_cleanup_static.py # communicator cleanup static regression check
│   ├── communicator_bf16_validation_static.py # BF16 validation-cache static regression check
│   ├── smoke.cu                       # CUDA device/P2P smoke test
│   ├── correctness.cu                 # single-host all_reduce correctness
│   ├── public_api.cu                   # public communicator API coverage
│   ├── p2p_step_counters.cu            # P2P step-counter coverage
│   ├── p2p_topology.cu                 # P2P topology coverage
│   ├── mpi_bootstrap.cu               # MPI bootstrap and socket smoke coverage
│   ├── mpi_correctness.cu             # MPI/socket correctness and fault injection
│   ├── socket_protocol_test.cc         # socket framing and proxy behavior
│   └── simple_protocol.cu              # Simple protocol layout invariants
└── LICENSE
```

Directory responsibilities:

- `include/nano_nccl/`: public headers; `mpi.h` is available only in an MPI build
- `src/core/`: infrastructure, collective-agnostic (buffer, NUMA, stream)
- `src/transport/`: transport abstraction, subdirectories per transport type
- `src/collective/`: collective operation abstraction, subdirectories per collective type
- `src/kernels/`: device kernel template headers (`.cuh`), referenced and instantiated by collective implementations
- `benchmarks/`: perf + correctness integrated benchmark
- `tests/`: single-host, MPI/socket, transport, and protocol regression coverage

## Naming Conventions

| Category | Style | Examples |
|---|---|---|
| Class/struct | PascalCase | `MappedBuffer`, `AllReduceRunner` |
| Function | snake_case | `wait_send_credit`, `run_ring_simple` |
| Constants | kPascalCase | `kRanks`, `kChannels`, `kSimpleFifoSteps` |
| Namespace | snake_case, layered | `nano_nccl::core`, `nano_nccl::transport::shm` |
| File | snake_case | `buffer.h`, `ring_simple.cu` |
| Algorithm name | snake_case | `ring_simple` |
| Kernel template | snake_case + template params | `ring_simple_kernel<T, RedOp>` |

Namespace layering maps 1:1 to directory structure:

- `nano_nccl::core` — buffer, numa, stream
- `nano_nccl::transport::p2p` — device FIFO, P2P ring validation
- `nano_nccl::transport::shm` — SHM FIFO, step counter
- `nano_nccl::transport::socket` — IPv4 endpoints and socket proxy threads
- `nano_nccl::transport::rdma` — RoCE/IB endpoint, RC QP, and send/recv proxy threads
- `nano_nccl::collective::all_reduce` — all_reduce host-side orchestration
- `nano_nccl::kernels` — device kernel templates

## Coding Standards

- **Indentation**: 4 spaces, no tabs
- **Error handling**: `CUDA_CHECK_THROW` macro + `throw std::runtime_error`; all CUDA API calls must check errors
- **Header guard**: `#pragma once`
- **Include order**: self → C++ standard library → CUDA → system
- **Line width**: no hard limit
- **Comments**:
  - Write comments when code logic is not intuitive
  - Comments explain "why", not "what"
  - Comments go on the line above the code
  - Write comments where things are easy to misunderstand
  - Do not add comments to obvious code

## Architecture

**Hybrid polymorphism: device templates + host runtime**

- **Device kernel** is templated: `template<typename T, typename RedOp> __global__ void ring_simple_kernel(...)` with `int nranks` as a runtime argument. dtype and reduce op are compile-time parameters; rank count is runtime (benchmarked lossless vs. template specialization on 4× A6000: geomean busbw ratio 1.003/0.997/0.998 for float/fp16/bf16 across 256 KiB – 64 MiB).
- **Host side** uses runtime parameters to select algo/transport/collective, does not require compiling all combinations.
- **Transport**: SHM FIFO, device P2P FIFO, MPI/socket, or optional MPI/RDMA for cross-process ring edges. SHM GPUs read/write mapped host memory directly over PCIe (`cudaHostAllocMapped`), with no proxy thread or `cudaMemcpy`; FIFO buffers are allocated on the receiver NUMA node to avoid cross-NUMA bandwidth loss. P2P FIFO buffers are allocated on the receiver GPU and require bidirectional CUDA peer access between every ring-neighbor pair. Socket uses an IPv4 listener chosen by `NANO_NCCL_SOCKET_IFNAME`; it is a trusted-network transport without TLS or auto reconnect. RDMA defaults to RC send/recv over a host-pinned, registered FIFO; set `NANO_NCCL_RDMA_USE_WRITE=1` for the WRITE+CTS host-pin data plane. The proxy multi-flights WQEs up to Simple FIFO slice depth and does not use GPUDirect RDMA (compare with `NCCL_NET_GDR_LEVEL=0`); it is selected only by explicit `--transport rdma` and requires the MPI/RDMA build plus `NANO_NCCL_RDMA_IFNAME`. Both hosts must build the same commit (`RdmaPeerInfo` is 64 bytes).

### Transport selection

The benchmark `--transport` option accepts `auto`, `shm`, `p2p`, and `rdma`. `auto`
(the default) resolves each ring edge independently: it selects P2P for an edge
only when that edge has an active, direct NVLink and CUDA peer access in both
directions (`rank i -> rank (i + 1) % nranks` and the reverse); all other edges
use SHM. Its aggregate transport is `shm`, `p2p`, or `mixed` according to the
resolved edge plan. `shm` always selects SHM. Explicit `p2p` validates those
directions and fails on the first unavailable direction; it does not fall back.
P2P is single-node only and is not a network transport.

`rdma` requires `NANO_NCCL_ENABLE_MPI=ON` and `NANO_NCCL_ENABLE_RDMA=ON` at
build time. It resolves cross-process ring edges to RDMA and resolves local
edges like `auto` (P2P when bidirectional NVLink peer access is available,
otherwise SHM). Aggregate transport is still `mixed` when local and RDMA edges
coexist. The default data plane is RC SEND/RECV; `NANO_NCCL_RDMA_USE_WRITE=1`
enables WRITE+CTS (RC `WRITE_WITH_IMM` plus a host-pinned CTS slot ring). Both
planes use registered host-pinned FIFO memory only — no GPUDirect RDMA; fair
NCCL comparisons use `NCCL_NET_GDR_LEVEL=0`. The host proxy multi-flights WQEs
up to Simple FIFO slice depth. SEND and WriteCts always post SGE from the
registered mapped FIFO; correctness relies on publisher `fence.acq_rel.sys`
plus relaxed `send_tail` store after the block sync (NCCL Simple postPeer
pairing; no host bounce path). Dual-host WRITE matrix
(float/fp16/bf16 × sum/avg/max/min, 256 KiB–64 MiB) validated `#wrong=0`.
Both hosts must build the same commit (`RdmaPeerInfo` is 64 bytes). Set
`NANO_NCCL_RDMA_IFNAME=<rdma-interface>` in every MPI process; set
`NANO_NCCL_RDMA_GID_INDEX` when the default GID table entry is not routable for
the target RoCE deployment. QP information and receive-ready coordination reuse
the trusted TCP bootstrap connection. There is no multi-host performance gate
for this path.

For MPI/socket launches, set `NANO_NCCL_SOCKET_IFNAME` to an interface with exactly one usable IPv4 address in every MPMD app context. `NANO_NCCL_SOCKET_TEST_FAULT_INJECTION=ON` builds a separate test-only library for `nano_nccl_mpi_correctness`; ordinary MPI benchmarks always link the library without the `NANO_NCCL_SOCKET_FAIL_AFTER_SLICES` hook.

**Key design points**:

- Step counter persists across iterations (matching NCCL `conn->step`); `run_batch` uses CUDA events for cross-stream barriers instead of per-iteration `cudaStreamSynchronize`, aligning with NCCL `BenchTime` timing methodology.
- `kSimpleFifoSliceSteps = 2` (two slices per chunk), matching the active Simple protocol constants.
- Wait cache (`send_head_cache`/`recv_tail_cache`) matches NCCL `connStepCache`, avoiding reloading step counter from host memory on every wait.
- Optional benchmark profiling is guarded by `NANO_NCCL_ENABLE_BENCH_PROFILING`; when `OFF`, the default performance path compiles no NVTX or CUDA profiler calls and has no iteration-time profiling condition. When `ON`, every size starts capture after warmup, creates an outer `all_reduce size=<bytes>B` range plus `all_reduce size=<bytes>B iteration=<iteration>` per-iteration ranges, synchronizes, stops capture, then validates. CUDA 12.8 warns that NVTX 2 `<nvToolsExt.h>` is deprecated; the supported range API remains in use.
- Optional `NANO_NCCL_RDMA_SHARED_PROGRESS` (default dedicated/`0`; `1` enables one shared host progress thread for all local RDMA proxies).
Optional `NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE` (default `OFF`) can record per-event RDMA host-proxy metrics including tail-vs-CTS arm times and inter-post spacing; investigation-only, not for production perf claims.

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_NRANKS=<your_gpu_count> \
  -DNANO_NCCL_CUDA_ARCH=<your_cuda_arch>
make -j$(nproc)
```

Optional benchmark profiling build (CUDA 12.8):

```bash
cmake -S . -B build-profile -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=86 \
  -DNANO_NCCL_ENABLE_BENCH_PROFILING=ON
cmake --build build-profile -j$(nproc)
```

Optional two-host MPI/socket build (both hosts must build the same commit and use the same Open MPI 4.1.2 ABI):

```bash
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_NRANKS=8 -DNANO_NCCL_CUDA_ARCH=86
cmake --build build-mpi -j$(nproc)
```

Build artifacts:

- `build/benchmarks/nano_nccl_all_reduce_bench` — perf + correctness benchmark
- `build/tests/nano_nccl_correctness` — correctness-only test
- `build/tests/nano_nccl_smoke` — smoke test
- `build/tests/nano_nccl_public_api` — public C++ API coverage
- `build/tests/nano_nccl_p2p_step_counters` — P2P step-counter coverage
- `build/tests/nano_nccl_p2p_topology` — P2P topology coverage
- `build/tests/nano_nccl_simple_protocol` — Simple protocol layout coverage
- `build-mpi/tests/nano_nccl_mpi_correctness` — MPI/socket correctness and test-only fault injection
- `build-mpi/tests/nano_nccl_mpi_bootstrap` — MPI bootstrap smoke test
- `build-mpi/tests/nano_nccl_socket_protocol` — socket framing and proxy behavior

When `BUILD_TESTING` is enabled (the default), `ctest --test-dir build
--output-on-failure` also runs the static BF16 capability-validation regression
and benchmark profiling static regressions.

## Acceptance

Current status: **REVALIDATION PENDING** (2026-07-16)

Candidate path: `ring_simple` (Ring + Simple protocol, `--transport auto`)

Use `-w 5 -n 20` for all future performance measurements and NCCL comparisons.

Only default-OFF binaries may provide candidate performance results; profiling-enabled binaries are observability-only.

The prior full `PASS` table is retired: its BF16 figures conflicted with the
README and predated the BF16 validation-cache fix. A same-round BF16 comparison
(out-of-place busbw) on 4× NVIDIA RTX A6000 (Ampere sm_86), CUDA 12.8, was
rerun on 2026-07-16. `auto` resolved to a mixed P2P/SHM edge plan. All points
had `#wrong=0`; however, the 4 MiB and 16 MiB points were below the NCCL
baseline, so this result restores BF16 performance but does not satisfy the
full per-size acceptance gate.

| dtype | 256 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1.066 | 1.028 | 0.971 | 0.987 | 1.025 | 1.015 |

Re-verification commands:

```bash
# Candidate
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --transport auto --dtype <float|fp16|bf16> \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20

# NCCL baseline (same-round, requires nccl-tests installed and NCCL library path)
cd <nccl-tests-build-dir>
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_LIBRARY_PATH=<nccl-lib-path> \
NCCL_ALGO=Ring NCCL_PROTO=Simple \
NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=33554432 \
./build/all_reduce_perf -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 -d <float|half|bfloat16>
```

Pass criterion: for each contract message size `s`, `candidate_busbw(s) >= nccl_busbw(s)`.

This is a single-host acceptance criterion only; no multi-host socket performance criterion has been established.

## Extension Guide

- **New dtype**: implement pack/unpack trait in `include/nano_nccl/traits.h`, add template instantiation in `ring_simple.cu`; current `float`, FP16, and BF16 support `sum`/`avg`/`max`/`min`, and BF16 requires SM80+
- **New reduce op**: implement RedOp trait in `include/nano_nccl/traits.h`, add template instantiation in `ring_simple.cu`
- **New rank count**: pass the new rank count to `ring_simple_kernel` as runtime `nranks` argument; `kRanks` (from `NANO_NCCL_NRANKS`) still controls host-side buffer sizing and array dimensions
- **New collective**: create subdirectory under `src/collective/`, implement collective interface
- **New transport**: create a subdirectory under `src/transport/` and implement the transport interface (for example, RoCE RC send/recv); RDMA v1 uses registered host FIFO memory, while RDMA write and GPUDirect RDMA remain future work
- **RDMA build**: `cmake -S . -B build-rdma -DCMAKE_BUILD_TYPE=Release -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON -DNANO_NCCL_NRANKS=<ranks> -DNANO_NCCL_CUDA_ARCH=<arch>`; requires `libibverbs-dev` and reuses the TCP bootstrap connection for `RdmaPeerInfo` exchange and receive-ready coordination. Default data plane is SEND/RECV; `NANO_NCCL_RDMA_USE_WRITE=1` enables WRITE+CTS (host-pin only; no GDR)
