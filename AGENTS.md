# AGENTS.md

## Project Positioning

This project is a GPU collective communication library targeting NCCL-equivalent All Reduce performance.

Current capabilities:

- Single-node multi-GPU (tested with `CUDA_VISIBLE_DEVICES=0,1,2,3`; rank count configurable via CMake `NANO_NCCL_NRANKS`)
- `float` dtype, `sum` reduce op, out-of-place
- Ring + Simple protocol, SHM FIFO transport
- Performance parity with NCCL `Ring` + `Simple` + 4 channels baseline achieved (see Acceptance section below)

Future expansion axes:

- **dtype**: `float` → `half`/`double`/`int8`
- **reduce op**: `sum` → `max`/`min`/`prod`
- **rank count**: 4 → 2/8/16 (templated, host-side dispatch)
- **collective**: `all_reduce` → `all_gather`/`reduce_scatter`/`broadcast`
- **transport**: SHM FIFO → P2P/NVLink/network

Only the above scenario is currently supported. Do not claim general NCCL replacement capability until expansion is complete.

## Directory Structure

```
nano-nccl/
├── CMakeLists.txt
├── AGENTS.md
├── include/nano_nccl/
│   ├── types.h                        # dtype/redop/config enums and structs
│   ├── traits.h                       # dtype traits (pack/unpack), redop traits
│   └── all_reduce.h                   # public API
├── src/
│   ├── CMakeLists.txt
│   ├── core/
│   │   ├── buffer.h / buffer.cu       # DeviceBuffer/MappedBuffer/RegisteredMappedBuffer
│   │   ├── numa.h / numa.cu           # NUMA mapping
│   │   └── stream.h                   # Stream/Event/GraphExec RAII wrappers
│   ├── transport/
│   │   ├── transport.h                # Transport interface (seam for P2P/network)
│   │   └── shm/
│   │       ├── shm_fifo.h / .cu       # SHM FIFO buffer management
│   │       └── shm_step.h             # step counter (wait/post)
│   ├── collective/
│   │   ├── collective.h               # Collective interface (seam for all_gather etc.)
│   │   └── all_reduce/
│   │       ├── ring_simple.h          # ring_simple entry declaration
│   │       └── ring_simple.cu         # host-side launcher + kernel template instantiation
│   └── kernels/
│       └── ring_simple_kernel.cuh     # device kernel template
├── benchmarks/
│   ├── CMakeLists.txt
│   └── all_reduce_bench.cu            # correctness + perf integrated benchmark
├── tests/
│   ├── CMakeLists.txt
│   ├── smoke.cu                       # CUDA device/P2P smoke test
│   └── correctness.cu                 # all_reduce correctness-only test
└── LICENSE
```

Directory responsibilities:

- `include/nano_nccl/`: public headers, external users only include from here
- `src/core/`: infrastructure, collective-agnostic (buffer, NUMA, stream)
- `src/transport/`: transport abstraction, subdirectories per transport type
- `src/collective/`: collective operation abstraction, subdirectories per collective type
- `src/kernels/`: device kernel template headers (`.cuh`), referenced and instantiated by collective implementations
- `benchmarks/`: perf + correctness integrated benchmark
- `tests/`: standalone correctness test and smoke test

## Naming Conventions

| Category | Style | Examples |
|---|---|---|
| Class/struct | PascalCase | `MappedBuffer`, `AllReduceRunner` |
| Function | snake_case | `wait_send_credit`, `run_ring_simple` |
| Constants | kPascalCase | `kRanks`, `kChannels`, `kSimpleFifoSteps` |
| Namespace | snake_case, layered | `nano_nccl::core`, `nano_nccl::transport::shm` |
| File | snake_case | `buffer.h`, `ring_simple.cu` |
| Algorithm name | snake_case | `ring_simple` |
| Kernel template | snake_case + template params | `ring_simple_kernel<NRanks, T, RedOp>` |

Namespace layering maps 1:1 to directory structure:

- `nano_nccl::core` — buffer, numa, stream
- `nano_nccl::transport::shm` — SHM FIFO, step counter
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

- **Device kernel** is templated: `template<int NRanks, typename T, typename RedOp> __global__ void ring_simple_kernel(...)`. dtype, reduce op, and rank count are compile-time parameters, enabling loop unrolling and zero virtual-call overhead.
- **Host side** uses runtime parameters to select algo/transport/collective, does not require compiling all combinations.
- **Transport**: SHM FIFO. GPUs read/write mapped host memory directly over PCIe (`cudaHostAllocMapped`), no proxy thread, no cudaMemcpy. FIFO buffers allocated on receiver NUMA node to avoid cross-NUMA bandwidth loss.

**Key design points**:

- Step counter persists across iterations (matching NCCL `conn->step`); `run_batch` uses CUDA events for cross-stream barriers instead of per-iteration `cudaStreamSynchronize`, aligning with NCCL `BenchTime` timing methodology.
- `kSimpleFifoSliceSteps = 4` (SlicePerChunk = 1), eliminating empty-slice barrier and polling overhead.
- Wait cache (`send_head_cache`/`recv_tail_cache`) matches NCCL `connStepCache`, avoiding reloading step counter from host memory on every wait.

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_NRANKS=<your_gpu_count> \
  -DNANO_NCCL_CUDA_ARCH=<your_cuda_arch>
make -j$(nproc)
```

Build artifacts:

- `build/benchmarks/nano_nccl_all_reduce_bench` — perf + correctness benchmark
- `build/tests/nano_nccl_correctness` — correctness-only test
- `build/tests/nano_nccl_smoke` — smoke test

## Acceptance

Current status: **PASS** (2026-07-05, re-verified after refactoring)

Candidate path: `ring_simple` (Ring + Simple protocol, SHM FIFO transport)

Same-round comparison (out-of-place busbw):

| size(bytes) | NCCL busbw(GB/s) | candidate busbw(GB/s) | ratio |
| ---: | ---: | ---: | ---: |
| 262144 | 4.46 | 4.88 | 1.094 |
| 1048576 | 7.12 | 7.78 | 1.093 |
| 4194304 | 8.51 | 8.85 | 1.040 |
| 16777216 | 8.59 | 8.85 | 1.030 |
| 67108864 | 8.71 | 8.86 | 1.017 |

geomean(ratio) = 1.054 ≥ 1.00 ✓

Environment: 4× GTX 1080 Ti (Pascal sm_61), CUDA 12.4, driver 550.127.05, no NVLink, GPU0/1 NUMA 0, GPU2/3 NUMA 1.

Re-verification commands:

```bash
# Candidate
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple -b 262144 -e 67108864 -f 4 -w 2 -n 5

# NCCL baseline (same-round, requires nccl-tests installed and NCCL library path)
cd <nccl-tests-build-dir>
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_LIBRARY_PATH=<nccl-lib-path> \
NCCL_ALGO=Ring NCCL_PROTO=Simple \
NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=33554432 \
./build/all_reduce_perf -b 262144 -e 67108864 -f 4 -g 4 -w 2 -n 5
```

Pass criterion: for each contract message size `s`, `candidate_busbw(s) >= nccl_busbw(s)`.

## Extension Guide

- **New dtype**: implement pack/unpack trait in `include/nano_nccl/traits.h`, add template instantiation in `ring_simple.cu`
- **New reduce op**: implement RedOp trait in `include/nano_nccl/traits.h`, add template instantiation in `ring_simple.cu`
- **New rank count**: add `switch(nranks)` branch in host-side dispatch in `ring_simple.cu`, instantiate kernel for that `NRanks`
- **New collective**: create subdirectory under `src/collective/`, implement collective interface
- **New transport**: create subdirectory under `src/transport/`, implement transport interface
