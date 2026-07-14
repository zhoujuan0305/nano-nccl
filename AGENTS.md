# AGENTS.md

## Project Positioning

This project is a GPU collective communication library targeting NCCL-equivalent All Reduce performance.

Current capabilities:

- Single-node multi-GPU (tested with `CUDA_VISIBLE_DEVICES=0,1,2,3`; rank count configurable via CMake `NANO_NCCL_NRANKS`)
- `float`, FP16, and BF16 dtypes with the `sum` reduce op, out-of-place; BF16 requires SM80+
- Ring + Simple protocol, with SHM FIFO and device P2P FIFO transports
- A6000 same-round results exceed the NCCL `Ring` + `Simple` + 4 channels baseline for float, FP16, and BF16 (see Acceptance section below)

Future expansion axes:

- **dtype**: `float`/FP16/BF16 → `double`/`int8`
- **reduce op**: `sum` → `max`/`min`/`prod`
- **rank count**: 4 → 2/8/16 (templated, host-side dispatch)
- **collective**: `all_reduce` → `all_gather`/`reduce_scatter`/`broadcast`
- **transport**: SHM FIFO/P2P FIFO → network

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
│   │   ├── transport.h                # Transport interface (seam for network)
│   │   ├── simple_protocol.h          # Shared Simple protocol constants
│   │   ├── p2p/
│   │   │   └── p2p_fifo.h / .cu       # Device FIFO and ring peer-access checks
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
- `nano_nccl::transport::p2p` — device FIFO, P2P ring validation
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
- **Transport**: SHM FIFO or device P2P FIFO. SHM GPUs read/write mapped host memory directly over PCIe (`cudaHostAllocMapped`), with no proxy thread or `cudaMemcpy`; FIFO buffers are allocated on the receiver NUMA node to avoid cross-NUMA bandwidth loss. P2P FIFO buffers are allocated on the receiver GPU and require bidirectional CUDA peer access between every ring-neighbor pair.

### Transport selection

The benchmark `--transport` option accepts `auto`, `shm`, and `p2p`. `auto`
(the default) resolves each ring edge independently: it selects P2P for an edge
only when that edge has an active, direct NVLink and CUDA peer access in both
directions (`rank i -> rank (i + 1) % nranks` and the reverse); all other edges
use SHM. Its aggregate transport is `shm`, `p2p`, or `mixed` according to the
resolved edge plan. `shm` always selects SHM. Explicit `p2p` validates those
directions and fails on the first unavailable direction; it does not fall back.
P2P is single-node only and is not a network transport.

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

Current status: **PASS** (2026-07-14)

Candidate path: `ring_simple` (Ring + Simple protocol, `--transport auto`)

Use `-w 5 -n 20` for all future performance measurements and NCCL comparisons.

Same-round comparison (out-of-place busbw) on 4× NVIDIA RTX A6000 (Ampere
sm_86), CUDA 12.8. `auto` resolved to a mixed P2P/SHM edge plan. The values
below are geomean `candidate_busbw / nccl_busbw`; all 15 measured points had
`#wrong=0` and met the per-size gate.

| dtype | 256 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float | 1.335 | 1.241 | 1.038 | 1.007 | 1.019 | 1.120 |
| FP16 | 1.388 | 1.207 | 1.038 | 1.009 | 1.017 | 1.123 |
| BF16 | 1.421 | 1.242 | 1.038 | 1.015 | 1.018 | 1.136 |

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

## Extension Guide

- **New dtype**: implement pack/unpack trait in `include/nano_nccl/traits.h`, add template instantiation in `ring_simple.cu`; current `float`, FP16, and BF16 support `sum`, and BF16 requires SM80+
- **New reduce op**: implement RedOp trait in `include/nano_nccl/traits.h`, add template instantiation in `ring_simple.cu`
- **New rank count**: add `switch(nranks)` branch in host-side dispatch in `ring_simple.cu`, instantiate kernel for that `NRanks`
- **New collective**: create subdirectory under `src/collective/`, implement collective interface
- **New transport**: create a subdirectory under `src/transport/` and implement the transport interface (for example, a network transport)
