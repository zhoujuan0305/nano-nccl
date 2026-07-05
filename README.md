# nano-nccl

[中文说明](README.zh.md)

A single-node multi-GPU All Reduce library that matches NCCL `Ring` + `Simple` + 4 channels baseline performance on 4× GTX 1080 Ti (Pascal, no NVLink).

---

## Performance

Compared against NCCL baseline in the same run (out-of-place busbw, `-w 2 -n 5`):

| size (bytes) | NCCL busbw (GB/s) | nano-nccl busbw (GB/s) | ratio |
| ---: | ---: | ---: | ---: |
| 262144 | 4.46 | 4.88 | 1.094 |
| 1048576 | 7.12 | 7.78 | 1.093 |
| 4194304 | 8.51 | 8.85 | 1.040 |
| 16777216 | 8.59 | 8.85 | 1.030 |
| 67108864 | 8.71 | 8.86 | 1.017 |

geomean(ratio) = 1.054, all sizes ≥ 1.00.

**Test environment**: 4× GTX 1080 Ti (Pascal sm_61), CUDA 12.4, driver 550.127.05, no NVLink, GPU0/1 NUMA 0 / GPU2/3 NUMA 1.

> Performance data applies only to the above hardware and parameter configuration.

---

## Build

Dependencies: CUDA 12+, CMake 3.18+, libnuma-dev

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_NRANKS=<your_gpu_count> \
  -DNANO_NCCL_CUDA_ARCH=<your_cuda_arch>
make -j$(nproc)
```

For example, on a 4-GPU Pascal (sm_61) system:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=61
```

On a 2-GPU Turing (sm_75) system:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNANO_NCCL_NRANKS=2 -DNANO_NCCL_CUDA_ARCH=75
```

Build artifacts:

- `build/benchmarks/nano_nccl_all_reduce_bench` — perf + correctness benchmark
- `build/tests/nano_nccl_correctness` — correctness-only test
- `build/tests/nano_nccl_smoke` — smoke test

### CMake options

| Option | Default | Description |
|---|---|---|
| `NANO_NCCL_NRANKS` | 4 | Number of GPU ranks |
| `NANO_NCCL_NCHANNELS` | 4 | Number of channels |
| `NANO_NCCL_CUDA_ARCH` | 61 | CUDA compute capability (e.g. 61 for sm_61, 75 for Turing, 86 for Ampere) |
| `NANO_NCCL_BLOCK_THREADS` | 512 | Threads per block |
| `NANO_NCCL_FIFO_BUFF_BYTES` | 33554432 | FIFO buffer size in bytes (32 MiB) |

NUMA topology is detected at runtime by reading `/sys/bus/pci/devices/*/numa_node` — no source code changes needed when moving to a different machine.

---

## Usage

```bash
# Benchmark (perf + correctness)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple -b 262144 -e 67108864 -f 4 -w 2 -n 5

# Correctness-only test
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_correctness

# Smoke test
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_smoke
```

---

## Limitations

Currently supports only:

- Single-node multi-GPU (tested with `CUDA_VISIBLE_DEVICES=0,1,2,3`; rank count is configurable via `NANO_NCCL_NRANKS`)
- `float` dtype
- `sum` reduce op
- out-of-place

Future expansion plans:

- dtype: `half` / `double` / `int8`
- reduce op: `max` / `min` / `prod`
- rank count: 2 / 8 / 16 (templated, host-side dispatch)
- collective: `all_gather` / `reduce_scatter` / `broadcast`
- transport: P2P / NVLink / network

---

## License

[MIT](LICENSE)
