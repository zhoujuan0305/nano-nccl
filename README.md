# nano-nccl

[中文说明](README.zh.md)

A single-node multi-GPU All Reduce library targeting NCCL `Ring` + `Simple` + 4 channels performance.

---

## Performance

Same-round comparison against NCCL (out-of-place busbw, `-w 5 -n 20`). nano-nccl used `--transport auto`, which resolved to a mixed P2P/SHM ring-edge plan.

| dtype | 256 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float | 1.335 | 1.241 | 1.038 | 1.007 | 1.019 | 1.120 |
| fp16 | 1.388 | 1.207 | 1.038 | 1.009 | 1.017 | 1.123 |
| bf16 | 1.421 | 1.242 | 1.038 | 1.015 | 1.018 | 1.136 |

Each cell is `nano-nccl busbw / NCCL busbw`; all 15 measured points were correct (`#wrong=0`) and at or above the NCCL baseline.

**Test environment**: 4× NVIDIA RTX A6000 (Ampere sm_86), CUDA 12.8, `CUDA_VISIBLE_DEVICES=0,1,2,3`. NCCL used `Ring` + `Simple`, four channels, and `NCCL_BUFFSIZE=33554432`.

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

For example, on the measured 4-GPU RTX A6000 (sm_86) system:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=86
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
  --algo ring_simple --dtype float --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20

# FP16 and BF16 correctness/performance runs (BF16 requires SM80+)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype fp16 --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype bf16 --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20

# Correctness-only test
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_correctness

# Smoke test
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_smoke
```

### Transport selection

`--transport` accepts `auto`, `shm`, and `p2p`.

- `auto` (the default) selects P2P independently for each ring edge only when
  it has a direct NVLink and CUDA peer access in both directions; other edges
  use SHM. The resulting transport is `shm`, `p2p`, or `mixed`.
- `shm` forces the mapped-host-memory SHM FIFO path.
- `p2p` requires that every ring edge has the required bidirectional peer
  access and fails during setup on the first unavailable direction.

P2P is a single-node transport. It requires CUDA peer access for the complete
configured ring; it is not a multi-node or network transport.

---

## Limitations

Currently supports only:

- Single-node multi-GPU (tested with `CUDA_VISIBLE_DEVICES=0,1,2,3`; rank count is configurable via `NANO_NCCL_NRANKS`)
- `float`, FP16 (`fp16`), and BF16 (`bf16`) dtypes; BF16 requires SM80+
- `sum` reduce op
- out-of-place
- SHM FIFO and device P2P FIFO transports; P2P is single-node only

Future expansion plans:

- dtype: `double` / `int8`
- reduce op: `max` / `min` / `prod`
- rank count: 2 / 8 / 16 (templated, host-side dispatch)
- collective: `all_gather` / `reduce_scatter` / `broadcast`
- transport: network

---

## License

[MIT](LICENSE)
