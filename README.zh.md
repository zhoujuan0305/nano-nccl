# nano-nccl

[English](README.md)

单机多 GPU All Reduce 库，目标是达到 NCCL `Ring` + `Simple` + 4 channels 的性能。

---

## 性能

与 NCCL 同轮对比（out-of-place busbw，`-w 2 -n 5`）。nano-nccl 使用 `--transport auto`，解析为 P2P/SHM 混合的 ring-edge 路径。

| dtype | 256 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float | 1.300 | 1.225 | 1.039 | 1.013 | 1.021 | 1.114 |
| fp16 | 1.334 | 1.250 | 1.331 | 1.322 | 1.150 | 1.276 |
| bf16 | 1.303 | 1.236 | 1.062 | 1.018 | 1.022 | 1.122 |

表中数值为 `nano-nccl busbw / NCCL busbw`；全部 15 个测点均正确（`#wrong=0`），并且不低于 NCCL 基线。

**测试环境**：4× NVIDIA RTX A6000 (Ampere sm_86)，CUDA 12.8，`CUDA_VISIBLE_DEVICES=0,1,2,3`。NCCL 使用 `Ring` + `Simple`、4 channels 和 `NCCL_BUFFSIZE=33554432`。

> 性能数据仅适用于上述硬件和参数场景。

---

## 构建

依赖：CUDA 12+、CMake 3.18+、libnuma-dev

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_NRANKS=<你的GPU数量> \
  -DNANO_NCCL_CUDA_ARCH=<你的CUDA算力>
make -j$(nproc)
```

例如，本次测试的 4 GPU RTX A6000 (sm_86) 系统：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=86
```

构建产物：

- `build/benchmarks/nano_nccl_all_reduce_bench` — 性能 + 正确性 benchmark
- `build/tests/nano_nccl_correctness` — 纯正确性测试
- `build/tests/nano_nccl_smoke` — 冒烟测试

### CMake 选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `NANO_NCCL_NRANKS` | 4 | GPU 数 |
| `NANO_NCCL_NCHANNELS` | 4 | channel 数 |
| `NANO_NCCL_CUDA_ARCH` | 61 | CUDA 算力（如 61 对应 sm_61，75 对应 Turing，86 对应 Ampere） |
| `NANO_NCCL_BLOCK_THREADS` | 512 | 每 block 线程数 |
| `NANO_NCCL_FIFO_BUFF_BYTES` | 33554432 | FIFO buffer 大小（字节，默认 32 MiB） |

NUMA 拓扑在运行时从 `/sys/bus/pci/devices/*/numa_node` 自动检测，换机器不需要改源码。

---

## 运行

```bash
# Benchmark（性能 + 正确性）
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype float --transport auto \
  -b 262144 -e 67108864 -f 4 -w 2 -n 5

# FP16 和 BF16 的正确性/性能运行（BF16 需要 SM80+）
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype fp16 --transport auto \
  -b 262144 -e 67108864 -f 4 -w 2 -n 5
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype bf16 --transport auto \
  -b 262144 -e 67108864 -f 4 -w 2 -n 5

# 纯正确性测试
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_correctness

# 冒烟测试
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_smoke
```

### 通信路径选择

`--transport` 接受 `auto`、`shm` 和 `p2p`。

- `auto`（默认值）对每条 ring edge 独立选择：只有具备 direct NVLink 和双向 CUDA peer access 时才选择 P2P；其余 edge 使用 SHM。最终路径可能是 `shm`、`p2p` 或 `mixed`。
- `shm` 强制使用 mapped host memory 的 SHM FIFO 路径。
- `p2p` 要求每条 ring edge 都具备所需的双向 peer access。任一方向不可用时，初始化会在第一个不可用方向报错，不会回退。

P2P 是单机通信路径，需要每对完整配置环邻居之间的双向 CUDA peer access；它不是多机或网络通信路径。

---

## 当前限制

当前仅支持：

- 单机多 GPU（已验证 `CUDA_VISIBLE_DEVICES=0,1,2,3`；rank 数可通过 `NANO_NCCL_NRANKS` 配置）
- `float`、FP16（`fp16`）和 BF16（`bf16`）类型；BF16 需要 SM80+
- `sum` 规约操作
- out-of-place
- SHM FIFO 和 device P2P FIFO 通信路径；P2P 仅支持单机

未来计划扩展：

- dtype：`double` / `int8`
- reduce op：`max` / `min` / `prod`
- rank 数：2 / 8 / 16（模板参数化，host 侧分发）
- collective：`all_gather` / `reduce_scatter` / `broadcast`
- 通信路径：网络

---

## 许可证

[MIT](LICENSE)
