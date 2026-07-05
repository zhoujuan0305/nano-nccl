# nano-nccl

[English](README.md)

单机多 GPU All Reduce 库，在 4× GTX 1080 Ti (Pascal, 无 NVLink) 上达到 NCCL `Ring` + `Simple` + 4 channels 基线等价的性能。

---

## 性能

与同轮 NCCL 基线对比（out-of-place busbw，`-w 2 -n 5`）：

| size (bytes) | NCCL busbw (GB/s) | nano-nccl busbw (GB/s) | ratio |
| ---: | ---: | ---: | ---: |
| 262144 | 4.46 | 4.88 | 1.094 |
| 1048576 | 7.12 | 7.78 | 1.093 |
| 4194304 | 8.51 | 8.85 | 1.040 |
| 16777216 | 8.59 | 8.85 | 1.030 |
| 67108864 | 8.71 | 8.86 | 1.017 |

geomean(ratio) = 1.054，所有尺寸均 ≥ 1.00。

**测试环境**：4× GTX 1080 Ti (Pascal sm_61)，CUDA 12.4，driver 550.127.05，无 NVLink，GPU0/1 NUMA 0 / GPU2/3 NUMA 1。

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

例如，4 GPU Pascal (sm_61) 系统：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=61
```

2 GPU Turing (sm_75) 系统：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNANO_NCCL_NRANKS=2 -DNANO_NCCL_CUDA_ARCH=75
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
  --algo ring_simple -b 262144 -e 67108864 -f 4 -w 2 -n 5

# 纯正确性测试
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_correctness

# 冒烟测试
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_smoke
```

---

## 当前限制

当前仅支持：

- 单机多 GPU（已验证 `CUDA_VISIBLE_DEVICES=0,1,2,3`；rank 数可通过 `NANO_NCCL_NRANKS` 配置）
- `float` 类型
- `sum` 规约操作
- out-of-place

未来计划扩展：

- dtype：`half` / `double` / `int8`
- reduce op：`max` / `min` / `prod`
- rank 数：2 / 8 / 16（模板参数化，host 侧分发）
- collective：`all_gather` / `reduce_scatter` / `broadcast`
- 通信路径：P2P / NVLink / 网络

---

## 许可证

[MIT](LICENSE)
