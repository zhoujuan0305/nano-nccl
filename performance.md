# Performance

## Test environment

| Item | Configuration |
|---|---|
| Hardware | Two hosts, two NVIDIA RTX A6000 GPUs per host (four GPUs total, SM86) |
| Software | CUDA 12.8.61, Open MPI 4.1.2, NCCL 2.30.7 |
| Interconnect | Local NVLink P2P; cross-host GPUDirect RDMA over 100 Gb/s Ethernet |
| Collective | Out-of-place float/sum AllReduce, Ring/Simple |
| Parameters | 1–4 channels; 32 MiB buffer per channel; Release build, profiling disabled |
| Measurement | `-w 5 -n 20`; four paired runs; per-collective completion, slowest-rank time |

## AllReduce matrix

Each entry is **nano-nccl / NCCL latency in µs**, using the median of four job averages.

| Message | 1 channel | 2 channels | 3 channels | 4 channels |
|---|---:|---:|---:|---:|
| 1 MiB | 194.44 / 229.12 | 163.47 / 199.26 | 156.94 / 187.21 | 154.98 / 195.28 |
| 9 MiB | 1416.87 / 1532.55 | 1262.10 / 1442.60 | 1246.85 / 1463.53 | 1241.38 / 1429.62 |
| 12 MiB | 1881.48 / 2067.82 | 1679.88 / 1836.99 | 1654.19 / 1798.06 | 1664.79 / 1787.08 |
| 16 MiB | 2513.69 / 2676.43 | 2227.34 / 2408.14 | 2215.30 / 2305.43 | 2204.70 / 2277.51 |
| 32 MiB | 5040.16 / 5184.35 | 4476.26 / 4577.50 | 4497.89 / 4604.35 | 4468.45 / 4579.76 |

All 20 cases passed correctness checks. The geometric mean of `NCCL latency / nano-nccl latency`
(equivalently, `nano-nccl busbw / NCCL busbw`) is **1.10×**.
