# Performance

All results below are out-of-place all-reduce measurements on **one host**. Bandwidth is `busbw` in GB/s. Every measured nano-nccl and NCCL result completed validation with zero wrong values. The `nano/NCCL` column is calculated from the unrounded measured time (`nccl_time_us / nano_time_us`).

## Test Topology And Environment

One host: two-socket Intel Xeon Platinum 8462Y+ (32 cores per socket, two threads per core), 4x NVIDIA RTX A6000 (SM86), CUDA 12.8.61, NCCL 2.30.7 built from source, nccl-tests 2.19.6, and Open MPI 4.1.2.

| Node | OS kernel | GPU driver | GPUs |
| --- | --- | --- | --- |
| A | Linux 5.15.0-136-generic | 580.82.07 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |

GPU0-GPU1 and GPU2-GPU3 are connected by four NVLinks. The two pairs are separated by `SYS` paths across NUMA nodes. Three single-host tables are reported separately because they are different transport classes: in-process auto (P2P/SHM), 4-rank TCP socket, and 4-rank host-pinned RDMA / NET-IB.

All measurements use a Release build with `NANO_NCCL_ENABLE_BENCH_PROFILING=OFF`, message sizes 256 KiB through 64 MiB, `-w 5`, and `-n 20`. NCCL uses `Ring`, `Simple`, four channels, and a 32 MiB buffer. RDMA WRITE+CTS posts from the registered mapped FIFO (no host bounce; visibility via publisher `st.release.sys(send_tail)` after block sync and host acquire loads).

## Single Host: 4 Ranks Over Auto (P2P/SHM)

In-process 4-GPU communicator. Nano `--transport auto` resolves each ring edge independently (P2P when bidirectional NVLink peer access is available, otherwise SHM). NCCL is the matching intra-node Ring+Simple path (P2P/SHM allowed).

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 67.52 | 5.82 | 77.91 | 5.05 | 1.15 |
| 1 MiB | 120.70 | 13.03 | 124.95 | 12.59 | 1.04 |
| 4 MiB | 337.15 | 18.66 | 332.74 | 18.91 | 0.99 |
| 16 MiB | 1150.55 | 21.87 | 1122.13 | 22.43 | 0.98 |
| 64 MiB | 4353.83 | 23.12 | 4402.32 | 22.87 | 1.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.98 | 5.87 | 77.68 | 5.06 | 1.16 |
| 1 MiB | 119.71 | 13.14 | 125.30 | 12.55 | 1.05 |
| 4 MiB | 337.10 | 18.66 | 335.08 | 18.78 | 0.99 |
| 16 MiB | 1153.10 | 21.82 | 1122.79 | 22.41 | 0.97 |
| 64 MiB | 4351.79 | 23.13 | 4385.77 | 22.95 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 68.05 | 5.78 | 80.75 | 4.87 | 1.19 |
| 1 MiB | 122.43 | 12.85 | 128.64 | 12.23 | 1.05 |
| 4 MiB | 337.75 | 18.63 | 339.06 | 18.56 | 1.00 |
| 16 MiB | 1154.52 | 21.80 | 1133.75 | 22.20 | 0.98 |
| 64 MiB | 4379.36 | 22.99 | 4400.01 | 22.88 | 1.00 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 67.55 | 5.82 | 80.62 | 4.88 | 1.19 |
| 1 MiB | 124.32 | 12.65 | 128.43 | 12.25 | 1.03 |
| 4 MiB | 337.99 | 18.61 | 338.79 | 18.57 | 1.00 |
| 16 MiB | 1152.07 | 21.84 | 1134.20 | 22.19 | 0.98 |
| 64 MiB | 4378.07 | 22.99 | 4400.18 | 22.88 | 1.01 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.99 | 5.87 | 75.87 | 5.18 | 1.13 |
| 1 MiB | 120.57 | 13.05 | 126.26 | 12.46 | 1.05 |
| 4 MiB | 335.60 | 18.75 | 336.84 | 18.68 | 1.00 |
| 16 MiB | 1150.17 | 21.88 | 1118.65 | 22.50 | 0.97 |
| 64 MiB | 4347.15 | 23.16 | 4390.02 | 22.93 | 1.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.81 | 5.89 | 76.56 | 5.14 | 1.15 |
| 1 MiB | 119.93 | 13.12 | 126.36 | 12.45 | 1.05 |
| 4 MiB | 337.02 | 18.67 | 334.45 | 18.81 | 0.99 |
| 16 MiB | 1149.59 | 21.89 | 1121.43 | 22.44 | 0.98 |
| 64 MiB | 4352.46 | 23.13 | 4386.52 | 22.95 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 67.72 | 5.81 | 81.06 | 4.85 | 1.20 |
| 1 MiB | 119.81 | 13.13 | 127.70 | 12.32 | 1.07 |
| 4 MiB | 336.97 | 18.67 | 334.58 | 18.80 | 0.99 |
| 16 MiB | 1148.00 | 21.92 | 1121.18 | 22.45 | 0.98 |
| 64 MiB | 4351.92 | 23.13 | 4387.43 | 22.94 | 1.01 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 67.97 | 5.79 | 79.82 | 4.93 | 1.17 |
| 1 MiB | 120.77 | 13.02 | 127.30 | 12.36 | 1.05 |
| 4 MiB | 336.45 | 18.70 | 334.51 | 18.81 | 0.99 |
| 16 MiB | 1148.29 | 21.92 | 1120.66 | 22.46 | 0.98 |
| 64 MiB | 4351.76 | 23.13 | 4390.82 | 22.93 | 1.01 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.98 | 5.87 | 74.49 | 5.28 | 1.11 |
| 1 MiB | 119.45 | 13.17 | 124.99 | 12.58 | 1.05 |
| 4 MiB | 338.27 | 18.60 | 334.38 | 18.82 | 0.99 |
| 16 MiB | 1149.32 | 21.90 | 1117.50 | 22.52 | 0.97 |
| 64 MiB | 4354.53 | 23.12 | 4384.98 | 22.96 | 1.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.43 | 5.92 | 76.60 | 5.13 | 1.15 |
| 1 MiB | 120.18 | 13.09 | 125.49 | 12.53 | 1.04 |
| 4 MiB | 336.38 | 18.70 | 335.97 | 18.73 | 1.00 |
| 16 MiB | 1149.77 | 21.89 | 1121.44 | 22.44 | 0.98 |
| 64 MiB | 4353.09 | 23.12 | 4387.01 | 22.95 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 67.36 | 5.84 | 80.40 | 4.89 | 1.19 |
| 1 MiB | 120.60 | 13.04 | 127.69 | 12.32 | 1.06 |
| 4 MiB | 337.28 | 18.65 | 336.27 | 18.71 | 1.00 |
| 16 MiB | 1152.07 | 21.84 | 1122.06 | 22.43 | 0.97 |
| 64 MiB | 4353.88 | 23.12 | 4388.52 | 22.94 | 1.01 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 67.28 | 5.84 | 80.90 | 4.86 | 1.20 |
| 1 MiB | 120.41 | 13.06 | 127.14 | 12.37 | 1.06 |
| 4 MiB | 337.68 | 18.63 | 337.15 | 18.66 | 1.00 |
| 16 MiB | 1150.50 | 21.87 | 1122.04 | 22.43 | 0.98 |
| 64 MiB | 4360.23 | 23.09 | 4387.45 | 22.94 | 1.01 |


## Single Host: 4 Ranks Over TCP Socket

Four MPI ranks, one GPU per rank. Nano `--transport auto` therefore places every ring edge on TCP socket. NCCL comparison forces the same class: `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_IB_DISABLE=1`, Ring+Simple, four channels, 32 MiB buffer. This is a loopback-TCP path, not the intra-node P2P/SHM table above.

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 347.48 | 1.13 | 693.69 | 0.57 | 2.00 |
| 1 MiB | 924.29 | 1.70 | 1170.89 | 1.34 | 1.27 |
| 4 MiB | 1776.64 | 3.54 | 2585.38 | 2.43 | 1.46 |
| 16 MiB | 3430.15 | 7.34 | 8380.80 | 3.00 | 2.44 |
| 64 MiB | 10688.4 | 9.42 | 41912.2 | 2.40 | 3.92 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 273.67 | 1.44 | 706.06 | 0.56 | 2.58 |
| 1 MiB | 471.70 | 3.33 | 1177.59 | 1.34 | 2.50 |
| 4 MiB | 1498.34 | 4.20 | 2636.00 | 2.39 | 1.76 |
| 16 MiB | 2828.07 | 8.90 | 8557.11 | 2.94 | 3.03 |
| 64 MiB | 10709.7 | 9.40 | 42416.4 | 2.37 | 3.96 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 260.57 | 1.51 | 701.69 | 0.56 | 2.69 |
| 1 MiB | 544.63 | 2.89 | 1153.11 | 1.36 | 2.12 |
| 4 MiB | 1432.58 | 4.39 | 2658.76 | 2.37 | 1.86 |
| 16 MiB | 3081.86 | 8.17 | 8718.92 | 2.89 | 2.83 |
| 64 MiB | 10764.3 | 9.35 | 42659.7 | 2.36 | 3.96 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 272.59 | 1.44 | 698.71 | 0.56 | 2.56 |
| 1 MiB | 628.40 | 2.50 | 1169.93 | 1.34 | 1.86 |
| 4 MiB | 1375.46 | 4.57 | 2607.25 | 2.41 | 1.90 |
| 16 MiB | 2738.74 | 9.19 | 8446.26 | 2.98 | 3.08 |
| 64 MiB | 11315.3 | 8.90 | 42064.9 | 2.39 | 3.72 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 233.01 | 1.69 | 704.26 | 0.56 | 3.02 |
| 1 MiB | 486.00 | 3.24 | 1150.70 | 1.37 | 2.37 |
| 4 MiB | 1528.68 | 4.12 | 2616.23 | 2.40 | 1.71 |
| 16 MiB | 2697.14 | 9.33 | 8522.31 | 2.95 | 3.16 |
| 64 MiB | 11040.1 | 9.12 | 42046.5 | 2.39 | 3.81 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 253.83 | 1.55 | 712.76 | 0.55 | 2.81 |
| 1 MiB | 499.86 | 3.15 | 1178.04 | 1.34 | 2.36 |
| 4 MiB | 1449.55 | 4.34 | 2628.62 | 2.39 | 1.81 |
| 16 MiB | 2937.15 | 8.57 | 8533.44 | 2.95 | 2.91 |
| 64 MiB | 10659.7 | 9.44 | 42675.8 | 2.36 | 4.00 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 259.36 | 1.52 | 711.48 | 0.55 | 2.74 |
| 1 MiB | 542.47 | 2.90 | 1209.97 | 1.30 | 2.23 |
| 4 MiB | 1358.75 | 4.63 | 2607.12 | 2.41 | 1.92 |
| 16 MiB | 2524.06 | 9.97 | 8696.91 | 2.89 | 3.45 |
| 64 MiB | 10672.9 | 9.43 | 41660.4 | 2.42 | 3.90 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 232.43 | 1.69 | 710.81 | 0.55 | 3.06 |
| 1 MiB | 533.99 | 2.95 | 1189.86 | 1.32 | 2.23 |
| 4 MiB | 1269.23 | 4.96 | 2682.64 | 2.35 | 2.11 |
| 16 MiB | 3004.02 | 8.38 | 8688.68 | 2.90 | 2.89 |
| 64 MiB | 10515.6 | 9.57 | 42270.5 | 2.38 | 4.02 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 259.69 | 1.51 | 695.02 | 0.57 | 2.68 |
| 1 MiB | 554.60 | 2.84 | 1163.87 | 1.35 | 2.10 |
| 4 MiB | 1725.60 | 3.65 | 2586.62 | 2.43 | 1.50 |
| 16 MiB | 2908.37 | 8.65 | 8424.09 | 2.99 | 2.90 |
| 64 MiB | 10695.7 | 9.41 | 42935.5 | 2.34 | 4.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 234.18 | 1.68 | 684.50 | 0.57 | 2.92 |
| 1 MiB | 457.94 | 3.43 | 1157.84 | 1.36 | 2.53 |
| 4 MiB | 1325.25 | 4.75 | 2585.14 | 2.43 | 1.95 |
| 16 MiB | 2807.36 | 8.96 | 8720.54 | 2.89 | 3.11 |
| 64 MiB | 10822.8 | 9.30 | 42224.9 | 2.38 | 3.90 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 273.56 | 1.44 | 703.69 | 0.56 | 2.57 |
| 1 MiB | 661.75 | 2.38 | 1176.18 | 1.34 | 1.78 |
| 4 MiB | 1386.37 | 4.54 | 2589.77 | 2.43 | 1.87 |
| 16 MiB | 3387.68 | 7.43 | 8587.64 | 2.93 | 2.53 |
| 64 MiB | 12512.8 | 8.04 | 42580.5 | 2.36 | 3.40 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 261.17 | 1.51 | 697.63 | 0.56 | 2.67 |
| 1 MiB | 527.41 | 2.98 | 1169.08 | 1.35 | 2.22 |
| 4 MiB | 1575.31 | 3.99 | 2705.78 | 2.33 | 1.72 |
| 16 MiB | 2647.42 | 9.51 | 8822.52 | 2.85 | 3.33 |
| 64 MiB | 11755.1 | 8.56 | 41540.7 | 2.42 | 3.53 |


## Single Host: 4 Ranks Over RDMA

Four MPI ranks, one GPU per rank. Nano `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` (WRITE+CTS over registered host-pinned FIFO; dedicated per-proxy threads). RTR `path_mtu` is `min(local, remote) port.active_mtu`. NCCL comparison: `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_NET_GDR_LEVEL=0` (host-pin NET/IB, no GPUDirect). Both engines sit on the NIC loopback band (~3 GB/s here), not the NVLink auto table.

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 139.42 | 2.82 | 156.42 | 2.51 | 1.12 |
| 1 MiB | 505.74 | 3.11 | 521.23 | 3.02 | 1.03 |
| 4 MiB | 2000.00 | 3.15 | 2007.49 | 3.13 | 1.00 |
| 16 MiB | 8001.40 | 3.15 | 8017.90 | 3.14 | 1.00 |
| 64 MiB | 31988.5 | 3.15 | 32247.0 | 3.12 | 1.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 139.67 | 2.82 | 155.06 | 2.54 | 1.11 |
| 1 MiB | 515.49 | 3.05 | 523.72 | 3.00 | 1.02 |
| 4 MiB | 2001.31 | 3.14 | 2053.68 | 3.06 | 1.03 |
| 16 MiB | 8000.92 | 3.15 | 8063.03 | 3.12 | 1.01 |
| 64 MiB | 32027.4 | 3.14 | 32267.7 | 3.12 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 142.06 | 2.77 | 156.00 | 2.52 | 1.10 |
| 1 MiB | 504.62 | 3.12 | 520.79 | 3.02 | 1.03 |
| 4 MiB | 2003.83 | 3.14 | 2008.20 | 3.13 | 1.00 |
| 16 MiB | 7999.14 | 3.15 | 8055.88 | 3.12 | 1.01 |
| 64 MiB | 32063.7 | 3.14 | 32262.5 | 3.12 | 1.01 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 140.55 | 2.80 | 155.12 | 2.53 | 1.10 |
| 1 MiB | 522.18 | 3.01 | 532.61 | 2.95 | 1.02 |
| 4 MiB | 2000.80 | 3.14 | 2011.78 | 3.13 | 1.01 |
| 16 MiB | 7997.39 | 3.15 | 8008.06 | 3.14 | 1.00 |
| 64 MiB | 32032.6 | 3.14 | 32140.1 | 3.13 | 1.00 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 141.01 | 2.79 | 156.42 | 2.51 | 1.11 |
| 1 MiB | 504.73 | 3.12 | 531.53 | 2.96 | 1.05 |
| 4 MiB | 2002.23 | 3.14 | 2012.76 | 3.13 | 1.01 |
| 16 MiB | 7997.14 | 3.15 | 8057.63 | 3.12 | 1.01 |
| 64 MiB | 32011.9 | 3.14 | 32064.1 | 3.14 | 1.00 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 140.32 | 2.80 | 157.16 | 2.50 | 1.12 |
| 1 MiB | 505.25 | 3.11 | 518.94 | 3.03 | 1.03 |
| 4 MiB | 2002.03 | 3.14 | 2007.23 | 3.13 | 1.00 |
| 16 MiB | 8007.77 | 3.14 | 8072.15 | 3.12 | 1.01 |
| 64 MiB | 31987.5 | 3.15 | 32259.5 | 3.12 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 140.92 | 2.79 | 157.56 | 2.50 | 1.12 |
| 1 MiB | 509.27 | 3.09 | 517.67 | 3.04 | 1.02 |
| 4 MiB | 2003.42 | 3.14 | 2007.86 | 3.13 | 1.00 |
| 16 MiB | 8026.79 | 3.14 | 8020.81 | 3.14 | 1.00 |
| 64 MiB | 32002.4 | 3.15 | 32271.2 | 3.12 | 1.01 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 139.08 | 2.83 | 155.74 | 2.52 | 1.12 |
| 1 MiB | 504.79 | 3.12 | 518.95 | 3.03 | 1.03 |
| 4 MiB | 2000.54 | 3.14 | 2054.43 | 3.06 | 1.03 |
| 16 MiB | 7998.10 | 3.15 | 8041.42 | 3.13 | 1.01 |
| 64 MiB | 31986.7 | 3.15 | 32226.5 | 3.12 | 1.01 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 139.17 | 2.83 | 154.81 | 2.54 | 1.11 |
| 1 MiB | 505.60 | 3.11 | 531.27 | 2.96 | 1.05 |
| 4 MiB | 2033.84 | 3.09 | 2013.46 | 3.12 | 0.99 |
| 16 MiB | 8012.28 | 3.14 | 7993.48 | 3.15 | 1.00 |
| 64 MiB | 31987.1 | 3.15 | 32088.6 | 3.14 | 1.00 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 140.15 | 2.81 | 166.57 | 2.36 | 1.19 |
| 1 MiB | 521.48 | 3.02 | 519.81 | 3.03 | 1.00 |
| 4 MiB | 2002.24 | 3.14 | 2046.69 | 3.07 | 1.02 |
| 16 MiB | 7996.42 | 3.15 | 8025.84 | 3.14 | 1.00 |
| 64 MiB | 32038.8 | 3.14 | 32156.4 | 3.13 | 1.00 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 138.89 | 2.83 | 156.00 | 2.52 | 1.12 |
| 1 MiB | 514.26 | 3.06 | 533.32 | 2.95 | 1.04 |
| 4 MiB | 2003.04 | 3.14 | 2007.75 | 3.13 | 1.00 |
| 16 MiB | 7997.42 | 3.15 | 7996.82 | 3.15 | 1.00 |
| 64 MiB | 32002.8 | 3.15 | 32085.0 | 3.14 | 1.00 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 142.06 | 2.77 | 154.94 | 2.54 | 1.09 |
| 1 MiB | 509.83 | 3.09 | 517.91 | 3.04 | 1.02 |
| 4 MiB | 2006.03 | 3.14 | 2007.13 | 3.13 | 1.00 |
| 16 MiB | 8003.89 | 3.14 | 8001.02 | 3.15 | 1.00 |
| 64 MiB | 32004.8 | 3.15 | 32277.0 | 3.12 | 1.01 |


## Reproduction

Build nano-nccl with CUDA 12.8, SM86, Release mode, and profiling disabled. The in-process auto binary uses four ranks in one process. Socket and RDMA tables use four MPI ranks (`NANO_NCCL_NRANKS=4`, one GPU per rank) from the same Open MPI 4.1.2 prefix.

```bash
# nano-nccl, in-process auto (P2P/SHM)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ./build-perf-single/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --transport auto --dtype <float|fp16|bf16> \
  --redop <sum|avg|max|min> -b 262144 -e 67108864 -f 4 -w 5 -n 20

# NCCL, in-process intra-node
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_LIBRARY_PATH=<path-to-nccl-lib> \
NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 \
NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=33554432 \
  <path-to-nccl-tests>/build/all_reduce_perf \
  -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 \
  -d <float|half|bfloat16> -o <sum|avg|max|min>
```

For single-host socket, launch four MPI ranks with one GPU each (`CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK`). Nano uses `--transport auto` (cross-process edges are socket). NCCL sets `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, and `NCCL_IB_DISABLE=1`.

For single-host RDMA, the same 4-rank launch uses nano `--transport rdma` and `NANO_NCCL_RDMA_USE_WRITE=1`. Set `NANO_NCCL_SOCKET_IFNAME=<interface>` for bootstrap and `NANO_NCCL_RDMA_IFNAME=<rdma-interface>` (and `NANO_NCCL_RDMA_GID_INDEX` when required). NCCL sets `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_NET_GDR_LEVEL=0`, `NCCL_IB_HCA=<rdma-hca>`, and `NCCL_IB_GID_INDEX` when required.

```bash
cmake -S . -B build-perf-rdma-n4 -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON \
  -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=86 \
  -DNANO_NCCL_ENABLE_BENCH_PROFILING=OFF
cmake --build build-perf-rdma-n4 -j<jobs>
```

Or regenerate this file from a completed matrix JSON:

```bash
scripts/run_performance_matrix.sh --nccl-bin <path> --nccl-lib <dir> --out-dir <out>
python3 scripts/render_performance_md.py <out>/matrix.json -o performance.md
```
