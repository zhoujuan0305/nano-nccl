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
| 256 KiB | 67.93 | 5.79 | 83.20 | 4.73 | 1.22 |
| 1 MiB | 120.44 | 13.06 | 130.59 | 12.04 | 1.08 |
| 4 MiB | 336.74 | 18.68 | 337.63 | 18.63 | 1.00 |
| 16 MiB | 1150.45 | 21.87 | 1125.48 | 22.36 | 0.98 |
| 64 MiB | 4346.46 | 23.16 | 4394.95 | 22.90 | 1.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.40 | 5.92 | 83.56 | 4.71 | 1.26 |
| 1 MiB | 119.99 | 13.11 | 130.12 | 12.09 | 1.08 |
| 4 MiB | 337.06 | 18.67 | 337.78 | 18.63 | 1.00 |
| 16 MiB | 1148.99 | 21.90 | 1120.51 | 22.46 | 0.98 |
| 64 MiB | 4350.92 | 23.14 | 4394.31 | 22.91 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 67.36 | 5.84 | 81.78 | 4.81 | 1.21 |
| 1 MiB | 122.91 | 12.80 | 129.62 | 12.13 | 1.05 |
| 4 MiB | 337.95 | 18.62 | 336.14 | 18.72 | 0.99 |
| 16 MiB | 1152.91 | 21.83 | 1124.97 | 22.37 | 0.98 |
| 64 MiB | 4375.50 | 23.01 | 4395.07 | 22.90 | 1.00 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 67.41 | 5.83 | 82.84 | 4.75 | 1.23 |
| 1 MiB | 122.52 | 12.84 | 129.76 | 12.12 | 1.06 |
| 4 MiB | 337.78 | 18.63 | 336.37 | 18.70 | 1.00 |
| 16 MiB | 1153.74 | 21.81 | 1128.90 | 22.29 | 0.98 |
| 64 MiB | 4375.01 | 23.01 | 4410.57 | 22.82 | 1.01 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.48 | 5.92 | 79.40 | 4.95 | 1.19 |
| 1 MiB | 120.33 | 13.07 | 128.71 | 12.22 | 1.07 |
| 4 MiB | 336.68 | 18.69 | 337.64 | 18.63 | 1.00 |
| 16 MiB | 1148.30 | 21.92 | 1120.21 | 22.47 | 0.98 |
| 64 MiB | 4351.57 | 23.13 | 4390.24 | 22.93 | 1.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.96 | 5.87 | 82.57 | 4.76 | 1.23 |
| 1 MiB | 120.07 | 13.10 | 130.59 | 12.04 | 1.09 |
| 4 MiB | 337.04 | 18.67 | 336.66 | 18.69 | 1.00 |
| 16 MiB | 1149.64 | 21.89 | 1123.07 | 22.41 | 0.98 |
| 64 MiB | 4349.16 | 23.15 | 4387.09 | 22.95 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 69.47 | 5.66 | 85.11 | 4.62 | 1.23 |
| 1 MiB | 122.28 | 12.86 | 132.58 | 11.86 | 1.08 |
| 4 MiB | 352.10 | 17.87 | 340.58 | 18.47 | 0.97 |
| 16 MiB | 1222.35 | 20.59 | 1120.88 | 22.45 | 0.92 |
| 64 MiB | 4665.66 | 21.58 | 4390.43 | 22.93 | 0.94 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 68.68 | 5.72 | 83.43 | 4.71 | 1.21 |
| 1 MiB | 122.10 | 12.88 | 131.80 | 11.93 | 1.08 |
| 4 MiB | 352.57 | 17.84 | 341.10 | 18.44 | 0.97 |
| 16 MiB | 1219.43 | 20.64 | 1122.45 | 22.42 | 0.92 |
| 64 MiB | 4664.90 | 21.58 | 4390.12 | 22.93 | 0.94 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.49 | 5.91 | 79.57 | 4.94 | 1.20 |
| 1 MiB | 119.79 | 13.13 | 129.52 | 12.14 | 1.08 |
| 4 MiB | 337.63 | 18.63 | 337.80 | 18.62 | 1.00 |
| 16 MiB | 1150.44 | 21.88 | 1121.75 | 22.43 | 0.98 |
| 64 MiB | 4349.88 | 23.14 | 4387.85 | 22.94 | 1.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.53 | 5.91 | 82.61 | 4.76 | 1.24 |
| 1 MiB | 119.60 | 13.15 | 130.96 | 12.01 | 1.09 |
| 4 MiB | 336.76 | 18.68 | 340.48 | 18.48 | 1.01 |
| 16 MiB | 1148.68 | 21.91 | 1122.72 | 22.41 | 0.98 |
| 64 MiB | 4349.94 | 23.14 | 4393.22 | 22.91 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 68.68 | 5.73 | 83.37 | 4.72 | 1.21 |
| 1 MiB | 124.71 | 12.61 | 131.92 | 11.92 | 1.06 |
| 4 MiB | 358.29 | 17.56 | 339.18 | 18.55 | 0.95 |
| 16 MiB | 1254.46 | 20.06 | 1123.64 | 22.40 | 0.90 |
| 64 MiB | 4785.61 | 21.03 | 4390.96 | 22.93 | 0.92 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 68.24 | 5.76 | 84.94 | 4.63 | 1.24 |
| 1 MiB | 124.32 | 12.65 | 132.77 | 11.85 | 1.07 |
| 4 MiB | 357.76 | 17.59 | 339.85 | 18.51 | 0.95 |
| 16 MiB | 1251.72 | 20.10 | 1122.04 | 22.43 | 0.90 |
| 64 MiB | 4794.12 | 21.00 | 4391.63 | 22.92 | 0.92 |


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
