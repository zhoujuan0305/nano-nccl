# Performance

All results below are out-of-place all-reduce measurements. Bandwidth is `busbw` in GB/s. Every measured nano-nccl and NCCL result completed validation with zero wrong values. The `nano/NCCL` column is calculated from the unrounded measured time (`nccl_time_us / nano_time_us`).

## Test Topology And Environment

Both hosts use two-socket Intel Xeon Platinum 8462Y+ CPUs (32 cores per socket, two threads per core), 4x NVIDIA RTX A6000 GPUs (SM86), CUDA 12.8.61, NCCL 2.30.7 built from source, nccl-tests 2.19.6, and Open MPI 4.1.2.

| Node | OS kernel | GPU driver | GPUs |
| --- | --- | --- | --- |
| A | Linux 5.15.0-136-generic | 580.82.07 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |
| B | Linux 6.8.0-124-generic | 580.173.02 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |

On each host GPU0-GPU1 and GPU2-GPU3 are connected by four NVLinks. The two pairs are separated by `SYS` paths across NUMA nodes. The nano-nccl `auto` plan resolves each ring edge independently (P2P when bidirectional NVLink peer access is available, otherwise SHM). Two-host socket runs use TCP; NCCL socket runs set `NCCL_IB_DISABLE=1`. Two-host RDMA runs use nano `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` (WRITE+CTS; local edges P2P when NVLink peer access is available, otherwise SHM; dedicated per-proxy threads by default) and NCCL IB/RoCE with `NCCL_NET_GDR_LEVEL=0`.

All measurements use a Release build with `NANO_NCCL_ENABLE_BENCH_PROFILING=OFF`, message sizes 256 KiB through 64 MiB, `-w 5`, and `-n 20`. NCCL uses `Ring`, `Simple`, four channels, and a 32 MiB buffer. RDMA SEND/WriteCts always post from the registered mapped FIFO (no host bounce; visibility via publisher `fence.acq_rel.sys` + relaxed `send_tail` store, matching NCCL Simple postPeer).

## Single Host: 4 Ranks

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.52 | 5.91 | 81.92 | 4.80 | 1.23 |
| 1 MiB | 119.53 | 13.16 | 130.35 | 12.07 | 1.09 |
| 4 MiB | 335.20 | 18.77 | 337.91 | 18.62 | 1.01 |
| 16 MiB | 1148.74 | 21.91 | 1121.97 | 22.43 | 0.98 |
| 64 MiB | 4350.52 | 23.14 | 4395.69 | 22.90 | 1.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 67.10 | 5.86 | 83.82 | 4.69 | 1.25 |
| 1 MiB | 122.22 | 12.87 | 130.40 | 12.06 | 1.07 |
| 4 MiB | 336.13 | 18.72 | 340.08 | 18.50 | 1.01 |
| 16 MiB | 1151.83 | 21.85 | 1122.80 | 22.41 | 0.97 |
| 64 MiB | 4349.44 | 23.14 | 4397.12 | 22.89 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 67.88 | 5.79 | 83.08 | 4.73 | 1.22 |
| 1 MiB | 123.23 | 12.76 | 130.06 | 12.09 | 1.06 |
| 4 MiB | 338.08 | 18.61 | 337.58 | 18.64 | 1.00 |
| 16 MiB | 1155.64 | 21.78 | 1125.98 | 22.35 | 0.97 |
| 64 MiB | 4376.94 | 23.00 | 4400.16 | 22.88 | 1.01 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 68.23 | 5.76 | 81.40 | 4.83 | 1.19 |
| 1 MiB | 122.30 | 12.86 | 129.32 | 12.16 | 1.06 |
| 4 MiB | 337.71 | 18.63 | 338.48 | 18.59 | 1.00 |
| 16 MiB | 1155.38 | 21.78 | 1125.86 | 22.35 | 0.97 |
| 64 MiB | 4375.54 | 23.01 | 4399.64 | 22.88 | 1.01 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.61 | 5.90 | 82.09 | 4.79 | 1.23 |
| 1 MiB | 120.01 | 13.11 | 129.66 | 12.13 | 1.08 |
| 4 MiB | 337.60 | 18.64 | 336.40 | 18.70 | 1.00 |
| 16 MiB | 1149.88 | 21.89 | 1120.11 | 22.47 | 0.97 |
| 64 MiB | 4349.93 | 23.14 | 4394.39 | 22.91 | 1.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.93 | 5.88 | 81.23 | 4.84 | 1.21 |
| 1 MiB | 119.96 | 13.11 | 130.26 | 12.07 | 1.09 |
| 4 MiB | 336.11 | 18.72 | 338.00 | 18.61 | 1.01 |
| 16 MiB | 1150.19 | 21.88 | 1123.39 | 22.40 | 0.98 |
| 64 MiB | 4353.54 | 23.12 | 4392.21 | 22.92 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 68.93 | 5.70 | 84.97 | 4.63 | 1.23 |
| 1 MiB | 121.37 | 12.96 | 132.86 | 11.84 | 1.09 |
| 4 MiB | 349.82 | 17.98 | 340.96 | 18.45 | 0.97 |
| 16 MiB | 1218.38 | 20.66 | 1123.93 | 22.39 | 0.92 |
| 64 MiB | 4668.76 | 21.56 | 4395.28 | 22.90 | 0.94 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 68.45 | 5.74 | 83.64 | 4.70 | 1.22 |
| 1 MiB | 122.24 | 12.87 | 131.57 | 11.95 | 1.08 |
| 4 MiB | 351.82 | 17.88 | 340.03 | 18.50 | 0.97 |
| 16 MiB | 1220.38 | 20.62 | 1122.62 | 22.42 | 0.92 |
| 64 MiB | 4662.99 | 21.59 | 4396.51 | 22.90 | 0.94 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.87 | 5.88 | 80.86 | 4.86 | 1.21 |
| 1 MiB | 120.01 | 13.11 | 129.35 | 12.16 | 1.08 |
| 4 MiB | 337.19 | 18.66 | 338.82 | 18.57 | 1.00 |
| 16 MiB | 1151.68 | 21.85 | 1123.00 | 22.41 | 0.98 |
| 64 MiB | 4351.74 | 23.13 | 4392.05 | 22.92 | 1.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 66.40 | 5.92 | 82.56 | 4.76 | 1.24 |
| 1 MiB | 120.02 | 13.11 | 130.16 | 12.08 | 1.08 |
| 4 MiB | 336.90 | 18.67 | 339.66 | 18.52 | 1.01 |
| 16 MiB | 1149.20 | 21.90 | 1120.55 | 22.46 | 0.98 |
| 64 MiB | 4352.25 | 23.13 | 4392.13 | 22.92 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 69.56 | 5.65 | 84.93 | 4.63 | 1.22 |
| 1 MiB | 124.62 | 12.62 | 131.53 | 11.96 | 1.06 |
| 4 MiB | 356.89 | 17.63 | 341.07 | 18.45 | 0.96 |
| 16 MiB | 1250.33 | 20.13 | 1120.98 | 22.45 | 0.90 |
| 64 MiB | 4787.29 | 21.03 | 4397.09 | 22.89 | 0.92 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 68.96 | 5.70 | 85.95 | 4.57 | 1.25 |
| 1 MiB | 123.02 | 12.79 | 132.40 | 11.88 | 1.08 |
| 4 MiB | 357.34 | 17.61 | 341.11 | 18.44 | 0.95 |
| 16 MiB | 1250.36 | 20.13 | 1123.01 | 22.41 | 0.90 |
| 64 MiB | 4787.54 | 21.03 | 4394.15 | 22.91 | 0.92 |


## Two Hosts: 8 Ranks Over TCP Socket

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 492.64 | 0.93 | 1182.57 | 0.39 | 2.40 |
| 1 MiB | 583.51 | 3.14 | 3101.30 | 0.59 | 5.31 |
| 4 MiB | 1656.95 | 4.43 | 7427.38 | 0.99 | 4.48 |
| 16 MiB | 10366.5 | 2.83 | 25832.0 | 1.14 | 2.49 |
| 64 MiB | 32560.2 | 3.61 | 88976.0 | 1.32 | 2.73 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 442.06 | 1.04 | 880.69 | 0.52 | 1.99 |
| 1 MiB | 561.26 | 3.27 | 1407.77 | 1.30 | 2.51 |
| 4 MiB | 1600.22 | 4.59 | 4269.96 | 1.72 | 2.67 |
| 16 MiB | 7895.76 | 3.72 | 14422.4 | 2.04 | 1.83 |
| 64 MiB | 27618.4 | 4.25 | 51787.8 | 2.27 | 1.88 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 422.30 | 1.09 | 1190.28 | 0.39 | 2.82 |
| 1 MiB | 516.50 | 3.55 | 1457.29 | 1.26 | 2.82 |
| 4 MiB | 914.87 | 8.02 | 6803.55 | 1.08 | 7.44 |
| 16 MiB | 2756.37 | 10.65 | 22719.9 | 1.29 | 8.24 |
| 64 MiB | 15923.6 | 7.38 | 94480.8 | 1.24 | 5.93 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 3164.19 | 0.14 | 1131.64 | 0.41 | 0.36 |
| 1 MiB | 626.00 | 2.93 | 1438.17 | 1.28 | 2.30 |
| 4 MiB | 1242.77 | 5.91 | 5510.84 | 1.33 | 4.43 |
| 16 MiB | 7837.72 | 3.75 | 25345.6 | 1.16 | 3.23 |
| 64 MiB | 19124.7 | 6.14 | 104480.0 | 1.12 | 5.46 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 527.90 | 0.87 | 1193.54 | 0.38 | 2.26 |
| 1 MiB | 518.43 | 3.54 | 1422.90 | 1.29 | 2.74 |
| 4 MiB | 1112.19 | 6.60 | 3893.55 | 1.89 | 3.50 |
| 16 MiB | 4074.29 | 7.21 | 14381.7 | 2.04 | 3.53 |
| 64 MiB | 17521.3 | 6.70 | 52205.4 | 2.25 | 2.98 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 578.01 | 0.79 | 1205.07 | 0.38 | 2.08 |
| 1 MiB | 498.89 | 3.68 | 1444.41 | 1.27 | 2.90 |
| 4 MiB | 1124.73 | 6.53 | 4249.73 | 1.73 | 3.78 |
| 16 MiB | 3540.67 | 8.29 | 15038.4 | 1.95 | 4.25 |
| 64 MiB | 16526.2 | 7.11 | 52469.1 | 2.24 | 3.17 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 521.04 | 0.88 | 1075.83 | 0.43 | 2.06 |
| 1 MiB | 532.91 | 3.44 | 1438.47 | 1.28 | 2.70 |
| 4 MiB | 974.51 | 7.53 | 3996.33 | 1.84 | 4.10 |
| 16 MiB | 3144.47 | 9.34 | 14908.5 | 1.97 | 4.74 |
| 64 MiB | 14470.4 | 8.12 | 54474.1 | 2.16 | 3.76 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 532.16 | 0.86 | 1233.65 | 0.37 | 2.32 |
| 1 MiB | 499.32 | 3.68 | 1662.37 | 1.10 | 3.33 |
| 4 MiB | 975.43 | 7.52 | 4831.29 | 1.52 | 4.95 |
| 16 MiB | 2960.70 | 9.92 | 16657.7 | 1.76 | 5.63 |
| 64 MiB | 11370.2 | 10.33 | 56023.0 | 2.10 | 4.93 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 405.31 | 1.13 | 1091.94 | 0.42 | 2.69 |
| 1 MiB | 509.20 | 3.60 | 1497.06 | 1.23 | 2.94 |
| 4 MiB | 1015.61 | 7.23 | 4327.40 | 1.70 | 4.26 |
| 16 MiB | 3006.81 | 9.76 | 15483.4 | 1.90 | 5.15 |
| 64 MiB | 13312.3 | 8.82 | 61248.5 | 1.92 | 4.60 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 520.12 | 0.88 | 1225.96 | 0.37 | 2.36 |
| 1 MiB | 550.04 | 3.34 | 1638.36 | 1.12 | 2.98 |
| 4 MiB | 993.90 | 7.39 | 4843.74 | 1.52 | 4.87 |
| 16 MiB | 3432.56 | 8.55 | 16871.1 | 1.74 | 4.92 |
| 64 MiB | 18253.7 | 6.43 | 57179.0 | 2.05 | 3.13 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 437.29 | 1.05 | 891.35 | 0.51 | 2.04 |
| 1 MiB | 533.20 | 3.44 | 1373.56 | 1.34 | 2.58 |
| 4 MiB | 1168.86 | 6.28 | 4860.10 | 1.51 | 4.16 |
| 16 MiB | 3571.86 | 8.22 | 17348.1 | 1.69 | 4.86 |
| 64 MiB | 14114.3 | 8.32 | 54389.2 | 2.16 | 3.85 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 652.34 | 0.70 | 1094.48 | 0.42 | 1.68 |
| 1 MiB | 595.69 | 3.08 | 1536.66 | 1.19 | 2.58 |
| 4 MiB | 1120.40 | 6.55 | 4573.60 | 1.60 | 4.08 |
| 16 MiB | 3240.00 | 9.06 | 15671.3 | 1.87 | 4.84 |
| 64 MiB | 16540.3 | 7.10 | 55652.1 | 2.11 | 3.36 |


## Two Hosts: 8 Ranks Over RDMA

The nano-nccl runs explicitly request `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` (WRITE+CTS over registered host-pinned FIFO). Aggregate transport is `mixed`: local ring edges resolve like `auto` (P2P when bidirectional NVLink peer access is available, otherwise SHM) while cross-host edges use RDMA. Host RDMA progress defaults to dedicated per-proxy threads (`NANO_NCCL_RDMA_SHARED_PROGRESS` unset/`0`); set `NANO_NCCL_RDMA_SHARED_PROGRESS=1` for a single shared progress thread. NCCL uses RDMA with `NCCL_NET_GDR_LEVEL=0` (host-pin / no GPUDirect RDMA). Ring Simple publish uses NCCL-style `fence.acq_rel.sys` + relaxed `send_tail` store after the publisher block sync. No two-host performance acceptance threshold has been established. Small-message NCCL points can show cold-start spikes (very low NCCL busbw / high nano/NCCL ratio); treat those as baseline noise rather than nano speedups.

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 207.75 | 2.21 | 203.22 | 2.26 | 0.98 |
| 1 MiB | 252.56 | 7.27 | 271.83 | 6.75 | 1.08 |
| 4 MiB | 725.12 | 10.12 | 682.05 | 10.76 | 0.94 |
| 16 MiB | 2666.24 | 11.01 | 2556.23 | 11.49 | 0.96 |
| 64 MiB | 10617.5 | 11.06 | 10347.8 | 11.35 | 0.97 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 188.98 | 2.43 | 209.91 | 2.19 | 1.11 |
| 1 MiB | 244.87 | 7.49 | 260.03 | 7.06 | 1.06 |
| 4 MiB | 717.01 | 10.24 | 655.80 | 11.19 | 0.91 |
| 16 MiB | 2666.78 | 11.01 | 2556.45 | 11.48 | 0.96 |
| 64 MiB | 10634.7 | 11.04 | 10298.9 | 11.40 | 0.97 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 189.74 | 2.42 | 557.13 | 0.82 | 2.94 |
| 1 MiB | 241.61 | 7.59 | 255.82 | 7.17 | 1.06 |
| 4 MiB | 716.72 | 10.24 | 671.14 | 10.94 | 0.94 |
| 16 MiB | 2657.78 | 11.05 | 2554.76 | 11.49 | 0.96 |
| 64 MiB | 10524.0 | 11.16 | 10354.5 | 11.34 | 0.98 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 191.92 | 2.39 | 190.75 | 2.41 | 0.99 |
| 1 MiB | 240.69 | 7.62 | 256.59 | 7.15 | 1.07 |
| 4 MiB | 713.19 | 10.29 | 654.29 | 11.22 | 0.92 |
| 16 MiB | 2661.92 | 11.03 | 2555.72 | 11.49 | 0.96 |
| 64 MiB | 10659.6 | 11.02 | 10314.5 | 11.39 | 0.97 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 196.30 | 2.34 | 191.40 | 2.40 | 0.98 |
| 1 MiB | 243.40 | 7.54 | 256.25 | 7.16 | 1.05 |
| 4 MiB | 712.50 | 10.30 | 654.10 | 11.22 | 0.92 |
| 16 MiB | 2661.64 | 11.03 | 2555.11 | 11.49 | 0.96 |
| 64 MiB | 10619.5 | 11.06 | 10293.7 | 11.41 | 0.97 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 193.54 | 2.37 | 190.28 | 2.41 | 0.98 |
| 1 MiB | 241.83 | 7.59 | 255.18 | 7.19 | 1.06 |
| 4 MiB | 715.44 | 10.26 | 652.77 | 11.24 | 0.91 |
| 16 MiB | 2663.53 | 11.02 | 2568.23 | 11.43 | 0.96 |
| 64 MiB | 10588.5 | 11.09 | 10326.7 | 11.37 | 0.98 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 194.20 | 2.36 | 190.31 | 2.41 | 0.98 |
| 1 MiB | 249.14 | 7.37 | 262.40 | 6.99 | 1.05 |
| 4 MiB | 718.23 | 10.22 | 654.52 | 11.21 | 0.91 |
| 16 MiB | 2663.06 | 11.02 | 2565.08 | 11.45 | 0.96 |
| 64 MiB | 10612.4 | 11.07 | 10368.8 | 11.33 | 0.98 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 191.70 | 2.39 | 193.23 | 2.37 | 1.01 |
| 1 MiB | 247.57 | 7.41 | 258.61 | 7.10 | 1.04 |
| 4 MiB | 717.71 | 10.23 | 655.78 | 11.19 | 0.91 |
| 16 MiB | 2658.90 | 11.04 | 2559.18 | 11.47 | 0.96 |
| 64 MiB | 10583.1 | 11.10 | 10326.3 | 11.37 | 0.98 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 190.24 | 2.41 | 187.96 | 2.44 | 0.99 |
| 1 MiB | 240.59 | 7.63 | 255.64 | 7.18 | 1.06 |
| 4 MiB | 717.82 | 10.23 | 653.10 | 11.24 | 0.91 |
| 16 MiB | 2659.99 | 11.04 | 2553.27 | 11.50 | 0.96 |
| 64 MiB | 10622.1 | 11.06 | 10236.2 | 11.47 | 0.96 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 190.00 | 2.41 | 192.66 | 2.38 | 1.01 |
| 1 MiB | 241.46 | 7.60 | 257.59 | 7.12 | 1.07 |
| 4 MiB | 715.46 | 10.26 | 658.28 | 11.15 | 0.92 |
| 16 MiB | 2658.63 | 11.04 | 2555.67 | 11.49 | 0.96 |
| 64 MiB | 10610.2 | 11.07 | 10318.8 | 11.38 | 0.97 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 196.71 | 2.33 | 190.74 | 2.41 | 0.97 |
| 1 MiB | 253.15 | 7.25 | 259.58 | 7.07 | 1.03 |
| 4 MiB | 718.06 | 10.22 | 655.23 | 11.20 | 0.91 |
| 16 MiB | 2660.97 | 11.03 | 2556.50 | 11.48 | 0.96 |
| 64 MiB | 10619.0 | 11.06 | 10414.5 | 11.28 | 0.98 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 205.12 | 2.24 | 206.73 | 2.22 | 1.01 |
| 1 MiB | 248.78 | 7.38 | 258.94 | 7.09 | 1.04 |
| 4 MiB | 723.01 | 10.15 | 653.34 | 11.23 | 0.90 |
| 16 MiB | 2654.32 | 11.06 | 2556.87 | 11.48 | 0.96 |
| 64 MiB | 10645.4 | 11.03 | 10312.0 | 11.39 | 0.97 |


## Reproduction

Build nano-nccl with CUDA 12.8, SM86, Release mode, and profiling disabled. The single-host binary uses four ranks. The distributed binary on both hosts uses eight global ranks and the same Open MPI 4.1.2 prefix. Place matching trees under `<path-to-worktree>` on every host.

```bash
# nano-nccl, single host
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ./build-perf-single/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --transport auto --dtype <float|fp16|bf16> \
  --redop <sum|avg|max|min> -b 262144 -e 67108864 -f 4 -w 5 -n 20

# NCCL, single host
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_LIBRARY_PATH=<path-to-nccl-lib> \
NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 \
NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=33554432 \
  <path-to-nccl-tests>/build/all_reduce_perf \
  -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 \
  -d <float|half|bfloat16> -o <sum|avg|max|min>
```

For two hosts over TCP Socket, launch one 4-GPU process on each host with an Open MPI 4.1.2 launcher, set `NANO_NCCL_SOCKET_IFNAME=<interface>` for nano-nccl, and set `NCCL_SOCKET_IFNAME=<interface>`, `NCCL_IB_DISABLE=1`, `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, and `NCCL_BUFFSIZE=33554432` for NCCL. Bind MPI TCP/OOB to the same bootstrap interface (`btl_tcp_if_include` / `oob_tcp_if_include`).

For RDMA, build nano-nccl with MPI and RDMA enabled on both hosts. Set `NANO_NCCL_SOCKET_IFNAME=<interface>` for bootstrap and `NANO_NCCL_RDMA_IFNAME=<rdma-interface>` (and `NANO_NCCL_RDMA_GID_INDEX` when required). Set `NCCL_SOCKET_IFNAME=<interface>`, `NCCL_IB_HCA=<rdma-hca>`, `NCCL_IB_GID_INDEX` when required, and `NCCL_NET_GDR_LEVEL=0` for the host-pin baseline. Clear inherited `NCCL_IB_DISABLE`.

```bash
cmake -S . -B build-perf-rdma -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON \
  -DNANO_NCCL_NRANKS=8 -DNANO_NCCL_CUDA_ARCH=86 \
  -DNANO_NCCL_ENABLE_BENCH_PROFILING=OFF
cmake --build build-perf-rdma -j<jobs>
```

Or regenerate this file from a completed matrix JSON:

```bash
scripts/run_performance_matrix.sh --nccl-bin <path> --nccl-lib <dir> --out-dir <out>
python3 scripts/render_performance_md.py <out>/matrix.json -o performance.md
```
