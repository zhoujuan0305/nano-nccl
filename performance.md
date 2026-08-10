# Performance

All results below are out-of-place all-reduce measurements. Bandwidth is `busbw` in GB/s. Every measured nano-nccl and NCCL result completed validation with zero wrong values. The `nano/NCCL` column is calculated from the unrounded measured time (`nccl_time_us / nano_time_us`).

## Test Topology And Environment

Both hosts use two-socket Intel Xeon Platinum 8462Y+ CPUs (32 cores per socket, two threads per core), 4x NVIDIA RTX A6000 GPUs (SM86), CUDA 12.8.61, NCCL 2.30.7 built from source, nccl-tests 2.19.6, and Open MPI 4.1.2.

| Node | OS kernel | GPU driver | GPUs |
| --- | --- | --- | --- |
| A | Linux 5.15.0-136-generic | 580.82.07 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |
| B | Linux 6.8.0-124-generic | 580.173.02 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |

On each host GPU0-GPU1 and GPU2-GPU3 are connected by four NVLinks. The two pairs are separated by `SYS` paths across NUMA nodes. The nano-nccl `auto` plan resolves each ring edge independently (P2P when bidirectional NVLink peer access is available, otherwise SHM). Two-host socket runs use TCP; NCCL socket runs set `NCCL_IB_DISABLE=1`. Two-host RDMA runs use nano `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` (WRITE+CTS; local edges P2P when NVLink peer access is available, otherwise SHM; dedicated per-proxy threads by default) and NCCL IB/RoCE with `NCCL_NET_GDR_LEVEL=0`.

All measurements use a Release build with `NANO_NCCL_ENABLE_BENCH_PROFILING=OFF`, message sizes 256 KiB through 64 MiB, `-w 5`, and `-n 20`. NCCL uses `Ring`, `Simple`, four channels, and a 32 MiB buffer. RDMA SEND/WriteCts always post from the registered mapped FIFO (no host bounce; visibility via publisher `st.release.sys(send_tail)` after block sync and host acquire loads).

## Single Host: 4 Ranks

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


## Two Hosts: 8 Ranks Over TCP Socket

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4243.69 | 0.11 | 4038.95 | 0.11 | 0.95 |
| 1 MiB | 16067.8 | 0.11 | 16107.3 | 0.11 | 1.00 |
| 4 MiB | 63345.9 | 0.12 | 64960.6 | 0.11 | 1.03 |
| 16 MiB | 252423.2 | 0.12 | 254811.0 | 0.12 | 1.01 |
| 64 MiB | 1008512.3 | 0.12 | 1011757.0 | 0.12 | 1.00 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4183.56 | 0.11 | 4016.15 | 0.11 | 0.96 |
| 1 MiB | 15948.5 | 0.12 | 15899.2 | 0.12 | 1.00 |
| 4 MiB | 63170.7 | 0.12 | 63205.6 | 0.12 | 1.00 |
| 16 MiB | 253101.9 | 0.12 | 253059.0 | 0.12 | 1.00 |
| 64 MiB | 1032403.2 | 0.11 | 1011620.0 | 0.12 | 0.98 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4285.79 | 0.11 | 4003.96 | 0.11 | 0.93 |
| 1 MiB | 16113.9 | 0.11 | 15818.0 | 0.12 | 0.98 |
| 4 MiB | 63473.4 | 0.12 | 63127.4 | 0.12 | 0.99 |
| 16 MiB | 252573.7 | 0.12 | 253815.0 | 0.12 | 1.00 |
| 64 MiB | 1099605.7 | 0.11 | 1008635.0 | 0.12 | 0.92 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4171.95 | 0.11 | 4021.81 | 0.11 | 0.96 |
| 1 MiB | 16072.0 | 0.11 | 15820.4 | 0.12 | 0.98 |
| 4 MiB | 64563.4 | 0.11 | 63136.3 | 0.12 | 0.98 |
| 16 MiB | 262654.8 | 0.11 | 251980.0 | 0.12 | 0.96 |
| 64 MiB | 1057761.4 | 0.11 | 1007948.0 | 0.12 | 0.95 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4259.78 | 0.11 | 4016.65 | 0.11 | 0.94 |
| 1 MiB | 16133.9 | 0.11 | 15857.1 | 0.12 | 0.98 |
| 4 MiB | 63395.5 | 0.12 | 65316.0 | 0.11 | 1.03 |
| 16 MiB | 272405.3 | 0.11 | 264291.0 | 0.11 | 0.97 |
| 64 MiB | 1008845.0 | 0.12 | 1037770.0 | 0.11 | 1.03 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4298.80 | 0.11 | 4062.90 | 0.11 | 0.95 |
| 1 MiB | 17094.4 | 0.11 | 16144.4 | 0.11 | 0.94 |
| 4 MiB | 63525.1 | 0.12 | 65104.6 | 0.11 | 1.02 |
| 16 MiB | 253311.6 | 0.12 | 255392.0 | 0.11 | 1.01 |
| 64 MiB | 1009582.8 | 0.12 | 1025545.0 | 0.11 | 1.02 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4256.36 | 0.11 | 5671.00 | 0.08 | 1.33 |
| 1 MiB | 16037.9 | 0.11 | 15991.1 | 0.11 | 1.00 |
| 4 MiB | 63405.6 | 0.12 | 63098.1 | 0.12 | 1.00 |
| 16 MiB | 252573.2 | 0.12 | 251849.0 | 0.12 | 1.00 |
| 64 MiB | 1081623.2 | 0.11 | 1007187.0 | 0.12 | 0.93 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4191.84 | 0.11 | 4011.45 | 0.11 | 0.96 |
| 1 MiB | 16089.0 | 0.11 | 15839.2 | 0.12 | 0.98 |
| 4 MiB | 67151.3 | 0.11 | 64176.9 | 0.11 | 0.96 |
| 16 MiB | 271357.0 | 0.11 | 256648.0 | 0.11 | 0.95 |
| 64 MiB | 1127465.0 | 0.10 | 1014853.0 | 0.12 | 0.90 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4281.98 | 0.11 | 4021.00 | 0.11 | 0.94 |
| 1 MiB | 16484.1 | 0.11 | 15833.8 | 0.12 | 0.96 |
| 4 MiB | 63329.2 | 0.12 | 64817.6 | 0.11 | 1.02 |
| 16 MiB | 252401.0 | 0.12 | 254235.0 | 0.12 | 1.01 |
| 64 MiB | 1114487.5 | 0.11 | 1007963.0 | 0.12 | 0.90 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4171.35 | 0.11 | 4044.54 | 0.11 | 0.97 |
| 1 MiB | 16694.4 | 0.11 | 16941.0 | 0.11 | 1.01 |
| 4 MiB | 63256.5 | 0.12 | 65752.0 | 0.11 | 1.04 |
| 16 MiB | 252300.5 | 0.12 | 253168.0 | 0.12 | 1.00 |
| 64 MiB | 1049629.5 | 0.11 | 1007670.0 | 0.12 | 0.96 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4333.95 | 0.11 | 4015.46 | 0.11 | 0.93 |
| 1 MiB | 16195.9 | 0.11 | 15832.2 | 0.12 | 0.98 |
| 4 MiB | 63252.8 | 0.12 | 63019.4 | 0.12 | 1.00 |
| 16 MiB | 251859.6 | 0.12 | 251677.0 | 0.12 | 1.00 |
| 64 MiB | 1080596.3 | 0.11 | 1005652.0 | 0.12 | 0.93 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4274.25 | 0.11 | 4014.05 | 0.11 | 0.94 |
| 1 MiB | 16127.7 | 0.11 | 15803.8 | 0.12 | 0.98 |
| 4 MiB | 63523.9 | 0.12 | 63095.9 | 0.12 | 0.99 |
| 16 MiB | 260661.0 | 0.11 | 252192.0 | 0.12 | 0.97 |
| 64 MiB | 1058100.9 | 0.11 | 1007715.0 | 0.12 | 0.95 |


## Two Hosts: 8 Ranks Over RDMA

The nano-nccl runs explicitly request `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` (WRITE+CTS over registered host-pinned FIFO). Aggregate transport is `mixed`: local ring edges resolve like `auto` (P2P when bidirectional NVLink peer access is available, otherwise SHM) while cross-host edges use RDMA. Host RDMA progress defaults to dedicated per-proxy threads (`NANO_NCCL_RDMA_SHARED_PROGRESS` unset/`0`); set `NANO_NCCL_RDMA_SHARED_PROGRESS=1` for a single shared progress thread. NCCL uses RDMA with `NCCL_NET_GDR_LEVEL=0` (host-pin / no GPUDirect RDMA). Ring Simple publish advances `send_tail` with `st.release.sys` after the publisher block sync (host acquire loads). No two-host performance acceptance threshold has been established. Small-message NCCL points can show cold-start spikes (very low NCCL busbw / high nano/NCCL ratio); treat those as baseline noise rather than nano speedups.

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 199.41 | 2.30 | 219.22 | 2.09 | 1.10 |
| 1 MiB | 248.92 | 7.37 | 284.25 | 6.46 | 1.14 |
| 4 MiB | 722.60 | 10.16 | 690.49 | 10.63 | 0.96 |
| 16 MiB | 2779.49 | 10.56 | 2940.65 | 9.98 | 1.06 |
| 64 MiB | 12099.1 | 9.71 | 13077.8 | 8.98 | 1.08 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 211.16 | 2.17 | 210.08 | 2.18 | 0.99 |
| 1 MiB | 279.68 | 6.56 | 277.61 | 6.61 | 0.99 |
| 4 MiB | 725.32 | 10.12 | 959.34 | 7.65 | 1.32 |
| 16 MiB | 2781.89 | 10.55 | 2744.82 | 10.70 | 0.99 |
| 64 MiB | 12179.1 | 9.64 | 11849.0 | 9.91 | 0.97 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 205.23 | 2.24 | 231.80 | 1.98 | 1.13 |
| 1 MiB | 278.20 | 6.60 | 282.41 | 6.50 | 1.02 |
| 4 MiB | 719.22 | 10.21 | 675.87 | 10.86 | 0.94 |
| 16 MiB | 2803.49 | 10.47 | 3145.93 | 9.33 | 1.12 |
| 64 MiB | 12230.1 | 9.60 | 13864.6 | 8.47 | 1.13 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 192.86 | 2.38 | 215.02 | 2.13 | 1.11 |
| 1 MiB | 250.16 | 7.34 | 280.97 | 6.53 | 1.12 |
| 4 MiB | 717.24 | 10.23 | 687.39 | 10.68 | 0.96 |
| 16 MiB | 2658.77 | 11.04 | 2754.29 | 10.66 | 1.04 |
| 64 MiB | 10574.8 | 11.11 | 11919.1 | 9.85 | 1.13 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 198.85 | 2.31 | 199.25 | 2.30 | 1.00 |
| 1 MiB | 246.50 | 7.44 | 265.90 | 6.90 | 1.08 |
| 4 MiB | 719.80 | 10.20 | 657.52 | 11.16 | 0.91 |
| 16 MiB | 2678.09 | 10.96 | 2656.92 | 11.05 | 0.99 |
| 64 MiB | 10672.4 | 11.00 | 11608.0 | 10.12 | 1.09 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 191.65 | 2.39 | 193.90 | 2.37 | 1.01 |
| 1 MiB | 241.83 | 7.59 | 259.59 | 7.07 | 1.07 |
| 4 MiB | 723.26 | 10.15 | 653.20 | 11.24 | 0.90 |
| 16 MiB | 2663.01 | 11.03 | 2600.74 | 11.29 | 0.98 |
| 64 MiB | 10530.9 | 11.15 | 10961.2 | 10.71 | 1.04 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 353.16 | 1.30 | 206.88 | 2.22 | 0.59 |
| 1 MiB | 270.20 | 6.79 | 285.64 | 6.42 | 1.06 |
| 4 MiB | 721.83 | 10.17 | 693.49 | 10.58 | 0.96 |
| 16 MiB | 2693.22 | 10.90 | 2776.06 | 10.58 | 1.03 |
| 64 MiB | 10915.3 | 10.76 | 12055.0 | 9.74 | 1.10 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 203.81 | 2.25 | 486.64 | 0.94 | 2.39 |
| 1 MiB | 259.63 | 7.07 | 282.36 | 6.50 | 1.09 |
| 4 MiB | 723.59 | 10.14 | 688.90 | 10.65 | 0.95 |
| 16 MiB | 2713.39 | 10.82 | 2731.12 | 10.75 | 1.01 |
| 64 MiB | 10945.6 | 10.73 | 11638.2 | 10.09 | 1.06 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 198.90 | 2.31 | 212.64 | 2.16 | 1.07 |
| 1 MiB | 260.80 | 7.04 | 283.19 | 6.48 | 1.09 |
| 4 MiB | 725.21 | 10.12 | 690.56 | 10.63 | 0.95 |
| 16 MiB | 2701.49 | 10.87 | 3180.64 | 9.23 | 1.18 |
| 64 MiB | 10948.7 | 10.73 | 17895.2 | 6.56 | 1.63 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 199.64 | 2.30 | 214.39 | 2.14 | 1.07 |
| 1 MiB | 258.09 | 7.11 | 644.42 | 2.85 | 2.50 |
| 4 MiB | 743.80 | 9.87 | 655.61 | 11.20 | 0.88 |
| 16 MiB | 2740.34 | 10.71 | 2752.24 | 10.67 | 1.00 |
| 64 MiB | 10955.4 | 10.72 | 11761.4 | 9.99 | 1.07 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 194.68 | 2.36 | 518.84 | 0.88 | 2.67 |
| 1 MiB | 248.23 | 7.39 | 257.84 | 7.12 | 1.04 |
| 4 MiB | 722.22 | 10.16 | 653.52 | 11.23 | 0.90 |
| 16 MiB | 2664.94 | 11.02 | 2553.19 | 11.50 | 0.96 |
| 64 MiB | 10532.4 | 11.15 | 10225.8 | 11.48 | 0.97 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 190.76 | 2.40 | 215.60 | 2.13 | 1.13 |
| 1 MiB | 247.47 | 7.42 | 282.93 | 6.49 | 1.14 |
| 4 MiB | 717.37 | 10.23 | 687.87 | 10.67 | 0.96 |
| 16 MiB | 2661.19 | 11.03 | 2818.20 | 10.42 | 1.06 |
| 64 MiB | 10521.4 | 11.16 | 11824.1 | 9.93 | 1.12 |


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
