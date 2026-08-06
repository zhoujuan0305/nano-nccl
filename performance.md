# Performance

All results below are out-of-place all-reduce measurements. Bandwidth is `busbw` in GB/s. Every measured nano-nccl and NCCL result completed validation with zero wrong values. The `nano/NCCL` column is calculated from the unrounded measured time (`nccl_time_us / nano_time_us`).

## Test Topology And Environment

Both hosts use two-socket Intel Xeon Platinum 8462Y+ CPUs (32 cores per socket, two threads per core), 4x NVIDIA RTX A6000 GPUs (SM86), CUDA 12.8.61, NCCL 2.30.7 built from source, nccl-tests 2.19.6, and Open MPI 4.1.2.

| Node | OS kernel | GPU driver | GPUs |
| --- | --- | --- | --- |
| A | Linux 5.15.0-136-generic | 580.82.07 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |
| B | Linux 6.8.0-124-generic | 580.173.02 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |

On each host GPU0-GPU1 and GPU2-GPU3 are connected by four NVLinks. The two pairs are separated by `SYS` paths across NUMA nodes. The nano-nccl `auto` plan resolves each ring edge independently (P2P when bidirectional NVLink peer access is available, otherwise SHM). Two-host socket runs use TCP; NCCL socket runs set `NCCL_IB_DISABLE=1`. Two-host RDMA runs use nano `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` (WRITE+CTS) and NCCL IB/RoCE with `NCCL_NET_GDR_LEVEL=0`.

All measurements use a Release build with `NANO_NCCL_ENABLE_BENCH_PROFILING=OFF`, message sizes 256 KiB through 64 MiB, `-w 5`, and `-n 20`. NCCL uses `Ring`, `Simple`, four channels, and a 32 MiB buffer. RDMA SEND/WriteCts always post from the registered mapped FIFO (no host bounce; visibility via `__threadfence_system` + release/acquire steps).

## Single Host: 4 Ranks

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 56.87 | 6.91 | 83.45 | 4.71 | 1.47 |
| 1 MiB | 110.97 | 14.17 | 130.94 | 12.01 | 1.18 |
| 4 MiB | 324.92 | 19.36 | 338.60 | 18.58 | 1.04 |
| 16 MiB | 1127.26 | 22.32 | 1123.90 | 22.39 | 1.00 |
| 64 MiB | 4321.42 | 23.29 | 4393.67 | 22.91 | 1.02 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 58.67 | 6.70 | 82.86 | 4.75 | 1.41 |
| 1 MiB | 108.11 | 14.55 | 130.24 | 12.08 | 1.20 |
| 4 MiB | 326.50 | 19.27 | 339.04 | 18.56 | 1.04 |
| 16 MiB | 1128.24 | 22.31 | 1123.04 | 22.41 | 1.00 |
| 64 MiB | 4320.79 | 23.30 | 4396.04 | 22.90 | 1.02 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 57.02 | 6.90 | 82.65 | 4.76 | 1.45 |
| 1 MiB | 113.53 | 13.85 | 130.39 | 12.06 | 1.15 |
| 4 MiB | 327.77 | 19.19 | 337.30 | 18.65 | 1.03 |
| 16 MiB | 1130.34 | 22.26 | 1124.81 | 22.37 | 1.00 |
| 64 MiB | 4354.95 | 23.11 | 4400.78 | 22.87 | 1.01 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 59.29 | 6.63 | 81.77 | 4.81 | 1.38 |
| 1 MiB | 109.32 | 14.39 | 130.40 | 12.06 | 1.19 |
| 4 MiB | 325.20 | 19.35 | 336.63 | 18.69 | 1.04 |
| 16 MiB | 1130.86 | 22.25 | 1123.70 | 22.40 | 0.99 |
| 64 MiB | 4351.32 | 23.13 | 4400.88 | 22.87 | 1.01 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 58.74 | 6.69 | 81.24 | 4.84 | 1.38 |
| 1 MiB | 109.73 | 14.33 | 129.35 | 12.16 | 1.18 |
| 4 MiB | 325.32 | 19.34 | 337.66 | 18.63 | 1.04 |
| 16 MiB | 1128.13 | 22.31 | 1121.01 | 22.45 | 0.99 |
| 64 MiB | 4323.95 | 23.28 | 4459.52 | 22.57 | 1.03 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 57.11 | 6.89 | 81.01 | 4.85 | 1.42 |
| 1 MiB | 109.82 | 14.32 | 130.19 | 12.08 | 1.19 |
| 4 MiB | 326.89 | 19.25 | 337.23 | 18.66 | 1.03 |
| 16 MiB | 1124.80 | 22.37 | 1122.49 | 22.42 | 1.00 |
| 64 MiB | 4325.25 | 23.27 | 4394.66 | 22.91 | 1.02 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 59.45 | 6.61 | 84.48 | 4.65 | 1.42 |
| 1 MiB | 110.60 | 14.22 | 132.16 | 11.90 | 1.19 |
| 4 MiB | 341.54 | 18.42 | 341.36 | 18.43 | 1.00 |
| 16 MiB | 1196.71 | 21.03 | 1124.85 | 22.37 | 0.94 |
| 64 MiB | 4649.46 | 21.65 | 4392.34 | 22.92 | 0.94 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 59.35 | 6.63 | 84.74 | 4.64 | 1.43 |
| 1 MiB | 111.30 | 14.13 | 132.65 | 11.86 | 1.19 |
| 4 MiB | 340.42 | 18.48 | 342.38 | 18.38 | 1.01 |
| 16 MiB | 1196.29 | 21.04 | 1122.67 | 22.42 | 0.94 |
| 64 MiB | 4645.89 | 21.67 | 4393.47 | 22.91 | 0.95 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 59.18 | 6.64 | 80.26 | 4.90 | 1.36 |
| 1 MiB | 108.62 | 14.48 | 129.76 | 12.12 | 1.19 |
| 4 MiB | 326.71 | 19.26 | 340.01 | 18.50 | 1.04 |
| 16 MiB | 1129.46 | 22.28 | 1122.19 | 22.43 | 0.99 |
| 64 MiB | 4323.44 | 23.28 | 4391.89 | 22.92 | 1.02 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 57.30 | 6.86 | 82.93 | 4.74 | 1.45 |
| 1 MiB | 109.47 | 14.37 | 130.03 | 12.10 | 1.19 |
| 4 MiB | 327.74 | 19.20 | 340.15 | 18.50 | 1.04 |
| 16 MiB | 1129.76 | 22.28 | 1123.55 | 22.40 | 0.99 |
| 64 MiB | 4325.65 | 23.27 | 4395.88 | 22.90 | 1.02 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 58.75 | 6.69 | 84.66 | 4.64 | 1.44 |
| 1 MiB | 112.35 | 14.00 | 132.98 | 11.83 | 1.18 |
| 4 MiB | 347.64 | 18.10 | 340.70 | 18.47 | 0.98 |
| 16 MiB | 1222.49 | 20.59 | 1123.74 | 22.39 | 0.92 |
| 64 MiB | 4757.42 | 21.16 | 4394.39 | 22.91 | 0.92 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 59.30 | 6.63 | 83.88 | 4.69 | 1.41 |
| 1 MiB | 112.42 | 13.99 | 133.29 | 11.80 | 1.19 |
| 4 MiB | 345.48 | 18.21 | 340.90 | 18.46 | 0.99 |
| 16 MiB | 1224.38 | 20.55 | 1121.46 | 22.44 | 0.92 |
| 64 MiB | 4758.23 | 21.16 | 4392.23 | 22.92 | 0.92 |


## Two Hosts: 8 Ranks Over TCP Socket

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4663.35 | 0.10 | 4010.91 | 0.11 | 0.86 |
| 1 MiB | 18315.3 | 0.10 | 15837.0 | 0.12 | 0.86 |
| 4 MiB | 63281.0 | 0.12 | 63175.0 | 0.12 | 1.00 |
| 16 MiB | 252122.6 | 0.12 | 251979.0 | 0.12 | 1.00 |
| 64 MiB | 1065552.2 | 0.11 | 1006634.0 | 0.12 | 0.94 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4173.52 | 0.11 | 4018.19 | 0.11 | 0.96 |
| 1 MiB | 16036.4 | 0.11 | 16768.5 | 0.11 | 1.05 |
| 4 MiB | 67018.1 | 0.11 | 64188.7 | 0.11 | 0.96 |
| 16 MiB | 275353.7 | 0.11 | 252119.0 | 0.12 | 0.92 |
| 64 MiB | 1116616.1 | 0.11 | 1007592.0 | 0.12 | 0.90 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4169.59 | 0.11 | 3989.02 | 0.12 | 0.96 |
| 1 MiB | 15959.0 | 0.11 | 15880.9 | 0.12 | 1.00 |
| 4 MiB | 63215.1 | 0.12 | 62967.6 | 0.12 | 1.00 |
| 16 MiB | 252141.2 | 0.12 | 252054.0 | 0.12 | 1.00 |
| 64 MiB | 1007939.3 | 0.12 | 1005942.0 | 0.12 | 1.00 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4670.59 | 0.10 | 3997.05 | 0.11 | 0.86 |
| 1 MiB | 17959.9 | 0.10 | 15873.3 | 0.12 | 0.88 |
| 4 MiB | 72335.1 | 0.10 | 63829.5 | 0.11 | 0.88 |
| 16 MiB | 274562.4 | 0.11 | 251970.0 | 0.12 | 0.92 |
| 64 MiB | 1123405.4 | 0.10 | 1006644.0 | 0.12 | 0.90 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4659.71 | 0.10 | 4038.54 | 0.11 | 0.87 |
| 1 MiB | 17955.8 | 0.10 | 16628.6 | 0.11 | 0.93 |
| 4 MiB | 63248.6 | 0.12 | 66318.4 | 0.11 | 1.05 |
| 16 MiB | 280845.5 | 0.10 | 261423.0 | 0.11 | 0.93 |
| 64 MiB | 1127906.4 | 0.10 | 1010996.0 | 0.12 | 0.90 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4675.43 | 0.10 | 3995.23 | 0.11 | 0.85 |
| 1 MiB | 18088.5 | 0.10 | 15826.7 | 0.12 | 0.87 |
| 4 MiB | 72481.4 | 0.10 | 63190.6 | 0.12 | 0.87 |
| 16 MiB | 279942.8 | 0.10 | 252123.0 | 0.12 | 0.90 |
| 64 MiB | 1123083.0 | 0.10 | 1006423.0 | 0.12 | 0.90 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4795.24 | 0.10 | 3993.21 | 0.11 | 0.83 |
| 1 MiB | 18613.4 | 0.10 | 15790.5 | 0.12 | 0.85 |
| 4 MiB | 72271.3 | 0.10 | 63000.3 | 0.12 | 0.87 |
| 16 MiB | 280391.2 | 0.10 | 251682.0 | 0.12 | 0.90 |
| 64 MiB | 1124150.7 | 0.10 | 1006647.0 | 0.12 | 0.90 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4653.40 | 0.10 | 4039.66 | 0.11 | 0.87 |
| 1 MiB | 18208.1 | 0.10 | 15818.5 | 0.12 | 0.87 |
| 4 MiB | 72768.5 | 0.10 | 63130.0 | 0.12 | 0.87 |
| 16 MiB | 283989.5 | 0.10 | 251705.0 | 0.12 | 0.89 |
| 64 MiB | 1125060.2 | 0.10 | 1005888.0 | 0.12 | 0.89 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4264.97 | 0.11 | 4017.81 | 0.11 | 0.94 |
| 1 MiB | 16075.4 | 0.11 | 16045.9 | 0.11 | 1.00 |
| 4 MiB | 63600.5 | 0.12 | 63410.3 | 0.12 | 1.00 |
| 16 MiB | 252920.2 | 0.12 | 252994.0 | 0.12 | 1.00 |
| 64 MiB | 1022485.0 | 0.11 | 1008570.0 | 0.12 | 0.99 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4682.73 | 0.10 | 4010.66 | 0.11 | 0.86 |
| 1 MiB | 17906.3 | 0.10 | 15862.7 | 0.12 | 0.89 |
| 4 MiB | 63196.5 | 0.12 | 63194.5 | 0.12 | 1.00 |
| 16 MiB | 278039.2 | 0.11 | 253132.0 | 0.12 | 0.91 |
| 64 MiB | 1123606.1 | 0.10 | 1008956.0 | 0.12 | 0.90 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4286.01 | 0.11 | 4001.74 | 0.11 | 0.93 |
| 1 MiB | 17879.1 | 0.10 | 15851.8 | 0.12 | 0.89 |
| 4 MiB | 68370.0 | 0.11 | 63282.7 | 0.12 | 0.93 |
| 16 MiB | 271850.3 | 0.11 | 252718.0 | 0.12 | 0.93 |
| 64 MiB | 1068649.9 | 0.11 | 1009941.0 | 0.12 | 0.95 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4190.62 | 0.11 | 3997.09 | 0.11 | 0.95 |
| 1 MiB | 16890.9 | 0.11 | 15994.2 | 0.11 | 0.95 |
| 4 MiB | 65651.8 | 0.11 | 63233.8 | 0.12 | 0.96 |
| 16 MiB | 276494.9 | 0.11 | 252149.0 | 0.12 | 0.91 |
| 64 MiB | 1139506.8 | 0.10 | 1008912.0 | 0.12 | 0.89 |


## Two Hosts: 8 Ranks Over RDMA

The nano-nccl runs explicitly request `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` (WRITE+CTS over registered host-pinned FIFO). Aggregate transport output is `mixed` because local ring edges retain their local transport while cross-host edges use RDMA. NCCL uses RDMA with `NCCL_NET_GDR_LEVEL=0` (host-pin / no GPUDirect RDMA). No two-host performance acceptance threshold has been established.

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 249.88 | 1.84 | 187.13 | 2.45 | 0.75 |
| 1 MiB | 315.10 | 5.82 | 259.45 | 7.07 | 0.82 |
| 4 MiB | 727.56 | 10.09 | 655.17 | 11.20 | 0.90 |
| 16 MiB | 2662.57 | 11.03 | 2617.44 | 11.22 | 0.98 |
| 64 MiB | 10587.2 | 11.09 | 10354.7 | 11.34 | 0.98 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 250.79 | 1.83 | 204.64 | 2.24 | 0.82 |
| 1 MiB | 319.26 | 5.75 | 260.72 | 7.04 | 0.82 |
| 4 MiB | 727.10 | 10.09 | 656.36 | 11.18 | 0.90 |
| 16 MiB | 2664.58 | 11.02 | 2557.52 | 11.48 | 0.96 |
| 64 MiB | 10569.8 | 11.11 | 10623.7 | 11.05 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 249.19 | 1.84 | 189.98 | 2.41 | 0.76 |
| 1 MiB | 313.99 | 5.84 | 256.82 | 7.15 | 0.82 |
| 4 MiB | 724.62 | 10.13 | 653.53 | 11.23 | 0.90 |
| 16 MiB | 2707.70 | 10.84 | 2553.55 | 11.50 | 0.94 |
| 64 MiB | 10559.8 | 11.12 | 10236.8 | 11.47 | 0.97 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 243.06 | 1.89 | 202.61 | 2.26 | 0.83 |
| 1 MiB | 313.35 | 5.86 | 273.21 | 6.72 | 0.87 |
| 4 MiB | 726.82 | 10.10 | 679.48 | 10.80 | 0.93 |
| 16 MiB | 2665.52 | 11.01 | 2562.93 | 11.46 | 0.96 |
| 64 MiB | 10566.5 | 11.11 | 10286.4 | 11.42 | 0.97 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 242.93 | 1.89 | 525.16 | 0.87 | 2.16 |
| 1 MiB | 312.66 | 5.87 | 256.81 | 7.15 | 0.82 |
| 4 MiB | 722.73 | 10.16 | 652.33 | 11.25 | 0.90 |
| 16 MiB | 2654.39 | 11.06 | 2554.98 | 11.49 | 0.96 |
| 64 MiB | 10566.8 | 11.11 | 10233.7 | 11.48 | 0.97 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 246.10 | 1.86 | 498.44 | 0.92 | 2.03 |
| 1 MiB | 314.88 | 5.83 | 259.15 | 7.08 | 0.82 |
| 4 MiB | 723.99 | 10.14 | 653.73 | 11.23 | 0.90 |
| 16 MiB | 2658.52 | 11.04 | 2903.75 | 10.11 | 1.09 |
| 64 MiB | 10560.5 | 11.12 | 10236.6 | 11.47 | 0.97 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 251.22 | 1.83 | 207.70 | 2.21 | 0.83 |
| 1 MiB | 325.84 | 5.63 | 262.08 | 7.00 | 0.80 |
| 4 MiB | 735.51 | 9.98 | 655.58 | 11.20 | 0.89 |
| 16 MiB | 2659.93 | 11.04 | 2554.02 | 11.50 | 0.96 |
| 64 MiB | 10551.9 | 11.13 | 10332.9 | 11.37 | 0.98 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 246.50 | 1.86 | 212.61 | 2.16 | 0.86 |
| 1 MiB | 315.30 | 5.82 | 256.77 | 7.15 | 0.81 |
| 4 MiB | 726.66 | 10.10 | 654.85 | 11.21 | 0.90 |
| 16 MiB | 2656.48 | 11.05 | 2581.22 | 11.37 | 0.97 |
| 64 MiB | 10559.0 | 11.12 | 10346.8 | 11.35 | 0.98 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 247.77 | 1.85 | 515.26 | 0.89 | 2.08 |
| 1 MiB | 317.20 | 5.79 | 260.16 | 7.05 | 0.82 |
| 4 MiB | 727.67 | 10.09 | 654.30 | 11.22 | 0.90 |
| 16 MiB | 2658.08 | 11.05 | 2553.90 | 11.50 | 0.96 |
| 64 MiB | 10577.6 | 11.10 | 10343.2 | 11.35 | 0.98 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 245.62 | 1.87 | 200.12 | 2.29 | 0.81 |
| 1 MiB | 324.34 | 5.66 | 273.68 | 6.70 | 0.84 |
| 4 MiB | 723.91 | 10.14 | 680.03 | 10.79 | 0.94 |
| 16 MiB | 2659.51 | 11.04 | 2564.52 | 11.45 | 0.96 |
| 64 MiB | 10566.4 | 11.11 | 10229.9 | 11.48 | 0.97 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 253.02 | 1.81 | 508.62 | 0.90 | 2.01 |
| 1 MiB | 326.42 | 5.62 | 276.37 | 6.64 | 0.85 |
| 4 MiB | 739.13 | 9.93 | 678.12 | 10.82 | 0.92 |
| 16 MiB | 2690.24 | 10.91 | 2931.01 | 10.02 | 1.09 |
| 64 MiB | 10554.0 | 11.13 | 10233.1 | 11.48 | 0.97 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 250.35 | 1.83 | 202.86 | 2.26 | 0.81 |
| 1 MiB | 325.45 | 5.64 | 273.60 | 6.71 | 0.84 |
| 4 MiB | 735.28 | 9.98 | 679.80 | 10.80 | 0.92 |
| 16 MiB | 2662.48 | 11.03 | 2553.83 | 11.50 | 0.96 |
| 64 MiB | 10554.7 | 11.13 | 10235.5 | 11.47 | 0.97 |


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

For RDMA WRITE+CTS, build nano-nccl with MPI and RDMA enabled on both hosts. Set `NANO_NCCL_SOCKET_IFNAME=<interface>` for bootstrap, `NANO_NCCL_RDMA_IFNAME=<rdma-interface>` (and `NANO_NCCL_RDMA_GID_INDEX` when required), and `NANO_NCCL_RDMA_USE_WRITE=1`. Set `NCCL_SOCKET_IFNAME=<interface>`, `NCCL_IB_HCA=<rdma-hca>`, `NCCL_IB_GID_INDEX` when required, and `NCCL_NET_GDR_LEVEL=0` for the host-pin baseline. Clear inherited `NCCL_IB_DISABLE`.

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
