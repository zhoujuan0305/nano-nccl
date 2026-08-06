# Performance

All results below are out-of-place all-reduce measurements. Bandwidth is `busbw` in GB/s. Every measured nano-nccl and NCCL result completed validation with zero wrong values. The `nano/NCCL` column is calculated from the unrounded measured time (`nccl_time_us / nano_time_us`).

Measurement tree matches the RDMA host-pin path documented in the READMEs (direct registered-FIFO send, empty-slice elision, selective CQ signaling). Tables were regenerated from a full 180-cell matrix (`float`/`fp16`/`bf16` × `sum`/`avg`/`max`/`min` × single/socket/RDMA). Generated at `2026-08-06T02:24:33.603799+00:00` (UTC).

## Test Topology And Environment

Both hosts use two-socket Intel Xeon Platinum 8462Y+ CPUs (32 cores per socket, two threads per core), 4x NVIDIA RTX A6000 GPUs (SM86), CUDA 12.8.61, NCCL 2.30.7 built from source, nccl-tests 2.19.6, and Open MPI 4.1.2.

| Node | OS kernel | GPU driver | GPUs |
| --- | --- | --- | --- |
| A | Linux 5.15.0-136-generic | 580.82.07 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |
| B | Linux 6.8.0-124-generic | 580.173.02 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |

On each host GPU0-GPU1 and GPU2-GPU3 are connected by four NVLinks. The two pairs are separated by `SYS` paths across NUMA nodes. The nano-nccl `auto` plan resolves each ring edge independently (P2P when bidirectional NVLink peer access is available, otherwise SHM). Two-host socket runs use TCP; NCCL socket runs set `NCCL_IB_DISABLE=1`. Two-host RDMA runs use nano `--transport rdma` and NCCL IB/RoCE with `NCCL_NET_GDR_LEVEL=0`.

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

The nano-nccl runs explicitly request `--transport rdma`. Their aggregate transport output is `mixed` because local ring edges retain their local transport while cross-host edges use RDMA. NCCL uses RDMA with `NCCL_NET_GDR_LEVEL=0` (host-pin / no GPUDirect RDMA), matching nano-nccl's registered host FIFO path. No two-host performance acceptance threshold has been established.

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 782.39 | 0.59 | 210.28 | 2.18 | 0.27 |
| 1 MiB | 438.71 | 4.18 | 263.72 | 6.96 | 0.60 |
| 4 MiB | 722.69 | 10.16 | 652.51 | 11.25 | 0.90 |
| 16 MiB | 2676.00 | 10.97 | 2554.28 | 11.49 | 0.95 |
| 64 MiB | 10903.6 | 10.77 | 10244.7 | 11.46 | 0.94 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 203.12 | 2.26 | 212.20 | 2.16 | 1.04 |
| 1 MiB | 382.02 | 4.80 | 276.71 | 6.63 | 0.72 |
| 4 MiB | 730.59 | 10.05 | 685.28 | 10.71 | 0.94 |
| 16 MiB | 2701.47 | 10.87 | 2781.22 | 10.56 | 1.03 |
| 64 MiB | 10875.3 | 10.80 | 12466.0 | 9.42 | 1.15 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 400.96 | 1.14 | 211.05 | 2.17 | 0.53 |
| 1 MiB | 526.46 | 3.49 | 279.80 | 6.56 | 0.53 |
| 4 MiB | 734.35 | 10.00 | 658.33 | 11.15 | 0.90 |
| 16 MiB | 3211.55 | 9.14 | 2801.81 | 10.48 | 0.87 |
| 64 MiB | 10743.7 | 10.93 | 12326.5 | 9.53 | 1.15 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 190.76 | 2.40 | 207.91 | 2.21 | 1.09 |
| 1 MiB | 521.91 | 3.52 | 275.76 | 6.65 | 0.53 |
| 4 MiB | 726.23 | 10.11 | 658.25 | 11.15 | 0.91 |
| 16 MiB | 4188.38 | 7.01 | 2632.17 | 11.15 | 0.63 |
| 64 MiB | 10698.0 | 10.98 | 11143.8 | 10.54 | 1.04 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 199.87 | 2.30 | 204.00 | 2.25 | 1.02 |
| 1 MiB | 480.82 | 3.82 | 269.72 | 6.80 | 0.56 |
| 4 MiB | 729.79 | 10.06 | 657.90 | 11.16 | 0.90 |
| 16 MiB | 4015.04 | 7.31 | 2758.17 | 10.64 | 0.69 |
| 64 MiB | 11065.4 | 10.61 | 12500.1 | 9.40 | 1.13 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 203.78 | 2.25 | 211.93 | 2.16 | 1.04 |
| 1 MiB | 359.62 | 5.10 | 279.58 | 6.56 | 0.78 |
| 4 MiB | 743.47 | 9.87 | 686.94 | 10.69 | 0.92 |
| 16 MiB | 3236.13 | 9.07 | 2664.27 | 11.02 | 0.82 |
| 64 MiB | 11283.1 | 10.41 | 11319.8 | 10.37 | 1.00 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 404.74 | 1.13 | 211.44 | 2.17 | 0.52 |
| 1 MiB | 506.59 | 3.62 | 283.53 | 6.47 | 0.56 |
| 4 MiB | 731.65 | 10.03 | 655.95 | 11.19 | 0.90 |
| 16 MiB | 3207.17 | 9.15 | 2598.56 | 11.30 | 0.81 |
| 64 MiB | 10600.5 | 11.08 | 10783.2 | 10.89 | 1.02 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 205.12 | 2.24 | 212.89 | 2.15 | 1.04 |
| 1 MiB | 509.28 | 3.60 | 270.87 | 6.77 | 0.53 |
| 4 MiB | 735.84 | 9.98 | 659.78 | 11.12 | 0.90 |
| 16 MiB | 3881.30 | 7.56 | 2764.46 | 10.62 | 0.71 |
| 64 MiB | 11224.7 | 10.46 | 12497.6 | 9.40 | 1.11 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 1063.30 | 0.43 | 455.33 | 1.01 | 0.43 |
| 1 MiB | 464.64 | 3.95 | 265.02 | 6.92 | 0.57 |
| 4 MiB | 739.43 | 9.93 | 657.00 | 11.17 | 0.89 |
| 16 MiB | 2754.85 | 10.66 | 2721.25 | 10.79 | 0.99 |
| 64 MiB | 10844.0 | 10.83 | 11439.1 | 10.27 | 1.05 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 191.75 | 2.39 | 189.64 | 2.42 | 0.99 |
| 1 MiB | 557.83 | 3.29 | 259.38 | 7.07 | 0.46 |
| 4 MiB | 722.01 | 10.17 | 681.19 | 10.78 | 0.94 |
| 16 MiB | 2655.07 | 11.06 | 2925.80 | 10.03 | 1.10 |
| 64 MiB | 10568.5 | 11.11 | 10356.9 | 11.34 | 0.98 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 197.10 | 2.33 | 489.76 | 0.94 | 2.48 |
| 1 MiB | 526.84 | 3.48 | 261.67 | 7.01 | 0.50 |
| 4 MiB | 724.36 | 10.13 | 659.13 | 11.14 | 0.91 |
| 16 MiB | 4173.97 | 7.03 | 2564.80 | 11.45 | 0.61 |
| 64 MiB | 11239.8 | 10.45 | 10336.6 | 11.36 | 0.92 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 200.92 | 2.28 | 221.18 | 2.07 | 1.10 |
| 1 MiB | 276.82 | 6.63 | 277.48 | 6.61 | 1.00 |
| 4 MiB | 734.25 | 10.00 | 681.76 | 10.77 | 0.93 |
| 16 MiB | 3914.41 | 7.50 | 2567.65 | 11.43 | 0.66 |
| 64 MiB | 10952.9 | 10.72 | 10357.8 | 11.34 | 0.95 |


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
