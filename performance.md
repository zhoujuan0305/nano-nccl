# Performance

All results below are out-of-place all-reduce measurements. Bandwidth is `busbw` in GB/s. Every measured nano-nccl and NCCL result completed validation with zero wrong values. The `nano/NCCL` column is calculated from the unrounded measured time (`nccl_time_us / nano_time_us`).

## Test Topology And Environment

Two hosts, each two-socket Intel Xeon Platinum 8462Y+ (32 cores per socket, two threads per core), 4x NVIDIA RTX A6000 (SM86), CUDA 12.8.61, NCCL 2.30.7 built from source, nccl-tests 2.19.6, and Open MPI 4.1.2.

| Node | OS kernel | GPU driver | GPUs |
| --- | --- | --- | --- |
| A | Linux 5.15.0-136-generic | 580.82.07 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |
| B | Linux 5.15.0-136-generic | 580.82.07 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |

On each host, GPU0-GPU1 and GPU2-GPU3 are connected by four NVLinks. The two pairs are separated by `SYS` paths across NUMA nodes. Tables are separate transport classes: in-process auto (P2P/SHM) on one host, two-host TCP socket, two-host host-pinned RDMA, and two-host RDMA with GPUDirect (device FIFO).

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


## Two Hosts: 8 Ranks Over TCP Socket

2 hosts x 4 GPUs, one MPI process per host. Nano `--transport auto` keeps local ring edges on P2P/SHM and places cross-host edges on TCP socket. NCCL is Ring+Simple with `NCCL_IB_DISABLE=1` (P2P/SHM allowed intra-node). Bootstrap uses the management IPv4 interface, not loopback.

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4199.45 | 0.11 | 4008.83 | 0.11 | 0.95 |
| 1 MiB | 17074.2 | 0.11 | 16258.5 | 0.11 | 0.95 |
| 4 MiB | 63260.4 | 0.12 | 63166.4 | 0.12 | 1.00 |
| 16 MiB | 264058.5 | 0.11 | 251931.0 | 0.12 | 0.95 |
| 64 MiB | 1096773.3 | 0.11 | 1006180.0 | 0.12 | 0.92 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4181.72 | 0.11 | 4023.11 | 0.11 | 0.96 |
| 1 MiB | 15999.0 | 0.11 | 16033.4 | 0.11 | 1.00 |
| 4 MiB | 63183.6 | 0.12 | 63236.8 | 0.12 | 1.00 |
| 16 MiB | 252113.6 | 0.12 | 251996.0 | 0.12 | 1.00 |
| 64 MiB | 1066403.7 | 0.11 | 1007925.0 | 0.12 | 0.95 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4211.92 | 0.11 | 4047.25 | 0.11 | 0.96 |
| 1 MiB | 15966.7 | 0.11 | 16577.8 | 0.11 | 1.04 |
| 4 MiB | 63239.1 | 0.12 | 66112.7 | 0.11 | 1.05 |
| 16 MiB | 252234.2 | 0.12 | 265972.0 | 0.11 | 1.05 |
| 64 MiB | 1008537.9 | 0.12 | 1029399.0 | 0.11 | 1.02 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4227.29 | 0.11 | 4027.80 | 0.11 | 0.95 |
| 1 MiB | 15951.0 | 0.12 | 16405.9 | 0.11 | 1.03 |
| 4 MiB | 63144.4 | 0.12 | 64499.6 | 0.11 | 1.02 |
| 16 MiB | 251958.6 | 0.12 | 255736.0 | 0.11 | 1.01 |
| 64 MiB | 1071066.9 | 0.11 | 1008520.0 | 0.12 | 0.94 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4212.90 | 0.11 | 4225.55 | 0.11 | 1.00 |
| 1 MiB | 16014.6 | 0.11 | 15808.1 | 0.12 | 0.99 |
| 4 MiB | 66094.9 | 0.11 | 63490.4 | 0.12 | 0.96 |
| 16 MiB | 272025.5 | 0.11 | 252124.0 | 0.12 | 0.93 |
| 64 MiB | 1094216.7 | 0.11 | 1007292.0 | 0.12 | 0.92 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4202.58 | 0.11 | 4034.00 | 0.11 | 0.96 |
| 1 MiB | 16014.1 | 0.11 | 15832.7 | 0.12 | 0.99 |
| 4 MiB | 63209.9 | 0.12 | 63085.1 | 0.12 | 1.00 |
| 16 MiB | 252232.3 | 0.12 | 252049.0 | 0.12 | 1.00 |
| 64 MiB | 1073197.6 | 0.11 | 1007120.0 | 0.12 | 0.94 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4168.72 | 0.11 | 4009.58 | 0.11 | 0.96 |
| 1 MiB | 15983.2 | 0.11 | 15895.0 | 0.12 | 0.99 |
| 4 MiB | 63186.1 | 0.12 | 63457.6 | 0.12 | 1.00 |
| 16 MiB | 252267.5 | 0.12 | 252773.0 | 0.12 | 1.00 |
| 64 MiB | 1037097.5 | 0.11 | 1008294.0 | 0.12 | 0.97 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4230.79 | 0.11 | 4024.61 | 0.11 | 0.95 |
| 1 MiB | 15960.9 | 0.11 | 16166.6 | 0.11 | 1.01 |
| 4 MiB | 63188.8 | 0.12 | 65774.3 | 0.11 | 1.04 |
| 16 MiB | 252151.8 | 0.12 | 253565.0 | 0.12 | 1.01 |
| 64 MiB | 1087603.4 | 0.11 | 1007385.0 | 0.12 | 0.93 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4198.78 | 0.11 | 4009.17 | 0.11 | 0.95 |
| 1 MiB | 15965.1 | 0.11 | 15861.2 | 0.12 | 0.99 |
| 4 MiB | 63345.1 | 0.12 | 64111.9 | 0.11 | 1.01 |
| 16 MiB | 253003.6 | 0.12 | 254344.0 | 0.12 | 1.01 |
| 64 MiB | 1074977.0 | 0.11 | 1010608.0 | 0.12 | 0.94 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4195.62 | 0.11 | 4041.08 | 0.11 | 0.96 |
| 1 MiB | 16261.8 | 0.11 | 16543.4 | 0.11 | 1.02 |
| 4 MiB | 63401.8 | 0.12 | 65651.9 | 0.11 | 1.04 |
| 16 MiB | 260609.5 | 0.11 | 254536.0 | 0.12 | 0.98 |
| 64 MiB | 1040630.3 | 0.11 | 1011167.0 | 0.12 | 0.97 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4217.38 | 0.11 | 3991.62 | 0.11 | 0.95 |
| 1 MiB | 15949.3 | 0.12 | 15940.5 | 0.12 | 1.00 |
| 4 MiB | 63188.8 | 0.12 | 63047.9 | 0.12 | 1.00 |
| 16 MiB | 252100.4 | 0.12 | 251894.0 | 0.12 | 1.00 |
| 64 MiB | 1070230.1 | 0.11 | 1005791.0 | 0.12 | 0.94 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4202.51 | 0.11 | 3993.85 | 0.11 | 0.95 |
| 1 MiB | 16010.6 | 0.11 | 15921.5 | 0.12 | 0.99 |
| 4 MiB | 63180.5 | 0.12 | 63076.2 | 0.12 | 1.00 |
| 16 MiB | 252213.3 | 0.12 | 252373.0 | 0.12 | 1.00 |
| 64 MiB | 1064346.9 | 0.11 | 1006867.0 | 0.12 | 0.95 |


## Two Hosts: 8 Ranks Over RDMA

2 hosts x 4 GPUs, one MPI process per host. Nano `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` (WRITE+CTS over registered host-pinned FIFO). Local edges stay P2P/SHM; cross-host edges are RDMA. RTR `path_mtu` is `min(local, remote) port.active_mtu`. NCCL: Ring+Simple, `NCCL_NET_GDR_LEVEL=0`, P2P/SHM allowed intra-node. NCCL 256 KiB OOP that collapsed (~0.8 GB/s) was re-run isolated up to four times; cells that stayed collapsed are not nano wins.

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 194.33 | 2.36 | 197.33 | 2.32 | 1.02 |
| 1 MiB | 243.74 | 7.53 | 263.40 | 6.97 | 1.08 |
| 4 MiB | 717.83 | 10.23 | 655.29 | 11.20 | 0.91 |
| 16 MiB | 2653.41 | 11.07 | 2556.62 | 11.48 | 0.96 |
| 64 MiB | 10527.7 | 11.16 | 10317.4 | 11.38 | 0.98 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 180.53 | 2.54 | 198.07 | 2.32 | 1.10 |
| 1 MiB | 235.26 | 7.80 | 266.82 | 6.88 | 1.13 |
| 4 MiB | 714.94 | 10.27 | 656.02 | 11.19 | 0.92 |
| 16 MiB | 2652.93 | 11.07 | 2555.88 | 11.49 | 0.96 |
| 64 MiB | 10526.3 | 11.16 | 10356.0 | 11.34 | 0.98 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 182.83 | 2.51 | 198.18 | 2.31 | 1.08 |
| 1 MiB | 236.12 | 7.77 | 267.54 | 6.86 | 1.13 |
| 4 MiB | 714.85 | 10.27 | 661.39 | 11.10 | 0.93 |
| 16 MiB | 2673.26 | 10.98 | 2555.40 | 11.49 | 0.96 |
| 64 MiB | 10529.3 | 11.15 | 10330.7 | 11.37 | 0.98 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 189.80 | 2.42 | 195.94 | 2.34 | 1.03 |
| 1 MiB | 240.70 | 7.62 | 262.09 | 7.00 | 1.09 |
| 4 MiB | 715.37 | 10.26 | 656.77 | 11.18 | 0.92 |
| 16 MiB | 2654.46 | 11.06 | 2556.69 | 11.48 | 0.96 |
| 64 MiB | 10528.1 | 11.15 | 10369.1 | 11.33 | 0.98 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 185.66 | 2.47 | 195.91 | 2.34 | 1.06 |
| 1 MiB | 235.14 | 7.80 | 258.52 | 7.10 | 1.10 |
| 4 MiB | 712.66 | 10.30 | 655.64 | 11.20 | 0.92 |
| 16 MiB | 2651.20 | 11.07 | 2556.31 | 11.49 | 0.96 |
| 64 MiB | 10527.6 | 11.16 | 10305.0 | 11.40 | 0.98 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 184.14 | 2.49 | 199.03 | 2.30 | 1.08 |
| 1 MiB | 240.27 | 7.64 | 267.44 | 6.86 | 1.11 |
| 4 MiB | 715.07 | 10.26 | 660.60 | 11.11 | 0.92 |
| 16 MiB | 2656.19 | 11.05 | 2756.45 | 10.65 | 1.04 |
| 64 MiB | 10529.0 | 11.15 | 10248.6 | 11.46 | 0.97 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 188.28 | 2.44 | 195.04 | 2.35 | 1.04 |
| 1 MiB | 239.36 | 7.67 | 264.28 | 6.94 | 1.10 |
| 4 MiB | 715.53 | 10.26 | 655.52 | 11.20 | 0.92 |
| 16 MiB | 2653.35 | 11.07 | 2554.61 | 11.49 | 0.96 |
| 64 MiB | 10529.7 | 11.15 | 10350.8 | 11.35 | 0.98 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 181.61 | 2.53 | 196.59 | 2.33 | 1.08 |
| 1 MiB | 234.96 | 7.81 | 266.98 | 6.87 | 1.14 |
| 4 MiB | 717.67 | 10.23 | 655.48 | 11.20 | 0.91 |
| 16 MiB | 2653.90 | 11.06 | 2665.47 | 11.01 | 1.00 |
| 64 MiB | 10523.8 | 11.16 | 10272.8 | 11.43 | 0.98 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 187.54 | 2.45 | 193.50 | 2.37 | 1.03 |
| 1 MiB | 239.51 | 7.66 | 262.31 | 7.00 | 1.10 |
| 4 MiB | 716.89 | 10.24 | 655.72 | 11.19 | 0.91 |
| 16 MiB | 2649.11 | 11.08 | 2556.68 | 11.48 | 0.97 |
| 64 MiB | 10525.8 | 11.16 | 10250.3 | 11.46 | 0.97 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 180.02 | 2.55 | 197.01 | 2.33 | 1.09 |
| 1 MiB | 234.13 | 7.84 | 259.17 | 7.08 | 1.11 |
| 4 MiB | 782.37 | 9.38 | 653.96 | 11.22 | 0.84 |
| 16 MiB | 2652.68 | 11.07 | 2556.68 | 11.48 | 0.96 |
| 64 MiB | 10528.7 | 11.15 | 10305.2 | 11.40 | 0.98 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 188.45 | 2.43 | 197.19 | 2.33 | 1.05 |
| 1 MiB | 237.85 | 7.72 | 262.75 | 6.98 | 1.10 |
| 4 MiB | 712.68 | 10.30 | 667.91 | 10.99 | 0.94 |
| 16 MiB | 2649.80 | 11.08 | 2578.54 | 11.39 | 0.97 |
| 64 MiB | 10523.8 | 11.16 | 10271.5 | 11.43 | 0.98 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 181.05 | 2.53 | 194.77 | 2.36 | 1.08 |
| 1 MiB | 233.13 | 7.87 | 277.19 | 6.62 | 1.19 |
| 4 MiB | 715.75 | 10.25 | 659.24 | 11.13 | 0.92 |
| 16 MiB | 2661.14 | 11.03 | 2559.70 | 11.47 | 0.96 |
| 64 MiB | 10528.4 | 11.15 | 10264.2 | 11.44 | 0.97 |


## Two Hosts: 8 Ranks Over RDMA (GDR)

Same 2x4 topology as the host-pinned RDMA table. Nano `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` and `NANO_NCCL_RDMA_GDR=1` (WRITE+CTS from a registered GPU FIFO; host proxy still posts). NCCL: Ring+Simple, `NCCL_NET_GDR_LEVEL=LOC`. Collapsed NCCL 256 KiB OOP cells were re-run isolated. This is host-proxy GDR, not GPU-initiated IBGDA.

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 179.71 | 2.55 | 198.27 | 2.31 | 1.10 |
| 1 MiB | 220.82 | 8.31 | 264.53 | 6.94 | 1.20 |
| 4 MiB | 692.08 | 10.61 | 655.25 | 11.20 | 0.95 |
| 16 MiB | 2605.97 | 11.27 | 2556.19 | 11.49 | 0.98 |
| 64 MiB | 10300.8 | 11.40 | 10263.2 | 11.44 | 1.00 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 167.88 | 2.73 | 198.92 | 2.31 | 1.18 |
| 1 MiB | 212.66 | 8.63 | 263.27 | 6.97 | 1.24 |
| 4 MiB | 689.83 | 10.64 | 656.12 | 11.19 | 0.95 |
| 16 MiB | 2603.82 | 11.28 | 2556.68 | 11.48 | 0.98 |
| 64 MiB | 10297.2 | 11.41 | 10263.0 | 11.44 | 1.00 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 176.00 | 2.61 | 206.55 | 2.22 | 1.17 |
| 1 MiB | 218.60 | 8.39 | 261.68 | 7.01 | 1.20 |
| 4 MiB | 692.27 | 10.60 | 655.70 | 11.19 | 0.95 |
| 16 MiB | 2620.14 | 11.21 | 2557.70 | 11.48 | 0.98 |
| 64 MiB | 10296.8 | 11.41 | 10245.0 | 11.46 | 0.99 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 178.31 | 2.57 | 197.33 | 2.32 | 1.11 |
| 1 MiB | 218.96 | 8.38 | 263.45 | 6.97 | 1.20 |
| 4 MiB | 694.25 | 10.57 | 657.00 | 11.17 | 0.95 |
| 16 MiB | 2698.03 | 10.88 | 2557.59 | 11.48 | 0.95 |
| 64 MiB | 10300.3 | 11.40 | 10310.4 | 11.39 | 1.00 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 173.03 | 2.65 | 197.12 | 2.33 | 1.14 |
| 1 MiB | 218.97 | 8.38 | 272.44 | 6.74 | 1.24 |
| 4 MiB | 693.52 | 10.58 | 655.24 | 11.20 | 0.94 |
| 16 MiB | 2622.74 | 11.19 | 2555.79 | 11.49 | 0.97 |
| 64 MiB | 10297.1 | 11.41 | 10283.9 | 11.42 | 1.00 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 179.31 | 2.56 | 196.22 | 2.34 | 1.09 |
| 1 MiB | 221.23 | 8.29 | 268.07 | 6.85 | 1.21 |
| 4 MiB | 693.64 | 10.58 | 655.74 | 11.19 | 0.95 |
| 16 MiB | 2729.80 | 10.76 | 2556.37 | 11.49 | 0.94 |
| 64 MiB | 10303.3 | 11.40 | 10310.1 | 11.39 | 1.00 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 183.50 | 2.50 | 196.15 | 2.34 | 1.07 |
| 1 MiB | 220.28 | 8.33 | 265.01 | 6.92 | 1.20 |
| 4 MiB | 692.62 | 10.60 | 655.79 | 11.19 | 0.95 |
| 16 MiB | 2690.67 | 10.91 | 2554.99 | 11.49 | 0.95 |
| 64 MiB | 10304.6 | 11.40 | 10262.1 | 11.44 | 1.00 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 167.28 | 2.74 | 206.26 | 2.22 | 1.23 |
| 1 MiB | 213.32 | 8.60 | 263.43 | 6.97 | 1.23 |
| 4 MiB | 692.17 | 10.60 | 655.23 | 11.20 | 0.95 |
| 16 MiB | 2699.08 | 10.88 | 2555.70 | 11.49 | 0.95 |
| 64 MiB | 10301.2 | 11.40 | 10262.3 | 11.44 | 1.00 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 167.53 | 2.74 | 196.07 | 2.34 | 1.17 |
| 1 MiB | 213.48 | 8.60 | 263.60 | 6.96 | 1.23 |
| 4 MiB | 692.20 | 10.60 | 656.44 | 11.18 | 0.95 |
| 16 MiB | 2670.53 | 10.99 | 2555.89 | 11.49 | 0.96 |
| 64 MiB | 10323.9 | 11.38 | 10296.7 | 11.41 | 1.00 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 174.95 | 2.62 | 198.99 | 2.31 | 1.14 |
| 1 MiB | 220.40 | 8.33 | 256.57 | 7.15 | 1.16 |
| 4 MiB | 692.44 | 10.60 | 658.26 | 11.15 | 0.95 |
| 16 MiB | 2797.12 | 10.50 | 2556.07 | 11.49 | 0.91 |
| 64 MiB | 10303.5 | 11.40 | 10249.3 | 11.46 | 0.99 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 175.04 | 2.62 | 195.53 | 2.35 | 1.12 |
| 1 MiB | 217.53 | 8.44 | 267.79 | 6.85 | 1.23 |
| 4 MiB | 692.66 | 10.60 | 656.62 | 11.18 | 0.95 |
| 16 MiB | 2633.30 | 11.15 | 2556.05 | 11.49 | 0.97 |
| 64 MiB | 10299.8 | 11.40 | 10258.4 | 11.45 | 1.00 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 179.67 | 2.55 | 200.35 | 2.29 | 1.12 |
| 1 MiB | 220.12 | 8.34 | 266.85 | 6.88 | 1.21 |
| 4 MiB | 695.27 | 10.56 | 657.19 | 11.17 | 0.95 |
| 16 MiB | 2719.40 | 10.80 | 2556.58 | 11.48 | 0.94 |
| 64 MiB | 10301.2 | 11.40 | 10252.7 | 11.45 | 1.00 |


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
cmake -S . -B build-perf-rdma-n8 -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON \
  -DNANO_NCCL_NRANKS=8 -DNANO_NCCL_CUDA_ARCH=86 \
  -DNANO_NCCL_ENABLE_BENCH_PROFILING=OFF
cmake --build build-perf-rdma-n8 -j<jobs>
```

Or regenerate this file from a completed matrix JSON:

```bash
scripts/run_performance_matrix.sh --nccl-bin <path> --nccl-lib <dir> --out-dir <out>
python3 scripts/render_performance_md.py <out>/matrix.json -o performance.md
```
