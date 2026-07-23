# Performance

All results below are out-of-place all-reduce measurements. Bandwidth is `busbw` in GB/s. Every measured nano-nccl and NCCL result completed validation with zero wrong values. The `nano/NCCL` column is calculated from the unrounded measured time.

## Test Topology And Environment

Measurements were taken on 2026-07-22.

Both hosts use two-socket Intel Xeon Platinum 8462Y+ CPUs (32 cores per socket, two threads per core), 4x NVIDIA RTX A6000 GPUs (SM86), CUDA 12.8.61, NCCL 2.30.7 built from source, nccl-tests 2.19.6, and Open MPI 4.1.2.

| Node | OS kernel | GPU driver | GPUs |
| --- | --- | --- | --- |
| A | Linux 5.15.0-136-generic | 580.82.07 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |
| B | Linux 6.8.0-111-generic | 580.159.03 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |

On each host GPU0-GPU1 and GPU2-GPU3 are connected by four NVLinks. The two pairs are separated by `SYS` paths across NUMA nodes. The nano-nccl `auto` plan resolved to `mixed` on the single host and to P2P/SHM/socket `mixed` paths across the two-host ring. The distributed runs use TCP sockets; NCCL has `NCCL_IB_DISABLE=1`.

All measurements use a Release build with `NANO_NCCL_ENABLE_BENCH_PROFILING=OFF`, message sizes 256 KiB through 64 MiB, `-w 5`, and `-n 20`. NCCL uses `Ring`, `Simple`, four channels, and a 32 MiB buffer.

## Single Host: 4 Ranks

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 58.30 | 6.75 | 69.94 | 5.62 | 1.20 |
| 1 MiB | 109.51 | 14.36 | 122.17 | 12.87 | 1.12 |
| 4 MiB | 326.37 | 19.28 | 332.04 | 18.95 | 1.02 |
| 16 MiB | 1120.64 | 22.46 | 1117.53 | 22.52 | 1.00 |
| 64 MiB | 4321.19 | 23.30 | 4402.58 | 22.86 | 1.02 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 57.92 | 6.79 | 70.69 | 5.56 | 1.22 |
| 1 MiB | 107.40 | 14.65 | 123.59 | 12.73 | 1.15 |
| 4 MiB | 325.54 | 19.33 | 331.61 | 18.97 | 1.02 |
| 16 MiB | 1119.91 | 22.47 | 1117.69 | 22.52 | 1.00 |
| 64 MiB | 4312.14 | 23.34 | 4411.96 | 22.82 | 1.02 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 56.66 | 6.94 | 70.27 | 5.60 | 1.24 |
| 1 MiB | 108.17 | 14.54 | 121.78 | 12.92 | 1.13 |
| 4 MiB | 324.41 | 19.39 | 330.25 | 19.05 | 1.02 |
| 16 MiB | 1121.42 | 22.44 | 1121.51 | 22.44 | 1.00 |
| 64 MiB | 4323.16 | 23.28 | 4412.87 | 22.81 | 1.02 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 56.67 | 6.94 | 71.43 | 5.51 | 1.26 |
| 1 MiB | 106.61 | 14.75 | 120.56 | 13.05 | 1.13 |
| 4 MiB | 324.46 | 19.39 | 330.97 | 19.01 | 1.02 |
| 16 MiB | 1121.85 | 22.43 | 1120.15 | 22.47 | 1.00 |
| 64 MiB | 4319.16 | 23.31 | 4408.05 | 22.84 | 1.02 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 58.44 | 6.73 | 68.66 | 5.73 | 1.17 |
| 1 MiB | 108.88 | 14.45 | 121.15 | 12.98 | 1.11 |
| 4 MiB | 325.62 | 19.32 | 328.39 | 19.16 | 1.01 |
| 16 MiB | 1120.38 | 22.46 | 1107.81 | 22.72 | 0.99 |
| 64 MiB | 4308.14 | 23.37 | 4372.63 | 23.02 | 1.01 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 58.59 | 6.71 | 70.05 | 5.61 | 1.20 |
| 1 MiB | 110.81 | 14.19 | 122.23 | 12.87 | 1.10 |
| 4 MiB | 327.13 | 19.23 | 328.88 | 19.13 | 1.01 |
| 16 MiB | 1119.62 | 22.48 | 1110.11 | 22.67 | 0.99 |
| 64 MiB | 4317.57 | 23.31 | 4371.74 | 23.03 | 1.01 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 58.38 | 6.74 | 71.64 | 5.49 | 1.23 |
| 1 MiB | 109.49 | 14.36 | 123.68 | 12.72 | 1.13 |
| 4 MiB | 325.48 | 19.33 | 328.41 | 19.16 | 1.01 |
| 16 MiB | 1125.03 | 22.37 | 1110.86 | 22.65 | 0.99 |
| 64 MiB | 4379.26 | 22.99 | 4373.98 | 23.01 | 1.00 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 57.84 | 6.80 | 71.36 | 5.51 | 1.23 |
| 1 MiB | 108.97 | 14.43 | 125.02 | 12.58 | 1.15 |
| 4 MiB | 326.74 | 19.26 | 329.51 | 19.09 | 1.01 |
| 16 MiB | 1130.15 | 22.27 | 1112.59 | 22.62 | 0.98 |
| 64 MiB | 4378.73 | 22.99 | 4376.42 | 23.00 | 1.00 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 58.53 | 6.72 | 67.89 | 5.79 | 1.16 |
| 1 MiB | 107.08 | 14.69 | 123.22 | 12.76 | 1.15 |
| 4 MiB | 326.62 | 19.26 | 331.40 | 18.98 | 1.01 |
| 16 MiB | 1119.98 | 22.47 | 1114.61 | 22.58 | 1.00 |
| 64 MiB | 4313.15 | 23.34 | 4498.15 | 22.38 | 1.04 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 58.18 | 6.76 | 69.61 | 5.65 | 1.20 |
| 1 MiB | 109.18 | 14.41 | 122.81 | 12.81 | 1.12 |
| 4 MiB | 326.03 | 19.30 | 332.50 | 18.92 | 1.02 |
| 16 MiB | 1120.29 | 22.46 | 1115.93 | 22.55 | 1.00 |
| 64 MiB | 4319.71 | 23.30 | 4388.95 | 22.94 | 1.02 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 58.76 | 6.69 | 72.52 | 5.42 | 1.23 |
| 1 MiB | 109.46 | 14.37 | 124.44 | 12.64 | 1.14 |
| 4 MiB | 327.85 | 19.19 | 330.15 | 19.06 | 1.01 |
| 16 MiB | 1137.58 | 22.12 | 1116.07 | 22.55 | 0.98 |
| 64 MiB | 4424.93 | 22.75 | 4392.74 | 22.92 | 0.99 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 58.96 | 6.67 | 70.90 | 5.55 | 1.20 |
| 1 MiB | 109.05 | 14.42 | 124.15 | 12.67 | 1.14 |
| 4 MiB | 327.91 | 19.19 | 331.39 | 18.98 | 1.01 |
| 16 MiB | 1141.52 | 22.05 | 1118.18 | 22.51 | 0.98 |
| 64 MiB | 4423.54 | 22.76 | 4398.27 | 22.89 | 0.99 |

## Two Hosts: 8 Ranks Over TCP Socket

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 27252.12 | 0.02 | 4027.47 | 0.11 | 0.15 |
| 1 MiB | 26808.62 | 0.07 | 16690.20 | 0.11 | 0.62 |
| 4 MiB | 73644.58 | 0.10 | 65372.40 | 0.11 | 0.89 |
| 16 MiB | 290761.75 | 0.10 | 252612.00 | 0.12 | 0.87 |
| 64 MiB | 1082645.27 | 0.11 | 1007645.00 | 0.12 | 0.93 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 28906.84 | 0.02 | 4002.27 | 0.11 | 0.14 |
| 1 MiB | 31697.44 | 0.06 | 15847.90 | 0.12 | 0.50 |
| 4 MiB | 78013.15 | 0.09 | 63297.00 | 0.12 | 0.81 |
| 16 MiB | 275141.00 | 0.11 | 253709.00 | 0.12 | 0.92 |
| 64 MiB | 1097478.11 | 0.11 | 1010271.00 | 0.12 | 0.92 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 30220.20 | 0.02 | 4023.08 | 0.11 | 0.13 |
| 1 MiB | 23281.40 | 0.08 | 16008.40 | 0.11 | 0.69 |
| 4 MiB | 66598.51 | 0.11 | 63581.90 | 0.12 | 0.95 |
| 16 MiB | 253615.12 | 0.12 | 253898.00 | 0.12 | 1.00 |
| 64 MiB | 1010477.29 | 0.12 | 1009121.00 | 0.12 | 1.00 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 20662.01 | 0.02 | 4048.23 | 0.11 | 0.20 |
| 1 MiB | 29647.55 | 0.06 | 16677.20 | 0.11 | 0.56 |
| 4 MiB | 74035.66 | 0.10 | 64706.30 | 0.11 | 0.87 |
| 16 MiB | 279835.70 | 0.10 | 255763.00 | 0.11 | 0.91 |
| 64 MiB | 1060956.78 | 0.11 | 1007911.00 | 0.12 | 0.95 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 30162.10 | 0.02 | 4007.65 | 0.11 | 0.13 |
| 1 MiB | 29045.56 | 0.06 | 15844.20 | 0.12 | 0.55 |
| 4 MiB | 74996.66 | 0.10 | 63167.80 | 0.12 | 0.84 |
| 16 MiB | 264902.26 | 0.11 | 252087.00 | 0.12 | 0.95 |
| 64 MiB | 1010803.28 | 0.12 | 1006240.00 | 0.12 | 1.00 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 36792.77 | 0.01 | 3998.28 | 0.11 | 0.11 |
| 1 MiB | 31647.49 | 0.06 | 16818.70 | 0.11 | 0.53 |
| 4 MiB | 63669.82 | 0.12 | 63635.90 | 0.12 | 1.00 |
| 16 MiB | 266239.43 | 0.11 | 253663.00 | 0.12 | 0.95 |
| 64 MiB | 1014392.78 | 0.12 | 1008920.00 | 0.12 | 0.99 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 25039.53 | 0.02 | 4051.68 | 0.11 | 0.16 |
| 1 MiB | 23399.01 | 0.08 | 15803.40 | 0.12 | 0.68 |
| 4 MiB | 63923.13 | 0.11 | 63627.70 | 0.12 | 1.00 |
| 16 MiB | 263619.95 | 0.11 | 252942.00 | 0.12 | 0.96 |
| 64 MiB | 1075918.55 | 0.11 | 1007632.00 | 0.12 | 0.94 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 26350.20 | 0.02 | 4024.01 | 0.11 | 0.15 |
| 1 MiB | 23635.46 | 0.08 | 16173.40 | 0.11 | 0.68 |
| 4 MiB | 76189.91 | 0.10 | 63408.60 | 0.12 | 0.83 |
| 16 MiB | 254011.86 | 0.12 | 252738.00 | 0.12 | 0.99 |
| 64 MiB | 1071362.78 | 0.11 | 1007143.00 | 0.12 | 0.94 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 29852.53 | 0.02 | 3993.67 | 0.11 | 0.13 |
| 1 MiB | 23206.84 | 0.08 | 15858.10 | 0.12 | 0.68 |
| 4 MiB | 70779.04 | 0.10 | 65519.30 | 0.11 | 0.93 |
| 16 MiB | 276717.75 | 0.11 | 252376.00 | 0.12 | 0.91 |
| 64 MiB | 1135852.20 | 0.10 | 1007258.00 | 0.12 | 0.89 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 27757.75 | 0.02 | 4000.55 | 0.11 | 0.14 |
| 1 MiB | 29661.53 | 0.06 | 15790.20 | 0.12 | 0.53 |
| 4 MiB | 71672.21 | 0.10 | 63470.20 | 0.12 | 0.89 |
| 16 MiB | 254081.02 | 0.12 | 252948.00 | 0.12 | 1.00 |
| 64 MiB | 1121259.98 | 0.10 | 1007197.00 | 0.12 | 0.90 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 21840.70 | 0.02 | 3981.46 | 0.12 | 0.18 |
| 1 MiB | 35377.69 | 0.05 | 15743.90 | 0.12 | 0.45 |
| 4 MiB | 63275.35 | 0.12 | 62803.10 | 0.12 | 0.99 |
| 16 MiB | 286907.65 | 0.10 | 251108.00 | 0.12 | 0.88 |
| 64 MiB | 1106327.17 | 0.11 | 1004368.00 | 0.12 | 0.91 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 27626.48 | 0.02 | 4029.53 | 0.11 | 0.15 |
| 1 MiB | 31455.49 | 0.06 | 15972.60 | 0.11 | 0.51 |
| 4 MiB | 75246.13 | 0.10 | 63148.10 | 0.12 | 0.84 |
| 16 MiB | 292737.02 | 0.10 | 251809.00 | 0.12 | 0.86 |
| 64 MiB | 1131329.61 | 0.10 | 1007667.00 | 0.12 | 0.89 |

## Reproduction

Build nano-nccl with CUDA 12.8, SM86, Release mode, and profiling disabled. The single-host binary uses four ranks. The distributed binary on both hosts uses eight global ranks and the same Open MPI 4.1.2 prefix.

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

For two hosts, launch one 4-GPU process on each host with an Open MPI 4.1.2 launcher, set `NANO_NCCL_SOCKET_IFNAME=<interface>` for nano-nccl, and set `NCCL_SOCKET_IFNAME=<interface>`, `NCCL_IB_DISABLE=1`, `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, and `NCCL_BUFFSIZE=33554432` for NCCL. Both commands must set `LD_LIBRARY_PATH` to the matching NCCL and Open MPI library directories.
