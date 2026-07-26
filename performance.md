# Performance

All results below are out-of-place all-reduce measurements. Bandwidth is `busbw` in GB/s. Every measured nano-nccl and NCCL result completed validation with zero wrong values. The `nano/NCCL` column is calculated from the unrounded measured time.

## Test Topology And Environment

Both hosts use two-socket Intel Xeon Platinum 8462Y+ CPUs (32 cores per socket, two threads per core), 4x NVIDIA RTX A6000 GPUs (SM86), CUDA 12.8.61, NCCL 2.30.7 built from source, nccl-tests 2.19.6, and Open MPI 4.1.2.

| Node | OS kernel | GPU driver | GPUs |
| --- | --- | --- | --- |
| A | Linux 5.15.0-136-generic | 580.82.07 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |
| B | Linux 6.8.0-111-generic | 580.173.02 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |

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
| 256 KiB | 4184.23 | 0.11 | 4021.96 | 0.11 | 0.96 |
| 1 MiB | 15981.39 | 0.11 | 15932.30 | 0.12 | 1.00 |
| 4 MiB | 63410.73 | 0.12 | 63604.50 | 0.12 | 1.00 |
| 16 MiB | 280561.68 | 0.10 | 252330.00 | 0.12 | 0.90 |
| 64 MiB | 1110723.13 | 0.11 | 1005487.00 | 0.12 | 0.91 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4185.38 | 0.11 | 3999.31 | 0.11 | 0.96 |
| 1 MiB | 15946.47 | 0.12 | 16133.00 | 0.11 | 1.01 |
| 4 MiB | 63162.30 | 0.12 | 63331.30 | 0.12 | 1.00 |
| 16 MiB | 252053.61 | 0.12 | 253241.00 | 0.12 | 1.00 |
| 64 MiB | 1068414.85 | 0.11 | 1004605.00 | 0.12 | 0.94 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4167.49 | 0.11 | 4009.70 | 0.11 | 0.96 |
| 1 MiB | 15973.78 | 0.11 | 16066.30 | 0.11 | 1.01 |
| 4 MiB | 63147.18 | 0.12 | 65324.60 | 0.11 | 1.03 |
| 16 MiB | 252136.02 | 0.12 | 254913.00 | 0.12 | 1.01 |
| 64 MiB | 1075747.36 | 0.11 | 1006728.00 | 0.12 | 0.94 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4184.45 | 0.11 | 3990.34 | 0.11 | 0.95 |
| 1 MiB | 16120.14 | 0.11 | 15726.60 | 0.12 | 0.98 |
| 4 MiB | 63199.79 | 0.12 | 62728.40 | 0.12 | 0.99 |
| 16 MiB | 252172.77 | 0.12 | 250524.00 | 0.12 | 0.99 |
| 64 MiB | 1007794.11 | 0.12 | 1001545.00 | 0.12 | 0.99 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4176.28 | 0.11 | 3983.18 | 0.12 | 0.95 |
| 1 MiB | 16294.99 | 0.11 | 15756.90 | 0.12 | 0.97 |
| 4 MiB | 64824.52 | 0.11 | 62923.00 | 0.12 | 0.97 |
| 16 MiB | 262021.29 | 0.11 | 252016.00 | 0.12 | 0.96 |
| 64 MiB | 1046875.94 | 0.11 | 1004148.00 | 0.12 | 0.96 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4165.83 | 0.11 | 3993.31 | 0.11 | 0.96 |
| 1 MiB | 15970.30 | 0.11 | 15776.20 | 0.12 | 0.99 |
| 4 MiB | 63183.33 | 0.12 | 63049.30 | 0.12 | 1.00 |
| 16 MiB | 252025.12 | 0.12 | 251897.00 | 0.12 | 1.00 |
| 64 MiB | 1007348.92 | 0.12 | 1005341.00 | 0.12 | 1.00 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4180.97 | 0.11 | 3982.46 | 0.12 | 0.95 |
| 1 MiB | 15992.83 | 0.11 | 15838.80 | 0.12 | 0.99 |
| 4 MiB | 63195.04 | 0.12 | 63008.30 | 0.12 | 1.00 |
| 16 MiB | 252054.85 | 0.12 | 250999.00 | 0.12 | 1.00 |
| 64 MiB | 1007491.79 | 0.12 | 1003307.00 | 0.12 | 1.00 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4204.17 | 0.11 | 4017.16 | 0.11 | 0.96 |
| 1 MiB | 16000.07 | 0.11 | 16800.50 | 0.11 | 1.05 |
| 4 MiB | 63259.23 | 0.12 | 65626.40 | 0.11 | 1.04 |
| 16 MiB | 252329.28 | 0.12 | 253631.00 | 0.12 | 1.01 |
| 64 MiB | 1079674.32 | 0.11 | 1006518.00 | 0.12 | 0.93 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4204.25 | 0.11 | 4003.40 | 0.11 | 0.95 |
| 1 MiB | 15989.30 | 0.11 | 15847.80 | 0.12 | 0.99 |
| 4 MiB | 63191.63 | 0.12 | 65236.50 | 0.11 | 1.03 |
| 16 MiB | 252405.32 | 0.12 | 251883.00 | 0.12 | 1.00 |
| 64 MiB | 1096876.07 | 0.11 | 1005809.00 | 0.12 | 0.92 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4168.96 | 0.11 | 4010.28 | 0.11 | 0.96 |
| 1 MiB | 16045.44 | 0.11 | 16131.90 | 0.11 | 1.01 |
| 4 MiB | 66449.38 | 0.11 | 65772.90 | 0.11 | 0.99 |
| 16 MiB | 274920.81 | 0.11 | 253556.00 | 0.12 | 0.92 |
| 64 MiB | 1123350.09 | 0.10 | 1005908.00 | 0.12 | 0.90 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4189.75 | 0.11 | 4017.59 | 0.11 | 0.96 |
| 1 MiB | 15988.77 | 0.11 | 16375.50 | 0.11 | 1.02 |
| 4 MiB | 63253.04 | 0.12 | 64985.90 | 0.11 | 1.03 |
| 16 MiB | 253163.93 | 0.12 | 260351.00 | 0.11 | 1.03 |
| 64 MiB | 1008121.73 | 0.12 | 1008367.00 | 0.12 | 1.00 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4168.85 | 0.11 | 3961.22 | 0.12 | 0.95 |
| 1 MiB | 15983.46 | 0.11 | 15707.20 | 0.12 | 0.98 |
| 4 MiB | 63219.34 | 0.12 | 62658.00 | 0.12 | 0.99 |
| 16 MiB | 252270.61 | 0.12 | 250287.00 | 0.12 | 0.99 |
| 64 MiB | 1007692.38 | 0.12 | 1000984.00 | 0.12 | 0.99 |

## Two Hosts: 8 Ranks Over RDMA

The nano-nccl runs explicitly request `--transport rdma`. Their aggregate transport output is `mixed` because local ring edges retain their local transport while cross-host edges use RDMA. NCCL uses RDMA. No two-host performance acceptance threshold has been established.

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 462.79 | 0.99 | 541.05 | 0.85 | 1.17 |
| 1 MiB | 620.00 | 2.96 | 269.69 | 6.80 | 0.43 |
| 4 MiB | 1423.53 | 5.16 | 674.38 | 10.88 | 0.47 |
| 16 MiB | 3217.55 | 9.12 | 2562.40 | 11.46 | 0.80 |
| 64 MiB | 10627.07 | 11.05 | 10231.20 | 11.48 | 0.96 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 380.07 | 1.21 | 197.87 | 2.32 | 0.52 |
| 1 MiB | 738.61 | 2.48 | 268.01 | 6.85 | 0.36 |
| 4 MiB | 1324.61 | 5.54 | 675.50 | 10.87 | 0.51 |
| 16 MiB | 3244.73 | 9.05 | 2562.79 | 11.46 | 0.79 |
| 64 MiB | 10626.01 | 11.05 | 10221.80 | 11.49 | 0.96 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 1051.56 | 0.44 | 176.10 | 2.61 | 0.17 |
| 1 MiB | 556.63 | 3.30 | 245.01 | 7.49 | 0.44 |
| 4 MiB | 1433.85 | 5.12 | 653.45 | 11.23 | 0.46 |
| 16 MiB | 3267.64 | 8.99 | 2553.76 | 11.50 | 0.78 |
| 64 MiB | 11185.24 | 10.50 | 10222.10 | 11.49 | 0.91 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 489.53 | 0.94 | 177.72 | 2.58 | 0.36 |
| 1 MiB | 603.85 | 3.04 | 251.18 | 7.31 | 0.42 |
| 4 MiB | 879.71 | 8.34 | 652.04 | 11.26 | 0.74 |
| 16 MiB | 3219.60 | 9.12 | 2553.63 | 11.50 | 0.79 |
| 64 MiB | 10632.64 | 11.05 | 10219.90 | 11.49 | 0.96 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 767.16 | 0.60 | 178.75 | 2.57 | 0.23 |
| 1 MiB | 793.63 | 2.31 | 250.08 | 7.34 | 0.32 |
| 4 MiB | 1322.24 | 5.55 | 653.25 | 11.24 | 0.49 |
| 16 MiB | 3264.23 | 8.99 | 2552.86 | 11.50 | 0.78 |
| 64 MiB | 11180.63 | 10.50 | 10219.20 | 11.49 | 0.91 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 854.63 | 0.54 | 178.57 | 2.57 | 0.21 |
| 1 MiB | 596.55 | 3.08 | 270.59 | 6.78 | 0.45 |
| 4 MiB | 1320.48 | 5.56 | 675.75 | 10.86 | 0.51 |
| 16 MiB | 2696.97 | 10.89 | 2560.14 | 11.47 | 0.95 |
| 64 MiB | 11193.81 | 10.49 | 10220.50 | 11.49 | 0.91 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 979.43 | 0.47 | 176.85 | 2.59 | 0.18 |
| 1 MiB | 557.25 | 3.29 | 253.81 | 7.23 | 0.46 |
| 4 MiB | 804.40 | 9.12 | 653.27 | 11.24 | 0.81 |
| 16 MiB | 3226.48 | 9.10 | 2554.56 | 11.49 | 0.79 |
| 64 MiB | 11208.52 | 10.48 | 10220.20 | 11.49 | 0.91 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 955.21 | 0.48 | 198.03 | 2.32 | 0.21 |
| 1 MiB | 652.37 | 2.81 | 262.95 | 6.98 | 0.40 |
| 4 MiB | 794.06 | 9.24 | 676.16 | 10.86 | 0.85 |
| 16 MiB | 2694.35 | 10.90 | 2552.31 | 11.50 | 0.95 |
| 64 MiB | 11161.83 | 10.52 | 10301.20 | 11.40 | 0.92 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 863.30 | 0.53 | 529.20 | 0.87 | 0.61 |
| 1 MiB | 455.09 | 4.03 | 250.68 | 7.32 | 0.55 |
| 4 MiB | 797.97 | 9.20 | 651.14 | 11.27 | 0.82 |
| 16 MiB | 2703.33 | 10.86 | 2554.46 | 11.49 | 0.94 |
| 64 MiB | 11162.84 | 10.52 | 10284.80 | 11.42 | 0.92 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 886.01 | 0.52 | 198.93 | 2.31 | 0.22 |
| 1 MiB | 570.10 | 3.22 | 256.37 | 7.16 | 0.45 |
| 4 MiB | 1322.45 | 5.55 | 652.39 | 11.25 | 0.49 |
| 16 MiB | 2697.94 | 10.88 | 2563.22 | 11.45 | 0.95 |
| 64 MiB | 11252.97 | 10.44 | 10434.10 | 11.26 | 0.93 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 836.83 | 0.55 | 184.09 | 2.49 | 0.22 |
| 1 MiB | 456.99 | 4.02 | 252.15 | 7.28 | 0.55 |
| 4 MiB | 1306.60 | 5.62 | 651.25 | 11.27 | 0.50 |
| 16 MiB | 3260.97 | 9.00 | 2552.37 | 11.50 | 0.78 |
| 64 MiB | 11178.50 | 10.51 | 10224.50 | 11.49 | 0.91 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 866.01 | 0.53 | 199.33 | 2.30 | 0.23 |
| 1 MiB | 455.84 | 4.03 | 268.03 | 6.85 | 0.59 |
| 4 MiB | 1992.90 | 3.68 | 676.06 | 10.86 | 0.34 |
| 16 MiB | 3225.10 | 9.10 | 2553.65 | 11.50 | 0.79 |
| 64 MiB | 10607.48 | 11.07 | 10229.20 | 11.48 | 0.96 |

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

For two hosts over TCP Socket, launch one 4-GPU process on each host with an Open MPI 4.1.2 launcher, set `NANO_NCCL_SOCKET_IFNAME=<interface>` for nano-nccl, and set `NCCL_SOCKET_IFNAME=<interface>`, `NCCL_IB_DISABLE=1`, `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, and `NCCL_BUFFSIZE=33554432` for NCCL. Both commands must set `LD_LIBRARY_PATH` to the matching NCCL and Open MPI library directories.

For RDMA, build nano-nccl with MPI and RDMA enabled on both hosts. Launch one 4-GPU process per host; both hosts must use matching local NCCL and nccl-tests builds. Set `NANO_NCCL_SOCKET_IFNAME=<interface>` for bootstrap and `NANO_NCCL_RDMA_IFNAME=<rdma-interface>` for nano-nccl. Set `NCCL_SOCKET_IFNAME=<interface>` and `NCCL_IB_HCA=<rdma-hca>` for NCCL. The former is a network interface and the latter is an HCA name. The command clears any inherited `NCCL_IB_DISABLE` value.

```bash
cmake -S . -B build-perf-rdma -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON \
  -DNANO_NCCL_NRANKS=8 -DNANO_NCCL_CUDA_ARCH=86 \
  -DNANO_NCCL_ENABLE_BENCH_PROFILING=OFF
cmake --build build-perf-rdma -j<jobs>
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_LIBRARY_PATH=<path-to-nccl-lib>:$LD_LIBRARY_PATH \
NANO_NCCL_SOCKET_IFNAME=<interface> NANO_NCCL_RDMA_IFNAME=<rdma-interface> \
mpirun --host <host-a>:1 -np 1 -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH \
  -x NANO_NCCL_SOCKET_IFNAME -x NANO_NCCL_RDMA_IFNAME \
  build-perf-rdma/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --transport rdma --dtype <float|fp16|bf16> \
  --redop <sum|avg|max|min> -b 262144 -e 67108864 -f 4 -w 5 -n 20 : \
  --host <host-b>:1 -np 1 -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH \
  -x NANO_NCCL_SOCKET_IFNAME -x NANO_NCCL_RDMA_IFNAME \
  build-perf-rdma/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --transport rdma --dtype <float|fp16|bf16> \
  --redop <sum|avg|max|min> -b 262144 -e 67108864 -f 4 -w 5 -n 20
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_LIBRARY_PATH=<path-to-nccl-lib>:$LD_LIBRARY_PATH \
NCCL_SOCKET_IFNAME=<interface> NCCL_IB_HCA=<rdma-hca> \
NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 \
NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=33554432 \
env -u NCCL_IB_DISABLE mpirun --host <host-a>:1 -np 1 -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH \
  -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x NCCL_ALGO -x NCCL_PROTO \
  -x NCCL_MIN_NCHANNELS -x NCCL_MAX_NCHANNELS -x NCCL_BUFFSIZE \
  <path-to-nccl-tests>/build/all_reduce_perf \
  -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 \
  -d <float|half|bfloat16> -o <sum|avg|max|min> : \
  --host <host-b>:1 -np 1 -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH \
  -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x NCCL_ALGO -x NCCL_PROTO \
  -x NCCL_MIN_NCHANNELS -x NCCL_MAX_NCHANNELS -x NCCL_BUFFSIZE \
  <path-to-nccl-tests>/build/all_reduce_perf \
  -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 \
  -d <float|half|bfloat16> -o <sum|avg|max|min>
```
