# Performance

## 2026-10-15: Two-host four-rank GDR

The latest campaign compares the nano implementation now integrated into main
with **NCCL 2.30.7** on two hosts, using physical GPU0 and GPU1 on each host (four MPI ranks,
one process per GPU). It measures out-of-place `float`/`sum` AllReduce with
Ring/Simple, local NVLink P2P, and bidirectional cross-host GDR over a 100 Gb/s
link with MTU 1500. Existing authorized vLLM workers remained running.

### Method and configuration

- Channels: 1, 2, 3, and 4; seven message sizes, including two scalar tails.
  Each cell contains four pairs in AB/BA/AB/BA order, for 224 formal jobs.
  All jobs completed with exit status 0 and zero wrong values; slow samples
  were retained.
- Both implementations: CUDA 12.8.61, SM86 hardware, Open MPI 4.1.2,
  32 MiB FIFO/buffer per channel, `-w 5 -n 20`.
  Nano uses `--wait-mode query` and maximum-rank aggregation; nccl-tests uses
  `-g 1 -z 2 -a 3 -m 1`. NCCL was explicitly Ring/Simple with matching channel
  counts and `NCCL_NTHREADS=256`; metadata confirmed nine warps.
- Nano: U8 vector-loop unrolling, 16-byte-aligned channel partitions,
  vector body/scalar tail, 288 logical threads/256 workers, original Simple
  synchronization, shared progress, WRITE+CTS, and selective send CQEs.
  Profiling and hot-path investigation instrumentation were off during timing;
  NCCL CPU profiler metadata was collected separately before timing.
- The measurements were collected before integration from local commit
  `cf473a44b9bdc89a1dacb88ccfef612a3e4deba9` plus frozen development changes
  (kernel SHA256 `0c31cff07ee5fb71f765307777bd84d78de5812351d4e8dbbae762e2c0d17518`).
  Main now includes those channel alignment, vector-body/scalar-tail, float4
  batching, and shared-progress mechanisms, with regular build options and
  the original Simple synchronization. Experimental NCCL-style synchronization,
  forced SHM edges, all-data signaled CQEs, and hot-path diagnostics are not
  enabled by this integration. The timings below remain the original measured
  development-run data, not a newly timed main run.

### Implementation and reproduction

Main defaults to 288 block threads, U8 float4 batching, 16-byte-aligned
AllReduce channel boundaries, and shared RDMA progress. The last channel can
have a scalar tail; unaligned pointers keep the scalar path. The partition
change applies to AllReduce, while vector worker changes are shared by the
three collectives. `NANO_NCCL_FLOAT4_UNROLL=1` retains the unbatched comparison.
GDR remains an explicit request and fails rather than silently falling back;
WRITE+CTS also remains opt-in. Empty slices still skip unnecessary data WRs.

Build a separate directory on both hosts for each channel count (1–4):

```bash
cmake -S . -B build-gdr-c<channels> -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_PREFIX_PATH=<openmpi-4.1.2-prefix> \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON \
  -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_NCHANNELS=<channels> \
  -DNANO_NCCL_CUDA_ARCH=86 -DNANO_NCCL_BLOCK_THREADS=288 \
  -DNANO_NCCL_FLOAT4_UNROLL=8 -DNANO_NCCL_ENABLE_BENCH_PROFILING=OFF
cmake --build build-gdr-c<channels> -j<jobs>
```

For nano, export `NANO_NCCL_RDMA_GDR=1`, `NANO_NCCL_RDMA_USE_WRITE=1`, and
`NANO_NCCL_RDMA_SHARED_PROGRESS=1`, together with the correct bootstrap/RDMA
interface and GID. Launch two MPI ranks per host and expose physical GPU0/1,
using a local-rank device selection. For each message size, run
`nano_nccl_all_reduce_bench --algo ring_simple --transport rdma --dtype float
--redop sum --wait-mode query -b <bytes> -e <bytes> -w 5 -n 20`.

For NCCL 2.30.7, match Ring/Simple, channel count, `NCCL_NTHREADS=256`,
`NCCL_BUFFSIZE=33554432`, `NCCL_NET_GDR_LEVEL=SYS`, and `NCCL_NET_GDR_READ=1`;
use nccl-tests `-g 1 -z 2 -a 3 -m 1` with the same sizes/warmups/iterations
and rank/device mapping. Verify actual per-edge local P2P and cross-host GDR
in both implementations' logs. A working GDR registration provider and
compatible end-to-end RoCE configuration are prerequisites.

### Latency matrix

Each entry is **nano / NCCL time in microseconds**. Each arm's value is the
median of four independent job averages, with 20 measured collectives per job;
it is not a median of individual collective latencies.

| Message | 1 channel | 2 channels | 3 channels | 4 channels |
|---|---:|---:|---:|---:|
| 1 MiB | 194.44 / 229.12 | 163.47 / 199.26 | 156.94 / 187.21 | 154.98 / 195.28 |
| 1 MiB + 4 B | 201.08 / 227.79 | 164.37 / 195.81 | 157.03 / 187.29 | 157.22 / 188.59 |
| 9 MiB | 1416.87 / 1532.55 | 1262.10 / 1442.60 | 1246.85 / 1463.53 | 1241.38 / 1429.62 |
| 12 MiB | 1881.48 / 2067.82 | 1679.88 / 1836.99 | 1654.19 / 1798.06 | 1664.79 / 1787.08 |
| 16 MiB | 2513.69 / 2676.43 | 2227.34 / 2408.14 | 2215.30 / 2305.43 | 2204.70 / 2277.51 |
| 16 MiB + 4 B | 2661.34 / 2619.15 | 2243.06 / 2396.92 | 2211.47 / 2350.96 | 2214.41 / 2274.05 |
| 32 MiB | 5040.16 / 5184.35 | 4476.26 / 4577.50 | 4497.89 / 4604.35 | 4468.45 / 4579.76 |

Nano has lower median time in **27/28 cells** and wins all four pairs in
**25/28 cells**. The equally weighted geometric mean of the 28 ratios
`NCCL median time / nano median time` is **1.1024x**. This is a summary of this
matrix, not a training speedup or proof that one implementation mechanism
caused the advantage. At 32 MiB the median-time advantage is about 2–3%.

### Exceptions, variation, and evidence

- The only median regression is 1 channel / 16 MiB + 4 B: nano is 1.61% slower
  (2661.34 vs 2619.15 us). Source inspection confirms that nano executes a
  second outer loop with one valid float at this boundary, while NCCL remains
  in one outer loop. Its contribution to the timing gap remains `Unknown`;
  this is an open performance gap, `GAP-RANK4-CHUNK-CAP-001`.
- Both implementations have slow job samples. For 4 channels / 12 MiB,
  nano spans 1654.29–2311.95 us; its median is faster, but its arithmetic mean
  is 1823.95 vs NCCL's 1779.62 us. NCCL's 2-channel / 9 MiB samples span
  1362.46–2045.11 us. Using the arithmetic mean of all four job averages,
  nano is faster in 26/28 cells. No samples were removed.
- All four authorized background workers remained present. The 2116 selected-GPU
  health samples showed recovery action `None` and sampled GPU utilization 0%.
  Sampling cannot exclude intervening activity. Unselected GPUs also carried
  compute load (sampled utilization up to 84%), and CPU, PCIe, memory, and
  network resources were shared.
- These four job averages do not establish per-collective p99 or show that
  tails are eliminated. This campaign does not isolate U8, partition alignment,
  vectorization, synchronization, or protocol contributions, and it does not
  complete the full dtype/op/collective contract or the 3% nano-baseline gate.
  The existing NCCL reference build lacks complete compiler logs; its listed
  cubins include SM80 but not SM86, so byte-for-byte build reproducibility is
  not claimed.

### Evidence and source references

The full original measurements (224 rows), 112 pairs, 28-cell statistics,
verification, build manifests, and logs remain in the independent experiments
repository under `nano-nccl/two-host-rdma-gdr/runs/run-20261015-001/`.
The NCCL source is v2.30.7-1 commit
`73cf112295c33aee2b895f329f592f2a9b4b0f97`; the reference library SHA256 is
`45d7b107276c3b865e06c25b24b5988850dda99a1db71af2eae7bc077b933c35` and
nccl-tests commit is `a0b82b2260cf5152b9f8c061bbf7eaf0ba096432`.

The boundary difference above follows from nano
`src/collective/all_reduce/ring_simple_geometry.h::cbd_part`,
`communicator.cu::ensure_fifo_buffers`, and
`src/kernels/ring_simple_kernel.cuh`: 32 MiB / 8 FIFO steps gives a 4 MiB slot,
so the four-rank, one-channel 16 MiB + 4 B input requires a second loop with
one valid float. NCCL `src/include/collectives.h`,
`src/enqueue.cc::calcCollChunking`, and
`src/device/all_reduce.h::runRing` use a nominal 16 MiB/rank chunk limit,
then shrink the actual final chunk for this input; the nominal outer stride
is 64 MiB and this input stays within one loop. Neither nominal capacity
nor loop count alone proves actual WR traffic or timing causality.

### Integration validation

The integrated implementation was freshly built and checked on both hosts
with `/usr/local/cuda-12.8/bin/nvcc` (12.8.61), keeping the authorized vLLM
workers running:

- Eight Release MPI/RDMA builds passed: channels 1–4 on both hosts, using
  default U8/288 threads and the vector-aligned AllReduce partition.
- 63 CTest executions across host/channel configurations passed, including
  vector tails, Simple publication, RDMA protocol/proxy progress, GDR
  registration/flush checks, C API/static contracts, and the minimum SM70
  default-build contract. These are executions, not 63 distinct test cases.
  The profiling static check was updated to match the already implemented
  barrier-followed-by-final-warmup timing contract.
- A separate U1 build passed float/FP16/BF16 vector-worker tests, including
  aligned bodies, scalar tails, misalignment, aliases, NaNs, and guard values.
- Six dual-host, four-rank native GDR jobs passed: WRITE+CTS on channels 1–4,
  plus SEND/RECV and dedicated-progress compatibility on channel 3. Each
  checked all three collectives, float/FP16/BF16, two rounds, and 5,242,883
  elements; reductions used `sum`. This is finite execution coverage, not
  the full dtype/op/rank contract.
- Twelve bounded AllReduce smoke jobs produced 32 size/channel results with
  zero wrong values: 1/2/4/8/16/32 MiB and 1 MiB + 4 B / 16 MiB + 4 B on
  each channel count. Both cross-host edges reported GPU FIFO placement,
  and both local edges reported P2P. These used `-w 2 -n 3` for correctness;
  their times do not replace the paired performance matrix above.
- Both hosts matched the 98-file implementation/test manifest. All 236
  selected-GPU health readings in 118 host snapshots reported recovery action
  `None`; the original four authorized workers remained present. Discrete
  checks do not replace an unrestricted kernel-log audit.

Full build/test logs and verification are retained locally in the integration
worktree's ignored `.local/integration-20261015/` directory. The matrix above
remains the original 224-job measurement. Full-scope performance acceptance,
the packed-type regression questions, and the accepted nano baseline gate
remain open.

## Historical measurements: timing caveat

The remaining tables are historical out-of-place AllReduce measurements.
Bandwidth is `busbw` in GB/s; the `nano/NCCL` column is calculated from the
unrounded time as `nccl_time_us / nano_time_us`. Every measured result in
these tables completed benchmark validation with zero wrong values.

These historical matrices predate the matched benchmark contract. Nano
synchronized after every collective and reported the slowest MPI rank;
nccl-tests used its default asynchronous batch and averaged MPI processes.
Nano used one MPI process per GPU while NCCL used multiple GPUs per MPI
process. Their ratios cannot be used for formal NCCL-relative acceptance.
The reproduction commands below use the corrected protocol and will not
reproduce these historical timing values.

## Historical test topology and environment

Two hosts, each two-socket Intel Xeon Platinum 8462Y+ (32 cores per socket, two threads per core), 4x NVIDIA RTX A6000 (SM86), CUDA 12.8.61, NCCL 2.30.7 built from source, nccl-tests 2.19.6, and Open MPI 4.1.2.

| Node | OS kernel | GPU driver | GPUs |
| --- | --- | --- | --- |
| A | Linux 5.15.0-136-generic | 580.82.07 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |
| B | Linux 5.15.0-136-generic | 580.82.07 | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |

On each host, GPU0-GPU1 and GPU2-GPU3 are connected by four NVLinks. The two pairs are separated by `SYS` paths across NUMA nodes. Tables are separate transport classes: in-process auto (P2P/SHM) on one host, two-host TCP socket, two-host host-pinned RDMA, and two-host RDMA with GPUDirect (device FIFO).

These historical measurements use a Release build with `NANO_NCCL_ENABLE_BENCH_PROFILING=OFF`, message sizes 256 KiB through 64 MiB, `-w 5`, and `-n 20`. NCCL uses `Ring`, `Simple`, four channels, and a 32 MiB buffer. RDMA WRITE+CTS posts from the registered mapped FIFO (no host bounce; publication via publisher `st.release.sys(send_tail)` after block sync and host acquire loads). The GDR table reports the median of five full-matrix repetitions for each implementation.

## Historical single host: 4 Ranks Over Auto (P2P/SHM)

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


## Historical two hosts: 8 Ranks Over TCP Socket

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


## Historical two hosts: 8 Ranks Over RDMA

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


## Historical two hosts: 8 Ranks Over RDMA (GDR)

Same 2x4 topology as the host-pinned RDMA table. Nano `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` and `NANO_NCCL_RDMA_GDR=1` (WRITE+CTS from a registered GPU FIFO; host proxy still posts). After a receive CQE, nano flushes third-party GDR writes before publishing `recv_tail` when the device lacks native GPU/RDMA write ordering. NCCL: Ring+Simple with `NCCL_NET_GDR_LEVEL=SYS`; debug logs confirmed `/GDRDMA` on the cross-host edges. Each cell is the median of five complete, alternating-order repetitions; no isolated retries replace matrix samples. This is host-proxy GDR, not GPU-initiated IBGDA.

This historical 60-cell table used the unmatched timing and rank layout noted above; its ratios are not NCCL-relative acceptance results. Under that historical setup, nano/NCCL busbw geomean was 1.101, the minimum cell was 0.937, and 0 cells were below 0.90. The corrected matched two-host WRITE+CTS matrix is recorded in the experiments repository under `nano-nccl/two-host-rdma-gdr/runs/run-20260927-003/`.

No accepted GDR nano baseline exists, so the formal 3% baseline-regression gate is not adjudicated. Cells below NCCL remain `Unknown` performance gaps until controlled causal experiments explain them. The existing packed FP16/BF16 `max`/`min` single-NaN propagation failure is unchanged in GDR=0 and GDR=1; these tables validate their ordinary benchmark inputs but do not establish complete dtype/redop contract correctness.

### Float

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 171.75 | 2.67 | 214.30 | 2.14 | 1.25 |
| 1 MiB | 218.14 | 8.41 | 281.85 | 6.51 | 1.29 |
| 4 MiB | 694.13 | 10.57 | 651.74 | 11.26 | 0.94 |
| 16 MiB | 2637.25 | 11.13 | 2546.43 | 11.53 | 0.97 |
| 64 MiB | 10299.7 | 11.40 | 10168.1 | 11.55 | 0.99 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 170.57 | 2.69 | 214.56 | 2.14 | 1.26 |
| 1 MiB | 217.74 | 8.43 | 286.64 | 6.40 | 1.32 |
| 4 MiB | 693.52 | 10.58 | 651.26 | 11.27 | 0.94 |
| 16 MiB | 2619.23 | 11.21 | 2546.96 | 11.53 | 0.97 |
| 64 MiB | 10299.8 | 11.40 | 10166.6 | 11.55 | 0.99 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 179.20 | 2.56 | 224.67 | 2.04 | 1.25 |
| 1 MiB | 221.87 | 8.27 | 290.04 | 6.33 | 1.31 |
| 4 MiB | 693.09 | 10.59 | 651.42 | 11.27 | 0.94 |
| 16 MiB | 2617.71 | 11.22 | 2544.91 | 11.54 | 0.97 |
| 64 MiB | 10303.3 | 11.40 | 10166.4 | 11.55 | 0.99 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 171.22 | 2.68 | 230.90 | 1.99 | 1.35 |
| 1 MiB | 219.74 | 8.35 | 282.95 | 6.49 | 1.29 |
| 4 MiB | 694.35 | 10.57 | 652.72 | 11.25 | 0.94 |
| 16 MiB | 2621.07 | 11.20 | 2546.50 | 11.53 | 0.97 |
| 64 MiB | 10337.4 | 11.36 | 10166.0 | 11.55 | 0.98 |

### FP16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 176.94 | 2.59 | 450.07 | 1.02 | 2.54 |
| 1 MiB | 218.62 | 8.39 | 281.63 | 6.52 | 1.29 |
| 4 MiB | 693.51 | 10.58 | 651.28 | 11.27 | 0.94 |
| 16 MiB | 2717.22 | 10.81 | 2546.08 | 11.53 | 0.94 |
| 64 MiB | 10299.8 | 11.40 | 10165.2 | 11.55 | 0.99 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 178.10 | 2.58 | 215.27 | 2.13 | 1.21 |
| 1 MiB | 222.27 | 8.26 | 288.94 | 6.35 | 1.30 |
| 4 MiB | 693.36 | 10.59 | 654.21 | 11.22 | 0.94 |
| 16 MiB | 2662.71 | 11.03 | 2544.99 | 11.54 | 0.96 |
| 64 MiB | 10300.7 | 11.40 | 10164.7 | 11.55 | 0.99 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 177.62 | 2.58 | 214.64 | 2.14 | 1.21 |
| 1 MiB | 218.58 | 8.39 | 289.12 | 6.35 | 1.32 |
| 4 MiB | 692.55 | 10.60 | 653.10 | 11.24 | 0.94 |
| 16 MiB | 2715.09 | 10.81 | 2545.31 | 11.54 | 0.94 |
| 64 MiB | 10299.5 | 11.40 | 10165.0 | 11.55 | 0.99 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 173.55 | 2.64 | 217.83 | 2.11 | 1.26 |
| 1 MiB | 218.90 | 8.38 | 289.83 | 6.33 | 1.32 |
| 4 MiB | 693.37 | 10.59 | 651.98 | 11.26 | 0.94 |
| 16 MiB | 2709.19 | 10.84 | 2547.36 | 11.53 | 0.94 |
| 64 MiB | 10299.6 | 11.40 | 10165.9 | 11.55 | 0.99 |

### BF16

#### Sum

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 177.05 | 2.59 | 219.25 | 2.09 | 1.24 |
| 1 MiB | 220.15 | 8.34 | 281.12 | 6.53 | 1.28 |
| 4 MiB | 690.97 | 10.62 | 651.51 | 11.27 | 0.94 |
| 16 MiB | 2627.65 | 11.17 | 2545.45 | 11.53 | 0.97 |
| 64 MiB | 10300.1 | 11.40 | 10165.5 | 11.55 | 0.99 |

#### Avg

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 170.77 | 2.69 | 459.51 | 1.00 | 2.69 |
| 1 MiB | 218.06 | 8.42 | 279.41 | 6.57 | 1.28 |
| 4 MiB | 689.67 | 10.64 | 650.21 | 11.29 | 0.94 |
| 16 MiB | 2717.25 | 10.81 | 2546.03 | 11.53 | 0.94 |
| 64 MiB | 10300.6 | 11.40 | 10165.8 | 11.55 | 0.99 |

#### Max

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 177.59 | 2.58 | 214.80 | 2.14 | 1.21 |
| 1 MiB | 218.50 | 8.40 | 286.07 | 6.41 | 1.31 |
| 4 MiB | 692.93 | 10.59 | 653.29 | 11.24 | 0.94 |
| 16 MiB | 2690.57 | 10.91 | 2547.34 | 11.53 | 0.95 |
| 64 MiB | 10303.2 | 11.40 | 10168.5 | 11.55 | 0.99 |

#### Min

| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 177.90 | 2.58 | 214.96 | 2.13 | 1.21 |
| 1 MiB | 217.77 | 8.43 | 283.33 | 6.48 | 1.30 |
| 4 MiB | 693.22 | 10.59 | 653.20 | 11.24 | 0.94 |
| 16 MiB | 2709.29 | 10.84 | 2545.74 | 11.53 | 0.94 |
| 64 MiB | 10301.4 | 11.40 | 10165.1 | 11.55 | 0.99 |


## Reproduction of the matched protocol

For a new matched comparison, build nano-nccl with CUDA 12.8, SM86, Release mode, and profiling disabled. Run one MPI process per GPU for both implementations: four local ranks for a single-host comparison or four ranks per host with a global `NANO_NCCL_NRANKS=8` for two hosts. Use the same Open MPI 4.1.2 prefix.

```bash
# nano-nccl, four local MPI ranks over auto (P2P/SHM)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  mpirun --bind-to none -np 4 ./build-perf-single/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --transport auto --dtype <float|fp16|bf16> \
  --redop <sum|avg|max|min> -b 262144 -e 67108864 -f 4 -w 5 -n 20 --wait-mode query

# NCCL, four local MPI ranks, one GPU per process
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_LIBRARY_PATH=<path-to-nccl-lib> \
NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 \
NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=33554432 \
  mpirun --bind-to none -np 4 <path-to-nccl-tests>/build/all_reduce_perf \
  -b 262144 -e 67108864 -f 4 -g 1 -w 5 -n 20 -z 2 -a 3 -m 1 \
  -d <float|half|bfloat16> -o <sum|avg|max|min>
```

For socket runs, launch four MPI processes with four visible GPUs per host. Nano uses `--transport auto` (cross-host edges are socket). NCCL sets `NCCL_IB_DISABLE=1`; local P2P/SHM remain enabled.

For host-pinned RDMA, use nano `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1`. Set `NANO_NCCL_SOCKET_IFNAME=<interface>` for bootstrap and `NANO_NCCL_RDMA_IFNAME=<rdma-interface>` (and `NANO_NCCL_RDMA_GID_INDEX` when required). NCCL sets `NCCL_NET_GDR_LEVEL=0`, `NCCL_IB_HCA=<rdma-hca>`, and `NCCL_IB_GID_INDEX` when required.

For a two-host GDR matrix, build both hosts for eight ranks and launch four MPI processes per host with four visible GPUs. Add `NANO_NCCL_RDMA_GDR=1` for nano and use `NCCL_NET_GDR_LEVEL=SYS` and `NCCL_NET_GDR_READ=1` for NCCL. Run `nano_nccl_rdma_gdr` before the matrix; nano's explicit GDR request fails instead of falling back. The benchmark prints the aggregate transport and each directed edge's backend and RDMA FIFO placement (`gdr`, `host-pinned`, or `n/a`). Verify `/GDRDMA` in both NCCL send and receive logs.

```bash
cmake -S . -B build-perf-rdma-n8 -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON \
  -DNANO_NCCL_NRANKS=8 -DNANO_NCCL_CUDA_ARCH=86 \
  -DNANO_NCCL_ENABLE_BENCH_PROFILING=OFF
cmake --build build-perf-rdma-n8 -j<jobs>
```

The legacy renderer replaces the whole document. Render historical matrices to a separate path and preserve the current implementation and measured campaign section above:

```bash
scripts/run_performance_matrix.sh --nccl-bin <path> --nccl-lib <dir> --out-dir <out>
python3 scripts/render_performance_md.py <out>/matrix.json -o <out>/performance-generated.md
```
