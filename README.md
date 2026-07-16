# nano-nccl

[中文说明](README.zh.md)

An All Reduce library targeting NCCL `Ring` + `Simple` + 4 channels performance on its validated single-host path. An optional MPI/socket path supports multi-host correctness runs.

---

## Performance

The tables below report out-of-place `busbw` in GB/s from the current measurements (`-w 5 -n 20`).

### Single host: 4× RTX A6000

CUDA 12.8, `CUDA_VISIBLE_DEVICES=0,1,2,3`, and `--transport auto` (resolved to `mixed`). NCCL used `Ring` + `Simple`, four channels, and `NCCL_BUFFSIZE=33554432`. Each size cell is `nano-nccl / NCCL` bus bandwidth in GB/s.

| dtype | 256 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | nano/NCCL geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float | 6.88 / 6.43 | 14.80 / 14.14 | 19.30 / 19.50 | 22.47 / 22.61 | 23.29 / 22.95 | 1.023 |
| fp16 | 6.84 / 6.33 | 14.43 / 14.29 | 19.32 / 19.70 | 22.44 / 22.73 | 23.29 / 23.17 | 1.012 |
| bf16 | 0.11 / 6.35 | 0.45 / 14.42 | 1.56 / 19.94 | 5.82 / 22.77 | 16.42 / 23.17 | 0.095 |

### Two hosts: 2×4 RTX A6000 over TCP socket

One four-GPU MPI process per host, `eno2`, CUDA 12.8, and Open MPI 4.1.2. nano-nccl used `--algo ring_simple --transport auto` and resolved to `mixed` (socket for cross-host edges). NCCL 2.25.1 used `Ring` + `Simple`, four channels, `NCCL_SOCKET_IFNAME=eno2`, and `NCCL_IB_DISABLE=1`.

| size | nano-nccl time (us) | nano busbw (GB/s) | NCCL time (us) | NCCL busbw (GB/s) | nano/NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4203.72 | 0.109130 | 3989.82 | 0.114981 | 0.949116 |
| 1 MiB | 16030.16 | 0.114472 | 16012.80 | 0.114596 | 0.998917 |
| 4 MiB | 66742.16 | 0.109976 | 63165.10 | 0.116204 | 0.946405 |
| 16 MiB | 275982.46 | 0.106384 | 252055.00 | 0.116483 | 0.913301 |
| 64 MiB | 1107119.12 | 0.106078 | 1019285.00 | 0.115219 | 0.920664 |

---

## Build

Dependencies: CUDA 12+, CMake 3.18+, libnuma-dev. The optional MPI/socket build requires Open MPI 4.1.2 on every host; every launched host must use the same Open MPI ABI.

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_NRANKS=<your_gpu_count> \
  -DNANO_NCCL_CUDA_ARCH=<your_cuda_arch>
make -j$(nproc)
```

For example, on the measured 4-GPU RTX A6000 (sm_86) system:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=86
```

Build artifacts:

- `build/benchmarks/nano_nccl_all_reduce_bench` — perf + correctness benchmark
- `build/tests/nano_nccl_correctness` — correctness-only test
- `build/tests/nano_nccl_smoke` — smoke test

### CMake options

| Option | Default | Description |
|---|---|---|
| `NANO_NCCL_NRANKS` | 4 | Number of GPU ranks |
| `NANO_NCCL_NCHANNELS` | 4 | Number of channels |
| `NANO_NCCL_CUDA_ARCH` | 61 | CUDA compute capability (e.g. 61 for sm_61, 75 for Turing, 86 for Ampere) |
| `NANO_NCCL_BLOCK_THREADS` | 512 | Threads per block |
| `NANO_NCCL_FIFO_BUFF_BYTES` | 33554432 | FIFO buffer size in bytes (32 MiB) |
| `NANO_NCCL_ENABLE_MPI` | `OFF` | Build the MPI communicator bootstrap and distributed benchmark/test |
| `NANO_NCCL_SOCKET_TEST_FAULT_INJECTION` | `OFF` | Build a separate test-only fault-injection library for `nano_nccl_mpi_correctness`; ordinary MPI benchmarks never include the hook |

NUMA topology is detected at runtime by reading `/sys/bus/pci/devices/*/numa_node` — no source code changes needed when moving to a different machine.

### Optional MPI/socket build

Build the same commit on each host with the global GPU count. The socket listener is IPv4-only; set `NANO_NCCL_SOCKET_IFNAME` to an interface that resolves to exactly one usable IPv4 address. The socket connection has no TLS, authentication, or automatic reconnect, so use it only on a trusted private network.

```bash
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release \
  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_NRANKS=8 -DNANO_NCCL_CUDA_ARCH=86
cmake --build build-mpi -j$(nproc)
```

For a two-host, four-GPU-per-host launch, pass the interface export in both MPMD app contexts:

```bash
mpirun \
  -np 1 --host 192.168.104.246 -x NANO_NCCL_SOCKET_IFNAME=eno2 \
    ./build-mpi/tests/nano_nccl_mpi_correctness --dtype float \
  : -np 1 --host 192.168.104.247 -x NANO_NCCL_SOCKET_IFNAME=eno2 \
    ./build-mpi/tests/nano_nccl_mpi_correctness --dtype float
```

---

## Usage

```bash
# Benchmark (perf + correctness)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype float --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20

# FP16 and BF16 correctness/performance runs (BF16 requires SM80+)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype fp16 --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype bf16 --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20

# Correctness-only test
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_correctness

# Smoke test
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_smoke
```

## Public C++ API

`nano_nccl/communicator.h` exposes a move-only `Communicator` for one process
that owns all configured local GPUs. The caller owns the device buffers and
CUDA streams. Buffer and stream arrays must have one entry per device, in the
same order as `CommunicatorConfig::devices`.

```cpp
#include "nano_nccl/communicator.h"

#include <memory>
#include <vector>

std::vector<int> devices{0, 1, 2, 3};
nano_nccl::CommunicatorConfig config{devices};
std::unique_ptr<nano_nccl::Communicator> communicator =
    nano_nccl::create_communicator(config);

std::vector<const void*> send_buffers(devices.size());
std::vector<void*> recv_buffers(devices.size());
std::vector<cudaStream_t> streams(devices.size());

// Allocate one out-of-place send/receive pair and one stream on each device.
// send_buffers[i], recv_buffers[i], and streams[i] must belong to devices[i].
// ... cudaSetDevice(devices[i]), cudaMalloc, cudaStreamCreateWithFlags ...

constexpr std::size_t count = 1 << 20;  // Elements per local rank.
nano_nccl::CollectiveArgs args{
    send_buffers,
    recv_buffers,
    streams,
    count,
    nano_nccl::DType::Float,
    nano_nccl::RedOp::Sum,
};

communicator->all_reduce(args);  // Enqueues work; it does not synchronize.

for (std::size_t i = 0; i < devices.size(); ++i) {
    cudaSetDevice(devices[i]);
    cudaStreamSynchronize(streams[i]);
}
communicator->check_async_error();
```

The single-host adapter requires `devices` to be the visible-device sequence
`{0, ..., NANO_NCCL_NRANKS - 1}`. In an MPI build, `nano_nccl/mpi.h` provides
`create_communicator_from_mpi(MPI_COMM_WORLD, config)` for a distributed
communicator. `all_reduce` is out-of-place and supports `float`, FP16, BF16,
and `sum`. `reduce_scatter` and `all_gather` are present in the public interface
but throw an unsupported-operation error.

### Transport selection

`--transport` accepts `auto`, `shm`, and `p2p` on a single host. Distributed
MPI communicators accept `auto`; cross-process ring edges resolve to socket.

- `auto` (the default) selects P2P independently for each ring edge only when
  it has a direct NVLink and CUDA peer access in both directions; other edges
  use SHM. The resulting transport is `shm`, `p2p`, or `mixed`.
- `shm` forces the mapped-host-memory SHM FIFO path.
- `p2p` requires that every ring edge has the required bidirectional peer
  access and fails during setup on the first unavailable direction.

P2P is a single-node transport. It requires CUDA peer access for the complete
configured ring; it is not a multi-node or network transport. Socket uses a
trusted, IPv4-only TCP network boundary and has no TLS or auto reconnect.

---

## Limitations

Currently supports only:

- Single-node multi-GPU performance path (tested with `CUDA_VISIBLE_DEVICES=0,1,2,3`); optional MPI/socket multi-host `all_reduce` correctness path
- `float`, FP16 (`fp16`), and BF16 (`bf16`) dtypes; BF16 requires SM80+
- `sum` reduce op
- out-of-place
- SHM FIFO and device P2P FIFO transports, plus optional MPI/socket for cross-process ring edges; P2P is single-node only

No multi-host performance comparison or performance acceptance gate has been established. This project is not a general NCCL replacement.

Future expansion plans:

- dtype: `double` / `int8`
- reduce op: `max` / `min` / `prod`
- rank count: 2 / 8 / 16 (templated, host-side dispatch)
- collective: `all_gather` / `reduce_scatter` / `broadcast`
- transport: network

---

## License

[MIT](LICENSE)
