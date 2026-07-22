# nano-nccl

[中文说明](README.zh.md)

A GPU collective communication library for single-host multi-GPU All Reduce, targeting NCCL `Ring` + `Simple` + 4 channels performance. An optional MPI/socket path supports multi-host correctness runs.

---

## Performance

[Detailed single-host and two-host performance results](performance.md) record the tested topology, environment, all dtype/reduction combinations, and point-by-point NCCL comparisons.

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

For example, for a 4-GPU RTX A6000 (sm_86) system:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=86
```

Build artifacts:

- `build/benchmarks/nano_nccl_all_reduce_bench` — perf + correctness benchmark
- `build/tests/nano_nccl_correctness` — correctness-only test
- `build/tests/nano_nccl_smoke` — smoke test
- `build/tests/nano_nccl_public_api` — public C++ API coverage
- `build/tests/nano_nccl_p2p_step_counters` — P2P step-counter coverage
- `build/tests/nano_nccl_p2p_topology` — P2P topology coverage
- `build/tests/nano_nccl_simple_protocol` — Simple protocol layout coverage
- `build-mpi/tests/nano_nccl_mpi_correctness` — MPI/socket correctness test (MPI build)
- `build-mpi/tests/nano_nccl_mpi_bootstrap` — MPI bootstrap smoke test (MPI build)
- `build-mpi/tests/nano_nccl_socket_protocol` — socket framing and proxy test (MPI build)

When `BUILD_TESTING` is enabled (the default), `ctest --test-dir build
--output-on-failure` also runs the static BF16 capability-validation regression
and benchmark profiling static regressions.

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
| `NANO_NCCL_ENABLE_BENCH_PROFILING` | `OFF` | Compile NVTX/CUDA-profiler instrumentation into the all-reduce benchmark; keep `OFF` for reported bandwidth |

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
  -np 1 --host <host-a> -x NANO_NCCL_SOCKET_IFNAME=<interface> \
    ./build-mpi/tests/nano_nccl_mpi_correctness --dtype float \
  : -np 1 --host <host-b> -x NANO_NCCL_SOCKET_IFNAME=<interface> \
    ./build-mpi/tests/nano_nccl_mpi_correctness --dtype float
```

---

## Usage

```bash
# Benchmark (perf + correctness)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype float --redop sum --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20

# FP16 and BF16 correctness/performance runs (BF16 requires SM80+)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype fp16 --redop max --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --dtype bf16 --redop avg --transport auto \
  -b 262144 -e 67108864 -f 4 -w 5 -n 20

# Correctness-only test
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_correctness

# Smoke test
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/nano_nccl_smoke
```

`--redop` accepts `sum` (the default), `avg`, `max`, and `min`. `avg` is the
element-wise `sum / nranks`. `max` and `min` propagate NaN when either operand
is NaN. The selected reduction operation is compiled into the device kernel;
the rank count remains a runtime kernel parameter.

### Optional NVTX/CUDA profiling

Build a separate profiling binary; do not use this build for performance comparisons:

```bash
cmake -S . -B build-profile -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DNANO_NCCL_NRANKS=4 -DNANO_NCCL_CUDA_ARCH=86 \
  -DNANO_NCCL_ENABLE_BENCH_PROFILING=ON
cmake --build build-profile -j$(nproc)
nsys profile --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --output=bench-nvtx-profile \
  ./build-profile/benchmarks/nano_nccl_all_reduce_bench \
  --algo ring_simple --transport auto --dtype float -b 262144 -e 262144 -f 2 -w 1 -n 2
nsys stats --report nvtx_pushpop_sum bench-nvtx-profile.nsys-rep
```

For every message size, the capture contains an outer `all_reduce size=<bytes>B` range and one `all_reduce size=<bytes>B iteration=<iteration>` range per measured iteration. Warmup is outside capture. CUDA 12.8 emits an NVTX 2 deprecation notice for `<nvToolsExt.h>`; it does not invalidate the capture.

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
and `sum`, `avg`, `max`, and `min`. `avg` is `sum / nranks`; `max` and `min`
propagate NaN. `reduce_scatter` and `all_gather` are present in the public interface
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
- `sum`, `avg`, `max`, and `min` reduce ops; `avg` is `sum / nranks`, and `max`/`min` propagate NaN
- out-of-place
- SHM FIFO and device P2P FIFO transports, plus optional MPI/socket for cross-process ring edges; P2P is single-node only

No multi-host performance acceptance gate has been established. This project is not a general NCCL replacement.

Future expansion plans:

- dtype: `double` / `int8`
- reduce op: `prod`
- rank count: 2 / 8 / 16 (runtime kernel parameter)
- collective: `all_gather` / `reduce_scatter` / `broadcast`
- transport: network

---

## License

[MIT](LICENSE)
