#include "nano_nccl/all_reduce.h"
#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
#include "collective/all_reduce/bench_profiling.h"
#endif
#if defined(NANO_NCCL_ENABLE_MPI)
#include "nano_nccl/mpi.h"
#include "nano_nccl/traits.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#if defined(NANO_NCCL_ENABLE_MPI)
#include <mpi.h>
#endif

#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
namespace bench_profiling = nano_nccl::collective::all_reduce::bench_profiling;
#endif

namespace {

bool read_size_arg(int argc, char** argv, int* index, std::size_t* out) {
    if (*index + 1 >= argc) {
        return false;
    }
    *out = static_cast<std::size_t>(std::strtoull(argv[++(*index)], nullptr, 10));
    return true;
}

bool read_int_arg(int argc, char** argv, int* index, int* out) {
    if (*index + 1 >= argc) {
        return false;
    }
    *out = std::atoi(argv[++(*index)]);
    return true;
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--algo auto|ring_simple] "
                 "[--dtype float|fp16|bf16] "
                 "[--transport auto|shm|p2p] "
                 "[-b bytes] [-e bytes] [-f factor] [-w warmup] [-n iters]\n",
                 argv0);
}

#if defined(NANO_NCCL_ENABLE_MPI)

bool mpi_ok(int status, const char* operation) {
    if (status == MPI_SUCCESS) return true;
    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    std::fprintf(stderr, "%s failed: %.*s\n", operation, length, error);
    return false;
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

std::vector<int> visible_devices(int count) {
    std::vector<int> devices(count);
    for (int device = 0; device < count; ++device) devices[device] = device;
    return devices;
}

template <nano_nccl::DType kDType>
int run_mpi_bench_typed(const nano_nccl::BenchConfig& config,
                        std::vector<nano_nccl::BenchResult>* results,
                        const std::vector<int>& devices,
                        int local_rank_offset) {
    using Traits = nano_nccl::DTypeTraits<kDType>;
    using T = typename Traits::type;
    try {
        if (config.algo != "auto" && config.algo != "ring_simple") {
            throw std::runtime_error("unsupported algo: " + config.algo);
        }
        const std::vector<std::size_t> sizes = nano_nccl::make_sizes(
            config.min_bytes, config.max_bytes, config.factor);
        if (sizes.empty()) throw std::runtime_error("invalid size range");
        for (std::size_t bytes : sizes) {
            if (bytes % sizeof(T) != 0) {
                throw std::runtime_error("size must be divisible by dtype size");
            }
        }

        nano_nccl::CommunicatorConfig communicator_config;
        communicator_config.devices = devices;
        communicator_config.transport = config.transport;
        std::unique_ptr<nano_nccl::Communicator> communicator =
            nano_nccl::create_communicator_from_mpi(MPI_COMM_WORLD,
                                                    communicator_config);
        const nano_nccl::TransportKind transport = communicator->transport();
        const std::size_t max_count = sizes.back() / sizeof(T);
        std::vector<void*> send_buffers(devices.size());
        std::vector<void*> recv_buffers(devices.size());
        std::vector<cudaStream_t> streams(devices.size());
        for (std::size_t local_rank = 0; local_rank < devices.size(); ++local_rank) {
            cuda_check(cudaSetDevice(devices[local_rank]), "cudaSetDevice");
            cuda_check(cudaMalloc(&send_buffers[local_rank], max_count * sizeof(T)),
                       "cudaMalloc(send)");
            cuda_check(cudaMalloc(&recv_buffers[local_rank], max_count * sizeof(T)),
                       "cudaMalloc(recv)");
            cuda_check(cudaStreamCreateWithFlags(&streams[local_rank], cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");
        }

        for (std::size_t bytes : sizes) {
            const std::size_t count = bytes / sizeof(T);
            std::vector<std::vector<T>> host_inputs(devices.size(),
                                                     std::vector<T>(count));
            std::vector<T> host_output(count);
            float expected = 0.0f;
            for (int global_rank = 0; global_rank < nano_nccl::kRanks; ++global_rank) {
                expected += Traits::to_float(Traits::from_float(
                    static_cast<float>(global_rank + 1)));
            }
            for (std::size_t local_rank = 0; local_rank < devices.size(); ++local_rank) {
                const float value = static_cast<float>(local_rank_offset +
                    static_cast<int>(local_rank) + 1);
                std::fill(host_inputs[local_rank].begin(), host_inputs[local_rank].end(),
                          Traits::from_float(value));
                cuda_check(cudaSetDevice(devices[local_rank]), "cudaSetDevice");
                cuda_check(cudaMemcpyAsync(send_buffers[local_rank],
                                           host_inputs[local_rank].data(), bytes,
                                           cudaMemcpyHostToDevice, streams[local_rank]),
                           "cudaMemcpyAsync");
            }

            auto launch_and_wait = [&] {
                std::vector<const void*> send_const(send_buffers.begin(), send_buffers.end());
                communicator->all_reduce({send_const, recv_buffers, streams, count, kDType,
                                          nano_nccl::RedOp::Sum});
                for (std::size_t local_rank = 0; local_rank < devices.size(); ++local_rank) {
                    cuda_check(cudaSetDevice(devices[local_rank]), "cudaSetDevice");
                    cuda_check(cudaStreamSynchronize(streams[local_rank]),
                               "cudaStreamSynchronize");
                }
                communicator->check_async_error();
            };
            for (int iteration = 0; iteration < config.warmup_iters; ++iteration) {
                launch_and_wait();
            }
            mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(benchmark start)");
            double local_time_us = 0.0;
#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
            bench_profiling::ProfilerSession profiler;
            {
                bench_profiling::NvtxRange size_range(
                    bench_profiling::all_reduce_size_range_name(bytes));
#endif
            const auto start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < config.iters; ++iteration) {
#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
                bench_profiling::NvtxRange iteration_range(
                    bench_profiling::all_reduce_iteration_range_name(bytes, iteration));
#endif
                launch_and_wait();
            }
            const auto end = std::chrono::steady_clock::now();
            local_time_us = std::chrono::duration<double, std::micro>(end - start).count() /
                            static_cast<double>(config.iters);
#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
            }
            profiler.stop();
#endif
            double max_time_us = 0.0;
            mpi_ok(MPI_Allreduce(&local_time_us, &max_time_us, 1, MPI_DOUBLE, MPI_MAX,
                                 MPI_COMM_WORLD), "MPI_Allreduce(max elapsed)");

            int local_wrong = 0;
            float local_max_abs_error = 0.0f;
            for (std::size_t local_rank = 0; local_rank < devices.size(); ++local_rank) {
                cuda_check(cudaSetDevice(devices[local_rank]), "cudaSetDevice");
                cuda_check(cudaMemcpy(host_output.data(), recv_buffers[local_rank], bytes,
                                       cudaMemcpyDeviceToHost), "cudaMemcpy");
                for (T value : host_output) {
                    const float error = std::fabs(Traits::to_float(value) - expected);
                    local_max_abs_error = std::max(local_max_abs_error, error);
                    if (error > Traits::kDefaultEpsilon) {
                        ++local_wrong;
                        break;
                    }
                }
            }
            int global_wrong = 0;
            float global_max_abs_error = 0.0f;
            mpi_ok(MPI_Allreduce(&local_wrong, &global_wrong, 1, MPI_INT, MPI_SUM,
                                 MPI_COMM_WORLD), "MPI_Allreduce(wrong)");
            mpi_ok(MPI_Allreduce(&local_max_abs_error, &global_max_abs_error, 1, MPI_FLOAT,
                                 MPI_MAX, MPI_COMM_WORLD), "MPI_Allreduce(max abs)");

            int mpi_rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            if (mpi_rank == 0) {
                nano_nccl::BenchResult result;
                result.algo = config.algo == "auto" ? "ring_simple" : config.algo;
                result.dtype = kDType;
                result.transport = transport;
                result.bytes = bytes;
                result.count = count;
                result.time_us = max_time_us;
                result.algbw = nano_nccl::algbw_gbs(bytes, max_time_us);
                result.busbw = nano_nccl::all_reduce_busbw_gbs(result.algbw,
                                                               nano_nccl::kRanks);
                result.wrong = global_wrong;
                result.max_abs_error = global_max_abs_error;
                results->push_back(std::move(result));
            }
        }

        for (std::size_t local_rank = 0; local_rank < devices.size(); ++local_rank) {
            cudaSetDevice(devices[local_rank]);
            cudaFree(send_buffers[local_rank]);
            cudaFree(recv_buffers[local_rank]);
            cudaStreamDestroy(streams[local_rank]);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MPI benchmark failed: %s\n", error.what());
        return 1;
    }
}

int run_mpi_bench(const nano_nccl::BenchConfig& config,
                  std::vector<nano_nccl::BenchResult>* results) {
    int mpi_rank = 0;
    if (!mpi_ok(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank), "MPI_Comm_rank")) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    int local_device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&local_device_count);
    int local_devices_ready =
        device_status == cudaSuccess && local_device_count > 0 ? 1 : 0;
    if (local_devices_ready == 0) {
        if (device_status != cudaSuccess) {
            std::fprintf(stderr, "MPI benchmark rank %d: cudaGetDeviceCount failed: %s\n",
                         mpi_rank, cudaGetErrorString(device_status));
        } else {
            std::fprintf(stderr, "MPI benchmark rank %d: no visible CUDA devices\n",
                         mpi_rank);
        }
    }

    int all_devices_ready = 0;
    if (!mpi_ok(MPI_Allreduce(&local_devices_ready, &all_devices_ready, 1, MPI_INT,
                              MPI_MIN, MPI_COMM_WORLD),
                "MPI_Allreduce(cudaGetDeviceCount)")) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }
    if (all_devices_ready == 0) {
        if (mpi_rank == 0) {
            std::fprintf(stderr,
                         "MPI benchmark aborted because at least one rank cannot enumerate CUDA devices\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    const std::vector<int> devices = visible_devices(local_device_count);
    int local_rank_offset = 0;
    if (!mpi_ok(MPI_Exscan(&local_device_count, &local_rank_offset, 1, MPI_INT, MPI_SUM,
                           MPI_COMM_WORLD), "MPI_Exscan(local rank offset)")) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }
    if (mpi_rank == 0) local_rank_offset = 0;
    switch (config.dtype) {
        case nano_nccl::DType::Float:
            return run_mpi_bench_typed<nano_nccl::DType::Float>(
                config, results, devices, local_rank_offset);
        case nano_nccl::DType::Float16:
            return run_mpi_bench_typed<nano_nccl::DType::Float16>(
                config, results, devices, local_rank_offset);
        case nano_nccl::DType::BFloat16:
            return run_mpi_bench_typed<nano_nccl::DType::BFloat16>(
                config, results, devices, local_rank_offset);
    }
    return 2;
}

#endif

}  // namespace

int main(int argc, char** argv) {
    nano_nccl::BenchConfig config;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--algo") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 2;
            }
            config.algo = argv[++i];
        } else if (std::strcmp(argv[i], "--dtype") == 0) {
            if (i + 1 >= argc ||
                !nano_nccl::parse_dtype(argv[++i], &config.dtype)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--transport") == 0) {
            if (i + 1 >= argc ||
                !nano_nccl::parse_transport(argv[++i], &config.transport)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "-b") == 0) {
            if (!read_size_arg(argc, argv, &i, &config.min_bytes)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "-e") == 0) {
            if (!read_size_arg(argc, argv, &i, &config.max_bytes)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "-f") == 0) {
            if (!read_int_arg(argc, argv, &i, &config.factor)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "-w") == 0) {
            if (!read_int_arg(argc, argv, &i, &config.warmup_iters)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "-n") == 0) {
            if (!read_int_arg(argc, argv, &i, &config.iters)) {
                usage(argv[0]);
                return 2;
            }
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    std::vector<nano_nccl::BenchResult> results;
    int rc = 0;
    int mpi_rank = 0;
#if defined(NANO_NCCL_ENABLE_MPI)
    if (!mpi_ok(MPI_Init(&argc, &argv), "MPI_Init")) return 1;
    int mpi_size = 1;
    mpi_ok(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank), "MPI_Comm_rank");
    mpi_ok(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size), "MPI_Comm_size");
    rc = mpi_size > 1 ? run_mpi_bench(config, &results)
                      : nano_nccl::run_all_reduce_bench(config, &results);
#else
    rc = nano_nccl::run_all_reduce_bench(config, &results);
#endif

    if (mpi_rank == 0) {
        std::printf("# nano-nccl all_reduce_bench\n");
        std::printf("# algo %s dtype %s transport %s nGpus %d warmup iters: %d iters: %d validation: 1\n",
                    config.algo.c_str(), nano_nccl::dtype_name(config.dtype),
                    nano_nccl::transport_name(config.transport),
                    nano_nccl::kRanks, config.warmup_iters, config.iters);
        std::printf("# %14s %8s %10s %12s %12s %10s %10s %10s %8s %12s\n", "algo",
                    "dtype", "transport", "size(B)", "count", "time(us)", "algbw",
                    "busbw", "#wrong", "max_abs");
        for (const auto& result : results) {
            std::printf("%14s %8s %10s %12zu %12zu %10.2f %10.2f %10.2f %8d %12.6g\n",
                        result.algo.c_str(), nano_nccl::dtype_name(result.dtype),
                        nano_nccl::transport_name(result.transport), result.bytes,
                        result.count, result.time_us, result.algbw, result.busbw,
                        result.wrong, result.max_abs_error);
        }
    }

    int exit_code = rc;
    if (rc == 0) {
        for (const auto& result : results) {
            if (result.wrong != 0) exit_code = 1;
        }
    }
#if defined(NANO_NCCL_ENABLE_MPI)
    int global_exit_code = 0;
    mpi_ok(MPI_Allreduce(&exit_code, &global_exit_code, 1, MPI_INT, MPI_MAX,
                         MPI_COMM_WORLD), "MPI_Allreduce(exit code)");
    MPI_Finalize();
    return global_exit_code;
#else
    return exit_code;
#endif
}
