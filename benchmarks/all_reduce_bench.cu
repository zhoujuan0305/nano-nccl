#include "nano_nccl/mpi.h"
#include "nano_nccl/traits.h"
#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
#include "collective/all_reduce/bench_profiling.h"
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
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
namespace bench_profiling = nano_nccl::collective::all_reduce::bench_profiling;
#endif

namespace {

void mpi_check(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    throw std::runtime_error(std::string(operation) + ": " +
                             std::string(error, static_cast<std::size_t>(length)));
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

void usage(const char* program) {
    std::fprintf(stderr,
                 "Usage: %s [--algo auto|ring_simple] "
                 "[--dtype float|fp16|bf16] [--redop sum|avg|max|min] "
                 "[--transport auto|shm|p2p|socket|rdma] "
                 "[--device local-cuda-index] "
                 "[-b bytes] [-e bytes] [-f factor] [-w warmup] [-n iters]\n",
                 program);
}

template <nano_nccl::DType kDType>
int run_typed(const nano_nccl::BenchConfig& config,
              std::vector<nano_nccl::BenchResult>* results, int rank) {
    using Traits = nano_nccl::DTypeTraits<kDType>;
    using T = typename Traits::type;
    const std::vector<std::size_t> sizes = nano_nccl::make_sizes(
        config.min_bytes, config.max_bytes, config.factor);
    if (sizes.empty()) throw std::invalid_argument("invalid size range");
    for (std::size_t bytes : sizes) {
        if (bytes % sizeof(T) != 0) {
            throw std::invalid_argument("size must be divisible by dtype size");
        }
    }

    nano_nccl::CommunicatorConfig communicator_config;
    communicator_config.device = config.device;
    communicator_config.transport = config.transport;
    std::unique_ptr<nano_nccl::Communicator> communicator =
        nano_nccl::create_communicator_from_mpi(MPI_COMM_WORLD,
                                                communicator_config);
    const nano_nccl::TransportKind actual_transport = communicator->transport();

    const std::size_t max_count = sizes.back() / sizeof(T);
    T* send = nullptr;
    T* recv = nullptr;
    cudaStream_t stream = nullptr;
    cuda_check(cudaSetDevice(config.device), "cudaSetDevice");
    cuda_check(cudaMalloc(&send, max_count * sizeof(T)), "cudaMalloc(send)");
    cuda_check(cudaMalloc(&recv, max_count * sizeof(T)), "cudaMalloc(recv)");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");

    for (std::size_t bytes : sizes) {
        const std::size_t count = bytes / sizeof(T);
        const T rank_value = Traits::from_float(static_cast<float>(rank + 1));
        std::vector<T> input(count, rank_value);
        std::vector<T> output(count);
        cuda_check(cudaMemcpyAsync(send, input.data(), bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync(input)");

        float expected = Traits::to_float(Traits::from_float(1.0f));
        for (int peer = 1; peer < nano_nccl::kRanks; ++peer) {
            const float next = Traits::to_float(
                Traits::from_float(static_cast<float>(peer + 1)));
            if (config.redop == nano_nccl::RedOp::Sum ||
                config.redop == nano_nccl::RedOp::Avg) {
                expected += next;
            } else if (config.redop == nano_nccl::RedOp::Max) {
                expected = std::fmax(expected, next);
            } else {
                expected = std::fmin(expected, next);
            }
        }
        if (config.redop == nano_nccl::RedOp::Avg) {
            expected /= static_cast<float>(nano_nccl::kRanks);
        }

        auto launch_and_wait = [&] {
            communicator->all_reduce({send, recv, stream, count, kDType,
                                      config.redop});
            cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
            communicator->check_async_error();
        };
        for (int iteration = 0; iteration < config.warmup_iters; ++iteration) {
            launch_and_wait();
        }
        mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(benchmark start)");

#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
        bench_profiling::ProfilerSession profiler;
        bench_profiling::NvtxRange size_range(
            bench_profiling::all_reduce_size_range_name(bytes));
#endif
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < config.iters; ++iteration) {
#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
            bench_profiling::NvtxRange iteration_range(
                bench_profiling::all_reduce_iteration_range_name(bytes,
                                                                  iteration));
#endif
            launch_and_wait();
        }
        const auto end = std::chrono::steady_clock::now();
#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
        profiler.stop();
#endif
        const double local_time_us =
            std::chrono::duration<double, std::micro>(end - start).count() /
            static_cast<double>(config.iters);
        double max_time_us = 0.0;
        mpi_check(MPI_Allreduce(&local_time_us, &max_time_us, 1, MPI_DOUBLE,
                                MPI_MAX, MPI_COMM_WORLD),
                  "MPI_Allreduce(max elapsed)");

        cuda_check(cudaMemcpy(output.data(), recv, bytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(output)");
        int local_wrong = 0;
        float local_max_abs_error = 0.0f;
        for (T value : output) {
            const float error = std::fabs(Traits::to_float(value) - expected);
            local_max_abs_error = std::max(local_max_abs_error, error);
            if (error > Traits::kDefaultEpsilon) {
                local_wrong = 1;
                break;
            }
        }
        int global_wrong = 0;
        float global_max_abs_error = 0.0f;
        mpi_check(MPI_Allreduce(&local_wrong, &global_wrong, 1, MPI_INT,
                                MPI_SUM, MPI_COMM_WORLD),
                  "MPI_Allreduce(wrong)");
        mpi_check(MPI_Allreduce(&local_max_abs_error, &global_max_abs_error, 1,
                                MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD),
                  "MPI_Allreduce(max abs)");
        if (rank == 0) {
            nano_nccl::BenchResult result;
            result.algo = "ring_simple";
            result.dtype = kDType;
            result.redop = config.redop;
            result.transport = actual_transport;
            result.bytes = bytes;
            result.count = count;
            result.time_us = max_time_us;
            result.algbw = nano_nccl::algbw_gbs(bytes, max_time_us);
            result.busbw = nano_nccl::all_reduce_busbw_gbs(
                result.algbw, nano_nccl::kRanks);
            result.wrong = global_wrong;
            result.max_abs_error = global_max_abs_error;
            results->push_back(result);
        }
    }

    communicator.reset();
    const cudaError_t stream_status = cudaStreamDestroy(stream);
    const cudaError_t recv_status = cudaFree(recv);
    const cudaError_t send_status = cudaFree(send);
    cuda_check(stream_status, "cudaStreamDestroy");
    cuda_check(recv_status, "cudaFree(recv)");
    cuda_check(send_status, "cudaFree(send)");
    return 0;
}

int run_benchmark(const nano_nccl::BenchConfig& config,
                  std::vector<nano_nccl::BenchResult>* results, int rank) {
    switch (config.dtype) {
        case nano_nccl::DType::Float:
            return run_typed<nano_nccl::DType::Float>(config, results, rank);
        case nano_nccl::DType::Float16:
            return run_typed<nano_nccl::DType::Float16>(config, results, rank);
        case nano_nccl::DType::BFloat16:
            return run_typed<nano_nccl::DType::BFloat16>(config, results, rank);
    }
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    nano_nccl::BenchConfig config;
    for (int index = 1; index < argc; ++index) {
        const char* argument = argv[index];
        auto next = [&]() -> const char* {
            if (++index >= argc) throw std::invalid_argument("missing option value");
            return argv[index];
        };
        try {
            if (std::strcmp(argument, "--algo") == 0) {
                config.algo = next();
            } else if (std::strcmp(argument, "--dtype") == 0) {
                if (!nano_nccl::parse_dtype(next(), &config.dtype)) {
                    throw std::invalid_argument("invalid dtype");
                }
            } else if (std::strcmp(argument, "--redop") == 0) {
                if (!nano_nccl::parse_redop(next(), &config.redop)) {
                    throw std::invalid_argument("invalid redop");
                }
            } else if (std::strcmp(argument, "--transport") == 0) {
                if (!nano_nccl::parse_transport(next(), &config.transport)) {
                    throw std::invalid_argument("invalid transport");
                }
            } else if (std::strcmp(argument, "--device") == 0) {
                config.device = std::atoi(next());
            } else if (std::strcmp(argument, "-b") == 0) {
                config.min_bytes = std::strtoull(next(), nullptr, 10);
            } else if (std::strcmp(argument, "-e") == 0) {
                config.max_bytes = std::strtoull(next(), nullptr, 10);
            } else if (std::strcmp(argument, "-f") == 0) {
                config.factor = std::atoi(next());
            } else if (std::strcmp(argument, "-w") == 0) {
                config.warmup_iters = std::atoi(next());
            } else if (std::strcmp(argument, "-n") == 0) {
                config.iters = std::atoi(next());
            } else {
                throw std::invalid_argument("unknown option");
            }
        } catch (const std::exception&) {
            usage(argv[0]);
            return 2;
        }
    }
    if (config.algo != "auto" && config.algo != "ring_simple") {
        usage(argv[0]);
        return 2;
    }

    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int exit_code = 0;
    try {
        int visible_devices = 0;
        cuda_check(cudaGetDeviceCount(&visible_devices), "cudaGetDeviceCount");
        if (config.device < 0) {
            const char* local_rank = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
            config.device = local_rank == nullptr ? 0 : std::atoi(local_rank);
        }
        if (config.device < 0 || config.device >= visible_devices) {
            throw std::runtime_error("benchmark selected an invalid CUDA device");
        }
        if (size != nano_nccl::kRanks) {
            throw std::runtime_error("MPI process count must match kRanks");
        }
        std::vector<nano_nccl::BenchResult> results;
        exit_code = run_benchmark(config, &results, rank);
        if (rank == 0) {
            std::printf("# nano-nccl one-process-per-GPU all_reduce_bench\n");
            std::printf("# %14s %8s %8s %10s %12s %12s %10s %10s %10s %8s %12s\n",
                        "algo", "dtype", "redop", "transport", "size(B)",
                        "count", "time(us)", "algbw", "busbw", "#wrong",
                        "max_abs");
            for (const auto& result : results) {
                std::printf("%14s %8s %8s %10s %12zu %12zu %10.2f %10.2f %10.2f %8d %12.6g\n",
                            result.algo.c_str(),
                            nano_nccl::dtype_name(result.dtype),
                            nano_nccl::redop_name(result.redop),
                            nano_nccl::transport_name(result.transport),
                            result.bytes, result.count, result.time_us,
                            result.algbw, result.busbw, result.wrong,
                            result.max_abs_error);
                if (result.wrong != 0) exit_code = 1;
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "rank=%d benchmark failed: %s\n", rank, error.what());
        exit_code = 1;
    }
    int global_exit_code = 0;
    MPI_Allreduce(&exit_code, &global_exit_code, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Finalize();
    return global_exit_code;
}
