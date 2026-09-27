#include "nano_nccl/communicator.h"
#include "nano_nccl/mpi.h"
#include "nano_nccl/traits.h"
#include "bench_wait.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

namespace {

enum class Collective { ReduceScatter, AllGather };

void mpi_check(int status, const char* op) {
    if (status == MPI_SUCCESS) return;
    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    throw std::runtime_error(std::string(op) + ": " + std::string(error, length));
}

void cuda_check(cudaError_t status, const char* op) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
}

void usage(const char* program) {
    std::fprintf(stderr, "Usage: %s --collective reducescatter|allgather [--dtype float|fp16|bf16] [--redop sum|avg|max|min] [--transport auto|shm|p2p|socket|rdma] [--wait-mode query|sync] [--device local-cuda-index] [-b bytes] [-e bytes] [-f factor] [-w warmup] [-n iters]\n", program);
}

template <nano_nccl::DType D>
int run_typed(const nano_nccl::BenchConfig& cfg, Collective collective,
              std::vector<nano_nccl::BenchResult>* results, int rank,
              nano_nccl::benchmarks::WaitMode wait_mode) {
    using Traits = nano_nccl::DTypeTraits<D>;
    using T = typename Traits::type;
    const auto sizes = nano_nccl::make_sizes(cfg.min_bytes, cfg.max_bytes, cfg.factor);
    if (sizes.empty()) throw std::invalid_argument("invalid size range");
    for (std::size_t bytes : sizes) {
        if (bytes % (sizeof(T) * static_cast<std::size_t>(nano_nccl::kRanks)) != 0)
            throw std::invalid_argument("bytes must be divisible by dtype size * nranks");
    }

    nano_nccl::CommunicatorConfig cc;
    cc.device = cfg.device;
    cc.transport = cfg.transport;
    auto communicator = nano_nccl::create_communicator_from_mpi(MPI_COMM_WORLD, cc);
    for (int edge = 0; edge < nano_nccl::kRanks; ++edge) {
        const int local_gdr = communicator->edge_uses_gdr(edge) ? 1 : 0;
        int min_gdr = 0, max_gdr = 0;
        mpi_check(MPI_Allreduce(&local_gdr, &min_gdr, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD), "MPI_Allreduce(min GDR)");
        mpi_check(MPI_Allreduce(&local_gdr, &max_gdr, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), "MPI_Allreduce(max GDR)");
        if (min_gdr != max_gdr) throw std::runtime_error("inconsistent GDR placement across ranks");
        if (rank == 0) {
            const auto backend = communicator->edge_transport(edge);
            const char* placement = backend == nano_nccl::TransportKind::Rdma
                ? (max_gdr ? "gdr" : "host-pinned") : "n/a";
            std::printf("# edge %d->%d transport=%s rdma_memory=%s\n", edge,
                        (edge + 1) % nano_nccl::kRanks,
                        nano_nccl::transport_name(backend), placement);
        }
    }
    cuda_check(cudaSetDevice(cfg.device), "cudaSetDevice");
    std::size_t max_count = sizes.back() / sizeof(T);
    T *send = nullptr, *recv = nullptr;
    cudaStream_t stream = nullptr;
    cuda_check(cudaMalloc(&send, max_count * sizeof(T)), "cudaMalloc(send)");
    cuda_check(cudaMalloc(&recv, max_count * sizeof(T)), "cudaMalloc(recv)");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

    for (std::size_t bytes : sizes) {
        const std::size_t count = bytes / sizeof(T);
        const std::size_t rank_count = count / static_cast<std::size_t>(nano_nccl::kRanks);
        const std::size_t input_count = collective == Collective::ReduceScatter ? count : rank_count;
        const std::size_t out_count = collective == Collective::ReduceScatter ? rank_count : count;
        std::vector<T> input(input_count);
        for (std::size_t i = 0; i < input_count; ++i) {
            // Encode both rank and position so misplaced chunks/elements fail validation.
            input[i] = Traits::from_float(static_cast<float>(rank + 1) +
                                          static_cast<float>(i % 4) * 0.125f);
        }
        cuda_check(cudaMemcpyAsync(send, input.data(), input_count * sizeof(T), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(input)");
        const bool rank_timing = std::getenv("NANO_NCCL_BENCH_DIAG_RANK_TIMING") != nullptr;
        bool measuring = false;
        double enqueue_sum_us = 0, wait_sum_us = 0;
        double enqueue_max_us = 0, wait_max_us = 0;
        double wait_first_us = 0;
        int measured_iteration = -1, wait_max_iteration = -1;
        int wait_over_300_count = 0, wait_over_1000_count = 0;
        auto launch = [&] {
            const auto begin = rank_timing && measuring
                ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (collective == Collective::ReduceScatter)
                communicator->reduce_scatter({send, recv, stream, out_count, D, cfg.redop});
            else
                communicator->all_gather({send, recv, stream, rank_count, D});
            const auto enqueued = rank_timing && measuring
                ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            nano_nccl::benchmarks::wait_for_stream(
                stream, wait_mode, [&] { communicator->check_async_error(); });
            if (rank_timing && measuring) {
                const auto completed = std::chrono::steady_clock::now();
                const double enqueue_us =
                    std::chrono::duration<double, std::micro>(enqueued - begin).count();
                const double wait_us =
                    std::chrono::duration<double, std::micro>(completed - enqueued).count();
                enqueue_sum_us += enqueue_us;
                wait_sum_us += wait_us;
                enqueue_max_us = std::max(enqueue_max_us, enqueue_us);
                if (measured_iteration == 0) wait_first_us = wait_us;
                if (wait_us > wait_max_us) {
                    wait_max_us = wait_us;
                    wait_max_iteration = measured_iteration;
                }
                wait_over_300_count += wait_us > 300.0;
                wait_over_1000_count += wait_us > 1000.0;
            }
        };
        for (int i = 1; i < cfg.warmup_iters; ++i) launch();
        mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(start)");
        // Barrier returns can be skewed across ranks; absorb that skew outside
        // the timed iterations with the final warmup collective.
        if (cfg.warmup_iters > 0) launch();
        const auto start = std::chrono::steady_clock::now();
        measuring = true;
        for (int i = 0; i < cfg.iters; ++i) {
            measured_iteration = i;
            launch();
        }
        measuring = false;
        const auto end = std::chrono::steady_clock::now();
        const double local_us = std::chrono::duration<double, std::micro>(end - start).count() / cfg.iters;
        if (rank_timing) {
            const auto start_steady_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    start.time_since_epoch()).count();
            std::fprintf(stderr,
                         "# rank_timing collective=%s rank=%d size=%zu start_steady_ns=%lld local_us=%.2f enqueue_avg_us=%.2f enqueue_max_us=%.2f wait_avg_us=%.2f wait_max_us=%.2f wait_first_us=%.2f wait_max_iteration=%d wait_over_300_count=%d wait_over_1000_count=%d\n",
                         collective == Collective::ReduceScatter ? "reduce_scatter" : "all_gather",
                         rank, bytes, static_cast<long long>(start_steady_ns),
                         local_us, enqueue_sum_us / cfg.iters,
                         enqueue_max_us, wait_sum_us / cfg.iters, wait_max_us,
                         wait_first_us, wait_max_iteration, wait_over_300_count,
                         wait_over_1000_count);
        }
        double max_us = 0;
        mpi_check(MPI_Allreduce(&local_us, &max_us, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), "MPI_Allreduce(time)");

        std::vector<T> output(out_count);
        cuda_check(cudaMemcpy(output.data(), recv, out_count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy(output)");
        int local_wrong = 0;
        float local_error = 0;
        for (std::size_t i = 0; i < out_count; ++i) {
            float want = 0;
            if (collective == Collective::ReduceScatter) {
                const std::size_t global_index =
                    static_cast<std::size_t>(rank) * rank_count + i;
                for (int peer = 0; peer < nano_nccl::kRanks; ++peer) {
                    const float value = Traits::to_float(Traits::from_float(
                        static_cast<float>(peer + 1) +
                        static_cast<float>(global_index % 4) * 0.125f));
                    if (cfg.redop == nano_nccl::RedOp::Sum ||
                        cfg.redop == nano_nccl::RedOp::Avg) want += value;
                    else if (cfg.redop == nano_nccl::RedOp::Max) want = std::fmax(want, value);
                    else want = peer == 0 ? value : std::fmin(want, value);
                }
                if (cfg.redop == nano_nccl::RedOp::Avg) want /= nano_nccl::kRanks;
            } else {
                const std::size_t source_rank = i / rank_count;
                const std::size_t source_index = i % rank_count;
                want = Traits::to_float(Traits::from_float(
                    static_cast<float>(source_rank + 1) +
                    static_cast<float>(source_index % 4) * 0.125f));
            }
            const float error = std::fabs(Traits::to_float(output[i]) - want);
            local_error = std::max(local_error, error);
            if (error > Traits::kDefaultEpsilon) local_wrong = 1;
        }
        int wrong = 0;
        float max_error = 0;
        mpi_check(MPI_Allreduce(&local_wrong, &wrong, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), "MPI_Allreduce(wrong)");
        mpi_check(MPI_Allreduce(&local_error, &max_error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD), "MPI_Allreduce(error)");
        if (rank == 0) {
            nano_nccl::BenchResult r;
            r.algo = "ring_simple"; r.dtype = D; r.redop = cfg.redop; r.transport = communicator->transport();
            r.bytes = bytes; r.count = count; r.time_us = max_us;
            r.algbw = nano_nccl::algbw_gbs(bytes, max_us);
            r.busbw = r.algbw * (static_cast<double>(nano_nccl::kRanks - 1) / nano_nccl::kRanks);
            r.wrong = wrong; r.max_abs_error = max_error;
            results->push_back(r);
        }
    }
    communicator.reset();
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    cuda_check(cudaFree(recv), "cudaFree(recv)");
    cuda_check(cudaFree(send), "cudaFree(send)");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    nano_nccl::BenchConfig cfg;
    cfg.warmup_iters = 5;
    cfg.iters = 20;
    auto wait_mode = nano_nccl::benchmarks::WaitMode::Query;
    Collective collective = Collective::ReduceScatter;
    bool selected = false, redop_selected = false;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        auto next = [&]() -> const char* { if (++i >= argc) throw std::invalid_argument("missing option value"); return argv[i]; };
        try {
            if (!std::strcmp(arg, "--collective")) {
                const char* v = next();
                if (!std::strcmp(v, "reducescatter") || !std::strcmp(v, "reduce_scatter")) collective = Collective::ReduceScatter;
                else if (!std::strcmp(v, "allgather") || !std::strcmp(v, "all_gather")) collective = Collective::AllGather;
                else throw std::invalid_argument("invalid collective");
                selected = true;
            } else if (!std::strcmp(arg, "--dtype")) {
                if (!nano_nccl::parse_dtype(next(), &cfg.dtype)) throw std::invalid_argument("invalid dtype");
            } else if (!std::strcmp(arg, "--redop")) {
                if (!nano_nccl::parse_redop(next(), &cfg.redop)) throw std::invalid_argument("invalid redop");
                redop_selected = true;
            } else if (!std::strcmp(arg, "--transport")) {
                if (!nano_nccl::parse_transport(next(), &cfg.transport)) throw std::invalid_argument("invalid transport");
            } else if (!std::strcmp(arg, "--wait-mode")) {
                if (!nano_nccl::benchmarks::parse_wait_mode(next(), &wait_mode)) throw std::invalid_argument("invalid wait mode");
            } else if (!std::strcmp(arg, "--device")) cfg.device = std::atoi(next());
            else if (!std::strcmp(arg, "-b")) cfg.min_bytes = std::strtoull(next(), nullptr, 10);
            else if (!std::strcmp(arg, "-e")) cfg.max_bytes = std::strtoull(next(), nullptr, 10);
            else if (!std::strcmp(arg, "-f")) cfg.factor = std::atoi(next());
            else if (!std::strcmp(arg, "-w")) cfg.warmup_iters = std::atoi(next());
            else if (!std::strcmp(arg, "-n")) cfg.iters = std::atoi(next());
            else throw std::invalid_argument("unknown option");
        } catch (...) { usage(argv[0]); return 2; }
    }
    if (!selected || cfg.warmup_iters < 0 || cfg.iters <= 0 ||
        (collective == Collective::AllGather && redop_selected)) { usage(argv[0]); return 2; }
    MPI_Init(&argc, &argv);
    int rank = 0, world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
    int exit_code = 0;
    try {
        int devices = 0; cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (cfg.device < 0) { const char* local = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"); cfg.device = local ? std::atoi(local) : 0; }
        if (cfg.device < 0 || cfg.device >= devices) throw std::runtime_error("invalid CUDA device");
        if (world != nano_nccl::kRanks) throw std::runtime_error("MPI process count must match kRanks");
        std::vector<nano_nccl::BenchResult> results;
        switch (cfg.dtype) {
            case nano_nccl::DType::Float: run_typed<nano_nccl::DType::Float>(cfg, collective, &results, rank, wait_mode); break;
            case nano_nccl::DType::Float16: run_typed<nano_nccl::DType::Float16>(cfg, collective, &results, rank, wait_mode); break;
            case nano_nccl::DType::BFloat16: run_typed<nano_nccl::DType::BFloat16>(cfg, collective, &results, rank, wait_mode); break;
        }
        if (rank == 0) {
            std::printf("# nano-nccl %s benchmark; size is per-rank collective buffer bytes\n", collective == Collective::ReduceScatter ? "reduce_scatter" : "all_gather");
            std::printf("# timing=host_wall_per_iteration_complete wait_mode=%s rank_aggregation=max warmup=%d iters=%d\n",
                        nano_nccl::benchmarks::wait_mode_name(wait_mode),
                        cfg.warmup_iters, cfg.iters);
            std::printf("# algo dtype redop transport size(B) count time(us) algbw busbw wrong max_abs\n");
            for (const auto& r : results) {
                std::printf("%s %s %s %s %zu %zu %.2f %.2f %.2f %d %.6g\n", r.algo.c_str(), nano_nccl::dtype_name(r.dtype), nano_nccl::redop_name(r.redop), nano_nccl::transport_name(r.transport), r.bytes, r.count, r.time_us, r.algbw, r.busbw, r.wrong, r.max_abs_error);
                if (r.wrong) exit_code = 1;
            }
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "rank=%d benchmark failed: %s\n", rank, e.what()); exit_code = 1; }
    int global = 0; MPI_Allreduce(&exit_code, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); MPI_Finalize();
    return global;
}
