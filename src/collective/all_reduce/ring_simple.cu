#include "collective/all_reduce/ring_simple.h"

#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
#include "collective/all_reduce/bench_profiling.h"
#endif

#include "core/buffer.h"
#include "core/stream.h"
#include "nano_nccl/communicator.h"
#include "nano_nccl/traits.h"
#include "transport/p2p/p2p_topology.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace nano_nccl::collective::all_reduce {

namespace {

using nano_nccl::core::DeviceBuffer;
using nano_nccl::core::Stream;

float input_value(int rank, std::size_t index) {
    int bucket = static_cast<int>(index % 251);
    int rank_weight = (rank + static_cast<int>(index % kRanks)) % kRanks;
    return static_cast<float>(rank_weight + 1) * 0.125f +
           static_cast<float>(bucket) * 0.00025f;
}

std::string select_auto_algo(std::size_t /*bytes*/) { return "ring_simple"; }

template <DType kDType, RedOp kRedOp>
void fill_inputs(
    std::vector<typename DTypeTraits<kDType>::type> host_inputs[kRanks],
    std::vector<float>* expected, std::size_t count) {
    using T = typename DTypeTraits<kDType>::type;
    expected->resize(count);
    for (int rank = 0; rank < kRanks; ++rank) {
        host_inputs[rank].resize(count);
        for (std::size_t index = 0; index < count; ++index) {
            host_inputs[rank][index] =
                DTypeTraits<kDType>::from_float(input_value(rank, index));
        }
    }
    for (std::size_t index = 0; index < count; ++index) {
        float value = DTypeTraits<kDType>::to_float(host_inputs[0][index]);
        for (int rank = 1; rank < kRanks; ++rank) {
            float next = DTypeTraits<kDType>::to_float(host_inputs[rank][index]);
            if constexpr (kRedOp == RedOp::Sum || kRedOp == RedOp::Avg) {
                value += next;
            } else if constexpr (kRedOp == RedOp::Max) {
                value = std::fmax(value, next);
            } else {
                value = std::fmin(value, next);
            }
        }
        if constexpr (kRedOp == RedOp::Avg) value /= static_cast<float>(kRanks);
        (*expected)[index] = value;
    }
}

template <DType kDType>
int verify_outputs(
    DeviceBuffer<typename DTypeTraits<kDType>::type>* outputs[kRanks],
    const std::vector<float>& expected, float epsilon, float* max_abs_error) {
    using T = typename DTypeTraits<kDType>::type;
    std::vector<T> actual(expected.size());
    int wrong = 0;
    *max_abs_error = 0.0f;
    bool debug_first_wrong = std::getenv("NANO_NCCL_DEBUG_FIRST_WRONG") != nullptr;
    for (int rank = 0; rank < kRanks; ++rank) {
        CUDA_CHECK_THROW(cudaSetDevice(rank));
        CUDA_CHECK_THROW(cudaMemcpy(actual.data(), outputs[rank]->get(),
                                    actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
        for (std::size_t index = 0; index < expected.size(); ++index) {
            float actual_value = DTypeTraits<kDType>::to_float(actual[index]);
            float error = std::fabs(actual_value - expected[index]);
            *max_abs_error = std::max(*max_abs_error, error);
            if (error > epsilon) {
                ++wrong;
                if (debug_first_wrong) {
                    std::fprintf(stderr,
                                 "first wrong rank=%d index=%zu actual=%g expected=%g abs=%g\\n",
                                 rank, index, actual_value, expected[index], error);
                }
                break;
            }
        }
    }
    return wrong;
}

void sync_streams(Stream* streams[kRanks]) {
    for (int rank = 0; rank < kRanks; ++rank) {
        CUDA_CHECK_THROW(cudaSetDevice(rank));
        CUDA_CHECK_THROW(cudaStreamSynchronize(streams[rank]->get()));
    }
}

template <typename T, DType kDType, RedOp kRedOp>
class AllReduceBenchRunner {
public:
    explicit AllReduceBenchRunner(std::size_t max_count, TransportKind transport) {
        CommunicatorConfig config;
        for (int rank = 0; rank < kRanks; ++rank) config.devices.push_back(rank);
        config.transport = transport;
        communicator_ = create_communicator(config);
        resolved_transport_ = transport::p2p::resolve_ring_transport(transport).resolved_kind();
        for (int rank = 0; rank < kRanks; ++rank) {
            inputs_[rank] = std::make_unique<DeviceBuffer<T>>(rank, max_count);
            outputs_[rank] = std::make_unique<DeviceBuffer<T>>(rank, max_count);
            streams_[rank] = std::make_unique<Stream>(rank);
        }
    }

    void load_inputs(std::vector<T> host_inputs[kRanks], std::size_t count) {
        std::size_t bytes = count * sizeof(T);
        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK_THROW(cudaSetDevice(rank));
            CUDA_CHECK_THROW(cudaMemcpy(inputs_[rank]->get(), host_inputs[rank].data(), bytes,
                                        cudaMemcpyHostToDevice));
            CUDA_CHECK_THROW(cudaMemset(outputs_[rank]->get(), 0, bytes));
        }
        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK_THROW(cudaSetDevice(rank));
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
        }
    }

    void run_once(std::size_t count) {
        communicator_->all_reduce(make_args(count));
        sync_streams(raw_streams());
    }

    void run_batch(std::size_t count, int iters, std::size_t bytes) {
        cudaEvent_t iter_events[kRanks];
        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK_THROW(cudaSetDevice(rank));
            CUDA_CHECK_THROW(cudaEventCreateWithFlags(&iter_events[rank], cudaEventDisableTiming));
        }

        Stream* streams[kRanks];
        for (int rank = 0; rank < kRanks; ++rank) streams[rank] = streams_[rank].get();
        for (int iteration = 0; iteration < iters; ++iteration) {
            if (iteration > 0) {
                for (int stream_rank = 0; stream_rank < kRanks; ++stream_rank) {
                    CUDA_CHECK_THROW(cudaSetDevice(stream_rank));
                    for (int event_rank = 0; event_rank < kRanks; ++event_rank) {
                        CUDA_CHECK_THROW(cudaStreamWaitEvent(streams[stream_rank]->get(),
                                                             iter_events[event_rank], 0));
                    }
                }
            }
#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
            bench_profiling::NvtxRange iteration_range(
                bench_profiling::all_reduce_iteration_range_name(bytes, iteration));
#endif
            communicator_->all_reduce(make_args(count));
            for (int rank = 0; rank < kRanks; ++rank) {
                CUDA_CHECK_THROW(cudaSetDevice(rank));
                CUDA_CHECK_THROW(cudaEventRecord(iter_events[rank], streams[rank]->get()));
            }
        }

        sync_streams(streams);
        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK_THROW(cudaSetDevice(rank));
            CUDA_CHECK_THROW(cudaEventDestroy(iter_events[rank]));
        }
    }

    int verify(const std::vector<float>& expected, float epsilon, float* max_abs_error) {
        DeviceBuffer<T>* outputs[kRanks];
        for (int rank = 0; rank < kRanks; ++rank) outputs[rank] = outputs_[rank].get();
        return verify_outputs<kDType>(outputs, expected, epsilon, max_abs_error);
    }

    TransportKind transport() const { return resolved_transport_; }

private:
    CollectiveArgs make_args(std::size_t count) const {
        CollectiveArgs args;
        args.count = count;
        args.dtype = kDType;
        args.redop = kRedOp;
        for (int rank = 0; rank < kRanks; ++rank) {
            args.send_buffers.push_back(inputs_[rank]->get());
            args.recv_buffers.push_back(outputs_[rank]->get());
            args.streams.push_back(streams_[rank]->get());
        }
        return args;
    }

    Stream** raw_streams() {
        for (int rank = 0; rank < kRanks; ++rank) raw_streams_[rank] = streams_[rank].get();
        return raw_streams_;
    }

    std::unique_ptr<Communicator> communicator_;
    std::unique_ptr<DeviceBuffer<T>> inputs_[kRanks];
    std::unique_ptr<DeviceBuffer<T>> outputs_[kRanks];
    std::unique_ptr<Stream> streams_[kRanks];
    Stream* raw_streams_[kRanks]{};
    TransportKind resolved_transport_ = TransportKind::Shm;
};

template <DType kDType, RedOp kRedOp>
int run_ring_simple_bench_typed(const BenchConfig& config,
                                std::vector<BenchResult>* results) {
    using T = typename DTypeTraits<kDType>::type;
    try {
        auto sizes = make_sizes(config.min_bytes, config.max_bytes, config.factor);
        if (sizes.empty()) {
            std::fprintf(stderr, "invalid size range\\n");
            return 2;
        }
        for (std::size_t bytes : sizes) {
            if (bytes % sizeof(T) != 0) {
                std::fprintf(stderr, "size must be divisible by dtype size: %zu\\n", bytes);
                return 2;
            }
        }

        AllReduceBenchRunner<T, kDType, kRedOp> runner(
            sizes.back() / sizeof(T), config.transport);
        for (std::size_t bytes : sizes) {
            std::size_t count = bytes / sizeof(T);
            std::vector<T> host_inputs[kRanks];
            std::vector<float> expected;
            fill_inputs<kDType, kRedOp>(host_inputs, &expected, count);
            runner.load_inputs(host_inputs, count);
            std::string algo = config.algo == "auto" ? select_auto_algo(bytes) : config.algo;
            if (algo != "ring_simple") {
                std::fprintf(stderr, "unsupported algo: %s\\n", algo.c_str());
                return 2;
            }

            for (int iteration = 0; iteration < config.warmup_iters; ++iteration) {
                runner.run_once(count);
            }
            double time_us;
#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
            bench_profiling::ProfilerSession profiler;
            {
                bench_profiling::NvtxRange size_range(
                    bench_profiling::all_reduce_size_range_name(bytes));
#endif
            auto start = std::chrono::steady_clock::now();
            runner.run_batch(count, config.iters, bytes);
            auto end = std::chrono::steady_clock::now();
            time_us = std::chrono::duration<double, std::micro>(end - start).count() /
                      static_cast<double>(config.iters);
#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)
            }
            profiler.stop();
#endif

            float max_abs_error = 0.0f;
            float epsilon = config.epsilon <= 0.0f ? DTypeTraits<kDType>::kDefaultEpsilon
                                                    : config.epsilon;
            int wrong = runner.verify(expected, epsilon, &max_abs_error);
            BenchResult result;
            result.algo = algo;
            result.dtype = kDType;
            result.redop = kRedOp;
            result.transport = runner.transport();
            result.bytes = bytes;
            result.count = count;
            result.time_us = time_us;
            result.algbw = algbw_gbs(bytes, time_us);
            result.busbw = all_reduce_busbw_gbs(result.algbw, kRanks);
            result.wrong = wrong;
            result.max_abs_error = max_abs_error;
            results->push_back(result);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\\n", error.what());
        return 1;
    }
}

}  // namespace

template <DType kDType>
int run_ring_simple_bench_redop(const BenchConfig& config,
                                std::vector<BenchResult>* results) {
    switch (config.redop) {
        case RedOp::Sum:
            return run_ring_simple_bench_typed<kDType, RedOp::Sum>(config, results);
        case RedOp::Avg:
            return run_ring_simple_bench_typed<kDType, RedOp::Avg>(config, results);
        case RedOp::Max:
            return run_ring_simple_bench_typed<kDType, RedOp::Max>(config, results);
        case RedOp::Min:
            return run_ring_simple_bench_typed<kDType, RedOp::Min>(config, results);
    }
    std::fprintf(stderr, "unsupported reduction operation\n");
    return 2;
}

int run_ring_simple_bench(const BenchConfig& config,
                          std::vector<BenchResult>* results) {
    switch (config.dtype) {
        case DType::Float:
            return run_ring_simple_bench_redop<DType::Float>(config, results);
        case DType::Float16:
            return run_ring_simple_bench_redop<DType::Float16>(config, results);
        case DType::BFloat16:
            return run_ring_simple_bench_redop<DType::BFloat16>(config, results);
    }
    std::fprintf(stderr, "unsupported dtype\\n");
    return 2;
}

}  // namespace nano_nccl::collective::all_reduce

namespace nano_nccl {

int run_all_reduce_bench(const BenchConfig& config,
                         std::vector<BenchResult>* results) {
    return collective::all_reduce::run_ring_simple_bench(config, results);
}

std::vector<std::size_t> make_sizes(std::size_t min_bytes,
                                    std::size_t max_bytes, int factor) {
    std::vector<std::size_t> sizes;
    if (min_bytes == 0 || max_bytes < min_bytes || factor < 2) return sizes;
    for (std::size_t size = min_bytes; size <= max_bytes; size *= factor) {
        sizes.push_back(size);
        if (size > std::numeric_limits<std::size_t>::max() /
                       static_cast<std::size_t>(factor)) {
            break;
        }
    }
    return sizes;
}

double algbw_gbs(std::size_t bytes, double time_us) {
    return time_us <= 0.0 ? 0.0 : static_cast<double>(bytes) / time_us / 1000.0;
}

double all_reduce_busbw_gbs(double algbw, int nranks) {
    return algbw * (2.0 * static_cast<double>(nranks - 1) /
                    static_cast<double>(nranks));
}

}  // namespace nano_nccl
