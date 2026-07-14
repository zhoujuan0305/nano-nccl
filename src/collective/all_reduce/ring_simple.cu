#include "collective/all_reduce/ring_simple.h"
#include "core/buffer.h"
#include "core/numa.h"
#include "core/stream.h"
#include "kernels/ring_simple_kernel.cuh"
#include "nano_nccl/all_reduce.h"
#include "nano_nccl/traits.h"
#include "nano_nccl/types.h"
#include "transport/p2p/p2p_fifo.h"
#include "transport/p2p/p2p_step_counters.h"
#include "transport/shm/shm_fifo.h"
#include "transport/shm/shm_step.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace nano_nccl::collective::all_reduce {

namespace {

using nano_nccl::core::DeviceBuffer;
using nano_nccl::core::MappedBuffer;
using nano_nccl::core::MappedU64Array;
using nano_nccl::core::Stream;
using nano_nccl::transport::SimpleControlArgs;
using nano_nccl::transport::SimpleFifoArgs;
using nano_nccl::kernels::ring_simple_kernel;
using nano_nccl::DType;
using nano_nccl::RedOp;
using nano_nccl::TransportKind;

float input_value(int rank, std::size_t idx) {
    int bucket = static_cast<int>(idx % 251);
    return static_cast<float>(rank + 1) * 0.125f +
           static_cast<float>(bucket) * 0.00025f;
}

// 当前只保留 ring_simple 路径，auto 直接选它。
std::string select_auto_algo(std::size_t /*bytes*/) {
    return "ring_simple";
}

template <DType kDType>
void fill_inputs(
    std::vector<typename DTypeTraits<kDType>::type> host_inputs[kRanks],
    std::vector<float>* expected, std::size_t count) {
    using T = typename DTypeTraits<kDType>::type;
    expected->assign(count, 0.0f);
    for (int rank = 0; rank < kRanks; ++rank) {
        host_inputs[rank].resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            host_inputs[rank][i] =
                DTypeTraits<kDType>::from_float(input_value(rank, i));
        }
    }
    // 期望值按 Sum redop 对所有 rank 求和，rank 数由 kRanks 决定。
    for (std::size_t i = 0; i < count; ++i) {
        float sum = 0.0f;
        for (int rank = 0; rank < kRanks; ++rank) {
            sum += DTypeTraits<kDType>::to_float(host_inputs[rank][i]);
        }
        (*expected)[i] = sum;
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
    bool debug_first_wrong =
        std::getenv("NANO_NCCL_DEBUG_FIRST_WRONG") != nullptr;
    std::size_t bytes = expected.size() * sizeof(T);
    for (int rank = 0; rank < kRanks; ++rank) {
        CUDA_CHECK_THROW(cudaSetDevice(rank));
        CUDA_CHECK_THROW(cudaMemcpy(actual.data(), outputs[rank]->get(), bytes,
                                    cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < expected.size(); ++i) {
            float actual_value = DTypeTraits<kDType>::to_float(actual[i]);
            float err = std::fabs(actual_value - expected[i]);
            *max_abs_error = std::max(*max_abs_error, err);
            if (err > epsilon) {
                ++wrong;
                if (debug_first_wrong) {
                    std::fprintf(stderr,
                                 "first wrong rank=%d index=%zu actual=%g "
                                 "expected=%g abs=%g\n",
                                 rank, i, actual_value, expected[i], err);
                }
                break;
            }
        }
    }
    return wrong;
}

void sync_streams(Stream* streams[kRanks]) {
    for (int dev = 0; dev < kRanks; ++dev) {
        CUDA_CHECK_THROW(cudaSetDevice(dev));
        CUDA_CHECK_THROW(cudaStreamSynchronize(streams[dev]->get()));
    }
}

void require_devices() {
    int device_count = 0;
    CUDA_CHECK_THROW(cudaGetDeviceCount(&device_count));
    if (device_count < kRanks) {
        throw std::runtime_error("need at least " +
                                 std::to_string(kRanks) +
                                 " visible CUDA devices");
    }
}

void require_bf16_devices() {
    for (int rank = 0; rank < kRanks; ++rank) {
        cudaDeviceProp props{};
        CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, rank));
        if (props.major < 8) {
            throw std::runtime_error(
                "bf16 requires compute capability 8.0 or newer");
        }
    }
}

template <typename T, DType kDType>
class AllReduceRunner {
public:
    explicit AllReduceRunner(std::size_t max_count, TransportKind requested)
        : max_count_(max_count) {
        try {
            require_devices();
            transport_plan_ = transport::p2p::resolve_ring_transport(requested);
            if (transport_plan_.uses_p2p()) {
                transport::p2p::enable_p2p_ring_peer_access_or_throw(
                    transport_plan_);
            }

            for (int rank = 0; rank < kRanks; ++rank) {
                inputs_[rank] = new DeviceBuffer<T>(rank, max_count_);
                outputs_[rank] = new DeviceBuffer<T>(rank, max_count_);
                streams_[rank] = new Stream(rank);
            }

            // step counter 跨迭代持久化，容量覆盖 2 * kChannels * kRanks（kind/channel/edge）。
            simple_fifo_steps_.reset(2 * kChannels * kRanks);
            simple_fifo_base_step_.reset(kRanks * kChannels);
            for (int channel = 0; channel < kChannels; ++channel) {
                std::size_t part_offset = 0;
                std::size_t part_count = 0;
                std::size_t chunk_count = 0;
                transport::shm::cbd_part<T>(max_count_, channel, &part_offset,
                                            &part_count, &chunk_count);
                simple_fifo_slot_elems_ =
                    std::max(simple_fifo_slot_elems_, chunk_count);
            }
            simple_fifo_slot_elems_ =
                std::max<std::size_t>(simple_fifo_slot_elems_, 1);
            if (transport_plan_.uses_p2p()) {
                p2p_fifo_ = new transport::p2p::P2pFifo<T>(
                    simple_fifo_slot_elems_, transport_plan_);
                p2p_steps_ = new transport::p2p::P2pStepCounters(
                    transport_plan_);
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~AllReduceRunner() { cleanup(); }

    void load_inputs(std::vector<T> host_inputs[kRanks], std::size_t count) {
        std::size_t bytes = count * sizeof(T);
        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK_THROW(cudaSetDevice(rank));
            CUDA_CHECK_THROW(cudaMemcpy(inputs_[rank]->get(),
                                        host_inputs[rank].data(), bytes,
                                        cudaMemcpyHostToDevice));
            CUDA_CHECK_THROW(cudaMemset(outputs_[rank]->get(), 0, bytes));
        }
        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK_THROW(cudaSetDevice(rank));
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
        }
    }

    void run_once(std::size_t count) {
        ensure_fifo_buffers();
        reset_control();
        launch_ring_simple(count);
        sync_streams(streams_);
    }

    // 批量异步提交：用 CUDA events 做跨流屏障代替每迭代 cudaStreamSynchronize，
    // 对齐 NCCL BenchTime 计时口径。
    void run_batch(std::size_t count, int iters) {
        ensure_fifo_buffers();
        reset_control();

        cudaEvent_t iter_events[kRanks];
        for (int r = 0; r < kRanks; ++r) {
            CUDA_CHECK_THROW(cudaSetDevice(r));
            CUDA_CHECK_THROW(cudaEventCreateWithFlags(&iter_events[r],
                                                      cudaEventDisableTiming));
        }

        for (int iter = 0; iter < iters; ++iter) {
            if (iter > 0) {
                // 下一轮所有 stream 必须等上一轮所有 rank 的 kernel 完成，
                // 否则跨 rank 的 step counter 读写会乱序。
                for (int s = 0; s < kRanks; ++s) {
                    CUDA_CHECK_THROW(cudaSetDevice(s));
                    for (int r = 0; r < kRanks; ++r) {
                        CUDA_CHECK_THROW(cudaStreamWaitEvent(streams_[s]->get(),
                                                             iter_events[r], 0));
                    }
                }
            }
            launch_ring_simple(count);
            for (int r = 0; r < kRanks; ++r) {
                CUDA_CHECK_THROW(cudaSetDevice(r));
                CUDA_CHECK_THROW(cudaEventRecord(iter_events[r],
                                                 streams_[r]->get()));
            }
        }

        sync_streams(streams_);
        for (int r = 0; r < kRanks; ++r) {
            CUDA_CHECK_THROW(cudaEventDestroy(iter_events[r]));
        }
    }

    int verify(const std::vector<float>& expected, float epsilon,
               float* max_abs_error) {
        return verify_outputs<kDType>(outputs_, expected, epsilon,
                                      max_abs_error);
    }

    TransportKind transport() const { return transport_plan_.resolved_kind(); }

private:
    void cleanup() {
        for (int channel = 0; channel < kChannels; ++channel) {
            for (int edge = 0; edge < kRanks; ++edge) {
                delete simple_fifo_[channel][edge];
                simple_fifo_[channel][edge] = nullptr;
            }
        }
        delete p2p_fifo_;
        p2p_fifo_ = nullptr;
        delete p2p_steps_;
        p2p_steps_ = nullptr;
        for (int rank = 0; rank < kRanks; ++rank) {
            delete streams_[rank];
            streams_[rank] = nullptr;
            delete outputs_[rank];
            outputs_[rank] = nullptr;
            delete inputs_[rank];
            inputs_[rank] = nullptr;
        }
    }

    // Ring 拓扑：edge i 的 sender 是 rank i，receiver 是 rank (i+1)%kRanks。
    int ring_edge_recv_gpu(int edge) const {
        return (edge + 1) % kRanks;
    }

    [[maybe_unused]] int ring_edge_send_gpu(int edge) const {
        return edge;
    }

    // FIFO buffer 按 receiver NUMA 节点分配，避免跨 NUMA 带宽损失。
    void ensure_fifo_buffers() {
        for (int channel = 0; channel < kChannels; ++channel) {
            for (int edge = 0; edge < kRanks; ++edge) {
                if (transport_plan_.edge_kind(edge) != TransportKind::Shm) {
                    continue;
                }
                if (simple_fifo_[channel][edge] == nullptr) {
                    int recv_gpu = ring_edge_recv_gpu(edge);
                    int numa_node = core::gpu_numa_node(recv_gpu);
                    simple_fifo_[channel][edge] = new MappedBuffer<T>(
                        transport::shm::kSimpleFifoSteps *
                            simple_fifo_slot_elems_,
                        numa_node);
                }
            }
        }
    }

    void reset_control() {
        simple_fifo_steps_.clear_host();
        simple_fifo_base_step_.clear_host();
        if (transport_plan_.uses_p2p()) {
            cudaStream_t raw_streams[kRanks];
            for (int rank = 0; rank < kRanks; ++rank) {
                raw_streams[rank] = streams_[rank]->get();
            }
            p2p_steps_->reset(raw_streams);

            cudaEvent_t reset_events[kRanks];
            for (int rank = 0; rank < kRanks; ++rank) {
                CUDA_CHECK_THROW(cudaSetDevice(rank));
                CUDA_CHECK_THROW(cudaEventCreateWithFlags(
                    &reset_events[rank], cudaEventDisableTiming));
                CUDA_CHECK_THROW(cudaEventRecord(reset_events[rank],
                                                 streams_[rank]->get()));
            }
            // 本 rank 的 kernel 会发布相邻 GPU 的 counter，必须先等所有
            // device-resident counter 都完成清零，避免清零覆盖首个发布。
            for (int stream_rank = 0; stream_rank < kRanks; ++stream_rank) {
                CUDA_CHECK_THROW(cudaSetDevice(stream_rank));
                for (int event_rank = 0; event_rank < kRanks; ++event_rank) {
                    CUDA_CHECK_THROW(cudaStreamWaitEvent(
                        streams_[stream_rank]->get(), reset_events[event_rank],
                        0));
                }
            }
            for (int rank = 0; rank < kRanks; ++rank) {
                CUDA_CHECK_THROW(cudaSetDevice(rank));
                CUDA_CHECK_THROW(cudaEventDestroy(reset_events[rank]));
            }
        }
    }

    void launch_ring_simple(std::size_t count) {
        for (int rank = 0; rank < kRanks; ++rank) {
            SimpleFifoArgs<T> args{};
            args.rank = rank;
            args.count = count;
            args.slot_elems = simple_fifo_slot_elems_;
            args.step_elems = transport::shm::simple_fifo_step_elems<T>();
            args.input = inputs_[rank]->get();
            args.output = outputs_[rank]->get();
            SimpleControlArgs shm_control =
                transport::shm::make_simple_control_args(
                    simple_fifo_steps_.device_ptr(rank),
                    simple_fifo_base_step_.device_ptr(rank), rank);
            SimpleControlArgs p2p_control{};
            if (p2p_steps_ != nullptr) {
                p2p_control = p2p_steps_->control_args(rank);
            }

            int next = (rank + 1) % kRanks;
            int prev = (rank + kRanks - 1) % kRanks;
            int send_edge = transport::shm::ring_edge_index(rank, next, kRanks);
            int recv_edge = transport::shm::ring_edge_index(prev, rank, kRanks);
            if (send_edge < 0 || recv_edge < 0) {
                throw std::runtime_error(
                    "ring_simple saw an unexpected ring edge");
            }
            for (int channel = 0; channel < kChannels; ++channel) {
                if (transport_plan_.edge_kind(send_edge) == TransportKind::P2p) {
                    args.send_fifo[channel] =
                        p2p_fifo_->edge_ptr(channel, send_edge);
                } else {
                    args.send_fifo[channel] =
                        simple_fifo_[channel][send_edge]->device_ptr(rank);
                }
                if (transport_plan_.edge_kind(recv_edge) == TransportKind::P2p) {
                    args.recv_fifo[channel] =
                        p2p_fifo_->edge_ptr(channel, recv_edge);
                } else {
                    args.recv_fifo[channel] =
                        simple_fifo_[channel][recv_edge]->device_ptr(rank);
                }
                args.control.send_head[channel] =
                    transport_plan_.edge_kind(send_edge) == TransportKind::P2p
                        ? p2p_control.send_head[channel]
                        : shm_control.send_head[channel];
                args.control.send_tail[channel] =
                    transport_plan_.edge_kind(send_edge) == TransportKind::P2p
                        ? p2p_control.send_tail[channel]
                        : shm_control.send_tail[channel];
                args.control.recv_tail[channel] =
                    transport_plan_.edge_kind(recv_edge) == TransportKind::P2p
                        ? p2p_control.recv_tail[channel]
                        : shm_control.recv_tail[channel];
                args.control.recv_head[channel] =
                    transport_plan_.edge_kind(recv_edge) == TransportKind::P2p
                        ? p2p_control.recv_head[channel]
                        : shm_control.recv_head[channel];
            }
            args.control.base_steps = p2p_control.base_steps != nullptr
                                          ? p2p_control.base_steps
                                          : shm_control.base_steps;

            CUDA_CHECK_THROW(cudaSetDevice(rank));
            ring_simple_kernel<kRanks, T, RedOp::Sum>
                <<<kChannels, kBlockThreads, 0, streams_[rank]->get()>>>(args);
            CUDA_CHECK_THROW(cudaGetLastError());
        }
    }

    std::size_t max_count_ = 0;
    DeviceBuffer<T>* inputs_[kRanks]{};
    DeviceBuffer<T>* outputs_[kRanks]{};
    Stream* streams_[kRanks]{};
    MappedU64Array simple_fifo_steps_;
    MappedU64Array simple_fifo_base_step_;
    MappedBuffer<T>* simple_fifo_[kChannels][kRanks]{};
    transport::p2p::P2pFifo<T>* p2p_fifo_ = nullptr;
    transport::p2p::P2pStepCounters* p2p_steps_ = nullptr;
    std::size_t simple_fifo_slot_elems_ = 0;
    transport::p2p::RingTransportPlan transport_plan_ =
        transport::p2p::RingTransportPlan::uniform(TransportKind::Shm);
};

template <DType kDType>
int run_ring_simple_bench_typed(const BenchConfig& config,
                                std::vector<BenchResult>* results) {
    using T = typename DTypeTraits<kDType>::type;
    try {
        auto sizes = make_sizes(config.min_bytes, config.max_bytes, config.factor);
        if (sizes.empty()) {
            std::fprintf(stderr, "invalid size range\n");
            return 2;
        }
        for (std::size_t bytes : sizes) {
            if (bytes % sizeof(T) != 0) {
                std::fprintf(stderr,
                             "size must be divisible by dtype size: %zu\n",
                             bytes);
                return 2;
            }
        }
        require_devices();
        if constexpr (kDType == DType::BFloat16) {
            require_bf16_devices();
        }
        AllReduceRunner<T, kDType> runner(sizes.back() / sizeof(T),
                                          config.transport);

        for (std::size_t bytes : sizes) {
            std::size_t count = bytes / sizeof(T);
            std::vector<T> host_inputs[kRanks];
            std::vector<float> expected;
            fill_inputs<kDType>(host_inputs, &expected, count);
            runner.load_inputs(host_inputs, count);
            std::string algo =
                (config.algo == "auto") ? select_auto_algo(bytes) : config.algo;
            if (algo != "ring_simple") {
                std::fprintf(stderr, "unsupported algo: %s\n", algo.c_str());
                return 2;
            }

            for (int i = 0; i < config.warmup_iters; ++i) {
                runner.run_once(count);
            }

            auto start = std::chrono::steady_clock::now();
            runner.run_batch(count, config.iters);
            auto end = std::chrono::steady_clock::now();
            double total_us =
                std::chrono::duration<double, std::micro>(end - start).count();
            double time_us = total_us / static_cast<double>(config.iters);

            float max_abs_error = 0.0f;
            float epsilon = config.epsilon <= 0.0f
                                ? DTypeTraits<kDType>::kDefaultEpsilon
                                : config.epsilon;
            int wrong = runner.verify(expected, epsilon, &max_abs_error);

            BenchResult result;
            result.algo = algo;
            result.dtype = kDType;
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
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}

}  // namespace

int run_ring_simple_bench(const BenchConfig& config,
                          std::vector<BenchResult>* results) {
    switch (config.dtype) {
        case DType::Float:
            return run_ring_simple_bench_typed<DType::Float>(config, results);
        case DType::Float16:
            return run_ring_simple_bench_typed<DType::Float16>(config, results);
        case DType::BFloat16:
            return run_ring_simple_bench_typed<DType::BFloat16>(config, results);
    }
    std::fprintf(stderr, "unsupported dtype\n");
    return 2;
}

}  // namespace nano_nccl::collective::all_reduce

// 显式实例化：NRanks/kRanks 由 CMake 配置，dtype 和 RedOp 由 host dispatch 选择。
// 必须在 kernels 命名空间内显式实例化。
namespace nano_nccl::kernels {
template __global__ void ring_simple_kernel<nano_nccl::kRanks, float, nano_nccl::RedOp::Sum>(
    nano_nccl::transport::SimpleFifoArgs<float>);
template __global__ void ring_simple_kernel<nano_nccl::kRanks, __half, nano_nccl::RedOp::Sum>(
    nano_nccl::transport::SimpleFifoArgs<__half>);
template __global__ void ring_simple_kernel<nano_nccl::kRanks, __nv_bfloat16,
                                            nano_nccl::RedOp::Sum>(
    nano_nccl::transport::SimpleFifoArgs<__nv_bfloat16>);
}  // namespace nano_nccl::kernels

namespace nano_nccl {

// 公共 API 委托到 collective::all_reduce 实现。
int run_all_reduce_bench(const BenchConfig& config,
                         std::vector<BenchResult>* results) {
    return collective::all_reduce::run_ring_simple_bench(config, results);
}

std::vector<std::size_t> make_sizes(std::size_t min_bytes,
                                    std::size_t max_bytes, int factor) {
    std::vector<std::size_t> sizes;
    if (min_bytes == 0 || max_bytes < min_bytes || factor < 2) {
        return sizes;
    }
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
    if (time_us <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / time_us / 1000.0;
}

double all_reduce_busbw_gbs(double algbw, int nranks) {
    return algbw * (2.0 * static_cast<double>(nranks - 1) /
                    static_cast<double>(nranks));
}

}  // namespace nano_nccl
