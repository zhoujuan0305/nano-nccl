#include "collective/all_reduce/ring_simple.h"
#include "core/buffer.h"
#include "core/numa.h"
#include "core/stream.h"
#include "kernels/ring_simple_kernel.cuh"
#include "nano_nccl/all_reduce.h"
#include "nano_nccl/traits.h"
#include "nano_nccl/types.h"
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
using nano_nccl::transport::shm::ShmFifoArgs;
using nano_nccl::kernels::ring_simple_kernel;
using nano_nccl::DType;
using nano_nccl::RedOp;

float input_value(int rank, std::size_t idx) {
    int bucket = static_cast<int>(idx % 251);
    return static_cast<float>(rank + 1) * 0.125f +
           static_cast<float>(bucket) * 0.00025f;
}

// 当前只保留 ring_simple 路径，auto 直接选它。
std::string select_auto_algo(std::size_t /*bytes*/) {
    return "ring_simple";
}

void fill_inputs(std::vector<float> host_inputs[kRanks],
                 std::vector<float>* expected, std::size_t count) {
    expected->assign(count, 0.0f);
    for (int rank = 0; rank < kRanks; ++rank) {
        host_inputs[rank].resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            host_inputs[rank][i] = input_value(rank, i);
        }
    }
    // 期望值按 Sum redop 对所有 rank 求和，rank 数由 kRanks 决定。
    for (std::size_t i = 0; i < count; ++i) {
        float sum = 0.0f;
        for (int rank = 0; rank < kRanks; ++rank) {
            sum += host_inputs[rank][i];
        }
        (*expected)[i] = sum;
    }
}

int verify_outputs(DeviceBuffer<float>* outputs[kRanks],
                   const std::vector<float>& expected, float epsilon,
                   float* max_abs_error) {
    std::vector<float> actual(expected.size());
    int wrong = 0;
    *max_abs_error = 0.0f;
    bool debug_first_wrong =
        std::getenv("NANO_NCCL_DEBUG_FIRST_WRONG") != nullptr;
    std::size_t bytes = expected.size() * sizeof(float);
    for (int rank = 0; rank < kRanks; ++rank) {
        CUDA_CHECK_THROW(cudaSetDevice(rank));
        CUDA_CHECK_THROW(cudaMemcpy(actual.data(), outputs[rank]->get(), bytes,
                                    cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < expected.size(); ++i) {
            float err = std::fabs(actual[i] - expected[i]);
            *max_abs_error = std::max(*max_abs_error, err);
            if (err > epsilon) {
                ++wrong;
                if (debug_first_wrong) {
                    std::fprintf(stderr,
                                 "first wrong rank=%d index=%zu actual=%g "
                                 "expected=%g abs=%g\n",
                                 rank, i, actual[i], expected[i], err);
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

void enable_peer_access_or_throw() {
    for (int src = 0; src < kRanks; ++src) {
        CUDA_CHECK_THROW(cudaSetDevice(src));
        for (int dst = 0; dst < kRanks; ++dst) {
            if (src == dst) {
                continue;
            }
            int can_access = 0;
            CUDA_CHECK_THROW(cudaDeviceCanAccessPeer(&can_access, src, dst));
            if (!can_access) {
                continue;
            }
            cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();
            } else if (err != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }
}

class AllReduceRunner {
public:
    explicit AllReduceRunner(std::size_t max_count) : max_count_(max_count) {
        require_devices();
        enable_peer_access_or_throw();

        for (int rank = 0; rank < kRanks; ++rank) {
            inputs_[rank] = new DeviceBuffer<float>(rank, max_count_);
            outputs_[rank] = new DeviceBuffer<float>(rank, max_count_);
            streams_[rank] = new Stream(rank);
        }

        // step counter 跨迭代持久化，容量覆盖 2 * kChannels * kRanks（kind/channel/edge）。
        simple_fifo_steps_.reset(2 * kChannels * kRanks);
        simple_fifo_base_step_.reset(kRanks * kChannels);
        for (int channel = 0; channel < kChannels; ++channel) {
            std::size_t part_offset = 0;
            std::size_t part_count = 0;
            std::size_t chunk_count = 0;
            transport::shm::cbd_part(max_count_, channel, &part_offset,
                                     &part_count, &chunk_count);
            simple_fifo_slot_elems_ =
                std::max(simple_fifo_slot_elems_, chunk_count);
        }
        simple_fifo_slot_elems_ =
            std::max<std::size_t>(simple_fifo_slot_elems_, 1);
    }

    ~AllReduceRunner() {
        for (int channel = 0; channel < kChannels; ++channel) {
            for (int edge = 0; edge < kRanks; ++edge) {
                delete simple_fifo_[channel][edge];
            }
        }
        for (int rank = 0; rank < kRanks; ++rank) {
            delete streams_[rank];
            delete outputs_[rank];
            delete inputs_[rank];
        }
    }

    void load_inputs(std::vector<float> host_inputs[kRanks], std::size_t count) {
        std::size_t bytes = count * sizeof(float);
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
        simple_fifo_steps_.clear_host();
        simple_fifo_base_step_.clear_host();
        launch_ring_simple(count);
        sync_streams(streams_);
    }

    // 批量异步提交：用 CUDA events 做跨流屏障代替每迭代 cudaStreamSynchronize，
    // 对齐 NCCL BenchTime 计时口径。
    void run_batch(std::size_t count, int iters) {
        ensure_fifo_buffers();
        simple_fifo_steps_.clear_host();
        simple_fifo_base_step_.clear_host();

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
        return verify_outputs(outputs_, expected, epsilon, max_abs_error);
    }

private:
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
                if (simple_fifo_[channel][edge] == nullptr) {
                    int recv_gpu = ring_edge_recv_gpu(edge);
                    int numa_node = core::gpu_numa_node(recv_gpu);
                    simple_fifo_[channel][edge] = new MappedBuffer(
                        transport::shm::kSimpleFifoSteps *
                            simple_fifo_slot_elems_,
                        numa_node);
                }
            }
        }
    }

    void launch_ring_simple(std::size_t count) {
        for (int rank = 0; rank < kRanks; ++rank) {
            ShmFifoArgs<float> args{};
            args.rank = rank;
            args.count = count;
            args.slot_elems = simple_fifo_slot_elems_;
            args.step_elems = transport::shm::kSimpleFifoStepElems;
            args.input = inputs_[rank]->get();
            args.output = outputs_[rank]->get();
            args.steps = simple_fifo_steps_.device_ptr(rank);
            args.base_steps = simple_fifo_base_step_.device_ptr(rank) +
                              static_cast<std::size_t>(rank) * kChannels;

            int next = (rank + 1) % kRanks;
            int prev = (rank + kRanks - 1) % kRanks;
            int send_edge = transport::shm::ring_edge_index(rank, next, kRanks);
            int recv_edge = transport::shm::ring_edge_index(prev, rank, kRanks);
            if (send_edge < 0 || recv_edge < 0) {
                throw std::runtime_error(
                    "ring_simple saw an unexpected ring edge");
            }
            for (int channel = 0; channel < kChannels; ++channel) {
                args.send_fifo[channel] =
                    simple_fifo_[channel][send_edge]->device_ptr(rank);
                args.recv_fifo[channel] =
                    simple_fifo_[channel][recv_edge]->device_ptr(rank);
            }

            CUDA_CHECK_THROW(cudaSetDevice(rank));
            ring_simple_kernel<kRanks, DTypeTraits<DType::Float>::type,
                               RedOp::Sum>
                <<<kChannels, kBlockThreads, 0, streams_[rank]->get()>>>(args);
            CUDA_CHECK_THROW(cudaGetLastError());
        }
    }

    std::size_t max_count_ = 0;
    DeviceBuffer<float>* inputs_[kRanks]{};
    DeviceBuffer<float>* outputs_[kRanks]{};
    Stream* streams_[kRanks]{};
    MappedU64Array simple_fifo_steps_;
    MappedU64Array simple_fifo_base_step_;
    MappedBuffer* simple_fifo_[kChannels][kRanks]{};
    std::size_t simple_fifo_slot_elems_ = 0;
};

}  // namespace

int run_ring_simple_bench(const BenchConfig& config,
                          std::vector<BenchResult>* results) {
    try {
        auto sizes = make_sizes(config.min_bytes, config.max_bytes, config.factor);
        if (sizes.empty()) {
            std::fprintf(stderr, "invalid size range\n");
            return 2;
        }
        std::size_t max_count = sizes.back() / sizeof(float);
        AllReduceRunner runner(max_count);

        for (std::size_t bytes : sizes) {
            if (bytes % sizeof(float) != 0) {
                std::fprintf(stderr,
                             "size must be divisible by sizeof(float): %zu\n",
                             bytes);
                return 2;
            }
            std::size_t count = bytes / sizeof(float);
            std::vector<float> host_inputs[kRanks];
            std::vector<float> expected;
            fill_inputs(host_inputs, &expected, count);
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
            int wrong = runner.verify(expected, config.epsilon, &max_abs_error);

            BenchResult result;
            result.algo = algo;
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

}  // namespace nano_nccl::collective::all_reduce

// 显式实例化：NRanks/kRanks 由 CMake 配置，T=float、RedOp=Sum 为当前唯一特化。
// 必须在 kernels 命名空间内显式实例化。
namespace nano_nccl::kernels {
template __global__ void ring_simple_kernel<nano_nccl::kRanks, float, nano_nccl::RedOp::Sum>(
    nano_nccl::transport::shm::ShmFifoArgs<float>);
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
