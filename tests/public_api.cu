#include "nano_nccl/communicator.h"

#include "collective/all_reduce/topology.h"
#include "core/buffer.h"
#include "kernels/ring_simple_kernel.cuh"
#include "transport/p2p/p2p_fifo.h"
#include "transport/p2p/p2p_step_counters.h"
#include "transport/p2p/p2p_topology.h"
#include "transport/simple_protocol.h"

#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err__));                \
            return 1;                                                         \
        }                                                                     \
    } while (0)

namespace {

constexpr int kRanks = 4;
constexpr std::size_t kCount = 1024;
constexpr std::size_t kGrowCount = kCount * 1024;

struct StreamGate {
    std::condition_variable condition;
    std::mutex mutex;
    bool released = false;

    ~StreamGate() { release(); }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [this] { return released; });
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        condition.notify_one();
    }
};

void CUDART_CB wait_for_stream_gate(void* user_data) {
    static_cast<StreamGate*>(user_data)->wait();
}

float input_value(int rank, std::size_t index, int round) {
    return static_cast<float>(round * (rank + 1)) +
           static_cast<float>(index) * 0.25f;
}

bool throws_with_message(void (nano_nccl::Communicator::*operation)(
                             const nano_nccl::CollectiveArgs&),
                         nano_nccl::Communicator* communicator,
                         const nano_nccl::CollectiveArgs& args,
                         const char* message) {
    try {
        (communicator->*operation)(args);
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(message) != std::string::npos;
    }
    return false;
}

bool all_reduce_throws_with_message(nano_nccl::Communicator* communicator,
                                    const nano_nccl::CollectiveArgs& args,
                                    const char* message) {
    try {
        communicator->all_reduce(args);
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(message) != std::string::npos;
    }
    return false;
}

bool create_communicator_throws_with_message(
    const nano_nccl::CommunicatorConfig& config, const char* message) {
    try {
        auto communicator = nano_nccl::create_communicator(config);
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(message) != std::string::npos;
    }
    return false;
}

bool synchronize_streams(const std::vector<cudaStream_t>& streams) {
    for (int rank = 0; rank < kRanks; ++rank) {
        if (cudaSetDevice(rank) != cudaSuccess ||
            cudaStreamSynchronize(streams[rank]) != cudaSuccess) {
            return false;
        }
    }
    return true;
}

bool verify_output(const std::vector<float>& output, int round) {
    for (std::size_t index = 0; index < kCount; ++index) {
        float expected = 0.0f;
        for (int input_rank = 0; input_rank < kRanks; ++input_rank) {
            expected += input_value(input_rank, index, round);
        }
        if (std::fabs(output[index] - expected) > 1e-6f) {
            std::fprintf(stderr, "wrong round=%d index=%zu actual=%g expected=%g\n",
                         round, index, output[index], expected);
            return false;
        }
    }
    return true;
}

std::vector<nano_nccl::collective::all_reduce::ProcessTopology>
make_synthetic_topologies(const std::vector<int>& local_rank_counts) {
    std::vector<nano_nccl::collective::all_reduce::ProcessTopology> topologies;
    int global_rank_count = 0;
    for (int count : local_rank_counts) global_rank_count += count;

    int offset = 0;
    for (int count : local_rank_counts) {
        nano_nccl::collective::all_reduce::ProcessTopology topology;
        topology.global_rank_count = global_rank_count;
        topology.local_rank_offset = offset;
        topology.devices.resize(count);
        for (int local_rank = 0; local_rank < count; ++local_rank) {
            topology.devices[local_rank] = local_rank;
        }
        topology.edge_kinds.assign(global_rank_count, nano_nccl::TransportKind::Shm);
        topology.distributed = true;
        topologies.push_back(std::move(topology));
        offset += count;
    }

    for (int edge = 0; edge < global_rank_count; ++edge) {
        int receiver = (edge + 1) % global_rank_count;
        int sender_process = 0;
        int receiver_process = 0;
        while (edge >= topologies[sender_process].local_rank_offset +
                           static_cast<int>(topologies[sender_process].devices.size())) {
            ++sender_process;
        }
        while (receiver >= topologies[receiver_process].local_rank_offset +
                               static_cast<int>(topologies[receiver_process].devices.size())) {
            ++receiver_process;
        }
        if (sender_process != receiver_process) {
            for (auto& topology : topologies) {
                topology.edge_kinds[edge] = nano_nccl::TransportKind::Socket;
            }
        }
    }
    return topologies;
}

bool run_topology_test() {
    using nano_nccl::TransportKind;
    using nano_nccl::collective::all_reduce::ProcessTopology;
    using nano_nccl::collective::all_reduce::validate_process_topology;
    using nano_nccl::transport::p2p::P2pFifo;
    using nano_nccl::transport::p2p::P2pStepCounters;
    using nano_nccl::transport::p2p::RingTransportPlan;

    ProcessTopology single_process{
        4,
        0,
        {0, 1, 2, 3},
        {TransportKind::P2p, TransportKind::Shm,
         TransportKind::P2p, TransportKind::Shm},
        false,
    };
    validate_process_topology(single_process);
    for (int local_rank = 0; local_rank < 4; ++local_rank) {
        if (single_process.local_rank_offset + local_rank != local_rank) {
            return false;
        }
    }
    if (single_process.edge_kinds.size() != 4 ||
        single_process.edge_kinds[0] != TransportKind::P2p ||
        single_process.edge_kinds[1] != TransportKind::Shm ||
        single_process.edge_kinds[2] != TransportKind::P2p ||
        single_process.edge_kinds[3] != TransportKind::Shm) {
        return false;
    }

    auto two_process = make_synthetic_topologies({2, 2});
    if (two_process.size() != 2 ||
        two_process[0].local_rank_offset != 0 ||
        two_process[1].local_rank_offset != 2 ||
        two_process[0].global_rank_count != 4 ||
        two_process[1].global_rank_count != 4 ||
        two_process[0].edge_kinds[1] != TransportKind::Socket ||
        two_process[0].edge_kinds[3] != TransportKind::Socket) {
        return false;
    }
    validate_process_topology(two_process[0]);
    validate_process_topology(two_process[1]);

    ProcessTopology second_process = two_process[1];
    auto resolved = nano_nccl::transport::p2p::resolve_ring_transport(
        TransportKind::Auto, second_process);
    if (resolved.edge_kind(1) != TransportKind::Socket ||
        resolved.edge_kind(3) != TransportKind::Socket ||
        resolved.edge_kind(2) == TransportKind::Socket) {
        return false;
    }

    RingTransportPlan local_p2p_plan({TransportKind::Socket, TransportKind::Socket,
                                      TransportKind::P2p, TransportKind::Socket});
    cudaStream_t local_streams[2]{};
    if (cudaSetDevice(0) != cudaSuccess ||
        cudaStreamCreateWithFlags(&local_streams[0], cudaStreamNonBlocking) != cudaSuccess ||
        cudaSetDevice(1) != cudaSuccess ||
        cudaStreamCreateWithFlags(&local_streams[1], cudaStreamNonBlocking) != cudaSuccess) {
        return false;
    }
    try {
        P2pFifo<float> fifo(1, local_p2p_plan, second_process);
        P2pStepCounters counters(local_p2p_plan, second_process);
        counters.reset({local_streams[0], local_streams[1]});
        if (cudaSetDevice(0) != cudaSuccess ||
            cudaStreamSynchronize(local_streams[0]) != cudaSuccess ||
            cudaSetDevice(1) != cudaSuccess ||
            cudaStreamSynchronize(local_streams[1]) != cudaSuccess ||
            fifo.edge_ptr(0, 2) == nullptr) {
            return false;
        }
    } catch (const std::exception&) {
        return false;
    }
    if (cudaSetDevice(0) != cudaSuccess ||
        cudaStreamDestroy(local_streams[0]) != cudaSuccess ||
        cudaSetDevice(1) != cudaSuccess ||
        cudaStreamDestroy(local_streams[1]) != cudaSuccess) {
        return false;
    }

    std::puts("topology=PASS");
    return true;
}

__global__ void abort_aware_send_wait(
    nano_nccl::transport::SimpleChannelArgs<float> args, int* result) {
    std::uint64_t cache = 0;
    __shared__ int wait_status;
    bool ready = nano_nccl::kernels::wait_send_credit<float>(
        args, 0, &cache, &wait_status);
    if (threadIdx.x == 0) result[0] = ready ? 0 : 1;
}

__global__ void abort_after_send_wait_starts(
    nano_nccl::transport::SimpleChannelArgs<float> args, int* result) {
    std::uint64_t cache = 0;
    __shared__ int wait_status;
    bool ready = nano_nccl::kernels::wait_send_credit<float>(
        args, 8, &cache, &wait_status);
    if (threadIdx.x == 0) result[0] = ready ? 0 : 1;
}

__global__ void abort_after_recv_wait_starts(
    nano_nccl::transport::SimpleChannelArgs<float> args, int* result) {
    std::uint64_t cache = 0;
    __shared__ int wait_status;
    bool ready = nano_nccl::kernels::wait_recv_ready<float>(
        args, 0, &cache, &wait_status);
    if (threadIdx.x == 0) result[0] = ready ? 0 : 1;
}

__global__ void publish_socket_slice(
    nano_nccl::transport::SimpleChannelArgs<float> args, const float* input) {
    std::uint64_t step = 0;
    std::uint64_t cache = 0;
    __shared__ int wait_status;
    nano_nccl::kernels::direct_send<float,
                                    nano_nccl::RedOp::Sum>(
        args, input, 4, &step, &cache, blockDim.x, &wait_status);
}

bool run_socket_abort_test() {
    using nano_nccl::core::MappedU32Array;
    using nano_nccl::core::MappedU64Array;
    using nano_nccl::transport::SimpleChannelArgs;

    constexpr int kDevice = 0;
    MappedU64Array counters;
    counters.reset(4, -1, {kDevice});
    MappedU32Array abort;
    abort.reset(1, -1, {kDevice});
    MappedU32Array started;
    started.reset(1, -1, {kDevice});
    int* result = nullptr;
    if (cudaSetDevice(kDevice) != cudaSuccess ||
        cudaMalloc(&result, sizeof(int)) != cudaSuccess) {
        return false;
    }
    SimpleChannelArgs<float> args{
        1, 1, nullptr, nullptr,
        counters.device_ptr(kDevice), counters.device_ptr(kDevice) + 1,
        counters.device_ptr(kDevice) + 2, counters.device_ptr(kDevice) + 3,
        nullptr, nullptr, abort.device_ptr(kDevice), started.device_ptr(kDevice),
    };
    abort_after_send_wait_starts<<<1, 32>>>(args, result);
    for (int spin = 0; spin < 1000000 &&
         __atomic_load_n(started.host_ptr(), __ATOMIC_ACQUIRE) == 0; ++spin) {}
    const bool send_observed =
        __atomic_load_n(started.host_ptr(), __ATOMIC_ACQUIRE) != 0;
    __atomic_store_n(abort.host_ptr(), 1U, __ATOMIC_RELEASE);
    int aborted = 0;
    bool ok = cudaGetLastError() == cudaSuccess &&
              cudaMemcpy(&aborted, result, sizeof(aborted), cudaMemcpyDeviceToHost) ==
                  cudaSuccess &&
              aborted == 1 && send_observed;
    if (ok) {
        __atomic_store_n(abort.host_ptr(), 0U, __ATOMIC_RELEASE);
        __atomic_store_n(started.host_ptr(), 0U, __ATOMIC_RELEASE);
        abort_after_recv_wait_starts<<<1, 32>>>(args, result);
        for (int spin = 0; spin < 1000000 &&
             __atomic_load_n(started.host_ptr(), __ATOMIC_ACQUIRE) == 0; ++spin) {}
        const bool recv_observed =
            __atomic_load_n(started.host_ptr(), __ATOMIC_ACQUIRE) != 0;
        __atomic_store_n(abort.host_ptr(), 1U, __ATOMIC_RELEASE);
        aborted = 0;
        ok = cudaGetLastError() == cudaSuccess &&
             cudaMemcpy(&aborted, result, sizeof(aborted), cudaMemcpyDeviceToHost) ==
                 cudaSuccess &&
             aborted == 1 && recv_observed;
    }
    cudaFree(result);
    std::puts(ok ? "socket_abort=PASS" : "socket_abort=FAIL");
    return ok;
}

bool run_socket_kernel_test() {
    using nano_nccl::core::MappedBuffer;
    using nano_nccl::core::MappedU32Array;
    using nano_nccl::core::MappedU64Array;
    using nano_nccl::transport::SimpleChannelArgs;

    constexpr int kDevice = 0;
    std::vector<int> devices{kDevice};
    MappedBuffer<float> fifo(nano_nccl::transport::kSimpleFifoSteps * 16, -1,
                             devices);
    MappedU64Array counters;
    counters.reset(4, -1, devices);
    MappedU32Array sizes;
    sizes.reset(nano_nccl::transport::kSimpleFifoSteps, -1, devices);
    MappedU32Array abort;
    abort.reset(1, -1, devices);

    float* input = nullptr;
    int* result = nullptr;
    if (cudaSetDevice(kDevice) != cudaSuccess ||
        cudaMalloc(&input, 4 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&result, sizeof(int)) != cudaSuccess) {
        return false;
    }
    bool ok = false;
    try {
        SimpleChannelArgs<float> args{
            16,
            16,
            fifo.device_ptr(kDevice),
            fifo.device_ptr(kDevice),
            counters.device_ptr(kDevice),
            counters.device_ptr(kDevice) + 1,
            counters.device_ptr(kDevice) + 2,
            counters.device_ptr(kDevice) + 3,
            sizes.device_ptr(kDevice),
            nullptr,
            abort.device_ptr(kDevice),
        };

        __atomic_store_n(abort.host_ptr(), 1U, __ATOMIC_RELEASE);
        abort_aware_send_wait<<<1, 32>>>(args, result);
        int aborted = 0;
        if (cudaGetLastError() == cudaSuccess &&
            cudaMemcpy(&aborted, result, sizeof(aborted), cudaMemcpyDeviceToHost) ==
                cudaSuccess &&
            aborted == 1) {
            __atomic_store_n(abort.host_ptr(), 0U, __ATOMIC_RELEASE);
            publish_socket_slice<<<1, 32>>>(args, input);
            std::uint64_t ready = 0;
            for (int spin = 0; spin < 1000000 && ready == 0; ++spin) {
                ready = __atomic_load_n(counters.host_ptr() + 2, __ATOMIC_ACQUIRE);
            }
            ok = cudaGetLastError() == cudaSuccess &&
                 ready >= nano_nccl::transport::kSimpleFifoSliceSteps &&
                 __atomic_load_n(sizes.host_ptr(), __ATOMIC_ACQUIRE) ==
                     4 * sizeof(float) &&
                 cudaDeviceSynchronize() == cudaSuccess;
            if (!ok) {
                std::fprintf(stderr, "socket_kernel detail tail=%llu size=%u\n",
                             static_cast<unsigned long long>(ready), sizes.host_ptr()[0]);
            }
        }
    } catch (const std::exception&) {
        ok = false;
    }
    cudaFree(input);
    cudaFree(result);
    std::puts(ok ? "socket_kernel=PASS" : "socket_kernel=FAIL");
    return ok;
}

bool run_socket_stride_test() {
    constexpr std::size_t kCount = 128ULL * 1024 * 1024 / sizeof(float);
    std::size_t part_offset = 0;
    std::size_t part_count = 0;
    std::size_t chunk_count = 0;
    nano_nccl::transport::shm::cbd_part<float>(kCount, 0, &part_offset,
                                               &part_count, &chunk_count);
    const std::size_t stride = nano_nccl::transport::shm::simple_fifo_step_elems<float>();
    const std::size_t socket_loop_chunk =
        nano_nccl::transport::shm::simple_fifo_loop_chunk_elems<float>(chunk_count, stride);
    const std::size_t local_loop_chunk =
        nano_nccl::transport::shm::simple_fifo_loop_chunk_elems<float>(chunk_count, 2 * stride);
    bool ok = chunk_count > stride && socket_loop_chunk == stride &&
              local_loop_chunk == 2 * stride &&
              nano_nccl::transport::kSimpleFifoSteps * stride * sizeof(float) ==
                  nano_nccl::transport::kSimpleFifoBuffBytes;
    std::puts(ok ? "socket_stride=PASS" : "socket_stride=FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--topology") {
        return run_topology_test() ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--socket-kernel") {
        return run_socket_kernel_test() ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--socket-abort") {
        return run_socket_abort_test() ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--socket-stride") {
        return run_socket_stride_test() ? 0 : 1;
    }
    std::vector<const void*> send_buffers(kRanks);
    std::vector<void*> recv_buffers(kRanks);
    std::vector<cudaStream_t> streams(kRanks);
    std::vector<std::vector<float>> inputs(kRanks, std::vector<float>(kCount));
    std::vector<float> output(kCount);

    for (int rank = 0; rank < kRanks; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        void* send_buffer = nullptr;
        CUDA_CHECK(cudaMalloc(&send_buffer, kGrowCount * sizeof(float)));
        send_buffers[rank] = send_buffer;
        CUDA_CHECK(cudaMalloc(&recv_buffers[rank], kGrowCount * sizeof(float)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking));
    }

    try {
        nano_nccl::CommunicatorConfig config;
        config.devices = {0, 1, 2, 3};
        auto invalid_length_config = config;
        invalid_length_config.devices.pop_back();
        auto invalid_sequence_config = config;
        invalid_sequence_config.devices = {0, 1, 3, 2};
        if (!create_communicator_throws_with_message(invalid_length_config,
                                                     "exactly") ||
            !create_communicator_throws_with_message(invalid_sequence_config,
                                                     "visible sequence")) {
            std::fprintf(stderr, "invalid communicator configuration lacked a diagnostic\n");
            return 1;
        }

        auto communicator = nano_nccl::create_communicator(config);
        nano_nccl::CollectiveArgs args{
            send_buffers, recv_buffers, streams, kCount,
            nano_nccl::DType::Float, nano_nccl::RedOp::Sum,
        };

        auto wrong_rank_args = args;
        wrong_rank_args.send_buffers.pop_back();
        auto null_buffer_args = args;
        null_buffer_args.recv_buffers[1] = nullptr;
        auto null_stream_args = args;
        null_stream_args.streams[2] = nullptr;
        auto zero_count_args = args;
        zero_count_args.count = 0;
        auto non_sum_args = args;
        non_sum_args.redop = static_cast<nano_nccl::RedOp>(1);
        auto in_place_args = args;
        in_place_args.recv_buffers[3] = const_cast<void*>(in_place_args.send_buffers[3]);
        auto invalid_dtype_args = args;
        invalid_dtype_args.dtype = static_cast<nano_nccl::DType>(99);
        if (!all_reduce_throws_with_message(communicator.get(), wrong_rank_args,
                                            "one entry per rank") ||
            !all_reduce_throws_with_message(communicator.get(), null_buffer_args,
                                            "non-null") ||
            !all_reduce_throws_with_message(communicator.get(), null_stream_args,
                                            "non-null") ||
            !all_reduce_throws_with_message(communicator.get(), zero_count_args,
                                            "positive") ||
            !all_reduce_throws_with_message(communicator.get(), non_sum_args,
                                            "sum") ||
            !all_reduce_throws_with_message(communicator.get(), in_place_args,
                                            "in-place") ||
            !all_reduce_throws_with_message(communicator.get(), invalid_dtype_args,
                                            "unsupported dtype")) {
            std::fprintf(stderr, "all_reduce validation did not report a diagnostic message\n");
            return 1;
        }

        if (!throws_with_message(&nano_nccl::Communicator::all_gather,
                                 communicator.get(), args, "all_gather") ||
            !throws_with_message(&nano_nccl::Communicator::reduce_scatter,
                                 communicator.get(), args, "reduce_scatter")) {
            std::fprintf(stderr, "unsupported collective did not report its name\n");
            return 1;
        }

        for (int round = 1; round <= 2; ++round) {
            for (int rank = 0; rank < kRanks; ++rank) {
                for (std::size_t index = 0; index < kCount; ++index) {
                    inputs[rank][index] = input_value(rank, index, round);
                }
                CUDA_CHECK(cudaSetDevice(rank));
                CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(send_buffers[rank]),
                                           inputs[rank].data(), kCount * sizeof(float),
                                           cudaMemcpyHostToDevice, streams[rank]));
            }
            communicator->all_reduce(args);
            if (!synchronize_streams(streams)) {
                std::fprintf(stderr, "normal all_reduce stream synchronization failed\n");
                return 1;
            }
            communicator->check_async_error();
            for (int rank = 0; rank < kRanks; ++rank) {
                CUDA_CHECK(cudaSetDevice(rank));
                CUDA_CHECK(cudaMemcpy(output.data(), recv_buffers[rank],
                                      kCount * sizeof(float), cudaMemcpyDeviceToHost));
                if (!verify_output(output, round)) return 1;
            }
        }

        StreamGate stream_gate;
        cudaEvent_t gate = nullptr;
        cudaStream_t gate_stream = nullptr;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreateWithFlags(&gate_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));
        CUDA_CHECK(cudaLaunchHostFunc(gate_stream, wait_for_stream_gate, &stream_gate));
        CUDA_CHECK(cudaEventRecord(gate, gate_stream));
        CUDA_CHECK(cudaStreamWaitEvent(streams[0], gate, 0));

        // The event cannot complete until its producer stream is released, so
        // FIFO growth must reject without relying on work duration.
        communicator->all_reduce(args);
        auto grow_args = args;
        grow_args.count = kGrowCount;
        bool grow_rejected = all_reduce_throws_with_message(
            communicator.get(), grow_args, "cannot grow communicator FIFO");
        stream_gate.release();
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamSynchronize(gate_stream));
        CUDA_CHECK(cudaEventDestroy(gate));
        CUDA_CHECK(cudaStreamDestroy(gate_stream));
        if (!grow_rejected) {
            std::fprintf(stderr, "in-flight FIFO growth did not report a diagnostic\n");
            return 1;
        }
        if (!synchronize_streams(streams)) {
            std::fprintf(stderr, "in-flight FIFO regression stream synchronization failed\n");
            return 1;
        }
        communicator->check_async_error();

        // Both launches share cross-rank dependencies, while callers wait only
        // after the second launch has been submitted.
        for (int round = 3; round <= 4; ++round) {
            for (int rank = 0; rank < kRanks; ++rank) {
                for (std::size_t index = 0; index < kCount; ++index) {
                    inputs[rank][index] = input_value(rank, index, round);
                }
                CUDA_CHECK(cudaSetDevice(rank));
                CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(send_buffers[rank]),
                                           inputs[rank].data(), kCount * sizeof(float),
                                           cudaMemcpyHostToDevice, streams[rank]));
            }
            communicator->all_reduce(args);
        }
        if (!synchronize_streams(streams)) {
            std::fprintf(stderr, "back-to-back all_reduce stream synchronization failed\n");
            return 1;
        }
        communicator->check_async_error();
        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaMemcpy(output.data(), recv_buffers[rank],
                                  kCount * sizeof(float), cudaMemcpyDeviceToHost));
            if (!verify_output(output, 4)) return 1;
        }

        auto move_source = nano_nccl::create_communicator(config);
        for (int rank = 0; rank < kRanks; ++rank) {
            for (std::size_t index = 0; index < kCount; ++index) {
                inputs[rank][index] = input_value(rank, index, 5);
            }
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(send_buffers[rank]),
                                       inputs[rank].data(), kCount * sizeof(float),
                                       cudaMemcpyHostToDevice, streams[rank]));
        }
        communicator->all_reduce(args);
        // Move assignment releases the old communicator while its launch is in flight.
        *communicator = std::move(*move_source);
        if (!synchronize_streams(streams)) {
            std::fprintf(stderr, "move-assignment stream synchronization failed\n");
            return 1;
        }
        communicator->check_async_error();
        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaMemcpy(output.data(), recv_buffers[rank],
                                  kCount * sizeof(float), cudaMemcpyDeviceToHost));
            if (!verify_output(output, 5)) return 1;
        }

        for (int rank = 0; rank < kRanks; ++rank) {
            for (std::size_t index = 0; index < kCount; ++index) {
                inputs[rank][index] = input_value(rank, index, 6);
            }
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(send_buffers[rank]),
                                       inputs[rank].data(), kCount * sizeof(float),
                                       cudaMemcpyHostToDevice, streams[rank]));
        }
        communicator->all_reduce(args);

        // Destruction must wait for the launch even though callers have not
        // synchronized their streams yet.
        communicator.reset();

        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
            CUDA_CHECK(cudaMemcpy(output.data(), recv_buffers[rank],
                                  kCount * sizeof(float), cudaMemcpyDeviceToHost));
            if (!verify_output(output, 6)) return 1;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }

    for (int rank = 0; rank < kRanks; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(cudaFree(const_cast<void*>(send_buffers[rank])));
        CUDA_CHECK(cudaFree(recv_buffers[rank]));
        CUDA_CHECK(cudaStreamDestroy(streams[rank]));
    }
    std::puts("public_api=PASS");
    return 0;
}
