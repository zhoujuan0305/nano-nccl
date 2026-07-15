#include "nano_nccl/communicator.h"

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

}  // namespace

int main() {
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
