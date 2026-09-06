#include "nano_nccl/communicator.h"
#include "nano_nccl/traits.h"

#include "transport/p2p/p2p_fifo.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

// 4096 stays on the <=4 KiB early-credit path for float, while 16385 crosses
// it for every dtype after four-channel partitioning. The larger case makes
// every dtype use both Simple slices in a chunk.
constexpr std::size_t kCounts[] = {1, 5, 4096, 16385};
constexpr std::size_t kSliceBoundaryCount = 1048577;
constexpr std::size_t kMaxCount = kSliceBoundaryCount;

void cuda_check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

void cuda_cleanup_check(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s failed during test cleanup: %s\n", operation,
                 cudaGetErrorString(status));
    std::terminate();
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer(int device, std::size_t count) : device_(device) {
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");
        cuda_check(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ == nullptr) return;
        cuda_cleanup_check(cudaSetDevice(device_), "cudaSetDevice");
        cuda_cleanup_check(cudaFree(data_), "cudaFree");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

private:
    int device_ = 0;
    T* data_ = nullptr;
};

class Stream {
public:
    explicit Stream(int device) : device_(device) {
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
    }

    ~Stream() {
        if (stream_ == nullptr) return;
        cuda_cleanup_check(cudaSetDevice(device_), "cudaSetDevice");
        cuda_cleanup_check(cudaStreamDestroy(stream_), "cudaStreamDestroy");
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    cudaStream_t get() const { return stream_; }

private:
    int device_ = 0;
    cudaStream_t stream_ = nullptr;
};

template <typename T>
class CollectiveBuffers {
public:
    CollectiveBuffers() {
        for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
            send_.push_back(std::make_unique<DeviceBuffer<T>>(
                rank, kMaxCount * nano_nccl::kRanks));
            recv_.push_back(std::make_unique<DeviceBuffer<T>>(
                rank, kMaxCount * nano_nccl::kRanks));
            streams_.push_back(std::make_unique<Stream>(rank));
        }
    }

    std::vector<const void*> send_buffers() const {
        std::vector<const void*> result;
        for (const auto& buffer : send_) result.push_back(buffer->get());
        return result;
    }

    std::vector<void*> recv_buffers() const {
        std::vector<void*> result;
        for (const auto& buffer : recv_) result.push_back(buffer->get());
        return result;
    }

    std::vector<cudaStream_t> streams() const {
        std::vector<cudaStream_t> result;
        for (const auto& stream : streams_) result.push_back(stream->get());
        return result;
    }

    T* send(int rank) const { return send_[rank]->get(); }
    T* recv(int rank) const { return recv_[rank]->get(); }
    cudaStream_t stream(int rank) const { return streams_[rank]->get(); }

    void synchronize() const {
        for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
            cuda_check(cudaSetDevice(rank), "cudaSetDevice");
            cuda_check(cudaStreamSynchronize(stream(rank)),
                       "cudaStreamSynchronize");
        }
    }

private:
    std::vector<std::unique_ptr<DeviceBuffer<T>>> send_;
    std::vector<std::unique_ptr<DeviceBuffer<T>>> recv_;
    std::vector<std::unique_ptr<Stream>> streams_;
};

float input_value(int source_rank, std::size_t index) {
    return static_cast<float>(source_rank + 1) +
           static_cast<float>(index % 4) * 0.25f;
}

float reduce_expected(nano_nccl::RedOp redop, std::size_t index) {
    float result = input_value(0, index);
    for (int rank = 1; rank < nano_nccl::kRanks; ++rank) {
        const float value = input_value(rank, index);
        switch (redop) {
            case nano_nccl::RedOp::Sum:
            case nano_nccl::RedOp::Avg:
                result += value;
                break;
            case nano_nccl::RedOp::Max:
                result = std::fmax(result, value);
                break;
            case nano_nccl::RedOp::Min:
                result = std::fmin(result, value);
                break;
        }
    }
    if (redop == nano_nccl::RedOp::Avg) {
        result /= static_cast<float>(nano_nccl::kRanks);
    }
    return result;
}

template <nano_nccl::DType kDType>
void run_all_gather(nano_nccl::Communicator* communicator) {
    using Traits = nano_nccl::DTypeTraits<kDType>;
    using T = typename Traits::type;
    CollectiveBuffers<T> buffers;
    auto run_case = [&](std::size_t send_count) {
        std::vector<T> input(send_count);
        std::vector<T> output(send_count * nano_nccl::kRanks);
        for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
            for (std::size_t index = 0; index < send_count; ++index) {
                input[index] = Traits::from_float(input_value(rank, index));
            }
            cuda_check(cudaSetDevice(rank), "cudaSetDevice");
            cuda_check(cudaMemcpyAsync(buffers.send(rank), input.data(),
                                       send_count * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       buffers.stream(rank)),
                       "cudaMemcpyAsync");
        }

        nano_nccl::AllGatherArgs args{
            buffers.send_buffers(), buffers.recv_buffers(), buffers.streams(),
            send_count, kDType,
        };
        communicator->all_gather(args);
        buffers.synchronize();

        for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
            cuda_check(cudaSetDevice(rank), "cudaSetDevice");
            cuda_check(cudaMemcpy(output.data(), buffers.recv(rank),
                                  output.size() * sizeof(T),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
            for (int source_rank = 0; source_rank < nano_nccl::kRanks;
                 ++source_rank) {
                for (std::size_t index = 0; index < send_count; ++index) {
                    const std::size_t output_index =
                        static_cast<std::size_t>(source_rank) * send_count +
                        index;
                    const float expected = Traits::to_float(
                        Traits::from_float(input_value(source_rank, index)));
                    const float actual = Traits::to_float(output[output_index]);
                    if (actual != expected) {
                        throw std::runtime_error(
                            "all_gather output mismatch at rank " +
                            std::to_string(rank));
                    }
                }
            }
        }
    };
    for (std::size_t send_count : kCounts) run_case(send_count);
    run_case(kSliceBoundaryCount);
}

template <nano_nccl::DType kDType>
void run_reduce_scatter(nano_nccl::Communicator* communicator) {
    using Traits = nano_nccl::DTypeTraits<kDType>;
    using T = typename Traits::type;
    CollectiveBuffers<T> buffers;
    const nano_nccl::RedOp redops[] = {
        nano_nccl::RedOp::Sum,
        nano_nccl::RedOp::Avg,
        nano_nccl::RedOp::Max,
        nano_nccl::RedOp::Min,
    };
    auto run_case = [&](nano_nccl::RedOp redop, std::size_t recv_count) {
        const std::size_t input_count = recv_count * nano_nccl::kRanks;
        std::vector<T> input(input_count);
        std::vector<T> output(recv_count);
        for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
            for (std::size_t index = 0; index < input_count; ++index) {
                input[index] = Traits::from_float(input_value(rank, index));
            }
            cuda_check(cudaSetDevice(rank), "cudaSetDevice");
            cuda_check(cudaMemcpyAsync(buffers.send(rank), input.data(),
                                       input_count * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       buffers.stream(rank)),
                       "cudaMemcpyAsync");
        }

        nano_nccl::ReduceScatterArgs args{
            buffers.send_buffers(), buffers.recv_buffers(), buffers.streams(),
            recv_count, kDType, redop,
        };
        communicator->reduce_scatter(args);
        buffers.synchronize();

        for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
            cuda_check(cudaSetDevice(rank), "cudaSetDevice");
            cuda_check(cudaMemcpy(output.data(), buffers.recv(rank),
                                  output.size() * sizeof(T),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
            for (std::size_t index = 0; index < recv_count; ++index) {
                const std::size_t global_index =
                    static_cast<std::size_t>(rank) * recv_count + index;
                const float expected = reduce_expected(redop, global_index);
                const float actual = Traits::to_float(output[index]);
                if (std::fabs(actual - expected) > Traits::kDefaultEpsilon) {
                    throw std::runtime_error(
                        "reduce_scatter output mismatch at rank " +
                        std::to_string(rank));
                }
            }
        }
    };
    for (nano_nccl::RedOp redop : redops) {
        for (std::size_t recv_count : kCounts) run_case(redop, recv_count);
    }
    run_case(nano_nccl::RedOp::Sum, kSliceBoundaryCount);
}

template <nano_nccl::DType kDType>
void run_reduce_scatter_nan(nano_nccl::Communicator* communicator) {
    using Traits = nano_nccl::DTypeTraits<kDType>;
    using T = typename Traits::type;
    constexpr std::size_t kRecvCount = 32;
    constexpr std::size_t kNanIndex = 3;
    const std::size_t input_count = kRecvCount * nano_nccl::kRanks;
    CollectiveBuffers<T> buffers;
    std::vector<T> input(input_count);
    std::vector<T> output(kRecvCount);

    for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
        for (std::size_t index = 0; index < input_count; ++index) {
            input[index] = Traits::from_float(input_value(rank, index));
        }
        if (rank == 1) {
            for (int output_rank = 0; output_rank < nano_nccl::kRanks;
                 ++output_rank) {
                input[static_cast<std::size_t>(output_rank) * kRecvCount +
                      kNanIndex] = Traits::from_float(NAN);
            }
        }
        cuda_check(cudaSetDevice(rank), "cudaSetDevice");
        cuda_check(cudaMemcpyAsync(buffers.send(rank), input.data(),
                                   input_count * sizeof(T),
                                   cudaMemcpyHostToDevice,
                                   buffers.stream(rank)),
                   "cudaMemcpyAsync");
    }

    for (nano_nccl::RedOp redop :
         {nano_nccl::RedOp::Sum, nano_nccl::RedOp::Avg,
          nano_nccl::RedOp::Max, nano_nccl::RedOp::Min}) {
        communicator->reduce_scatter({
            buffers.send_buffers(), buffers.recv_buffers(), buffers.streams(),
            kRecvCount, kDType, redop,
        });
        buffers.synchronize();
        for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
            cuda_check(cudaSetDevice(rank), "cudaSetDevice");
            cuda_check(cudaMemcpy(output.data(), buffers.recv(rank),
                                  output.size() * sizeof(T),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
            if (!std::isnan(Traits::to_float(output[kNanIndex]))) {
                throw std::runtime_error(
                    "reduce_scatter did not propagate NaN at rank " +
                    std::to_string(rank));
            }
        }
    }
}

bool bf16_supported() {
    for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, rank),
                   "cudaGetDeviceProperties");
        if (properties.major < 8) return false;
    }
    return true;
}

nano_nccl::TransportKind parse_requested_transport(const char* value) {
    nano_nccl::TransportKind transport;
    if (!nano_nccl::parse_transport(value, &transport)) {
        throw std::runtime_error("expected transport shm, auto, or p2p");
    }
    return transport;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: nano_nccl_collectives <transport>");
        }
        int visible_devices = 0;
        cudaError_t status = cudaGetDeviceCount(&visible_devices);
        if (status != cudaSuccess || visible_devices < nano_nccl::kRanks) {
            return 77;
        }
        const nano_nccl::TransportKind transport =
            parse_requested_transport(argv[1]);
        if (transport == nano_nccl::TransportKind::P2p &&
            !nano_nccl::transport::p2p::p2p_ring_available()) {
            return 77;
        }

        nano_nccl::CommunicatorConfig config;
        config.transport = transport;
        for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
            config.devices.push_back(rank);
        }
        auto communicator = nano_nccl::create_communicator(config);

        run_all_gather<nano_nccl::DType::Float>(communicator.get());
        run_all_gather<nano_nccl::DType::Float16>(communicator.get());
        run_reduce_scatter<nano_nccl::DType::Float>(communicator.get());
        run_reduce_scatter_nan<nano_nccl::DType::Float>(communicator.get());
        run_reduce_scatter<nano_nccl::DType::Float16>(communicator.get());
        run_reduce_scatter_nan<nano_nccl::DType::Float16>(communicator.get());
        if (bf16_supported()) {
            run_all_gather<nano_nccl::DType::BFloat16>(communicator.get());
            run_reduce_scatter<nano_nccl::DType::BFloat16>(communicator.get());
            run_reduce_scatter_nan<nano_nccl::DType::BFloat16>(
                communicator.get());
        }
        communicator->check_async_error();
        std::printf("collectives=PASS transport=%s\n",
                    nano_nccl::transport_name(communicator->transport()));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
