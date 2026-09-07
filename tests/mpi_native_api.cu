#include "nano_nccl/mpi.h"
#include "nano_nccl/traits.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

namespace {

constexpr std::size_t kCount = 16385;

void cuda_check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

float input_value(int rank, std::size_t index) {
    return static_cast<float>(rank + 1) +
           static_cast<float>(index % 4) * 0.25f;
}

float reduced_value(nano_nccl::RedOp redop, std::size_t index) {
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
    return redop == nano_nccl::RedOp::Avg
        ? result / static_cast<float>(nano_nccl::kRanks)
        : result;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        cuda_check(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ == nullptr) return;
        const cudaError_t status = cudaFree(data_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "cudaFree during test cleanup: %s\n",
                         cudaGetErrorString(status));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const noexcept { return data_; }

private:
    T* data_ = nullptr;
};

void expect_invalid_descriptors(nano_nccl::Communicator* communicator,
                                cudaStream_t stream) {
    DeviceBuffer<float> buffer(64);
    auto expect_rejected = [&](const nano_nccl::AllReduceArgs& args,
                               const char* expected) {
        try {
            communicator->all_reduce(args);
        } catch (const std::exception& error) {
            if (std::string(error.what()).find(expected) != std::string::npos) {
                return;
            }
            throw;
        }
        throw std::runtime_error("invalid descriptor was accepted");
    };
    expect_rejected(
        {buffer.get(), buffer.get(), stream, 32, nano_nccl::DType::Float,
         nano_nccl::RedOp::Sum},
        "overlapping all_reduce");
    expect_rejected(
        {buffer.get(), buffer.get() + 1, stream, 32,
         nano_nccl::DType::Float, nano_nccl::RedOp::Sum},
        "overlapping all_reduce");
    expect_rejected(
        {buffer.get(), buffer.get() + 32, nullptr, 32,
         nano_nccl::DType::Float, nano_nccl::RedOp::Sum},
        "must be non-null");
    expect_rejected(
        {buffer.get(), buffer.get() + 32, stream, 0,
         nano_nccl::DType::Float, nano_nccl::RedOp::Sum},
        "count must be positive");
}

template <nano_nccl::DType kDType>
void run_collective_matrix(nano_nccl::Communicator* communicator, int rank,
                           cudaStream_t stream) {
    using Traits = nano_nccl::DTypeTraits<kDType>;
    using T = typename Traits::type;
    const std::size_t large_count = kCount * nano_nccl::kRanks;
    DeviceBuffer<T> send(large_count);
    DeviceBuffer<T> recv(large_count);
    std::vector<T> input(large_count);
    std::vector<T> output(large_count);

    for (std::size_t index = 0; index < large_count; ++index) {
        input[index] = Traits::from_float(input_value(rank, index));
    }
    cuda_check(cudaMemcpyAsync(send.get(), input.data(), large_count * sizeof(T),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(reduce input)");

    for (nano_nccl::RedOp redop :
         {nano_nccl::RedOp::Sum, nano_nccl::RedOp::Avg,
          nano_nccl::RedOp::Max, nano_nccl::RedOp::Min}) {
        communicator->all_reduce(
            {send.get(), recv.get(), stream, kCount, kDType, redop});
        cuda_check(cudaMemcpyAsync(output.data(), recv.get(), kCount * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync(all_reduce output)");
        cuda_check(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize(all_reduce)");
        for (std::size_t index = 0; index < kCount; ++index) {
            const float actual = Traits::to_float(output[index]);
            const float expected = Traits::to_float(
                Traits::from_float(reduced_value(redop, index)));
            if (std::fabs(actual - expected) > Traits::kDefaultEpsilon) {
                throw std::runtime_error("all_reduce output mismatch");
            }
        }

        communicator->reduce_scatter(
            {send.get(), recv.get(), stream, kCount, kDType, redop});
        cuda_check(cudaMemcpyAsync(output.data(), recv.get(), kCount * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync(reduce_scatter output)");
        cuda_check(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize(reduce_scatter)");
        for (std::size_t index = 0; index < kCount; ++index) {
            const std::size_t global_index =
                static_cast<std::size_t>(rank) * kCount + index;
            const float actual = Traits::to_float(output[index]);
            const float expected = Traits::to_float(
                Traits::from_float(reduced_value(redop, global_index)));
            if (std::fabs(actual - expected) > Traits::kDefaultEpsilon) {
                throw std::runtime_error("reduce_scatter output mismatch");
            }
        }
    }

    communicator->all_gather(
        {send.get(), recv.get(), stream, kCount, kDType});
    cuda_check(cudaMemcpyAsync(output.data(), recv.get(),
                               large_count * sizeof(T),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(all_gather output)");
    cuda_check(cudaStreamSynchronize(stream),
               "cudaStreamSynchronize(all_gather)");
    for (int source_rank = 0; source_rank < nano_nccl::kRanks; ++source_rank) {
        for (std::size_t index = 0; index < kCount; ++index) {
            const std::size_t output_index =
                static_cast<std::size_t>(source_rank) * kCount + index;
            const float actual = Traits::to_float(output[output_index]);
            const float expected = Traits::to_float(
                Traits::from_float(input_value(source_rank, index)));
            if (actual != expected) {
                throw std::runtime_error("all_gather output mismatch");
            }
        }
    }
}

template <nano_nccl::DType kDType>
void run_nan_matrix(nano_nccl::Communicator* communicator, int rank,
                    cudaStream_t stream) {
    using Traits = nano_nccl::DTypeTraits<kDType>;
    using T = typename Traits::type;
    constexpr std::size_t kNanCount = 32;
    constexpr std::size_t kNanIndex = 3;
    const std::size_t large_count = kNanCount * nano_nccl::kRanks;
    DeviceBuffer<T> send(large_count);
    DeviceBuffer<T> recv(large_count);
    std::vector<T> input(large_count, Traits::from_float(rank + 1.0f));
    std::vector<T> output(kNanCount);
    if (rank == 1) {
        for (int output_rank = 0; output_rank < nano_nccl::kRanks;
             ++output_rank) {
            input[static_cast<std::size_t>(output_rank) * kNanCount +
                  kNanIndex] = Traits::from_float(NAN);
        }
    }
    cuda_check(cudaMemcpyAsync(send.get(), input.data(), large_count * sizeof(T),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(NaN input)");

    for (nano_nccl::RedOp redop :
         {nano_nccl::RedOp::Sum, nano_nccl::RedOp::Avg,
          nano_nccl::RedOp::Max, nano_nccl::RedOp::Min}) {
        communicator->all_reduce(
            {send.get(), recv.get(), stream, kNanCount, kDType, redop});
        cuda_check(cudaMemcpyAsync(output.data(), recv.get(),
                                   kNanCount * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync(all_reduce NaN output)");
        cuda_check(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize(all_reduce NaN)");
        if (!std::isnan(Traits::to_float(output[kNanIndex]))) {
            throw std::runtime_error("all_reduce did not propagate NaN");
        }

        communicator->reduce_scatter(
            {send.get(), recv.get(), stream, kNanCount, kDType, redop});
        cuda_check(cudaMemcpyAsync(output.data(), recv.get(),
                                   kNanCount * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync(reduce_scatter NaN output)");
        cuda_check(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize(reduce_scatter NaN)");
        if (!std::isnan(Traits::to_float(output[kNanIndex]))) {
            throw std::runtime_error("reduce_scatter did not propagate NaN");
        }
    }
}

nano_nccl::TransportKind parse_transport(int argc, char** argv) {
    if (argc != 2) {
        throw std::invalid_argument(
            "usage: nano_nccl_mpi_native_api <auto|shm|p2p|socket|rdma>");
    }
    if (std::string(argv[1]) == "auto") return nano_nccl::TransportKind::Auto;
    if (std::string(argv[1]) == "shm") return nano_nccl::TransportKind::Shm;
    if (std::string(argv[1]) == "p2p") return nano_nccl::TransportKind::P2p;
    if (std::string(argv[1]) == "socket") {
        return nano_nccl::TransportKind::Socket;
    }
    if (std::string(argv[1]) == "rdma") return nano_nccl::TransportKind::Rdma;
    throw std::invalid_argument("unknown transport");
}

}  // namespace

int main(int argc, char** argv) {
    const char* local_rank_text = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (local_rank_text == nullptr) {
        std::fprintf(stderr, "run this test under Open MPI\n");
        return EXIT_FAILURE;
    }
    const bool use_visible_local_rank =
        std::getenv("NANO_NCCL_TEST_VISIBLE_LOCAL_RANK") != nullptr;
    if (!use_visible_local_rank) setenv("CUDA_VISIBLE_DEVICES", local_rank_text, 1);
    MPI_Init(&argc, &argv);

    int exit_code = EXIT_SUCCESS;
    try {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        nano_nccl::CommunicatorConfig config;
        config.device = use_visible_local_rank ? std::stoi(local_rank_text) : 0;
        config.transport = parse_transport(argc, argv);
        std::unique_ptr<nano_nccl::Communicator> communicator =
            nano_nccl::create_communicator_from_mpi(MPI_COMM_WORLD, config);
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
        expect_invalid_descriptors(communicator.get(), stream);
        run_collective_matrix<nano_nccl::DType::Float>(communicator.get(), rank,
                                                       stream);
        run_nan_matrix<nano_nccl::DType::Float>(communicator.get(), rank, stream);
        run_collective_matrix<nano_nccl::DType::Float16>(communicator.get(), rank,
                                                         stream);
        run_nan_matrix<nano_nccl::DType::Float16>(communicator.get(), rank, stream);
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, config.device),
                   "cudaGetDeviceProperties");
        if (properties.major >= 8) {
            run_collective_matrix<nano_nccl::DType::BFloat16>(
                communicator.get(), rank, stream);
            run_nan_matrix<nano_nccl::DType::BFloat16>(communicator.get(), rank,
                                                       stream);
        }
        communicator->check_async_error();
        if (communicator->local_rank_count() != 1 ||
            communicator->global_rank_count() != nano_nccl::kRanks ||
            (config.transport != nano_nccl::TransportKind::Auto &&
             communicator->transport() != config.transport) ||
            (config.transport == nano_nccl::TransportKind::Auto &&
             communicator->transport() == nano_nccl::TransportKind::Auto)) {
            throw std::runtime_error("communicator metadata mismatch");
        }
        for (int edge = 0; edge < nano_nccl::kRanks; ++edge) {
            const nano_nccl::TransportKind edge_transport =
                communicator->edge_transport(edge);
            if (edge_transport == nano_nccl::TransportKind::Auto ||
                edge_transport == nano_nccl::TransportKind::Mixed) {
                throw std::runtime_error("Ring edge transport was not resolved");
            }
        }
        try {
            communicator->edge_transport(-1);
            throw std::runtime_error("invalid Ring edge was accepted");
        } catch (const std::invalid_argument&) {
        }
        // Exercise non-collective SHM/CUDA IPC teardown with deliberately
        // asymmetric process timing.
        if (rank == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        communicator.reset();
        cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        exit_code = std::string(error.what()).find("failed the CUDA IPC probe") !=
                std::string::npos
            ? 77
            : EXIT_FAILURE;
    }

    int global_exit_code = EXIT_FAILURE;
    if (MPI_Allreduce(&exit_code, &global_exit_code, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) {
        global_exit_code = EXIT_FAILURE;
    }
    if (MPI_Finalize() != MPI_SUCCESS) return EXIT_FAILURE;
    return global_exit_code;
}
