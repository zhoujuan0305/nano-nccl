#include "nano_nccl/mpi.h"
#include "nano_nccl/traits.h"

#include <algorithm>
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

namespace {

bool mpi_ok(int status, const char* operation) {
    if (status == MPI_SUCCESS) return true;
    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    std::fprintf(stderr, "%s failed: %.*s\n", operation, length, error);
    return false;
}

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool visible_devices(std::vector<int>* devices) {
    int count = 0;
    if (!cuda_ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount") || count <= 0) {
        return false;
    }
    devices->resize(count);
    for (int device = 0; device < count; ++device) (*devices)[device] = device;
    return true;
}

struct Options {
    nano_nccl::DType dtype = nano_nccl::DType::Float;
    bool fault_injection = false;
};

bool parse_options(int argc, char** argv, Options* options) {
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--dtype") == 0) {
            if (index + 1 == argc ||
                !nano_nccl::parse_dtype(argv[++index], &options->dtype)) {
                return false;
            }
        } else if (std::strcmp(argv[index], "--fault-injection") == 0) {
            options->fault_injection = true;
        } else {
            return false;
        }
    }
    return true;
}

template <nano_nccl::DType kDType>
bool run_collective(nano_nccl::Communicator* communicator,
                    const std::vector<int>& devices, int global_rank_offset,
                    bool fault_injection, int* wrong_count,
                    bool* stream_waits_completed, bool* async_socket_failure) {
    using Traits = nano_nccl::DTypeTraits<kDType>;
    using T = typename Traits::type;
    constexpr std::size_t kCount = 256 * 1024;

    std::vector<std::vector<T>> host_inputs(devices.size(), std::vector<T>(kCount));
    std::vector<T> host_output(kCount);
    std::vector<const void*> send_buffers(devices.size());
    std::vector<void*> recv_buffers(devices.size());
    std::vector<cudaStream_t> streams(devices.size());
    bool ok = true;

    float expected = 0.0f;
    for (int global_rank = 0; global_rank < nano_nccl::kRanks; ++global_rank) {
        expected += Traits::to_float(Traits::from_float(static_cast<float>(global_rank + 1)));
    }
    for (std::size_t local_rank = 0; local_rank < devices.size(); ++local_rank) {
        const float value = static_cast<float>(global_rank_offset +
                                               static_cast<int>(local_rank) + 1);
        std::fill(host_inputs[local_rank].begin(), host_inputs[local_rank].end(),
                  Traits::from_float(value));
        ok = cuda_ok(cudaSetDevice(devices[local_rank]), "cudaSetDevice") && ok;
        ok = cuda_ok(cudaMalloc(const_cast<void**>(&send_buffers[local_rank]),
                                kCount * sizeof(T)), "cudaMalloc(send)") && ok;
        ok = cuda_ok(cudaMalloc(&recv_buffers[local_rank], kCount * sizeof(T)),
                     "cudaMalloc(recv)") && ok;
        ok = cuda_ok(cudaStreamCreateWithFlags(&streams[local_rank], cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags") && ok;
        if (ok) {
            ok = cuda_ok(cudaMemcpyAsync(const_cast<void*>(send_buffers[local_rank]),
                                         host_inputs[local_rank].data(),
                                         kCount * sizeof(T), cudaMemcpyHostToDevice,
                                         streams[local_rank]), "cudaMemcpyAsync") && ok;
        }
    }

    try {
        if (ok) {
            try {
                communicator->all_reduce({send_buffers, recv_buffers, streams, kCount,
                                          kDType, nano_nccl::RedOp::Sum});
            } catch (const std::exception& error) {
                if (!fault_injection || std::strstr(error.what(), "socket") == nullptr) {
                    throw;
                }
                *async_socket_failure = true;
            }
            for (std::size_t local_rank = 0; local_rank < devices.size(); ++local_rank) {
                ok = cuda_ok(cudaSetDevice(devices[local_rank]), "cudaSetDevice") && ok;
                ok = cuda_ok(cudaStreamSynchronize(streams[local_rank]),
                             "cudaStreamSynchronize") && ok;
            }
            *stream_waits_completed = ok;
            if (fault_injection) {
                try {
                    communicator->check_async_error();
                } catch (const std::exception& error) {
                    *async_socket_failure =
                        *async_socket_failure || std::strstr(error.what(), "socket") != nullptr;
                }
                ok = *stream_waits_completed && *async_socket_failure;
            } else {
                communicator->check_async_error();
                const float epsilon = Traits::kDefaultEpsilon;
                for (std::size_t local_rank = 0; local_rank < devices.size(); ++local_rank) {
                    ok = cuda_ok(cudaSetDevice(devices[local_rank]), "cudaSetDevice") && ok;
                    ok = cuda_ok(cudaMemcpy(host_output.data(), recv_buffers[local_rank],
                                             kCount * sizeof(T), cudaMemcpyDeviceToHost),
                                 "cudaMemcpy") && ok;
                    for (T value : host_output) {
                        if (std::fabs(Traits::to_float(value) - expected) > epsilon) {
                            ++*wrong_count;
                            break;
                        }
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mpi correctness collective failed: %s\n", error.what());
        ok = false;
    }

    for (std::size_t local_rank = 0; local_rank < devices.size(); ++local_rank) {
        cudaSetDevice(devices[local_rank]);
        if (send_buffers[local_rank] != nullptr) cudaFree(const_cast<void*>(send_buffers[local_rank]));
        if (recv_buffers[local_rank] != nullptr) cudaFree(recv_buffers[local_rank]);
        if (streams[local_rank] != nullptr) cudaStreamDestroy(streams[local_rank]);
    }
    return ok;
}

bool dispatch_collective(nano_nccl::DType dtype, nano_nccl::Communicator* communicator,
                         const std::vector<int>& devices, int global_rank_offset,
                         bool fault_injection, int* wrong_count,
                         bool* stream_waits_completed, bool* async_socket_failure) {
    switch (dtype) {
        case nano_nccl::DType::Float:
            return run_collective<nano_nccl::DType::Float>(
                communicator, devices, global_rank_offset, fault_injection, wrong_count,
                stream_waits_completed, async_socket_failure);
        case nano_nccl::DType::Float16:
            return run_collective<nano_nccl::DType::Float16>(
                communicator, devices, global_rank_offset, fault_injection, wrong_count,
                stream_waits_completed, async_socket_failure);
        case nano_nccl::DType::BFloat16:
            return run_collective<nano_nccl::DType::BFloat16>(
                communicator, devices, global_rank_offset, fault_injection, wrong_count,
                stream_waits_completed, async_socket_failure);
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        std::fprintf(stderr, "Usage: %s [--dtype float|fp16|bf16] [--fault-injection]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (!mpi_ok(MPI_Init(&argc, &argv), "MPI_Init")) return EXIT_FAILURE;

    int mpi_rank = 0;
    int local_rank_offset = 0;
    int local_device_count = 0;
    int global_device_count = 0;
    bool ok = mpi_ok(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank), "MPI_Comm_rank");
    std::vector<int> devices;
    ok = visible_devices(&devices) && ok;
    local_device_count = static_cast<int>(devices.size());
    ok = mpi_ok(MPI_Exscan(&local_device_count, &local_rank_offset, 1, MPI_INT, MPI_SUM,
                           MPI_COMM_WORLD), "MPI_Exscan") && ok;
    if (mpi_rank == 0) local_rank_offset = 0;
    ok = mpi_ok(MPI_Allreduce(&local_device_count, &global_device_count, 1, MPI_INT, MPI_SUM,
                               MPI_COMM_WORLD), "MPI_Allreduce(device count)") && ok;
    ok = global_device_count == nano_nccl::kRanks && ok;

    int local_wrong = 0;
    bool stream_waits_completed = false;
    bool async_socket_failure = false;
    std::unique_ptr<nano_nccl::Communicator> communicator;
    try {
        if (ok) {
            nano_nccl::CommunicatorConfig config;
            config.devices = devices;
            config.transport = nano_nccl::TransportKind::Auto;
            communicator = nano_nccl::create_communicator_from_mpi(MPI_COMM_WORLD, config);
            ok = communicator->transport() == nano_nccl::TransportKind::Socket ||
                 communicator->transport() == nano_nccl::TransportKind::Mixed;
            ok = dispatch_collective(options.dtype, communicator.get(), devices,
                                     local_rank_offset, options.fault_injection,
                                     &local_wrong, &stream_waits_completed,
                                     &async_socket_failure) && ok;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mpi correctness setup failed: %s\n", error.what());
        ok = false;
    }

    int global_wrong = 0;
    int local_ok = ok ? 1 : 0;
    int global_ok = 0;
    mpi_ok(MPI_Allreduce(&local_wrong, &global_wrong, 1, MPI_INT, MPI_SUM,
                         MPI_COMM_WORLD), "MPI_Allreduce(wrong)");
    mpi_ok(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                         MPI_COMM_WORLD), "MPI_Allreduce(status)");

    if (!options.fault_injection) {
        mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(before communicator destroy)");
        communicator.reset();
    }

    if (mpi_rank == 0 && global_ok != 0 && global_wrong == 0) {
        if (options.fault_injection) {
            std::puts("mpi_fault_injection=PASS");
        } else {
            std::puts("mpi_correctness=PASS");
        }
    }
    if (options.fault_injection) {
        MPI_Abort(MPI_COMM_WORLD, global_ok != 0 && global_wrong == 0 ? 0 : 1);
        return EXIT_FAILURE;
    }
    if (global_ok == 0 || global_wrong != 0) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    return mpi_ok(MPI_Finalize(), "MPI_Finalize") ? EXIT_SUCCESS : EXIT_FAILURE;
}
