#include "nano_nccl/mpi.h"

#include "collective/all_reduce/communicator_internal.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

bool check_mpi(int status, const char* operation) {
    if (status == MPI_SUCCESS) return true;
    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    std::fprintf(stderr, "%s failed: %.*s\n", operation, length, error);
    return false;
}

bool visible_devices(std::vector<int>* devices) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        std::fprintf(stderr, "no visible CUDA devices\n");
        return false;
    }
    devices->resize(count);
    for (int device = 0; device < count; ++device) {
        (*devices)[device] = device;
    }
    return true;
}

bool factory_closes_socket_fds_when_construction_throws() {
    int sockets[2]{};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
        std::perror("socketpair");
        return false;
    }

    bool threw = false;
    try {
        nano_nccl::CommunicatorConfig config;
        config.devices = {0};
        nano_nccl::collective::all_reduce::ProcessTopology topology{
            nano_nccl::kRanks,
            0,
            {1},
            std::vector<nano_nccl::TransportKind>(
                nano_nccl::kRanks, nano_nccl::TransportKind::Shm),
            true,
        };
        nano_nccl::collective::all_reduce::CommunicatorFactory::create(
            config, std::move(topology),
            nano_nccl::collective::all_reduce::SocketFdOwner({sockets[0]}));
    } catch (const std::runtime_error&) {
        threw = true;
    }

    errno = 0;
    bool closed = fcntl(sockets[0], F_GETFD) == -1 && errno == EBADF;
    if (!closed) close(sockets[0]);
    close(sockets[1]);
    return threw && closed;
}

bool fd_is_closed(int fd) {
    errno = 0;
    return fcntl(fd, F_GETFD) == -1 && errno == EBADF;
}

void close_if_open(int fd) {
    if (!fd_is_closed(fd)) close(fd);
}

bool socket_fd_owner_moves_without_leaks() {
    int constructed[2]{};
    int replaced[2]{};
    int replacement[2]{};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, constructed) != 0 ||
        socketpair(AF_UNIX, SOCK_STREAM, 0, replaced) != 0 ||
        socketpair(AF_UNIX, SOCK_STREAM, 0, replacement) != 0) {
        std::perror("socketpair");
        return false;
    }

    bool move_constructed_stays_open = false;
    {
        std::optional<nano_nccl::collective::all_reduce::SocketFdOwner> moved;
        {
            nano_nccl::collective::all_reduce::SocketFdOwner source(
                {constructed[0]});
            moved.emplace(std::move(source));
        }
        move_constructed_stays_open = !fd_is_closed(constructed[0]);
    }
    bool move_constructed_closed = fd_is_closed(constructed[0]);

    bool replacement_closed = false;
    bool moved_assignment_stays_open = false;
    {
        nano_nccl::collective::all_reduce::SocketFdOwner destination(
            {replaced[0]});
        {
            nano_nccl::collective::all_reduce::SocketFdOwner source(
                {replacement[0]});
            destination = std::move(source);
            replacement_closed = fd_is_closed(replaced[0]);
        }
        moved_assignment_stays_open = !fd_is_closed(replacement[0]);
    }
    bool moved_assignment_closed = fd_is_closed(replacement[0]);

    close_if_open(constructed[0]);
    close_if_open(replaced[0]);
    close_if_open(replacement[0]);
    close(constructed[1]);
    close(replaced[1]);
    close(replacement[1]);
    return move_constructed_stays_open && move_constructed_closed &&
           replacement_closed && moved_assignment_stays_open &&
           moved_assignment_closed;
}

bool run_socket_all_reduce_smoke(nano_nccl::Communicator* communicator,
                                 const std::vector<int>& devices,
                                 int local_rank_offset) {
    constexpr std::size_t kCount = 128ULL * 1024 * 1024 / sizeof(float);
    std::vector<const void*> send_buffers(devices.size());
    std::vector<void*> recv_buffers(devices.size());
    std::vector<cudaStream_t> streams(devices.size());
    std::vector<float> input(kCount);
    std::vector<float> output(kCount);
    bool ok = true;
    for (std::size_t local_rank = 0; local_rank < devices.size(); ++local_rank) {
        const int device = devices[local_rank];
        const float value = static_cast<float>(local_rank_offset +
                                               static_cast<int>(local_rank) + 1);
        std::fill(input.begin(), input.end(), value);
        void* send = nullptr;
        ok = ok && cudaSetDevice(device) == cudaSuccess &&
             cudaMalloc(&send, kCount * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&recv_buffers[local_rank], kCount * sizeof(float)) == cudaSuccess &&
             cudaStreamCreateWithFlags(&streams[local_rank], cudaStreamNonBlocking) ==
                 cudaSuccess;
        send_buffers[local_rank] = send;
        if (ok) {
            ok = cudaMemcpyAsync(send, input.data(), kCount * sizeof(float),
                                 cudaMemcpyHostToDevice, streams[local_rank]) == cudaSuccess;
        }
    }
    try {
        if (ok) {
            communicator->all_reduce({send_buffers, recv_buffers, streams, kCount,
                                      nano_nccl::DType::Float, nano_nccl::RedOp::Sum});
            for (std::size_t local_rank = 0; local_rank < devices.size(); ++local_rank) {
                ok = ok && cudaSetDevice(devices[local_rank]) == cudaSuccess &&
                     cudaStreamSynchronize(streams[local_rank]) == cudaSuccess &&
                     cudaMemcpy(output.data(), recv_buffers[local_rank],
                                kCount * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
                for (float value : output) {
                    const float expected =
                        static_cast<float>(nano_nccl::kRanks * (nano_nccl::kRanks + 1) / 2);
                    if (std::fabs(value - expected) > 1e-6f) ok = false;
                }
            }
            communicator->check_async_error();
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "socket all_reduce smoke failed: %s\n", error.what());
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

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--factory-fd-ownership") {
        return factory_closes_socket_fds_when_construction_throws() &&
                   socket_fd_owner_moves_without_leaks()
            ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (!check_mpi(MPI_Init(&argc, &argv), "MPI_Init")) return EXIT_FAILURE;

    int mpi_rank = 0;
    const bool run_socket_smoke =
        argc == 2 && std::string(argv[1]) == "--socket-all-reduce-smoke";
    const bool run_invalid_interface_consensus =
        argc == 2 && std::string(argv[1]) == "--invalid-interface-consensus";
    bool ok = check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank), "MPI_Comm_rank");
    if (run_invalid_interface_consensus && mpi_rank == 0) {
        setenv("NANO_NCCL_SOCKET_IFNAME", "nano_nccl_invalid_interface", 1);
    }
    std::vector<int> devices;
    ok = ok && visible_devices(&devices);

    if (ok) {
        try {
            nano_nccl::CommunicatorConfig config;
            config.devices = devices;
            std::unique_ptr<nano_nccl::Communicator> communicator;
            if (run_invalid_interface_consensus) {
                try {
                    communicator = nano_nccl::create_communicator_from_mpi(
                        MPI_COMM_WORLD, config);
                    std::fprintf(stderr, "invalid interface was accepted\n");
                    ok = false;
                } catch (const std::exception& error) {
                    ok = std::string(error.what()).find("socket listener setup failed on MPI rank 0") !=
                         std::string::npos;
                }
            } else {
                communicator = nano_nccl::create_communicator_from_mpi(
                    MPI_COMM_WORLD, config);
            }
            if (!run_invalid_interface_consensus) {
                ok = communicator->local_rank_count() == static_cast<int>(devices.size()) &&
                     communicator->global_rank_count() == nano_nccl::kRanks;
                if (ok && run_socket_smoke) {
                    ok = run_socket_all_reduce_smoke(
                        communicator.get(), devices,
                        mpi_rank * static_cast<int>(devices.size()));
                    ok = check_mpi(MPI_Barrier(MPI_COMM_WORLD),
                                   "MPI_Barrier(socket all_reduce smoke)") && ok;
                }
                if (!ok) {
                    std::fprintf(stderr, "communicator rank counts are incorrect\n");
                }
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "bootstrap failed: %s\n", error.what());
            ok = false;
        }
    }

    int local_ok = ok ? 1 : 0;
    int global_ok = 0;
    if (!check_mpi(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                                 MPI_COMM_WORLD), "MPI_Allreduce")) {
        ok = false;
    } else {
        ok = global_ok != 0;
    }
    if (mpi_rank == 0 && ok) {
        std::printf("global_rank_count=%d\n", nano_nccl::kRanks);
        if (run_socket_smoke) std::puts("socket_all_reduce=PASS");
    }

    if (!check_mpi(MPI_Finalize(), "MPI_Finalize")) return EXIT_FAILURE;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
