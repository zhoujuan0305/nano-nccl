#include "nano_nccl/mpi.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>

#include <mpi.h>

namespace {

enum class Result { Pass, Skip, Fail };

Result run_fault(const char* point, nano_nccl::TransportKind transport) {
    setenv("NANO_NCCL_TEST_TRANSPORT_FAILURE", point, 1);
    bool injected = false;
    try {
        nano_nccl::CommunicatorConfig config;
        config.device = 0;
        config.transport = transport;
        auto communicator =
            nano_nccl::create_communicator_from_mpi(MPI_COMM_WORLD, config);
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (transport == nano_nccl::TransportKind::P2p &&
            message.find("failed the CUDA IPC probe") != std::string::npos) {
            unsetenv("NANO_NCCL_TEST_TRANSPORT_FAILURE");
            return Result::Skip;
        }
        injected = message.find("injected") != std::string::npos;
        if (!injected) {
            std::fprintf(stderr, "%s produced unexpected error: %s\n", point,
                         error.what());
        }
    }
    unsetenv("NANO_NCCL_TEST_TRANSPORT_FAILURE");
    if (!injected) return Result::Fail;

    try {
        nano_nccl::CommunicatorConfig config;
        config.device = 0;
        config.transport = transport;
        std::unique_ptr<nano_nccl::Communicator> communicator =
            nano_nccl::create_communicator_from_mpi(MPI_COMM_WORLD, config);
        communicator.reset();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s retry failed after cleanup: %s\n", point,
                     error.what());
        return Result::Fail;
    }
    return Result::Pass;
}

}  // namespace

int main(int argc, char** argv) {
    const char* local_rank = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (local_rank == nullptr) return EXIT_FAILURE;
    setenv("CUDA_VISIBLE_DEVICES", local_rank, 1);
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return EXIT_FAILURE;

    Result result = Result::Pass;
    for (const char* point : {"shm_allocation", "shm_registration"}) {
        if (run_fault(point, nano_nccl::TransportKind::Shm) != Result::Pass) {
            result = Result::Fail;
            break;
        }
    }
    if (result == Result::Pass) {
        for (const char* point : {"p2p_allocation", "p2p_open"}) {
            const Result current =
                run_fault(point, nano_nccl::TransportKind::P2p);
            if (current != Result::Pass) {
                result = current;
                break;
            }
        }
    }

    int local_result = static_cast<int>(result);
    int global_result = static_cast<int>(Result::Fail);
    if (MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) {
        global_result = static_cast<int>(Result::Fail);
    }
    if (MPI_Finalize() != MPI_SUCCESS) return EXIT_FAILURE;
    if (global_result == static_cast<int>(Result::Skip)) return 77;
    return global_result == static_cast<int>(Result::Pass)
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
