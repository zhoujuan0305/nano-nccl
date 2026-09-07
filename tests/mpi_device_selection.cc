#include "nano_nccl/mpi.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    bool rejected = false;
    try {
        nano_nccl::CommunicatorConfig config;
        config.device = 0;
        config.transport = nano_nccl::TransportKind::Shm;
        auto communicator =
            nano_nccl::create_communicator_from_mpi(MPI_COMM_WORLD, config);
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("same CUDA device") !=
                   std::string::npos;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "unexpected device validation error: %s\n",
                     error.what());
    }

    int local_ok = rejected ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_ok != 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
