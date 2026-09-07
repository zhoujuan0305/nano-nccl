#pragma once

#include "nano_nccl/types.h"
#include "transport/connection.h"

#include <vector>

#include <mpi.h>

namespace nano_nccl::transport::shm {

ConnectionResources create_mpi_shm_connections(
    MPI_Comm control_comm, int global_rank, int device,
    const std::vector<TransportKind>& edge_kinds);

}  // namespace nano_nccl::transport::shm
