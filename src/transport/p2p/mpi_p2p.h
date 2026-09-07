#pragma once

#include "nano_nccl/types.h"
#include "transport/connection.h"

#include <vector>

#include <mpi.h>

namespace nano_nccl::transport::p2p {

std::vector<bool> probe_mpi_p2p_edges(
    MPI_Comm control_comm, int global_rank, int device,
    const std::vector<int>& node_ids);

ConnectionResources create_mpi_p2p_connections(
    MPI_Comm control_comm, int global_rank, int device,
    const std::vector<TransportKind>& edge_kinds);

}  // namespace nano_nccl::transport::p2p
