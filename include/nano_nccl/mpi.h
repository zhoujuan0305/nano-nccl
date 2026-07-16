#pragma once

#include "nano_nccl/communicator.h"

#include <memory>

#include <mpi.h>

namespace nano_nccl {

std::unique_ptr<Communicator> create_communicator_from_mpi(
    MPI_Comm control_comm, const CommunicatorConfig& config);

}  // namespace nano_nccl
