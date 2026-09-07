#pragma once

#include "nano_nccl/communicator.h"

#include <memory>

#include <mpi.h>

namespace nano_nccl {

// Every rank in control_comm owns exactly one GPU selected by config.device.
// control_comm must contain kRanks processes and config.device must be visible.
std::unique_ptr<Communicator> create_communicator_from_mpi(
    MPI_Comm control_comm, const CommunicatorConfig& config);

}  // namespace nano_nccl
