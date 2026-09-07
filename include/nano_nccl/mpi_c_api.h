#pragma once

#include "nano_nccl/nano_nccl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Create a one-process-per-GPU communicator from an MPI_COMM_WORLD subgroup.
// Every world process must call this function with a color; key determines the
// rank order inside that subgroup. Each subgroup must contain the build-time
// rank count. device selects this process's one GPU rank from the visible
// CUDA devices; callers may either mask to cuda:0 or pass their local rank.
// MPI is initialized here when the embedding application has not initialized
// it already. Destruction is collective within the created subgroup so every
// member must call nano_nccl_destroy_communicator in the same communicator
// order.
typedef struct NanoNcclMpiSubgroupConfig {
    int color;
    int key;
    int device;
    nano_nccl_transport_t transport;
} nano_nccl_mpi_subgroup_config_t;

nano_nccl_status_t nano_nccl_create_mpi_subgroup_communicator(
    const nano_nccl_mpi_subgroup_config_t* config,
    nano_nccl_communicator_t** communicator);

nano_nccl_status_t nano_nccl_mpi_channel_count(int* channel_count);

#ifdef __cplusplus
}
#endif
