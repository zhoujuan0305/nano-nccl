#pragma once

#include "nano_nccl/nano_nccl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Create one communicator from an MPI_COMM_WORLD subgroup. Every process in
// MPI_COMM_WORLD must call this function with a color; key determines the rank
// order inside that subgroup. MPI is initialized here when the embedding
// application has not initialized it already.
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
