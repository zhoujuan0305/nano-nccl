#pragma once

#include <cuda_runtime_api.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NANO_NCCL_ABI_VERSION 2u

typedef struct NanoNcclCommunicator nano_nccl_communicator_t;

typedef int32_t nano_nccl_status_t;
enum {
    NANO_NCCL_STATUS_SUCCESS = 0,
    NANO_NCCL_STATUS_INVALID_ARGUMENT = 1,
    NANO_NCCL_STATUS_UNSUPPORTED = 2,
    NANO_NCCL_STATUS_ERROR = 3,
};

typedef int32_t nano_nccl_dtype_t;
enum {
    NANO_NCCL_DTYPE_FLOAT = 0,
    NANO_NCCL_DTYPE_FLOAT16 = 1,
    NANO_NCCL_DTYPE_BFLOAT16 = 2,
};

typedef int32_t nano_nccl_redop_t;
enum {
    NANO_NCCL_REDOP_SUM = 0,
    NANO_NCCL_REDOP_AVG = 1,
    NANO_NCCL_REDOP_MAX = 2,
    NANO_NCCL_REDOP_MIN = 3,
};

typedef int32_t nano_nccl_transport_t;
enum {
    NANO_NCCL_TRANSPORT_AUTO = 0,
    NANO_NCCL_TRANSPORT_SHM = 1,
    NANO_NCCL_TRANSPORT_P2P = 2,
    NANO_NCCL_TRANSPORT_SOCKET = 3,
    NANO_NCCL_TRANSPORT_RDMA = 4,
    NANO_NCCL_TRANSPORT_MIXED = 5,
};

typedef struct NanoNcclAllReduceArgs {
    const void* send_buffer;
    void* recv_buffer;
    cudaStream_t stream;
    size_t count;  // Input and output elements per rank.
    nano_nccl_dtype_t dtype;
    nano_nccl_redop_t redop;
} nano_nccl_all_reduce_args_t;

typedef struct NanoNcclReduceScatterArgs {
    const void* send_buffer;
    void* recv_buffer;
    cudaStream_t stream;
    size_t recv_count;  // Output elements per rank; input has global-rank multiple.
    nano_nccl_dtype_t dtype;
    nano_nccl_redop_t redop;
} nano_nccl_reduce_scatter_args_t;

typedef struct NanoNcclAllGatherArgs {
    const void* send_buffer;
    void* recv_buffer;
    cudaStream_t stream;
    size_t send_count;  // Input elements per rank; output has global-rank multiple.
    nano_nccl_dtype_t dtype;
} nano_nccl_all_gather_args_t;

uint32_t nano_nccl_abi_version(void);
const char* nano_nccl_status_string(nano_nccl_status_t status);

// The returned pointer remains valid until the next stateful nano-nccl C ABI
// call on the same thread. ABI/status queries do not clear the diagnostic.
const char* nano_nccl_get_last_error(void);

nano_nccl_status_t nano_nccl_destroy_communicator(
    nano_nccl_communicator_t* communicator);

nano_nccl_status_t nano_nccl_all_reduce(
    nano_nccl_communicator_t* communicator,
    const nano_nccl_all_reduce_args_t* args);
nano_nccl_status_t nano_nccl_reduce_scatter(
    nano_nccl_communicator_t* communicator,
    const nano_nccl_reduce_scatter_args_t* args);
nano_nccl_status_t nano_nccl_all_gather(
    nano_nccl_communicator_t* communicator,
    const nano_nccl_all_gather_args_t* args);

nano_nccl_status_t nano_nccl_check_async_error(
    const nano_nccl_communicator_t* communicator);
nano_nccl_status_t nano_nccl_local_rank_count(
    const nano_nccl_communicator_t* communicator, int* rank_count);
nano_nccl_status_t nano_nccl_global_rank_count(
    const nano_nccl_communicator_t* communicator, int* rank_count);
nano_nccl_status_t nano_nccl_transport(
    const nano_nccl_communicator_t* communicator,
    nano_nccl_transport_t* transport);
// Query source_global_rank -> (source_global_rank + 1) % global_rank_count.
nano_nccl_status_t nano_nccl_edge_transport(
    const nano_nccl_communicator_t* communicator, int source_global_rank,
    nano_nccl_transport_t* transport);

#ifdef __cplusplus
}
#endif
