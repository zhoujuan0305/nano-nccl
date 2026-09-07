#include "nano_nccl/nano_nccl.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

_Static_assert(NANO_NCCL_ABI_VERSION == 2u,
               "one-process-per-GPU descriptors require C ABI version 2");
_Static_assert(sizeof(nano_nccl_status_t) == sizeof(int32_t),
               "status ABI must remain 32-bit");
_Static_assert(sizeof(nano_nccl_dtype_t) == sizeof(int32_t),
               "dtype ABI must remain 32-bit");
_Static_assert(sizeof(nano_nccl_redop_t) == sizeof(int32_t),
               "redop ABI must remain 32-bit");
_Static_assert(sizeof(nano_nccl_transport_t) == sizeof(int32_t),
               "transport ABI must remain 32-bit");
_Static_assert(sizeof(((nano_nccl_all_reduce_args_t*)0)->send_buffer) ==
                   sizeof(void*),
               "all-reduce send buffer must be scalar");
_Static_assert(sizeof(((nano_nccl_reduce_scatter_args_t*)0)->recv_buffer) ==
                   sizeof(void*),
               "reduce-scatter receive buffer must be scalar");
_Static_assert(sizeof(((nano_nccl_all_gather_args_t*)0)->stream) ==
                   sizeof(cudaStream_t),
               "all-gather stream must be scalar");

int main(void) {
    if (nano_nccl_abi_version() != 2u) {
        fprintf(stderr, "unexpected nano-nccl ABI version\n");
        return 1;
    }
    if (strcmp(nano_nccl_status_string(NANO_NCCL_STATUS_SUCCESS),
               "success") != 0) {
        fprintf(stderr, "unexpected success status string\n");
        return 1;
    }
    if (nano_nccl_destroy_communicator(NULL) != NANO_NCCL_STATUS_SUCCESS) {
        fprintf(stderr, "destroying a null communicator must succeed\n");
        return 1;
    }
    if (nano_nccl_all_reduce(NULL, NULL) !=
        NANO_NCCL_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "a null communicator was not rejected\n");
        return 1;
    }
    if (strstr(nano_nccl_get_last_error(), "communicator") == NULL) {
        fprintf(stderr, "invalid handle did not publish a diagnostic\n");
        return 1;
    }
    if (nano_nccl_edge_transport(NULL, 0, NULL) !=
        NANO_NCCL_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "a null edge-transport handle was not rejected\n");
        return 1;
    }
    return 0;
}
