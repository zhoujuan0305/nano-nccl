#include "nano_nccl/nano_nccl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { kCount = 16 };

_Static_assert(sizeof(nano_nccl_status_t) == sizeof(int32_t),
               "status ABI must remain 32-bit");
_Static_assert(sizeof(nano_nccl_dtype_t) == sizeof(int32_t),
               "dtype ABI must remain 32-bit");
_Static_assert(sizeof(nano_nccl_redop_t) == sizeof(int32_t),
               "redop ABI must remain 32-bit");
_Static_assert(sizeof(nano_nccl_transport_t) == sizeof(int32_t),
               "transport ABI must remain 32-bit");

static int fail_status(const char* call, nano_nccl_status_t status) {
    fprintf(stderr, "%s failed: status=%s detail=%s\n", call,
            nano_nccl_status_string(status), nano_nccl_get_last_error());
    return 1;
}

int main(void) {
    nano_nccl_communicator_t* communicator = NULL;
    const void* send_buffers[NANO_NCCL_NRANKS] = {0};
    void* recv_buffers[NANO_NCCL_NRANKS] = {0};
    cudaStream_t streams[NANO_NCCL_NRANKS] = {0};
    float inputs[NANO_NCCL_NRANKS][kCount];
    float output[kCount];
    const size_t bytes = kCount * sizeof(float);
    int devices[NANO_NCCL_NRANKS + 1];
    int result = 1;
    nano_nccl_status_t status = NANO_NCCL_STATUS_SUCCESS;

    for (int rank = 0; rank <= NANO_NCCL_NRANKS; ++rank) {
        devices[rank] = rank;
    }
    if (nano_nccl_abi_version() != NANO_NCCL_ABI_VERSION) {
        fprintf(stderr, "ABI version mismatch\n");
        return 1;
    }
    if (nano_nccl_create_communicator(NULL, &communicator) !=
        NANO_NCCL_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "null config was not rejected\n");
        return 1;
    }
    if (strstr(nano_nccl_get_last_error(), "config") == NULL) {
        fprintf(stderr, "invalid argument did not publish a diagnostic\n");
        return 1;
    }

    nano_nccl_communicator_config_t wrong_rank_count_config = {
        devices,
        NANO_NCCL_NRANKS + 1,
        NANO_NCCL_TRANSPORT_AUTO,
    };
    status = nano_nccl_create_communicator(&wrong_rank_count_config, &communicator);
    if (status != NANO_NCCL_STATUS_ERROR || communicator != NULL ||
        strstr(nano_nccl_get_last_error(), "exactly") == NULL) {
        fprintf(stderr, "C++ setup exception crossed or was translated incorrectly\n");
        return 1;
    }

    int visible_devices = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&visible_devices);
    if (cuda_status != cudaSuccess || visible_devices < NANO_NCCL_NRANKS) {
        return 77;
    }

    nano_nccl_communicator_config_t config = {
        devices,
        NANO_NCCL_NRANKS,
        NANO_NCCL_TRANSPORT_AUTO,
    };
    status = nano_nccl_create_communicator(&config, &communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        return fail_status("nano_nccl_create_communicator", status);
    }

    int local_rank_count = 0;
    int global_rank_count = 0;
    nano_nccl_transport_t transport = NANO_NCCL_TRANSPORT_AUTO;
    if ((status = nano_nccl_local_rank_count(communicator, &local_rank_count)) !=
            NANO_NCCL_STATUS_SUCCESS ||
        (status = nano_nccl_global_rank_count(communicator, &global_rank_count)) !=
            NANO_NCCL_STATUS_SUCCESS ||
        (status = nano_nccl_transport(communicator, &transport)) !=
            NANO_NCCL_STATUS_SUCCESS) {
        fail_status("communicator query", status);
        goto cleanup;
    }
    if (local_rank_count != NANO_NCCL_NRANKS ||
        global_rank_count != NANO_NCCL_NRANKS ||
        (transport != NANO_NCCL_TRANSPORT_SHM &&
         transport != NANO_NCCL_TRANSPORT_P2P &&
         transport != NANO_NCCL_TRANSPORT_MIXED)) {
        fprintf(stderr, "communicator query returned unexpected values\n");
        goto cleanup;
    }

    if (nano_nccl_all_reduce(communicator, NULL) !=
            NANO_NCCL_STATUS_INVALID_ARGUMENT ||
        strstr(nano_nccl_get_last_error(), "all_reduce args") == NULL) {
        fprintf(stderr, "null all_reduce args were not rejected\n");
        goto cleanup;
    }

    for (int rank = 0; rank < NANO_NCCL_NRANKS; ++rank) {
        void* send_buffer = NULL;
        cuda_status = cudaSetDevice(devices[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        cuda_status = cudaMalloc(&send_buffer, bytes);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "send cudaMalloc failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        send_buffers[rank] = send_buffer;
        cuda_status = cudaMalloc(&recv_buffers[rank], bytes);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "recv cudaMalloc failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        cuda_status =
            cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaStreamCreateWithFlags failed for rank %d: %s\n",
                    rank, cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        for (int index = 0; index < kCount; ++index) {
            inputs[rank][index] = (float)(rank + 1);
        }
        cuda_status = cudaMemcpyAsync(send_buffer, inputs[rank], bytes,
                                      cudaMemcpyHostToDevice, streams[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
    }

    nano_nccl_all_reduce_args_t all_reduce_args = {
        send_buffers,
        recv_buffers,
        streams,
        kCount,
        NANO_NCCL_DTYPE_FLOAT,
        NANO_NCCL_REDOP_SUM,
    };
    status = nano_nccl_all_reduce(communicator, &all_reduce_args);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        fail_status("nano_nccl_all_reduce", status);
        goto cleanup;
    }
    if (nano_nccl_get_last_error()[0] != '\0') {
        fprintf(stderr, "successful all_reduce did not clear the diagnostic\n");
        goto cleanup;
    }

    const float expected =
        (float)(NANO_NCCL_NRANKS * (NANO_NCCL_NRANKS + 1) / 2);
    for (int rank = 0; rank < NANO_NCCL_NRANKS; ++rank) {
        cuda_status = cudaSetDevice(devices[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        cuda_status = cudaStreamSynchronize(streams[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaStreamSynchronize failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        cuda_status = cudaMemcpy(output, recv_buffers[rank], bytes,
                                 cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        for (int index = 0; index < kCount; ++index) {
            if (fabsf(output[index] - expected) > 1e-6f) {
                fprintf(stderr, "all_reduce mismatch rank=%d index=%d got=%f expected=%f\n",
                        rank, index, output[index], expected);
                goto cleanup;
            }
        }
    }

    nano_nccl_reduce_scatter_args_t reduce_scatter_args = {
        send_buffers,
        recv_buffers,
        streams,
        kCount,
        NANO_NCCL_DTYPE_FLOAT,
        NANO_NCCL_REDOP_SUM,
    };
    status = nano_nccl_reduce_scatter(communicator, &reduce_scatter_args);
    if (status != NANO_NCCL_STATUS_UNSUPPORTED ||
        strstr(nano_nccl_get_last_error(), "reduce_scatter") == NULL) {
        fprintf(stderr, "reduce_scatter did not return UNSUPPORTED with a diagnostic\n");
        goto cleanup;
    }

    nano_nccl_all_gather_args_t all_gather_args = {
        send_buffers,
        recv_buffers,
        streams,
        kCount,
        NANO_NCCL_DTYPE_FLOAT,
    };
    status = nano_nccl_all_gather(communicator, &all_gather_args);
    if (status != NANO_NCCL_STATUS_UNSUPPORTED ||
        strstr(nano_nccl_get_last_error(), "all_gather") == NULL) {
        fprintf(stderr, "all_gather did not return UNSUPPORTED with a diagnostic\n");
        goto cleanup;
    }

    status = nano_nccl_check_async_error(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        fail_status("nano_nccl_check_async_error", status);
        goto cleanup;
    }
    result = 0;

cleanup:
    status = nano_nccl_destroy_communicator(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        fail_status("nano_nccl_destroy_communicator", status);
        result = 1;
    }
    for (int rank = 0; rank < NANO_NCCL_NRANKS; ++rank) {
        if (send_buffers[rank] == NULL && recv_buffers[rank] == NULL &&
            streams[rank] == NULL) {
            continue;
        }
        cuda_status = cudaSetDevice(devices[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cleanup cudaSetDevice failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            result = 1;
            continue;
        }
        if (streams[rank] != NULL) {
            cuda_status = cudaStreamSynchronize(streams[rank]);
            if (cuda_status != cudaSuccess) {
                fprintf(stderr, "cleanup cudaStreamSynchronize failed for rank %d: %s\n",
                        rank, cudaGetErrorString(cuda_status));
                result = 1;
            }
        }
        if (send_buffers[rank] != NULL) {
            cuda_status = cudaFree((void*)send_buffers[rank]);
            if (cuda_status != cudaSuccess) {
                fprintf(stderr, "send cudaFree failed for rank %d: %s\n", rank,
                        cudaGetErrorString(cuda_status));
                result = 1;
            }
        }
        if (recv_buffers[rank] != NULL) {
            cuda_status = cudaFree(recv_buffers[rank]);
            if (cuda_status != cudaSuccess) {
                fprintf(stderr, "recv cudaFree failed for rank %d: %s\n", rank,
                        cudaGetErrorString(cuda_status));
                result = 1;
            }
        }
        if (streams[rank] != NULL) {
            cuda_status = cudaStreamDestroy(streams[rank]);
            if (cuda_status != cudaSuccess) {
                fprintf(stderr, "cudaStreamDestroy failed for rank %d: %s\n", rank,
                        cudaGetErrorString(cuda_status));
                result = 1;
            }
        }
    }
    return result;
}
