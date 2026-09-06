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

static uint16_t encoded_rank_value(nano_nccl_dtype_t dtype, int value) {
    static const uint16_t kFloat16Values[] = {
        0x0000u, 0x3c00u, 0x4000u, 0x4200u, 0x4400u,
        0x4500u, 0x4600u, 0x4700u, 0x4800u,
    };
    static const uint16_t kBfloat16Values[] = {
        0x0000u, 0x3f80u, 0x4000u, 0x4040u, 0x4080u,
        0x40a0u, 0x40c0u, 0x40e0u, 0x4100u,
    };
    return dtype == NANO_NCCL_DTYPE_FLOAT16 ? kFloat16Values[value]
                                             : kBfloat16Values[value];
}

static float reduce_matrix_expected(nano_nccl_redop_t redop) {
    if (redop == NANO_NCCL_REDOP_SUM) {
        return (float)(NANO_NCCL_NRANKS * (NANO_NCCL_NRANKS + 1) / 2);
    }
    if (redop == NANO_NCCL_REDOP_AVG) {
        return (float)(NANO_NCCL_NRANKS + 1) / 2.0f;
    }
    if (redop == NANO_NCCL_REDOP_MAX) return (float)NANO_NCCL_NRANKS;
    return 1.0f;
}

static uint16_t encoded_reduce_matrix_expected(nano_nccl_dtype_t dtype,
                                                nano_nccl_redop_t redop) {
    if (redop == NANO_NCCL_REDOP_MAX) {
        return encoded_rank_value(dtype, NANO_NCCL_NRANKS);
    }
    if (redop == NANO_NCCL_REDOP_MIN) {
        return encoded_rank_value(dtype, 1);
    }
    if (dtype == NANO_NCCL_DTYPE_FLOAT16) {
        if (NANO_NCCL_NRANKS == 2) {
            return redop == NANO_NCCL_REDOP_SUM ? 0x4200u : 0x3e00u;
        }
        if (NANO_NCCL_NRANKS == 4) {
            return redop == NANO_NCCL_REDOP_SUM ? 0x4900u : 0x4100u;
        }
        return redop == NANO_NCCL_REDOP_SUM ? 0x5080u : 0x4480u;
    }
    if (NANO_NCCL_NRANKS == 2) {
        return redop == NANO_NCCL_REDOP_SUM ? 0x4040u : 0x3fc0u;
    }
    if (NANO_NCCL_NRANKS == 4) {
        return redop == NANO_NCCL_REDOP_SUM ? 0x4120u : 0x4020u;
    }
    return redop == NANO_NCCL_REDOP_SUM ? 0x4210u : 0x4090u;
}

int main(void) {
    nano_nccl_communicator_t* communicator = NULL;
    const void* send_buffers[NANO_NCCL_NRANKS] = {0};
    void* recv_buffers[NANO_NCCL_NRANKS] = {0};
    cudaStream_t streams[NANO_NCCL_NRANKS] = {0};
    float inputs[NANO_NCCL_NRANKS][kCount];
    float output[kCount];
    uint16_t inputs16[NANO_NCCL_NRANKS][kCount];
    uint16_t output16[kCount];
    const size_t bytes = kCount * sizeof(float);
    int devices[NANO_NCCL_NRANKS + 1];
    int result = 1;
    nano_nccl_status_t status = NANO_NCCL_STATUS_SUCCESS;
    int bf16_supported = 1;

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
        struct cudaDeviceProp properties;
        cuda_status = cudaGetDeviceProperties(&properties, devices[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for rank %d: %s\n",
                    rank, cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        if (properties.major < 8) bf16_supported = 0;
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
    void* saved_recv_buffer = recv_buffers[0];
    recv_buffers[0] =
        (void*)((unsigned char*)send_buffers[0] + sizeof(float));
    status = nano_nccl_all_reduce(communicator, &all_reduce_args);
    recv_buffers[0] = saved_recv_buffer;
    if (status != NANO_NCCL_STATUS_INVALID_ARGUMENT ||
        strstr(nano_nccl_get_last_error(), "overlapping") == NULL) {
        fprintf(stderr, "partial-overlap all_reduce was not rejected\n");
        goto cleanup;
    }
    saved_recv_buffer = recv_buffers[1];
    recv_buffers[1] =
        (void*)((unsigned char*)send_buffers[0] + sizeof(float));
    status = nano_nccl_all_reduce(communicator, &all_reduce_args);
    recv_buffers[1] = saved_recv_buffer;
    if (status != NANO_NCCL_STATUS_INVALID_ARGUMENT ||
        strstr(nano_nccl_get_last_error(), "overlapping") == NULL) {
        fprintf(stderr, "cross-entry overlap was not rejected\n");
        goto cleanup;
    }
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

    const size_t recv_count = kCount / NANO_NCCL_NRANKS;
    for (int rank = 0; rank < NANO_NCCL_NRANKS; ++rank) {
        cuda_status = cudaSetDevice(devices[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        for (int index = 0; index < kCount; ++index) {
            inputs[rank][index] = (float)(rank * 10 + index);
        }
        cuda_status = cudaMemcpyAsync((void*)send_buffers[rank], inputs[rank],
                                      bytes, cudaMemcpyHostToDevice,
                                      streams[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "reduce_scatter input copy failed for rank %d: %s\n",
                    rank, cudaGetErrorString(cuda_status));
            goto cleanup;
        }
    }

    nano_nccl_reduce_scatter_args_t reduce_scatter_args = {
        send_buffers,
        recv_buffers,
        streams,
        recv_count,
        NANO_NCCL_DTYPE_FLOAT,
        NANO_NCCL_REDOP_SUM,
    };
    saved_recv_buffer = recv_buffers[0];
    recv_buffers[0] =
        (void*)((unsigned char*)send_buffers[0] + sizeof(float));
    status = nano_nccl_reduce_scatter(communicator, &reduce_scatter_args);
    recv_buffers[0] = saved_recv_buffer;
    if (status != NANO_NCCL_STATUS_INVALID_ARGUMENT ||
        strstr(nano_nccl_get_last_error(), "overlapping") == NULL) {
        fprintf(stderr, "partial-overlap reduce_scatter was not rejected\n");
        goto cleanup;
    }
    status = nano_nccl_reduce_scatter(communicator, &reduce_scatter_args);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        fail_status("nano_nccl_reduce_scatter", status);
        goto cleanup;
    }
    for (int rank = 0; rank < NANO_NCCL_NRANKS; ++rank) {
        cuda_status = cudaSetDevice(devices[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        cuda_status = cudaStreamSynchronize(streams[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "reduce_scatter sync failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        cuda_status = cudaMemcpy(output, recv_buffers[rank],
                                 recv_count * sizeof(float),
                                 cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "reduce_scatter output copy failed for rank %d: %s\n",
                    rank, cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        for (size_t index = 0; index < recv_count; ++index) {
            const size_t global_index = (size_t)rank * recv_count + index;
            const float reduce_scatter_expected =
                (float)(10 * NANO_NCCL_NRANKS * (NANO_NCCL_NRANKS - 1) / 2) +
                (float)(global_index * NANO_NCCL_NRANKS);
            if (output[index] != reduce_scatter_expected) {
                fprintf(stderr,
                        "reduce_scatter mismatch rank=%d index=%zu got=%f "
                        "expected=%f\n",
                        rank, index, output[index], reduce_scatter_expected);
                goto cleanup;
            }
        }
    }

    const nano_nccl_dtype_t reduce_dtypes[] = {
        NANO_NCCL_DTYPE_FLOAT,
        NANO_NCCL_DTYPE_FLOAT16,
        NANO_NCCL_DTYPE_BFLOAT16,
    };
    const nano_nccl_redop_t reduce_ops[] = {
        NANO_NCCL_REDOP_SUM,
        NANO_NCCL_REDOP_AVG,
        NANO_NCCL_REDOP_MAX,
        NANO_NCCL_REDOP_MIN,
    };
    for (size_t dtype_index = 0;
         dtype_index < sizeof(reduce_dtypes) / sizeof(reduce_dtypes[0]);
         ++dtype_index) {
        const nano_nccl_dtype_t dtype = reduce_dtypes[dtype_index];
        if (dtype == NANO_NCCL_DTYPE_BFLOAT16 && !bf16_supported) continue;
        const size_t element_size =
            dtype == NANO_NCCL_DTYPE_FLOAT ? sizeof(float) : sizeof(uint16_t);
        for (size_t op_index = 0;
             op_index < sizeof(reduce_ops) / sizeof(reduce_ops[0]);
             ++op_index) {
            const nano_nccl_redop_t redop = reduce_ops[op_index];
            for (int rank = 0; rank < NANO_NCCL_NRANKS; ++rank) {
                cuda_status = cudaSetDevice(devices[rank]);
                if (cuda_status != cudaSuccess) goto cleanup;
                if (dtype == NANO_NCCL_DTYPE_FLOAT) {
                    for (int index = 0; index < kCount; ++index) {
                        inputs[rank][index] = (float)(rank + 1);
                    }
                    cuda_status = cudaMemcpyAsync(
                        (void*)send_buffers[rank], inputs[rank],
                        kCount * element_size, cudaMemcpyHostToDevice,
                        streams[rank]);
                } else {
                    for (int index = 0; index < kCount; ++index) {
                        inputs16[rank][index] =
                            encoded_rank_value(dtype, rank + 1);
                    }
                    cuda_status = cudaMemcpyAsync(
                        (void*)send_buffers[rank], inputs16[rank],
                        kCount * element_size, cudaMemcpyHostToDevice,
                        streams[rank]);
                }
                if (cuda_status != cudaSuccess) {
                    fprintf(stderr, "reduce_scatter matrix input copy failed\n");
                    goto cleanup;
                }
            }
            reduce_scatter_args.dtype = dtype;
            reduce_scatter_args.redop = redop;
            status = nano_nccl_reduce_scatter(communicator,
                                              &reduce_scatter_args);
            if (status != NANO_NCCL_STATUS_SUCCESS) {
                fail_status("nano_nccl_reduce_scatter matrix", status);
                goto cleanup;
            }
            for (int rank = 0; rank < NANO_NCCL_NRANKS; ++rank) {
                cuda_status = cudaSetDevice(devices[rank]);
                if (cuda_status != cudaSuccess) goto cleanup;
                cuda_status = cudaStreamSynchronize(streams[rank]);
                if (cuda_status != cudaSuccess) goto cleanup;
                if (dtype == NANO_NCCL_DTYPE_FLOAT) {
                    cuda_status = cudaMemcpy(
                        output, recv_buffers[rank], recv_count * element_size,
                        cudaMemcpyDeviceToHost);
                    if (cuda_status != cudaSuccess) {
                        fprintf(stderr,
                                "reduce_scatter matrix output copy failed\n");
                        goto cleanup;
                    }
                    const float matrix_expected =
                        reduce_matrix_expected(redop);
                    for (size_t index = 0; index < recv_count; ++index) {
                        if (output[index] != matrix_expected) {
                            fprintf(stderr,
                                    "reduce_scatter float ABI matrix mismatch\n");
                            goto cleanup;
                        }
                    }
                } else {
                    cuda_status = cudaMemcpy(
                        output16, recv_buffers[rank], recv_count * element_size,
                        cudaMemcpyDeviceToHost);
                    if (cuda_status != cudaSuccess) {
                        fprintf(stderr,
                                "reduce_scatter matrix output copy failed\n");
                        goto cleanup;
                    }
                    const uint16_t matrix_expected =
                        encoded_reduce_matrix_expected(dtype, redop);
                    for (size_t index = 0; index < recv_count; ++index) {
                        if (output16[index] != matrix_expected) {
                            fprintf(stderr,
                                    "reduce_scatter 16-bit ABI matrix mismatch\n");
                            goto cleanup;
                        }
                    }
                }
            }
        }
    }

    reduce_scatter_args.dtype = NANO_NCCL_DTYPE_FLOAT;
    reduce_scatter_args.redop = NANO_NCCL_REDOP_SUM;

    const size_t send_count = kCount / NANO_NCCL_NRANKS;
    for (int rank = 0; rank < NANO_NCCL_NRANKS; ++rank) {
        cuda_status = cudaSetDevice(devices[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        for (size_t index = 0; index < send_count; ++index) {
            inputs[rank][index] = (float)(rank * 100) + (float)index;
        }
        cuda_status = cudaMemcpyAsync((void*)send_buffers[rank], inputs[rank],
                                      send_count * sizeof(float),
                                      cudaMemcpyHostToDevice, streams[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "all_gather input copy failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
    }

    nano_nccl_all_gather_args_t all_gather_args = {
        send_buffers,
        recv_buffers,
        streams,
        send_count,
        NANO_NCCL_DTYPE_FLOAT,
    };
    saved_recv_buffer = recv_buffers[0];
    recv_buffers[0] =
        (void*)((unsigned char*)send_buffers[0] + sizeof(float));
    status = nano_nccl_all_gather(communicator, &all_gather_args);
    recv_buffers[0] = saved_recv_buffer;
    if (status != NANO_NCCL_STATUS_INVALID_ARGUMENT ||
        strstr(nano_nccl_get_last_error(), "overlapping") == NULL) {
        fprintf(stderr, "partial-overlap all_gather was not rejected\n");
        goto cleanup;
    }
    status = nano_nccl_all_gather(communicator, &all_gather_args);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        fail_status("nano_nccl_all_gather", status);
        goto cleanup;
    }
    for (int rank = 0; rank < NANO_NCCL_NRANKS; ++rank) {
        cuda_status = cudaSetDevice(devices[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        cuda_status = cudaStreamSynchronize(streams[rank]);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "all_gather sync failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        cuda_status = cudaMemcpy(output, recv_buffers[rank], bytes,
                                 cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "all_gather output copy failed for rank %d: %s\n", rank,
                    cudaGetErrorString(cuda_status));
            goto cleanup;
        }
        for (int source_rank = 0; source_rank < NANO_NCCL_NRANKS; ++source_rank) {
            for (size_t index = 0; index < send_count; ++index) {
                const size_t output_index =
                    (size_t)source_rank * send_count + index;
                const float all_gather_expected =
                    (float)(source_rank * 100) + (float)index;
                if (output[output_index] != all_gather_expected) {
                    fprintf(stderr,
                            "all_gather mismatch rank=%d source=%d index=%zu "
                            "got=%f expected=%f\n",
                            rank, source_rank, index, output[output_index],
                            all_gather_expected);
                    goto cleanup;
                }
            }
        }
    }

    for (size_t dtype_index = 1;
         dtype_index < sizeof(reduce_dtypes) / sizeof(reduce_dtypes[0]);
         ++dtype_index) {
        const nano_nccl_dtype_t dtype = reduce_dtypes[dtype_index];
        if (dtype == NANO_NCCL_DTYPE_BFLOAT16 && !bf16_supported) continue;
        for (int rank = 0; rank < NANO_NCCL_NRANKS; ++rank) {
            cuda_status = cudaSetDevice(devices[rank]);
            if (cuda_status != cudaSuccess) goto cleanup;
            for (size_t index = 0; index < send_count; ++index) {
                inputs16[rank][index] =
                    (uint16_t)((size_t)rank * send_count + index + 1);
            }
            cuda_status = cudaMemcpyAsync(
                (void*)send_buffers[rank], inputs16[rank],
                send_count * sizeof(uint16_t), cudaMemcpyHostToDevice,
                streams[rank]);
            if (cuda_status != cudaSuccess) goto cleanup;
        }
        all_gather_args.dtype = dtype;
        status = nano_nccl_all_gather(communicator, &all_gather_args);
        if (status != NANO_NCCL_STATUS_SUCCESS) {
            fail_status("nano_nccl_all_gather 16-bit matrix", status);
            goto cleanup;
        }
        for (int rank = 0; rank < NANO_NCCL_NRANKS; ++rank) {
            cuda_status = cudaSetDevice(devices[rank]);
            if (cuda_status != cudaSuccess) goto cleanup;
            cuda_status = cudaStreamSynchronize(streams[rank]);
            if (cuda_status != cudaSuccess) goto cleanup;
            cuda_status = cudaMemcpy(output16, recv_buffers[rank],
                                     kCount * sizeof(uint16_t),
                                     cudaMemcpyDeviceToHost);
            if (cuda_status != cudaSuccess) goto cleanup;
            for (int source_rank = 0; source_rank < NANO_NCCL_NRANKS;
                 ++source_rank) {
                for (size_t index = 0; index < send_count; ++index) {
                    const size_t output_index =
                        (size_t)source_rank * send_count + index;
                    const uint16_t expected_bits =
                        (uint16_t)(output_index + 1);
                    if (output16[output_index] != expected_bits) {
                        fprintf(stderr, "all_gather 16-bit ABI matrix mismatch\n");
                        goto cleanup;
                    }
                }
            }
        }
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
