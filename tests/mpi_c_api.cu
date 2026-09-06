#include "nano_nccl/mpi_c_api.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

template <typename T>
T from_float(float value);

template <>
float from_float(float value) { return value; }

template <>
__nv_bfloat16 from_float(float value) { return __float2bfloat16(value); }

template <typename T>
float to_float(T value);

template <>
float to_float(float value) { return value; }

template <>
float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T>
bool run_case(nano_nccl_communicator_t* communicator, int rank,
              nano_nccl_dtype_t dtype) {
    constexpr size_t count = 1024 * 1024;
    std::vector<T> input(count, from_float<T>(static_cast<float>(rank + 1)));
    std::vector<T> output(count);
    T* send = nullptr;
    T* recv = nullptr;
    cudaStream_t stream = nullptr;
    bool ok = cuda_ok(cudaMalloc(&send, count * sizeof(T)), "cudaMalloc(send)") &&
              cuda_ok(cudaMalloc(&recv, count * sizeof(T)), "cudaMalloc(recv)") &&
              cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate") &&
              cuda_ok(cudaMemcpyAsync(send, input.data(), count * sizeof(T),
                                      cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync(input)");
    const void* send_buffers[] = {send};
    void* recv_buffers[] = {recv};
    cudaStream_t streams[] = {stream};
    nano_nccl_all_reduce_args_t args{
        send_buffers, recv_buffers, streams, count, dtype, NANO_NCCL_REDOP_SUM,
    };
    nano_nccl_status_t status = nano_nccl_all_reduce(communicator, &args);
    if (ok && status != NANO_NCCL_STATUS_SUCCESS) {
        std::fprintf(stderr, "all-reduce failed: %s\n",
                     nano_nccl_get_last_error());
        ok = false;
    }
    ok = cuda_ok(cudaMemcpyAsync(output.data(), recv, count * sizeof(T),
                                 cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync(output)") && ok;
    ok = cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize") && ok;
    status = nano_nccl_check_async_error(communicator);
    if (ok && status != NANO_NCCL_STATUS_SUCCESS) {
        std::fprintf(stderr, "async check failed: %s\n",
                     nano_nccl_get_last_error());
        ok = false;
    }
    for (T value : output) {
        if (std::fabs(to_float(value) - 3.0f) > 0.001f) {
            ok = false;
            break;
        }
    }
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (recv != nullptr) cudaFree(recv);
    if (send != nullptr) cudaFree(send);
    return ok;
}

}  // namespace

int main() {
    const char* rank_text = std::getenv("OMPI_COMM_WORLD_RANK");
    const char* local_rank_text = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (rank_text == nullptr || local_rank_text == nullptr) {
        std::fprintf(stderr, "run this smoke test under Open MPI\n");
        return EXIT_FAILURE;
    }
    const int rank = std::stoi(rank_text);
    setenv("CUDA_VISIBLE_DEVICES", local_rank_text, 1);

    nano_nccl_mpi_subgroup_config_t config{
        0, rank, 0, NANO_NCCL_TRANSPORT_SOCKET,
    };
    nano_nccl_communicator_t* communicator = nullptr;
    nano_nccl_status_t status = nano_nccl_create_mpi_subgroup_communicator(
        &config, &communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        std::fprintf(stderr, "communicator creation failed: %s\n",
                     nano_nccl_get_last_error());
        return EXIT_FAILURE;
    }
    int channels = 0;
    bool ok = nano_nccl_mpi_channel_count(&channels) == NANO_NCCL_STATUS_SUCCESS &&
              channels == 4 &&
              run_case<float>(communicator, rank, NANO_NCCL_DTYPE_FLOAT) &&
              run_case<__nv_bfloat16>(communicator, rank,
                                      NANO_NCCL_DTYPE_BFLOAT16);
    status = nano_nccl_destroy_communicator(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        std::fprintf(stderr, "communicator destruction failed: %s\n",
                     nano_nccl_get_last_error());
        ok = false;
    }
    if (ok) std::printf("rank=%d mpi_c_api float+bf16 OK\n", rank);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
