#include "nano_nccl/mpi_c_api.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

static_assert(NANO_NCCL_ABI_VERSION == 2u,
              "one-process-per-GPU descriptors require C ABI version 2");

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
              int rank_count, nano_nccl_dtype_t dtype) {
    constexpr size_t count = 1024 * 1024;
    const size_t large_count = count * static_cast<size_t>(rank_count);
    const float expected_sum =
        static_cast<float>(rank_count * (rank_count + 1) / 2);
    std::vector<T> input(
        large_count, from_float<T>(static_cast<float>(rank + 1)));
    std::vector<T> output(large_count);
    T* send = nullptr;
    T* recv = nullptr;
    cudaStream_t stream = nullptr;
    bool ok = cuda_ok(cudaMalloc(&send, large_count * sizeof(T)),
                      "cudaMalloc(send)") &&
              cuda_ok(cudaMalloc(&recv, large_count * sizeof(T)),
                      "cudaMalloc(recv)") &&
              cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate") &&
              cuda_ok(cudaMemcpyAsync(send, input.data(),
                                      large_count * sizeof(T),
                                      cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync(input)");
    nano_nccl_all_reduce_args_t args{};
    args.send_buffer = send;
    args.recv_buffer = recv;
    args.stream = stream;
    args.count = count;
    args.dtype = dtype;
    args.redop = NANO_NCCL_REDOP_SUM;
    nano_nccl_all_reduce_args_t invalid_args = args;
    invalid_args.recv_buffer = send;
    nano_nccl_status_t status = nano_nccl_all_reduce(
        communicator, &invalid_args);
    if (status != NANO_NCCL_STATUS_INVALID_ARGUMENT) {
        std::fprintf(stderr, "in-place descriptor was not rejected\n");
        ok = false;
    }
    status = nano_nccl_all_reduce(communicator, &args);
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
    for (size_t index = 0; index < count; ++index) {
        if (std::fabs(to_float(output[index]) - expected_sum) > 0.001f) {
            ok = false;
            break;
        }
    }

    nano_nccl_reduce_scatter_args_t reduce_scatter_args{};
    reduce_scatter_args.send_buffer = send;
    reduce_scatter_args.recv_buffer = recv;
    reduce_scatter_args.stream = stream;
    reduce_scatter_args.recv_count = count;
    reduce_scatter_args.dtype = dtype;
    reduce_scatter_args.redop = NANO_NCCL_REDOP_SUM;
    status = nano_nccl_reduce_scatter(communicator, &reduce_scatter_args);
    ok = status == NANO_NCCL_STATUS_SUCCESS && ok;
    ok = cuda_ok(cudaMemcpyAsync(output.data(), recv, count * sizeof(T),
                                 cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync(reduce_scatter output)") && ok;
    ok = cuda_ok(cudaStreamSynchronize(stream),
                 "cudaStreamSynchronize(reduce_scatter)") && ok;
    for (size_t index = 0; index < count; ++index) {
        if (std::fabs(to_float(output[index]) - expected_sum) > 0.001f) {
            ok = false;
            break;
        }
    }

    nano_nccl_all_gather_args_t all_gather_args{};
    all_gather_args.send_buffer = send;
    all_gather_args.recv_buffer = recv;
    all_gather_args.stream = stream;
    all_gather_args.send_count = count;
    all_gather_args.dtype = dtype;
    status = nano_nccl_all_gather(communicator, &all_gather_args);
    ok = status == NANO_NCCL_STATUS_SUCCESS && ok;
    ok = cuda_ok(cudaMemcpyAsync(output.data(), recv,
                                 large_count * sizeof(T),
                                 cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync(all_gather output)") && ok;
    ok = cuda_ok(cudaStreamSynchronize(stream),
                 "cudaStreamSynchronize(all_gather)") && ok;
    for (int source_rank = 0; source_rank < rank_count; ++source_rank) {
        for (size_t index = 0; index < count; ++index) {
            const size_t output_index =
                static_cast<size_t>(source_rank) * count + index;
            if (std::fabs(to_float(output[output_index]) -
                          static_cast<float>(source_rank + 1)) > 0.001f) {
                ok = false;
                break;
            }
        }
    }
    status = nano_nccl_check_async_error(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        std::fprintf(stderr, "final async check failed: %s\n",
                     nano_nccl_get_last_error());
        ok = false;
    }
    if (stream != nullptr) {
        ok = cuda_ok(cudaStreamDestroy(stream), "cudaStreamDestroy") && ok;
    }
    if (recv != nullptr) ok = cuda_ok(cudaFree(recv), "cudaFree(recv)") && ok;
    if (send != nullptr) ok = cuda_ok(cudaFree(send), "cudaFree(send)") && ok;
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const char* rank_text = std::getenv("OMPI_COMM_WORLD_RANK");
    const char* local_rank_text = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (rank_text == nullptr || local_rank_text == nullptr) {
        std::fprintf(stderr, "run this smoke test under Open MPI\n");
        return EXIT_FAILURE;
    }
    const int rank = std::stoi(rank_text);
    setenv("CUDA_VISIBLE_DEVICES", local_rank_text, 1);

    nano_nccl_transport_t requested_transport = NANO_NCCL_TRANSPORT_SOCKET;
    nano_nccl_transport_t expected_transport = NANO_NCCL_TRANSPORT_SOCKET;
    if (argc == 2 && std::string(argv[1]) == "p2p") {
        requested_transport = NANO_NCCL_TRANSPORT_P2P;
        expected_transport = NANO_NCCL_TRANSPORT_P2P;
    } else if (argc == 2 && std::string(argv[1]) == "auto") {
        requested_transport = NANO_NCCL_TRANSPORT_AUTO;
        expected_transport = NANO_NCCL_TRANSPORT_P2P;
    } else if (argc == 2 && std::string(argv[1]) == "shm") {
        requested_transport = NANO_NCCL_TRANSPORT_SHM;
        expected_transport = NANO_NCCL_TRANSPORT_SHM;
    } else if (argc != 1 && !(argc == 2 && std::string(argv[1]) == "socket")) {
        std::fprintf(stderr, "usage: %s [socket|shm|p2p|auto]\n", argv[0]);
        return EXIT_FAILURE;
    }

    nano_nccl_mpi_subgroup_config_t config{
        0, rank, 0, requested_transport,
    };
    nano_nccl_communicator_t* communicator = nullptr;
    nano_nccl_status_t status = nano_nccl_create_mpi_subgroup_communicator(
        &config, &communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        std::fprintf(stderr, "communicator creation failed: %s\n",
                     nano_nccl_get_last_error());
        if (requested_transport == NANO_NCCL_TRANSPORT_P2P &&
            std::string(nano_nccl_get_last_error()).find(
                "failed the CUDA IPC probe") != std::string::npos) {
            return 77;
        }
        return EXIT_FAILURE;
    }
    nano_nccl_communicator_t* second_communicator = nullptr;
    status = nano_nccl_create_mpi_subgroup_communicator(
        &config, &second_communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        std::fprintf(stderr, "second communicator creation failed: %s\n",
                     nano_nccl_get_last_error());
        nano_nccl_destroy_communicator(communicator);
        return EXIT_FAILURE;
    }
    status = nano_nccl_destroy_communicator(communicator);
    communicator = second_communicator;
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        std::fprintf(stderr, "first communicator destruction failed: %s\n",
                     nano_nccl_get_last_error());
        nano_nccl_destroy_communicator(communicator);
        return EXIT_FAILURE;
    }
    nano_nccl_communicator_t* replacement_communicator = nullptr;
    status = nano_nccl_create_mpi_subgroup_communicator(
        &config, &replacement_communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "communicator creation after partial teardown failed: %s\n",
                     nano_nccl_get_last_error());
        nano_nccl_destroy_communicator(communicator);
        return EXIT_FAILURE;
    }
    status = nano_nccl_destroy_communicator(replacement_communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        std::fprintf(stderr, "replacement communicator destruction failed: %s\n",
                     nano_nccl_get_last_error());
        nano_nccl_destroy_communicator(communicator);
        return EXIT_FAILURE;
    }
    int channels = 0;
    int global_rank_count = 0;
    nano_nccl_transport_t actual_transport = NANO_NCCL_TRANSPORT_AUTO;
    bool ok = nano_nccl_mpi_channel_count(&channels) == NANO_NCCL_STATUS_SUCCESS &&
              nano_nccl_global_rank_count(communicator, &global_rank_count) ==
                  NANO_NCCL_STATUS_SUCCESS &&
              nano_nccl_transport(communicator, &actual_transport) ==
                  NANO_NCCL_STATUS_SUCCESS &&
              channels == 4 &&
              (requested_transport == NANO_NCCL_TRANSPORT_AUTO
                   ? (actual_transport == NANO_NCCL_TRANSPORT_P2P ||
                      actual_transport == NANO_NCCL_TRANSPORT_SHM ||
                      actual_transport == NANO_NCCL_TRANSPORT_MIXED)
                   : actual_transport == expected_transport);
    for (int edge = 0; edge < global_rank_count && ok; ++edge) {
        nano_nccl_transport_t edge_transport = NANO_NCCL_TRANSPORT_AUTO;
        ok = nano_nccl_edge_transport(communicator, edge, &edge_transport) ==
                 NANO_NCCL_STATUS_SUCCESS &&
             (requested_transport == NANO_NCCL_TRANSPORT_AUTO
                  ? (edge_transport == NANO_NCCL_TRANSPORT_P2P ||
                     edge_transport == NANO_NCCL_TRANSPORT_SHM)
                  : edge_transport == expected_transport);
    }
    nano_nccl_transport_t invalid_edge_transport = NANO_NCCL_TRANSPORT_AUTO;
    ok = nano_nccl_edge_transport(communicator, -1,
                                  &invalid_edge_transport) ==
             NANO_NCCL_STATUS_INVALID_ARGUMENT &&
         ok;
    ok = ok &&
              run_case<float>(communicator, rank, global_rank_count,
                              NANO_NCCL_DTYPE_FLOAT) &&
              run_case<__nv_bfloat16>(communicator, rank, global_rank_count,
                                      NANO_NCCL_DTYPE_BFLOAT16);
    if (actual_transport == NANO_NCCL_TRANSPORT_SOCKET && rank == 0) {
        // A rank may begin communicator destruction as soon as its last GPU
        // work is complete. Its peers must not observe that normal close as an
        // asynchronous transport failure while entering their own teardown.
        std::this_thread::sleep_for(std::chrono::seconds(1));
        status = nano_nccl_check_async_error(communicator);
        if (status != NANO_NCCL_STATUS_SUCCESS) {
            std::fprintf(stderr, "staggered teardown latched an async error: %s\n",
                         nano_nccl_get_last_error());
            ok = false;
        }
    }
    status = nano_nccl_destroy_communicator(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) {
        std::fprintf(stderr, "communicator destruction failed: %s\n",
                     nano_nccl_get_last_error());
        ok = false;
    }
    if (ok) {
        std::printf("rank=%d transport=%d mpi_c_api collectives OK\n", rank,
                    actual_transport);
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
