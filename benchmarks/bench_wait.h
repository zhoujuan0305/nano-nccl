#pragma once

#include <stdexcept>
#include <string>
#include <thread>

#include <cuda_runtime.h>

namespace nano_nccl::benchmarks {

enum class WaitMode { Query, Sync };

inline const char* wait_mode_name(WaitMode mode) {
    return mode == WaitMode::Query ? "query" : "sync";
}

inline bool parse_wait_mode(const char* value, WaitMode* mode) {
    const std::string name(value);
    if (name == "query") {
        *mode = WaitMode::Query;
        return true;
    }
    if (name == "sync") {
        *mode = WaitMode::Sync;
        return true;
    }
    return false;
}

// Match nccl-tests -z 2 for a single local stream: query until complete,
// check asynchronous transport failures, then yield while the stream is busy.
template <typename CheckAsyncError>
void wait_for_stream(cudaStream_t stream, WaitMode mode,
                     CheckAsyncError&& check_async_error) {
    if (mode == WaitMode::Sync) {
        const cudaError_t status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamSynchronize: ") +
                                     cudaGetErrorString(status));
        }
    } else {
        for (;;) {
            const cudaError_t status = cudaStreamQuery(stream);
            if (status == cudaSuccess) break;
            if (status != cudaErrorNotReady) {
                throw std::runtime_error(std::string("cudaStreamQuery: ") +
                                         cudaGetErrorString(status));
            }
            check_async_error();
            std::this_thread::yield();
        }
    }
    check_async_error();
}

}  // namespace nano_nccl::benchmarks
