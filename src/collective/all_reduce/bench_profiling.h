#pragma once

#if defined(NANO_NCCL_ENABLE_BENCH_PROFILING)

#include "core/buffer.h"

#include <cstddef>
#include <string>
#include <utility>

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

namespace nano_nccl::collective::all_reduce::bench_profiling {

inline std::string all_reduce_size_range_name(std::size_t bytes) {
    return "all_reduce size=" + std::to_string(bytes) + "B";
}

inline std::string all_reduce_iteration_range_name(std::size_t bytes, int iteration) {
    return all_reduce_size_range_name(bytes) + " iteration=" + std::to_string(iteration);
}

class NvtxRange {
public:
    explicit NvtxRange(const std::string& name) { nvtxRangePushA(name.c_str()); }
    ~NvtxRange() { nvtxRangePop(); }
};

class ProfilerSession {
public:
    ProfilerSession() {
        CUDA_CHECK_THROW(cudaProfilerStart());
        started_ = true;
    }

    ~ProfilerSession() {
        if (started_) cudaProfilerStop();
    }

    void stop() {
        if (!started_) return;
        cudaError_t status = cudaProfilerStop();
        started_ = false;
        CUDA_CHECK_THROW(status);
    }

private:
    bool started_ = false;
};

}  // namespace nano_nccl::collective::all_reduce::bench_profiling

#endif
