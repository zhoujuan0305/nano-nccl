#include "nano_nccl/all_reduce.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err__));                \
            return 1;                                                         \
        }                                                                     \
    } while (0)

// 独立正确性测试：覆盖合同消息大小，只校验 #wrong=0，不报告性能。
int main() {
    nano_nccl::BenchConfig config;
    config.algo = "ring_simple";
    config.min_bytes = 262144;
    config.max_bytes = 67108864;
    config.factor = 4;
    config.warmup_iters = 1;
    config.iters = 1;

    int total_wrong = 0;
    const nano_nccl::DType dtypes[] = {
        nano_nccl::DType::Float,
        nano_nccl::DType::Float16,
    };
    std::vector<nano_nccl::BenchResult> results;
    for (nano_nccl::DType dtype : dtypes) {
        config.dtype = dtype;
        results.clear();
        int rc = nano_nccl::run_all_reduce_bench(config, &results);
        if (rc != 0) return 1;
        for (const auto& result : results) {
            std::printf("correctness algo=%s dtype=%s bytes=%zu wrong=%d max_abs=%g\n",
                        result.algo.c_str(), nano_nccl::dtype_name(result.dtype),
                        result.bytes, result.wrong, result.max_abs_error);
            total_wrong += result.wrong;
        }
    }

    int first_bf16_unsupported_device = -1;
    cudaDeviceProp first_bf16_unsupported_props{};
    for (int device = 0; device < nano_nccl::kRanks; ++device) {
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        if (props.major < 8 && first_bf16_unsupported_device < 0) {
            first_bf16_unsupported_device = device;
            first_bf16_unsupported_props = props;
        }
    }
    if (first_bf16_unsupported_device < 0) {
        config.dtype = nano_nccl::DType::BFloat16;
        results.clear();
        int rc = nano_nccl::run_all_reduce_bench(config, &results);
        if (rc != 0) return 1;
        for (const auto& result : results) {
            std::printf("correctness algo=%s dtype=%s bytes=%zu wrong=%d max_abs=%g\n",
                        result.algo.c_str(), nano_nccl::dtype_name(result.dtype),
                        result.bytes, result.wrong, result.max_abs_error);
            total_wrong += result.wrong;
        }
    } else {
        std::printf(
            "correctness skip dtype=bf16 reason=device%d (%s) is SM%d%d; "
            "bf16 requires SM80+\n",
            first_bf16_unsupported_device, first_bf16_unsupported_props.name,
            first_bf16_unsupported_props.major,
            first_bf16_unsupported_props.minor);
    }

    if (total_wrong != 0) {
        std::fprintf(stderr, "correctness=FAIL total_wrong=%d\n", total_wrong);
        return 1;
    }
    std::puts("correctness=PASS");
    return 0;
}
