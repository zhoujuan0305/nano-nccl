#include "nano_nccl/all_reduce.h"
#include "transport/p2p/p2p_fifo.h"

#include <cstddef>
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

constexpr std::size_t kPacked16ScalarFallbackBytes = 262146;

bool report_results(const std::vector<nano_nccl::BenchResult>& results,
                    nano_nccl::TransportKind requested_transport,
                    nano_nccl::TransportKind expected_transport,
                    int* total_wrong) {
    for (const auto& result : results) {
        std::printf("correctness algo=%s dtype=%s redop=%s transport=%s bytes=%zu wrong=%d max_abs=%g\n",
                    result.algo.c_str(), nano_nccl::dtype_name(result.dtype),
                    nano_nccl::redop_name(result.redop),
                    nano_nccl::transport_name(result.transport), result.bytes,
                    result.wrong, result.max_abs_error);
        if (result.transport != expected_transport) {
            std::fprintf(stderr,
                         "correctness=FAIL requested transport=%s expected transport=%s "
                         "resolved transport=%s\n",
                         nano_nccl::transport_name(requested_transport),
                         nano_nccl::transport_name(expected_transport),
                         nano_nccl::transport_name(result.transport));
            return false;
        }
        *total_wrong += result.wrong;
    }
    return true;
}

int run_dtype(nano_nccl::BenchConfig* config, nano_nccl::DType dtype,
              nano_nccl::RedOp redop,
              nano_nccl::TransportKind requested_transport,
              nano_nccl::TransportKind expected_transport, int* total_wrong) {
    config->dtype = dtype;
    config->redop = redop;
    config->transport = requested_transport;
    std::vector<nano_nccl::BenchResult> results;
    int rc = nano_nccl::run_all_reduce_bench(*config, &results);
    if (rc != 0 || results.empty() ||
        !report_results(results, requested_transport, expected_transport,
                        total_wrong)) {
        return 1;
    }
    return 0;
}

int run_bf16_if_supported(nano_nccl::BenchConfig* config,
                           nano_nccl::RedOp redop,
                           nano_nccl::TransportKind requested_transport,
                          nano_nccl::TransportKind expected_transport,
                          int* total_wrong) {
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
        return run_dtype(config, nano_nccl::DType::BFloat16, redop,
                          requested_transport, expected_transport, total_wrong);
    }

    std::printf(
        "correctness skip dtype=bf16 transport=%s reason=device%d (%s) is SM%d%d; "
        "bf16 requires SM80+\n",
        nano_nccl::transport_name(requested_transport),
        first_bf16_unsupported_device,
        first_bf16_unsupported_props.name,
        first_bf16_unsupported_props.major,
        first_bf16_unsupported_props.minor);
    return 0;
}

int run_dtype_matrix(nano_nccl::BenchConfig* config,
                     nano_nccl::TransportKind requested_transport,
                     nano_nccl::TransportKind expected_transport,
                     int* total_wrong) {
    const nano_nccl::DType dtypes[] = {
        nano_nccl::DType::Float,
        nano_nccl::DType::Float16,
    };
    const nano_nccl::RedOp redops[] = {
        nano_nccl::RedOp::Sum,
        nano_nccl::RedOp::Avg,
        nano_nccl::RedOp::Max,
        nano_nccl::RedOp::Min,
    };
    for (nano_nccl::RedOp redop : redops) {
        for (nano_nccl::DType dtype : dtypes) {
            if (run_dtype(config, dtype, redop, requested_transport,
                          expected_transport, total_wrong) != 0) {
                return 1;
            }
        }
        if (run_bf16_if_supported(config, redop, requested_transport,
                                  expected_transport, total_wrong) != 0) {
            return 1;
        }
    }
    return 0;
}

int run_packed16_scalar_matrix(nano_nccl::BenchConfig* config,
                               nano_nccl::TransportKind requested_transport,
                               nano_nccl::TransportKind expected_transport,
                               int* total_wrong) {
    const nano_nccl::RedOp redops[] = {
        nano_nccl::RedOp::Sum,
        nano_nccl::RedOp::Avg,
        nano_nccl::RedOp::Max,
        nano_nccl::RedOp::Min,
    };
    for (nano_nccl::RedOp redop : redops) {
        if (run_dtype(config, nano_nccl::DType::Float16, redop,
                      requested_transport, expected_transport, total_wrong) != 0 ||
            run_bf16_if_supported(config, redop, requested_transport,
                                  expected_transport, total_wrong) != 0) {
            return 1;
        }
    }
    return 0;
}

int run_transport_matrix(nano_nccl::BenchConfig* config,
                         nano_nccl::TransportKind requested_transport,
                         nano_nccl::TransportKind expected_transport,
                         int* total_wrong) {
    if (run_dtype_matrix(config, requested_transport, expected_transport,
                         total_wrong) != 0) {
        return 1;
    }

    nano_nccl::BenchConfig non_vector_config = *config;
    // 该大小保留 16-bit dtype 的整除性，但使元素数不满足 packed-16 向量宽度。
    non_vector_config.min_bytes = kPacked16ScalarFallbackBytes;
    non_vector_config.max_bytes = kPacked16ScalarFallbackBytes;
    non_vector_config.factor = 2;
    return run_packed16_scalar_matrix(&non_vector_config, requested_transport,
                                      expected_transport, total_wrong);
}

int main() {
    nano_nccl::BenchConfig config;
    config.algo = "ring_simple";
    config.min_bytes = 262144;
    config.max_bytes = 67108864;
    config.factor = 4;
    config.warmup_iters = 1;
    config.iters = 1;

    int total_wrong = 0;
    if (run_transport_matrix(&config, nano_nccl::TransportKind::Shm,
                             nano_nccl::TransportKind::Shm,
                             &total_wrong) != 0) {
        return 1;
    }

    nano_nccl::TransportKind expected_auto =
        nano_nccl::transport::p2p::resolve_ring_transport(
            nano_nccl::TransportKind::Auto)
            .resolved_kind();
    if (run_transport_matrix(&config, nano_nccl::TransportKind::Auto,
                             expected_auto, &total_wrong) != 0) {
        return 1;
    }

    if (nano_nccl::transport::p2p::p2p_ring_available()) {
        if (run_transport_matrix(&config, nano_nccl::TransportKind::P2p,
                                 nano_nccl::TransportKind::P2p,
                                 &total_wrong) != 0) {
            return 1;
        }
    } else {
        std::printf(
            "correctness skip transport=p2p reason=full directed ring P2P "
            "support is unavailable\n");
    }

    if (total_wrong != 0) {
        std::fprintf(stderr, "correctness=FAIL total_wrong=%d\n", total_wrong);
        return 1;
    }
    std::puts("correctness=PASS");
    return 0;
}
