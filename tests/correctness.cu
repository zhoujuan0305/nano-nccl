#include "nano_nccl/all_reduce.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

// 独立正确性测试：覆盖合同消息大小，只校验 #wrong=0，不报告性能。
int main() {
    nano_nccl::BenchConfig config;
    config.algo = "ring_simple";
    config.min_bytes = 262144;
    config.max_bytes = 67108864;
    config.factor = 4;
    config.warmup_iters = 1;
    config.iters = 1;

    std::vector<nano_nccl::BenchResult> results;
    int rc = nano_nccl::run_all_reduce_bench(config, &results);
    if (rc != 0) {
        std::fprintf(stderr, "correctness run failed rc=%d\n", rc);
        return 1;
    }

    int total_wrong = 0;
    for (const auto& r : results) {
        std::printf("correctness algo=%s bytes=%zu wrong=%d max_abs=%g\n",
                    r.algo.c_str(), r.bytes, r.wrong, r.max_abs_error);
        total_wrong += r.wrong;
    }

    if (total_wrong != 0) {
        std::fprintf(stderr, "correctness=FAIL total_wrong=%d\n", total_wrong);
        return 1;
    }
    std::puts("correctness=PASS");
    return 0;
}
