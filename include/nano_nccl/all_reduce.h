#pragma once

#include "nano_nccl/types.h"

#include <vector>

namespace nano_nccl {

// 公共 API：执行 All Reduce benchmark，按 config.algo 选择路径，逐 size 输出结果。
int run_all_reduce_bench(const BenchConfig& config,
                         std::vector<BenchResult>* results);

}  // namespace nano_nccl
