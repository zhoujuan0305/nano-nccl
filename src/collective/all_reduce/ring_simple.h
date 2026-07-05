#pragma once

#include "nano_nccl/types.h"

#include <vector>

namespace nano_nccl::collective::all_reduce {

// ring_simple 路径的 benchmark 入口：当前实现托管在 ring_simple.cu 内的
// AllReduceRunner（匿名 namespace）中。nano_nccl::run_all_reduce_bench 委托到这里。
int run_ring_simple_bench(const BenchConfig& config,
                          std::vector<BenchResult>* results);

}  // namespace nano_nccl::collective::all_reduce
