#pragma once

namespace nano_nccl::core {

// GPU -> NUMA 节点映射，运行时从 sysfs 读取，避免硬编码本机拓扑。
int gpu_numa_node(int gpu);

// 封装 libnuma numa_set_preferred；node<0 表示恢复默认策略。
void numa_set_prefer(int node);

// libnuma 可用性封装；不可用时返回 false。
bool numa_available_();

}  // namespace nano_nccl::core
