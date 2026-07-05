#include "core/numa.h"

#include <cstdio>

#include <cuda_runtime.h>
#include <numa.h>

namespace nano_nccl::core {

// 运行时从 /sys/bus/pci/devices/<domain:bus:device.0>/numa_node 读取 GPU 所在
// NUMA 节点。读失败或节点为负（无 NUMA）时退回 0，保证单 NUMA / 容器环境也能跑。
// cudaDeviceProp 的 pci* 字段是十进制 int，sysfs 路径用十六进制，故用 %x 格式化。
int gpu_numa_node(int gpu) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, gpu) != cudaSuccess) {
        return 0;
    }
    char path[256];
    std::snprintf(path, sizeof(path),
                  "/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node",
                  prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    std::FILE* f = std::fopen(path, "r");
    if (f == nullptr) {
        return 0;
    }
    int node = 0;
    if (std::fscanf(f, "%d", &node) != 1) {
        node = 0;
    }
    std::fclose(f);
    if (node < 0) {
        node = 0;
    }
    return node;
}

void numa_set_prefer(int node) {
    if (numa_available() >= 0) {
        numa_set_preferred(node);
    }
}

bool numa_available_() {
    return numa_available() >= 0;
}

}  // namespace nano_nccl::core
