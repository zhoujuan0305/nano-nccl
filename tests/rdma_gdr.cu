// Register CUDA device memory with the local HCA (GPUDirect / peermem).
//
// Env: NANO_NCCL_RDMA_IFNAME, optional NANO_NCCL_RDMA_GID_INDEX.
// Exit: 0 PASS, 1 FAIL, 77 SKIP.

#include "transport/rdma/rdma_endpoint.h"
#include "transport/rdma/rdma_gdr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <infiniband/verbs.h>

namespace {

using nano_nccl::transport::rdma::RdmaEndpoint;
using nano_nccl::transport::rdma::RdmaMemoryPlacement;
using nano_nccl::transport::rdma::RdmaRegisteredMemory;
using nano_nccl::transport::rdma::parse_rdma_memory_placement_env;

#define CUDA_OR_THROW(expr)                                                    \
    do {                                                                       \
        cudaError_t err = (expr);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string(#expr) + ": " +               \
                                     cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

}  // namespace

int main() {
    try {
        setenv("NANO_NCCL_RDMA_GDR", "0", 1);
        if (parse_rdma_memory_placement_env() != RdmaMemoryPlacement::HostPin) {
            std::fprintf(stderr, "GDR=0 must parse as HostPin\n");
            return 1;
        }
        setenv("NANO_NCCL_RDMA_GDR", "1", 1);
        if (parse_rdma_memory_placement_env() !=
            RdmaMemoryPlacement::GpuDirect) {
            std::fprintf(stderr, "GDR=1 must parse as GpuDirect\n");
            return 1;
        }

        if (std::getenv("NANO_NCCL_RDMA_IFNAME") == nullptr ||
            std::getenv("NANO_NCCL_RDMA_IFNAME")[0] == '\0') {
            std::fprintf(stderr, "SKIP: NANO_NCCL_RDMA_IFNAME unset\n");
            return 77;
        }

        RdmaEndpoint endpoint = RdmaEndpoint::create_from_environment();
        constexpr std::size_t kBytes = 4096;
        void* device = nullptr;
        CUDA_OR_THROW(cudaMalloc(&device, kBytes));
        CUDA_OR_THROW(cudaMemset(device, 0x5a, kBytes));
        CUDA_OR_THROW(cudaDeviceSynchronize());

        try {
            RdmaRegisteredMemory mem = RdmaRegisteredMemory::register_device(
                endpoint.pd(), device, kBytes);
            if (mem.mr() == nullptr || mem.rkey() == 0 || !mem.is_device()) {
                std::fprintf(stderr, "device MR incomplete\n");
                CUDA_OR_THROW(cudaFree(device));
                return 1;
            }
            if (mem.addr() != device) {
                std::fprintf(stderr, "device MR addr mismatch\n");
                CUDA_OR_THROW(cudaFree(device));
                return 1;
            }
        } catch (const std::exception& ex) {
            CUDA_OR_THROW(cudaFree(device));
            const char* msg = ex.what();
            if (std::strstr(msg, "unavailable") != nullptr) {
                std::fprintf(stderr, "SKIP: %s\n", msg);
                return 77;
            }
            std::fprintf(stderr, "register_device: %s\n", msg);
            return 1;
        }
        CUDA_OR_THROW(cudaFree(device));
        std::printf("rdma_gdr=PASS\n");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "rdma_gdr: %s\n", ex.what());
        return 1;
    }
}
