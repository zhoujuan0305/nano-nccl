#pragma once

#include <cstddef>
#include <cstdint>

struct ibv_pd;
struct ibv_mr;

namespace nano_nccl::transport::rdma {

enum class RdmaMemoryPlacement { HostPin, GpuDirect };

// NANO_NCCL_RDMA_GDR unset/""/0/false/off → HostPin;
// 1/true/on → GpuDirect; otherwise throws.
RdmaMemoryPlacement parse_rdma_memory_placement_env();

// Owns one ibv_mr. Host-pin uses ibv_reg_mr on CPU memory. GpuDirect
// registers CUDA device memory (peermem and/or DMA-BUF).
class RdmaRegisteredMemory {
public:
    RdmaRegisteredMemory() = default;
    ~RdmaRegisteredMemory();

    RdmaRegisteredMemory(const RdmaRegisteredMemory&) = delete;
    RdmaRegisteredMemory& operator=(const RdmaRegisteredMemory&) = delete;
    RdmaRegisteredMemory(RdmaRegisteredMemory&& other) noexcept;
    RdmaRegisteredMemory& operator=(RdmaRegisteredMemory&& other) noexcept;

    static RdmaRegisteredMemory register_host(ibv_pd* pd, void* addr,
                                              std::size_t bytes);
    static RdmaRegisteredMemory register_device(ibv_pd* pd, void* device_addr,
                                                std::size_t bytes);

    ibv_mr* mr() const noexcept { return mr_; }
    void* addr() const noexcept { return addr_; }
    std::size_t bytes() const noexcept { return bytes_; }
    std::uint32_t lkey() const noexcept;
    std::uint32_t rkey() const noexcept;
    bool is_device() const noexcept { return device_; }

private:
    RdmaRegisteredMemory(ibv_mr* mr, void* addr, std::size_t bytes,
                         bool device) noexcept;

    void reset() noexcept;

    ibv_mr* mr_ = nullptr;
    void* addr_ = nullptr;
    std::size_t bytes_ = 0;
    bool device_ = false;
};

}  // namespace nano_nccl::transport::rdma
