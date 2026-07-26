#pragma once

#include <cstdint>
#include <memory>

struct ibv_context;
struct ibv_pd;
struct ibv_cq;

namespace nano_nccl::transport::rdma {

// RAII 拥有一个 ibv_context* + ibv_pd*。create_from_environment 按
// NANO_NCCL_RDMA_IFNAME 选 HCA，按 NANO_NCCL_RDMA_GID_INDEX 选 GID（默认 0）。
class RdmaEndpoint {
public:
    RdmaEndpoint() = default;
    ~RdmaEndpoint();

    RdmaEndpoint(const RdmaEndpoint&) = delete;
    RdmaEndpoint& operator=(const RdmaEndpoint&) = delete;
    RdmaEndpoint(RdmaEndpoint&& other) noexcept;
    RdmaEndpoint& operator=(RdmaEndpoint&& other) noexcept;

    static RdmaEndpoint create_from_environment();

    ibv_context* context() const noexcept { return context_; }
    ibv_pd* pd() const noexcept { return pd_; }
    std::uint16_t port_lid() const noexcept { return port_lid_; }
    std::uint16_t gid_index() const noexcept { return gid_index_; }
    const std::uint8_t (&gid() const noexcept)[16];

    std::unique_ptr<struct ibv_cq, void(*)(struct ibv_cq*)> allocate_cq(int wr_depth);

private:
    RdmaEndpoint(ibv_context* ctx, ibv_pd* pd, std::uint16_t port_lid,
                 std::uint16_t gid_index, const std::uint8_t gid[16]) noexcept;

    void close() noexcept;

    ibv_context* context_ = nullptr;
    ibv_pd* pd_ = nullptr;
    std::uint16_t port_lid_ = 0;
    std::uint16_t gid_index_ = 0;
    std::uint8_t gid_[16]{};
};

}  // namespace nano_nccl::transport::rdma