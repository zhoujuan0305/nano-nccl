#pragma once

#include <cstdint>

namespace nano_nccl::transport::rdma {

// 定长 POD，通过 TCP bootstrap fd 在环上邻居间交换一次。
// send/recv 双边语义下不携带 rkey/addr：receiver FIFO 地址不暴露给 sender。
struct RdmaPeerInfo {
    std::uint32_t qpn = 0;        // 本端 QPN
    std::uint32_t psn = 0;        // 本端起始 PSN
    std::uint16_t port_lid = 0;   // RoCE/IB LID
    std::uint16_t gid_index = 0;  // 用哪个 GID
    std::uint8_t  gid[16]{};      // RoCEv2 用 IPv6 GID
};

static_assert(sizeof(RdmaPeerInfo) == 28);

}  // namespace nano_nccl::transport::rdma