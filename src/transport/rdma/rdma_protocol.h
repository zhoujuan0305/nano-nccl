#pragma once

#include <cstdint>

namespace nano_nccl::transport::rdma {

// Clear-To-Send slot published by the receiver into the sender-local CTS FIFO.
// Sender polls ready; after a successful data WRITE post (raddr/rkey already
// snapshotted), sender clears ready so a next-round CTS can reuse the slot.
struct RdmaCtsSlot {
    std::uint64_t raddr = 0;
    std::uint32_t rkey = 0;
    std::uint32_t nbytes = 0;
    std::uint64_t step_tag = 0;
    std::uint32_t ready = 0;
    std::uint32_t reserved = 0;
};
static_assert(sizeof(RdmaCtsSlot) == 32);
static_assert(alignof(RdmaCtsSlot) == 8);

// 定长 POD，通过 TCP bootstrap fd 在环上邻居间交换一次。
// QP 字段用于 RC 建链；recv/cts FIFO 字段供 WRITE+CTS 数据面使用。
// SEND/RECV 模式可将 FIFO/CTS 字段保持为 0。
// Layout: 4B pad after gid[16] before recv_fifo_addr → sizeof 64 (not 60).
struct RdmaPeerInfo {
    std::uint32_t qpn = 0;        // 本端 QPN
    std::uint32_t psn = 0;        // 本端起始 PSN
    std::uint16_t port_lid = 0;   // RoCE/IB LID
    std::uint16_t gid_index = 0;  // 用哪个 GID
    std::uint8_t  gid[16]{};      // RoCEv2 用 IPv6 GID
    std::uint64_t recv_fifo_addr = 0;
    std::uint32_t recv_fifo_rkey = 0;
    std::uint32_t recv_fifo_bytes = 0;
    std::uint64_t cts_fifo_addr = 0;
    std::uint32_t cts_fifo_rkey = 0;
    std::uint32_t cts_slot_count = 0;
};

static_assert(sizeof(RdmaPeerInfo) == 64);

}  // namespace nano_nccl::transport::rdma
