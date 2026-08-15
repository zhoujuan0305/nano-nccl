#pragma once

#include "transport/rdma/rdma_protocol.h"

#include <cstdint>

struct ibv_qp;
struct ibv_cq;
struct ibv_pd;
struct ibv_context;

namespace nano_nccl::transport::rdma {

class RdmaEndpoint;

// RAII 拥有一个 ibv_qp* + 独立的 ibv_cq*。一个 (edge, channel) 对应一个 RdmaQp。
// wr_id 与 slot 当前为同一概念（恒等映射），保留两层命名以便后续扩展为
// slot 高位编码 channel 时只改实现、不改调用点。
class RdmaQp {
public:
    static constexpr std::uint64_t slot_to_wr_id(std::uint64_t slot) noexcept {
        return slot;
    }
    static constexpr std::uint64_t wr_id_to_slot(std::uint64_t wr_id) noexcept {
        return wr_id;
    }

    RdmaQp() = default;
    ~RdmaQp();

    RdmaQp(const RdmaQp&) = delete;
    RdmaQp& operator=(const RdmaQp&) = delete;
    RdmaQp(RdmaQp&& other) noexcept;
    RdmaQp& operator=(RdmaQp&& other) noexcept;

    // CREATE + INIT：allocate_cq 由 endpoint 提供，所有权转移到 RdmaQp。
    static RdmaQp create_init(RdmaEndpoint& endpoint, int send_wr, int recv_wr);

    // RTR：用对端 qpn/psn/port_lid/gid 成心到 RTR。
    // path_mtu is min(local_active_mtu, remote.active_mtu) (ibv_mtu 1..5).
    void transition_to_rtr(const RdmaPeerInfo& remote,
                           std::uint16_t local_gid_index,
                           std::uint32_t local_active_mtu);

    // RTS：sq_psn 用 local_psn；remote_psn 已在 RTR 的 rq_psn 消费，此处忽略。
    void transition_to_rts(std::uint32_t local_psn, std::uint32_t remote_psn);

    ibv_qp* qp() const noexcept { return qp_; }
    ibv_cq* cq() const noexcept { return cq_; }
    // Actual max_inline_data granted by the HCA after create (may be 0).
    std::uint32_t max_inline_data() const noexcept { return max_inline_data_; }
    // 仅填 qpn；port_lid/gid_index/gid 由调用方从 RdmaEndpoint 抄入。
    RdmaPeerInfo local_info() const noexcept;

private:
    RdmaQp(ibv_qp* qp, ibv_cq* cq, std::uint32_t qpn,
           std::uint32_t max_inline_data) noexcept;

    void destroy() noexcept;

    ibv_qp* qp_ = nullptr;
    ibv_cq* cq_ = nullptr;
    std::uint32_t qpn_ = 0;
    std::uint32_t max_inline_data_ = 0;
};

}  // namespace nano_nccl::transport::rdma
