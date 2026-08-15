#include "transport/rdma/rdma_qp.h"
#include "transport/rdma/rdma_endpoint.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <infiniband/verbs.h>

namespace nano_nccl::transport::rdma {

namespace {

// 用 errno 当前值组装 "rdma qp <op> failed: <strerror>"。
// 适用于返回 nullptr 并设置 errno 的动词（ibv_create_qp）。
std::runtime_error qp_error(const char* op) {
    return std::runtime_error(std::string("rdma qp ") + op + " failed: " +
                              std::strerror(errno));
}

// 适用于直接返回 errno 的动词（ibv_modify_qp），返回值即错误码——
// 不能用全局 errno，因为这类动词不保证设置 errno。
std::runtime_error qp_rc_error(const char* op, int rc) {
    return std::runtime_error(std::string("rdma qp ") + op + " failed: " +
                              std::strerror(rc));
}

// CREATE 后立即转 INIT：开放本地写、远端写、远端读权限，端口 1，pkey 0。
void modify_qp_to_init(ibv_qp* qp) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ;
    int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
               IBV_QP_ACCESS_FLAGS;
    int rc = ibv_modify_qp(qp, &attr, mask);
    if (rc != 0) {
        throw qp_rc_error("ibv_modify_qp INIT", rc);
    }
}

}  // namespace

RdmaQp::~RdmaQp() { destroy(); }

RdmaQp::RdmaQp(RdmaQp&& other) noexcept
    : qp_(other.qp_),
      cq_(other.cq_),
      qpn_(other.qpn_),
      max_inline_data_(other.max_inline_data_) {
    other.qp_ = nullptr;
    other.cq_ = nullptr;
    other.qpn_ = 0;
    other.max_inline_data_ = 0;
}

RdmaQp& RdmaQp::operator=(RdmaQp&& other) noexcept {
    if (this != &other) {
        destroy();
        qp_ = other.qp_;
        cq_ = other.cq_;
        qpn_ = other.qpn_;
        max_inline_data_ = other.max_inline_data_;
        other.qp_ = nullptr;
        other.cq_ = nullptr;
        other.qpn_ = 0;
        other.max_inline_data_ = 0;
    }
    return *this;
}

RdmaQp::RdmaQp(ibv_qp* qp, ibv_cq* cq, std::uint32_t qpn,
               std::uint32_t max_inline_data) noexcept
    : qp_(qp), cq_(cq), qpn_(qpn), max_inline_data_(max_inline_data) {}

void RdmaQp::destroy() noexcept {
    // 销毁顺序与创建相反：先 qp 再 cq——qp 必须先于其关联的 CQ 释放。
    if (qp_ != nullptr) {
        ibv_destroy_qp(qp_);
    }
    if (cq_ != nullptr) {
        ibv_destroy_cq(cq_);
    }
    qp_ = nullptr;
    cq_ = nullptr;
    qpn_ = 0;
    max_inline_data_ = 0;
}

RdmaQp RdmaQp::create_init(RdmaEndpoint& endpoint, int send_wr, int recv_wr) {
    // Shared CQ must hold both SQ and RQ completions (WriteCts posts CTS SENDs
    // and IMM RECVs on the same CQ). Depth is send_wr + recv_wr.
    // sq_sig_all=0：仅带 IBV_SEND_SIGNALED 的 SEND 进 CQ，减少 multi-flight CQ 税。
    // allocate_cq 返回 unique_ptr，这里 .release() 把 raw 所有权转交给 RdmaQp。
    const int cq_depth = send_wr + recv_wr;
    ibv_cq* cq = endpoint.allocate_cq(cq_depth).release();

    ibv_qp_init_attr init_attr{};
    init_attr.qp_context = nullptr;
    init_attr.send_cq = cq;
    init_attr.recv_cq = cq;
    init_attr.srq = nullptr;
    init_attr.cap.max_send_wr = static_cast<std::uint32_t>(send_wr);
    init_attr.cap.max_recv_wr = static_cast<std::uint32_t>(recv_wr);
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 1;
    // Request enough inline room for a 32B RdmaCtsSlot (and a small margin).
    init_attr.cap.max_inline_data = 64;
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.sq_sig_all = 0;

    ibv_qp* qp = ibv_create_qp(endpoint.pd(), &init_attr);
    if (qp == nullptr) {
        // create_qp 失败需要手动释放已分配的 cq。
        ibv_destroy_cq(cq);
        throw qp_error("ibv_create_qp");
    }
    // Driver overwrites cap with the actual granted values.
    const std::uint32_t max_inline = init_attr.cap.max_inline_data;

    try {
        modify_qp_to_init(qp);
    } catch (...) {
        // INIT 失败回滚：销毁已创建的 qp 与 cq。
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        throw;
    }

    return RdmaQp(qp, cq, qp->qp_num, max_inline);
}

void RdmaQp::transition_to_rtr(const RdmaPeerInfo& remote,
                               std::uint16_t local_gid_index,
                               std::uint32_t local_active_mtu) {
    if (local_gid_index > 0xff) {
        throw std::runtime_error("local RDMA GID index exceeds verbs range");
    }

    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    // NCCL uses min(remote advertised mtu, local port.active_mtu).
    attr.path_mtu = static_cast<ibv_mtu>(
        negotiate_path_mtu(local_active_mtu, remote.active_mtu));
    attr.dest_qp_num = remote.qpn;
    attr.rq_psn = remote.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = remote.port_lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    // brief 记 IBV_RATE_PORT_WIDTH，但本机 verbs.h 无该枚举；取 IBV_RATE_MAX(=0)
    // 表示不限速、按端口原生带宽发送，等价语义。
    attr.ah_attr.static_rate = IBV_RATE_MAX;
    attr.ah_attr.port_num = 1;
    // RoCE 走 GRH：dgid 来自 peer，sgid_index 必须选择本端 source GID。
    attr.ah_attr.is_global = 1;
    std::memcpy(&attr.ah_attr.grh.dgid, remote.gid, 16);
    attr.ah_attr.grh.sgid_index = static_cast<std::uint8_t>(local_gid_index);
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.traffic_class = 0;

    int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
               IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
               IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    int rc = ibv_modify_qp(qp_, &attr, mask);
    if (rc != 0) {
        throw qp_rc_error("ibv_modify_qp RTR", rc);
    }
}

void RdmaQp::transition_to_rts(std::uint32_t local_psn,
                               std::uint32_t remote_psn) {
    // remote_psn 已在 RTR 的 rq_psn 中消费；RTS 仅用 local_psn 作为 sq_psn。
    (void)remote_psn;

    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = local_psn;
    attr.max_rd_atomic = 1;

    int mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
               IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
               IBV_QP_MAX_QP_RD_ATOMIC;
    int rc = ibv_modify_qp(qp_, &attr, mask);
    if (rc != 0) {
        throw qp_rc_error("ibv_modify_qp RTS", rc);
    }
}

RdmaPeerInfo RdmaQp::local_info() const noexcept {
    RdmaPeerInfo info{};
    info.qpn = qpn_;
    // port_lid/gid_index/gid 由调用方从 RdmaEndpoint 抄入——RdmaQp 不持有 endpoint 引用。
    return info;
}

}  // namespace nano_nccl::transport::rdma
