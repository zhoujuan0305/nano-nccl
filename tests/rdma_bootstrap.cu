// End-to-end RDMA bootstrap smoke: two in-process RC QPs on the same endpoint
// exchange RdmaPeerInfo over a localhost TCP. Sender posts one IBV_WR_SEND
// carrying "RDMA_BOOT_OK", receiver posts one pre-recv WQE. After both CQs
// report success the payload is memcompared.
//
// Inputs (env, runtime values only — never hard-coded):
//   NANO_NCCL_RDMA_IFNAME=<rdma-interface>     RoCE/IB port-tied netdev
//   NANO_NCCL_SOCKET_IFNAME=<socket-interface> IPv4 listener for the info swap
// Optional:
//   NANO_NCCL_RDMA_GID_INDEX=<n>               GID table index (default 0)
//
// Exit codes:
//   0  PASS
//   1  FAIL (with `rdma_bootstrap=FAIL <what>` on stderr)
//   77 SKIP (no HCA / NANO_NCCL_RDMA_IFNAME unset; ctest BMI skip)

#include "transport/rdma/rdma_endpoint.h"
#include "transport/rdma/rdma_protocol.h"
#include "transport/rdma/rdma_qp.h"
#include "transport/socket/socket_endpoint.h"
#include "transport/socket/socket_protocol.h"

#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <infiniband/verbs.h>
#include <unistd.h>

namespace {

using nano_nccl::transport::rdma::RdmaEndpoint;
using nano_nccl::transport::rdma::RdmaPeerInfo;
using nano_nccl::transport::rdma::RdmaQp;
using nano_nccl::transport::socket::SocketConnection;
using nano_nccl::transport::socket::SocketEndpoint;
using nano_nccl::transport::socket::SocketHello;
using nano_nccl::transport::socket::recv_all;
using nano_nccl::transport::socket::send_all;

// Build the local side's RdmaPeerInfo: qpn comes from the QP; port LID /
// gid index / gid come from the shared endpoint (per brief).
void build_local_info(const RdmaEndpoint& endpoint, const RdmaQp& qp,
                      RdmaPeerInfo* out) {
    *out = qp.local_info();           // only fills qpn
    out->port_lid = endpoint.port_lid();
    out->gid_index = endpoint.gid_index();
    std::memcpy(out->gid, endpoint.gid(), 16);
}

class MrOwner {
public:
    explicit MrOwner(ibv_mr* mr = nullptr) : mr_(mr) {}
    ~MrOwner() { if (mr_ != nullptr) ibv_dereg_mr(mr_); }

    MrOwner(const MrOwner&) = delete;
    MrOwner& operator=(const MrOwner&) = delete;

    ibv_mr* get() const noexcept { return mr_; }
    ibv_mr* operator->() const noexcept { return mr_; }

private:
    ibv_mr* mr_ = nullptr;
};

void poll_completion_or_throw(ibv_cq* cq, ibv_wc* wc, const char* side) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (true) {
        const int count = ibv_poll_cq(cq, 1, wc);
        if (count < 0) {
            throw std::runtime_error(std::string("ibv_poll_cq (") + side + ") failed");
        }
        if (count != 0) return;
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error(std::string("ibv_poll_cq (") + side + ") timed out");
        }
    }
}

}  // namespace

int main() {
    // ctest BMI skip when no RDMA HCA is reachable.
    const char* rdma_ifname = std::getenv("NANO_NCCL_RDMA_IFNAME");
    if (rdma_ifname == nullptr || rdma_ifname[0] == '\0') {
        std::printf("rdma_bootstrap=SKIP no RDMA HCA\n");
        return 77;
    }
    {
        int num_devices = 0;
        ibv_device** list = ibv_get_device_list(&num_devices);
        if (list == nullptr || num_devices <= 0) {
            if (list != nullptr) ibv_free_device_list(list);
            std::printf("rdma_bootstrap=SKIP no RDMA HCA\n");
            return 77;
        }
        ibv_free_device_list(list);
    }

    try {
        RdmaEndpoint endpoint = RdmaEndpoint::create_from_environment();
        // Two independent RC QPs on the same endpoint/HCA: A=sender, B=receiver.
        // Send CQ depth 8, recv CQ depth 8 — matches the brief's per-QP WQE budget.
        constexpr int kSendWr = 8;
        constexpr int kRecvWr = 8;
        RdmaQp qp_a = RdmaQp::create_init(endpoint, kSendWr, kRecvWr);
        RdmaQp qp_b = RdmaQp::create_init(endpoint, kSendWr, kRecvWr);
        if (qp_a.max_inline_data() < sizeof(nano_nccl::transport::rdma::RdmaCtsSlot) ||
            qp_b.max_inline_data() < sizeof(nano_nccl::transport::rdma::RdmaCtsSlot)) {
            throw std::runtime_error(
                "rdma_bootstrap: max_inline_data too small for RdmaCtsSlot");
        }

        // 16-byte payload (16 is mmap/max_inline friendly, comfortably holds
        // the 13-byte terminated C string "RDMA_BOOT_OK" + null).
        constexpr std::size_t kBufBytes = 16;
        constexpr char kPayload[] = "RDMA_BOOT_OK";
        constexpr std::size_t kPayloadLen = sizeof(kPayload);  // includes '\0'
        std::uint8_t send_buf[kBufBytes] = {};
        std::uint8_t recv_buf[kBufBytes] = {};
        std::memcpy(send_buf, kPayload, kPayloadLen);

        // Each side registers its own MR (send/recv double-sided; no rkey swap).
        constexpr int kMrAccess = IBV_ACCESS_LOCAL_WRITE |
                                  IBV_ACCESS_REMOTE_WRITE |
                                  IBV_ACCESS_REMOTE_READ;
        MrOwner send_mr(ibv_reg_mr(endpoint.pd(), send_buf, kBufBytes, kMrAccess));
        if (send_mr.get() == nullptr) {
            throw std::runtime_error(std::string("ibv_reg_mr send: ") +
                                     std::strerror(errno));
        }
        MrOwner recv_mr(ibv_reg_mr(endpoint.pd(), recv_buf, kBufBytes, kMrAccess));
        if (recv_mr.get() == nullptr) {
            throw std::runtime_error(std::string("ibv_reg_mr recv: ") +
                                     std::strerror(errno));
        }

        // Localhost TCP listener for the RdmaPeerInfo swap. The same
        // <socket-interface> listener accepts a connection from itself.
        SocketEndpoint listener = SocketEndpoint::create_from_environment();
        SocketConnection conn_a = listener.connect(
            listener.address(), SocketHello{0, 1, 0});
        SocketConnection conn_b = listener.accept();
        (void)listener.read_hello(conn_b);  // drain WireHello bytes
        const int fd_a = conn_a.fd();
        const int fd_b = conn_b.fd();

        // Swap RdmaPeerInfo each direction (64-byte fixed-layout POD).
        RdmaPeerInfo local_a{}, local_b{}, remote_a{}, remote_b{};
        build_local_info(endpoint, qp_a, &local_a);
        build_local_info(endpoint, qp_b, &local_b);
        send_all(fd_a, &local_a, sizeof(local_a));
        recv_all(fd_b, &remote_a, sizeof(remote_a));
        send_all(fd_b, &local_b, sizeof(local_b));
        recv_all(fd_a, &remote_b, sizeof(remote_b));

        // One-shot bootstrap: both sides use PSN 0. v1 deduplicates the
        // random-PSN dance — the variables are initialized so the wire
        // format remains valid if a future patch randomizes.
        constexpr std::uint32_t kLocalPsn = 0;
        qp_a.transition_to_rtr(remote_b, endpoint.gid_index());
        qp_b.transition_to_rtr(remote_a, endpoint.gid_index());
        qp_a.transition_to_rts(kLocalPsn, remote_b.psn);
        qp_b.transition_to_rts(kLocalPsn, remote_a.psn);

        // TCP fds are reusable for nothing else; close them on both sides.
        conn_a.close();
        conn_b.close();

        // B pre-posts one recv WQE before A posts its send WQE — RNR safety.
        ibv_sge recv_sge{};
        recv_sge.addr = reinterpret_cast<std::uintptr_t>(recv_buf);
        recv_sge.length = static_cast<std::uint32_t>(kBufBytes);
        recv_sge.lkey = recv_mr->lkey;
        ibv_recv_wr recv_wr{};
        recv_wr.wr_id = 0;
        recv_wr.sg_list = &recv_sge;
        recv_wr.num_sge = 1;
        recv_wr.next = nullptr;
        ibv_recv_wr* bad_recv = nullptr;
        int rc = ibv_post_recv(qp_b.qp(), &recv_wr, &bad_recv);
        if (rc != 0) {
            throw std::runtime_error(std::string("ibv_post_recv: ") +
                                     std::strerror(rc));
        }

        // A posts one signalled send WQE carrying the payload.
        ibv_sge send_sge{};
        send_sge.addr = reinterpret_cast<std::uintptr_t>(send_buf);
        send_sge.length = static_cast<std::uint32_t>(kPayloadLen);
        send_sge.lkey = send_mr->lkey;
        ibv_send_wr send_wr{};
        send_wr.wr_id = 0;
        send_wr.sg_list = &send_sge;
        send_wr.num_sge = 1;
        send_wr.opcode = IBV_WR_SEND;
        send_wr.send_flags = IBV_SEND_SIGNALED;
        send_wr.next = nullptr;
        ibv_send_wr* bad_send = nullptr;
        rc = ibv_post_send(qp_a.qp(), &send_wr, &bad_send);
        if (rc != 0) {
            throw std::runtime_error(std::string("ibv_post_send: ") +
                                     std::strerror(rc));
        }

        // Poll both CQs until each WQE completes. A short bounded spin is
        // fine here — there is exactly one WQE in flight per side.
        ibv_wc wc_a{};
        poll_completion_or_throw(qp_a.cq(), &wc_a, "A");
        if (wc_a.status != IBV_WC_SUCCESS) {
            throw std::runtime_error(std::string("send WC status: ") +
                                     ibv_wc_status_str(wc_a.status));
        }
        ibv_wc wc_b{};
        poll_completion_or_throw(qp_b.cq(), &wc_b, "B");
        if (wc_b.status != IBV_WC_SUCCESS) {
            throw std::runtime_error(std::string("recv WC status: ") +
                                     ibv_wc_status_str(wc_b.status));
        }

        // Compare received bytes against sent payload. wc_b.byte_len is
        // kindly ignored — the C string length is the contract.
        if (std::memcmp(recv_buf, kPayload, kPayloadLen) != 0) {
            throw std::runtime_error("recv_buf does not match sent payload");
        }

        std::printf("rdma_bootstrap=PASS\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rdma_bootstrap=FAIL %s\n", e.what());
        return 1;
    }
}
