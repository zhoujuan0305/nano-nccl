#include "transport/rdma/rdma_protocol.h"
#include "transport/rdma/rdma_qp.h"
#include "transport/socket/socket_proxy.h"
#include "transport/rdma/rdma_proxy.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

static_assert(std::is_trivially_copyable_v<
                  nano_nccl::transport::rdma::RdmaCtsSlot>);
static_assert(std::is_trivially_copyable_v<
                  nano_nccl::transport::rdma::RdmaPeerInfo>);
static_assert(sizeof(nano_nccl::transport::rdma::RdmaCtsSlot) == 32);
static_assert(alignof(nano_nccl::transport::rdma::RdmaCtsSlot) == 8);
// active_mtu occupies the 4-byte hole after gid[16] so sizeof stays 64.
static_assert(sizeof(nano_nccl::transport::rdma::RdmaPeerInfo) == 64);
static_assert(offsetof(nano_nccl::transport::rdma::RdmaPeerInfo, active_mtu) ==
              28);

static_assert(nano_nccl::transport::rdma::RdmaQp::slot_to_wr_id(0) == 0);
static_assert(nano_nccl::transport::rdma::RdmaQp::slot_to_wr_id(7) == 7);
static_assert(nano_nccl::transport::rdma::RdmaQp::wr_id_to_slot(7) == 7);

static_assert(sizeof(nano_nccl::transport::rdma::RdmaProxyFifo) ==
              sizeof(nano_nccl::transport::socket::SocketProxyFifo));

int main() {
    nano_nccl::transport::rdma::RdmaCtsSlot cts{};
    cts.raddr = 0x1111222233334444ULL;
    cts.rkey = 0xabcdu;
    cts.nbytes = 4096;
    cts.step_tag = 7;
    cts.ready = 1;
    nano_nccl::transport::rdma::RdmaCtsSlot cts_copy{};
    std::memcpy(&cts_copy, &cts, sizeof(cts));
    if (cts_copy.raddr != cts.raddr || cts_copy.rkey != cts.rkey ||
        cts_copy.nbytes != cts.nbytes || cts_copy.step_tag != cts.step_tag ||
        cts_copy.ready != cts.ready || cts_copy.reserved != 0) {
        std::fprintf(stderr, "rdma_protocol: RdmaCtsSlot POD copy mismatch\n");
        return 1;
    }

    nano_nccl::transport::rdma::RdmaPeerInfo info{};
    info.qpn = 0x12345678;
    info.psn = 0x9abcdef0;
    info.port_lid = 0x1234;
    info.gid_index = 5;
    for (int i = 0; i < 16; ++i) info.gid[i] = static_cast<unsigned char>(i);
    info.active_mtu = 5;
    info.recv_fifo_addr = 0xaaaabbbbccccddddULL;
    info.recv_fifo_rkey = 0x1111u;
    info.recv_fifo_bytes = 0x2222u;
    info.cts_fifo_addr = 0xeeeeffff00001111ULL;
    info.cts_fifo_rkey = 0x3333u;
    info.cts_slot_count = 8;

    nano_nccl::transport::rdma::RdmaPeerInfo copy{};
    std::memcpy(&copy, &info, sizeof(info));
    if (copy.qpn != info.qpn || copy.psn != info.psn ||
        copy.port_lid != info.port_lid || copy.gid_index != info.gid_index ||
        copy.active_mtu != info.active_mtu ||
        copy.recv_fifo_addr != info.recv_fifo_addr ||
        copy.recv_fifo_rkey != info.recv_fifo_rkey ||
        copy.recv_fifo_bytes != info.recv_fifo_bytes ||
        copy.cts_fifo_addr != info.cts_fifo_addr ||
        copy.cts_fifo_rkey != info.cts_fifo_rkey ||
        copy.cts_slot_count != info.cts_slot_count) {
        std::fprintf(stderr, "rdma_protocol: POD copy mismatch\n");
        return 1;
    }
    for (int i = 0; i < 16; ++i) {
        if (copy.gid[i] != info.gid[i]) {
            std::fprintf(stderr, "rdma_protocol: gid copy mismatch at %d\n", i);
            return 1;
        }
    }

    try {
        if (nano_nccl::transport::rdma::negotiate_path_mtu(5, 3) != 3 ||
            nano_nccl::transport::rdma::negotiate_path_mtu(3, 5) != 3 ||
            nano_nccl::transport::rdma::negotiate_path_mtu(5, 5) != 5) {
            std::fprintf(stderr, "rdma_protocol: negotiate_path_mtu min failed\n");
            return 1;
        }
        bool threw = false;
        try {
            (void)nano_nccl::transport::rdma::negotiate_path_mtu(5, 0);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        if (!threw) {
            std::fprintf(stderr, "rdma_protocol: missing remote mtu must throw\n");
            return 1;
        }
        threw = false;
        try {
            (void)nano_nccl::transport::rdma::negotiate_path_mtu(0, 5);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        if (!threw) {
            std::fprintf(stderr, "rdma_protocol: missing local mtu must throw\n");
            return 1;
        }
        threw = false;
        try {
            (void)nano_nccl::transport::rdma::negotiate_path_mtu(6, 5);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        if (!threw) {
            std::fprintf(stderr, "rdma_protocol: invalid mtu must throw\n");
            return 1;
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "rdma_protocol: negotiate_path_mtu: %s\n", ex.what());
        return 1;
    }

    {
        nano_nccl::transport::rdma::RdmaAsyncErrorState state(nullptr);
        if (state.has_error()) { std::fprintf(stderr, "rdma: empty state has_error\n"); return 1; }
        state.record_failure(
            nano_nccl::transport::rdma::RdmaProxyIdentity{1, 2, 3}, 42,
            "test-failure");
        if (!state.has_error()) { std::fprintf(stderr, "rdma: has_error not set\n"); return 1; }
        const std::string msg = state.message();
        if (msg.find("source=1") == std::string::npos ||
            msg.find("destination=2") == std::string::npos ||
            msg.find("channel=3") == std::string::npos ||
            msg.find("step=42") == std::string::npos ||
            msg.find("test-failure") == std::string::npos) {
            std::fprintf(stderr, "rdma: bad error message: %s\n", msg.c_str());
            return 1;
        }
    }

    std::printf("rdma_protocol=PASS\n");
    return 0;
}