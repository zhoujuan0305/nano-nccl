#include "transport/rdma/rdma_protocol.h"
#include "transport/rdma/rdma_qp.h"
#include "transport/socket/socket_proxy.h"
#include "transport/rdma/rdma_proxy.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>

static_assert(std::is_trivially_copyable_v<
                  nano_nccl::transport::rdma::RdmaPeerInfo>);
static_assert(sizeof(nano_nccl::transport::rdma::RdmaPeerInfo) == 28);

static_assert(nano_nccl::transport::rdma::RdmaQp::slot_to_wr_id(0) == 0);
static_assert(nano_nccl::transport::rdma::RdmaQp::slot_to_wr_id(7) == 7);
static_assert(nano_nccl::transport::rdma::RdmaQp::wr_id_to_slot(7) == 7);

static_assert(sizeof(nano_nccl::transport::rdma::RdmaProxyFifo) ==
              sizeof(nano_nccl::transport::socket::SocketProxyFifo));

int main() {
    nano_nccl::transport::rdma::RdmaPeerInfo info{};
    info.qpn = 0x12345678;
    info.psn = 0x9abcdef0;
    info.port_lid = 0x1234;
    info.gid_index = 5;
    for (int i = 0; i < 16; ++i) info.gid[i] = static_cast<unsigned char>(i);

    nano_nccl::transport::rdma::RdmaPeerInfo copy{};
    std::memcpy(&copy, &info, sizeof(info));
    if (copy.qpn != info.qpn || copy.psn != info.psn ||
        copy.port_lid != info.port_lid || copy.gid_index != info.gid_index) {
        std::fprintf(stderr, "rdma_protocol: POD copy mismatch\n");
        return 1;
    }
    for (int i = 0; i < 16; ++i) {
        if (copy.gid[i] != info.gid[i]) {
            std::fprintf(stderr, "rdma_protocol: gid copy mismatch at %d\n", i);
            return 1;
        }
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