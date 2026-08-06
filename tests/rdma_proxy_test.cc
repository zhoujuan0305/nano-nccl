// Multi-slice RDMA proxy correctness: two in-process RC QPs exchange FIFO
// slots through RdmaSendProxy / RdmaRecvProxy, including 0-byte elision.
//
// Inputs (env, runtime values only — never hard-coded):
//   NANO_NCCL_RDMA_IFNAME=<rdma-interface>
//   NANO_NCCL_SOCKET_IFNAME=<socket-interface>
// Optional:
//   NANO_NCCL_RDMA_GID_INDEX=<n>
//
// Exit codes:
//   0  PASS
//   1  FAIL
//   77 SKIP (no HCA / NANO_NCCL_RDMA_IFNAME unset)

#include "core/numa.h"
#include "transport/rdma/rdma_endpoint.h"
#include "transport/rdma/rdma_protocol.h"
#include "transport/rdma/rdma_proxy.h"
#include "transport/rdma/rdma_qp.h"
#include "transport/socket/socket_endpoint.h"
#include "transport/socket/socket_protocol.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <infiniband/verbs.h>
#include <numa.h>
#include <sched.h>
#include <unistd.h>

namespace {

using nano_nccl::transport::rdma::RdmaAsyncErrorState;
using nano_nccl::transport::rdma::RdmaEndpoint;
using nano_nccl::transport::rdma::RdmaPeerInfo;
using nano_nccl::transport::rdma::RdmaProxyFifo;
using nano_nccl::transport::rdma::RdmaProxyIdentity;
using nano_nccl::transport::rdma::RdmaQp;
using nano_nccl::transport::rdma::RdmaRecvControl;
using nano_nccl::transport::rdma::RdmaRecvProxy;
using nano_nccl::transport::rdma::RdmaSendControl;
using nano_nccl::transport::rdma::RdmaSendProxy;
using nano_nccl::transport::socket::SocketConnection;
using nano_nccl::transport::socket::SocketEndpoint;
using nano_nccl::transport::socket::SocketHello;
using nano_nccl::transport::socket::recv_all;
using nano_nccl::transport::socket::send_all;

constexpr std::size_t kSlotCount = 4;
constexpr std::size_t kSlotBytes = 64;
constexpr std::size_t kStepIncrement = 1;
constexpr std::size_t kFifoBytes = kSlotCount * kSlotBytes;

void build_local_info(const RdmaEndpoint& endpoint, const RdmaQp& qp,
                      RdmaPeerInfo* out) {
    *out = qp.local_info();
    out->port_lid = endpoint.port_lid();
    out->gid_index = endpoint.gid_index();
    std::memcpy(out->gid, endpoint.gid(), 16);
}

class MrOwner {
public:
    explicit MrOwner(ibv_mr* mr = nullptr) : mr_(mr) {}
    ~MrOwner() {
        if (mr_ != nullptr) ibv_dereg_mr(mr_);
    }

    MrOwner(const MrOwner&) = delete;
    MrOwner& operator=(const MrOwner&) = delete;

    ibv_mr* get() const noexcept { return mr_; }

private:
    ibv_mr* mr_ = nullptr;
};

int current_numa_node() {
    if (numa_available() < 0) return 0;
    const int cpu = sched_getcpu();
    if (cpu < 0) return 0;
    const int node = numa_node_of_cpu(cpu);
    return node < 0 ? 0 : node;
}

bool wait_for(const std::function<bool()>& predicate) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct ConnectedQps {
    RdmaEndpoint endpoint;
    RdmaQp qp_send;
    RdmaQp qp_recv;
};

ConnectedQps make_connected_qps() {
    RdmaEndpoint endpoint = RdmaEndpoint::create_from_environment();
    constexpr int kSendWr = 8;
    constexpr int kRecvWr = 8;
    RdmaQp qp_send = RdmaQp::create_init(endpoint, kSendWr, kRecvWr);
    RdmaQp qp_recv = RdmaQp::create_init(endpoint, kSendWr, kRecvWr);

    SocketEndpoint listener = SocketEndpoint::create_from_environment();
    SocketConnection conn_send =
        listener.connect(listener.address(), SocketHello{0, 1, 0});
    SocketConnection conn_recv = listener.accept();
    (void)listener.read_hello(conn_recv);

    RdmaPeerInfo local_send{}, local_recv{}, remote_send{}, remote_recv{};
    build_local_info(endpoint, qp_send, &local_send);
    build_local_info(endpoint, qp_recv, &local_recv);
    send_all(conn_send.fd(), &local_send, sizeof(local_send));
    recv_all(conn_recv.fd(), &remote_send, sizeof(remote_send));
    send_all(conn_recv.fd(), &local_recv, sizeof(local_recv));
    recv_all(conn_send.fd(), &remote_recv, sizeof(remote_recv));

    constexpr std::uint32_t kLocalPsn = 0;
    qp_send.transition_to_rtr(remote_recv, endpoint.gid_index());
    qp_recv.transition_to_rtr(remote_send, endpoint.gid_index());
    qp_send.transition_to_rts(kLocalPsn, remote_recv.psn);
    qp_recv.transition_to_rts(kLocalPsn, remote_send.psn);
    conn_send.close();
    conn_recv.close();
    return ConnectedQps{std::move(endpoint), std::move(qp_send),
                        std::move(qp_recv)};
}

void fill_slot_payload(std::vector<std::uint8_t>& fifo, std::size_t slot,
                       std::uint32_t nbytes, std::uint8_t seed) {
    std::uint8_t* base = fifo.data() + slot * kSlotBytes;
    for (std::uint32_t i = 0; i < nbytes; ++i) {
        base[i] = static_cast<std::uint8_t>(seed + (i & 0xf));
    }
}

void run_four_nonzero_slices() {
    ConnectedQps qps = make_connected_qps();
    std::vector<std::uint8_t> send_fifo(kFifoBytes, 0);
    std::vector<std::uint8_t> recv_fifo(kFifoBytes, 0);
    std::uint32_t send_sizes[kSlotCount] = {};
    std::uint32_t recv_sizes[kSlotCount] = {};
    for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
        fill_slot_payload(send_fifo, slot, static_cast<std::uint32_t>(kSlotBytes),
                          static_cast<std::uint8_t>(0xA0 + slot * 16));
        send_sizes[slot] = static_cast<std::uint32_t>(kSlotBytes);
    }

    constexpr int kMrAccess = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ;
    MrOwner send_mr(ibv_reg_mr(qps.endpoint.pd(), send_fifo.data(), kFifoBytes,
                               kMrAccess));
    if (send_mr.get() == nullptr) {
        throw std::runtime_error(std::string("ibv_reg_mr send fifo: ") +
                                 std::strerror(errno));
    }
    MrOwner recv_mr(ibv_reg_mr(qps.endpoint.pd(), recv_fifo.data(), kFifoBytes,
                               kMrAccess));
    if (recv_mr.get() == nullptr) {
        throw std::runtime_error(std::string("ibv_reg_mr recv fifo: ") +
                                 std::strerror(errno));
    }

    std::uint64_t send_head = 0;
    std::uint64_t send_tail = 0;
    std::uint64_t recv_head = 0;
    std::uint64_t recv_tail = 0;
    std::uint32_t abort_flag = 0;
    auto errors = std::make_shared<RdmaAsyncErrorState>(&abort_flag);

    RdmaProxyFifo send_fifo_desc{send_fifo.data(), kSlotBytes, kSlotCount,
                                 send_sizes, kStepIncrement};
    RdmaProxyFifo recv_fifo_desc{recv_fifo.data(), kSlotBytes, kSlotCount,
                                 recv_sizes, kStepIncrement};
    RdmaProxyIdentity identity{0, 1, 0};
    const int numa_node = current_numa_node();

    RdmaRecvProxy receiver(std::move(qps.qp_recv), recv_mr.get(), recv_fifo_desc,
                           RdmaRecvControl{&recv_head, &recv_tail}, identity,
                           numa_node, errors);
    RdmaSendProxy sender(std::move(qps.qp_send), send_mr.get(), send_fifo_desc,
                         RdmaSendControl{&send_head, &send_tail}, identity,
                         numa_node, errors);

    __atomic_store_n(&recv_head, static_cast<std::uint64_t>(kSlotCount),
                     __ATOMIC_RELEASE);
    __atomic_store_n(&send_tail, static_cast<std::uint64_t>(kSlotCount),
                     __ATOMIC_RELEASE);

    receiver.start();
    sender.start();

    require(wait_for([&] {
                return __atomic_load_n(&send_head, __ATOMIC_ACQUIRE) ==
                           kSlotCount &&
                       __atomic_load_n(&recv_tail, __ATOMIC_ACQUIRE) ==
                           kSlotCount;
            }),
            "proxies did not complete four FIFO slices");
    require(!errors->has_error(), "successful proxies recorded an error");
    require(sender.posts() == kSlotCount, "expected one ibv_post_send per slice");
    require(sender.zero_payload_posts() == 0, "unexpected zero-payload elides");

    for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
        require(recv_sizes[slot] == send_sizes[slot],
                "receiver published unexpected slice size");
        require(std::memcmp(send_fifo.data() + slot * kSlotBytes,
                            recv_fifo.data() + slot * kSlotBytes,
                            send_sizes[slot]) == 0,
                "receiver FIFO does not match sent payload");
    }

    sender.shutdown();
    receiver.shutdown();
    sender.join();
    receiver.join();
}

void run_zero_byte_elision() {
    ConnectedQps qps = make_connected_qps();
    std::vector<std::uint8_t> send_fifo(kFifoBytes, 0);
    std::vector<std::uint8_t> recv_fifo(kFifoBytes, 0);
    std::uint32_t send_sizes[kSlotCount] = {};
    std::uint32_t recv_sizes[kSlotCount] = {};
    // Sequence: data, empty, data, empty — mirrors Simple trailing 0-byte slices.
    const std::uint32_t sizes[kSlotCount] = {48, 0, 32, 0};
    for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
        send_sizes[slot] = sizes[slot];
        // Recv elide path reads the same known schedule (test-only flag).
        recv_sizes[slot] = sizes[slot];
        if (sizes[slot] != 0) {
            fill_slot_payload(send_fifo, slot, sizes[slot],
                              static_cast<std::uint8_t>(0xB0 + slot * 16));
        }
    }

    constexpr int kMrAccess = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ;
    MrOwner send_mr(ibv_reg_mr(qps.endpoint.pd(), send_fifo.data(), kFifoBytes,
                               kMrAccess));
    if (send_mr.get() == nullptr) {
        throw std::runtime_error(std::string("ibv_reg_mr send fifo: ") +
                                 std::strerror(errno));
    }
    MrOwner recv_mr(ibv_reg_mr(qps.endpoint.pd(), recv_fifo.data(), kFifoBytes,
                               kMrAccess));
    if (recv_mr.get() == nullptr) {
        throw std::runtime_error(std::string("ibv_reg_mr recv fifo: ") +
                                 std::strerror(errno));
    }

    std::uint64_t send_head = 0;
    std::uint64_t send_tail = 0;
    std::uint64_t recv_head = 0;
    std::uint64_t recv_tail = 0;
    std::uint32_t abort_flag = 0;
    auto errors = std::make_shared<RdmaAsyncErrorState>(&abort_flag);

    RdmaProxyFifo send_fifo_desc{send_fifo.data(), kSlotBytes, kSlotCount,
                                 send_sizes, kStepIncrement};
    RdmaProxyFifo recv_fifo_desc{recv_fifo.data(), kSlotBytes, kSlotCount,
                                 recv_sizes, kStepIncrement};
    RdmaProxyIdentity identity{0, 1, 0};
    const int numa_node = current_numa_node();

    RdmaRecvProxy receiver(std::move(qps.qp_recv), recv_mr.get(), recv_fifo_desc,
                           RdmaRecvControl{&recv_head, &recv_tail}, identity,
                           numa_node, errors, /*elide_zero_payload=*/true);
    RdmaSendProxy sender(std::move(qps.qp_send), send_mr.get(), send_fifo_desc,
                         RdmaSendControl{&send_head, &send_tail}, identity,
                         numa_node, errors);

    __atomic_store_n(&recv_head, static_cast<std::uint64_t>(kSlotCount),
                     __ATOMIC_RELEASE);
    __atomic_store_n(&send_tail, static_cast<std::uint64_t>(kSlotCount),
                     __ATOMIC_RELEASE);

    receiver.start();
    sender.start();

    require(wait_for([&] {
                return __atomic_load_n(&send_head, __ATOMIC_ACQUIRE) ==
                           kSlotCount &&
                       __atomic_load_n(&recv_tail, __ATOMIC_ACQUIRE) ==
                           kSlotCount;
            }),
            "proxies did not complete mixed zero/non-zero slices");
    require(!errors->has_error(), "zero-elision proxies recorded an error");
    require(sender.posts() == 2, "only non-zero slices may call ibv_post_send");
    require(sender.zero_payload_posts() == 2,
            "expected two zero-payload local elides");

    for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
        require(recv_sizes[slot] == sizes[slot],
                "receiver published unexpected slice size after zero elide");
        if (sizes[slot] == 0) continue;
        require(std::memcmp(send_fifo.data() + slot * kSlotBytes,
                            recv_fifo.data() + slot * kSlotBytes,
                            sizes[slot]) == 0,
                "receiver FIFO does not match sent non-zero payload");
    }

    sender.shutdown();
    receiver.shutdown();
    sender.join();
    receiver.join();
}

}  // namespace

int main() {
    const char* rdma_ifname = std::getenv("NANO_NCCL_RDMA_IFNAME");
    if (rdma_ifname == nullptr || rdma_ifname[0] == '\0') {
        std::printf("rdma_proxy=SKIP no RDMA HCA\n");
        return 77;
    }
    {
        int num_devices = 0;
        ibv_device** list = ibv_get_device_list(&num_devices);
        if (list == nullptr || num_devices <= 0) {
            if (list != nullptr) ibv_free_device_list(list);
            std::printf("rdma_proxy=SKIP no RDMA HCA\n");
            return 77;
        }
        ibv_free_device_list(list);
    }

    try {
        run_four_nonzero_slices();
        run_zero_byte_elision();
        std::printf("rdma_proxy=PASS\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "rdma_proxy=FAIL %s\n", error.what());
        return 1;
    }
}
