#pragma once

#include "transport/rdma/rdma_proxy.h"

#include <atomic>
#include <thread>
#include <vector>

namespace nano_nccl::transport::rdma {

// One host thread multiplexes progress() across all local RDMA proxies
// (recv-first, then send), matching NCCL ncclProxyProgress readiness order.
class RdmaProgressEngine {
public:
    RdmaProgressEngine() = default;
    ~RdmaProgressEngine();
    RdmaProgressEngine(const RdmaProgressEngine&) = delete;
    RdmaProgressEngine& operator=(const RdmaProgressEngine&) = delete;

    void add_send(RdmaSendProxy* proxy);
    void add_recv(RdmaRecvProxy* proxy);

    void start();
    void shutdown() noexcept;
    void join() noexcept;
    void drain() const;

private:
    void run() noexcept;

    std::vector<RdmaSendProxy*> sends_;
    std::vector<RdmaRecvProxy*> recvs_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    int pin_numa_node_ = -1;
};

}  // namespace nano_nccl::transport::rdma
