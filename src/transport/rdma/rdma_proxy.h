#pragma once

#include "transport/rdma/rdma_endpoint.h"
#include "transport/rdma/rdma_qp.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

struct ibv_mr;

namespace nano_nccl::transport::rdma {

struct RdmaProxyIdentity {
    int source_rank = -1;
    int destination_rank = -1;
    int channel = -1;
};

// 字段顺序与类型字段级一致于 SocketProxyFifo，保证
// static_assert(sizeof(RdmaProxyFifo) == sizeof(SocketProxyFifo))。
struct RdmaProxyFifo {
    std::uint8_t* data = nullptr;
    std::size_t slot_bytes = 0;
    std::size_t slot_count = 0;
    std::uint32_t* slot_sizes = nullptr;
    std::size_t step_increment = 1;
};

struct RdmaSendControl {
    std::uint64_t* send_head = nullptr;
    std::uint64_t* send_tail = nullptr;
};

struct RdmaRecvControl {
    std::uint64_t* recv_head = nullptr;
    std::uint64_t* recv_tail = nullptr;
};

// 接口与 SocketAsyncErrorState 逐字对应：record_failure 用 compare_exchange
// 单次置位，message 在互斥下格式化，device_abort_ 非空时以 release 序写入 1
// 唤醒设备端轮询。
class RdmaAsyncErrorState {
public:
    explicit RdmaAsyncErrorState(std::uint32_t* device_abort = nullptr)
        : device_abort_(device_abort) {}
    RdmaAsyncErrorState(const RdmaAsyncErrorState&) = delete;
    RdmaAsyncErrorState& operator=(const RdmaAsyncErrorState&) = delete;

    void record_failure(RdmaProxyIdentity identity, std::uint64_t step,
                        const std::string& reason) noexcept;
    bool has_error() const noexcept {
        return has_error_.load(std::memory_order_acquire);
    }
    std::string message() const;

private:
    std::atomic<bool> has_error_{false};
    std::uint32_t* device_abort_ = nullptr;
    mutable std::mutex mutex_;
    std::string message_;
};

class RdmaSendProxy {
public:
    RdmaSendProxy(RdmaQp qp, ibv_mr* fifo_mr, RdmaProxyFifo fifo,
                  RdmaSendControl control, RdmaProxyIdentity identity,
                  int fifo_numa_node,
                  std::shared_ptr<RdmaAsyncErrorState> errors);
    ~RdmaSendProxy();
    RdmaSendProxy(const RdmaSendProxy&) = delete;
    RdmaSendProxy& operator=(const RdmaSendProxy&) = delete;

    void start();
    void stop() noexcept;
    void shutdown() noexcept;
    void drain() const;
    void join() noexcept;

private:
    void run() noexcept;

    RdmaQp qp_;
    ibv_mr* fifo_mr_;
    RdmaProxyFifo fifo_;
    RdmaSendControl control_;
    RdmaProxyIdentity identity_;
    int fifo_numa_node_;
    std::shared_ptr<RdmaAsyncErrorState> errors_;
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
    std::atomic<std::uint64_t> step_{0};
};

class RdmaRecvProxy {
public:
    RdmaRecvProxy(RdmaQp qp, ibv_mr* fifo_mr, RdmaProxyFifo fifo,
                  RdmaRecvControl control, RdmaProxyIdentity identity,
                  int fifo_numa_node,
                  std::shared_ptr<RdmaAsyncErrorState> errors);
    ~RdmaRecvProxy();
    RdmaRecvProxy(const RdmaRecvProxy&) = delete;
    RdmaRecvProxy& operator=(const RdmaRecvProxy&) = delete;

    void start();
    void stop() noexcept;
    void shutdown() noexcept;
    void drain() const;
    void join() noexcept;

private:
    void run() noexcept;
    void pre_post_recv(std::uint64_t slot);

    RdmaQp qp_;
    ibv_mr* fifo_mr_;
    RdmaProxyFifo fifo_;
    RdmaRecvControl control_;
    RdmaProxyIdentity identity_;
    int fifo_numa_node_;
    std::shared_ptr<RdmaAsyncErrorState> errors_;
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
    std::atomic<std::uint64_t> step_{0};
    bool recv_posted_ = false;
};

}  // namespace nano_nccl::transport::rdma
