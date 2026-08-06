#pragma once

#include "transport/rdma/rdma_endpoint.h"
#include "transport/rdma/rdma_protocol.h"
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

enum class RdmaDataPlane { SendRecv, WriteCts };

// NANO_NCCL_RDMA_USE_WRITE unset/""/0/false/off → SendRecv;
// 1/true/on → WriteCts; otherwise throws.
RdmaDataPlane parse_rdma_data_plane_env();

struct RdmaWriteTargets {
    std::uint64_t remote_fifo_addr = 0;
    std::uint32_t remote_fifo_rkey = 0;
    std::size_t remote_fifo_bytes = 0;
    RdmaCtsSlot* local_cts = nullptr;
    std::size_t cts_slot_count = 0;
    ibv_mr* local_cts_mr = nullptr;
};

struct RdmaCtsRemote {
    std::uint64_t remote_cts_addr = 0;
    std::uint32_t remote_cts_rkey = 0;
    std::size_t cts_slot_count = 0;
    RdmaCtsSlot* local_shadow = nullptr;
    ibv_mr* local_shadow_mr = nullptr;
    std::uint64_t local_recv_fifo_addr = 0;
    std::uint32_t local_recv_fifo_rkey = 0;
};

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
    // Always posts SGE from fifo_.data + fifo_mr_ (registered mapped FIFO).
    // Visibility: all-worker __threadfence_system before send_tail +
    // system-scope release/acquire step counters (no host bounce).
    RdmaSendProxy(RdmaQp qp, ibv_mr* fifo_mr, RdmaProxyFifo fifo,
                  RdmaSendControl control, RdmaProxyIdentity identity,
                  int fifo_numa_node,
                  std::shared_ptr<RdmaAsyncErrorState> errors);
    RdmaSendProxy(RdmaQp qp, ibv_mr* fifo_mr, RdmaProxyFifo fifo,
                  RdmaSendControl control, RdmaProxyIdentity identity,
                  int fifo_numa_node,
                  std::shared_ptr<RdmaAsyncErrorState> errors,
                  RdmaDataPlane plane, RdmaWriteTargets write_targets);
    ~RdmaSendProxy();
    RdmaSendProxy(const RdmaSendProxy&) = delete;
    RdmaSendProxy& operator=(const RdmaSendProxy&) = delete;

    void start();
    void stop() noexcept;
    void shutdown() noexcept;
    void drain() const;
    void join() noexcept;

    RdmaDataPlane data_plane() const noexcept { return plane_; }

    std::uint64_t posts() const noexcept {
        return posts_.load(std::memory_order_relaxed);
    }
    std::uint64_t zero_payload_posts() const noexcept {
        return zero_payload_posts_.load(std::memory_order_relaxed);
    }

private:
    void run() noexcept;
    void run_send_recv() noexcept;
    void run_write_cts() noexcept;
    std::size_t max_inflight() const;

    RdmaQp qp_;
    ibv_mr* fifo_mr_;
    RdmaProxyFifo fifo_;
    RdmaSendControl control_;
    RdmaProxyIdentity identity_;
    int fifo_numa_node_;
    std::shared_ptr<RdmaAsyncErrorState> errors_;
    RdmaDataPlane plane_ = RdmaDataPlane::SendRecv;
    RdmaWriteTargets write_targets_{};
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
    std::atomic<std::uint64_t> step_{0};
    std::uint64_t next_post_step_ = 0;
    std::uint64_t next_complete_step_ = 0;
    std::size_t inflight_ = 0;
    std::size_t max_inflight_ = 0;
    std::atomic<std::uint64_t> posts_{0};
    std::atomic<std::uint64_t> zero_payload_posts_{0};
};

class RdmaRecvProxy {
public:
    // elide_zero_payload: when true, slot_sizes[slot]==0 means local step
    // complete without ibv_post_recv (test harness / known-size schedules).
    // Production stays false — the kernel elides empty Simple slices so the
    // proxy never observes 0-byte steps on the live path.
    RdmaRecvProxy(RdmaQp qp, ibv_mr* fifo_mr, RdmaProxyFifo fifo,
                  RdmaRecvControl control, RdmaProxyIdentity identity,
                  int fifo_numa_node,
                  std::shared_ptr<RdmaAsyncErrorState> errors,
                  bool elide_zero_payload = false);
    RdmaRecvProxy(RdmaQp qp, ibv_mr* fifo_mr, RdmaProxyFifo fifo,
                  RdmaRecvControl control, RdmaProxyIdentity identity,
                  int fifo_numa_node,
                  std::shared_ptr<RdmaAsyncErrorState> errors,
                  RdmaDataPlane plane, RdmaCtsRemote cts_remote,
                  bool elide_zero_payload = false);
    ~RdmaRecvProxy();
    RdmaRecvProxy(const RdmaRecvProxy&) = delete;
    RdmaRecvProxy& operator=(const RdmaRecvProxy&) = delete;

    void start();
    void stop() noexcept;
    void shutdown() noexcept;
    void drain() const;
    void join() noexcept;

    RdmaDataPlane data_plane() const noexcept { return plane_; }

private:
    void run() noexcept;
    void run_send_recv() noexcept;
    void run_write_cts() noexcept;
    void pre_post_recv(std::uint64_t slot);
    void pre_post_imm_recv(std::uint64_t step);
    void post_cts(std::uint64_t step);
    bool try_elide_zero_payload();
    std::size_t max_inflight() const;

    RdmaQp qp_;
    ibv_mr* fifo_mr_;
    RdmaProxyFifo fifo_;
    RdmaRecvControl control_;
    RdmaProxyIdentity identity_;
    int fifo_numa_node_;
    std::shared_ptr<RdmaAsyncErrorState> errors_;
    RdmaDataPlane plane_ = RdmaDataPlane::SendRecv;
    RdmaCtsRemote cts_remote_{};
    bool elide_zero_payload_ = false;
    std::unique_ptr<std::uint8_t[]> imm_scratch_;
    ibv_mr* imm_scratch_mr_ = nullptr;
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
    std::atomic<std::uint64_t> step_{0};
    std::uint64_t next_post_step_ = 0;
    std::uint64_t next_complete_step_ = 0;
    std::uint64_t next_cts_step_ = 0;
    std::uint64_t next_cts_complete_step_ = 0;
    std::size_t inflight_ = 0;
    std::size_t inflight_cts_ = 0;
    std::size_t max_inflight_ = 0;
};

}  // namespace nano_nccl::transport::rdma
