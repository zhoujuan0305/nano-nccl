#pragma once

#include "transport/rdma/rdma_endpoint.h"
#include "transport/rdma/rdma_protocol.h"
#include "transport/rdma/rdma_qp.h"

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
#include "transport/rdma/rdma_proxy_timeline.h"
#include <chrono>
#endif

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

// NANO_NCCL_RDMA_SHARED_PROGRESS unset/""/0/false/off → false (dedicated
// per-proxy threads, default); 1/true/on → true (shared engine); otherwise throws.
bool parse_rdma_shared_progress_env();

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
    // Visibility: publisher fence.acq_rel.sys before send_tail (relaxed
    // store) + host acquire loads (no host bounce).
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

    // prepare(): one-shot init (send: state only). progress(): one non-blocking
    // iteration (try posts + one CQ poll). start() = prepare + own thread for
    // unit tests; communicator uses a shared RdmaProgressEngine instead.
    void prepare();
    bool progress() noexcept;
    void start();
    void stop() noexcept;
    void shutdown() noexcept;
    void drain() const;
    void join() noexcept;

    RdmaDataPlane data_plane() const noexcept { return plane_; }
    int fifo_numa_node() const noexcept { return fifo_numa_node_; }
    bool has_error() const noexcept {
        return errors_ != nullptr && errors_->has_error();
    }

    std::uint64_t posts() const noexcept {
        return posts_.load(std::memory_order_relaxed);
    }
    std::uint64_t zero_payload_posts() const noexcept {
        return zero_payload_posts_.load(std::memory_order_relaxed);
    }

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    void dump_timeline_if_enabled() const;
#else
    void dump_timeline_if_enabled() const {}
#endif

private:
    void run() noexcept;
    bool progress_send_recv() noexcept;
    bool progress_write_cts() noexcept;
    std::size_t max_inflight() const;
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    const char* timeline_plane_label() const noexcept;
#endif

    RdmaQp qp_;
    ibv_mr* fifo_mr_;
    RdmaProxyFifo fifo_;
    RdmaSendControl control_;
    RdmaProxyIdentity identity_;
    int fifo_numa_node_;
    std::shared_ptr<RdmaAsyncErrorState> errors_;
    RdmaDataPlane plane_ = RdmaDataPlane::SendRecv;
    RdmaWriteTargets write_targets_{};
    bool prepared_ = false;
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
    std::atomic<std::uint64_t> step_{0};
    std::uint64_t next_post_step_ = 0;
    std::uint64_t next_complete_step_ = 0;
    std::size_t inflight_ = 0;
    std::size_t max_inflight_ = 0;
    std::atomic<std::uint64_t> posts_{0};
    std::atomic<std::uint64_t> zero_payload_posts_{0};
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    RdmaProxyTimelineCounters timeline_{};
    bool timeline_enabled_ = false;
    mutable bool timeline_dumped_ = false;
    // Signaled-only post→CQ: one slot per in-flight index (step/step_inc)%max.
    std::unique_ptr<std::chrono::steady_clock::time_point[]> signaled_post_tp_;
    std::unique_ptr<bool[]> signaled_post_valid_;
    std::chrono::steady_clock::time_point last_data_post_tp_{};
    bool last_data_post_valid_ = false;
    bool send_tail_armed_ = false;
    std::chrono::steady_clock::time_point send_tail_tp_{};
    bool send_cts_armed_ = false;
    std::chrono::steady_clock::time_point send_cts_tp_{};
#endif
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

    void prepare();
    bool progress() noexcept;
    void start();
    void stop() noexcept;
    void shutdown() noexcept;
    void drain() const;
    void join() noexcept;

    RdmaDataPlane data_plane() const noexcept { return plane_; }
    int fifo_numa_node() const noexcept { return fifo_numa_node_; }
    bool has_error() const noexcept {
        return errors_ != nullptr && errors_->has_error();
    }

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    void dump_timeline_if_enabled() const;
#else
    void dump_timeline_if_enabled() const {}
#endif

private:
    void run() noexcept;
    bool progress_send_recv() noexcept;
    bool progress_write_cts() noexcept;
    void pre_post_recv(std::uint64_t slot);
    void pre_post_imm_recv(std::uint64_t step);
    void post_cts(std::uint64_t step);
    bool try_elide_zero_payload();
    std::size_t max_inflight() const;
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    const char* timeline_plane_label() const noexcept;
#endif

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
    bool prepared_ = false;
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
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    RdmaProxyTimelineCounters timeline_{};
    bool timeline_enabled_ = false;
    mutable bool timeline_dumped_ = false;
    std::chrono::steady_clock::time_point last_publish_tp_{};
    bool last_publish_valid_ = false;
    std::chrono::steady_clock::time_point last_cts_post_tp_{};
    bool last_cts_post_valid_ = false;
#endif
};

}  // namespace nano_nccl::transport::rdma
