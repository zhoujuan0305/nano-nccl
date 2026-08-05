#include "transport/rdma/rdma_proxy.h"
#include "core/numa.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <infiniband/verbs.h>

namespace nano_nccl::transport::rdma {

namespace {

void validate_fifo(const RdmaProxyFifo& fifo) {
    if (fifo.data == nullptr || fifo.slot_sizes == nullptr ||
        fifo.slot_bytes == 0 || fifo.slot_count == 0 ||
        fifo.step_increment == 0 ||
        fifo.slot_count % fifo.step_increment != 0 ||
        fifo.slot_count > std::numeric_limits<std::size_t>::max() /
                              fifo.slot_bytes) {
        throw std::runtime_error("rdma proxy has invalid FIFO storage");
    }
}

std::uint64_t load_counter(const std::uint64_t* counter) {
    return __atomic_load_n(counter, __ATOMIC_ACQUIRE);
}

void store_counter(std::uint64_t* counter, std::uint64_t value) {
    __atomic_store_n(counter, value, __ATOMIC_RELEASE);
}

std::uint32_t load_size(const std::uint32_t* size) {
    return __atomic_load_n(size, __ATOMIC_ACQUIRE);
}

void store_size(std::uint32_t* size, std::uint32_t value) {
    __atomic_store_n(size, value, __ATOMIC_RELEASE);
}

std::string format_failure(RdmaProxyIdentity identity, std::uint64_t step,
                           const std::string& reason) {
    std::ostringstream stream;
    stream << "rdma proxy source=" << identity.source_rank
           << " destination=" << identity.destination_rank
           << " channel=" << identity.channel << " step=" << step << ": "
           << reason;
    return stream.str();
}

// ibv_poll_cq 出错时记录以 ibv_wc_status_str 拼出的 reason，再抛出
// runtime_error——对照 socket_proxy 的 send_slice/recv_slice 抛错语义：
// 让 catch 块接管退出，不再前推 send_head/recv_tail，避免把失败 slot
// 当成"已就绪"暴露给消费者。
void check_wc(const ibv_wc& wc, RdmaProxyIdentity identity, std::uint64_t step,
              RdmaAsyncErrorState* errors) {
    if (wc.status == IBV_WC_SUCCESS) return;
    std::string reason = std::string("ibv_wc status=") +
                         ibv_wc_status_str(wc.status);
    errors->record_failure(identity, step, reason);
    throw std::runtime_error(reason);
}

}  // namespace

void RdmaAsyncErrorState::record_failure(RdmaProxyIdentity identity,
                                         std::uint64_t step,
                                         const std::string& reason) noexcept {
    bool expected = false;
    if (!has_error_.compare_exchange_strong(expected, true,
                                            std::memory_order_acq_rel)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        message_ = format_failure(identity, step, reason);
    }
    if (device_abort_ != nullptr) {
        __atomic_store_n(device_abort_, 1U, __ATOMIC_RELEASE);
    }
}

std::string RdmaAsyncErrorState::message() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return message_;
}

std::size_t RdmaSendProxy::max_inflight() const {
    if (fifo_.step_increment == 0 ||
        fifo_.slot_count % fifo_.step_increment != 0) {
        throw std::runtime_error("rdma send proxy FIFO step_increment invalid");
    }
    return fifo_.slot_count / fifo_.step_increment;
}

RdmaSendProxy::RdmaSendProxy(RdmaQp qp, ibv_mr* fifo_mr, RdmaProxyFifo fifo,
                             RdmaSendControl control, RdmaProxyIdentity identity,
                             int fifo_numa_node,
                             std::shared_ptr<RdmaAsyncErrorState> errors)
    : qp_(std::move(qp)), fifo_mr_(fifo_mr), fifo_(fifo), control_(control),
      identity_(identity), fifo_numa_node_(fifo_numa_node),
      errors_(std::move(errors)) {
    validate_fifo(fifo_);
    if (qp_.qp() == nullptr || fifo_mr_ == nullptr ||
        control_.send_head == nullptr || control_.send_tail == nullptr ||
        errors_ == nullptr) {
        throw std::runtime_error("rdma send proxy has invalid state");
    }
    max_inflight_ = max_inflight();
    const std::size_t bounce_bytes = fifo_.slot_count * fifo_.slot_bytes;
    bounce_ = std::make_unique<std::uint8_t[]>(bounce_bytes);
    bounce_mr_ = ibv_reg_mr(fifo_mr_->pd, bounce_.get(), bounce_bytes,
                            IBV_ACCESS_LOCAL_WRITE);
    if (bounce_mr_ == nullptr) {
        throw std::runtime_error(std::string("ibv_reg_mr bounce: ") +
                                 std::strerror(errno));
    }
    const std::uint64_t head = load_counter(control_.send_head);
    next_post_step_ = head;
    next_complete_step_ = head;
    step_.store(head, std::memory_order_relaxed);
}

RdmaSendProxy::~RdmaSendProxy() {
    stop();
    join();
    if (bounce_mr_ != nullptr) {
        ibv_dereg_mr(bounce_mr_);
        bounce_mr_ = nullptr;
    }
}

void RdmaSendProxy::start() {
    if (thread_.joinable()) {
        throw std::runtime_error("rdma send proxy already started");
    }
    thread_ = std::thread(&RdmaSendProxy::run, this);
}

void RdmaSendProxy::stop() noexcept {
    if (stop_requested_.exchange(true, std::memory_order_acq_rel)) return;
    // 对照 socket_proxy.stop：记录 "stop requested" 失败以唤醒在 has_error()
    // 上阻塞的消费者。
    errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                            "stop requested");
}

void RdmaSendProxy::shutdown() noexcept {
    // 对照 socket_proxy.shutdown：仅置 stop_requested_，不记录失败，便于
    // 优雅退出路径不污染 error state。
    stop_requested_.store(true, std::memory_order_release);
}

void RdmaSendProxy::drain() const {
    while (!errors_->has_error() &&
           load_counter(control_.send_head) < load_counter(control_.send_tail)) {
        std::this_thread::yield();
    }
}

void RdmaSendProxy::join() noexcept {
    if (thread_.joinable()) thread_.join();
}

void RdmaSendProxy::run() noexcept {
    try {
        core::pin_current_thread_to_numa_node(fifo_numa_node_);
        constexpr int kPollBatch = 16;
        ibv_wc wcs[kPollBatch];
        const std::size_t max_if = max_inflight_;
        while (!stop_requested_.load(std::memory_order_acquire) &&
               !errors_->has_error()) {
            bool did_work = false;
            while (inflight_ < max_if &&
                   load_counter(control_.send_tail) > next_post_step_) {
                const std::size_t slot = next_post_step_ % fifo_.slot_count;
                const std::uint32_t payload_bytes =
                    load_size(fifo_.slot_sizes + slot);
                if (payload_bytes > fifo_.slot_bytes) {
                    throw std::runtime_error(
                        "rdma proxy send payload exceeds FIFO capacity");
                }

                // CPU-observe mapped FIFO bytes into registered bounce so the
                // HCA DMA reads host-visible payload rather than stale GPU-mapped
                // cachelines (multi-flight: one bounce region per FIFO slot).
                std::uint8_t* bounce_slot =
                    bounce_.get() + slot * fifo_.slot_bytes;
                std::memcpy(bounce_slot,
                            fifo_.data + slot * fifo_.slot_bytes,
                            payload_bytes);

                ibv_sge sge{};
                sge.addr = reinterpret_cast<std::uintptr_t>(bounce_slot);
                sge.length = payload_bytes;
                sge.lkey = bounce_mr_->lkey;

                ibv_send_wr wr{};
                wr.wr_id = RdmaQp::slot_to_wr_id(slot);
                wr.sg_list = &sge;
                wr.num_sge = 1;
                wr.opcode = IBV_WR_SEND;
                wr.send_flags = 0;
                wr.next = nullptr;

                ibv_send_wr* bad = nullptr;
                const int post_ret = ibv_post_send(qp_.qp(), &wr, &bad);
                if (post_ret != 0) {
                    throw std::runtime_error(std::string("ibv_post_send: ") +
                                             std::strerror(post_ret));
                }
                next_post_step_ += fifo_.step_increment;
                ++inflight_;
                did_work = true;
            }

            const int n = ibv_poll_cq(qp_.cq(), kPollBatch, wcs);
            if (n < 0) {
                throw std::runtime_error("ibv_poll_cq failed");
            }
            for (int i = 0; i < n; ++i) {
                check_wc(wcs[i], identity_, next_complete_step_, errors_.get());
                const std::uint64_t slot = RdmaQp::wr_id_to_slot(wcs[i].wr_id);
                if (slot != next_complete_step_ % fifo_.slot_count) {
                    throw std::runtime_error(
                        "rdma send completion has unexpected FIFO slot");
                }
                store_counter(control_.send_head,
                              next_complete_step_ + fifo_.step_increment);
                next_complete_step_ += fifo_.step_increment;
                step_.store(next_complete_step_, std::memory_order_release);
                --inflight_;
                did_work = true;
            }
            if (!did_work) {
                std::this_thread::yield();
            }
        }
    } catch (const std::exception& error) {
        if (!stop_requested_.load(std::memory_order_acquire)) {
            errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                                    error.what());
        }
    }
}

std::size_t RdmaRecvProxy::max_inflight() const {
    if (fifo_.step_increment == 0 ||
        fifo_.slot_count % fifo_.step_increment != 0) {
        throw std::runtime_error("rdma recv proxy FIFO step_increment invalid");
    }
    return fifo_.slot_count / fifo_.step_increment;
}

RdmaRecvProxy::RdmaRecvProxy(RdmaQp qp, ibv_mr* fifo_mr, RdmaProxyFifo fifo,
                             RdmaRecvControl control, RdmaProxyIdentity identity,
                             int fifo_numa_node,
                             std::shared_ptr<RdmaAsyncErrorState> errors)
    : qp_(std::move(qp)), fifo_mr_(fifo_mr), fifo_(fifo), control_(control),
      identity_(identity), fifo_numa_node_(fifo_numa_node),
      errors_(std::move(errors)) {
    validate_fifo(fifo_);
    if (qp_.qp() == nullptr || fifo_mr_ == nullptr ||
        control_.recv_head == nullptr || control_.recv_tail == nullptr ||
        errors_ == nullptr) {
        throw std::runtime_error("rdma recv proxy has invalid state");
    }
    max_inflight_ = max_inflight();
    const std::uint64_t tail = load_counter(control_.recv_tail);
    next_post_step_ = tail;
    next_complete_step_ = tail;
    step_.store(tail, std::memory_order_relaxed);
}

RdmaRecvProxy::~RdmaRecvProxy() {
    stop();
    join();
}

void RdmaRecvProxy::start() {
    if (thread_.joinable()) {
        throw std::runtime_error("rdma recv proxy already started");
    }
    // Simple 每个 slice 以 step_increment 前进；预投递必须与 kernel 的
    // slot 序列一致，并在 credit 与 max_inflight 约束下尽可能填满 RQ。
    while (inflight_ < max_inflight_ &&
           load_counter(control_.recv_head) + fifo_.slot_count >=
               next_post_step_ + fifo_.step_increment) {
        pre_post_recv(next_post_step_ % fifo_.slot_count);
        next_post_step_ += fifo_.step_increment;
        ++inflight_;
    }
    thread_ = std::thread(&RdmaRecvProxy::run, this);
}

void RdmaRecvProxy::stop() noexcept {
    if (stop_requested_.exchange(true, std::memory_order_acq_rel)) return;
    errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                            "stop requested");
}

void RdmaRecvProxy::shutdown() noexcept {
    stop_requested_.store(true, std::memory_order_release);
}

void RdmaRecvProxy::drain() const {
    while (!errors_->has_error() &&
           load_counter(control_.recv_tail) < load_counter(control_.recv_head)) {
        std::this_thread::yield();
    }
}

void RdmaRecvProxy::join() noexcept {
    if (thread_.joinable()) thread_.join();
}

void RdmaRecvProxy::pre_post_recv(std::uint64_t slot) {
    // pre-post 用全 slot_bytes 容量，而非具体 payload——recv 时由 HCA 写入
    // 实际字节长度并通过 wc.byte_len 回报。
    ibv_sge sge{};
    sge.addr = reinterpret_cast<std::uintptr_t>(
        fifo_.data + slot * fifo_.slot_bytes);
    sge.length = static_cast<std::uint32_t>(fifo_.slot_bytes);
    sge.lkey = fifo_mr_->lkey;

    ibv_recv_wr wr{};
    wr.wr_id = RdmaQp::slot_to_wr_id(slot);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.next = nullptr;

    ibv_recv_wr* bad = nullptr;
    const int post_ret = ibv_post_recv(qp_.qp(), &wr, &bad);
    if (post_ret != 0) {
        throw std::runtime_error(std::string("ibv_post_recv: ") +
                                 std::strerror(post_ret));
    }
}

void RdmaRecvProxy::run() noexcept {
    try {
        core::pin_current_thread_to_numa_node(fifo_numa_node_);
        constexpr int kPollBatch = 16;
        ibv_wc wcs[kPollBatch];
        const std::size_t max_if = max_inflight_;
        while (!stop_requested_.load(std::memory_order_acquire) &&
               !errors_->has_error()) {
            bool did_work = false;
            while (inflight_ < max_if &&
                   load_counter(control_.recv_head) + fifo_.slot_count >=
                       next_post_step_ + fifo_.step_increment) {
                pre_post_recv(next_post_step_ % fifo_.slot_count);
                next_post_step_ += fifo_.step_increment;
                ++inflight_;
                did_work = true;
            }

            const int n = ibv_poll_cq(qp_.cq(), kPollBatch, wcs);
            if (n < 0) {
                throw std::runtime_error("ibv_poll_cq failed");
            }
            for (int i = 0; i < n; ++i) {
                check_wc(wcs[i], identity_, next_complete_step_, errors_.get());
                const std::uint64_t slot = RdmaQp::wr_id_to_slot(wcs[i].wr_id);
                if (slot != next_complete_step_ % fifo_.slot_count) {
                    throw std::runtime_error(
                        "rdma recv completion has unexpected FIFO slot");
                }
                store_size(fifo_.slot_sizes + slot, wcs[i].byte_len);
                store_counter(control_.recv_tail,
                              next_complete_step_ + fifo_.step_increment);
                next_complete_step_ += fifo_.step_increment;
                step_.store(next_complete_step_, std::memory_order_release);
                --inflight_;
                did_work = true;
            }
            if (!did_work) {
                std::this_thread::yield();
            }
        }
    } catch (const std::exception& error) {
        if (!stop_requested_.load(std::memory_order_acquire)) {
            errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                                    error.what());
        }
    }
}

}  // namespace nano_nccl::transport::rdma
