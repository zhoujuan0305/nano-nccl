#include "transport/rdma/rdma_proxy.h"
#include "core/numa.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
#include <chrono>
#endif

#include <endian.h>
#include <immintrin.h>
#include <infiniband/verbs.h>

namespace nano_nccl::transport::rdma {

namespace {

void validate_write_targets(const RdmaWriteTargets& targets) {
    if (targets.remote_fifo_addr == 0 || targets.remote_fifo_bytes == 0 ||
        targets.local_cts == nullptr || targets.cts_slot_count == 0) {
        throw std::runtime_error(
            "rdma send proxy WriteCts targets are incomplete");
    }
}

void validate_cts_remote(const RdmaCtsRemote& remote) {
    if (remote.remote_cts_addr == 0 || remote.cts_slot_count == 0 ||
        remote.local_recv_fifo_addr == 0 || remote.local_shadow == nullptr ||
        remote.local_shadow_mr == nullptr) {
        throw std::runtime_error(
            "rdma recv proxy WriteCts CTS remote is incomplete");
    }
}

// High bit tags CTS send CQEs so the shared CQ can distinguish them from
// data WRITE_WITH_IMM receive completions (wr_id = absolute step).
constexpr std::uint64_t kCtsWrIdBit = 1ull << 63;

constexpr bool is_cts_wr_id(std::uint64_t wr_id) noexcept {
    return (wr_id & kCtsWrIdBit) != 0;
}

constexpr std::uint64_t make_cts_wr_id(std::uint64_t step) noexcept {
    return kCtsWrIdBit | step;
}

constexpr std::uint64_t cts_wr_id_to_step(std::uint64_t wr_id) noexcept {
    return wr_id & ~kCtsWrIdBit;
}

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

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
using SteadyClock = std::chrono::steady_clock;
using SteadyTp = SteadyClock::time_point;

std::uint64_t timeline_ns_delta(SteadyTp start, SteadyTp end) noexcept {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0;
}

std::size_t timeline_flight_index(std::uint64_t step, std::uint64_t step_inc,
                                  std::size_t max_if) noexcept {
    if (max_if == 0 || step_inc == 0) {
        return 0;
    }
    return static_cast<std::size_t>((step / step_inc) % max_if);
}
#endif

}  // namespace

RdmaDataPlane parse_rdma_data_plane_env() {
    const char* env = std::getenv("NANO_NCCL_RDMA_USE_WRITE");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0 ||
        std::strcmp(env, "false") == 0 || std::strcmp(env, "off") == 0) {
        return RdmaDataPlane::SendRecv;
    }
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
        std::strcmp(env, "on") == 0) {
        return RdmaDataPlane::WriteCts;
    }
    throw std::runtime_error("NANO_NCCL_RDMA_USE_WRITE must be 0 or 1");
}

bool parse_rdma_shared_progress_env() {
    const char* env = std::getenv("NANO_NCCL_RDMA_SHARED_PROGRESS");
    // Default dedicated: A/B on this platform showed equal-or-better nano busbw
    // vs shared; keep shared available via explicit 1/true/on.
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0 ||
        std::strcmp(env, "false") == 0 || std::strcmp(env, "off") == 0) {
        return false;
    }
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
        std::strcmp(env, "on") == 0) {
        return true;
    }
    throw std::runtime_error(
        "NANO_NCCL_RDMA_SHARED_PROGRESS must be unset/\"\"/0/1/true/false/on/off");
}

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
    : RdmaSendProxy(std::move(qp), fifo_mr, fifo, control, identity,
                    fifo_numa_node, std::move(errors), RdmaDataPlane::SendRecv,
                    RdmaWriteTargets{}) {}

RdmaSendProxy::RdmaSendProxy(RdmaQp qp, ibv_mr* fifo_mr, RdmaProxyFifo fifo,
                             RdmaSendControl control, RdmaProxyIdentity identity,
                             int fifo_numa_node,
                             std::shared_ptr<RdmaAsyncErrorState> errors,
                             RdmaDataPlane plane, RdmaWriteTargets write_targets)
    : qp_(std::move(qp)), fifo_mr_(fifo_mr), fifo_(fifo), control_(control),
      identity_(identity), fifo_numa_node_(fifo_numa_node),
      errors_(std::move(errors)), plane_(plane),
      write_targets_(write_targets) {
    validate_fifo(fifo_);
    if (qp_.qp() == nullptr || fifo_mr_ == nullptr ||
        control_.send_head == nullptr || control_.send_tail == nullptr ||
        errors_ == nullptr) {
        throw std::runtime_error("rdma send proxy has invalid state");
    }
    if (plane_ == RdmaDataPlane::WriteCts) {
        validate_write_targets(write_targets_);
    }
    max_inflight_ = max_inflight();
    const std::uint64_t head = load_counter(control_.send_head);
    next_post_step_ = head;
    next_complete_step_ = head;
    step_.store(head, std::memory_order_relaxed);
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    timeline_enabled_ = rdma_proxy_timeline_env_enabled();
    if (timeline_enabled_ && max_inflight_ > 0) {
        signaled_post_tp_ =
            std::make_unique<SteadyTp[]>(max_inflight_);
        signaled_post_valid_ = std::make_unique<bool[]>(max_inflight_);
        for (std::size_t i = 0; i < max_inflight_; ++i) {
            signaled_post_valid_[i] = false;
        }
    }
#endif
}

RdmaSendProxy::~RdmaSendProxy() {
    stop();
    join();
    dump_timeline_if_enabled();
}

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
const char* RdmaSendProxy::timeline_plane_label() const noexcept {
    return plane_ == RdmaDataPlane::WriteCts ? "WriteCts" : "SendRecv";
}

void RdmaSendProxy::dump_timeline_if_enabled() const {
    if (!timeline_enabled_ || timeline_dumped_) {
        return;
    }
    timeline_dumped_ = true;
    // v2 per-event metrics only — lifetime stage spins poison interpretation.
    const std::string text = timeline_.format_v2(
        "send", identity_.source_rank, identity_.destination_rank,
        identity_.channel, timeline_plane_label());
    if (!text.empty()) {
        std::fprintf(stderr, "%s", text.c_str());
    }
}
#endif

void RdmaSendProxy::prepare() {
    if (prepared_) {
        throw std::runtime_error("rdma send proxy already prepared");
    }
    if (thread_.joinable()) {
        throw std::runtime_error("rdma send proxy already started");
    }
    prepared_ = true;
}

bool RdmaSendProxy::progress() noexcept {
    if (stop_requested_.load(std::memory_order_acquire) || errors_->has_error()) {
        return false;
    }
    if (plane_ == RdmaDataPlane::WriteCts) {
        return progress_write_cts();
    }
    return progress_send_recv();
}

void RdmaSendProxy::start() {
    if (thread_.joinable()) {
        throw std::runtime_error("rdma send proxy already started");
    }
    prepare();
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
        while (!stop_requested_.load(std::memory_order_acquire) &&
               !errors_->has_error()) {
            if (!progress()) {
                _mm_pause();
            }
        }
    } catch (const std::exception& error) {
        if (!stop_requested_.load(std::memory_order_acquire)) {
            errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                                    error.what());
        }
    }
}

bool RdmaSendProxy::progress_send_recv() noexcept {
    try {
        constexpr int kPollBatch = 16;
        // Signal every Kth post plus credit-full and burst-tail WRs so unsignaled
        // WQEs still free send_head in order without a CQ entry per slice.
        constexpr std::uint64_t kSignalEvery = 4;
        ibv_wc wcs[kPollBatch];
        const std::size_t max_if = max_inflight_;
        const std::uint64_t step_inc = fifo_.step_increment;
        {
            bool did_work = false;
            while (inflight_ < max_if) {
                const std::uint64_t tail = load_counter(control_.send_tail);
                const bool tail_ready = tail > next_post_step_;
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                if (timeline_enabled_ && tail_ready && !send_tail_armed_) {
                    send_tail_tp_ = SteadyClock::now();
                    send_tail_armed_ = true;
                    // Gap after prior data post until this step's tail arms
                    // (dominates send_inter_post under PIVOT=TAIL).
                    if (last_data_post_valid_) {
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::SendPostToNextTail,
                            timeline_ns_delta(last_data_post_tp_, send_tail_tp_));
                    }
                }
#endif
                if (!tail_ready) {
                    break;
                }
                const std::size_t slot = next_post_step_ % fifo_.slot_count;
                const std::uint32_t payload_bytes =
                    load_size(fifo_.slot_sizes + slot);
                if (payload_bytes > fifo_.slot_bytes) {
                    throw std::runtime_error(
                        "rdma proxy send payload exceeds FIFO capacity");
                }

                // 0-byte Simple trailing slices: local credit only — no RDMA op.
                // Kernel elides most empties; this is the host-side guard when a
                // 0-byte size is still published, and keeps send_head in lockstep.
                // Only elide at the completion head so send_head stays ordered with
                // prior in-flight SENDs; otherwise break and drain CQ first.
                if (payload_bytes == 0) {
                    if (next_post_step_ != next_complete_step_) {
                        break;
                    }
                    zero_payload_posts_.fetch_add(1, std::memory_order_relaxed);
                    store_counter(control_.send_head,
                                  next_post_step_ + step_inc);
                    next_post_step_ += step_inc;
                    next_complete_step_ += step_inc;
                    step_.store(next_complete_step_, std::memory_order_release);
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                    send_tail_armed_ = false;
                    send_cts_armed_ = false;
#endif
                    did_work = true;
                    continue;
                }

                posts_.fetch_add(1, std::memory_order_relaxed);

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                // Q1: local handle after can_post (same iteration; no spin).
                const SteadyTp t_ready =
                    timeline_enabled_ ? SteadyClock::now() : SteadyTp{};
#endif
                bool signal = false;
                ibv_sge sge{};
                sge.length = payload_bytes;
                sge.addr = reinterpret_cast<std::uintptr_t>(
                    fifo_.data + slot * fifo_.slot_bytes);
                sge.lkey = fifo_mr_->lkey;

                // wr_id = absolute step so one signaled CQE can batch-advance
                // send_head across prior unsignaled WRs (RC in-order).
                const std::uint64_t post_step = next_post_step_;
                const bool more_pending = tail > post_step + step_inc;
                // 0-byte elide only runs at the completion head. If a later
                // published step is empty, this WR must be SIGNALED or the
                // elide path waits forever on an unsignaled CQE.
                bool next_zero_needs_head = false;
                if (more_pending) {
                    const std::size_t next_slot =
                        (post_step + step_inc) % fifo_.slot_count;
                    next_zero_needs_head =
                        load_size(fifo_.slot_sizes + next_slot) == 0;
                }
                signal = (inflight_ + 1 >= max_if) || !more_pending ||
                         next_zero_needs_head ||
                         ((post_step / step_inc) % kSignalEvery == 0);

                ibv_send_wr wr{};
                wr.wr_id = RdmaQp::slot_to_wr_id(post_step);
                wr.sg_list = &sge;
                wr.num_sge = 1;
                wr.opcode = IBV_WR_SEND;
                wr.send_flags = signal ? IBV_SEND_SIGNALED : 0;
                wr.next = nullptr;

                ibv_send_wr* bad = nullptr;
                const int post_ret = ibv_post_send(qp_.qp(), &wr, &bad);
                if (post_ret != 0) {
                    throw std::runtime_error(std::string("ibv_post_send: ") +
                                             std::strerror(post_ret));
                }
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                if (timeline_enabled_) {
                    const SteadyTp post_done = SteadyClock::now();
                    timeline_.record_event(
                        RdmaProxyTimelineEvent::SendReadyToPost,
                        timeline_ns_delta(t_ready, post_done));
                    if (send_tail_armed_) {
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::SendTailArmToPost,
                            timeline_ns_delta(send_tail_tp_, post_done));
                        send_tail_armed_ = false;
                    }
                    // Q2: previous successful data post → this post.
                    if (last_data_post_valid_) {
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::SendInterPost,
                            timeline_ns_delta(last_data_post_tp_, post_done));
                    }
                    last_data_post_tp_ = post_done;
                    last_data_post_valid_ = true;
                    // Depth: inflight before ++ ; free slots from host credit view.
                    timeline_.inflight_at_post.record(
                        static_cast<std::uint64_t>(inflight_));
                    const std::uint64_t free_slots =
                        (next_complete_step_ +
                         static_cast<std::uint64_t>(fifo_.slot_count) - tail) /
                        step_inc;
                    timeline_.free_slots_at_post.record(free_slots);
                    // Q3 signaled-only: unsignaled WRs attribute to next CQE.
                    if (signal && signaled_post_tp_ && signaled_post_valid_) {
                        const std::size_t idx =
                            timeline_flight_index(post_step, step_inc, max_if);
                        signaled_post_tp_[idx] = post_done;
                        signaled_post_valid_[idx] = true;
                    }
                }
#endif
                next_post_step_ += step_inc;
                ++inflight_;
                did_work = true;
            }

            const int n = ibv_poll_cq(qp_.cq(), kPollBatch, wcs);
            if (n < 0) {
                throw std::runtime_error("ibv_poll_cq failed");
            }
            for (int i = 0; i < n; ++i) {
                check_wc(wcs[i], identity_, next_complete_step_, errors_.get());
                const std::uint64_t completed_through =
                    RdmaQp::wr_id_to_slot(wcs[i].wr_id);
                if (completed_through < next_complete_step_ ||
                    (completed_through - next_complete_step_) % step_inc != 0) {
                    throw std::runtime_error(
                        "rdma send completion has unexpected step");
                }
                const std::uint64_t advanced =
                    (completed_through - next_complete_step_) / step_inc + 1;
                if (advanced > inflight_) {
                    throw std::runtime_error(
                        "rdma send completion advances past inflight");
                }
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                // Q3 signaled-only: full interval attributed to the signaled WR.
                if (timeline_enabled_ && signaled_post_tp_ &&
                    signaled_post_valid_) {
                    const std::size_t idx = timeline_flight_index(
                        completed_through, step_inc, max_if);
                    if (signaled_post_valid_[idx]) {
                        const SteadyTp now = SteadyClock::now();
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::SendPostToCq,
                            timeline_ns_delta(signaled_post_tp_[idx], now));
                        signaled_post_valid_[idx] = false;
                    }
                }
                if (timeline_enabled_) {
                    timeline_.inflight_at_cqe.record(
                        static_cast<std::uint64_t>(inflight_));
                }
#endif
                // RC in-order: a signaled WR covers all prior unsignaled WRs.
                next_complete_step_ = completed_through + step_inc;
                store_counter(control_.send_head, next_complete_step_);
                step_.store(next_complete_step_, std::memory_order_release);
                inflight_ -= static_cast<std::size_t>(advanced);
                did_work = true;
            }
            return did_work;
        }
    } catch (const std::exception& error) {
        if (!stop_requested_.load(std::memory_order_acquire)) {
            errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                                    error.what());
        }
    }
    return false;
}

bool RdmaSendProxy::progress_write_cts() noexcept {
    try {
        constexpr int kPollBatch = 16;
        constexpr std::uint64_t kSignalEvery = 4;
        ibv_wc wcs[kPollBatch];
        const std::size_t max_if = max_inflight_;
        const std::uint64_t step_inc = fifo_.step_increment;
        {
            bool did_work = false;
            while (inflight_ < max_if) {
                const std::size_t slot = next_post_step_ % fifo_.slot_count;
                const std::uint64_t tail = load_counter(control_.send_tail);
                const bool tail_ready = tail > next_post_step_;
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                if (timeline_enabled_ && tail_ready && !send_tail_armed_) {
                    send_tail_tp_ = SteadyClock::now();
                    send_tail_armed_ = true;
                    // Gap after prior data post until this step's tail arms
                    // (dominates send_inter_post under PIVOT=TAIL).
                    if (last_data_post_valid_) {
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::SendPostToNextTail,
                            timeline_ns_delta(last_data_post_tp_, send_tail_tp_));
                    }
                }
#endif
                if (!tail_ready) {
                    break;
                }
                const std::uint32_t payload_bytes =
                    load_size(fifo_.slot_sizes + slot);
                if (payload_bytes > fifo_.slot_bytes) {
                    throw std::runtime_error(
                        "rdma proxy send payload exceeds FIFO capacity");
                }

                // 0-byte Simple trailing slices: local credit only — no RDMA.
                // Elide only at the completion head so ordering with in-flight
                // WRITEs is preserved.
                if (payload_bytes == 0) {
                    if (next_post_step_ != next_complete_step_) {
                        break;
                    }
                    zero_payload_posts_.fetch_add(1, std::memory_order_relaxed);
                    store_counter(control_.send_head,
                                  next_post_step_ + step_inc);
                    next_post_step_ += step_inc;
                    next_complete_step_ += step_inc;
                    step_.store(next_complete_step_, std::memory_order_release);
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                    send_tail_armed_ = false;
                    send_cts_armed_ = false;
#endif
                    did_work = true;
                    continue;
                }

                if (slot >= write_targets_.cts_slot_count) {
                    throw std::runtime_error(
                        "rdma WriteCts slot exceeds CTS FIFO");
                }
                RdmaCtsSlot* cts = write_targets_.local_cts + slot;
                const std::uint32_t ready =
                    __atomic_load_n(&cts->ready, __ATOMIC_ACQUIRE);
                const std::uint64_t step_tag =
                    ready == 1 ? __atomic_load_n(&cts->step_tag, __ATOMIC_RELAXED)
                               : 0;
                const bool cts_ready =
                    ready == 1 && step_tag == next_post_step_;
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                if (timeline_enabled_ && cts_ready && !send_cts_armed_) {
                    send_cts_tp_ = SteadyClock::now();
                    send_cts_armed_ = true;
                    if (send_tail_armed_) {
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::SendTailToCtsReady,
                            timeline_ns_delta(send_tail_tp_, send_cts_tp_));
                    }
                }
#endif
                if (!cts_ready) {
                    if (ready == 1 && step_tag != next_post_step_) {
                        throw std::runtime_error(
                            "rdma WriteCts CTS step_tag mismatch");
                    }
                    // Tail ready but CTS not yet — keep arm clocks across waits.
                    break;
                }
                const std::uint64_t remote_addr =
                    __atomic_load_n(&cts->raddr, __ATOMIC_RELAXED);
                const std::uint32_t rkey =
                    __atomic_load_n(&cts->rkey, __ATOMIC_RELAXED);
                const std::uint32_t nbytes_cap =
                    __atomic_load_n(&cts->nbytes, __ATOMIC_RELAXED);
                if (remote_addr == 0 || payload_bytes > nbytes_cap) {
                    throw std::runtime_error(
                        "rdma WriteCts CTS target invalid for payload");
                }

                posts_.fetch_add(1, std::memory_order_relaxed);

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                // Q1: can_post (tail+CTS) true → post return, same iteration.
                const SteadyTp t_ready =
                    timeline_enabled_ ? SteadyClock::now() : SteadyTp{};
#endif
                bool signal = false;
                ibv_sge sge{};
                sge.length = payload_bytes;
                sge.addr = reinterpret_cast<std::uintptr_t>(
                    fifo_.data + slot * fifo_.slot_bytes);
                sge.lkey = fifo_mr_->lkey;

                const std::uint64_t post_step = next_post_step_;
                const bool more_pending = tail > post_step + step_inc;
                bool next_zero_needs_head = false;
                if (more_pending) {
                    const std::size_t next_slot =
                        (post_step + step_inc) % fifo_.slot_count;
                    next_zero_needs_head =
                        load_size(fifo_.slot_sizes + next_slot) == 0;
                }
                signal = (inflight_ + 1 >= max_if) || !more_pending ||
                         next_zero_needs_head ||
                         ((post_step / step_inc) % kSignalEvery == 0);

                ibv_send_wr wr{};
                wr.wr_id = RdmaQp::slot_to_wr_id(post_step);
                wr.sg_list = &sge;
                wr.num_sge = 1;
                wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
                wr.send_flags = signal ? IBV_SEND_SIGNALED : 0;
                wr.imm_data = htobe32(payload_bytes);
                wr.wr.rdma.remote_addr = remote_addr;
                wr.wr.rdma.rkey = rkey;
                wr.next = nullptr;

                ibv_send_wr* bad = nullptr;
                const int post_ret = ibv_post_send(qp_.qp(), &wr, &bad);
                if (post_ret != 0) {
                    throw std::runtime_error(
                        std::string("ibv_post_send WRITE: ") +
                        std::strerror(post_ret));
                }
                // Clear ready immediately after a successful post. raddr/rkey
                // were already snapshotted into the WR; waiting until CQE would
                // race a next-round CTS that reuses the same slot index.
                __atomic_store_n(&cts->ready, 0U, __ATOMIC_RELEASE);
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                if (timeline_enabled_) {
                    const SteadyTp post_done = SteadyClock::now();
                    timeline_.record_event(
                        RdmaProxyTimelineEvent::SendReadyToPost,
                        timeline_ns_delta(t_ready, post_done));
                    if (send_tail_armed_) {
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::SendTailArmToPost,
                            timeline_ns_delta(send_tail_tp_, post_done));
                    }
                    if (send_cts_armed_) {
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::SendCtsArmToPost,
                            timeline_ns_delta(send_cts_tp_, post_done));
                    }
                    send_tail_armed_ = false;
                    send_cts_armed_ = false;
                    if (last_data_post_valid_) {
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::SendInterPost,
                            timeline_ns_delta(last_data_post_tp_, post_done));
                    }
                    last_data_post_tp_ = post_done;
                    last_data_post_valid_ = true;
                    timeline_.inflight_at_post.record(
                        static_cast<std::uint64_t>(inflight_));
                    const std::uint64_t free_slots =
                        (next_complete_step_ +
                         static_cast<std::uint64_t>(fifo_.slot_count) - tail) /
                        step_inc;
                    timeline_.free_slots_at_post.record(free_slots);
                    // Q3 signaled-only post→CQ attribution.
                    if (signal && signaled_post_tp_ && signaled_post_valid_) {
                        const std::size_t idx =
                            timeline_flight_index(post_step, step_inc, max_if);
                        signaled_post_tp_[idx] = post_done;
                        signaled_post_valid_[idx] = true;
                    }
                }
#endif
                next_post_step_ += step_inc;
                ++inflight_;
                did_work = true;
            }

            const int n = ibv_poll_cq(qp_.cq(), kPollBatch, wcs);
            if (n < 0) {
                throw std::runtime_error("ibv_poll_cq failed");
            }
            for (int i = 0; i < n; ++i) {
                check_wc(wcs[i], identity_, next_complete_step_, errors_.get());
                const std::uint64_t completed_through =
                    RdmaQp::wr_id_to_slot(wcs[i].wr_id);
                if (completed_through < next_complete_step_ ||
                    (completed_through - next_complete_step_) % step_inc != 0) {
                    throw std::runtime_error(
                        "rdma WriteCts send completion has unexpected step");
                }
                const std::uint64_t advanced =
                    (completed_through - next_complete_step_) / step_inc + 1;
                if (advanced > inflight_) {
                    throw std::runtime_error(
                        "rdma WriteCts send completion advances past inflight");
                }
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                // Q3 signaled-only: full interval attributed to the signaled WR.
                if (timeline_enabled_ && signaled_post_tp_ &&
                    signaled_post_valid_) {
                    const std::size_t idx = timeline_flight_index(
                        completed_through, step_inc, max_if);
                    if (signaled_post_valid_[idx]) {
                        const SteadyTp now = SteadyClock::now();
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::SendPostToCq,
                            timeline_ns_delta(signaled_post_tp_[idx], now));
                        signaled_post_valid_[idx] = false;
                    }
                }
                if (timeline_enabled_) {
                    timeline_.inflight_at_cqe.record(
                        static_cast<std::uint64_t>(inflight_));
                }
#endif
                // ready already cleared at post; CQE only advances credit.
                next_complete_step_ = completed_through + step_inc;
                store_counter(control_.send_head, next_complete_step_);
                step_.store(next_complete_step_, std::memory_order_release);
                inflight_ -= static_cast<std::size_t>(advanced);
                did_work = true;
            }
            return did_work;
        }
    } catch (const std::exception& error) {
        if (!stop_requested_.load(std::memory_order_acquire)) {
            errors_->record_failure(identity_,
                                    step_.load(std::memory_order_acquire),
                                    error.what());
        }
    }
    return false;
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
                             std::shared_ptr<RdmaAsyncErrorState> errors,
                             bool elide_zero_payload)
    : RdmaRecvProxy(std::move(qp), fifo_mr, fifo, control, identity,
                    fifo_numa_node, std::move(errors), RdmaDataPlane::SendRecv,
                    RdmaCtsRemote{}, elide_zero_payload) {}

RdmaRecvProxy::RdmaRecvProxy(RdmaQp qp, ibv_mr* fifo_mr, RdmaProxyFifo fifo,
                             RdmaRecvControl control, RdmaProxyIdentity identity,
                             int fifo_numa_node,
                             std::shared_ptr<RdmaAsyncErrorState> errors,
                             RdmaDataPlane plane, RdmaCtsRemote cts_remote,
                             bool elide_zero_payload)
    : qp_(std::move(qp)), fifo_mr_(fifo_mr), fifo_(fifo), control_(control),
      identity_(identity), fifo_numa_node_(fifo_numa_node),
      errors_(std::move(errors)), plane_(plane), cts_remote_(cts_remote),
      elide_zero_payload_(elide_zero_payload) {
    validate_fifo(fifo_);
    if (qp_.qp() == nullptr || fifo_mr_ == nullptr ||
        control_.recv_head == nullptr || control_.recv_tail == nullptr ||
        errors_ == nullptr) {
        throw std::runtime_error("rdma recv proxy has invalid state");
    }
    if (plane_ == RdmaDataPlane::WriteCts) {
        validate_cts_remote(cts_remote_);
        if (cts_remote_.cts_slot_count != fifo_.slot_count) {
            throw std::runtime_error(
                "rdma WriteCts CTS slot_count must match FIFO slot_count");
        }
    }
    max_inflight_ = max_inflight();
    if (plane_ == RdmaDataPlane::WriteCts) {
        // One scratch byte per concurrent IMM RECV WQE (HCA may reject 0-len).
        imm_scratch_ = std::make_unique<std::uint8_t[]>(max_inflight_);
        imm_scratch_mr_ =
            ibv_reg_mr(fifo_mr_->pd, imm_scratch_.get(), max_inflight_,
                       IBV_ACCESS_LOCAL_WRITE);
        if (imm_scratch_mr_ == nullptr) {
            throw std::runtime_error(std::string("ibv_reg_mr imm scratch: ") +
                                     std::strerror(errno));
        }
    }
    const std::uint64_t tail = load_counter(control_.recv_tail);
    next_post_step_ = tail;
    next_complete_step_ = tail;
    next_cts_step_ = tail;
    next_cts_complete_step_ = tail;
    step_.store(tail, std::memory_order_relaxed);
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    timeline_enabled_ = rdma_proxy_timeline_env_enabled();
#endif
}

RdmaRecvProxy::~RdmaRecvProxy() {
    stop();
    join();
    dump_timeline_if_enabled();
    if (imm_scratch_mr_ != nullptr) {
        ibv_dereg_mr(imm_scratch_mr_);
        imm_scratch_mr_ = nullptr;
    }
}

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
const char* RdmaRecvProxy::timeline_plane_label() const noexcept {
    return plane_ == RdmaDataPlane::WriteCts ? "WriteCts" : "SendRecv";
}

void RdmaRecvProxy::dump_timeline_if_enabled() const {
    if (!timeline_enabled_ || timeline_dumped_) {
        return;
    }
    timeline_dumped_ = true;
    // v2 per-event metrics only — lifetime stage spins poison interpretation.
    const std::string text = timeline_.format_v2(
        "recv", identity_.source_rank, identity_.destination_rank,
        identity_.channel, timeline_plane_label());
    if (!text.empty()) {
        std::fprintf(stderr, "%s", text.c_str());
    }
}
#endif

bool RdmaRecvProxy::try_elide_zero_payload() {
    if (!elide_zero_payload_) return false;
    if (next_post_step_ != next_complete_step_) return false;
    if (load_counter(control_.recv_head) + fifo_.slot_count <
        next_post_step_ + fifo_.step_increment) {
        return false;
    }
    const std::size_t slot = next_post_step_ % fifo_.slot_count;
    if (load_size(fifo_.slot_sizes + slot) != 0) return false;
    store_size(fifo_.slot_sizes + slot, 0);
    store_counter(control_.recv_tail,
                  next_complete_step_ + fifo_.step_increment);
    next_post_step_ += fifo_.step_increment;
    next_complete_step_ += fifo_.step_increment;
    step_.store(next_complete_step_, std::memory_order_release);
    return true;
}

void RdmaRecvProxy::prepare() {
    if (prepared_) {
        throw std::runtime_error("rdma recv proxy already prepared");
    }
    if (thread_.joinable()) {
        throw std::runtime_error("rdma recv proxy already started");
    }
    if (plane_ == RdmaDataPlane::WriteCts) {
        // Pre-post IMM RQ WQEs and initial CTS. Credit matches SEND/socket:
        // free while recv_head + slot_count covers the next step (head starts
        // at 0; GPU advances head as it consumes).
        const std::uint64_t step_inc = fifo_.step_increment;
        while (inflight_ < max_inflight_ &&
               load_counter(control_.recv_head) + fifo_.slot_count >=
                   next_post_step_ + step_inc) {
            pre_post_imm_recv(next_post_step_);
            next_post_step_ += step_inc;
            ++inflight_;
        }
        while (inflight_cts_ < max_inflight_ &&
               load_counter(control_.recv_head) + fifo_.slot_count >=
                   next_cts_step_ + step_inc) {
            post_cts(next_cts_step_);
            next_cts_step_ += step_inc;
            ++inflight_cts_;
        }
        prepared_ = true;
        return;
    }
    // Simple 每个 slice 以 step_increment 前进；预投递必须与 kernel 的
    // slot 序列一致，并在 credit 与 max_inflight 约束下尽可能填满 RQ。
    while (inflight_ < max_inflight_ &&
           load_counter(control_.recv_head) + fifo_.slot_count >=
               next_post_step_ + fifo_.step_increment) {
        if (try_elide_zero_payload()) continue;
        if (elide_zero_payload_ &&
            load_size(fifo_.slot_sizes +
                      (next_post_step_ % fifo_.slot_count)) == 0) {
            break;
        }
        pre_post_recv(next_post_step_ % fifo_.slot_count);
        next_post_step_ += fifo_.step_increment;
        ++inflight_;
    }
    prepared_ = true;
}

bool RdmaRecvProxy::progress() noexcept {
    if (stop_requested_.load(std::memory_order_acquire) || errors_->has_error()) {
        return false;
    }
    if (plane_ == RdmaDataPlane::WriteCts) {
        return progress_write_cts();
    }
    return progress_send_recv();
}

void RdmaRecvProxy::start() {
    if (thread_.joinable()) {
        throw std::runtime_error("rdma recv proxy already started");
    }
    prepare();
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

void RdmaRecvProxy::pre_post_imm_recv(std::uint64_t step) {
    // WRITE_WITH_IMM payload lands in the remote FIFO via RDMA; the RQ WQE
    // only harvests the IMM CQE. Scratch SGE satisfies HCAs that reject 0-len.
    const std::size_t scratch_idx = step % max_inflight_;
    ibv_sge sge{};
    sge.addr = reinterpret_cast<std::uintptr_t>(imm_scratch_.get() + scratch_idx);
    sge.length = 1;
    sge.lkey = imm_scratch_mr_->lkey;

    ibv_recv_wr wr{};
    wr.wr_id = RdmaQp::slot_to_wr_id(step);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.next = nullptr;

    ibv_recv_wr* bad = nullptr;
    const int post_ret = ibv_post_recv(qp_.qp(), &wr, &bad);
    if (post_ret != 0) {
        throw std::runtime_error(std::string("ibv_post_recv IMM: ") +
                                 std::strerror(post_ret));
    }
}

void RdmaRecvProxy::post_cts(std::uint64_t step) {
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    // Caller already verified credit; measure local CTS handle only.
    const SteadyTp t_ready =
        timeline_enabled_ ? SteadyClock::now() : SteadyTp{};
#endif
    const std::size_t slot = step % cts_remote_.cts_slot_count;
    RdmaCtsSlot* shadow = cts_remote_.local_shadow + slot;
    shadow->raddr =
        cts_remote_.local_recv_fifo_addr +
        static_cast<std::uint64_t>(slot) * fifo_.slot_bytes;
    shadow->rkey = cts_remote_.local_recv_fifo_rkey;
    shadow->nbytes = static_cast<std::uint32_t>(fifo_.slot_bytes);
    shadow->step_tag = step;
    shadow->reserved = 0;
    // ready last so a remote observer that sees ready==1 also sees fields.
    __atomic_store_n(&shadow->ready, 1U, __ATOMIC_RELEASE);

    // Selective SIGNAL mirrors the data path: reclaim SQ via RC in-order
    // batch on a subset of CQEs (credit-full, every N, or burst tail).
    constexpr std::uint64_t kSignalEvery = 4;
    const std::uint64_t step_inc = fifo_.step_increment;
    const bool more_credit =
        (inflight_cts_ + 1 < max_inflight_) &&
        (load_counter(control_.recv_head) + fifo_.slot_count >=
         step + 2 * step_inc);
    const bool signal = (inflight_cts_ + 1 >= max_inflight_) || !more_credit ||
                        ((step / step_inc) % kSignalEvery == 0);

    ibv_sge sge{};
    sge.addr = reinterpret_cast<std::uintptr_t>(shadow);
    sge.length = static_cast<std::uint32_t>(sizeof(RdmaCtsSlot));

    ibv_send_wr wr{};
    wr.wr_id = make_cts_wr_id(step);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    // Prefer INLINE when the QP grants enough room so the HCA copies from
    // the WR and skips MR fetch. Combine with optional SIGNALED.
    int send_flags = 0;
    if (sizeof(RdmaCtsSlot) <= qp_.max_inline_data()) {
        send_flags = IBV_SEND_INLINE;
        // lkey ignored for inline; addr/length identify the host buffer.
    } else {
        sge.lkey = cts_remote_.local_shadow_mr->lkey;
    }
    if (signal) {
        send_flags |= IBV_SEND_SIGNALED;
    }
    wr.send_flags = send_flags;
    wr.wr.rdma.remote_addr =
        cts_remote_.remote_cts_addr +
        static_cast<std::uint64_t>(slot) * sizeof(RdmaCtsSlot);
    wr.wr.rdma.rkey = cts_remote_.remote_cts_rkey;
    wr.next = nullptr;

    ibv_send_wr* bad = nullptr;
    const int post_ret = ibv_post_send(qp_.qp(), &wr, &bad);
    if (post_ret != 0) {
        throw std::runtime_error(std::string("ibv_post_send CTS: ") +
                                 std::strerror(post_ret));
    }
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    if (timeline_enabled_) {
        const SteadyTp post_done = SteadyClock::now();
        timeline_.record_event(RdmaProxyTimelineEvent::CtsReadyToPost,
                               timeline_ns_delta(t_ready, post_done));
        if (last_cts_post_valid_) {
            timeline_.record_event(
                RdmaProxyTimelineEvent::CtsInterPost,
                timeline_ns_delta(last_cts_post_tp_, post_done));
        }
        last_cts_post_tp_ = post_done;
        last_cts_post_valid_ = true;
    }
#endif
}

void RdmaRecvProxy::run() noexcept {
    try {
        core::pin_current_thread_to_numa_node(fifo_numa_node_);
        while (!stop_requested_.load(std::memory_order_acquire) &&
               !errors_->has_error()) {
            if (!progress()) {
                _mm_pause();
            }
        }
    } catch (const std::exception& error) {
        if (!stop_requested_.load(std::memory_order_acquire)) {
            errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                                    error.what());
        }
    }
}

bool RdmaRecvProxy::progress_send_recv() noexcept {
    try {
        constexpr int kPollBatch = 16;
        ibv_wc wcs[kPollBatch];
        const std::size_t max_if = max_inflight_;
        {
            bool did_work = false;
            while (inflight_ < max_if &&
                   load_counter(control_.recv_head) + fifo_.slot_count >=
                       next_post_step_ + fifo_.step_increment) {
                if (try_elide_zero_payload()) {
                    did_work = true;
                    continue;
                }
                if (elide_zero_payload_ &&
                    load_size(fifo_.slot_sizes +
                              (next_post_step_ % fifo_.slot_count)) == 0) {
                    break;
                }
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
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                // Q4: CQE seen → after recv_tail store (local publish handle).
                const SteadyTp t_cq =
                    timeline_enabled_ ? SteadyClock::now() : SteadyTp{};
#endif
                store_size(fifo_.slot_sizes + slot, wcs[i].byte_len);
                store_counter(control_.recv_tail,
                              next_complete_step_ + fifo_.step_increment);
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                if (timeline_enabled_) {
                    const SteadyTp t_pub = SteadyClock::now();
                    timeline_.record_event(
                        RdmaProxyTimelineEvent::RecvCqToPublish,
                        timeline_ns_delta(t_cq, t_pub));
                    // Q5: previous publish → this publish.
                    if (last_publish_valid_) {
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::RecvInterPublish,
                            timeline_ns_delta(last_publish_tp_, t_pub));
                    }
                    last_publish_tp_ = t_pub;
                    last_publish_valid_ = true;
                }
#endif
                next_complete_step_ += fifo_.step_increment;
                step_.store(next_complete_step_, std::memory_order_release);
                --inflight_;
                did_work = true;
            }
            return did_work;
        }
    } catch (const std::exception& error) {
        if (!stop_requested_.load(std::memory_order_acquire)) {
            errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                                    error.what());
        }
    }
    return false;
}

bool RdmaRecvProxy::progress_write_cts() noexcept {
    try {
        constexpr int kPollBatch = 16;
        ibv_wc wcs[kPollBatch];
        const std::size_t max_if = max_inflight_;
        const std::uint64_t step_inc = fifo_.step_increment;
        {
            bool did_work = false;

            // Refresh CTS for free slots (recv_head + slot_count credit).
            while (inflight_cts_ < max_if &&
                   load_counter(control_.recv_head) + fifo_.slot_count >=
                       next_cts_step_ + step_inc) {
                post_cts(next_cts_step_);
                next_cts_step_ += step_inc;
                ++inflight_cts_;
                did_work = true;
            }

            // Keep IMM RQ depth matched to available credit.
            while (inflight_ < max_if &&
                   load_counter(control_.recv_head) + fifo_.slot_count >=
                       next_post_step_ + step_inc) {
                pre_post_imm_recv(next_post_step_);
                next_post_step_ += step_inc;
                ++inflight_;
                did_work = true;
            }

            const int n = ibv_poll_cq(qp_.cq(), kPollBatch, wcs);
            if (n < 0) {
                throw std::runtime_error("ibv_poll_cq failed");
            }
            for (int i = 0; i < n; ++i) {
                check_wc(wcs[i], identity_, next_complete_step_, errors_.get());
                if (is_cts_wr_id(wcs[i].wr_id)) {
                    const std::uint64_t completed_through =
                        cts_wr_id_to_step(wcs[i].wr_id);
                    if (completed_through < next_cts_complete_step_ ||
                        (completed_through - next_cts_complete_step_) %
                                step_inc !=
                            0) {
                        throw std::runtime_error(
                            "rdma WriteCts CTS completion has unexpected step");
                    }
                    const std::uint64_t advanced =
                        (completed_through - next_cts_complete_step_) /
                            step_inc +
                        1;
                    if (advanced > inflight_cts_) {
                        throw std::runtime_error(
                            "rdma WriteCts CTS completion advances past "
                            "inflight");
                    }
                    // RC in-order: signaled CTS covers prior unsignaled CTS.
                    next_cts_complete_step_ = completed_through + step_inc;
                    inflight_cts_ -= static_cast<std::size_t>(advanced);
                    did_work = true;
                    continue;
                }

                // Data path only accepts WRITE_WITH_IMM; plain RECV has no imm.
                if (wcs[i].opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
                    throw std::runtime_error(
                        "rdma WriteCts unexpected recv WC opcode");
                }
                const std::uint64_t completed_step =
                    RdmaQp::wr_id_to_slot(wcs[i].wr_id);
                if (completed_step != next_complete_step_) {
                    throw std::runtime_error(
                        "rdma WriteCts IMM completion has unexpected step");
                }
                const std::size_t slot = completed_step % fifo_.slot_count;
                const std::uint32_t payload =
                    be32toh(static_cast<std::uint32_t>(wcs[i].imm_data));
                if (payload > fifo_.slot_bytes) {
                    throw std::runtime_error(
                        "rdma WriteCts IMM payload exceeds FIFO capacity");
                }
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                // Q4: CQE seen → after recv_tail store.
                const SteadyTp t_cq =
                    timeline_enabled_ ? SteadyClock::now() : SteadyTp{};
#endif
                store_size(fifo_.slot_sizes + slot, payload);
                store_counter(control_.recv_tail,
                              next_complete_step_ + step_inc);
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
                if (timeline_enabled_) {
                    const SteadyTp t_pub = SteadyClock::now();
                    timeline_.record_event(
                        RdmaProxyTimelineEvent::RecvCqToPublish,
                        timeline_ns_delta(t_cq, t_pub));
                    if (last_publish_valid_) {
                        timeline_.record_event(
                            RdmaProxyTimelineEvent::RecvInterPublish,
                            timeline_ns_delta(last_publish_tp_, t_pub));
                    }
                    last_publish_tp_ = t_pub;
                    last_publish_valid_ = true;
                }
#endif
                next_complete_step_ += step_inc;
                step_.store(next_complete_step_, std::memory_order_release);
                --inflight_;
                did_work = true;
            }
            return did_work;
        }
    } catch (const std::exception& error) {
        if (!stop_requested_.load(std::memory_order_acquire)) {
            errors_->record_failure(identity_,
                                    step_.load(std::memory_order_acquire),
                                    error.what());
        }
    }
    return false;
}

}  // namespace nano_nccl::transport::rdma
