#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>

namespace nano_nccl::transport::rdma {

enum class RdmaProxyTimelineStage : int {
    SendTailWait = 0,
    SendPost,
    SendCqWait,
    RecvCqWait,
    RecvPublish,
    CtsPost,
    CtsCq,
    kCount
};

enum class RdmaProxyTimelineEvent : int {
    SendReadyToPost = 0,
    SendInterPost,
    SendPostToCq,
    RecvCqToPublish,
    RecvInterPublish,
    CtsReadyToPost,
    CtsInterPost,
    SendTailArmToPost,
    SendCtsArmToPost,
    SendTailToCtsReady,
    SendPostToNextTail,
    kCount
};

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)

const char* rdma_proxy_timeline_stage_name(RdmaProxyTimelineStage stage);
const char* rdma_proxy_timeline_event_name(RdmaProxyTimelineEvent event);

struct RdmaProxyTimelineEventStats {
    static constexpr int kHistBins = 12;

    std::uint64_t count = 0;
    std::uint64_t sum_ns = 0;
    std::uint64_t min_ns = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_ns = 0;
    std::uint64_t hist[kHistBins]{};

    void record(std::uint64_t ns) noexcept;
};

// Depth histograms (not time): bins 0,1,2,3,>=4
struct RdmaProxyDepthHist {
    static constexpr int kBins = 5;
    std::uint64_t count[kBins]{};

    void record(std::uint64_t depth) noexcept;
};

struct RdmaProxyTimelineCounters {
    std::atomic<std::uint64_t> total_ns[static_cast<int>(RdmaProxyTimelineStage::kCount)]{};
    std::atomic<std::uint64_t> count[static_cast<int>(RdmaProxyTimelineStage::kCount)]{};
    RdmaProxyTimelineEventStats events[static_cast<int>(RdmaProxyTimelineEvent::kCount)]{};
    RdmaProxyDepthHist inflight_at_post{};
    RdmaProxyDepthHist free_slots_at_post{};
    RdmaProxyDepthHist inflight_at_cqe{};

    void add(RdmaProxyTimelineStage stage, std::uint64_t ns) noexcept;
    void record_event(RdmaProxyTimelineEvent event, std::uint64_t ns) noexcept;
    std::string format(const char* role_label, int source_rank, int destination_rank,
                       int channel, const char* plane) const;
    std::string format_v2(const char* role_label, int source_rank, int destination_rank,
                          int channel, const char* plane) const;
};

class RdmaProxyTimelineScope {
public:
    RdmaProxyTimelineScope(RdmaProxyTimelineCounters* counters, bool enabled,
                           RdmaProxyTimelineStage stage) noexcept;
    ~RdmaProxyTimelineScope();

    RdmaProxyTimelineScope(const RdmaProxyTimelineScope&) = delete;
    RdmaProxyTimelineScope& operator=(const RdmaProxyTimelineScope&) = delete;

private:
    RdmaProxyTimelineCounters* counters_;
    bool enabled_;
    RdmaProxyTimelineStage stage_;
    std::chrono::steady_clock::time_point start_{};
};

bool rdma_proxy_timeline_env_enabled();

#else

// OFF path: empty stub — no atomics, no scope residual cost in proxies.
struct RdmaProxyTimelineCounters {};

#endif

}  // namespace nano_nccl::transport::rdma
