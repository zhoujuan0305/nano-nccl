#include "transport/rdma/rdma_proxy_timeline.h"

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)

#include <cstdlib>
#include <sstream>

namespace nano_nccl::transport::rdma {

namespace {

constexpr const char* kStageNames[static_cast<int>(RdmaProxyTimelineStage::kCount)] = {
    "send_tail_wait",
    "send_post",
    "send_cq_wait",
    "recv_cq_wait",
    "recv_publish",
    "cts_post",
    "cts_cq",
};

constexpr const char* kEventNames[static_cast<int>(RdmaProxyTimelineEvent::kCount)] = {
    "send_ready_to_post",
    "send_inter_post",
    "send_post_to_cq",
    "recv_cq_to_publish",
    "recv_inter_publish",
    "cts_ready_to_post",
    "cts_inter_post",
    "send_tail_arm_to_post",
    "send_cts_arm_to_post",
    "send_tail_to_cts_ready",
    "send_post_to_next_tail",
};

// Upper-exclusive edges for bins 0..10; values >= last edge land in bin 11.
constexpr std::uint64_t kHistEdges[RdmaProxyTimelineEventStats::kHistBins - 1] = {
    250, 500, 1000, 2000, 5000, 8000, 10000, 12000, 15000, 20000, 30000,
};

int hist_bin(std::uint64_t ns) noexcept {
    for (int index = 0; index < RdmaProxyTimelineEventStats::kHistBins - 1; ++index) {
        if (ns < kHistEdges[index]) {
            return index;
        }
    }
    return RdmaProxyTimelineEventStats::kHistBins - 1;
}

}  // namespace

const char* rdma_proxy_timeline_stage_name(RdmaProxyTimelineStage stage) {
    const int index = static_cast<int>(stage);
    if (index < 0 || index >= static_cast<int>(RdmaProxyTimelineStage::kCount)) {
        return "unknown";
    }
    return kStageNames[index];
}

const char* rdma_proxy_timeline_event_name(RdmaProxyTimelineEvent event) {
    const int index = static_cast<int>(event);
    if (index < 0 || index >= static_cast<int>(RdmaProxyTimelineEvent::kCount)) {
        return "unknown";
    }
    return kEventNames[index];
}

void RdmaProxyTimelineEventStats::record(std::uint64_t ns) noexcept {
    ++count;
    sum_ns += ns;
    if (ns < min_ns) {
        min_ns = ns;
    }
    if (ns > max_ns) {
        max_ns = ns;
    }
    ++hist[hist_bin(ns)];
}

void RdmaProxyDepthHist::record(std::uint64_t depth) noexcept {
    const int bin = depth >= static_cast<std::uint64_t>(kBins - 1)
                        ? (kBins - 1)
                        : static_cast<int>(depth);
    ++count[bin];
}

void RdmaProxyTimelineCounters::add(RdmaProxyTimelineStage stage, std::uint64_t ns) noexcept {
    const int index = static_cast<int>(stage);
    if (index < 0 || index >= static_cast<int>(RdmaProxyTimelineStage::kCount)) {
        return;
    }
    total_ns[index].fetch_add(ns, std::memory_order_relaxed);
    count[index].fetch_add(1, std::memory_order_relaxed);
}

void RdmaProxyTimelineCounters::record_event(RdmaProxyTimelineEvent event,
                                             std::uint64_t ns) noexcept {
    const int index = static_cast<int>(event);
    if (index < 0 || index >= static_cast<int>(RdmaProxyTimelineEvent::kCount)) {
        return;
    }
    events[index].record(ns);
}

std::string RdmaProxyTimelineCounters::format(const char* role_label, int source_rank,
                                              int destination_rank, int channel,
                                              const char* plane) const {
    std::ostringstream out;
    out << "# rdma_proxy_timeline plane=" << (plane != nullptr ? plane : "") << '\n';
    out << "# src=" << source_rank << " dst=" << destination_rank << " ch=" << channel
        << " role=" << (role_label != nullptr ? role_label : "") << '\n';
    out << "# stage total_ns count avg_ns\n";
    for (int index = 0; index < static_cast<int>(RdmaProxyTimelineStage::kCount); ++index) {
        const std::uint64_t total = total_ns[index].load(std::memory_order_relaxed);
        const std::uint64_t n = count[index].load(std::memory_order_relaxed);
        const std::uint64_t avg = n == 0 ? 0 : total / n;
        out << kStageNames[index] << ' ' << total << ' ' << n << ' ' << avg << '\n';
    }
    return out.str();
}

std::string RdmaProxyTimelineCounters::format_v2(const char* role_label, int source_rank,
                                                 int destination_rank, int channel,
                                                 const char* plane) const {
    std::ostringstream out;
    out << "# rdma_proxy_timeline_v2 plane=" << (plane != nullptr ? plane : "")
        << " role=" << (role_label != nullptr ? role_label : "") << " src=" << source_rank
        << " dst=" << destination_rank << " ch=" << channel << '\n';
    out << "# metric count sum_ns avg_ns min_ns max_ns\n";
    for (int index = 0; index < static_cast<int>(RdmaProxyTimelineEvent::kCount); ++index) {
        const RdmaProxyTimelineEventStats& stats = events[index];
        const std::uint64_t avg = stats.count == 0 ? 0 : stats.sum_ns / stats.count;
        const std::uint64_t min_out =
            stats.count == 0 ? 0 : stats.min_ns;
        out << kEventNames[index] << ' ' << stats.count << ' ' << stats.sum_ns << ' ' << avg
            << ' ' << min_out << ' ' << stats.max_ns << '\n';
    }
    out << "# hist metric bin0 bin1 bin2 bin3 bin4 bin5 bin6 bin7 bin8 bin9 bin10 "
           "bin11\n";
    for (int index = 0; index < static_cast<int>(RdmaProxyTimelineEvent::kCount); ++index) {
        const RdmaProxyTimelineEventStats& stats = events[index];
        out << "hist " << kEventNames[index];
        for (int bin = 0; bin < RdmaProxyTimelineEventStats::kHistBins; ++bin) {
            out << ' ' << stats.hist[bin];
        }
        out << '\n';
    }
    out << "# depth_hist name c0 c1 c2 c3 c4\n";
    const auto dump_depth = [&out](const char* name, const RdmaProxyDepthHist& hist) {
        out << "depth_hist " << name;
        for (int bin = 0; bin < RdmaProxyDepthHist::kBins; ++bin) {
            out << ' ' << hist.count[bin];
        }
        out << '\n';
    };
    dump_depth("inflight_at_post", inflight_at_post);
    dump_depth("free_slots_at_post", free_slots_at_post);
    dump_depth("inflight_at_cqe", inflight_at_cqe);
    return out.str();
}

RdmaProxyTimelineScope::RdmaProxyTimelineScope(RdmaProxyTimelineCounters* counters, bool enabled,
                                               RdmaProxyTimelineStage stage) noexcept
    : counters_(counters),
      enabled_(enabled && counters != nullptr),
      stage_(stage) {
    if (enabled_) {
        start_ = std::chrono::steady_clock::now();
    }
}

RdmaProxyTimelineScope::~RdmaProxyTimelineScope() {
    if (!enabled_) {
        return;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
    if (elapsed > 0) {
        counters_->add(stage_, static_cast<std::uint64_t>(elapsed));
    } else {
        counters_->add(stage_, 0);
    }
}

bool rdma_proxy_timeline_env_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("NANO_NCCL_RDMA_PROXY_TIMELINE");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

}  // namespace nano_nccl::transport::rdma

#endif
