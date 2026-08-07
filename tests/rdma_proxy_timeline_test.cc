#include "transport/rdma/rdma_proxy_timeline.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using nano_nccl::transport::rdma::RdmaProxyDepthHist;
using nano_nccl::transport::rdma::RdmaProxyTimelineCounters;
using nano_nccl::transport::rdma::RdmaProxyTimelineEvent;
using nano_nccl::transport::rdma::RdmaProxyTimelineEventStats;
using nano_nccl::transport::rdma::RdmaProxyTimelineScope;
using nano_nccl::transport::rdma::RdmaProxyTimelineStage;
using nano_nccl::transport::rdma::rdma_proxy_timeline_event_name;
using nano_nccl::transport::rdma::rdma_proxy_timeline_stage_name;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void counters_accumulate_totals_counts_and_avg() {
    RdmaProxyTimelineCounters counters;
    counters.add(RdmaProxyTimelineStage::SendTailWait, 1000);
    counters.add(RdmaProxyTimelineStage::SendTailWait, 3000);
    counters.add(RdmaProxyTimelineStage::SendPost, 500);

    const int tail = static_cast<int>(RdmaProxyTimelineStage::SendTailWait);
    const int post = static_cast<int>(RdmaProxyTimelineStage::SendPost);

    require(counters.total_ns[tail].load() == 4000, "SendTailWait total_ns");
    require(counters.count[tail].load() == 2, "SendTailWait count");
    require(counters.total_ns[post].load() == 500, "SendPost total_ns");
    require(counters.count[post].load() == 1, "SendPost count");

    const std::string dump =
        counters.format("send", 0, 1, 2, "WriteCts");
    require(dump.find("# rdma_proxy_timeline plane=WriteCts") != std::string::npos,
            "dump plane header");
    require(dump.find("# src=0 dst=1 ch=2 role=send") != std::string::npos,
            "dump identity header");
    require(dump.find("send_tail_wait 4000 2 2000") != std::string::npos,
            "dump SendTailWait avg");
    require(dump.find("send_post 500 1 500") != std::string::npos, "dump SendPost avg");
}

void scope_records_elapsed_time() {
    RdmaProxyTimelineCounters counters;
    {
        RdmaProxyTimelineScope scope(&counters, true, RdmaProxyTimelineStage::RecvCqWait);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const int stage = static_cast<int>(RdmaProxyTimelineStage::RecvCqWait);
    require(counters.count[stage].load() == 1, "scope count");
    require(counters.total_ns[stage].load() >= 1'000'000, "scope total_ns lower bound");
}

void stage_names_match_canonical_ids() {
    require(std::string(rdma_proxy_timeline_stage_name(RdmaProxyTimelineStage::SendTailWait)) ==
                "send_tail_wait",
            "send_tail_wait name");
    require(std::string(rdma_proxy_timeline_stage_name(RdmaProxyTimelineStage::SendPost)) ==
                "send_post",
            "send_post name");
    require(std::string(rdma_proxy_timeline_stage_name(RdmaProxyTimelineStage::SendCqWait)) ==
                "send_cq_wait",
            "send_cq_wait name");
    require(std::string(rdma_proxy_timeline_stage_name(RdmaProxyTimelineStage::RecvCqWait)) ==
                "recv_cq_wait",
            "recv_cq_wait name");
    require(std::string(rdma_proxy_timeline_stage_name(RdmaProxyTimelineStage::RecvPublish)) ==
                "recv_publish",
            "recv_publish name");
    require(std::string(rdma_proxy_timeline_stage_name(RdmaProxyTimelineStage::CtsPost)) ==
                "cts_post",
            "cts_post name");
    require(std::string(rdma_proxy_timeline_stage_name(RdmaProxyTimelineStage::CtsCq)) == "cts_cq",
            "cts_cq name");
}

void disabled_scope_is_noop() {
    RdmaProxyTimelineCounters counters;
    {
        RdmaProxyTimelineScope scope(&counters, false, RdmaProxyTimelineStage::CtsPost);
    }
    const int stage = static_cast<int>(RdmaProxyTimelineStage::CtsPost);
    require(counters.count[stage].load() == 0, "disabled scope count");
    require(counters.total_ns[stage].load() == 0, "disabled scope total");
}

void event_stats_record_updates_count_sum_min_max_and_hist() {
    RdmaProxyTimelineEventStats stats;
    require(stats.count == 0, "initial count");
    require(stats.min_ns == std::numeric_limits<std::uint64_t>::max(), "initial min");
    require(stats.max_ns == 0, "initial max");
    require(RdmaProxyTimelineEventStats::kHistBins == 12, "12 hist bins");

    // edges: 250,500,1000,2000,5000,8000,10000,12000,15000,20000,30000,+inf
    // bins: [0,250), [250,500), [500,1000), [1000,2000), [2000,5000),
    //        [5000,8000), [8000,10000), [10000,12000), [12000,15000),
    //        [15000,20000), [20000,30000), [30000,inf)
    stats.record(100);
    stats.record(250);
    stats.record(999);
    stats.record(9000);
    stats.record(30000);
    stats.record(30001);

    require(stats.count == 6, "event count");
    require(stats.sum_ns == 100 + 250 + 999 + 9000 + 30000 + 30001, "event sum");
    require(stats.min_ns == 100, "event min");
    require(stats.max_ns == 30001, "event max");
    require(stats.hist[0] == 1, "hist bin0 <250");
    require(stats.hist[1] == 1, "hist bin1 250-500");
    require(stats.hist[2] == 1, "hist bin2 500-1000");
    require(stats.hist[3] == 0, "hist bin3 empty");
    require(stats.hist[6] == 1, "hist bin6 8k-10k");
    require(stats.hist[10] == 0, "hist bin10 20k-30k empty");
    require(stats.hist[11] == 2, "hist bin11 >=30000");
}

void event_names_match_canonical_ids() {
    require(std::string(rdma_proxy_timeline_event_name(
                RdmaProxyTimelineEvent::SendReadyToPost)) == "send_ready_to_post",
            "send_ready_to_post name");
    require(std::string(rdma_proxy_timeline_event_name(
                RdmaProxyTimelineEvent::SendInterPost)) == "send_inter_post",
            "send_inter_post name");
    require(std::string(rdma_proxy_timeline_event_name(
                RdmaProxyTimelineEvent::SendPostToCq)) == "send_post_to_cq",
            "send_post_to_cq name");
    require(std::string(rdma_proxy_timeline_event_name(
                RdmaProxyTimelineEvent::RecvCqToPublish)) == "recv_cq_to_publish",
            "recv_cq_to_publish name");
    require(std::string(rdma_proxy_timeline_event_name(
                RdmaProxyTimelineEvent::RecvInterPublish)) == "recv_inter_publish",
            "recv_inter_publish name");
    require(std::string(rdma_proxy_timeline_event_name(
                RdmaProxyTimelineEvent::CtsReadyToPost)) == "cts_ready_to_post",
            "cts_ready_to_post name");
    require(std::string(rdma_proxy_timeline_event_name(
                RdmaProxyTimelineEvent::CtsInterPost)) == "cts_inter_post",
            "cts_inter_post name");
    require(std::string(rdma_proxy_timeline_event_name(
                RdmaProxyTimelineEvent::SendTailArmToPost)) == "send_tail_arm_to_post",
            "send_tail_arm_to_post name");
    require(std::string(rdma_proxy_timeline_event_name(
                RdmaProxyTimelineEvent::SendCtsArmToPost)) == "send_cts_arm_to_post",
            "send_cts_arm_to_post name");
    require(std::string(rdma_proxy_timeline_event_name(
                RdmaProxyTimelineEvent::SendTailToCtsReady)) == "send_tail_to_cts_ready",
            "send_tail_to_cts_ready name");
    require(std::string(rdma_proxy_timeline_event_name(
                RdmaProxyTimelineEvent::SendPostToNextTail)) == "send_post_to_next_tail",
            "send_post_to_next_tail name");
}

void format_v2_contains_metric_and_hist_lines() {
    RdmaProxyTimelineCounters counters;
    counters.record_event(RdmaProxyTimelineEvent::SendReadyToPost, 100);
    counters.record_event(RdmaProxyTimelineEvent::SendReadyToPost, 300);
    counters.record_event(RdmaProxyTimelineEvent::SendInterPost, 1000);
    counters.record_event(RdmaProxyTimelineEvent::RecvCqToPublish, 50);
    counters.record_event(RdmaProxyTimelineEvent::SendTailArmToPost, 12000);
    counters.record_event(RdmaProxyTimelineEvent::SendCtsArmToPost, 8000);
    counters.record_event(RdmaProxyTimelineEvent::SendTailToCtsReady, 0);
    counters.record_event(RdmaProxyTimelineEvent::SendPostToNextTail, 15000);

    const std::string dump =
        counters.format_v2("send", 0, 1, 2, "WriteCts");
    require(dump.find("# rdma_proxy_timeline_v2 plane=WriteCts role=send src=0 dst=1 ch=2") !=
                std::string::npos,
            "v2 header");
    require(dump.find("# metric count sum_ns avg_ns min_ns max_ns") != std::string::npos,
            "v2 metric header");
    require(dump.find("send_ready_to_post 2 400 200 100 300") != std::string::npos,
            "v2 send_ready_to_post line");
    require(dump.find("send_inter_post 1 1000 1000 1000 1000") != std::string::npos,
            "v2 send_inter_post line");
    require(dump.find("recv_cq_to_publish 1 50 50 50 50") != std::string::npos,
            "v2 recv_cq_to_publish line");
    require(dump.find("send_tail_arm_to_post 1 12000 12000 12000 12000") != std::string::npos,
            "v2 send_tail_arm_to_post line");
    require(dump.find("send_cts_arm_to_post 1 8000 8000 8000 8000") != std::string::npos,
            "v2 send_cts_arm_to_post line");
    require(dump.find("send_tail_to_cts_ready 1 0 0 0 0") != std::string::npos,
            "v2 send_tail_to_cts_ready line");
    require(dump.find("send_post_to_next_tail 1 15000 15000 15000 15000") != std::string::npos,
            "v2 send_post_to_next_tail line");
    require(dump.find("# hist metric bin0 bin1 bin2 bin3 bin4 bin5 bin6 bin7 bin8 "
                      "bin9 bin10 bin11") != std::string::npos,
            "v2 hist header 12 bins");
    require(dump.find("hist send_ready_to_post ") != std::string::npos,
            "v2 hist send_ready_to_post");
    // 100 → bin0, 300 → bin1; twelve bins
    require(dump.find("hist send_ready_to_post 1 1 0 0 0 0 0 0 0 0 0 0") != std::string::npos,
            "v2 hist bin counts");
}

void depth_hist_bins_0_1_2_3_ge4() {
    require(RdmaProxyDepthHist::kBins == 5, "depth hist 5 bins");
    RdmaProxyDepthHist hist{};
    hist.record(0);
    hist.record(1);
    hist.record(2);
    hist.record(3);
    hist.record(4);
    hist.record(7);
    hist.record(1);
    require(hist.count[0] == 1, "depth bin0");
    require(hist.count[1] == 2, "depth bin1");
    require(hist.count[2] == 1, "depth bin2");
    require(hist.count[3] == 1, "depth bin3");
    require(hist.count[4] == 2, "depth bin4+");
}

void free_slots_at_post_unsigned_formula() {
    // free = (head + slot_count - tail) / step_inc with unsigned arithmetic
    const std::uint64_t head = 10;
    const std::uint64_t tail = 6;
    const std::uint64_t slot_count = 8;
    const std::uint64_t step_inc = 2;
    const std::uint64_t free_slots =
        (head + slot_count - tail) / step_inc;
    require(free_slots == 6, "free slots formula");
    // head just behind tail by almost a full ring (credit tight)
    const std::uint64_t tight =
        (std::uint64_t{4} + slot_count - std::uint64_t{10}) / step_inc;
    require(tight == 1, "tight free slots");
}

void format_v2_contains_depth_hist_lines() {
    RdmaProxyTimelineCounters counters;
    counters.inflight_at_post.record(0);
    counters.inflight_at_post.record(3);
    counters.free_slots_at_post.record(2);
    counters.free_slots_at_post.record(5);
    counters.inflight_at_cqe.record(1);
    counters.inflight_at_cqe.record(4);

    const std::string dump = counters.format_v2("send", 0, 1, 0, "SendRecv");
    require(dump.find("# depth_hist name c0 c1 c2 c3 c4") != std::string::npos,
            "depth_hist header");
    require(dump.find("depth_hist inflight_at_post 1 0 0 1 0") != std::string::npos,
            "inflight_at_post hist");
    require(dump.find("depth_hist free_slots_at_post 0 0 1 0 1") != std::string::npos,
            "free_slots_at_post hist");
    require(dump.find("depth_hist inflight_at_cqe 0 1 0 0 1") != std::string::npos,
            "inflight_at_cqe hist");
}

}  // namespace

int main() {
    try {
#if !defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
#error "rdma_proxy_timeline_test requires NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE"
#endif
        counters_accumulate_totals_counts_and_avg();
        scope_records_elapsed_time();
        stage_names_match_canonical_ids();
        disabled_scope_is_noop();
        event_stats_record_updates_count_sum_min_max_and_hist();
        event_names_match_canonical_ids();
        format_v2_contains_metric_and_hist_lines();
        depth_hist_bins_0_1_2_3_ge4();
        free_slots_at_post_unsigned_formula();
        format_v2_contains_depth_hist_lines();
        std::cout << "rdma_proxy_timeline_test: PASS\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "rdma_proxy_timeline_test: FAIL: " << ex.what() << "\n";
        return 1;
    }
}
