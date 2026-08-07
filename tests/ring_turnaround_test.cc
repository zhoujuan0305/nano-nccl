#include "collective/all_reduce/ring_turnaround.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

using nano_nccl::collective::all_reduce::format_ring_turnaround;
using nano_nccl::collective::all_reduce::RingTurnaroundStats;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void format_emits_header_and_segment_avgs() {
    RingTurnaroundStats stats{};
    stats.wait_wall_ns = 4000;
    stats.wait_recv_ns = 3000;
    stats.wait_send_ns = 1000;
    stats.compute_ns = 500;
    stats.post_ns = 200;
    stats.count = 2;

    const std::string dump = format_ring_turnaround(stats, 1, 3);
    require(dump.find("# ring_turnaround rank=1 ch=3") != std::string::npos,
            "header");
    require(dump.find("# segment total_ns count avg_ns") != std::string::npos,
            "column header");
    require(dump.find("wait_wall 4000 2 2000") != std::string::npos, "wait_wall");
    require(dump.find("wait_recv 3000 2 1500") != std::string::npos, "wait_recv");
    require(dump.find("wait_send 1000 2 500") != std::string::npos, "wait_send");
    require(dump.find("compute 500 2 250") != std::string::npos, "compute");
    require(dump.find("post 200 2 100") != std::string::npos, "post");
}

void format_zero_count_avgs_are_zero() {
    RingTurnaroundStats stats{};
    stats.wait_wall_ns = 10;
    const std::string dump = format_ring_turnaround(stats, 0, 0);
    require(dump.find("wait_wall 10 0 0") != std::string::npos, "zero count avg");
}

}  // namespace

int main() {
    try {
        format_emits_header_and_segment_avgs();
        format_zero_count_avgs_are_zero();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "ring_turnaround_test FAIL: %s\n", ex.what());
        return 1;
    }
    std::printf("ring_turnaround=PASS\n");
    return 0;
}
