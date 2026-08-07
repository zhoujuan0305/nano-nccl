#pragma once

#include <cstdint>
#include <sstream>
#include <string>

namespace nano_nccl::collective::all_reduce {

// Device/host POD: dual-wait RR phase segment totals (nanoseconds).
struct RingTurnaroundStats {
    unsigned long long wait_wall_ns;
    unsigned long long wait_recv_ns;
    unsigned long long wait_send_ns;
    unsigned long long compute_ns;
    unsigned long long post_ns;
    unsigned long long count;
};

inline std::string format_ring_turnaround(const RingTurnaroundStats& stats,
                                          int local_rank, int channel) {
    const auto avg = [](unsigned long long total, unsigned long long n) {
        return n == 0 ? 0ULL : total / n;
    };
    const unsigned long long n = stats.count;
    std::ostringstream out;
    out << "# ring_turnaround rank=" << local_rank << " ch=" << channel << '\n';
    out << "# segment total_ns count avg_ns\n";
    out << "wait_wall " << stats.wait_wall_ns << ' ' << n << ' '
        << avg(stats.wait_wall_ns, n) << '\n';
    out << "wait_recv " << stats.wait_recv_ns << ' ' << n << ' '
        << avg(stats.wait_recv_ns, n) << '\n';
    out << "wait_send " << stats.wait_send_ns << ' ' << n << ' '
        << avg(stats.wait_send_ns, n) << '\n';
    out << "compute " << stats.compute_ns << ' ' << n << ' '
        << avg(stats.compute_ns, n) << '\n';
    out << "post " << stats.post_ns << ' ' << n << ' ' << avg(stats.post_ns, n)
        << '\n';
    return out.str();
}

}  // namespace nano_nccl::collective::all_reduce
