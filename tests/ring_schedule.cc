#include "collective/all_reduce/ring_schedule.h"

#include <cstdio>

namespace {

bool expect(bool condition, const char* message) {
    if (condition) return true;
    std::fprintf(stderr, "ring schedule check failed: %s\n", message);
    return false;
}

}  // namespace

int main() {
    using nano_nccl::ChannelPolicy;
    using namespace nano_nccl::collective::all_reduce;
    constexpr int kTestRanks = 4;

    bool ok = true;
    ok = expect(ring_direction(ChannelPolicy::Forward, 1) ==
                    RingDirection::Forward,
                "forward policy changed odd channel") && ok;
    ok = expect(ring_direction(ChannelPolicy::CounterRotating, 0) ==
                    RingDirection::Forward,
                "counter-rotating even channel is not forward") && ok;
    ok = expect(ring_direction(ChannelPolicy::CounterRotating, 1) ==
                    RingDirection::Reverse,
                "counter-rotating odd channel is not reverse") && ok;

    for (int rank = 0; rank < kTestRanks; ++rank) {
        int forward_next = (rank + 1) % kTestRanks;
        int reverse_next = (rank + kTestRanks - 1) % kTestRanks;
        ok = expect(ring_next(rank, RingDirection::Forward, kTestRanks) ==
                        forward_next,
                    "forward successor mismatch") && ok;
        ok = expect(ring_next(rank, RingDirection::Reverse, kTestRanks) ==
                        reverse_next,
                    "reverse successor mismatch") && ok;
        ok = expect(ring_edge_index(rank, forward_next, kTestRanks) == rank,
                    "forward canonical edge mismatch") && ok;
        ok = expect(ring_edge_index(rank, reverse_next, kTestRanks) ==
                        reverse_next,
                    "reverse canonical edge mismatch") && ok;
        ok = expect(logical_ring_index(rank, RingDirection::Reverse, kTestRanks) ==
                        (kTestRanks - rank) % kTestRanks,
                    "reverse logical ring index mismatch") && ok;
    }

    for (int edge = 0; edge < kTestRanks; ++edge) {
        int reverse_source = (edge + 1) % kTestRanks;
        ok = expect(ring_source_for_edge(edge, RingDirection::Reverse,
                                         kTestRanks) == reverse_source,
                    "reverse source mismatch") && ok;
        ok = expect(ring_destination_for_edge(edge, RingDirection::Reverse,
                                              kTestRanks) == edge,
                    "reverse destination mismatch") && ok;
        ok = expect(ring_edge_index(reverse_source, edge, kTestRanks) == edge,
                    "reverse edge does not round trip") && ok;
    }

    ok = expect(ring_edge_index(0, 2, kTestRanks) == -1,
                "non-adjacent edge was accepted") && ok;
    if (!ok) return 1;
    std::puts("ring_schedule=PASS");
    return 0;
}
