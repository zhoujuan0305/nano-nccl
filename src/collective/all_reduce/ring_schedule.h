#pragma once

#include "nano_nccl/types.h"

namespace nano_nccl::collective::all_reduce {

enum class RingDirection { Forward, Reverse };

inline RingDirection ring_direction(ChannelPolicy policy, int channel) {
    return policy == ChannelPolicy::CounterRotating && channel % 2 != 0
        ? RingDirection::Reverse
        : RingDirection::Forward;
}

inline int ring_next(int rank, RingDirection direction, int nranks) {
    return direction == RingDirection::Forward
        ? (rank + 1) % nranks
        : (rank + nranks - 1) % nranks;
}

inline int ring_previous(int rank, RingDirection direction, int nranks) {
    return direction == RingDirection::Forward
        ? (rank + nranks - 1) % nranks
        : (rank + 1) % nranks;
}

// A canonical edge names the undirected adjacent pair by its forward source.
// This keeps topology/transport selection shared by opposite channel directions.
inline int ring_edge_index(int source, int destination, int nranks) {
    if (nranks <= 1 || source < 0 || source >= nranks || destination < 0 ||
        destination >= nranks) {
        return -1;
    }
    if (destination == (source + 1) % nranks) return source;
    if (destination == (source + nranks - 1) % nranks) return destination;
    return -1;
}

inline int ring_source_for_edge(int edge, RingDirection direction, int nranks) {
    return direction == RingDirection::Forward ? edge : (edge + 1) % nranks;
}

inline int ring_destination_for_edge(int edge, RingDirection direction,
                                     int nranks) {
    return direction == RingDirection::Forward ? (edge + 1) % nranks : edge;
}

inline int logical_ring_index(int rank, RingDirection direction, int nranks) {
    return direction == RingDirection::Forward ? rank : (nranks - rank) % nranks;
}

}  // namespace nano_nccl::collective::all_reduce
