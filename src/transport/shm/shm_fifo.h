#pragma once

#include "transport/simple/protocol.h"

#include <cstdint>

namespace nano_nccl::transport::shm {

__host__ __device__ inline int step_idx(int kind, int channel, int edge) {
    return (kind * kChannels + channel) * kRanks + edge;
}

inline transport::simple::ControlArgs make_simple_control_args(
    std::uint64_t* steps, std::uint64_t* base_steps, int rank) {
    transport::simple::ControlArgs control{};
    const int send_edge = rank;
    const int recv_edge = (rank + kRanks - 1) % kRanks;
    for (int channel = 0; channel < kChannels; ++channel) {
        control.send_head[channel] =
            steps + step_idx(0, channel, send_edge);
        control.recv_tail[channel] =
            steps + step_idx(1, channel, recv_edge);
        control.send_tail[channel] =
            steps + step_idx(1, channel, send_edge);
        control.recv_head[channel] =
            steps + step_idx(0, channel, recv_edge);
    }
    control.send_base_steps = base_steps + rank * kChannels;
    control.recv_base_steps =
        base_steps + (kRanks + rank) * kChannels;
    return control;
}

}  // namespace nano_nccl::transport::shm
