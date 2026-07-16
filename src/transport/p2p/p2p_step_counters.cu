#include "transport/p2p/p2p_step_counters.h"

#include "transport/p2p/p2p_fifo.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace nano_nccl::transport::p2p {

namespace {

collective::all_reduce::ProcessTopology make_single_process_topology(
    const RingTransportPlan& plan) {
    collective::all_reduce::ProcessTopology topology{
        kRanks,
        0,
        std::vector<int>(kRanks),
        plan.edge_kinds(),
        false,
    };
    for (int rank = 0; rank < kRanks; ++rank) topology.devices[rank] = rank;
    return topology;
}

}  // namespace

P2pStepCounters::P2pStepCounters(int nranks)
    : plan_(RingTransportPlan::uniform(TransportKind::P2p)),
      topology_(make_single_process_topology(plan_)) {
    if (nranks != kRanks) {
        throw std::runtime_error("P2pStepCounters requested nranks=" +
                                 std::to_string(nranks) +
                                 " does not match configured kRanks=" +
                                 std::to_string(kRanks));
    }
    require_p2p_ring();

    try {
        for (int local_rank = 0; local_rank < kRanks; ++local_rank) {
            counters_[local_rank] = new core::DeviceBuffer<std::uint64_t>(
                topology_.devices[local_rank], 3 * kChannels);
        }
    } catch (...) {
        cleanup();
        throw;
    }
}

P2pStepCounters::P2pStepCounters(const RingTransportPlan& plan)
    : P2pStepCounters(plan, make_single_process_topology(plan)) {}

P2pStepCounters::P2pStepCounters(
    const RingTransportPlan& plan,
    const collective::all_reduce::ProcessTopology& topology)
    : plan_(plan), topology_(topology) {
    collective::all_reduce::validate_process_topology(topology_);
    try {
        for (int local_rank = 0; local_rank <
                                  static_cast<int>(topology_.devices.size()); ++local_rank) {
            int global_rank = topology_.local_rank_offset + local_rank;
            int prev = (global_rank + kRanks - 1) % kRanks;
            if (plan_.edge_kind(global_rank) != TransportKind::P2p &&
                plan_.edge_kind(prev) != TransportKind::P2p) {
                continue;
            }
            counters_[global_rank] = new core::DeviceBuffer<std::uint64_t>(
                topology_.devices[local_rank], 3 * kChannels);
        }
    } catch (...) {
        cleanup();
        throw;
    }
}

P2pStepCounters::~P2pStepCounters() { cleanup(); }

void P2pStepCounters::reset(cudaStream_t streams[kRanks]) {
    reset(std::vector<cudaStream_t>(streams, streams + kRanks));
}

void P2pStepCounters::reset(const std::vector<cudaStream_t>& streams) {
    if (streams.size() != topology_.devices.size()) {
        throw std::runtime_error("P2pStepCounters requires one stream per local rank");
    }
    for (int local_rank = 0; local_rank <
                              static_cast<int>(topology_.devices.size()); ++local_rank) {
        int global_rank = topology_.local_rank_offset + local_rank;
        if (counters_[global_rank] == nullptr) {
            continue;
        }
        CUDA_CHECK_THROW(cudaSetDevice(topology_.devices[local_rank]));
        CUDA_CHECK_THROW(cudaMemsetAsync(
            counters_[global_rank]->get(), 0,
            3 * kChannels * sizeof(std::uint64_t), streams[local_rank]));
    }
}

SimpleControlArgs P2pStepCounters::control_args(int rank) const {
    SimpleControlArgs control{};
    int prev = (rank + kRanks - 1) % kRanks;
    for (int channel = 0; channel < kChannels; ++channel) {
        if (plan_.edge_kind(rank) == TransportKind::P2p) {
            control.send_head[channel] =
                counters_[rank]->get() + kHeadOffset + channel;
            control.send_tail[channel] =
                counters_[(rank + 1) % kRanks]->get() + kTailOffset + channel;
        }
        if (plan_.edge_kind(prev) == TransportKind::P2p) {
            control.recv_tail[channel] =
                counters_[rank]->get() + kTailOffset + channel;
            control.recv_head[channel] =
                counters_[prev]->get() + kHeadOffset + channel;
        }
    }
    if (counters_[rank] != nullptr) {
        control.base_steps = counters_[rank]->get() + kBaseStepOffset;
    }
    return control;
}

void P2pStepCounters::cleanup() {
    for (int rank = 0; rank < kRanks; ++rank) {
        delete counters_[rank];
        counters_[rank] = nullptr;
    }
}

}  // namespace nano_nccl::transport::p2p
