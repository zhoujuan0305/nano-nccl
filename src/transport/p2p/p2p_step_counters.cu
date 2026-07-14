#include "transport/p2p/p2p_step_counters.h"

#include "transport/p2p/p2p_fifo.h"

#include <stdexcept>
#include <string>

namespace nano_nccl::transport::p2p {

P2pStepCounters::P2pStepCounters(int nranks)
    : plan_(RingTransportPlan::uniform(TransportKind::P2p)) {
    if (nranks != kRanks) {
        throw std::runtime_error("P2pStepCounters requested nranks=" +
                                 std::to_string(nranks) +
                                 " does not match configured kRanks=" +
                                 std::to_string(kRanks));
    }
    require_p2p_ring();

    try {
        for (int rank = 0; rank < kRanks; ++rank) {
            counters_[rank] = new core::DeviceBuffer<std::uint64_t>(
                rank, 3 * kChannels);
        }
    } catch (...) {
        cleanup();
        throw;
    }
}

P2pStepCounters::P2pStepCounters(const RingTransportPlan& plan) : plan_(plan) {
    try {
        for (int rank = 0; rank < kRanks; ++rank) {
            int prev = (rank + kRanks - 1) % kRanks;
            if (plan_.edge_kind(rank) != TransportKind::P2p &&
                plan_.edge_kind(prev) != TransportKind::P2p) {
                continue;
            }
            counters_[rank] = new core::DeviceBuffer<std::uint64_t>(
                rank, 3 * kChannels);
        }
    } catch (...) {
        cleanup();
        throw;
    }
}

P2pStepCounters::~P2pStepCounters() { cleanup(); }

void P2pStepCounters::reset(cudaStream_t streams[kRanks]) {
    for (int rank = 0; rank < kRanks; ++rank) {
        if (counters_[rank] == nullptr) {
            continue;
        }
        CUDA_CHECK_THROW(cudaSetDevice(rank));
        CUDA_CHECK_THROW(cudaMemsetAsync(
            counters_[rank]->get(), 0,
            3 * kChannels * sizeof(std::uint64_t), streams[rank]));
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
