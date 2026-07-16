#pragma once

#include "collective/all_reduce/topology.h"
#include "core/buffer.h"
#include "transport/p2p/p2p_topology.h"
#include "transport/simple_protocol.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace nano_nccl::transport::p2p {

class P2pStepCounters {
public:
    explicit P2pStepCounters(int nranks);
    explicit P2pStepCounters(const RingTransportPlan& plan);
    P2pStepCounters(const RingTransportPlan& plan,
                    const collective::all_reduce::ProcessTopology& topology);
    ~P2pStepCounters();

    P2pStepCounters(const P2pStepCounters&) = delete;
    P2pStepCounters& operator=(const P2pStepCounters&) = delete;

    void reset(cudaStream_t streams[kRanks]);
    void reset(const std::vector<cudaStream_t>& streams);
    SimpleControlArgs control_args(int rank) const;

private:
    static constexpr std::size_t kHeadOffset = 0;
    static constexpr std::size_t kTailOffset = kChannels;
    static constexpr std::size_t kBaseStepOffset = 2 * kChannels;

    void cleanup();

    RingTransportPlan plan_;
    collective::all_reduce::ProcessTopology topology_;
    core::DeviceBuffer<std::uint64_t>* counters_[kRanks]{};
};

}  // namespace nano_nccl::transport::p2p
