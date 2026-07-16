#pragma once

#include "nano_nccl/types.h"

#include <vector>

namespace nano_nccl::collective::all_reduce {
struct ProcessTopology;
}

namespace nano_nccl::transport::p2p {

class RingTransportPlan {
public:
    explicit RingTransportPlan(
        std::vector<TransportKind> edge_kinds);

    static RingTransportPlan uniform(TransportKind kind, int rank_count = kRanks);

    TransportKind edge_kind(int edge) const;
    const std::vector<TransportKind>& edge_kinds() const noexcept;
    TransportKind resolved_kind() const;
    bool uses_p2p() const;

private:
    std::vector<TransportKind> edge_kinds_;
};

RingTransportPlan resolve_ring_transport(TransportKind requested);
RingTransportPlan resolve_ring_transport(
    TransportKind requested,
    const collective::all_reduce::ProcessTopology& topology);
void enable_p2p_ring_peer_access_or_throw(const RingTransportPlan& plan);
void enable_p2p_ring_peer_access_or_throw(
    const RingTransportPlan& plan,
    const collective::all_reduce::ProcessTopology& topology);

}  // namespace nano_nccl::transport::p2p
