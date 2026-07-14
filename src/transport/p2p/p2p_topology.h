#pragma once

#include "nano_nccl/types.h"

#include <array>

namespace nano_nccl::transport::p2p {

class RingTransportPlan {
public:
    explicit RingTransportPlan(
        std::array<TransportKind, kRanks> edge_kinds);

    static RingTransportPlan uniform(TransportKind kind);

    TransportKind edge_kind(int edge) const;
    TransportKind resolved_kind() const;
    bool uses_p2p() const;

private:
    std::array<TransportKind, kRanks> edge_kinds_;
};

RingTransportPlan resolve_ring_transport(TransportKind requested);
void enable_p2p_ring_peer_access_or_throw(const RingTransportPlan& plan);

}  // namespace nano_nccl::transport::p2p
