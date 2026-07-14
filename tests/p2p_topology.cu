#include "transport/p2p/p2p_topology.h"

#include <cstdio>

int main() {
    using nano_nccl::TransportKind;
    using nano_nccl::transport::p2p::RingTransportPlan;

    RingTransportPlan mixed({TransportKind::P2p, TransportKind::Shm,
                             TransportKind::P2p, TransportKind::Shm});
    if (mixed.edge_kind(0) != TransportKind::P2p ||
        mixed.edge_kind(1) != TransportKind::Shm ||
        mixed.resolved_kind() != TransportKind::Mixed) {
        return 1;
    }
    if (RingTransportPlan::uniform(TransportKind::P2p).resolved_kind() !=
            TransportKind::P2p ||
        RingTransportPlan::uniform(TransportKind::Shm).resolved_kind() !=
            TransportKind::Shm) {
        return 1;
    }

    std::puts("p2p_topology=PASS");
    return 0;
}
