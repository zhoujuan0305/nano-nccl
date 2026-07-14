#pragma once

#include "core/buffer.h"
#include "nano_nccl/types.h"
#include "transport/p2p/p2p_topology.h"

#include <cstddef>

namespace nano_nccl::transport::p2p {

template <typename T>
class P2pFifo {
public:
    P2pFifo(std::size_t slot_elems, int nranks);
    P2pFifo(std::size_t slot_elems, const RingTransportPlan& plan);
    ~P2pFifo();

    P2pFifo(const P2pFifo&) = delete;
    P2pFifo& operator=(const P2pFifo&) = delete;

    T* edge_ptr(int channel, int edge) const;

private:
    core::DeviceBuffer<T>* buffers_[kChannels][kRanks]{};
};

bool p2p_ring_available();
void require_p2p_ring();
void enable_p2p_ring_peer_access_or_throw();

}  // namespace nano_nccl::transport::p2p
