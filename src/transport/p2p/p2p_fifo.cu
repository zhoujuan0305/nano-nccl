#include "transport/p2p/p2p_fifo.h"

#include "collective/all_reduce/topology.h"
#include "transport/simple_protocol.h"

#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

RingTransportPlan make_full_p2p_plan(int nranks) {
    if (nranks != kRanks) {
        throw std::runtime_error("P2pFifo requested nranks=" +
                                 std::to_string(nranks) +
                                 " does not match configured kRanks=" +
                                 std::to_string(kRanks));
    }
    require_p2p_ring();
    return RingTransportPlan::uniform(TransportKind::P2p);
}

bool peer_access_available(int src, int dst) {
    int can_access = 0;
    CUDA_CHECK_THROW(cudaDeviceCanAccessPeer(&can_access, src, dst));
    return can_access;
}

void require_peer_access(int src, int dst) {
    if (!peer_access_available(src, dst)) {
        throw std::runtime_error("P2P unavailable for ring edge " +
                                 std::to_string(src) + " -> " +
                                 std::to_string(dst));
    }
}

void enable_peer_access_or_throw(int src, int dst) {
    CUDA_CHECK_THROW(cudaSetDevice(src));
    cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    } else if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

}  // namespace

bool p2p_ring_available() {
    for (int edge = 0; edge < kRanks; ++edge) {
        int receiver = (edge + 1) % kRanks;
        if (!peer_access_available(edge, receiver) ||
            !peer_access_available(receiver, edge)) {
            return false;
        }
    }
    return true;
}

void require_p2p_ring() {
    for (int edge = 0; edge < kRanks; ++edge) {
        int receiver = (edge + 1) % kRanks;
        require_peer_access(edge, receiver);
        require_peer_access(receiver, edge);
    }
}

void enable_p2p_ring_peer_access_or_throw() {
    require_p2p_ring();
    for (int edge = 0; edge < kRanks; ++edge) {
        int receiver = (edge + 1) % kRanks;
        enable_peer_access_or_throw(edge, receiver);
        enable_peer_access_or_throw(receiver, edge);
    }
}

template <typename T>
P2pFifo<T>::P2pFifo(std::size_t slot_elems, int nranks)
    : P2pFifo(slot_elems, make_full_p2p_plan(nranks)) {}

template <typename T>
P2pFifo<T>::P2pFifo(std::size_t slot_elems, const RingTransportPlan& plan)
    : P2pFifo(slot_elems, plan, make_single_process_topology(plan)) {}

template <typename T>
P2pFifo<T>::P2pFifo(
    std::size_t slot_elems, const RingTransportPlan& plan,
    const collective::all_reduce::ProcessTopology& topology) {
    collective::all_reduce::validate_process_topology(topology);
    try {
        for (int channel = 0; channel < kChannels; ++channel) {
            for (int edge = 0; edge < kRanks; ++edge) {
                if (plan.edge_kind(edge) != TransportKind::P2p) {
                    continue;
                }
                int receiver_global_rank = (edge + 1) % kRanks;
                if (!collective::all_reduce::is_local_global_rank(topology, edge) ||
                    !collective::all_reduce::is_local_global_rank(
                        topology, receiver_global_rank)) {
                    throw std::runtime_error("P2pFifo requires local p2p edges");
                }
                int receiver_local_rank =
                    collective::all_reduce::local_rank_for_global_rank(
                        topology, receiver_global_rank);
                buffers_[channel][edge] = new core::DeviceBuffer<T>(
                    topology.devices[receiver_local_rank],
                    kSimpleFifoSteps * slot_elems);
            }
        }
    } catch (...) {
        for (int channel = 0; channel < kChannels; ++channel) {
            for (int edge = 0; edge < kRanks; ++edge) {
                delete buffers_[channel][edge];
                buffers_[channel][edge] = nullptr;
            }
        }
        throw;
    }
}

template <typename T>
P2pFifo<T>::~P2pFifo() {
    for (int channel = 0; channel < kChannels; ++channel) {
        for (int edge = 0; edge < kRanks; ++edge) {
            delete buffers_[channel][edge];
        }
    }
}

template <typename T>
T* P2pFifo<T>::edge_ptr(int channel, int edge) const {
    if (buffers_[channel][edge] == nullptr) {
        throw std::runtime_error("P2pFifo requested SHM edge " +
                                 std::to_string(edge));
    }
    return buffers_[channel][edge]->get();
}

template class P2pFifo<float>;
template class P2pFifo<__half>;
template class P2pFifo<__nv_bfloat16>;

}  // namespace nano_nccl::transport::p2p
