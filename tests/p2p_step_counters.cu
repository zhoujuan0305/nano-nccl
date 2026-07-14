#include "transport/p2p/p2p_fifo.h"
#include "transport/p2p/p2p_step_counters.h"
#include "transport/p2p/p2p_topology.h"

#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__,    \
                         __LINE__, cudaGetErrorString(err__));                \
            return 1;                                                         \
        }                                                                     \
    } while (0)

namespace {

bool check_owner(const std::uint64_t* pointer, int expected_device) {
    cudaPointerAttributes attributes{};
    cudaError_t err = cudaPointerGetAttributes(&attributes, pointer);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__,
                     cudaGetErrorString(err));
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice ||
        attributes.device != expected_device) {
        std::fprintf(stderr,
                     "counter owner mismatch: expected device=%d got "
                     "type=%d device=%d\\n",
                     expected_device, static_cast<int>(attributes.type),
                     attributes.device);
        return false;
    }
    return true;
}

bool check_mixed_control_directions() {
    using nano_nccl::TransportKind;
    using nano_nccl::transport::p2p::P2pStepCounters;
    using nano_nccl::transport::p2p::RingTransportPlan;

    RingTransportPlan plan({TransportKind::P2p, TransportKind::Shm,
                            TransportKind::P2p, TransportKind::Shm});
    P2pStepCounters counters(plan);
    const auto rank0 = counters.control_args(0);
    if (rank0.send_head[0] == nullptr || rank0.recv_tail[0] != nullptr) {
        std::fprintf(stderr, "rank 0 mixed counter directions are incorrect\n");
        return false;
    }

    const auto rank1 = counters.control_args(1);
    if (rank1.send_head[0] != nullptr || rank1.recv_tail[0] == nullptr) {
        std::fprintf(stderr, "rank 1 mixed counter directions are incorrect\n");
        return false;
    }

    return true;
}

bool check_isolated_rank_control_directions(
    cudaStream_t streams[nano_nccl::kRanks]) {
    using nano_nccl::TransportKind;
    using nano_nccl::transport::p2p::P2pStepCounters;
    using nano_nccl::transport::p2p::RingTransportPlan;

    RingTransportPlan plan({TransportKind::P2p, TransportKind::Shm,
                            TransportKind::Shm, TransportKind::Shm});
    P2pStepCounters counters(plan);
    counters.reset(streams);

    for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
        cudaError_t error = cudaSetDevice(rank);
        if (error != cudaSuccess) {
            std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__,
                         __LINE__, cudaGetErrorString(error));
            return false;
        }
        error = cudaStreamSynchronize(streams[rank]);
        if (error != cudaSuccess) {
            std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__,
                         __LINE__, cudaGetErrorString(error));
            return false;
        }
    }

    const auto isolated = counters.control_args(2);
    for (int channel = 0; channel < nano_nccl::kChannels; ++channel) {
        if (isolated.send_head[channel] != nullptr ||
            isolated.send_tail[channel] != nullptr ||
            isolated.recv_tail[channel] != nullptr ||
            isolated.recv_head[channel] != nullptr) {
            std::fprintf(stderr,
                         "isolated rank mixed counter directions are incorrect\\n");
            return false;
        }
    }
    if (isolated.base_steps != nullptr) {
        std::fprintf(stderr, "isolated rank base steps are incorrect\\n");
        return false;
    }

    return true;
}

bool check_mixed_fifo_rejects_shm_edge() {
    using nano_nccl::TransportKind;
    using nano_nccl::transport::p2p::P2pFifo;
    using nano_nccl::transport::p2p::RingTransportPlan;

    RingTransportPlan plan({TransportKind::P2p, TransportKind::Shm,
                            TransportKind::P2p, TransportKind::Shm});
    P2pFifo<float> fifo(1, plan);
    if (fifo.edge_ptr(0, 0) == nullptr) {
        std::fprintf(stderr, "P2P FIFO edge has no storage\n");
        return false;
    }
    try {
        fifo.edge_ptr(0, 1);
    } catch (const std::runtime_error&) {
        return true;
    }
    std::fprintf(stderr, "SHM FIFO edge did not throw\n");
    return false;
}

}  // namespace

int main() {
    if (!check_mixed_control_directions() ||
        !check_mixed_fifo_rejects_shm_edge()) {
        return 1;
    }

    if (!nano_nccl::transport::p2p::p2p_ring_available()) {
        std::puts("p2p_step_counters=SKIP reason=full directed ring P2P support is unavailable");
        return 0;
    }
    nano_nccl::transport::p2p::enable_p2p_ring_peer_access_or_throw();

    cudaStream_t streams[nano_nccl::kRanks]{};
    for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[rank],
                                             cudaStreamNonBlocking));
    }

    if (!check_isolated_rank_control_directions(streams)) {
        for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaStreamDestroy(streams[rank]));
        }
        return 1;
    }

    int result = 0;
    {
        nano_nccl::transport::p2p::P2pStepCounters counters(
            nano_nccl::kRanks);
        counters.reset(streams);

        for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
            const auto control = counters.control_args(rank);
            for (int channel = 0; channel < nano_nccl::kChannels; ++channel) {
                if (!check_owner(control.send_head[channel], rank) ||
                    !check_owner(control.recv_tail[channel], rank) ||
                    !check_owner(control.base_steps + channel, rank) ||
                    !check_owner(control.send_tail[channel],
                                 (rank + 1) % nano_nccl::kRanks) ||
                    !check_owner(control.recv_head[channel],
                                 (rank + nano_nccl::kRanks - 1) %
                                     nano_nccl::kRanks)) {
                    result = 1;
                }
            }
        }
    }

    for (int rank = 0; rank < nano_nccl::kRanks; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(cudaStreamDestroy(streams[rank]));
    }

    if (result != 0) {
        return result;
    }
    std::puts("p2p_step_counters=PASS");
    return 0;
}
