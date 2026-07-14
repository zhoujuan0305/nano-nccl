#include "transport/p2p/p2p_topology.h"

#include "transport/p2p/p2p_fifo.h"

#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>
#include <nvml.h>

namespace nano_nccl::transport::p2p {

namespace {

class NvmlSession {
public:
    NvmlSession() : initialized_(nvmlInit_v2() == NVML_SUCCESS) {}

    ~NvmlSession() {
        if (initialized_) {
            nvmlShutdown();
        }
    }

    bool initialized() const { return initialized_; }

private:
    bool initialized_ = false;
};

bool cuda_device_pci_bus_id(int device, char* bus_id, int length) {
    return cudaDeviceGetPCIBusId(bus_id, length, device) == cudaSuccess;
}

bool pci_bus_ids_match(const char* left, const char* right) {
    unsigned int left_domain = 0;
    unsigned int left_bus = 0;
    unsigned int left_device = 0;
    unsigned int left_function = 0;
    unsigned int right_domain = 0;
    unsigned int right_bus = 0;
    unsigned int right_device = 0;
    unsigned int right_function = 0;
    return std::sscanf(left, "%x:%x:%x.%x", &left_domain, &left_bus,
                       &left_device, &left_function) == 4 &&
           std::sscanf(right, "%x:%x:%x.%x", &right_domain, &right_bus,
                       &right_device, &right_function) == 4 &&
           left_domain == right_domain && left_bus == right_bus &&
           left_device == right_device && left_function == right_function;
}

bool has_active_direct_nvlink(NvmlSession* nvml, int src, int dst) {
    if (!nvml->initialized()) {
        return false;
    }

    char src_pci_bus_id[32]{};
    char dst_pci_bus_id[32]{};
    if (!cuda_device_pci_bus_id(src, src_pci_bus_id, sizeof(src_pci_bus_id)) ||
        !cuda_device_pci_bus_id(dst, dst_pci_bus_id, sizeof(dst_pci_bus_id))) {
        return false;
    }

    nvmlDevice_t src_device;
    if (nvmlDeviceGetHandleByPciBusId(src_pci_bus_id, &src_device) !=
        NVML_SUCCESS) {
        return false;
    }

    for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link) {
        nvmlEnableState_t state;
        if (nvmlDeviceGetNvLinkState(src_device, link, &state) != NVML_SUCCESS) {
            return false;
        }
        if (state != NVML_FEATURE_ENABLED) {
            continue;
        }

        nvmlPciInfo_t remote_pci{};
        if (nvmlDeviceGetNvLinkRemotePciInfo(src_device, link, &remote_pci) !=
            NVML_SUCCESS) {
            return false;
        }
        if (pci_bus_ids_match(remote_pci.busId, dst_pci_bus_id)) {
            return true;
        }
    }
    return false;
}

bool cuda_peer_access_available(int src, int dst) {
    int can_access = 0;
    return cudaDeviceCanAccessPeer(&can_access, src, dst) == cudaSuccess &&
           can_access != 0;
}

bool edge_supports_p2p(NvmlSession* nvml, int sender, int receiver) {
    return has_active_direct_nvlink(nvml, sender, receiver) &&
           has_active_direct_nvlink(nvml, receiver, sender) &&
           cuda_peer_access_available(sender, receiver) &&
           cuda_peer_access_available(receiver, sender);
}

void enable_peer_access_or_throw(int src, int dst) {
    CUDA_CHECK_THROW(cudaSetDevice(src));
    cudaError_t error = cudaDeviceEnablePeerAccess(dst, 0);
    if (error == cudaErrorPeerAccessAlreadyEnabled) {
        cudaError_t last_error = cudaGetLastError();
        if (last_error != cudaSuccess &&
            last_error != cudaErrorPeerAccessAlreadyEnabled) {
            throw std::runtime_error(cudaGetErrorString(last_error));
        }
    } else if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

}  // namespace

RingTransportPlan::RingTransportPlan(
    std::array<TransportKind, kRanks> edge_kinds)
    : edge_kinds_(edge_kinds) {
    for (TransportKind kind : edge_kinds_) {
        if (kind == TransportKind::Auto || kind == TransportKind::Mixed) {
            throw std::invalid_argument(
                "ring transport edges must be p2p or shm");
        }
    }
}

RingTransportPlan RingTransportPlan::uniform(TransportKind kind) {
    std::array<TransportKind, kRanks> edge_kinds;
    edge_kinds.fill(kind);
    return RingTransportPlan(edge_kinds);
}

TransportKind RingTransportPlan::edge_kind(int edge) const {
    return edge_kinds_.at(edge);
}

TransportKind RingTransportPlan::resolved_kind() const {
    TransportKind first = edge_kinds_[0];
    for (TransportKind kind : edge_kinds_) {
        if (kind != first) {
            return TransportKind::Mixed;
        }
    }
    return first;
}

bool RingTransportPlan::uses_p2p() const {
    for (TransportKind kind : edge_kinds_) {
        if (kind == TransportKind::P2p) {
            return true;
        }
    }
    return false;
}

RingTransportPlan resolve_ring_transport(TransportKind requested) {
    if (requested == TransportKind::Shm) {
        return RingTransportPlan::uniform(TransportKind::Shm);
    }
    if (requested == TransportKind::P2p) {
        require_p2p_ring();
        return RingTransportPlan::uniform(TransportKind::P2p);
    }
    if (requested != TransportKind::Auto) {
        throw std::invalid_argument("mixed is a resolved transport only");
    }

    NvmlSession nvml;
    std::array<TransportKind, kRanks> edge_kinds{};
    for (int edge = 0; edge < kRanks; ++edge) {
        int receiver = (edge + 1) % kRanks;
        edge_kinds[edge] = edge_supports_p2p(&nvml, edge, receiver)
                               ? TransportKind::P2p
                               : TransportKind::Shm;
    }
    return RingTransportPlan(edge_kinds);
}

void enable_p2p_ring_peer_access_or_throw(const RingTransportPlan& plan) {
    for (int edge = 0; edge < kRanks; ++edge) {
        if (plan.edge_kind(edge) != TransportKind::P2p) {
            continue;
        }
        int receiver = (edge + 1) % kRanks;
        enable_peer_access_or_throw(edge, receiver);
        enable_peer_access_or_throw(receiver, edge);
    }
}

}  // namespace nano_nccl::transport::p2p
