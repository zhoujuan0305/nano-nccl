#include "collective/all_reduce/topology.h"

#include <stdexcept>

namespace nano_nccl::collective::all_reduce {

namespace {

bool is_resolved_edge_kind(TransportKind kind) {
    return kind == TransportKind::Shm || kind == TransportKind::P2p ||
           kind == TransportKind::Socket;
}

}  // namespace

ProcessTopology make_single_process_topology(
    std::vector<int> devices, std::vector<TransportKind> edge_kinds) {
    ProcessTopology topology{
        static_cast<int>(devices.size()),
        0,
        std::move(devices),
        std::move(edge_kinds),
        false,
    };
    validate_process_topology(topology);
    return topology;
}

void validate_process_topology(const ProcessTopology& topology) {
    if (topology.global_rank_count != kRanks) {
        throw std::runtime_error("process topology global rank count must match kRanks");
    }
    if (topology.devices.empty()) {
        throw std::runtime_error("process topology requires local devices");
    }
    if (topology.local_rank_offset < 0 ||
        topology.local_rank_offset + static_cast<int>(topology.devices.size()) >
            topology.global_rank_count) {
        throw std::runtime_error("process topology local ranks are out of range");
    }
    if (topology.edge_kinds.size() !=
        static_cast<std::size_t>(topology.global_rank_count)) {
        throw std::runtime_error("process topology requires one kind per global edge");
    }
    for (TransportKind kind : topology.edge_kinds) {
        if (!is_resolved_edge_kind(kind)) {
            throw std::runtime_error("process topology edge kind must be resolved");
        }
    }
    if (!topology.distributed &&
        (topology.local_rank_offset != 0 ||
         static_cast<int>(topology.devices.size()) != topology.global_rank_count)) {
        throw std::runtime_error("single-process topology must own every rank");
    }
}

bool is_local_global_rank(const ProcessTopology& topology, int global_rank) {
    return global_rank >= topology.local_rank_offset &&
           global_rank < topology.local_rank_offset +
                             static_cast<int>(topology.devices.size());
}

int local_rank_for_global_rank(const ProcessTopology& topology, int global_rank) {
    if (!is_local_global_rank(topology, global_rank)) {
        throw std::runtime_error("global rank is not local to this process");
    }
    return global_rank - topology.local_rank_offset;
}

}  // namespace nano_nccl::collective::all_reduce
