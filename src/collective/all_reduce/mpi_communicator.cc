#include "nano_nccl/mpi.h"

#include "collective/all_reduce/communicator_internal.h"
#include "collective/all_reduce/topology.h"
#include "transport/p2p/mpi_p2p.h"
#include "transport/p2p/p2p_topology.h"
#include "transport/shm/mpi_shm.h"
#include "transport/connection.h"
#include "transport/socket/socket_endpoint.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace nano_nccl {

namespace {

using collective::all_reduce::ProcessTopology;
using transport::socket::SocketAddress;
using transport::socket::SocketConnection;
using transport::socket::SocketEndpoint;
using transport::socket::SocketHello;

void mpi_check(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    throw std::runtime_error(std::string(operation) + ": " +
                             std::string(error, static_cast<std::size_t>(length)));
}

void validate_local_device_with_consensus(MPI_Comm control_comm, int mpi_rank,
                                          int device) {
    std::array<char, 256> local_error{};
    int local_ok = 1;
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        local_ok = 0;
        std::snprintf(local_error.data(), local_error.size(),
                      "MPI rank %d cudaGetDeviceCount failed: %s", mpi_rank,
                      cudaGetErrorString(status));
        cudaGetLastError();
    } else if (device < 0 || device >= device_count) {
        local_ok = 0;
        std::snprintf(
            local_error.data(), local_error.size(),
            "MPI rank %d selected invalid CUDA device %d (visible device count=%d)",
            mpi_rank, device, device_count);
    } else {
        status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            local_ok = 0;
            std::snprintf(local_error.data(), local_error.size(),
                          "MPI rank %d cudaSetDevice failed: %s", mpi_rank,
                          cudaGetErrorString(status));
            cudaGetLastError();
        }
    }

    int all_ok = 0;
    mpi_check(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN,
                            control_comm),
              "MPI_Allreduce(local device validation)");
    if (all_ok != 0) return;

    std::vector<std::array<char, 256>> errors(kRanks);
    mpi_check(MPI_Allgather(local_error.data(),
                            static_cast<int>(local_error.size()), MPI_CHAR,
                            errors.data(), static_cast<int>(local_error.size()),
                            MPI_CHAR, control_comm),
              "MPI_Allgather(local device validation errors)");
    for (const auto& error : errors) {
        if (error[0] != '\0') throw std::invalid_argument(error.data());
    }
    throw std::invalid_argument("one-process-per-GPU device validation failed");
}

void validate_unique_device_per_node(MPI_Comm control_comm, int mpi_rank,
                                     int device,
                                     const std::vector<int>& node_ids) {
    cudaDeviceProp properties{};
    std::array<char, 256> local_error{};
    int local_ok = 1;
    const cudaError_t status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        local_ok = 0;
        std::snprintf(local_error.data(), local_error.size(),
                      "MPI rank %d cudaGetDeviceProperties failed: %s", mpi_rank,
                      cudaGetErrorString(status));
        cudaGetLastError();
    }

    int all_ok = 0;
    mpi_check(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN,
                            control_comm),
              "MPI_Allreduce(CUDA device identity)");
    if (all_ok == 0) {
        std::vector<std::array<char, 256>> errors(kRanks);
        mpi_check(MPI_Allgather(local_error.data(),
                                static_cast<int>(local_error.size()), MPI_CHAR,
                                errors.data(),
                                static_cast<int>(local_error.size()), MPI_CHAR,
                                control_comm),
                  "MPI_Allgather(CUDA device identity errors)");
        for (const auto& error : errors) {
            if (error[0] != '\0') throw std::runtime_error(error.data());
        }
        throw std::runtime_error("CUDA device identity validation failed");
    }

    std::vector<cudaUUID_t> uuids(kRanks);
    mpi_check(MPI_Allgather(&properties.uuid, sizeof(properties.uuid), MPI_BYTE,
                            uuids.data(), sizeof(properties.uuid), MPI_BYTE,
                            control_comm),
              "MPI_Allgather(CUDA device identities)");
    for (int first = 0; first < kRanks; ++first) {
        for (int second = first + 1; second < kRanks; ++second) {
            if (node_ids[first] == node_ids[second] &&
                std::memcmp(&uuids[first], &uuids[second],
                            sizeof(cudaUUID_t)) == 0) {
                throw std::invalid_argument(
                    "MPI ranks " + std::to_string(first) + " and " +
                    std::to_string(second) +
                    " select the same CUDA device on one host");
            }
        }
    }
}

SocketEndpoint create_listener_with_consensus(MPI_Comm control_comm) {
    constexpr std::size_t kErrorBytes = 256;
    SocketEndpoint listener;
    std::array<char, kErrorBytes> local_error{};
    int local_ok = 1;
    try {
        listener = SocketEndpoint::create_from_environment();
    } catch (const std::exception& error) {
        local_ok = 0;
        std::strncpy(local_error.data(), error.what(), local_error.size() - 1);
    }

    int all_ok = 0;
    mpi_check(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN,
                            control_comm),
              "MPI_Allreduce(socket listener setup)");
    if (all_ok != 0) return listener;

    std::vector<std::array<char, kErrorBytes>> errors;
    int mpi_size = 0;
    mpi_check(MPI_Comm_size(control_comm, &mpi_size), "MPI_Comm_size");
    errors.resize(static_cast<std::size_t>(mpi_size));
    mpi_check(MPI_Allgather(local_error.data(), static_cast<int>(kErrorBytes), MPI_CHAR,
                            errors.data(), static_cast<int>(kErrorBytes), MPI_CHAR,
                            control_comm),
              "MPI_Allgather(socket listener setup errors)");

    std::ostringstream message;
    message << "socket listener setup failed";
    for (int rank = 0; rank < mpi_size; ++rank) {
        const char* error = errors[static_cast<std::size_t>(rank)].data();
        if (error[0] != '\0') {
            message << " on MPI rank " << rank << ": " << error;
            break;
        }
    }
    throw std::runtime_error(message.str());
}

int process_for_global_rank(const std::vector<int>& counts, int global_rank) {
    int offset = 0;
    for (int process = 0; process < static_cast<int>(counts.size()); ++process) {
        if (global_rank < offset + counts[process]) return process;
        offset += counts[process];
    }
    throw std::runtime_error("global rank does not belong to an MPI process");
}

std::uint64_t hello_key(const SocketHello& hello) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(hello.source_global_rank)) << 32) |
           (static_cast<std::uint64_t>(static_cast<std::uint16_t>(hello.destination_global_rank)) << 16) |
           static_cast<std::uint16_t>(hello.channel);
}

bool is_expected_hello(const SocketHello& hello, const ProcessTopology& topology,
                       const std::vector<int>& process_counts, int mpi_rank) {
    if (hello.source_global_rank < 0 || hello.source_global_rank >= kRanks ||
        hello.destination_global_rank != (hello.source_global_rank + 1) % kRanks ||
        hello.channel < 0 || hello.channel >= kChannels) {
        return false;
    }
    int source_process = process_for_global_rank(process_counts, hello.source_global_rank);
    int destination_process = process_for_global_rank(
        process_counts, hello.destination_global_rank);
    TransportKind edge_kind = topology.edge_kinds[hello.source_global_rank];
    return source_process != destination_process &&
           (source_process == mpi_rank || destination_process == mpi_rank) &&
           mpi_rank == std::max(source_process, destination_process) &&
           (edge_kind == TransportKind::Socket ||
            edge_kind == TransportKind::Rdma);
}

std::vector<int> discover_node_ids(MPI_Comm control_comm, int global_rank) {
    MPI_Comm node_comm = MPI_COMM_NULL;
    mpi_check(MPI_Comm_split_type(control_comm, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &node_comm),
              "MPI_Comm_split_type(topology)");
    int node_rank = 0;
    mpi_check(MPI_Comm_rank(node_comm, &node_rank),
              "MPI_Comm_rank(node topology)");
    int node_id = node_rank == 0 ? global_rank : -1;
    mpi_check(MPI_Bcast(&node_id, 1, MPI_INT, 0, node_comm),
              "MPI_Bcast(node id)");
    mpi_check(MPI_Comm_free(&node_comm), "MPI_Comm_free(node topology)");

    std::vector<int> node_ids(kRanks);
    mpi_check(MPI_Allgather(&node_id, 1, MPI_INT, node_ids.data(), 1,
                            MPI_INT, control_comm),
              "MPI_Allgather(node ids)");
    return node_ids;
}

std::vector<TransportKind> resolve_interprocess_transports(
    TransportKind requested, const std::vector<int>& node_ids,
    const std::vector<bool>& p2p_capable) {
    std::vector<TransportKind> edge_kinds(kRanks, TransportKind::Shm);
    for (int edge = 0; edge < kRanks; ++edge) {
        const int receiver = (edge + 1) % kRanks;
        const bool same_host = node_ids[edge] == node_ids[receiver];
        switch (requested) {
            case TransportKind::Auto:
                edge_kinds[edge] = same_host
                    ? (p2p_capable[edge] ? TransportKind::P2p
                                          : TransportKind::Shm)
                    : TransportKind::Socket;
                break;
            case TransportKind::Shm:
                if (!same_host) {
                    throw std::invalid_argument(
                        "SHM edge " + std::to_string(edge) + "->" +
                        std::to_string(receiver) +
                        " unavailable across hosts; actual available backend is socket");
                }
                edge_kinds[edge] = TransportKind::Shm;
                break;
            case TransportKind::P2p:
                if (!same_host) {
                    throw std::invalid_argument(
                        "P2P edge " + std::to_string(edge) + "->" +
                        std::to_string(receiver) +
                        " unavailable across hosts; actual available backend is socket");
                }
                if (!p2p_capable[edge]) {
                    throw std::invalid_argument(
                        "P2P edge " + std::to_string(edge) + "->" +
                        std::to_string(receiver) +
                        " failed the CUDA IPC probe; actual available backend is shm");
                }
                edge_kinds[edge] = TransportKind::P2p;
                break;
            case TransportKind::Socket:
                edge_kinds[edge] = TransportKind::Socket;
                break;
            case TransportKind::Rdma:
#if defined(NANO_NCCL_ENABLE_RDMA)
                edge_kinds[edge] = same_host
                    ? (p2p_capable[edge] ? TransportKind::P2p
                                          : TransportKind::Shm)
                    : TransportKind::Rdma;
#else
                throw std::invalid_argument(
                    "rdma transport requires NANO_NCCL_ENABLE_RDMA=ON");
#endif
                break;
            case TransportKind::Mixed:
                throw std::invalid_argument(
                    "mixed is a resolved transport and cannot be requested");
        }
    }
    return edge_kinds;
}

}  // namespace

std::unique_ptr<Communicator> create_communicator_from_mpi(
    MPI_Comm control_comm, const CommunicatorConfig& config) {
    int initialized = 0;
    mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
    if (initialized == 0) {
        throw std::runtime_error("MPI must be initialized before creating a communicator");
    }
    int finalized = 0;
    mpi_check(MPI_Finalized(&finalized), "MPI_Finalized");
    if (finalized != 0) {
        throw std::runtime_error("MPI has already been finalized");
    }
    int mpi_rank = 0;
    int mpi_size = 0;
    mpi_check(MPI_Comm_rank(control_comm, &mpi_rank), "MPI_Comm_rank");
    mpi_check(MPI_Comm_size(control_comm, &mpi_size), "MPI_Comm_size");
    if (mpi_size != kRanks) {
        throw std::runtime_error("MPI process count must match kRanks=" +
                                 std::to_string(kRanks));
    }
    validate_local_device_with_consensus(control_comm, mpi_rank, config.device);

    std::vector<int> process_counts(mpi_size, 1);
    const std::vector<int> node_ids = discover_node_ids(control_comm, mpi_rank);
    validate_unique_device_per_node(control_comm, mpi_rank, config.device,
                                    node_ids);
    std::vector<bool> p2p_capable(kRanks, false);
    if (config.transport == TransportKind::Auto ||
        config.transport == TransportKind::P2p ||
        config.transport == TransportKind::Rdma) {
        p2p_capable = transport::p2p::probe_mpi_p2p_edges(
            control_comm, mpi_rank, config.device, node_ids);
    }
    std::vector<TransportKind> edge_kinds = resolve_interprocess_transports(
        config.transport, node_ids, p2p_capable);

    ProcessTopology topology{
        kRanks,
        mpi_rank,
        {config.device},
        std::move(edge_kinds),
        true,
    };
    collective::all_reduce::validate_process_topology(topology);

    transport::ConnectionResources transport_connections;
    transport::merge_connection_resources(
        &transport_connections, transport::shm::create_mpi_shm_connections(
            control_comm, mpi_rank, config.device, topology.edge_kinds));
    transport::merge_connection_resources(
        &transport_connections, transport::p2p::create_mpi_p2p_connections(
            control_comm, mpi_rank, config.device, topology.edge_kinds));

    const bool uses_network =
        std::find(topology.edge_kinds.begin(), topology.edge_kinds.end(),
                  TransportKind::Socket) != topology.edge_kinds.end() ||
        std::find(topology.edge_kinds.begin(), topology.edge_kinds.end(),
                  TransportKind::Rdma) != topology.edge_kinds.end();
    SocketEndpoint listener;
    SocketAddress local_endpoint{};
    if (uses_network) {
        listener = create_listener_with_consensus(control_comm);
        local_endpoint = listener.address();
    }
    std::vector<SocketAddress> endpoints(mpi_size);
    mpi_check(MPI_Allgather(&local_endpoint, sizeof(local_endpoint), MPI_BYTE,
                            endpoints.data(), sizeof(local_endpoint), MPI_BYTE,
                            control_comm), "MPI_Allgather(socket endpoints)");

    std::vector<SocketConnection> connections;
    int expected_accepts = 0;
    for (int edge = 0; edge < kRanks; ++edge) {
        // Both Socket and Rdma edges walk through this TCP bootstrap. Rdma
        // re-uses the same fds afterwards for the RdmaPeerInfo swap inside
        // Communicator::Impl::setup_rdma_transport().
        if (topology.edge_kinds[edge] != TransportKind::Socket &&
            topology.edge_kinds[edge] != TransportKind::Rdma) {
            continue;
        }
        int receiver = (edge + 1) % kRanks;
        int source_process = process_for_global_rank(process_counts, edge);
        int destination_process = process_for_global_rank(process_counts, receiver);
        int remote_process = source_process == mpi_rank ? destination_process : source_process;
        if (source_process != mpi_rank && destination_process != mpi_rank) continue;
        for (int channel = 0; channel < kChannels; ++channel) {
            SocketHello hello{edge, receiver, channel};
            if (mpi_rank < remote_process) {
                connections.push_back(listener.connect(endpoints[remote_process], hello));
            } else {
                ++expected_accepts;
            }
        }
    }

    std::unordered_set<std::uint64_t> accepted_hellos;
    for (int index = 0; index < expected_accepts; ++index) {
        SocketConnection connection = listener.accept();
        SocketHello hello = listener.read_hello(connection);
        if (!is_expected_hello(hello, topology, process_counts, mpi_rank)) {
            throw std::runtime_error("socket HELLO describes an unexpected ring edge");
        }
        if (!accepted_hellos.insert(hello_key(hello)).second) {
            throw std::runtime_error("socket HELLO duplicates a ring edge and channel");
        }
        connection.set_hello(hello);
        connections.push_back(std::move(connection));
    }

    return collective::all_reduce::CommunicatorFactory::create(
        config, std::move(topology),
        collective::all_reduce::SocketFdOwner::from_connections(
            std::move(connections)),
        std::move(transport_connections));
}

}  // namespace nano_nccl
