#include "nano_nccl/mpi.h"

#include "collective/all_reduce/communicator_internal.h"
#include "collective/all_reduce/topology.h"
#include "transport/p2p/p2p_topology.h"
#include "transport/socket/socket_endpoint.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

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
    return source_process != destination_process &&
           (source_process == mpi_rank || destination_process == mpi_rank) &&
           mpi_rank == std::max(source_process, destination_process) &&
           topology.edge_kinds[hello.source_global_rank] == TransportKind::Socket;
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
    if (config.devices.empty()) {
        throw std::runtime_error("MPI communicator requires at least one local device");
    }

    int mpi_rank = 0;
    int mpi_size = 0;
    mpi_check(MPI_Comm_rank(control_comm, &mpi_rank), "MPI_Comm_rank");
    mpi_check(MPI_Comm_size(control_comm, &mpi_size), "MPI_Comm_size");
    if (mpi_size <= 0) throw std::runtime_error("MPI communicator has no processes");

    int local_count = static_cast<int>(config.devices.size());
    std::vector<int> process_counts(mpi_size);
    mpi_check(MPI_Allgather(&local_count, 1, MPI_INT, process_counts.data(), 1, MPI_INT,
                            control_comm), "MPI_Allgather(local GPU counts)");
    for (int count : process_counts) {
        if (count <= 0) {
            throw std::runtime_error("every MPI process must manage at least one GPU");
        }
    }

    int local_offset = 0;
    mpi_check(MPI_Exscan(&local_count, &local_offset, 1, MPI_INT, MPI_SUM,
                          control_comm), "MPI_Exscan(local GPU counts)");
    if (mpi_rank == 0) local_offset = 0;

    int global_count = 0;
    mpi_check(MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM,
                            control_comm), "MPI_Allreduce(global GPU count)");
    if (global_count != kRanks) {
        throw std::runtime_error("MPI global GPU count must match kRanks=" +
                                 std::to_string(kRanks));
    }

    ProcessTopology topology{
        global_count,
        local_offset,
        config.devices,
        std::vector<TransportKind>(global_count, TransportKind::Shm),
        true,
    };
    for (int edge = 0; edge < global_count; ++edge) {
        int receiver = (edge + 1) % global_count;
        if (process_for_global_rank(process_counts, edge) !=
            process_for_global_rank(process_counts, receiver)) {
            topology.edge_kinds[edge] = TransportKind::Socket;
        }
    }
    collective::all_reduce::validate_process_topology(topology);

    if (config.transport == TransportKind::Auto) {
        topology.edge_kinds = transport::p2p::resolve_ring_transport(
            config.transport, topology).edge_kinds();
    } else if (config.transport == TransportKind::Socket) {
        for (int edge = 0; edge < global_count; ++edge) {
            if (topology.edge_kinds[edge] != TransportKind::Socket) {
                topology.edge_kinds[edge] = TransportKind::Shm;
            }
        }
    } else {
        throw std::invalid_argument(
            "distributed communicators require auto or socket transport");
    }
    collective::all_reduce::validate_process_topology(topology);

    SocketEndpoint listener = create_listener_with_consensus(control_comm);
    SocketAddress local_endpoint = listener.address();
    std::vector<SocketAddress> endpoints(mpi_size);
    mpi_check(MPI_Allgather(&local_endpoint, sizeof(local_endpoint), MPI_BYTE,
                            endpoints.data(), sizeof(local_endpoint), MPI_BYTE,
                            control_comm), "MPI_Allgather(socket endpoints)");

    std::vector<SocketConnection> connections;
    int expected_accepts = 0;
    for (int edge = 0; edge < global_count; ++edge) {
        if (topology.edge_kinds[edge] != TransportKind::Socket) continue;
        int receiver = (edge + 1) % global_count;
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
            std::move(connections)));
}

}  // namespace nano_nccl
