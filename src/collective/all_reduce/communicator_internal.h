#pragma once

#include "collective/all_reduce/topology.h"
#include "nano_nccl/communicator.h"
#include "transport/socket/socket_endpoint.h"

#include <memory>
#include <vector>

namespace nano_nccl::collective::all_reduce {

class SocketFdOwner {
public:
    SocketFdOwner() = default;
    explicit SocketFdOwner(std::vector<int> fds);
    static SocketFdOwner from_connections(
        std::vector<transport::socket::SocketConnection> connections);
    ~SocketFdOwner();

    SocketFdOwner(const SocketFdOwner&) = delete;
    SocketFdOwner& operator=(const SocketFdOwner&) = delete;
    SocketFdOwner(SocketFdOwner&& other) noexcept;
    SocketFdOwner& operator=(SocketFdOwner&& other) noexcept;

    std::vector<transport::socket::SocketConnection> release_connections() noexcept;

private:
    void close_all() noexcept;

    std::vector<transport::socket::SocketConnection> connections_;
};

class CommunicatorFactory {
public:
    static std::unique_ptr<Communicator> create(
        const CommunicatorConfig& config, ProcessTopology topology,
        SocketFdOwner socket_fds);
};

}  // namespace nano_nccl::collective::all_reduce
