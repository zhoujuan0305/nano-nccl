#include "transport/socket/socket_endpoint.h"

#include <cstddef>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>

namespace nano_nccl::transport::socket {

namespace {

struct WireHello {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t channel;
    std::uint32_t source_global_rank;
    std::uint32_t destination_global_rank;
};

static_assert(sizeof(WireHello) == 16);

std::runtime_error socket_error(const char* operation) {
    return std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

void write_all(int fd, const void* data, std::size_t bytes) {
    const auto* cursor = static_cast<const char*>(data);
    while (bytes != 0) {
        ssize_t written = send(fd, cursor, bytes, MSG_NOSIGNAL);
        if (written < 0) {
            if (errno == EINTR) continue;
            throw socket_error("send");
        }
        if (written == 0) throw std::runtime_error("send returned zero bytes");
        cursor += written;
        bytes -= static_cast<std::size_t>(written);
    }
}

void read_all(int fd, void* data, std::size_t bytes) {
    auto* cursor = static_cast<char*>(data);
    while (bytes != 0) {
        ssize_t received = recv(fd, cursor, bytes, 0);
        if (received < 0) {
            if (errno == EINTR) continue;
            throw socket_error("recv");
        }
        if (received == 0) throw std::runtime_error("socket peer closed during HELLO");
        cursor += received;
        bytes -= static_cast<std::size_t>(received);
    }
}

SocketAddress select_interface_address() {
    const char* interface_name = std::getenv("NANO_NCCL_SOCKET_IFNAME");
    if (interface_name == nullptr || interface_name[0] == '\0') {
        throw std::runtime_error("NANO_NCCL_SOCKET_IFNAME must be set");
    }

    ifaddrs* interfaces = nullptr;
    if (getifaddrs(&interfaces) != 0) throw socket_error("getifaddrs");

    SocketAddress selected{};
    int matches = 0;
    for (ifaddrs* entry = interfaces; entry != nullptr; entry = entry->ifa_next) {
        if (entry->ifa_name == nullptr || entry->ifa_addr == nullptr ||
            std::strcmp(entry->ifa_name, interface_name) != 0 ||
            entry->ifa_addr->sa_family != AF_INET ||
            (entry->ifa_flags & IFF_UP) == 0 ||
            (entry->ifa_flags & IFF_LOOPBACK) != 0) {
            continue;
        }
        const auto* address = reinterpret_cast<const sockaddr_in*>(entry->ifa_addr);
        selected.ipv4_address = address->sin_addr.s_addr;
        ++matches;
    }
    freeifaddrs(interfaces);

    if (matches != 1) {
        throw std::runtime_error("NANO_NCCL_SOCKET_IFNAME must identify exactly one usable IPv4 address");
    }
    return selected;
}

}  // namespace

SocketConnection::SocketConnection(int fd, SocketHello hello) : fd_(fd), hello_(hello) {}

SocketConnection::~SocketConnection() { close(); }

SocketConnection::SocketConnection(SocketConnection&& other) noexcept
    : fd_(other.fd_), hello_(other.hello_) {
    other.fd_ = -1;
}

SocketConnection& SocketConnection::operator=(SocketConnection&& other) noexcept {
    if (this == &other) return *this;
    close();
    fd_ = other.fd_;
    hello_ = other.hello_;
    other.fd_ = -1;
    return *this;
}

void SocketConnection::close() noexcept {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

int SocketConnection::release() noexcept {
    int fd = fd_;
    fd_ = -1;
    return fd;
}

SocketEndpoint::SocketEndpoint(int listener_fd, SocketAddress address) noexcept
    : listener_fd_(listener_fd), address_(address) {}

SocketEndpoint SocketEndpoint::create_from_environment() {
    SocketAddress address = select_interface_address();
    int listener_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener_fd < 0) throw socket_error("socket");

    try {
        int reuse_address = 1;
        if (setsockopt(listener_fd, SOL_SOCKET, SO_REUSEADDR, &reuse_address,
                       sizeof(reuse_address)) != 0) {
            throw socket_error("setsockopt(SO_REUSEADDR)");
        }
        sockaddr_in bind_address{};
        bind_address.sin_family = AF_INET;
        bind_address.sin_addr.s_addr = address.ipv4_address;
        bind_address.sin_port = 0;
        if (bind(listener_fd, reinterpret_cast<const sockaddr*>(&bind_address),
                 sizeof(bind_address)) != 0) {
            throw socket_error("bind");
        }
        if (listen(listener_fd, SOMAXCONN) != 0) throw socket_error("listen");

        socklen_t length = sizeof(bind_address);
        if (getsockname(listener_fd, reinterpret_cast<sockaddr*>(&bind_address),
                        &length) != 0) {
            throw socket_error("getsockname");
        }
        address.port = ntohs(bind_address.sin_port);
    } catch (...) {
        ::close(listener_fd);
        throw;
    }
    return SocketEndpoint(listener_fd, address);
}

SocketEndpoint::~SocketEndpoint() { close(); }

SocketEndpoint::SocketEndpoint(SocketEndpoint&& other) noexcept
    : listener_fd_(other.listener_fd_), address_(other.address_) {
    other.listener_fd_ = -1;
}

SocketEndpoint& SocketEndpoint::operator=(SocketEndpoint&& other) noexcept {
    if (this == &other) return *this;
    close();
    listener_fd_ = other.listener_fd_;
    address_ = other.address_;
    other.listener_fd_ = -1;
    return *this;
}

void SocketEndpoint::close() noexcept {
    if (listener_fd_ >= 0) {
        ::close(listener_fd_);
        listener_fd_ = -1;
    }
}

SocketConnection SocketEndpoint::connect(SocketAddress remote, SocketHello hello) const {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw socket_error("socket");
    try {
        sockaddr_in remote_address{};
        remote_address.sin_family = AF_INET;
        remote_address.sin_addr.s_addr = remote.ipv4_address;
        remote_address.sin_port = htons(remote.port);
        if (::connect(fd, reinterpret_cast<const sockaddr*>(&remote_address),
                      sizeof(remote_address)) != 0) {
            throw socket_error("connect");
        }
        WireHello wire{
            htonl(kSocketHelloMagic),
            htons(kSocketProtocolVersion),
            htons(static_cast<std::uint16_t>(hello.channel)),
            htonl(static_cast<std::uint32_t>(hello.source_global_rank)),
            htonl(static_cast<std::uint32_t>(hello.destination_global_rank)),
        };
        write_all(fd, &wire, sizeof(wire));
    } catch (...) {
        ::close(fd);
        throw;
    }
    return SocketConnection(fd, hello);
}

SocketConnection SocketEndpoint::accept() {
    int fd = -1;
    do {
        fd = ::accept(listener_fd_, nullptr, nullptr);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) throw socket_error("accept");
    return SocketConnection(fd);
}

SocketHello SocketEndpoint::read_hello(const SocketConnection& connection) const {
    WireHello wire{};
    read_all(connection.fd(), &wire, sizeof(wire));
    if (ntohl(wire.magic) != kSocketHelloMagic) {
        throw std::runtime_error("socket HELLO has invalid magic");
    }
    if (ntohs(wire.version) != kSocketProtocolVersion) {
        throw std::runtime_error("socket HELLO has unsupported version");
    }
    return SocketHello{
        static_cast<int>(ntohl(wire.source_global_rank)),
        static_cast<int>(ntohl(wire.destination_global_rank)),
        static_cast<int>(ntohs(wire.channel)),
    };
}

}  // namespace nano_nccl::transport::socket
