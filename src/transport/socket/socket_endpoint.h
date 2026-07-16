#pragma once

#include <cstdint>

namespace nano_nccl::transport::socket {

constexpr std::uint32_t kSocketHelloMagic = 0x4e4e434c;
constexpr std::uint16_t kSocketProtocolVersion = 1;

struct SocketAddress {
    std::uint32_t ipv4_address = 0;
    std::uint16_t port = 0;
    std::uint16_t reserved = 0;
};

struct SocketHello {
    int source_global_rank = -1;
    int destination_global_rank = -1;
    int channel = -1;
};

class SocketConnection {
public:
    SocketConnection() = default;
    explicit SocketConnection(int fd, SocketHello hello = {});
    ~SocketConnection();

    SocketConnection(const SocketConnection&) = delete;
    SocketConnection& operator=(const SocketConnection&) = delete;
    SocketConnection(SocketConnection&& other) noexcept;
    SocketConnection& operator=(SocketConnection&& other) noexcept;

    int fd() const noexcept { return fd_; }
    void close() noexcept;
    int release() noexcept;
    const SocketHello& hello() const noexcept { return hello_; }
    void set_hello(SocketHello hello) noexcept { hello_ = hello; }

private:
    int fd_ = -1;
    SocketHello hello_{};
};

class SocketEndpoint {
public:
    static SocketEndpoint create_from_environment();

    SocketEndpoint() = default;
    ~SocketEndpoint();

    SocketEndpoint(const SocketEndpoint&) = delete;
    SocketEndpoint& operator=(const SocketEndpoint&) = delete;
    SocketEndpoint(SocketEndpoint&& other) noexcept;
    SocketEndpoint& operator=(SocketEndpoint&& other) noexcept;

    SocketAddress address() const noexcept { return address_; }
    SocketConnection connect(SocketAddress remote, SocketHello hello) const;
    SocketConnection accept();
    SocketHello read_hello(const SocketConnection& connection) const;

private:
    SocketEndpoint(int listener_fd, SocketAddress address) noexcept;
    void close() noexcept;

    int listener_fd_ = -1;
    SocketAddress address_{};
};

}  // namespace nano_nccl::transport::socket
