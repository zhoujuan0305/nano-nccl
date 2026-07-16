#include "transport/socket/socket_protocol.h"
#include "transport/socket/socket_proxy.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

class SocketPair {
public:
    SocketPair() {
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds_) != 0) {
            throw std::runtime_error("socketpair failed");
        }
    }

    ~SocketPair() {
        for (int& fd : fds_) {
            if (fd >= 0) ::close(fd);
        }
    }

    int first() const { return fds_[0]; }
    int second() const { return fds_[1]; }
    int release_first() {
        int fd = fds_[0];
        fds_[0] = -1;
        return fd;
    }
    int release_second() {
        int fd = fds_[1];
        fds_[1] = -1;
        return fd;
    }
    void close_first() {
        if (fds_[0] >= 0) {
            ::close(fds_[0]);
            fds_[0] = -1;
        }
    }

private:
    int fds_[2]{-1, -1};
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void sends_big_endian_header_and_payload() {
    SocketPair sockets;
    constexpr std::array<std::uint8_t, 3> kPayload{0x12, 0x34, 0x56};

    std::thread sender([&] {
        nano_nccl::transport::socket::send_slice(
            sockets.first(), kPayload.data(), kPayload.size());
    });

    nano_nccl::transport::socket::SocketSliceHeader header{};
    nano_nccl::transport::socket::recv_all(sockets.second(), &header,
                                            sizeof(header));
    require(ntohl(header.payload_bytes_be) == kPayload.size(),
            "header is not big-endian payload size");
    std::array<std::uint8_t, kPayload.size()> received{};
    nano_nccl::transport::socket::recv_all(sockets.second(), received.data(),
                                            received.size());
    require(received == kPayload, "payload differs from sent bytes");
    sender.join();
}

void zero_size_slice_advances_framing_order() {
    SocketPair sockets;
    constexpr std::array<std::uint8_t, 2> kPayload{0xab, 0xcd};

    std::thread sender([&] {
        nano_nccl::transport::socket::send_slice(sockets.first(), nullptr, 0);
        nano_nccl::transport::socket::send_slice(
            sockets.first(), kPayload.data(), kPayload.size());
    });

    std::array<std::uint8_t, 2> received{};
    const std::uint32_t zero = nano_nccl::transport::socket::recv_slice(
        sockets.second(), received.data(), received.size());
    require(zero == 0, "zero-sized slice was not received");
    const std::uint32_t size = nano_nccl::transport::socket::recv_slice(
        sockets.second(), received.data(), received.size());
    require(size == kPayload.size(), "second slice size is wrong");
    require(received == kPayload, "second slice payload is wrong");
    sender.join();
}

void rejects_oversized_header_before_payload_copy() {
    SocketPair sockets;
    nano_nccl::transport::socket::SocketSliceHeader header{
        htonl(3),
    };
    nano_nccl::transport::socket::send_all(sockets.first(), &header,
                                            sizeof(header));

    std::array<std::uint8_t, 2> buffer{0xee, 0xee};
    try {
        nano_nccl::transport::socket::recv_slice(sockets.second(), buffer.data(),
                                                  buffer.size());
        throw std::runtime_error("oversized header was accepted");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("payload exceeds capacity") !=
                    std::string::npos,
                "oversized header error is not contextual");
    }
    require(buffer[0] == 0xee && buffer[1] == 0xee,
            "oversized header copied payload bytes");
}

void closed_peer_reports_receive_error() {
    SocketPair sockets;
    sockets.close_first();
    try {
        std::uint8_t byte = 0;
        nano_nccl::transport::socket::recv_all(sockets.second(), &byte, 1);
        throw std::runtime_error("closed peer did not fail receive");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("peer closed") !=
                    std::string::npos,
                "closed peer error is not descriptive");
    }
}

bool wait_for(const std::function<bool()>& predicate) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

void proxies_copy_fifo_slice_and_publish_counters() {
    SocketPair sockets;
    std::array<std::uint8_t, 8> send_fifo{0x10, 0x20, 0x30};
    std::array<std::uint8_t, 8> recv_fifo{};
    std::uint32_t send_size = 3;
    std::uint32_t recv_size = 0;
    std::uint64_t send_head = 0;
    std::uint64_t send_tail = 1;
    std::uint64_t recv_head = 1;
    std::uint64_t recv_tail = 0;
    std::uint32_t abort = 0;
    auto errors = std::make_shared<
        nano_nccl::transport::socket::SocketAsyncErrorState>(&abort);
    const nano_nccl::transport::socket::SocketProxyIdentity identity{
        2, 3, 1};
    nano_nccl::transport::socket::SocketSendProxy sender(
        nano_nccl::transport::socket::SocketConnection(sockets.release_first()),
        {send_fifo.data(), send_fifo.size(), 1, &send_size},
        {&send_head, &send_tail}, identity, errors);
    nano_nccl::transport::socket::SocketRecvProxy receiver(
        nano_nccl::transport::socket::SocketConnection(sockets.release_second()),
        {recv_fifo.data(), recv_fifo.size(), 1, &recv_size},
        {&recv_head, &recv_tail}, identity, errors);

    receiver.start();
    sender.start();
    require(wait_for([&] { return __atomic_load_n(&send_head, __ATOMIC_ACQUIRE) == 1 &&
                                   __atomic_load_n(&recv_tail, __ATOMIC_ACQUIRE) == 1; }),
            "proxies did not publish completed slice");
    require(recv_size == send_size, "receiver did not publish slice size");
    require(std::memcmp(send_fifo.data(), recv_fifo.data(), send_size) == 0,
            "receiver FIFO does not contain sent slice");
    require(!errors->has_error(), "successful proxies recorded an error");
    sender.stop();
    receiver.stop();
    sender.join();
    receiver.join();
}

void closed_peer_records_contextual_proxy_error() {
    SocketPair sockets;
    std::array<std::uint8_t, 8> recv_fifo{};
    std::uint32_t recv_size = 0;
    std::uint64_t recv_head = 1;
    std::uint64_t recv_tail = 0;
    std::uint32_t abort = 0;
    auto errors = std::make_shared<
        nano_nccl::transport::socket::SocketAsyncErrorState>(&abort);
    nano_nccl::transport::socket::SocketRecvProxy receiver(
        nano_nccl::transport::socket::SocketConnection(sockets.release_second()),
        {recv_fifo.data(), recv_fifo.size(), 1, &recv_size},
        {&recv_head, &recv_tail}, {7, 8, 2}, errors);
    receiver.start();
    sockets.close_first();
    require(wait_for([&] { return errors->has_error(); }),
            "closed peer did not record proxy error");
    receiver.join();
    require(__atomic_load_n(&abort, __ATOMIC_ACQUIRE) == 1,
            "proxy error did not set device-visible abort");
    require(errors->message().find("source=7 destination=8 channel=2 step=0") !=
                std::string::npos,
            "proxy error omits edge context");
}

void protocol_error_shuts_down_peer_proxy() {
    SocketPair sockets;
    nano_nccl::transport::socket::SocketSliceHeader oversized_header{htonl(9)};
    nano_nccl::transport::socket::send_all(sockets.first(), &oversized_header,
                                            sizeof(oversized_header));

    std::array<std::uint8_t, 8> failing_fifo{};
    std::array<std::uint8_t, 8> peer_fifo{};
    std::uint32_t failing_size = 0;
    std::uint32_t peer_size = 0;
    std::uint64_t failing_head = 1;
    std::uint64_t failing_tail = 0;
    std::uint64_t peer_head = 1;
    std::uint64_t peer_tail = 0;
    std::uint32_t failing_abort = 0;
    std::uint32_t peer_abort = 0;
    auto failing_errors = std::make_shared<
        nano_nccl::transport::socket::SocketAsyncErrorState>(&failing_abort);
    auto peer_errors = std::make_shared<
        nano_nccl::transport::socket::SocketAsyncErrorState>(&peer_abort);
    nano_nccl::transport::socket::SocketRecvProxy failing_receiver(
        nano_nccl::transport::socket::SocketConnection(sockets.release_second()),
        {failing_fifo.data(), failing_fifo.size(), 1, &failing_size},
        {&failing_head, &failing_tail}, {3, 4, 0}, failing_errors);
    nano_nccl::transport::socket::SocketRecvProxy peer_receiver(
        nano_nccl::transport::socket::SocketConnection(sockets.release_first()),
        {peer_fifo.data(), peer_fifo.size(), 1, &peer_size},
        {&peer_head, &peer_tail}, {4, 3, 0}, peer_errors);

    peer_receiver.start();
    failing_receiver.start();
    require(wait_for([&] { return failing_errors->has_error(); }),
            "oversized receive did not record a proxy error");
    require(wait_for([&] { return peer_errors->has_error(); }),
            "protocol error did not unblock the peer proxy");
    failing_receiver.join();
    peer_receiver.join();
    require(__atomic_load_n(&failing_abort, __ATOMIC_ACQUIRE) == 1,
            "oversized receive did not set its abort flag");
    require(__atomic_load_n(&peer_abort, __ATOMIC_ACQUIRE) == 1,
            "peer proxy did not set its abort flag");
    require(failing_errors->message().find("payload exceeds capacity") !=
                std::string::npos,
            "oversized receive error is not preserved");
    require(peer_errors->message().find("peer closed") != std::string::npos,
            "peer proxy did not observe socket shutdown");
}

}  // namespace

int main() {
    try {
        sends_big_endian_header_and_payload();
        zero_size_slice_advances_framing_order();
        rejects_oversized_header_before_payload_copy();
        closed_peer_reports_receive_error();
        proxies_copy_fifo_slice_and_publish_counters();
        closed_peer_records_contextual_proxy_error();
        protocol_error_shuts_down_peer_proxy();
    } catch (const std::exception& error) {
        std::cerr << "socket protocol test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
