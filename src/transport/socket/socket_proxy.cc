#include "transport/socket/socket_proxy.h"
#include "transport/socket/socket_protocol.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

namespace nano_nccl::transport::socket {

namespace {

std::runtime_error socket_error(const char* operation) {
    return std::runtime_error(std::string(operation) + ": " +
                              std::strerror(errno));
}

void send_iov_all(int fd, const void* first, std::size_t first_bytes,
                  const void* second, std::size_t second_bytes) {
    iovec iov[2]{};
    int iov_count = 0;
    if (first_bytes != 0) {
        iov[iov_count++] = iovec{const_cast<void*>(first), first_bytes};
    }
    if (second_bytes != 0) {
        iov[iov_count++] = iovec{const_cast<void*>(second), second_bytes};
    }

    while (iov_count != 0) {
        msghdr message{};
        message.msg_iov = iov;
        message.msg_iovlen = static_cast<std::size_t>(iov_count);
        const ssize_t written = sendmsg(fd, &message, MSG_NOSIGNAL);
        if (written < 0) {
            if (errno == EINTR) continue;
            throw socket_error("sendmsg");
        }
        if (written == 0) {
            throw std::runtime_error("sendmsg returned zero bytes");
        }

        std::size_t consumed = static_cast<std::size_t>(written);
        while (iov_count != 0 && consumed >= iov[0].iov_len) {
            consumed -= iov[0].iov_len;
            if (iov_count > 1) iov[0] = iov[1];
            --iov_count;
        }
        if (iov_count != 0 && consumed != 0) {
            iov[0].iov_base = static_cast<char*>(iov[0].iov_base) + consumed;
            iov[0].iov_len -= consumed;
        }
    }
}

void validate_fifo(const SocketProxyFifo& fifo) {
    if (fifo.data == nullptr || fifo.slot_sizes == nullptr ||
        fifo.slot_bytes == 0 || fifo.slot_count == 0 ||
        fifo.step_increment == 0 ||
        fifo.slot_count > std::numeric_limits<std::size_t>::max() /
                              fifo.slot_bytes) {
        throw std::runtime_error("socket proxy has invalid FIFO storage");
    }
}

std::uint64_t load_counter(const std::uint64_t* counter) {
    return __atomic_load_n(counter, __ATOMIC_ACQUIRE);
}

void store_counter(std::uint64_t* counter, std::uint64_t value) {
    __atomic_store_n(counter, value, __ATOMIC_RELEASE);
}

std::uint32_t load_size(const std::uint32_t* size) {
    return __atomic_load_n(size, __ATOMIC_ACQUIRE);
}

void store_size(std::uint32_t* size, std::uint32_t value) {
    __atomic_store_n(size, value, __ATOMIC_RELEASE);
}

std::string format_failure(SocketProxyIdentity identity, std::uint64_t step,
                           const std::string& reason) {
    std::ostringstream stream;
    stream << "socket proxy source=" << identity.source_rank
           << " destination=" << identity.destination_rank
           << " channel=" << identity.channel << " step=" << step << ": "
           << reason;
    return stream.str();
}

#if defined(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION)
std::uint64_t socket_fault_after_slices() {
    const char* value = std::getenv("NANO_NCCL_SOCKET_FAIL_AFTER_SLICES");
    if (value == nullptr || value[0] == '\0') return 0;

    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0) return 0;
    return static_cast<std::uint64_t>(parsed);
}
#endif

}  // namespace

void set_tcp_nodelay(int fd) {
    const int enabled = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) != 0) {
        if (errno == ENOPROTOOPT || errno == EOPNOTSUPP) return;
        throw socket_error("setsockopt(TCP_NODELAY)");
    }
}

void send_all(int fd, const void* data, std::size_t bytes) {
    if (bytes != 0 && data == nullptr) {
        throw std::runtime_error("send_all payload is null");
    }
    send_iov_all(fd, data, bytes, nullptr, 0);
}

void recv_all(int fd, void* data, std::size_t bytes) {
    if (bytes != 0 && data == nullptr) {
        throw std::runtime_error("recv_all payload is null");
    }
    auto* cursor = static_cast<char*>(data);
    while (bytes != 0) {
        const ssize_t received = recv(fd, cursor, bytes, 0);
        if (received < 0) {
            if (errno == EINTR) continue;
            throw socket_error("recv");
        }
        if (received == 0) {
            throw std::runtime_error("recv: peer closed");
        }
        cursor += received;
        bytes -= static_cast<std::size_t>(received);
    }
}

void send_slice(int fd, const void* payload, std::uint32_t payload_bytes) {
    if (payload_bytes != 0 && payload == nullptr) {
        throw std::runtime_error("send_slice payload is null");
    }
    const SocketSliceHeader header{htonl(payload_bytes)};
    send_iov_all(fd, &header, sizeof(header), payload, payload_bytes);
}

std::uint32_t recv_slice(int fd, void* payload, std::size_t payload_capacity) {
    SocketSliceHeader header{};
    recv_all(fd, &header, sizeof(header));
    const std::uint32_t payload_bytes = ntohl(header.payload_bytes_be);
    if (payload_bytes > payload_capacity) {
        throw std::runtime_error("socket slice payload exceeds capacity");
    }
    if (payload_bytes != 0 && payload == nullptr) {
        throw std::runtime_error("recv_slice payload is null");
    }
    recv_all(fd, payload, payload_bytes);
    return payload_bytes;
}

void SocketAsyncErrorState::record_failure(SocketProxyIdentity identity,
                                           std::uint64_t step,
                                           const std::string& reason) noexcept {
    bool expected = false;
    if (!has_error_.compare_exchange_strong(expected, true,
                                            std::memory_order_acq_rel)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        message_ = format_failure(identity, step, reason);
    }
    if (device_abort_ != nullptr) {
        __atomic_store_n(device_abort_, 1U, __ATOMIC_RELEASE);
    }
}

std::string SocketAsyncErrorState::message() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return message_;
}

SocketSendProxy::SocketSendProxy(
    SocketConnection connection, SocketProxyFifo fifo, SocketSendControl control,
    SocketProxyIdentity identity, std::shared_ptr<SocketAsyncErrorState> errors)
    : connection_(std::move(connection)), fifo_(fifo), control_(control),
      identity_(identity), errors_(std::move(errors)) {
    validate_fifo(fifo_);
    if (connection_.fd() < 0 || control_.send_head == nullptr ||
        control_.send_tail == nullptr || errors_ == nullptr) {
        throw std::runtime_error("socket send proxy has invalid state");
    }
    step_.store(load_counter(control_.send_head), std::memory_order_relaxed);
}

SocketSendProxy::~SocketSendProxy() {
    stop();
    join();
}

void SocketSendProxy::start() {
    if (thread_.joinable()) {
        throw std::runtime_error("socket send proxy already started");
    }
    thread_ = std::thread(&SocketSendProxy::run, this);
}

void SocketSendProxy::stop() noexcept {
    if (stop_requested_.exchange(true, std::memory_order_acq_rel)) return;
    errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                            "stop requested");
    if (connection_.fd() >= 0) ::shutdown(connection_.fd(), SHUT_RDWR);
}

void SocketSendProxy::shutdown() noexcept {
    if (stop_requested_.exchange(true, std::memory_order_acq_rel)) return;
    if (connection_.fd() >= 0) ::shutdown(connection_.fd(), SHUT_RDWR);
}

void SocketSendProxy::drain() const {
    while (!errors_->has_error() &&
           load_counter(control_.send_head) < load_counter(control_.send_tail)) {
        std::this_thread::yield();
    }
}

void SocketSendProxy::join() noexcept {
    if (thread_.joinable()) thread_.join();
}

void SocketSendProxy::run() noexcept {
    try {
        set_tcp_nodelay(connection_.fd());
#if defined(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION)
        const std::uint64_t fault_after_slices = socket_fault_after_slices();
        std::uint64_t successful_slices = 0;
#endif
        while (!stop_requested_.load(std::memory_order_acquire) &&
               !errors_->has_error()) {
            const std::uint64_t step = step_.load(std::memory_order_relaxed);
            if (load_counter(control_.send_tail) <= step) {
                std::this_thread::yield();
                continue;
            }
            const std::size_t slot = step % fifo_.slot_count;
            const std::uint32_t payload_bytes = load_size(fifo_.slot_sizes + slot);
            if (payload_bytes > fifo_.slot_bytes) {
                throw std::runtime_error("socket proxy send payload exceeds FIFO capacity");
            }
            send_slice(connection_.fd(), fifo_.data + slot * fifo_.slot_bytes,
                       payload_bytes);
            store_counter(control_.send_head, step + fifo_.step_increment);
            step_.store(step + fifo_.step_increment, std::memory_order_release);
#if defined(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION)
            ++successful_slices;
            if (fault_after_slices != 0 && successful_slices == fault_after_slices) {
                connection_.close();
                throw std::runtime_error(
                    "socket fault injection closed connection after successful slices");
            }
#endif
        }
    } catch (const std::exception& error) {
        if (!stop_requested_.load(std::memory_order_acquire)) {
            errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                                    error.what());
        }
        if (connection_.fd() >= 0) ::shutdown(connection_.fd(), SHUT_RDWR);
    }
}

SocketRecvProxy::SocketRecvProxy(
    SocketConnection connection, SocketProxyFifo fifo, SocketRecvControl control,
    SocketProxyIdentity identity, std::shared_ptr<SocketAsyncErrorState> errors)
    : connection_(std::move(connection)), fifo_(fifo), control_(control),
      identity_(identity), errors_(std::move(errors)) {
    validate_fifo(fifo_);
    if (connection_.fd() < 0 || control_.recv_head == nullptr ||
        control_.recv_tail == nullptr || errors_ == nullptr) {
        throw std::runtime_error("socket recv proxy has invalid state");
    }
    step_.store(load_counter(control_.recv_tail), std::memory_order_relaxed);
}

SocketRecvProxy::~SocketRecvProxy() {
    stop();
    join();
}

void SocketRecvProxy::start() {
    if (thread_.joinable()) {
        throw std::runtime_error("socket recv proxy already started");
    }
    thread_ = std::thread(&SocketRecvProxy::run, this);
}

void SocketRecvProxy::stop() noexcept {
    if (stop_requested_.exchange(true, std::memory_order_acq_rel)) return;
    errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                            "stop requested");
    if (connection_.fd() >= 0) ::shutdown(connection_.fd(), SHUT_RDWR);
}

void SocketRecvProxy::shutdown() noexcept {
    if (stop_requested_.exchange(true, std::memory_order_acq_rel)) return;
    if (connection_.fd() >= 0) ::shutdown(connection_.fd(), SHUT_RDWR);
}

void SocketRecvProxy::drain() const {
    while (!errors_->has_error() &&
           load_counter(control_.recv_tail) < load_counter(control_.recv_head)) {
        std::this_thread::yield();
    }
}

void SocketRecvProxy::join() noexcept {
    if (thread_.joinable()) thread_.join();
}

void SocketRecvProxy::run() noexcept {
    try {
        set_tcp_nodelay(connection_.fd());
        while (!stop_requested_.load(std::memory_order_acquire) &&
               !errors_->has_error()) {
            const std::uint64_t step = step_.load(std::memory_order_relaxed);
            if (load_counter(control_.recv_head) + fifo_.slot_count <
                step + fifo_.step_increment) {
                std::this_thread::yield();
                continue;
            }
            const std::size_t slot = step % fifo_.slot_count;
            const std::uint32_t payload_bytes = recv_slice(
                connection_.fd(), fifo_.data + slot * fifo_.slot_bytes,
                fifo_.slot_bytes);
            store_size(fifo_.slot_sizes + slot, payload_bytes);
            store_counter(control_.recv_tail, step + fifo_.step_increment);
            step_.store(step + fifo_.step_increment, std::memory_order_release);
        }
    } catch (const std::exception& error) {
        if (!stop_requested_.load(std::memory_order_acquire)) {
            errors_->record_failure(identity_, step_.load(std::memory_order_acquire),
                                    error.what());
        }
        if (connection_.fd() >= 0) ::shutdown(connection_.fd(), SHUT_RDWR);
    }
}

}  // namespace nano_nccl::transport::socket
