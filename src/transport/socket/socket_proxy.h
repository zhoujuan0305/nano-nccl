#pragma once

#include "transport/socket/socket_endpoint.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace nano_nccl::transport::socket {

struct SocketProxyIdentity {
    int source_rank = -1;
    int destination_rank = -1;
    int channel = -1;
};

struct SocketProxyFifo {
    std::uint8_t* data = nullptr;
    std::size_t slot_bytes = 0;
    std::size_t slot_count = 0;
    std::uint32_t* slot_sizes = nullptr;
    std::size_t step_increment = 1;
};

struct SocketSendControl {
    std::uint64_t* send_head = nullptr;
    std::uint64_t* send_tail = nullptr;
};

struct SocketRecvControl {
    std::uint64_t* recv_head = nullptr;
    std::uint64_t* recv_tail = nullptr;
};

class SocketAsyncErrorState {
public:
    explicit SocketAsyncErrorState(std::uint32_t* device_abort = nullptr)
        : device_abort_(device_abort) {}

    SocketAsyncErrorState(const SocketAsyncErrorState&) = delete;
    SocketAsyncErrorState& operator=(const SocketAsyncErrorState&) = delete;

    void record_failure(SocketProxyIdentity identity, std::uint64_t step,
                        const std::string& reason) noexcept;
    bool has_error() const noexcept {
        return has_error_.load(std::memory_order_acquire);
    }
    std::string message() const;

private:
    std::atomic<bool> has_error_{false};
    std::uint32_t* device_abort_ = nullptr;
    mutable std::mutex mutex_;
    std::string message_;
};

class SocketSendProxy {
public:
    SocketSendProxy(SocketConnection connection, SocketProxyFifo fifo,
                    SocketSendControl control, SocketProxyIdentity identity,
                    std::shared_ptr<SocketAsyncErrorState> errors);
    ~SocketSendProxy();

    SocketSendProxy(const SocketSendProxy&) = delete;
    SocketSendProxy& operator=(const SocketSendProxy&) = delete;

    void start();
    void stop() noexcept;
    void shutdown() noexcept;
    void drain() const;
    void join() noexcept;

private:
    void run() noexcept;

    SocketConnection connection_;
    SocketProxyFifo fifo_;
    SocketSendControl control_;
    SocketProxyIdentity identity_;
    std::shared_ptr<SocketAsyncErrorState> errors_;
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
    std::atomic<std::uint64_t> step_{0};
};

class SocketRecvProxy {
public:
    SocketRecvProxy(SocketConnection connection, SocketProxyFifo fifo,
                    SocketRecvControl control, SocketProxyIdentity identity,
                    std::shared_ptr<SocketAsyncErrorState> errors);
    ~SocketRecvProxy();

    SocketRecvProxy(const SocketRecvProxy&) = delete;
    SocketRecvProxy& operator=(const SocketRecvProxy&) = delete;

    void start();
    void stop() noexcept;
    void shutdown() noexcept;
    void drain() const;
    void join() noexcept;

private:
    void run() noexcept;

    SocketConnection connection_;
    SocketProxyFifo fifo_;
    SocketRecvControl control_;
    SocketProxyIdentity identity_;
    std::shared_ptr<SocketAsyncErrorState> errors_;
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
    std::atomic<std::uint64_t> step_{0};
};

}  // namespace nano_nccl::transport::socket
