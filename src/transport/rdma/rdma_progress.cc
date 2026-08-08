#include "transport/rdma/rdma_progress.h"
#include "core/numa.h"

#include <immintrin.h>
#include <stdexcept>

namespace nano_nccl::transport::rdma {

RdmaProgressEngine::~RdmaProgressEngine() {
    shutdown();
    join();
}

void RdmaProgressEngine::add_send(RdmaSendProxy* proxy) {
    if (proxy == nullptr) {
        throw std::runtime_error("rdma progress engine add_send null");
    }
    if (thread_.joinable()) {
        throw std::runtime_error(
            "rdma progress engine cannot add_send after start");
    }
    if (pin_numa_node_ < 0) {
        pin_numa_node_ = proxy->fifo_numa_node();
    }
    sends_.push_back(proxy);
}

void RdmaProgressEngine::add_recv(RdmaRecvProxy* proxy) {
    if (proxy == nullptr) {
        throw std::runtime_error("rdma progress engine add_recv null");
    }
    if (thread_.joinable()) {
        throw std::runtime_error(
            "rdma progress engine cannot add_recv after start");
    }
    if (pin_numa_node_ < 0) {
        pin_numa_node_ = proxy->fifo_numa_node();
    }
    recvs_.push_back(proxy);
}

void RdmaProgressEngine::start() {
    if (thread_.joinable()) {
        throw std::runtime_error("rdma progress engine already started");
    }
    if (sends_.empty() && recvs_.empty()) {
        return;
    }
    stop_.store(false, std::memory_order_release);
    thread_ = std::thread(&RdmaProgressEngine::run, this);
}

void RdmaProgressEngine::shutdown() noexcept {
    stop_.store(true, std::memory_order_release);
}

void RdmaProgressEngine::join() noexcept {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void RdmaProgressEngine::drain() const {
    for (RdmaSendProxy* proxy : sends_) {
        if (proxy != nullptr) {
            proxy->drain();
        }
    }
}

void RdmaProgressEngine::run() noexcept {
    if (pin_numa_node_ >= 0) {
        core::pin_current_thread_to_numa_node(pin_numa_node_);
    }
    while (!stop_.load(std::memory_order_acquire)) {
        bool work = false;
        bool any_error = false;
        for (RdmaRecvProxy* proxy : recvs_) {
            if (proxy == nullptr) {
                continue;
            }
            if (proxy->has_error()) {
                any_error = true;
                continue;
            }
            work = proxy->progress() || work;
            if (proxy->has_error()) {
                any_error = true;
            }
        }
        for (RdmaSendProxy* proxy : sends_) {
            if (proxy == nullptr) {
                continue;
            }
            if (proxy->has_error()) {
                any_error = true;
                continue;
            }
            work = proxy->progress() || work;
            if (proxy->has_error()) {
                any_error = true;
            }
        }
        if (any_error) {
            break;
        }
        if (!work) {
            _mm_pause();
        }
    }
}

}  // namespace nano_nccl::transport::rdma
