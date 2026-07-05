#pragma once

#include "core/buffer.h"

#include <cstddef>

#include <cuda_runtime.h>

namespace nano_nccl::core {

class Stream {
public:
    explicit Stream(int device) : device_(device) {
        CUDA_CHECK_THROW(cudaSetDevice(device_));
        CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~Stream() {
        if (stream_ != nullptr) {
            cudaSetDevice(device_);
            cudaStreamDestroy(stream_);
        }
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    cudaStream_t get() const { return stream_; }

private:
    int device_ = 0;
    cudaStream_t stream_ = nullptr;
};

class Event {
public:
    explicit Event(int device) : device_(device) {
        CUDA_CHECK_THROW(cudaSetDevice(device_));
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }

    ~Event() {
        if (event_ != nullptr) {
            cudaSetDevice(device_);
            cudaEventDestroy(event_);
        }
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    int device_ = 0;
    cudaEvent_t event_ = nullptr;
};

class GraphExec {
public:
    GraphExec() = default;
    ~GraphExec() { release(); }

    GraphExec(const GraphExec&) = delete;
    GraphExec& operator=(const GraphExec&) = delete;

    void release() {
        if (exec_ != nullptr) {
            cudaGraphExecDestroy(exec_);
            exec_ = nullptr;
        }
        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        count_ = 0;
    }

    bool valid_for(std::size_t count) const {
        return exec_ != nullptr && count_ == count;
    }

    cudaGraph_t* graph_ptr() { return &graph_; }
    cudaGraph_t graph() const { return graph_; }
    cudaGraphExec_t* exec_ptr() { return &exec_; }
    cudaGraphExec_t exec() const { return exec_; }
    void set_count(std::size_t count) { count_ = count; }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    std::size_t count_ = 0;
};

class BatchGraphExec {
public:
    BatchGraphExec() = default;
    ~BatchGraphExec() { release(); }

    BatchGraphExec(const BatchGraphExec&) = delete;
    BatchGraphExec& operator=(const BatchGraphExec&) = delete;

    void release() {
        if (exec_ != nullptr) {
            cudaGraphExecDestroy(exec_);
            exec_ = nullptr;
        }
        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        count_ = 0;
        iters_ = 0;
    }

    bool valid_for(std::size_t count, int iters) const {
        return exec_ != nullptr && count_ == count && iters_ == iters;
    }

    cudaGraph_t* graph_ptr() { return &graph_; }
    cudaGraphExec_t* exec_ptr() { return &exec_; }
    cudaGraphExec_t exec() const { return exec_; }
    void set(std::size_t count, int iters) {
        count_ = count;
        iters_ = iters;
    }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    std::size_t count_ = 0;
    int iters_ = 0;
};

}  // namespace nano_nccl::core
