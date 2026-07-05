#pragma once

#include "core/numa.h"
#include "nano_nccl/types.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace nano_nccl::core {

// 所有 CUDA API 调用统一走这个宏；失败抛 std::runtime_error。
#define CUDA_CHECK_THROW(call)                                                \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            char msg__[512];                                                  \
            std::snprintf(msg__, sizeof(msg__), "CUDA error at %s:%d: %s",    \
                          __FILE__, __LINE__, cudaGetErrorString(err__));     \
            throw std::runtime_error(msg__);                                  \
        }                                                                     \
    } while (0)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(int device, std::size_t count) { reset(device, count); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void reset(int device, std::size_t count) {
        release();
        device_ = device;
        count_ = count;
        CUDA_CHECK_THROW(cudaSetDevice(device_));
        CUDA_CHECK_THROW(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    void release() {
        if (ptr_ != nullptr) {
            cudaSetDevice(device_);
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    T* get() const { return ptr_; }
    std::size_t count() const { return count_; }

private:
    int device_ = 0;
    std::size_t count_ = 0;
    T* ptr_ = nullptr;
};

template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(std::size_t count,
                          unsigned int flags = cudaHostAllocDefault) {
        reset(count, flags);
    }
    ~PinnedBuffer() { release(); }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void reset(std::size_t count, unsigned int flags = cudaHostAllocDefault) {
        release();
        count_ = count;
        flags_ = flags;
        CUDA_CHECK_THROW(cudaHostAlloc(&ptr_, count_ * sizeof(T), flags_));
    }

    void release() {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    T* get() const { return ptr_; }
    std::size_t count() const { return count_; }

private:
    std::size_t count_ = 0;
    unsigned int flags_ = cudaHostAllocDefault;
    T* ptr_ = nullptr;
};

// 跨 kRanks GPU 映射的 host-pinned buffer；GPU 经 PCIe 直接读写，无 proxy。
// 按 receiver NUMA 节点分配，避免跨 NUMA 带宽损失。
class MappedBuffer {
public:
    MappedBuffer() = default;
    explicit MappedBuffer(std::size_t count, int numa_node = -1);
    ~MappedBuffer() { release(); }

    MappedBuffer(const MappedBuffer&) = delete;
    MappedBuffer& operator=(const MappedBuffer&) = delete;

    void reset(std::size_t count, int numa_node = -1);
    void release();

    float* device_ptr(int dev) const { return device_ptrs_[dev]; }

private:
    std::size_t count_ = 0;
    float* host_ptr_ = nullptr;
    float* device_ptrs_[kRanks]{};
};

// 基于 /dev/shm + cudaHostRegister 的映射 buffer；mapped flag 路径使用。
class RegisteredMappedBuffer {
public:
    RegisteredMappedBuffer() = default;
    explicit RegisteredMappedBuffer(std::size_t count);
    ~RegisteredMappedBuffer() { release(); }

    RegisteredMappedBuffer(const RegisteredMappedBuffer&) = delete;
    RegisteredMappedBuffer& operator=(const RegisteredMappedBuffer&) = delete;

    void reset(std::size_t count);
    void release();

    float* device_ptr(int dev) const { return device_ptrs_[dev]; }
    float* host_ptr() const { return host_ptr_; }

private:
    static std::size_t round_up(std::size_t value, std::size_t alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    [[noreturn]] static void throw_errno(const char* call) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "%s failed: %s", call,
                      std::strerror(errno));
        throw std::runtime_error(msg);
    }

    std::size_t count_ = 0;
    std::size_t map_bytes_ = 0;
    int fd_ = -1;
    bool registered_ = false;
    float* host_ptr_ = nullptr;
    float* device_ptrs_[kRanks]{};
};

// 跨 kRanks GPU 映射的 u64 数组；step counter 用，NUMA-aware 分配。
class MappedU64Array {
public:
    MappedU64Array() = default;
    ~MappedU64Array() { release(); }

    MappedU64Array(const MappedU64Array&) = delete;
    MappedU64Array& operator=(const MappedU64Array&) = delete;

    void reset(int count, int numa_node = -1);
    void release();

    void clear_host();

    std::uint64_t* device_ptr(int dev) const { return device_ptrs_[dev]; }

private:
    int count_ = 0;
    std::uint64_t* host_ptr_ = nullptr;
    std::uint64_t* device_ptrs_[kRanks]{};
};

}  // namespace nano_nccl::core
