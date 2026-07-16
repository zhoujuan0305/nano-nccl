#include "core/buffer.h"
#include "nano_nccl/traits.h"

#include <algorithm>
#include <utility>

namespace nano_nccl::core {

template <typename T>
MappedBuffer<T>::MappedBuffer(std::size_t count, int numa_node,
                              std::vector<int> devices) {
    reset(count, numa_node, std::move(devices));
}

RegisteredMappedBuffer::RegisteredMappedBuffer(std::size_t count) {
    reset(count);
}

template <typename T>
void MappedBuffer<T>::reset(std::size_t count, int numa_node,
                            std::vector<int> devices) {
    release();
    if (devices.empty()) {
        throw std::runtime_error("mapped buffer requires local devices");
    }
    count_ = count;
    devices_ = std::move(devices);
    device_ptrs_.resize(devices_.size());
    // 先按 receiver NUMA 绑定分配页，再恢复默认策略，避免污染后续分配。
    if (numa_node >= 0 && numa_available_()) {
        numa_set_prefer(numa_node);
    }
    CUDA_CHECK_THROW(cudaHostAlloc(&host_ptr_, count_ * sizeof(T),
                                   cudaHostAllocMapped |
                                       cudaHostAllocPortable));
    if (numa_node >= 0 && numa_available_()) {
        numa_set_prefer(-1);
    }
    for (std::size_t local_rank = 0; local_rank < devices_.size(); ++local_rank) {
        CUDA_CHECK_THROW(cudaSetDevice(devices_[local_rank]));
        CUDA_CHECK_THROW(
            cudaHostGetDevicePointer(&device_ptrs_[local_rank], host_ptr_, 0));
    }
}

template <typename T>
void MappedBuffer<T>::release() {
    if (host_ptr_ != nullptr) {
        cudaFreeHost(host_ptr_);
        host_ptr_ = nullptr;
    }
    device_ptrs_.clear();
    devices_.clear();
    count_ = 0;
}

template <typename T>
T* MappedBuffer<T>::device_ptr(int device) const {
    auto it = std::find(devices_.begin(), devices_.end(), device);
    if (it == devices_.end()) {
        throw std::runtime_error("mapped buffer has no pointer for device");
    }
    return device_ptrs_[static_cast<std::size_t>(it - devices_.begin())];
}

template class MappedBuffer<float>;
template class MappedBuffer<__half>;
template class MappedBuffer<__nv_bfloat16>;
template class MappedBuffer<std::uint8_t>;

void RegisteredMappedBuffer::reset(std::size_t count) {
    release();
    count_ = count;
    std::size_t bytes = count_ * sizeof(float);
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
    }
    map_bytes_ = round_up(bytes, static_cast<std::size_t>(page_size));

    char path[] = "/dev/shm/nano-nccl-XXXXXX";
    fd_ = mkstemp(path);
    if (fd_ < 0) {
        throw_errno("mkstemp(/dev/shm/nano-nccl-XXXXXX)");
    }
    unlink(path);
    if (ftruncate(fd_, static_cast<off_t>(map_bytes_)) != 0) {
        throw_errno("ftruncate");
    }
    void* ptr =
        mmap(nullptr, map_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (ptr == MAP_FAILED) {
        host_ptr_ = nullptr;
        throw_errno("mmap");
    }
    host_ptr_ = static_cast<float*>(ptr);
    std::memset(host_ptr_, 0, map_bytes_);

    CUDA_CHECK_THROW(cudaSetDevice(0));
    CUDA_CHECK_THROW(cudaHostRegister(host_ptr_, map_bytes_,
                                      cudaHostRegisterPortable |
                                          cudaHostRegisterMapped));
    registered_ = true;
    for (int dev = 0; dev < kRanks; ++dev) {
        CUDA_CHECK_THROW(cudaSetDevice(dev));
        CUDA_CHECK_THROW(
            cudaHostGetDevicePointer(&device_ptrs_[dev], host_ptr_, 0));
    }
}

void RegisteredMappedBuffer::release() {
    if (registered_) {
        cudaHostUnregister(host_ptr_);
        registered_ = false;
    }
    if (host_ptr_ != nullptr) {
        munmap(host_ptr_, map_bytes_);
        host_ptr_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    for (int dev = 0; dev < kRanks; ++dev) {
        device_ptrs_[dev] = nullptr;
    }
    count_ = 0;
    map_bytes_ = 0;
}

void MappedU64Array::reset(int count, int numa_node, std::vector<int> devices) {
    release();
    if (devices.empty()) {
        throw std::runtime_error("mapped counter requires local devices");
    }
    count_ = count;
    devices_ = std::move(devices);
    device_ptrs_.resize(devices_.size());
    if (numa_node >= 0 && numa_available_()) {
        numa_set_prefer(numa_node);
    }
    CUDA_CHECK_THROW(cudaHostAlloc(&host_ptr_, count_ * sizeof(std::uint64_t),
                                   cudaHostAllocMapped |
                                       cudaHostAllocPortable));
    if (numa_node >= 0 && numa_available_()) {
        numa_set_prefer(-1);
    }
    clear_host();
    for (std::size_t local_rank = 0; local_rank < devices_.size(); ++local_rank) {
        CUDA_CHECK_THROW(cudaSetDevice(devices_[local_rank]));
        CUDA_CHECK_THROW(
            cudaHostGetDevicePointer(&device_ptrs_[local_rank], host_ptr_, 0));
    }
}

void MappedU64Array::release() {
    if (host_ptr_ != nullptr) {
        cudaFreeHost(host_ptr_);
        host_ptr_ = nullptr;
    }
    device_ptrs_.clear();
    devices_.clear();
    count_ = 0;
}

void MappedU64Array::clear_host() {
    if (host_ptr_ == nullptr) {
        return;
    }
    for (int i = 0; i < count_; ++i) {
        host_ptr_[i] = 0;
    }
}

std::uint64_t* MappedU64Array::device_ptr(int device) const {
    auto it = std::find(devices_.begin(), devices_.end(), device);
    if (it == devices_.end()) {
        throw std::runtime_error("mapped counter has no pointer for device");
    }
    return device_ptrs_[static_cast<std::size_t>(it - devices_.begin())];
}

void MappedU32Array::reset(int count, int numa_node, std::vector<int> devices) {
    release();
    if (devices.empty()) {
        throw std::runtime_error("mapped counter requires local devices");
    }
    count_ = count;
    devices_ = std::move(devices);
    device_ptrs_.resize(devices_.size());
    if (numa_node >= 0 && numa_available_()) {
        numa_set_prefer(numa_node);
    }
    CUDA_CHECK_THROW(cudaHostAlloc(&host_ptr_, count_ * sizeof(std::uint32_t),
                                   cudaHostAllocMapped |
                                       cudaHostAllocPortable));
    if (numa_node >= 0 && numa_available_()) {
        numa_set_prefer(-1);
    }
    clear_host();
    for (std::size_t local_rank = 0; local_rank < devices_.size(); ++local_rank) {
        CUDA_CHECK_THROW(cudaSetDevice(devices_[local_rank]));
        CUDA_CHECK_THROW(
            cudaHostGetDevicePointer(&device_ptrs_[local_rank], host_ptr_, 0));
    }
}

void MappedU32Array::release() {
    if (host_ptr_ != nullptr) {
        cudaFreeHost(host_ptr_);
        host_ptr_ = nullptr;
    }
    device_ptrs_.clear();
    devices_.clear();
    count_ = 0;
}

void MappedU32Array::clear_host() {
    if (host_ptr_ == nullptr) {
        return;
    }
    for (int i = 0; i < count_; ++i) {
        host_ptr_[i] = 0;
    }
}

std::uint32_t* MappedU32Array::device_ptr(int device) const {
    auto it = std::find(devices_.begin(), devices_.end(), device);
    if (it == devices_.end()) {
        throw std::runtime_error("mapped counter has no pointer for device");
    }
    return device_ptrs_[static_cast<std::size_t>(it - devices_.begin())];
}

}  // namespace nano_nccl::core
