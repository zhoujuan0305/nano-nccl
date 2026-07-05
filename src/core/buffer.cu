#include "core/buffer.h"

namespace nano_nccl::core {

MappedBuffer::MappedBuffer(std::size_t count, int numa_node) {
    reset(count, numa_node);
}

RegisteredMappedBuffer::RegisteredMappedBuffer(std::size_t count) {
    reset(count);
}

void MappedBuffer::reset(std::size_t count, int numa_node) {
    release();
    count_ = count;
    // 先按 receiver NUMA 绑定分配页，再恢复默认策略，避免污染后续分配。
    if (numa_node >= 0 && numa_available_()) {
        numa_set_prefer(numa_node);
    }
    CUDA_CHECK_THROW(cudaHostAlloc(&host_ptr_, count_ * sizeof(float),
                                   cudaHostAllocMapped |
                                       cudaHostAllocPortable));
    if (numa_node >= 0 && numa_available_()) {
        numa_set_prefer(-1);
    }
    for (int dev = 0; dev < kRanks; ++dev) {
        CUDA_CHECK_THROW(cudaSetDevice(dev));
        CUDA_CHECK_THROW(
            cudaHostGetDevicePointer(&device_ptrs_[dev], host_ptr_, 0));
    }
}

void MappedBuffer::release() {
    if (host_ptr_ != nullptr) {
        cudaFreeHost(host_ptr_);
        host_ptr_ = nullptr;
    }
    for (int dev = 0; dev < kRanks; ++dev) {
        device_ptrs_[dev] = nullptr;
    }
    count_ = 0;
}

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

void MappedU64Array::reset(int count, int numa_node) {
    release();
    count_ = count;
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
    for (int dev = 0; dev < kRanks; ++dev) {
        CUDA_CHECK_THROW(cudaSetDevice(dev));
        CUDA_CHECK_THROW(
            cudaHostGetDevicePointer(&device_ptrs_[dev], host_ptr_, 0));
    }
}

void MappedU64Array::release() {
    if (host_ptr_ != nullptr) {
        cudaFreeHost(host_ptr_);
        host_ptr_ = nullptr;
    }
    for (int dev = 0; dev < kRanks; ++dev) {
        device_ptrs_[dev] = nullptr;
    }
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

}  // namespace nano_nccl::core
