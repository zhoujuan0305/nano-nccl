#include "transport/rdma/rdma_gdr.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <infiniband/verbs.h>
#include <unistd.h>

namespace nano_nccl::transport::rdma {

namespace {

std::runtime_error gdr_error(const char* op) {
    return std::runtime_error(std::string("rdma gdr ") + op + " failed: " +
                              std::strerror(errno));
}

int access_flags() {
    return IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
           IBV_ACCESS_REMOTE_READ;
}

ibv_mr* try_reg_dmabuf(ibv_pd* pd, void* device_addr, std::size_t bytes) {
    CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(device_addr);
    const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    const CUdeviceptr aligned = ptr & ~(static_cast<CUdeviceptr>(page) - 1);
    const std::size_t offset = static_cast<std::size_t>(ptr - aligned);
    const std::size_t aligned_bytes = (offset + bytes + page - 1) & ~(page - 1);

    int fd = -1;
    CUresult cu = cuMemGetHandleForAddressRange(
        &fd, aligned, aligned_bytes, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (cu != CUDA_SUCCESS || fd < 0) return nullptr;

    ibv_mr* mr = ibv_reg_dmabuf_mr(pd, static_cast<uint64_t>(offset), bytes,
                                   reinterpret_cast<uint64_t>(device_addr), fd,
                                   access_flags());
    const int saved = errno;
    close(fd);
    if (mr == nullptr) errno = saved;
    return mr;
}

}  // namespace

RdmaMemoryPlacement parse_rdma_memory_placement_env() {
    const char* env = std::getenv("NANO_NCCL_RDMA_GDR");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0 ||
        std::strcmp(env, "false") == 0 || std::strcmp(env, "off") == 0) {
        return RdmaMemoryPlacement::HostPin;
    }
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
        std::strcmp(env, "on") == 0) {
        return RdmaMemoryPlacement::GpuDirect;
    }
    throw std::runtime_error("NANO_NCCL_RDMA_GDR must be 0 or 1");
}

RdmaRegisteredMemory::RdmaRegisteredMemory(ibv_mr* mr, void* addr,
                                           std::size_t bytes,
                                           bool device) noexcept
    : mr_(mr), addr_(addr), bytes_(bytes), device_(device) {}

RdmaRegisteredMemory::~RdmaRegisteredMemory() { reset(); }

RdmaRegisteredMemory::RdmaRegisteredMemory(RdmaRegisteredMemory&& other) noexcept
    : mr_(other.mr_), addr_(other.addr_), bytes_(other.bytes_),
      device_(other.device_) {
    other.mr_ = nullptr;
    other.addr_ = nullptr;
    other.bytes_ = 0;
    other.device_ = false;
}

RdmaRegisteredMemory& RdmaRegisteredMemory::operator=(
    RdmaRegisteredMemory&& other) noexcept {
    if (this == &other) return *this;
    reset();
    mr_ = other.mr_;
    addr_ = other.addr_;
    bytes_ = other.bytes_;
    device_ = other.device_;
    other.mr_ = nullptr;
    other.addr_ = nullptr;
    other.bytes_ = 0;
    other.device_ = false;
    return *this;
}

void RdmaRegisteredMemory::reset() noexcept {
    if (mr_ != nullptr) {
        ibv_dereg_mr(mr_);
        mr_ = nullptr;
    }
    addr_ = nullptr;
    bytes_ = 0;
    device_ = false;
}

std::uint32_t RdmaRegisteredMemory::lkey() const noexcept {
    return mr_ == nullptr ? 0 : mr_->lkey;
}

std::uint32_t RdmaRegisteredMemory::rkey() const noexcept {
    return mr_ == nullptr ? 0 : mr_->rkey;
}

RdmaRegisteredMemory RdmaRegisteredMemory::register_host(ibv_pd* pd, void* addr,
                                                         std::size_t bytes) {
    if (pd == nullptr || addr == nullptr || bytes == 0) {
        throw std::runtime_error("rdma gdr register_host: invalid args");
    }
    ibv_mr* mr = ibv_reg_mr(pd, addr, bytes, access_flags());
    if (mr == nullptr) throw gdr_error("ibv_reg_mr host");
    return RdmaRegisteredMemory(mr, addr, bytes, false);
}

RdmaRegisteredMemory RdmaRegisteredMemory::register_device(
    ibv_pd* pd, void* device_addr, std::size_t bytes) {
    if (pd == nullptr || device_addr == nullptr || bytes == 0) {
        throw std::runtime_error("rdma gdr register_device: invalid args");
    }

    ibv_mr* mr = try_reg_dmabuf(pd, device_addr, bytes);
    if (mr == nullptr) mr = ibv_reg_mr(pd, device_addr, bytes, access_flags());
    if (mr == nullptr) {
        throw std::runtime_error(
            std::string("rdma gdr register_device unavailable: ") +
            std::strerror(errno));
    }
    return RdmaRegisteredMemory(mr, device_addr, bytes, true);
}

}  // namespace nano_nccl::transport::rdma
