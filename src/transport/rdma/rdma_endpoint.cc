#include "transport/rdma/rdma_endpoint.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <infiniband/verbs.h>
#include <net/if.h>

namespace nano_nccl::transport::rdma {

namespace {

// 用 errno 当前值组装 "rdma endpoint <op> failed: <strerror>"。
std::runtime_error endpoint_error(const char* op) {
    return std::runtime_error(std::string("rdma endpoint ") + op + " failed: " +
                              std::strerror(errno));
}

// 解析 NANO_NCCL_RDMA_GID_INDEX：unsigned long base10，落在 [0, 0xffff]。
std::uint16_t parse_gid_index_env() {
    const char* env = std::getenv("NANO_NCCL_RDMA_GID_INDEX");
    if (env == nullptr) return 0;
    errno = 0;
    char* end = nullptr;
    unsigned long v = std::strtoul(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || v > 0xffff) {
        throw std::runtime_error("NANO_NCCL_RDMA_GID_INDEX out of range");
    }
    return static_cast<std::uint16_t>(v);
}

// /sys/class/infiniband/<hca>/device/net/<ifname> 存在则视为该 HCA 绑定该接口
// (Mellanox OFED 约定)。
bool sysfs_iface_match(const std::string& hca_name, const std::string& ifname) {
    return std::filesystem::exists("/sys/class/infiniband/" + hca_name +
                                   "/device/net/" + ifname);
}

// 退路：RoCEv2 GID raw[8..15] 即 IPv6 地址，遍历 getifaddrs 找同名接口比对。
bool gid_iface_match(const union ibv_gid& gid, const std::string& ifname) {
    ifaddrs* interfaces = nullptr;
    if (getifaddrs(&interfaces) != 0) return false;
    bool matched = false;
    for (ifaddrs* entry = interfaces; entry != nullptr; entry = entry->ifa_next) {
        if (entry->ifa_name == nullptr || entry->ifa_addr == nullptr) continue;
        if (ifname != entry->ifa_name) continue;
        if (entry->ifa_addr->sa_family != AF_INET6) continue;
        const auto* sin6 = reinterpret_cast<const sockaddr_in6*>(entry->ifa_addr);
        const unsigned char* addr_bytes =
            reinterpret_cast<const unsigned char*>(sin6->sin6_addr.s6_addr);
        if (std::memcmp(addr_bytes, gid.raw, 16) == 0) {
            matched = true;
            break;
        }
    }
    freeifaddrs(interfaces);
    return matched;
}

// 由当前设备 + 接口名 + gid_index 判定该 HCA 是否为期望接口。
// ifname 空指针或空串表示“任意活动 HCA”，首个即匹配。
bool iface_matches_hca(ibv_context* ctx, ibv_device* dev, const char* ifname,
                       std::uint16_t gid_index) {
    if (ifname == nullptr || ifname[0] == '\0') return true;
    const char* hca_name = ibv_get_device_name(dev);
    if (hca_name != nullptr &&
        sysfs_iface_match(hca_name, ifname)) {
        return true;
    }
    union ibv_gid gid{};
    if (ibv_query_gid(ctx, 1, static_cast<int>(gid_index), &gid) != 0) return false;
    return gid_iface_match(gid, ifname);
}

// RAII 关闭 ibv_open_device 打开过的 context，方便列举匹配循环早退。
struct ContextGuard {
    ibv_context* ctx = nullptr;
    ~ContextGuard() { if (ctx != nullptr) ibv_close_device(ctx); }
    ibv_context* release() { ibv_context* p = ctx; ctx = nullptr; return p; }
};

// RAII 释放设备列表。
struct DeviceListGuard {
    ibv_device** list = nullptr;
    ~DeviceListGuard() { if (list != nullptr) ibv_free_device_list(list); }
};

// RAII 释放 PD（匹配成功、后续步骤失败时回滚）。
struct PdGuard {
    ibv_pd* pd = nullptr;
    ~PdGuard() { if (pd != nullptr) ibv_dealloc_pd(pd); }
    ibv_pd* release() { ibv_pd* p = pd; pd = nullptr; return p; }
};

// ibv_create_cq 返回的 unique_ptr deleter。
void destroy_cq(ibv_cq* cq) {
    if (cq != nullptr) ibv_destroy_cq(cq);
}

}  // namespace

RdmaEndpoint::RdmaEndpoint(ibv_context* ctx, ibv_pd* pd, std::uint16_t port_lid,
                           std::uint16_t gid_index, const std::uint8_t gid[16]) noexcept
    : context_(ctx), pd_(pd), port_lid_(port_lid), gid_index_(gid_index) {
    std::memcpy(gid_, gid, 16);
}

RdmaEndpoint::~RdmaEndpoint() { close(); }

RdmaEndpoint::RdmaEndpoint(RdmaEndpoint&& other) noexcept
    : context_(other.context_), pd_(other.pd_),
      port_lid_(other.port_lid_), gid_index_(other.gid_index_) {
    std::memcpy(gid_, other.gid_, 16);
    other.context_ = nullptr;
    other.pd_ = nullptr;
    other.port_lid_ = 0;
    other.gid_index_ = 0;
    std::memset(other.gid_, 0, 16);
}

RdmaEndpoint& RdmaEndpoint::operator=(RdmaEndpoint&& other) noexcept {
    if (this == &other) return *this;
    close();
    context_ = other.context_;
    pd_ = other.pd_;
    port_lid_ = other.port_lid_;
    gid_index_ = other.gid_index_;
    std::memcpy(gid_, other.gid_, 16);
    other.context_ = nullptr;
    other.pd_ = nullptr;
    other.port_lid_ = 0;
    other.gid_index_ = 0;
    std::memset(other.gid_, 0, 16);
    return *this;
}

void RdmaEndpoint::close() noexcept {
    // 顺序与 alloc 相反：先 dealloc_pd 再 close_device。
    if (pd_ != nullptr) {
        ibv_dealloc_pd(pd_);
    }
    if (context_ != nullptr) {
        ibv_close_device(context_);
    }
    pd_ = nullptr;
    context_ = nullptr;
}

const std::uint8_t (&RdmaEndpoint::gid() const noexcept)[16] { return gid_; }

RdmaEndpoint RdmaEndpoint::create_from_environment() {
    const std::uint16_t gid_index = parse_gid_index_env();
    const char* ifname = std::getenv("NANO_NCCL_RDMA_IFNAME");

    int num_devices = 0;
    DeviceListGuard list_guard;
    list_guard.list = ibv_get_device_list(&num_devices);
    if (list_guard.list == nullptr) {
        throw endpoint_error("ibv_get_device_list");
    }
    if (num_devices <= 0) {
        throw std::runtime_error("rdma endpoint: no RDMA devices found");
    }

    for (int i = 0; i < num_devices; ++i) {
        ibv_device* dev = list_guard.list[i];
        ContextGuard ctx_guard;
        ctx_guard.ctx = ibv_open_device(dev);
        if (ctx_guard.ctx == nullptr) {
            // 跳过无法打开的 HCA，继续试下一块。
            continue;
        }

        ibv_port_attr port_attr{};
        int rc = ibv_query_port(ctx_guard.ctx, 1, &port_attr);
        if (rc != 0) {
            // 端口查询失败本设备不可用，继续。
            continue;
        }
        if (port_attr.state != IBV_PORT_ACTIVE) {
            continue;
        }

        if (!iface_matches_hca(ctx_guard.ctx, dev, ifname, gid_index)) {
            continue;
        }

        // 匹配成功。再查一次 GID（用最终取用的 gid_index）填充要带走的 gid。
        union ibv_gid gid{};
        rc = ibv_query_gid(ctx_guard.ctx, 1, static_cast<int>(gid_index), &gid);
        if (rc != 0) {
            const char* hca_name = ibv_get_device_name(dev);
            std::string head = (hca_name != nullptr)
                ? (std::string("rdma endpoint ibv_query_gid(") + hca_name + ") failed: ")
                : std::string("rdma endpoint ibv_query_gid failed: ");
            throw std::runtime_error(head + std::strerror(rc));
        }

        PdGuard pd_guard;
        pd_guard.pd = ibv_alloc_pd(ctx_guard.ctx);
        if (pd_guard.pd == nullptr) {
            const char* hca_name = ibv_get_device_name(dev);
            std::string head = (hca_name != nullptr)
                ? (std::string("rdma endpoint ibv_alloc_pd(") + hca_name + ") failed: ")
                : std::string("rdma endpoint ibv_alloc_pd failed: ");
            throw std::runtime_error(head + std::strerror(errno));
        }

        // 全部就绪，把所有权转交构造出来的 RdmaEndpoint。
        return RdmaEndpoint(ctx_guard.release(), pd_guard.release(),
                           port_attr.lid, gid_index, gid.raw);
    }

    // 走完一圈没匹配上。
    if (ifname != nullptr && ifname[0] != '\0') {
        throw std::runtime_error(
            std::string("rdma endpoint: no active RDMA HCA tied to interface ") +
            ifname);
    }
    throw std::runtime_error("rdma endpoint: no active RDMA HCA found");
}

std::unique_ptr<struct ibv_cq, void(*)(struct ibv_cq*)>
RdmaEndpoint::allocate_cq(int wr_depth) {
    if (context_ == nullptr) {
        throw std::runtime_error("rdma endpoint allocate_cq on null context");
    }
    ibv_cq* cq = ibv_create_cq(context_, wr_depth, nullptr, nullptr, 0);
    if (cq == nullptr) {
        throw endpoint_error("ibv_create_cq");
    }
    return std::unique_ptr<struct ibv_cq, void(*)(struct ibv_cq*)>(cq, &destroy_cq);
}

}  // namespace nano_nccl::transport::rdma