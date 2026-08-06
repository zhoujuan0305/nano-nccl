#include "nano_nccl/communicator.h"

#include "core/buffer.h"
#include "core/numa.h"
#include "collective/all_reduce/communicator_internal.h"
#include "collective/all_reduce/topology.h"
#include "kernels/ring_simple_kernel.cuh"
#include "transport/socket/socket_proxy.h"
#include "nano_nccl/traits.h"
#include "transport/p2p/p2p_fifo.h"
#include "transport/p2p/p2p_step_counters.h"
#include "transport/p2p/p2p_topology.h"
#include "transport/shm/shm_fifo.h"
#if defined(NANO_NCCL_ENABLE_RDMA)
#include "transport/rdma/rdma_endpoint.h"
#include "transport/rdma/rdma_proxy.h"
#include "transport/rdma/rdma_qp.h"
#include "transport/socket/socket_protocol.h"

#include <infiniband/verbs.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <cuda_runtime.h>

#include <unistd.h>

namespace nano_nccl {

namespace {

using core::MappedBuffer;
using core::MappedU32Array;
using core::MappedU64Array;
using kernels::ring_simple_kernel;
using transport::SimpleControlArgs;
using transport::SimpleFifoArgs;

void require_single_process_devices(const std::vector<int>& devices) {
    if (devices.size() != kRanks) {
        throw std::runtime_error("communicator requires exactly " +
                                 std::to_string(kRanks) + " local devices");
    }
}

void require_local_devices(const std::vector<int>& devices) {
    if (devices.empty()) {
        throw std::runtime_error("communicator requires at least one local device");
    }
    for (int rank = 0; rank < kRanks; ++rank) {
        if (rank >= static_cast<int>(devices.size())) break;
        if (devices[rank] != rank) {
            throw std::runtime_error(
                "communicator devices must be the visible sequence 0.." +
                std::to_string(devices.size() - 1));
        }
    }

    int device_count = 0;
    CUDA_CHECK_THROW(cudaGetDeviceCount(&device_count));
    if (device_count < static_cast<int>(devices.size())) {
        throw std::runtime_error("need at least " + std::to_string(devices.size()) +
                                 " visible CUDA devices");
    }
}

void require_bf16_devices(const std::vector<int>& devices) {
    for (int device : devices) {
        cudaDeviceProp props{};
        CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, device));
        if (props.major < 8) {
            throw std::runtime_error(
                "bf16 requires compute capability 8.0 or newer");
        }
    }
}

void report_cuda_error_noexcept(cudaError_t status, const char* operation) noexcept {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed during Communicator destruction: %s\n",
                     operation, cudaGetErrorString(status));
    }
}

void fail_stop_on_cuda_cleanup_error(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) return;
    report_cuda_error_noexcept(status, operation);
    std::terminate();
}

}  // namespace

collective::all_reduce::SocketFdOwner::SocketFdOwner(std::vector<int> fds) {
    connections_.reserve(fds.size());
    for (int fd : fds) connections_.emplace_back(fd);
}

collective::all_reduce::SocketFdOwner
collective::all_reduce::SocketFdOwner::from_connections(
    std::vector<transport::socket::SocketConnection> connections) {
    SocketFdOwner owner;
    owner.connections_ = std::move(connections);
    return owner;
}

collective::all_reduce::SocketFdOwner::SocketFdOwner(
    SocketFdOwner&& other) noexcept
    : connections_(std::move(other.connections_)) {
    other.connections_.clear();
}

collective::all_reduce::SocketFdOwner&
collective::all_reduce::SocketFdOwner::operator=(SocketFdOwner&& other) noexcept {
    if (this == &other) return *this;
    close_all();
    connections_ = std::move(other.connections_);
    other.connections_.clear();
    return *this;
}

collective::all_reduce::SocketFdOwner::~SocketFdOwner() { close_all(); }

void collective::all_reduce::SocketFdOwner::close_all() noexcept {
    connections_.clear();
}

std::vector<transport::socket::SocketConnection>
collective::all_reduce::SocketFdOwner::release_connections() noexcept {
    return std::move(connections_);
}

class Communicator::Impl {
public:
    explicit Impl(const CommunicatorConfig& config) : devices_(config.devices) {
        require_single_process_devices(devices_);
        require_local_devices(devices_);
        topology_ = collective::all_reduce::make_single_process_topology(
            devices_, std::vector<TransportKind>(kRanks, TransportKind::Shm));
        transport_plan_ = transport::p2p::resolve_ring_transport(
            config.transport, topology_);
        topology_.edge_kinds = transport_plan_.edge_kinds();
        collective::all_reduce::validate_process_topology(topology_);
        if (transport_plan_.uses_p2p()) {
            transport::p2p::enable_p2p_ring_peer_access_or_throw(
                transport_plan_, topology_);
        }
        simple_fifo_steps_.reset(2 * kChannels * kRanks, -1, devices_);
        simple_fifo_base_step_.reset(kRanks * kChannels, -1, devices_);
        completion_events_.resize(devices_.size());
        fallback_streams_.resize(devices_.size());
        completion_recorded_.resize(devices_.size());
        fallback_in_flight_.resize(devices_.size());
    }

    Impl(const CommunicatorConfig& config,
         collective::all_reduce::ProcessTopology topology,
         collective::all_reduce::SocketFdOwner socket_fds)
        : devices_(config.devices), topology_(std::move(topology)),
          socket_fds_(std::move(socket_fds)) {
        require_local_devices(devices_);
        if (topology_.devices != devices_) {
            throw std::runtime_error("communicator topology devices do not match configuration");
        }
        collective::all_reduce::validate_process_topology(topology_);
        transport_plan_ = transport::p2p::RingTransportPlan(topology_.edge_kinds);
        if (transport_plan_.uses_p2p()) {
            transport::p2p::enable_p2p_ring_peer_access_or_throw(
                transport_plan_, topology_);
        }
        simple_fifo_steps_.reset(2 * kChannels * kRanks, -1, devices_);
        simple_fifo_base_step_.reset(kRanks * kChannels, -1, devices_);
        completion_events_.resize(devices_.size());
        fallback_streams_.resize(devices_.size());
        completion_recorded_.resize(devices_.size());
        fallback_in_flight_.resize(devices_.size());
        setup_socket_transport();
#if defined(NANO_NCCL_ENABLE_RDMA)
        setup_rdma_transport();
#endif
    }

    ~Impl() {
        if (socket_errors_ != nullptr && socket_errors_->has_error()) {
            socket_abort_.host_ptr()[0] = 1;
        }
#if defined(NANO_NCCL_ENABLE_RDMA)
        if (rdma_errors_ != nullptr && rdma_errors_->has_error()) {
            rdma_abort_.host_ptr()[0] = 1;
        }
#endif
        release_lifetime_tracking();
        for (const auto& proxy : socket_send_proxies_) proxy->drain();
        for (const auto& proxy : socket_recv_proxies_) proxy->drain();
        for (const auto& proxy : socket_send_proxies_) proxy->shutdown();
        for (const auto& proxy : socket_recv_proxies_) proxy->shutdown();
        for (const auto& proxy : socket_send_proxies_) proxy->join();
        for (const auto& proxy : socket_recv_proxies_) proxy->join();
#if defined(NANO_NCCL_ENABLE_RDMA)
        for (const auto& proxy : rdma_send_proxies_) proxy->drain();
        for (const auto& proxy : rdma_recv_proxies_) proxy->drain();
        for (const auto& proxy : rdma_send_proxies_) proxy->shutdown();
        for (const auto& proxy : rdma_recv_proxies_) proxy->shutdown();
        for (const auto& proxy : rdma_send_proxies_) proxy->join();
        for (const auto& proxy : rdma_recv_proxies_) proxy->join();
#endif
    }

    void all_reduce(const CollectiveArgs& args) {
        check_async_error();
        validate_args(args);
        switch (args.dtype) {
            case DType::Float:
                all_reduce_typed<float>(args);
                return;
            case DType::Float16:
                all_reduce_typed<__half>(args);
                return;
            case DType::BFloat16:
                ensure_bf16_devices_validated();
                all_reduce_typed<__nv_bfloat16>(args);
                return;
        }
        throw std::runtime_error("unsupported dtype");
    }

    int local_rank_count() const noexcept { return static_cast<int>(devices_.size()); }

    TransportKind transport() const noexcept { return transport_plan_.resolved_kind(); }

    void check_async_error() const {
        if (socket_errors_ != nullptr && socket_errors_->has_error()) {
            throw std::runtime_error(socket_errors_->message());
        }
#if defined(NANO_NCCL_ENABLE_RDMA)
        if (rdma_errors_ != nullptr && rdma_errors_->has_error()) {
            throw std::runtime_error(rdma_errors_->message());
        }
#endif
    }

private:
    template <typename T>
    struct FifoResources {
        std::vector<std::vector<std::unique_ptr<MappedBuffer<T>>>> shm_fifo;
        std::unique_ptr<transport::p2p::P2pFifo<T>> p2p_fifo;
        std::size_t slot_elems = 0;
    };

    struct SocketChannelResources {
        std::unique_ptr<MappedBuffer<std::uint8_t>> fifo;
        MappedU64Array control;
        MappedU32Array payload_bytes;
    };

    std::unique_ptr<SocketChannelResources> make_socket_resources(int device) {
        auto resources = std::make_unique<SocketChannelResources>();
        resources->fifo = std::make_unique<MappedBuffer<std::uint8_t>>(
            transport::kSimpleFifoBuffBytes, core::gpu_numa_node(device), devices_);
        resources->control.reset(2, core::gpu_numa_node(device), devices_);
        resources->payload_bytes.reset(transport::kSimpleFifoSteps,
                                       core::gpu_numa_node(device), devices_);
        return resources;
    }

    void setup_socket_transport() {
        if (!topology_.distributed) return;
        socket_send_resources_.resize(kChannels);
        socket_recv_resources_.resize(kChannels);
        for (int channel = 0; channel < kChannels; ++channel) {
            socket_send_resources_[channel].resize(kRanks);
            socket_recv_resources_[channel].resize(kRanks);
        }
        socket_abort_.reset(1, -1, devices_);
        socket_errors_ = std::make_shared<transport::socket::SocketAsyncErrorState>(
            socket_abort_.host_ptr());
        for (int edge = 0; edge < kRanks; ++edge) {
            if (transport_plan_.edge_kind(edge) != TransportKind::Socket) continue;
            int receiver = (edge + 1) % kRanks;
            if (collective::all_reduce::is_local_global_rank(topology_, edge)) {
                int local = collective::all_reduce::local_rank_for_global_rank(topology_, edge);
                for (int channel = 0; channel < kChannels; ++channel) {
                    socket_send_resources_[channel][edge] =
                        make_socket_resources(devices_[local]);
                }
            }
            if (collective::all_reduce::is_local_global_rank(topology_, receiver)) {
                int local = collective::all_reduce::local_rank_for_global_rank(topology_, receiver);
                for (int channel = 0; channel < kChannels; ++channel) {
                    socket_recv_resources_[channel][edge] =
                        make_socket_resources(devices_[local]);
                }
            }
        }

        auto connections = socket_fds_.release_connections();
#if defined(NANO_NCCL_ENABLE_RDMA)
        // The RDMA path reuses the bootstrap TCP fds for the QP info swap,
        // so Rdma-edged connections are handed back to socket_fds_ here for
        // setup_rdma_transport() to drain.
        std::vector<transport::socket::SocketConnection> rdma_residual;
#endif
        for (auto& connection : connections) {
            const auto hello = connection.hello();
            if (hello.channel < 0 || hello.channel >= kChannels ||
                hello.source_global_rank < 0 || hello.source_global_rank >= kRanks ||
                hello.destination_global_rank != (hello.source_global_rank + 1) % kRanks) {
                throw std::runtime_error("socket connection has invalid ring identity");
            }
            const int edge = hello.source_global_rank;
            const int receiver = (edge + 1) % kRanks;
#if defined(NANO_NCCL_ENABLE_RDMA)
            if (transport_plan_.edge_kind(edge) == TransportKind::Rdma) {
                rdma_residual.push_back(std::move(connection));
                continue;
            }
#endif
            if (collective::all_reduce::is_local_global_rank(topology_, edge)) {
                const int local = collective::all_reduce::local_rank_for_global_rank(
                    topology_, edge);
                const int fifo_numa_node = core::gpu_numa_node(devices_[local]);
                auto& resources = *socket_send_resources_[hello.channel][edge];
                transport::socket::SocketProxyFifo fifo{
                    resources.fifo->host_ptr(),
                    transport::kSimpleFifoBuffBytes / transport::kSimpleFifoSteps,
                    transport::kSimpleFifoSteps,
                    resources.payload_bytes.host_ptr(),
                    transport::shm::kSimpleFifoSliceSteps,
                };
                socket_send_proxies_.push_back(std::make_unique<transport::socket::SocketSendProxy>(
                    std::move(connection), fifo,
                    transport::socket::SocketSendControl{
                        resources.control.host_ptr(), resources.control.host_ptr() + 1},
                    transport::socket::SocketProxyIdentity{
                        edge, (edge + 1) % kRanks, hello.channel}, fifo_numa_node,
                    socket_errors_));
            } else {
                const int local = collective::all_reduce::local_rank_for_global_rank(
                    topology_, receiver);
                const int fifo_numa_node = core::gpu_numa_node(devices_[local]);
                auto& resources = *socket_recv_resources_[hello.channel][edge];
                transport::socket::SocketProxyFifo fifo{
                    resources.fifo->host_ptr(),
                    transport::kSimpleFifoBuffBytes / transport::kSimpleFifoSteps,
                    transport::kSimpleFifoSteps,
                    resources.payload_bytes.host_ptr(),
                    transport::shm::kSimpleFifoSliceSteps,
                };
                socket_recv_proxies_.push_back(std::make_unique<transport::socket::SocketRecvProxy>(
                    std::move(connection), fifo,
                    transport::socket::SocketRecvControl{
                        resources.control.host_ptr(), resources.control.host_ptr() + 1},
                    transport::socket::SocketProxyIdentity{
                        edge, (edge + 1) % kRanks, hello.channel}, fifo_numa_node,
                    socket_errors_));
            }
        }
        for (const auto& proxy : socket_send_proxies_) proxy->start();
        for (const auto& proxy : socket_recv_proxies_) proxy->start();
#if defined(NANO_NCCL_ENABLE_RDMA)
        // Re-assign the Rdma-edged residual back into socket_fds_ so that
        // setup_rdma_transport() can call socket_fds_.release_connections()
        // on the same SocketFdOwner instance.
        socket_fds_ = collective::all_reduce::SocketFdOwner::from_connections(
            std::move(rdma_residual));
#endif
    }

#if defined(NANO_NCCL_ENABLE_RDMA)
    // 1:1 mirror of SocketChannelResources plus the RC QP, the FIFO MR and
    // the peer info cached for the proxy construction. The MR is owned by
    // the unique_ptr below (RdmaMrDeleter); QP ownership transfers into the
    // proxy during setup_rdma_transport. fifo/control/payload_bytes mirror
    // the socket resource layout so a future kernel-wire-up task can reuse
    // the same slot indexing.
    struct IbvMrDeleter {
        void operator()(ibv_mr* mr) const noexcept {
            if (mr != nullptr) ibv_dereg_mr(mr);
        }
    };
    using RdmaMrPtr = std::unique_ptr<ibv_mr, IbvMrDeleter>;

    struct RdmaChannelResources {
        std::unique_ptr<MappedBuffer<std::uint8_t>> fifo;
        MappedU64Array control;
        MappedU32Array payload_bytes;
        RdmaMrPtr registered;            // dereg's fifo_mr_raw on destruction
        ibv_mr* fifo_mr_raw = nullptr;  // owned by registered; read-only view
        // WriteCts: sender owns CTS FIFO; receiver owns local CTS shadow.
        std::unique_ptr<transport::rdma::RdmaCtsSlot[]> cts_slots;
        RdmaMrPtr cts_registered;
        ibv_mr* cts_mr_raw = nullptr;
        std::unique_ptr<transport::rdma::RdmaQp> qp;
        transport::rdma::RdmaPeerInfo peer_info{};
    };

    std::unique_ptr<RdmaChannelResources> make_rdma_resources(int device) {
        auto resources = std::make_unique<RdmaChannelResources>();
        resources->fifo = std::make_unique<MappedBuffer<std::uint8_t>>(
            transport::kSimpleFifoBuffBytes, core::gpu_numa_node(device), devices_);
        resources->control.reset(2, core::gpu_numa_node(device), devices_);
        resources->payload_bytes.reset(transport::kSimpleFifoSteps,
                                       core::gpu_numa_node(device), devices_);
        return resources;
    }

    void register_rdma_cts_buffer(RdmaChannelResources& r, bool remote_write) {
        constexpr std::size_t kCtsBytes =
            static_cast<std::size_t>(transport::kSimpleFifoSteps) *
            sizeof(transport::rdma::RdmaCtsSlot);
        r.cts_slots.reset(new transport::rdma::RdmaCtsSlot[transport::kSimpleFifoSteps]());
        int access = IBV_ACCESS_LOCAL_WRITE;
        if (remote_write) {
            access |= IBV_ACCESS_REMOTE_WRITE;
        }
        ibv_mr* mr = ibv_reg_mr(rdma_endpoint_->pd(), r.cts_slots.get(), kCtsBytes,
                                access);
        if (mr == nullptr) {
            throw std::runtime_error(std::string("ibv_reg_mr rdma cts: ") +
                                     std::strerror(errno));
        }
        r.cts_mr_raw = mr;
        r.cts_registered = RdmaMrPtr(mr);
    }

    void setup_rdma_transport() {
        if (!topology_.distributed) return;
        bool has_rdma_edge = false;
        for (int edge = 0; edge < kRanks; ++edge) {
            if (transport_plan_.edge_kind(edge) == TransportKind::Rdma) {
                has_rdma_edge = true;
                break;
            }
        }
        if (!has_rdma_edge) return;

        const transport::rdma::RdmaDataPlane plane =
            transport::rdma::parse_rdma_data_plane_env();
        const bool write_cts = plane == transport::rdma::RdmaDataPlane::WriteCts;

        rdma_endpoint_ = std::make_shared<transport::rdma::RdmaEndpoint>(
            transport::rdma::RdmaEndpoint::create_from_environment());
        rdma_send_resources_.resize(kChannels);
        rdma_recv_resources_.resize(kChannels);
        for (int channel = 0; channel < kChannels; ++channel) {
            rdma_send_resources_[channel].resize(kRanks);
            rdma_recv_resources_[channel].resize(kRanks);
        }
        rdma_abort_.reset(1, -1, devices_);
        rdma_errors_ = std::make_shared<transport::rdma::RdmaAsyncErrorState>(
            rdma_abort_.host_ptr());

        // 1. For each Rdma edge: allocate send/recv FIFO + control, create RC
        //    QP, register the FIFO MR with bidirectional access flags.
        //    WriteCts also registers sender CTS FIFO and receiver CTS shadow.
        for (int edge = 0; edge < kRanks; ++edge) {
            if (transport_plan_.edge_kind(edge) != TransportKind::Rdma) continue;
            int receiver = (edge + 1) % kRanks;
            if (collective::all_reduce::is_local_global_rank(topology_, edge)) {
                int local = collective::all_reduce::local_rank_for_global_rank(
                    topology_, edge);
                for (int channel = 0; channel < kChannels; ++channel) {
                    rdma_send_resources_[channel][edge] =
                        make_rdma_resources(devices_[local]);
                    auto& r = *rdma_send_resources_[channel][edge];
                    constexpr int kRdmaProxyWr = 16;
                    r.qp = std::make_unique<transport::rdma::RdmaQp>(
                        transport::rdma::RdmaQp::create_init(
                            *rdma_endpoint_, kRdmaProxyWr, kRdmaProxyWr));
                    ibv_mr* mr = ibv_reg_mr(rdma_endpoint_->pd(), r.fifo->host_ptr(),
                        transport::kSimpleFifoBuffBytes,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_READ);
                    if (mr == nullptr) {
                        throw std::runtime_error(std::string("ibv_reg_mr rdma send fifo: ") +
                                                 std::strerror(errno));
                    }
                    r.fifo_mr_raw = mr;
                    r.registered = RdmaMrPtr(mr);
                    if (write_cts) {
                        // Sender owns CTS FIFO; peer receiver RDMA_WRITEs into it.
                        register_rdma_cts_buffer(r, /*remote_write=*/true);
                    }
                }
            }
            if (collective::all_reduce::is_local_global_rank(topology_, receiver)) {
                int local = collective::all_reduce::local_rank_for_global_rank(
                    topology_, receiver);
                for (int channel = 0; channel < kChannels; ++channel) {
                    rdma_recv_resources_[channel][edge] =
                        make_rdma_resources(devices_[local]);
                    auto& r = *rdma_recv_resources_[channel][edge];
                    constexpr int kRdmaProxyWr = 16;
                    r.qp = std::make_unique<transport::rdma::RdmaQp>(
                        transport::rdma::RdmaQp::create_init(
                            *rdma_endpoint_, kRdmaProxyWr, kRdmaProxyWr));
                    ibv_mr* mr = ibv_reg_mr(rdma_endpoint_->pd(), r.fifo->host_ptr(),
                        transport::kSimpleFifoBuffBytes,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_READ);
                    if (mr == nullptr) {
                        throw std::runtime_error(std::string("ibv_reg_mr rdma recv fifo: ") +
                                                 std::strerror(errno));
                    }
                    r.fifo_mr_raw = mr;
                    r.registered = RdmaMrPtr(mr);
                    if (write_cts) {
                        // Receiver owns CTS shadow SGE source (local write only).
                        register_rdma_cts_buffer(r, /*remote_write=*/false);
                    }
                }
            }
        }

        // 2. Iterate the TCP fds left in socket_fds_ by setup_socket_transport.
        //    Each fd is a ring edge; Rdma edges swap RdmaPeerInfo (64 bytes),
        //    then transition RTR/RTS on both sides before the fd closes.
        auto connections = socket_fds_.release_connections();
        std::vector<transport::socket::SocketConnection> rdma_ready_connections;
        for (auto& connection : connections) {
            const auto hello = connection.hello();
            if (hello.channel < 0 || hello.channel >= kChannels ||
                hello.source_global_rank < 0 || hello.source_global_rank >= kRanks ||
                hello.destination_global_rank != (hello.source_global_rank + 1) % kRanks) {
                throw std::runtime_error("rdma connection has invalid ring identity");
            }
            const int edge = hello.source_global_rank;
            if (transport_plan_.edge_kind(edge) != TransportKind::Rdma) continue;
            const int receiver = (edge + 1) % kRanks;
            const int fd = connection.fd();

            // Decide local role: sender (edge rank lives here) or receiver.
            const bool is_send =
                collective::all_reduce::is_local_global_rank(topology_, edge);
            const bool is_recv =
                collective::all_reduce::is_local_global_rank(topology_, receiver);
            if (!is_send && !is_recv) continue;  // defensive; should not happen

            const int local = collective::all_reduce::local_rank_for_global_rank(
                topology_, is_send ? edge : receiver);
            const int fifo_numa_node = core::gpu_numa_node(devices_[local]);

            RdmaChannelResources& r = is_send
                ? *rdma_send_resources_[hello.channel][edge]
                : *rdma_recv_resources_[hello.channel][edge];

            // Build local RdmaPeerInfo: qpn from QP, port LID / gid_index /
            // gid from the shared endpoint. WriteCts fills owned FIFO/CTS fields.
            transport::rdma::RdmaPeerInfo local_info = r.qp->local_info();
            local_info.port_lid = rdma_endpoint_->port_lid();
            local_info.gid_index = rdma_endpoint_->gid_index();
            std::memcpy(local_info.gid, rdma_endpoint_->gid(), 16);
            if (write_cts) {
                if (is_recv) {
                    local_info.recv_fifo_addr =
                        reinterpret_cast<std::uint64_t>(r.fifo->host_ptr());
                    local_info.recv_fifo_rkey = r.fifo_mr_raw->rkey;
                    local_info.recv_fifo_bytes =
                        static_cast<std::uint32_t>(transport::kSimpleFifoBuffBytes);
                }
                if (is_send) {
                    local_info.cts_fifo_addr =
                        reinterpret_cast<std::uint64_t>(r.cts_slots.get());
                    local_info.cts_fifo_rkey = r.cts_mr_raw->rkey;
                    local_info.cts_slot_count =
                        static_cast<std::uint32_t>(transport::kSimpleFifoSteps);
                }
            }

            transport::rdma::RdmaPeerInfo remote_info{};
            transport::socket::send_all(fd, &local_info, sizeof(local_info));
            transport::socket::recv_all(fd, &remote_info, sizeof(remote_info));

            if (write_cts) {
                if (is_send &&
                    (remote_info.recv_fifo_addr == 0 || remote_info.recv_fifo_rkey == 0 ||
                     remote_info.recv_fifo_bytes == 0)) {
                    throw std::runtime_error(
                        "rdma WriteCts peer missing recv FIFO endpoint");
                }
                if (is_recv &&
                    (remote_info.cts_fifo_addr == 0 || remote_info.cts_fifo_rkey == 0 ||
                     remote_info.cts_slot_count == 0)) {
                    throw std::runtime_error(
                        "rdma WriteCts peer missing CTS FIFO endpoint");
                }
                if (is_recv &&
                    remote_info.cts_slot_count !=
                        static_cast<std::uint32_t>(transport::kSimpleFifoSteps)) {
                    throw std::runtime_error(
                        "rdma WriteCts peer CTS slot_count mismatch");
                }
            }

            // v1 one-shot bootstrap: both PSNs start at 0.
            constexpr std::uint32_t kLocalPsn = 0;
            r.qp->transition_to_rtr(remote_info, rdma_endpoint_->gid_index());
            r.qp->transition_to_rts(kLocalPsn, remote_info.psn);

            // Move QP ownership into the proxy; the proxy RAII handles the
            // qp/cq destruction. SocketConnection closes fd on scope exit.
            auto qp_taken = std::move(r.qp);

            transport::rdma::RdmaProxyFifo fifo{
                r.fifo->host_ptr(),
                transport::kSimpleFifoBuffBytes / transport::kSimpleFifoSteps,
                transport::kSimpleFifoSteps,
                r.payload_bytes.host_ptr(),
                transport::shm::kSimpleFifoSliceSteps,
            };
            transport::rdma::RdmaProxyIdentity identity{
                edge, receiver, hello.channel};
            if (is_send) {
                if (write_cts) {
                    transport::rdma::RdmaWriteTargets targets{};
                    targets.remote_fifo_addr = remote_info.recv_fifo_addr;
                    targets.remote_fifo_rkey = remote_info.recv_fifo_rkey;
                    targets.remote_fifo_bytes = remote_info.recv_fifo_bytes;
                    targets.local_cts = r.cts_slots.get();
                    targets.cts_slot_count =
                        static_cast<std::size_t>(transport::kSimpleFifoSteps);
                    targets.local_cts_mr = r.cts_mr_raw;
                    rdma_send_proxies_.push_back(
                        std::make_unique<transport::rdma::RdmaSendProxy>(
                            std::move(*qp_taken), r.fifo_mr_raw, fifo,
                            transport::rdma::RdmaSendControl{
                                r.control.host_ptr(), r.control.host_ptr() + 1},
                            identity, fifo_numa_node, rdma_errors_,
                            transport::rdma::RdmaDataPlane::WriteCts, targets));
                } else {
                    rdma_send_proxies_.push_back(
                        std::make_unique<transport::rdma::RdmaSendProxy>(
                            std::move(*qp_taken), r.fifo_mr_raw, fifo,
                            transport::rdma::RdmaSendControl{
                                r.control.host_ptr(), r.control.host_ptr() + 1},
                            identity, fifo_numa_node, rdma_errors_));
                }
            } else {
                if (write_cts) {
                    transport::rdma::RdmaCtsRemote cts_remote{};
                    cts_remote.remote_cts_addr = remote_info.cts_fifo_addr;
                    cts_remote.remote_cts_rkey = remote_info.cts_fifo_rkey;
                    cts_remote.cts_slot_count = remote_info.cts_slot_count;
                    cts_remote.local_shadow = r.cts_slots.get();
                    cts_remote.local_shadow_mr = r.cts_mr_raw;
                    cts_remote.local_recv_fifo_addr =
                        reinterpret_cast<std::uint64_t>(r.fifo->host_ptr());
                    cts_remote.local_recv_fifo_rkey = r.fifo_mr_raw->rkey;
                    rdma_recv_proxies_.push_back(
                        std::make_unique<transport::rdma::RdmaRecvProxy>(
                            std::move(*qp_taken), r.fifo_mr_raw, fifo,
                            transport::rdma::RdmaRecvControl{
                                r.control.host_ptr(), r.control.host_ptr() + 1},
                            identity, fifo_numa_node, rdma_errors_,
                            transport::rdma::RdmaDataPlane::WriteCts, cts_remote));
                } else {
                    rdma_recv_proxies_.push_back(
                        std::make_unique<transport::rdma::RdmaRecvProxy>(
                            std::move(*qp_taken), r.fifo_mr_raw, fifo,
                            transport::rdma::RdmaRecvControl{
                                r.control.host_ptr(), r.control.host_ptr() + 1},
                            identity, fifo_numa_node, rdma_errors_));
                }
            }
            rdma_ready_connections.push_back(std::move(connection));
            // qp_taken (moved-from state) destructs here; r.qp holds nullptr.
        }

        // 3. Start receive proxies first. start() posts the next expected
        //    Simple-protocol slot (and WriteCts initial CTS) before launching
        //    its worker; a peer-ready byte proves the first sender WQE cannot
        //    hit an empty RQ / missing CTS.
        for (const auto& proxy : rdma_recv_proxies_) proxy->start();
        const std::uint8_t ready = 1;
        for (const auto& connection : rdma_ready_connections) {
            std::uint8_t peer_ready = 0;
            transport::socket::send_all(connection.fd(), &ready, sizeof(ready));
            transport::socket::recv_all(connection.fd(), &peer_ready, sizeof(peer_ready));
            if (peer_ready != ready) {
                throw std::runtime_error("rdma peer did not confirm recv readiness");
            }
        }
        rdma_ready_connections.clear();

        // Both sides confirmed pre-posted recv WQEs; send proxies can now run.
        for (const auto& proxy : rdma_send_proxies_) proxy->start();
    }
#endif

    class ResetEvents {
    public:
        explicit ResetEvents(int rank_count) : rank_count_(rank_count) {}
        ~ResetEvents() { destroy_noexcept(); }

        void create(int rank) {
            CUDA_CHECK_THROW(cudaSetDevice(rank));
            CUDA_CHECK_THROW(cudaEventCreateWithFlags(&events_[rank],
                                                      cudaEventDisableTiming));
        }

        void record(int rank, cudaStream_t stream) {
            CUDA_CHECK_THROW(cudaSetDevice(rank));
            CUDA_CHECK_THROW(cudaEventRecord(events_[rank], stream));
        }

        cudaEvent_t at(int rank) const { return events_[rank]; }

        void destroy() {
            for (int rank = 0; rank < rank_count_; ++rank) {
                if (events_[rank] == nullptr) continue;
                CUDA_CHECK_THROW(cudaSetDevice(rank));
                cudaEvent_t event = events_[rank];
                CUDA_CHECK_THROW(cudaEventDestroy(event));
                events_[rank] = nullptr;
            }
        }

    private:
        void destroy_noexcept() noexcept {
            for (int rank = 0; rank < rank_count_; ++rank) {
                if (events_[rank] == nullptr) continue;
                cudaError_t status = cudaSetDevice(rank);
                fail_stop_on_cuda_cleanup_error(status, "cudaSetDevice");
                status = cudaEventDestroy(events_[rank]);
                fail_stop_on_cuda_cleanup_error(status, "cudaEventDestroy");
                events_[rank] = nullptr;
            }
        }

        cudaEvent_t events_[kRanks]{};
        int rank_count_ = 0;
    };

    void validate_args(const CollectiveArgs& args) const {
        if (args.send_buffers.size() != devices_.size() ||
            args.recv_buffers.size() != devices_.size() ||
            args.streams.size() != devices_.size()) {
            throw std::runtime_error("collective arguments must have one entry per rank");
        }
        if (args.count == 0) {
            throw std::runtime_error("collective count must be positive");
        }
        switch (args.redop) {
            case RedOp::Sum:
            case RedOp::Avg:
            case RedOp::Max:
            case RedOp::Min:
                break;
            default:
                throw std::runtime_error("unsupported reduction operation");
        }
        for (int rank = 0; rank < local_rank_count(); ++rank) {
            if (args.send_buffers[rank] == nullptr || args.recv_buffers[rank] == nullptr ||
                args.streams[rank] == nullptr) {
                throw std::runtime_error("collective buffers and streams must be non-null");
            }
            if (args.send_buffers[rank] == args.recv_buffers[rank]) {
                throw std::runtime_error("in-place all_reduce is unsupported");
            }
        }
    }

    void require_previous_launch_complete() const {
        if (has_untracked_launch_) {
            throw std::runtime_error(
                "cannot grow communicator FIFO after completion tracking failed");
        }
        if (!has_launch_) return;
        for (int rank = 0; rank < local_rank_count(); ++rank) {
            CUDA_CHECK_THROW(cudaSetDevice(devices_[rank]));
            cudaError_t status = cudaEventQuery(completion_events_[rank]);
            if (status == cudaErrorNotReady) {
                throw std::runtime_error(
                    "cannot grow communicator FIFO while a prior all_reduce is running");
            }
            CUDA_CHECK_THROW(status);
        }
    }

    void wait_for_previous_launch_on_streams(
        const std::vector<cudaStream_t>& streams) const {
        if (has_untracked_launch_) {
            throw std::runtime_error(
                "cannot launch all_reduce after completion tracking failed");
        }
        if (!has_launch_) return;
        for (int stream_rank = 0; stream_rank < local_rank_count(); ++stream_rank) {
            CUDA_CHECK_THROW(cudaSetDevice(devices_[stream_rank]));
            for (int event_rank = 0; event_rank < local_rank_count(); ++event_rank) {
                CUDA_CHECK_THROW(cudaStreamWaitEvent(streams[stream_rank],
                                                     completion_events_[event_rank], 0));
            }
        }
    }

    template <typename T>
    void ensure_fifo_buffers(FifoResources<T>* resources, std::size_t count) {
        std::size_t required_slot_elems = 0;
        for (int channel = 0; channel < kChannels; ++channel) {
            std::size_t part_offset = 0;
            std::size_t part_count = 0;
            std::size_t chunk_count = 0;
            transport::shm::cbd_part<T>(count, channel, &part_offset,
                                        &part_count, &chunk_count);
            required_slot_elems = std::max(required_slot_elems, chunk_count);
        }
        required_slot_elems = std::max<std::size_t>(required_slot_elems, 1);
        if (topology_.distributed) {
            required_slot_elems = transport::shm::simple_fifo_step_elems<T>();
        }
        if (required_slot_elems <= resources->slot_elems) return;

        // Replacing FIFO storage while a kernel can still access it would race.
        require_previous_launch_complete();
        FifoResources<T> replacement;
        replacement.slot_elems = required_slot_elems;
        for (int channel = 0; channel < kChannels; ++channel) {
            for (int edge = 0; edge < topology_.global_rank_count; ++edge) {
                if (transport_plan_.edge_kind(edge) != TransportKind::Shm) continue;
                int receiver = (edge + 1) % kRanks;
                if (!collective::all_reduce::is_local_global_rank(topology_, edge) ||
                    !collective::all_reduce::is_local_global_rank(topology_, receiver)) {
                    continue;
                }
                int receiver_local =
                    collective::all_reduce::local_rank_for_global_rank(topology_, receiver);
                if (replacement.shm_fifo.empty()) {
                    replacement.shm_fifo.resize(kChannels);
                    for (auto& fifo_by_edge : replacement.shm_fifo) {
                        fifo_by_edge.resize(topology_.global_rank_count);
                    }
                }
                replacement.shm_fifo[channel][edge] = std::make_unique<MappedBuffer<T>>(
                    transport::shm::kSimpleFifoSteps * replacement.slot_elems,
                    core::gpu_numa_node(devices_[receiver_local]), devices_);
            }
        }
        if (transport_plan_.uses_p2p()) {
            replacement.p2p_fifo = std::make_unique<transport::p2p::P2pFifo<T>>(
                replacement.slot_elems, transport_plan_, topology_);
        }
        *resources = std::move(replacement);
    }

    void reset_control(const std::vector<cudaStream_t>& streams) {
        simple_fifo_steps_.clear_host();
        simple_fifo_base_step_.clear_host();
        if (!transport_plan_.uses_p2p()) return;

        p2p_steps_->reset(streams);

        ResetEvents reset_events(local_rank_count());
        for (int rank = 0; rank < local_rank_count(); ++rank) {
            reset_events.create(rank);
            reset_events.record(rank, streams[rank]);
        }
        for (int stream_rank = 0; stream_rank < local_rank_count(); ++stream_rank) {
            CUDA_CHECK_THROW(cudaSetDevice(stream_rank));
            for (int event_rank = 0; event_rank < local_rank_count(); ++event_rank) {
                CUDA_CHECK_THROW(cudaStreamWaitEvent(streams[stream_rank],
                                                     reset_events.at(event_rank), 0));
            }
        }
        reset_events.destroy();
    }

    template <typename T, RedOp kRedOp>
    void launch_ring_simple(const CollectiveArgs& args, FifoResources<T>* resources) {
        ensure_completion_events();
        for (int rank = 0; rank < local_rank_count(); ++rank) {
            SimpleFifoArgs<T> kernel_args{};
            int global_rank = topology_.local_rank_offset + rank;
            kernel_args.rank = global_rank;
            kernel_args.count = args.count;
            kernel_args.slot_elems = resources->slot_elems;
            kernel_args.step_elems = transport::shm::simple_fifo_step_elems<T>();
            kernel_args.input = static_cast<const T*>(args.send_buffers[rank]);
            kernel_args.output = static_cast<T*>(args.recv_buffers[rank]);

            SimpleControlArgs shm_control = transport::shm::make_simple_control_args(
                simple_fifo_steps_.device_ptr(devices_[rank]),
                simple_fifo_base_step_.device_ptr(devices_[rank]), global_rank);
            SimpleControlArgs p2p_control{};
            if (p2p_steps_ != nullptr) {
                p2p_control = p2p_steps_->control_args(global_rank);
            }

            int next = (global_rank + 1) % kRanks;
            int previous = (global_rank + kRanks - 1) % kRanks;
            int send_edge = transport::shm::ring_edge_index(global_rank, next, kRanks);
            int recv_edge = transport::shm::ring_edge_index(previous, global_rank, kRanks);
            if (send_edge < 0 || recv_edge < 0) {
                throw std::runtime_error("ring_simple saw an unexpected ring edge");
            }
            for (int channel = 0; channel < kChannels; ++channel) {
                bool send_p2p = transport_plan_.edge_kind(send_edge) == TransportKind::P2p;
                bool recv_p2p = transport_plan_.edge_kind(recv_edge) == TransportKind::P2p;
                bool send_socket = transport_plan_.edge_kind(send_edge) == TransportKind::Socket;
                bool recv_socket = transport_plan_.edge_kind(recv_edge) == TransportKind::Socket;
#if defined(NANO_NCCL_ENABLE_RDMA)
                bool send_rdma = transport_plan_.edge_kind(send_edge) == TransportKind::Rdma;
                bool recv_rdma = transport_plan_.edge_kind(recv_edge) == TransportKind::Rdma;
#endif
                if (send_socket) {
                    auto& socket = *socket_send_resources_[channel][send_edge];
                    kernel_args.send_fifo[channel] = reinterpret_cast<T*>(
                        socket.fifo->device_ptr(devices_[rank]));
                    kernel_args.control.send_head[channel] = socket.control.device_ptr(devices_[rank]);
                    kernel_args.control.send_tail[channel] = socket.control.device_ptr(devices_[rank]) + 1;
                    kernel_args.send_payload_bytes[channel] =
                        socket.payload_bytes.device_ptr(devices_[rank]);
#if defined(NANO_NCCL_ENABLE_RDMA)
                } else if (send_rdma) {
                    auto& rdma = *rdma_send_resources_[channel][send_edge];
                    kernel_args.send_fifo[channel] = reinterpret_cast<T*>(
                        rdma.fifo->device_ptr(devices_[rank]));
                    kernel_args.control.send_head[channel] = rdma.control.device_ptr(devices_[rank]);
                    kernel_args.control.send_tail[channel] = rdma.control.device_ptr(devices_[rank]) + 1;
                    kernel_args.send_payload_bytes[channel] =
                        rdma.payload_bytes.device_ptr(devices_[rank]);
#endif
                } else {
                    kernel_args.send_fifo[channel] = send_p2p
                        ? resources->p2p_fifo->edge_ptr(channel, send_edge)
                        : resources->shm_fifo[channel][send_edge]->device_ptr(devices_[rank]);
                    kernel_args.control.send_head[channel] = send_p2p
                        ? p2p_control.send_head[channel] : shm_control.send_head[channel];
                    kernel_args.control.send_tail[channel] = send_p2p
                        ? p2p_control.send_tail[channel] : shm_control.send_tail[channel];
                }
                if (recv_socket) {
                    auto& socket = *socket_recv_resources_[channel][recv_edge];
                    kernel_args.recv_fifo[channel] = reinterpret_cast<const T*>(
                        socket.fifo->device_ptr(devices_[rank]));
                    kernel_args.control.recv_head[channel] = socket.control.device_ptr(devices_[rank]);
                    kernel_args.control.recv_tail[channel] = socket.control.device_ptr(devices_[rank]) + 1;
                    kernel_args.recv_payload_bytes[channel] =
                        socket.payload_bytes.device_ptr(devices_[rank]);
#if defined(NANO_NCCL_ENABLE_RDMA)
                } else if (recv_rdma) {
                    auto& rdma = *rdma_recv_resources_[channel][recv_edge];
                    kernel_args.recv_fifo[channel] = reinterpret_cast<const T*>(
                        rdma.fifo->device_ptr(devices_[rank]));
                    kernel_args.control.recv_head[channel] = rdma.control.device_ptr(devices_[rank]);
                    kernel_args.control.recv_tail[channel] = rdma.control.device_ptr(devices_[rank]) + 1;
                    kernel_args.recv_payload_bytes[channel] =
                        rdma.payload_bytes.device_ptr(devices_[rank]);
#endif
                } else {
                    kernel_args.recv_fifo[channel] = recv_p2p
                        ? resources->p2p_fifo->edge_ptr(channel, recv_edge)
                        : resources->shm_fifo[channel][recv_edge]->device_ptr(devices_[rank]);
                    kernel_args.control.recv_tail[channel] = recv_p2p
                        ? p2p_control.recv_tail[channel] : shm_control.recv_tail[channel];
                    kernel_args.control.recv_head[channel] = recv_p2p
                        ? p2p_control.recv_head[channel] : shm_control.recv_head[channel];
                }
            }
#if defined(NANO_NCCL_ENABLE_RDMA)
            kernel_args.abort = rdma_errors_ != nullptr
                ? rdma_abort_.device_ptr(devices_[rank])
                : (socket_errors_ == nullptr
                    ? nullptr : socket_abort_.device_ptr(devices_[rank]));
#else
            kernel_args.abort = socket_errors_ == nullptr
                ? nullptr : socket_abort_.device_ptr(devices_[rank]);
#endif
            kernel_args.control.base_steps = p2p_control.base_steps != nullptr
                ? p2p_control.base_steps : shm_control.base_steps;

            CUDA_CHECK_THROW(cudaSetDevice(devices_[rank]));
            ring_simple_kernel<T, kRedOp>
                <<<kChannels, kBlockThreads, 0, args.streams[rank]>>>(kernel_args, kRanks);
            CUDA_CHECK_THROW(cudaGetLastError());
            record_completion(rank, args.streams[rank]);
        }
    }

    void ensure_completion_events() {
        for (int rank = 0; rank < local_rank_count(); ++rank) {
            CUDA_CHECK_THROW(cudaSetDevice(devices_[rank]));
            if (completion_events_[rank] == nullptr) {
                CUDA_CHECK_THROW(cudaEventCreateWithFlags(&completion_events_[rank],
                                                          cudaEventDisableTiming));
            }
        }
    }

    void record_completion(int rank, cudaStream_t stream) {
        cudaError_t status = cudaEventRecord(completion_events_[rank], stream);
        if (status != cudaSuccess) {
            has_untracked_launch_ = true;
            CUDA_CHECK_THROW(status);
        }
        completion_recorded_[rank] = true;
        fallback_in_flight_[rank] = false;
        has_launch_ = true;
    }

    void begin_fallback_tracking(const std::vector<cudaStream_t>& streams) {
        for (int rank = 0; rank < local_rank_count(); ++rank) {
            fallback_streams_[rank] = streams[rank];
            fallback_in_flight_[rank] = true;
        }
    }

    void release_lifetime_tracking() noexcept {
        for (int rank = 0; rank < local_rank_count(); ++rank) {
            cudaError_t status = cudaSetDevice(rank);
            fail_stop_on_cuda_cleanup_error(status, "cudaSetDevice");
            if (fallback_in_flight_[rank]) {
                status = cudaStreamSynchronize(fallback_streams_[rank]);
                fail_stop_on_cuda_cleanup_error(status, "cudaStreamSynchronize");
            }
            if (completion_events_[rank] != nullptr && completion_recorded_[rank]) {
                status = cudaEventSynchronize(completion_events_[rank]);
                fail_stop_on_cuda_cleanup_error(status, "cudaEventSynchronize");
            }
            if (completion_events_[rank] != nullptr) {
                status = cudaEventDestroy(completion_events_[rank]);
                fail_stop_on_cuda_cleanup_error(status, "cudaEventDestroy");
                completion_events_[rank] = nullptr;
            }
        }
    }

    void ensure_bf16_devices_validated() {
        if (bf16_devices_validated_) return;
        require_bf16_devices(devices_);
        bf16_devices_validated_ = true;
    }

    template <typename T>
    void all_reduce_typed(const CollectiveArgs& args) {
        switch (args.redop) {
            case RedOp::Sum:
                all_reduce_typed<T, RedOp::Sum>(args);
                return;
            case RedOp::Avg:
                all_reduce_typed<T, RedOp::Avg>(args);
                return;
            case RedOp::Max:
                all_reduce_typed<T, RedOp::Max>(args);
                return;
            case RedOp::Min:
                all_reduce_typed<T, RedOp::Min>(args);
                return;
        }
        throw std::runtime_error("unsupported reduction operation");
    }

    template <typename T, RedOp kRedOp>
    void all_reduce_typed(const CollectiveArgs& args) {
        FifoResources<T>* resources = nullptr;
        if constexpr (std::is_same_v<T, float>) {
            resources = &float_resources_;
        } else if constexpr (std::is_same_v<T, __half>) {
            resources = &float16_resources_;
        } else {
            resources = &bfloat16_resources_;
        }
        // Reject an unsafe replacement before adding any caller-stream work.
        ensure_fifo_buffers(resources, args.count);
        if (has_untracked_launch_) {
            throw std::runtime_error(
                "cannot launch all_reduce after completion tracking failed");
        }
        // Every caller stream has a fallback before reset, waits, or a launch
        // can enqueue work. A successful completion event clears its fallback.
        begin_fallback_tracking(args.streams);
        try {
            wait_for_previous_launch_on_streams(args.streams);
            if (!control_initialized_) {
                if (transport_plan_.uses_p2p()) {
                    p2p_steps_ =
                        std::make_unique<transport::p2p::P2pStepCounters>(
                            transport_plan_, topology_);
                }
                // Counters are reset once; subsequent launches advance persistent steps.
                reset_control(args.streams);
                control_initialized_ = true;
            }
            launch_ring_simple<T, kRedOp>(args, resources);
        } catch (...) {
            has_untracked_launch_ = true;
            throw;
        }
    }

    std::vector<int> devices_;
    collective::all_reduce::ProcessTopology topology_{};
    collective::all_reduce::SocketFdOwner socket_fds_;
    std::vector<std::vector<std::unique_ptr<SocketChannelResources>>> socket_send_resources_;
    std::vector<std::vector<std::unique_ptr<SocketChannelResources>>> socket_recv_resources_;
    MappedU32Array socket_abort_;
    std::shared_ptr<transport::socket::SocketAsyncErrorState> socket_errors_;
    std::vector<std::unique_ptr<transport::socket::SocketSendProxy>> socket_send_proxies_;
    std::vector<std::unique_ptr<transport::socket::SocketRecvProxy>> socket_recv_proxies_;
#if defined(NANO_NCCL_ENABLE_RDMA)
    std::shared_ptr<transport::rdma::RdmaEndpoint> rdma_endpoint_;
    std::vector<std::vector<std::unique_ptr<RdmaChannelResources>>> rdma_send_resources_;
    std::vector<std::vector<std::unique_ptr<RdmaChannelResources>>> rdma_recv_resources_;
    MappedU32Array rdma_abort_;
    std::shared_ptr<transport::rdma::RdmaAsyncErrorState> rdma_errors_;
    std::vector<std::unique_ptr<transport::rdma::RdmaSendProxy>> rdma_send_proxies_;
    std::vector<std::unique_ptr<transport::rdma::RdmaRecvProxy>> rdma_recv_proxies_;
#endif
    MappedU64Array simple_fifo_steps_;
    MappedU64Array simple_fifo_base_step_;
    std::unique_ptr<transport::p2p::P2pStepCounters> p2p_steps_;
    FifoResources<float> float_resources_;
    FifoResources<__half> float16_resources_;
    FifoResources<__nv_bfloat16> bfloat16_resources_;
    transport::p2p::RingTransportPlan transport_plan_ =
        transport::p2p::RingTransportPlan::uniform(TransportKind::Shm);
    std::vector<cudaEvent_t> completion_events_;
    std::vector<cudaStream_t> fallback_streams_;
    std::vector<bool> completion_recorded_;
    std::vector<bool> fallback_in_flight_;
    bool bf16_devices_validated_ = false;
    bool control_initialized_ = false;
    bool has_launch_ = false;
    bool has_untracked_launch_ = false;
};

Communicator::Communicator(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Communicator::Communicator(Communicator&&) noexcept = default;
Communicator& Communicator::operator=(Communicator&&) noexcept = default;
Communicator::~Communicator() = default;

void Communicator::all_reduce(const CollectiveArgs& args) { impl_->all_reduce(args); }

void Communicator::reduce_scatter(const CollectiveArgs&) {
    throw std::runtime_error("reduce_scatter is unsupported");
}

void Communicator::all_gather(const CollectiveArgs&) {
    throw std::runtime_error("all_gather is unsupported");
}

void Communicator::check_async_error() const { impl_->check_async_error(); }

int Communicator::local_rank_count() const noexcept { return impl_->local_rank_count(); }

int Communicator::global_rank_count() const noexcept { return kRanks; }

TransportKind Communicator::transport() const noexcept { return impl_->transport(); }

std::unique_ptr<Communicator> create_communicator(const CommunicatorConfig& config) {
    return std::unique_ptr<Communicator>(new Communicator(std::make_unique<Communicator::Impl>(config)));
}

std::unique_ptr<Communicator> collective::all_reduce::CommunicatorFactory::create(
    const CommunicatorConfig& config, ProcessTopology topology,
    SocketFdOwner socket_fds) {
    return std::unique_ptr<Communicator>(new Communicator(
        std::make_unique<Communicator::Impl>(config, std::move(topology),
                                             std::move(socket_fds))));
}

}  // namespace nano_nccl
