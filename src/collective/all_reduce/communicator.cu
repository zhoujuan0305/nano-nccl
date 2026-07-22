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

#include <algorithm>
#include <cstddef>
#include <cstdio>
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
    }

    ~Impl() {
        if (socket_errors_ != nullptr && socket_errors_->has_error()) {
            socket_abort_.host_ptr()[0] = 1;
        }
        release_lifetime_tracking();
        for (const auto& proxy : socket_send_proxies_) proxy->drain();
        for (const auto& proxy : socket_recv_proxies_) proxy->drain();
        for (const auto& proxy : socket_send_proxies_) proxy->shutdown();
        for (const auto& proxy : socket_recv_proxies_) proxy->shutdown();
        for (const auto& proxy : socket_send_proxies_) proxy->join();
        for (const auto& proxy : socket_recv_proxies_) proxy->join();
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
        for (auto& connection : connections) {
            const auto hello = connection.hello();
            if (hello.channel < 0 || hello.channel >= kChannels ||
                hello.source_global_rank < 0 || hello.source_global_rank >= kRanks ||
                hello.destination_global_rank != (hello.source_global_rank + 1) % kRanks) {
                throw std::runtime_error("socket connection has invalid ring identity");
            }
            const int edge = hello.source_global_rank;
            if (collective::all_reduce::is_local_global_rank(topology_, edge)) {
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
                        edge, (edge + 1) % kRanks, hello.channel}, socket_errors_));
            } else {
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
                        edge, (edge + 1) % kRanks, hello.channel}, socket_errors_));
            }
        }
        for (const auto& proxy : socket_send_proxies_) proxy->start();
        for (const auto& proxy : socket_recv_proxies_) proxy->start();
    }

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
                if (send_socket) {
                    auto& socket = *socket_send_resources_[channel][send_edge];
                    kernel_args.send_fifo[channel] = reinterpret_cast<T*>(
                        socket.fifo->device_ptr(devices_[rank]));
                    kernel_args.control.send_head[channel] = socket.control.device_ptr(devices_[rank]);
                    kernel_args.control.send_tail[channel] = socket.control.device_ptr(devices_[rank]) + 1;
                    kernel_args.send_payload_bytes[channel] =
                        socket.payload_bytes.device_ptr(devices_[rank]);
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
            kernel_args.abort = socket_errors_ == nullptr
                ? nullptr : socket_abort_.device_ptr(devices_[rank]);
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
