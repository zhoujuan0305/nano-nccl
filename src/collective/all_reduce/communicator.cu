#include "nano_nccl/communicator.h"

#include "core/buffer.h"
#include "core/numa.h"
#include "kernels/ring_simple_kernel.cuh"
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

namespace nano_nccl {

namespace {

using core::MappedBuffer;
using core::MappedU64Array;
using kernels::ring_simple_kernel;
using transport::SimpleControlArgs;
using transport::SimpleFifoArgs;

void require_supported_devices(const std::vector<int>& devices) {
    if (devices.size() != kRanks) {
        throw std::runtime_error("communicator requires exactly " +
                                 std::to_string(kRanks) + " local devices");
    }
    for (int rank = 0; rank < kRanks; ++rank) {
        if (devices[rank] != rank) {
            throw std::runtime_error(
                "communicator devices must be the visible sequence 0.." +
                std::to_string(kRanks - 1));
        }
    }

    int device_count = 0;
    CUDA_CHECK_THROW(cudaGetDeviceCount(&device_count));
    if (device_count < kRanks) {
        throw std::runtime_error("need at least " + std::to_string(kRanks) +
                                 " visible CUDA devices");
    }
}

void require_bf16_devices() {
    for (int rank = 0; rank < kRanks; ++rank) {
        cudaDeviceProp props{};
        CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, rank));
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

class Communicator::Impl {
public:
    explicit Impl(const CommunicatorConfig& config) : devices_(config.devices) {
        require_supported_devices(devices_);
        transport_plan_ = transport::p2p::resolve_ring_transport(config.transport);
        if (transport_plan_.uses_p2p()) {
            transport::p2p::enable_p2p_ring_peer_access_or_throw(transport_plan_);
        }
        simple_fifo_steps_.reset(2 * kChannels * kRanks);
        simple_fifo_base_step_.reset(kRanks * kChannels);
    }

    ~Impl() { release_lifetime_tracking(); }

    void all_reduce(const CollectiveArgs& args) {
        validate_args(args);
        switch (args.dtype) {
            case DType::Float:
                all_reduce_typed<float>(args);
                return;
            case DType::Float16:
                all_reduce_typed<__half>(args);
                return;
            case DType::BFloat16:
                require_bf16_devices();
                all_reduce_typed<__nv_bfloat16>(args);
                return;
        }
        throw std::runtime_error("unsupported dtype");
    }

    int local_rank_count() const noexcept { return static_cast<int>(devices_.size()); }

private:
    template <typename T>
    struct FifoResources {
        std::unique_ptr<MappedBuffer<T>> shm_fifo[kChannels][kRanks];
        std::unique_ptr<transport::p2p::P2pFifo<T>> p2p_fifo;
        std::size_t slot_elems = 0;
    };

    class ResetEvents {
    public:
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
            for (int rank = 0; rank < kRanks; ++rank) {
                if (events_[rank] == nullptr) continue;
                CUDA_CHECK_THROW(cudaSetDevice(rank));
                cudaEvent_t event = events_[rank];
                CUDA_CHECK_THROW(cudaEventDestroy(event));
                events_[rank] = nullptr;
            }
        }

    private:
        void destroy_noexcept() noexcept {
            for (int rank = 0; rank < kRanks; ++rank) {
                if (events_[rank] == nullptr) continue;
                cudaError_t status = cudaSetDevice(rank);
                fail_stop_on_cuda_cleanup_error(status, "cudaSetDevice");
                status = cudaEventDestroy(events_[rank]);
                fail_stop_on_cuda_cleanup_error(status, "cudaEventDestroy");
                events_[rank] = nullptr;
            }
        }

        cudaEvent_t events_[kRanks]{};
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
        if (args.redop != RedOp::Sum) {
            throw std::runtime_error("only sum reduction is supported");
        }
        for (int rank = 0; rank < kRanks; ++rank) {
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
        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK_THROW(cudaSetDevice(rank));
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
        for (int stream_rank = 0; stream_rank < kRanks; ++stream_rank) {
            CUDA_CHECK_THROW(cudaSetDevice(stream_rank));
            for (int event_rank = 0; event_rank < kRanks; ++event_rank) {
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
        if (required_slot_elems <= resources->slot_elems) return;

        // Replacing FIFO storage while a kernel can still access it would race.
        require_previous_launch_complete();
        FifoResources<T> replacement;
        replacement.slot_elems = required_slot_elems;
        for (int channel = 0; channel < kChannels; ++channel) {
            for (int edge = 0; edge < kRanks; ++edge) {
                if (transport_plan_.edge_kind(edge) != TransportKind::Shm) continue;
                int receiver = (edge + 1) % kRanks;
                replacement.shm_fifo[channel][edge] = std::make_unique<MappedBuffer<T>>(
                    transport::shm::kSimpleFifoSteps * replacement.slot_elems,
                    core::gpu_numa_node(receiver));
            }
        }
        if (transport_plan_.uses_p2p()) {
            replacement.p2p_fifo = std::make_unique<transport::p2p::P2pFifo<T>>(
                replacement.slot_elems, transport_plan_);
        }
        *resources = std::move(replacement);
    }

    void reset_control(const std::vector<cudaStream_t>& streams) {
        simple_fifo_steps_.clear_host();
        simple_fifo_base_step_.clear_host();
        if (!transport_plan_.uses_p2p()) return;

        cudaStream_t raw_streams[kRanks];
        for (int rank = 0; rank < kRanks; ++rank) raw_streams[rank] = streams[rank];
        p2p_steps_->reset(raw_streams);

        ResetEvents reset_events;
        for (int rank = 0; rank < kRanks; ++rank) {
            reset_events.create(rank);
            reset_events.record(rank, streams[rank]);
        }
        for (int stream_rank = 0; stream_rank < kRanks; ++stream_rank) {
            CUDA_CHECK_THROW(cudaSetDevice(stream_rank));
            for (int event_rank = 0; event_rank < kRanks; ++event_rank) {
                CUDA_CHECK_THROW(cudaStreamWaitEvent(streams[stream_rank],
                                                     reset_events.at(event_rank), 0));
            }
        }
        reset_events.destroy();
    }

    template <typename T>
    void launch_ring_simple(const CollectiveArgs& args, FifoResources<T>* resources) {
        ensure_completion_events();
        for (int rank = 0; rank < kRanks; ++rank) {
            SimpleFifoArgs<T> kernel_args{};
            kernel_args.rank = rank;
            kernel_args.count = args.count;
            kernel_args.slot_elems = resources->slot_elems;
            kernel_args.step_elems = transport::shm::simple_fifo_step_elems<T>();
            kernel_args.input = static_cast<const T*>(args.send_buffers[rank]);
            kernel_args.output = static_cast<T*>(args.recv_buffers[rank]);

            SimpleControlArgs shm_control = transport::shm::make_simple_control_args(
                simple_fifo_steps_.device_ptr(rank), simple_fifo_base_step_.device_ptr(rank),
                rank);
            SimpleControlArgs p2p_control{};
            if (p2p_steps_ != nullptr) p2p_control = p2p_steps_->control_args(rank);

            int next = (rank + 1) % kRanks;
            int previous = (rank + kRanks - 1) % kRanks;
            int send_edge = transport::shm::ring_edge_index(rank, next, kRanks);
            int recv_edge = transport::shm::ring_edge_index(previous, rank, kRanks);
            if (send_edge < 0 || recv_edge < 0) {
                throw std::runtime_error("ring_simple saw an unexpected ring edge");
            }
            for (int channel = 0; channel < kChannels; ++channel) {
                bool send_p2p = transport_plan_.edge_kind(send_edge) == TransportKind::P2p;
                bool recv_p2p = transport_plan_.edge_kind(recv_edge) == TransportKind::P2p;
                kernel_args.send_fifo[channel] = send_p2p
                    ? resources->p2p_fifo->edge_ptr(channel, send_edge)
                    : resources->shm_fifo[channel][send_edge]->device_ptr(rank);
                kernel_args.recv_fifo[channel] = recv_p2p
                    ? resources->p2p_fifo->edge_ptr(channel, recv_edge)
                    : resources->shm_fifo[channel][recv_edge]->device_ptr(rank);
                kernel_args.control.send_head[channel] = send_p2p
                    ? p2p_control.send_head[channel] : shm_control.send_head[channel];
                kernel_args.control.send_tail[channel] = send_p2p
                    ? p2p_control.send_tail[channel] : shm_control.send_tail[channel];
                kernel_args.control.recv_tail[channel] = recv_p2p
                    ? p2p_control.recv_tail[channel] : shm_control.recv_tail[channel];
                kernel_args.control.recv_head[channel] = recv_p2p
                    ? p2p_control.recv_head[channel] : shm_control.recv_head[channel];
            }
            kernel_args.control.base_steps = p2p_control.base_steps != nullptr
                ? p2p_control.base_steps : shm_control.base_steps;

            CUDA_CHECK_THROW(cudaSetDevice(rank));
            ring_simple_kernel<kRanks, T, RedOp::Sum>
                <<<kChannels, kBlockThreads, 0, args.streams[rank]>>>(kernel_args);
            CUDA_CHECK_THROW(cudaGetLastError());
            record_completion(rank, args.streams[rank]);
        }
    }

    void ensure_completion_events() {
        for (int rank = 0; rank < kRanks; ++rank) {
            CUDA_CHECK_THROW(cudaSetDevice(rank));
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
        for (int rank = 0; rank < kRanks; ++rank) {
            fallback_streams_[rank] = streams[rank];
            fallback_in_flight_[rank] = true;
        }
    }

    void release_lifetime_tracking() noexcept {
        for (int rank = 0; rank < kRanks; ++rank) {
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

    template <typename T>
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
                        std::make_unique<transport::p2p::P2pStepCounters>(transport_plan_);
                }
                // Counters are reset once; subsequent launches advance persistent steps.
                reset_control(args.streams);
                control_initialized_ = true;
            }
            launch_ring_simple(args, resources);
        } catch (...) {
            has_untracked_launch_ = true;
            throw;
        }
    }

    std::vector<int> devices_;
    MappedU64Array simple_fifo_steps_;
    MappedU64Array simple_fifo_base_step_;
    std::unique_ptr<transport::p2p::P2pStepCounters> p2p_steps_;
    FifoResources<float> float_resources_;
    FifoResources<__half> float16_resources_;
    FifoResources<__nv_bfloat16> bfloat16_resources_;
    transport::p2p::RingTransportPlan transport_plan_ =
        transport::p2p::RingTransportPlan::uniform(TransportKind::Shm);
    cudaEvent_t completion_events_[kRanks]{};
    cudaStream_t fallback_streams_[kRanks]{};
    bool completion_recorded_[kRanks]{};
    bool fallback_in_flight_[kRanks]{};
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

void Communicator::check_async_error() const {}

int Communicator::local_rank_count() const noexcept { return impl_->local_rank_count(); }

int Communicator::global_rank_count() const noexcept { return impl_->local_rank_count(); }

std::unique_ptr<Communicator> create_communicator(const CommunicatorConfig& config) {
    return std::unique_ptr<Communicator>(new Communicator(std::make_unique<Communicator::Impl>(config)));
}

}  // namespace nano_nccl
