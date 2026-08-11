#pragma once

#include "nano_nccl/types.h"

#include <cstddef>
#include <cstdint>

#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
namespace nano_nccl::collective::all_reduce {
struct RingTurnaroundStats;
}
#endif

namespace nano_nccl::transport::simple {

constexpr int kFifoSteps = 8;
constexpr int kSliceSteps = 2;
constexpr int kChunkSteps = 4;
constexpr std::size_t kVectorBytes = 16;

#ifndef NANO_NCCL_FIFO_BUFF_BYTES
#define NANO_NCCL_FIFO_BUFF_BYTES 33554432
#endif
constexpr std::size_t kFifoBytes = NANO_NCCL_FIFO_BUFF_BYTES;

struct ControlArgs {
    std::uint64_t* send_head[kChannels];
    std::uint64_t* recv_tail[kChannels];
    std::uint64_t* send_tail[kChannels];
    std::uint64_t* recv_head[kChannels];
    std::uint64_t* base_steps;
};

template <typename T>
struct FifoArgs {
    int rank;
    std::size_t count;
    std::size_t slot_elems;
    std::size_t step_elems;
    const T* input;
    T* output;
    T* send_fifo[kChannels];
    const T* recv_fifo[kChannels];
    std::uint32_t* send_payload_bytes[kChannels];
    const std::uint32_t* recv_payload_bytes[kChannels];
    const std::uint32_t* abort;
    ControlArgs control;
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    collective::all_reduce::RingTurnaroundStats* turnaround[kChannels];
#endif
};

template <typename T>
struct ChannelArgs {
    std::size_t slot_elems;
    std::size_t step_elems;
    T* send_fifo;
    const T* recv_fifo;
    std::uint64_t* send_head;
    std::uint64_t* recv_tail;
    std::uint64_t* send_tail;
    std::uint64_t* recv_head;
    std::uint32_t* send_payload_bytes;
    const std::uint32_t* recv_payload_bytes;
    const std::uint32_t* abort;
    std::uint32_t* wait_observer;
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    collective::all_reduce::RingTurnaroundStats* turnaround;
#endif
};

}  // namespace nano_nccl::transport::simple
