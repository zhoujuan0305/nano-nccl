#pragma once

#include "transport/simple_protocol.h"
#include "transport/shm/shm_step.h"

#include <cstddef>
#include <cstdint>

namespace nano_nccl::transport::shm {

// SHM 保留 edge-indexed mapped counter backing，kernel 仅接收本 rank 的方向指针。
inline SimpleControlArgs make_simple_control_args(std::uint64_t* steps,
                                                  std::uint64_t* base_steps,
                                                  int rank) {
    SimpleControlArgs control{};
    int send_edge = rank;
    int recv_edge = (rank + kRanks - 1) % kRanks;
    for (int channel = 0; channel < kChannels; ++channel) {
        control.send_head[channel] =
            steps + step_idx(0, channel, send_edge);
        control.recv_tail[channel] =
            steps + step_idx(1, channel, recv_edge);
        control.send_tail[channel] =
            steps + step_idx(1, channel, send_edge);
        control.recv_head[channel] =
            steps + step_idx(0, channel, recv_edge);
    }
    control.base_steps = base_steps + rank * kChannels;
    return control;
}

__host__ __device__ inline std::size_t div_up(std::size_t value,
                                              std::size_t divisor) {
    return (value + divisor - 1) / divisor;
}

__host__ __device__ inline std::size_t align_up(std::size_t value,
                                                std::size_t alignment) {
    return div_up(value, alignment) * alignment;
}

// Ring 拓扑：rank i 发给 rank (i+1)%NRanks，edge 编号即 src rank。
// 非法 src->dst 组合返回 -1。nranks 由调用方传入（kernel 用 NRanks 模板参数，
// host 用 kRanks），保持本 helper 与具体 rank 数解耦。
__host__ __device__ inline int ring_edge_index(int src, int dst, int nranks) {
    if (dst == (src + 1) % nranks) return src;
    return -1;
}

// channel-based 分片：把 count 按 channel 等分，再按 NRanks 切 chunk，
// chunk 大小按 grain 对齐，保证向量访问对齐。
template <typename T>
__host__ __device__ inline void cbd_part(std::size_t count, int channel,
                                         std::size_t* part_offset,
                                         std::size_t* part_count,
                                         std::size_t* chunk_count) {
    std::size_t begin =
        count * static_cast<std::size_t>(channel) / kChannels;
    std::size_t end =
        count * static_cast<std::size_t>(channel + 1) / kChannels;
    *part_offset = begin;
    *part_count = end - begin;
    if (*part_count == 0) {
        *chunk_count = 0;
        return;
    }
    *chunk_count = align_up(div_up(*part_count, kRanks),
                            simple_fifo_grain_elems<T>());
}

// The active FIFO allocation defines the safe ring-loop stride.  Distributed
// socket paths set slot_elems to one Simple step; local paths may grow it.
template <typename T>
__host__ __device__ inline std::size_t simple_fifo_loop_chunk_elems(
    std::size_t chunk_elems, std::size_t slot_elems) {
    return chunk_elems < slot_elems ? chunk_elems : slot_elems;
}

template <typename T>
__device__ inline std::size_t slice_elems(std::size_t nelem,
                                          std::size_t step_elems =
                                              simple_fifo_step_elems<T>()) {
    // slice 取 max(按向量访问对齐, 按 step 容量对齐)，保证既不超 step 容量。
    constexpr std::size_t kVectorElems = kSimpleFifoVectorBytes / sizeof(T);
    std::size_t by_count = div_up(nelem, kVectorElems * kSimpleFifoChunkSteps /
                                             kSimpleFifoSliceSteps) *
                           kVectorElems;
    std::size_t by_step = (step_elems * kSimpleFifoSliceSteps) / 32;
    return by_count > by_step ? by_count : by_step;
}

__device__ inline std::size_t nelem(std::size_t chunk_count,
                                     std::size_t rem_count,
                                     std::size_t chunk_offset) {
    if (chunk_offset >= rem_count) {
        return 0;
    }
    std::size_t rem = rem_count - chunk_offset;
    return rem < chunk_count ? rem : chunk_count;
}

}  // namespace nano_nccl::transport::shm
