#pragma once

#include "nano_nccl/types.h"
#include "transport/simple/geometry.h"

#include <cstddef>

namespace nano_nccl::collective::all_reduce {

template <typename T>
__host__ __device__ constexpr std::size_t simple_grain_elems() {
    return 512 / sizeof(T);
}

__host__ __device__ constexpr int ring_edge_index(int src, int dst, int nranks) {
    return dst == (src + 1) % nranks ? src : -1;
}

template <typename T>
__host__ __device__ constexpr void cbd_part(std::size_t count, int channel,
                                            std::size_t* part_offset,
                                            std::size_t* part_count,
                                            std::size_t* chunk_count) {
    const std::size_t begin = count * static_cast<std::size_t>(channel) / kChannels;
    const std::size_t end = count * static_cast<std::size_t>(channel + 1) / kChannels;
    *part_offset = begin;
    *part_count = end - begin;
    *chunk_count = *part_count == 0
        ? 0
        : transport::simple::align_up(
              transport::simple::div_up(*part_count, kRanks),
              simple_grain_elems<T>());
}

template <typename T>
__host__ __device__ constexpr std::size_t loop_chunk_elems(
    std::size_t chunk_elems, std::size_t slot_elems) {
    return chunk_elems < slot_elems ? chunk_elems : slot_elems;
}

__host__ __device__ constexpr std::size_t nelem(std::size_t chunk_count,
                                                std::size_t rem_count,
                                                std::size_t chunk_offset) {
    if (chunk_offset >= rem_count) return 0;
    const std::size_t rem = rem_count - chunk_offset;
    return rem < chunk_count ? rem : chunk_count;
}

}  // namespace nano_nccl::collective::all_reduce
