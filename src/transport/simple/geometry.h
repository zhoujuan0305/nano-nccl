#pragma once

#include "transport/simple/protocol.h"

#include <cstddef>

namespace nano_nccl::transport::simple {

__host__ __device__ constexpr std::size_t div_up(std::size_t value,
                                                 std::size_t divisor) {
    return (value + divisor - 1) / divisor;
}

__host__ __device__ constexpr std::size_t align_up(std::size_t value,
                                                   std::size_t alignment) {
    return div_up(value, alignment) * alignment;
}

template <typename T>
__host__ __device__ constexpr std::size_t slice_elems(
    std::size_t count, std::size_t nominal_step_elems) {
    constexpr std::size_t kVectorElems = kVectorBytes / sizeof(T);
    const std::size_t by_count =
        div_up(count, kVectorElems * kChunkSteps / kSliceSteps) * kVectorElems;
    const std::size_t by_step = nominal_step_elems * kSliceSteps / 32;
    return by_count > by_step ? by_count : by_step;
}

}  // namespace nano_nccl::transport::simple
