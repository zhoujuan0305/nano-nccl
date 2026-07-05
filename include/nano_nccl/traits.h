#pragma once

#include "nano_nccl/types.h"

#include <cstddef>

namespace nano_nccl {

// dtype trait：当前只 float 特化，未来 half/double 等在此扩展 pack/unpack。
template <DType>
struct DTypeTraits;

template <>
struct DTypeTraits<DType::Float> {
    using type = float;
    static constexpr std::size_t kSize = sizeof(float);
};

// redop trait：apply 提供 element-wise 规约。当前只 Sum 特化。
template <RedOp, typename T>
struct RedOpTraits;

template <typename T>
struct RedOpTraits<RedOp::Sum, T> {
    static __device__ __forceinline__ T apply(T a, T b) { return a + b; }
};

}  // namespace nano_nccl
