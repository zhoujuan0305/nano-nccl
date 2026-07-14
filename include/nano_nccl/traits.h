#pragma once

#include "nano_nccl/types.h"

#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace nano_nccl {

// dtype trait 将公共 dtype 映射为 CUDA 存储类型和 host-side 转换。
template <DType>
struct DTypeTraits;

template <>
struct DTypeTraits<DType::Float> {
    using type = float;
    static constexpr std::size_t kSize = sizeof(type);
    static constexpr float kDefaultEpsilon = 1e-5f;
    static type from_float(float value) { return value; }
    static float to_float(type value) { return value; }
};

template <>
struct DTypeTraits<DType::Float16> {
    using type = __half;
    static constexpr std::size_t kSize = sizeof(type);
    static constexpr float kDefaultEpsilon = 1e-2f;
    static type from_float(float value) { return __float2half(value); }
    static float to_float(type value) { return __half2float(value); }
};

template <>
struct DTypeTraits<DType::BFloat16> {
    using type = __nv_bfloat16;
    static constexpr std::size_t kSize = sizeof(type);
    static constexpr float kDefaultEpsilon = 1e-1f;
    static type from_float(float value) { return __float2bfloat16(value); }
    static float to_float(type value) { return __bfloat162float(value); }
};

// redop trait：apply 提供 element-wise 规约。当前只 Sum 特化。
template <RedOp, typename T>
struct RedOpTraits;

template <typename T>
struct RedOpTraits<RedOp::Sum, T> {
    static __device__ __forceinline__ T apply(T a, T b) { return a + b; }
};

}  // namespace nano_nccl
