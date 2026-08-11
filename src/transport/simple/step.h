#pragma once

#include "transport/simple/protocol.h"

#include <cstdint>

namespace nano_nccl::transport::simple {

template <typename T>
__host__ __device__ constexpr std::size_t step_elems() {
    return kFifoBytes / kFifoSteps / sizeof(T);
}

__device__ __forceinline__ std::uint64_t load_step(std::uint64_t* ptr) {
    std::uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(__cvta_generic_to_global(ptr))
                 : "memory");
    return value;
}

__device__ __forceinline__ void store_step(std::uint64_t* ptr,
                                           std::uint64_t value) {
    asm volatile("st.release.sys.global.u64 [%0], %1;"
                 :: "l"(__cvta_generic_to_global(ptr)), "l"(value)
                 : "memory");
}

__device__ __forceinline__ std::uint32_t load_abort(const std::uint32_t* ptr) {
    std::uint32_t value;
    asm volatile("ld.volatile.global.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(__cvta_generic_to_global(ptr))
                 : "memory");
    return value;
}

}  // namespace nano_nccl::transport::simple
