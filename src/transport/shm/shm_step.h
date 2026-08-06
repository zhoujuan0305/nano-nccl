#pragma once

#include "nano_nccl/types.h"
#include "transport/simple_protocol.h"

#include <cstdint>

namespace nano_nccl::transport::shm {

using ::nano_nccl::transport::kSimpleFifoBuffBytes;
using ::nano_nccl::transport::kSimpleFifoChunkSteps;
using ::nano_nccl::transport::kSimpleFifoSliceSteps;
using ::nano_nccl::transport::kSimpleFifoSteps;
using ::nano_nccl::transport::kSimpleFifoVectorBytes;

template <typename T>
__host__ __device__ constexpr std::size_t simple_fifo_step_elems() {
    return kSimpleFifoBuffBytes / kSimpleFifoSteps / sizeof(T);
}

// 512 是内存对齐 grain（字节），不是线程数，保持固定。
template <typename T>
__host__ __device__ constexpr std::size_t simple_fifo_grain_elems() {
    return 512 / sizeof(T);
}

// step counter 在 mapped host memory 上；sm_70+ 用 system release/acquire，
// 保证 GPU FIFO 写对 host proxy 可见后再推进 send_tail。
// kind: 0=send head / recv credit, 1=send ready / recv tail。
__host__ __device__ inline int step_idx(int kind, int channel, int edge) {
    return (kind * kChannels + channel) * kRanks + edge;
}

__device__ __forceinline__ std::uint64_t load_step(std::uint64_t* ptr) {
    std::uint64_t ans;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                 : "=l"(ans)
                 : "l"(__cvta_generic_to_global(ptr))
                 : "memory");
#else
    asm volatile("ld.volatile.global.u64 %0, [%1];"
                 : "=l"(ans)
                 : "l"(__cvta_generic_to_global(ptr))
                 : "memory");
#endif
    return ans;
}

__device__ __forceinline__ void store_step(std::uint64_t* ptr,
                                           std::uint64_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u64 [%0], %1;"
                 ::"l"(__cvta_generic_to_global(ptr)), "l"(value)
                 : "memory");
#else
    asm volatile("st.volatile.global.u64 [%0], %1;"
                 ::"l"(__cvta_generic_to_global(ptr)), "l"(value)
                 : "memory");
#endif
}

__device__ __forceinline__ std::uint32_t load_abort(const std::uint32_t* ptr) {
    std::uint32_t ans;
    asm volatile("ld.volatile.global.u32 %0, [%1];"
                 : "=r"(ans)
                 : "l"(__cvta_generic_to_global(ptr))
                 : "memory");
    return ans;
}

}  // namespace nano_nccl::transport::shm
