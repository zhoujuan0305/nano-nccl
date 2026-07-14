#pragma once

#include "nano_nccl/types.h"

#include <cstddef>
#include <cstdint>

namespace nano_nccl::transport::shm {

// SHM FIFO 布局常量，对齐 NCCL Simple 协议：
//   buff 切成 kSimpleFifoSteps 个 step，容量按 storage dtype 计算；
//   一个 chunk 占 kSimpleFifoChunkSteps 个 step，按 kSimpleFifoSliceSteps 粒度收发。
//   SlicePerChunk=1（kSimpleFifoChunkSteps == kSimpleFifoSliceSteps）消除空 slice 屏障。
constexpr int kSimpleFifoSteps = 8;
constexpr int kSimpleFifoSliceSteps = 4;
constexpr int kSimpleFifoChunkSteps = 4;
constexpr std::size_t kSimpleFifoVectorBytes = 16;
// FIFO buffer 总字节由 CMake 通过 NANO_NCCL_FIFO_BUFF_BYTES 注入，默认 32MiB。
#ifndef NANO_NCCL_FIFO_BUFF_BYTES
#define NANO_NCCL_FIFO_BUFF_BYTES 33554432
#endif
constexpr std::size_t kSimpleFifoBuffBytes = NANO_NCCL_FIFO_BUFF_BYTES;

template <typename T>
__host__ __device__ constexpr std::size_t simple_fifo_step_elems() {
    return kSimpleFifoBuffBytes / kSimpleFifoSteps / sizeof(T);
}

// 512 是内存对齐 grain（字节），不是线程数，保持固定。
template <typename T>
__host__ __device__ constexpr std::size_t simple_fifo_grain_elems() {
    return 512 / sizeof(T);
}

// step counter 在 mapped host memory 上，GPU 用 volatile u64 读写。
// kind: 0=send head / recv credit, 1=send ready / recv tail。
__device__ inline int step_idx(int kind, int channel, int edge) {
    return (kind * kChannels + channel) * kRanks + edge;
}

__device__ __forceinline__ std::uint64_t load_step(std::uint64_t* ptr) {
    std::uint64_t ans;
    asm volatile("ld.volatile.global.u64 %0, [%1];"
                 : "=l"(ans)
                 : "l"(__cvta_generic_to_global(ptr))
                 : "memory");
    return ans;
}

__device__ __forceinline__ void store_step(std::uint64_t* ptr,
                                           std::uint64_t value) {
    asm volatile("st.volatile.global.u64 [%0], %1;"
                 ::"l"(__cvta_generic_to_global(ptr)), "l"(value)
                 : "memory");
}

}  // namespace nano_nccl::transport::shm
