#pragma once

#include "nano_nccl/types.h"

#include <cstddef>
#include <cstdint>

namespace nano_nccl::transport {

// FIFO 布局常量，对齐 NCCL Simple 协议：
//   buff 切成 kSimpleFifoSteps 个 step，容量按 storage dtype 计算；
//   一个 chunk 占 kSimpleFifoChunkSteps 个 step，按 kSimpleFifoSliceSteps 粒度收发。
//   一个 chunk 固定分为两个 slice，与 NCCL ProtoSimple<2, 2> 对齐。
constexpr int kSimpleFifoSteps = 8;
constexpr int kSimpleFifoSliceSteps = 2;
constexpr int kSimpleFifoChunkSteps = 4;
constexpr std::size_t kSimpleFifoVectorBytes = 16;

// FIFO buffer 总字节由 CMake 通过 NANO_NCCL_FIFO_BUFF_BYTES 注入，默认 32MiB。
#ifndef NANO_NCCL_FIFO_BUFF_BYTES
#define NANO_NCCL_FIFO_BUFF_BYTES 33554432
#endif
constexpr std::size_t kSimpleFifoBuffBytes = NANO_NCCL_FIFO_BUFF_BYTES;

struct SimpleControlArgs {
    std::uint64_t* send_head[kChannels];
    std::uint64_t* recv_tail[kChannels];
    std::uint64_t* send_tail[kChannels];
    std::uint64_t* recv_head[kChannels];
    std::uint64_t* base_steps;
};

template <typename T>
struct SimpleFifoArgs {
    int rank;
    std::size_t count;
    std::size_t slot_elems;
    std::size_t step_elems;
    const T* input;
    T* output;
    T* send_fifo[kChannels];
    const T* recv_fifo[kChannels];
    SimpleControlArgs control;
};

template <typename T>
struct SimpleChannelArgs {
    std::size_t slot_elems;
    std::size_t step_elems;
    T* send_fifo;
    const T* recv_fifo;
    std::uint64_t* send_head;
    std::uint64_t* recv_tail;
    std::uint64_t* send_tail;
    std::uint64_t* recv_head;
};

}  // namespace nano_nccl::transport
