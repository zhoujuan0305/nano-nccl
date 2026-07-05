#pragma once

#include "nano_nccl/traits.h"
#include "transport/shm/shm_fifo.h"
#include "transport/shm/shm_step.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace nano_nccl::kernels {

__device__ __forceinline__ unsigned long long global_address(const void* ptr) {
    return static_cast<unsigned long long>(__cvta_generic_to_global(ptr));
}

__device__ __forceinline__ float4 load_volatile_float4(const float* ptr) {
    float4 v;
    asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(global_address(ptr))
                 : "memory");
    return v;
}

__device__ inline bool aligned_vec4(const float* a, const float* b,
                                    const float* c, std::size_t count) {
    unsigned long long addr =
        reinterpret_cast<unsigned long long>(a) |
        reinterpret_cast<unsigned long long>(b) |
        reinterpret_cast<unsigned long long>(c);
    return (count % 4 == 0) && ((addr & 0xfULL) == 0);
}

__device__ inline void copy_worker(const float* src, float* dst,
                                    std::size_t count, int nworkers) {
    if (threadIdx.x >= nworkers) return;
    if (aligned_vec4(src, dst, dst, count)) {
        std::size_t vec_count = count / 4;
        const float4* src4 = reinterpret_cast<const float4*>(src);
        float4* dst4 = reinterpret_cast<float4*>(dst);
        for (std::size_t i = threadIdx.x; i < vec_count; i += nworkers) {
            dst4[i] = src4[i];
        }
        return;
    }
    for (std::size_t i = threadIdx.x; i < count; i += nworkers) {
        dst[i] = src[i];
    }
}

__device__ inline void copy_volatile_worker(const float* src, float* dst,
                                             std::size_t count, int nworkers) {
    if (threadIdx.x >= nworkers) return;
    if (aligned_vec4(src, dst, dst, count)) {
        std::size_t vec_count = count / 4;
        float4* dst4 = reinterpret_cast<float4*>(dst);
        for (std::size_t i = threadIdx.x; i < vec_count; i += nworkers) {
            dst4[i] = load_volatile_float4(src + 4 * i);
        }
        return;
    }
    volatile const float* src_v = reinterpret_cast<volatile const float*>(src);
    for (std::size_t i = threadIdx.x; i < count; i += nworkers) {
        dst[i] = src_v[i];
    }
}

// local + recv 规约后写 dst。reduce 通过 RedOpTraits，未来扩展 max/min 直接换 trait。
template <int NRanks, typename T, RedOp kRedOp>
__device__ inline void reduce_volatile_worker(const float* local,
                                              const float* recv, float* dst,
                                              std::size_t count,
                                              int nworkers) {
    if (threadIdx.x >= nworkers) return;
    if (aligned_vec4(local, recv, dst, count)) {
        std::size_t vec_count = count / 4;
        const float4* local4 = reinterpret_cast<const float4*>(local);
        float4* dst4 = reinterpret_cast<float4*>(dst);
        for (std::size_t i = threadIdx.x; i < vec_count; i += nworkers) {
            float4 a = local4[i];
            float4 b = load_volatile_float4(recv + 4 * i);
            dst4[i] = make_float4(
                RedOpTraits<kRedOp, T>::apply(a.x, b.x),
                RedOpTraits<kRedOp, T>::apply(a.y, b.y),
                RedOpTraits<kRedOp, T>::apply(a.z, b.z),
                RedOpTraits<kRedOp, T>::apply(a.w, b.w));
        }
        return;
    }
    volatile const float* recv_v = reinterpret_cast<volatile const float*>(recv);
    for (std::size_t i = threadIdx.x; i < count; i += nworkers) {
        dst[i] = RedOpTraits<kRedOp, T>::apply(local[i], recv_v[i]);
    }
}

template <int NRanks, typename T, RedOp kRedOp>
__device__ inline void copy_broadcast_volatile_worker(
    const float* src, float* dst0, float* dst1, std::size_t count,
    int nworkers) {
    if (threadIdx.x >= nworkers) return;
    if (aligned_vec4(src, dst0, dst1, count)) {
        std::size_t vec_count = count / 4;
        float4* dst04 = reinterpret_cast<float4*>(dst0);
        float4* dst14 = reinterpret_cast<float4*>(dst1);
        for (std::size_t i = threadIdx.x; i < vec_count; i += nworkers) {
            float4 v = load_volatile_float4(src + 4 * i);
            dst04[i] = v;
            dst14[i] = v;
        }
        return;
    }
    volatile const float* src_v = reinterpret_cast<volatile const float*>(src);
    for (std::size_t i = threadIdx.x; i < count; i += nworkers) {
        float v = src_v[i];
        dst0[i] = v;
        dst1[i] = v;
    }
}

template <int NRanks, typename T, RedOp kRedOp>
__device__ inline void reduce_broadcast_volatile_worker(
    const float* local, const float* recv, float* dst0, float* dst1,
    std::size_t count, int nworkers) {
    if (threadIdx.x >= nworkers) return;
    if (aligned_vec4(local, recv, dst0, count) &&
        aligned_vec4(local, recv, dst1, count)) {
        std::size_t vec_count = count / 4;
        const float4* local4 = reinterpret_cast<const float4*>(local);
        float4* dst04 = reinterpret_cast<float4*>(dst0);
        float4* dst14 = reinterpret_cast<float4*>(dst1);
        for (std::size_t i = threadIdx.x; i < vec_count; i += nworkers) {
            float4 a = local4[i];
            float4 b = load_volatile_float4(recv + 4 * i);
            float4 v = make_float4(
                RedOpTraits<kRedOp, T>::apply(a.x, b.x),
                RedOpTraits<kRedOp, T>::apply(a.y, b.y),
                RedOpTraits<kRedOp, T>::apply(a.z, b.z),
                RedOpTraits<kRedOp, T>::apply(a.w, b.w));
            dst04[i] = v;
            dst14[i] = v;
        }
        return;
    }
    volatile const float* recv_v = reinterpret_cast<volatile const float*>(recv);
    for (std::size_t i = threadIdx.x; i < count; i += nworkers) {
        float v = RedOpTraits<kRedOp, T>::apply(local[i], recv_v[i]);
        dst0[i] = v;
        dst1[i] = v;
    }
}

// 等待 send 方有可用 slot（credit）。专用 threadIdx==1 轮询，对齐 NCCL connStepCache。
template <int NRanks, typename T>
__device__ inline void wait_send_credit(
    transport::shm::ShmFifoArgs<T> args, int channel, std::uint64_t step,
    std::uint64_t* head_cache) {
    int edge = transport::shm::ring_edge_index(args.rank,
                                               (args.rank + 1) % NRanks, NRanks);
    std::uint64_t* head =
        args.steps + transport::shm::step_idx(0, channel, edge);
    if (threadIdx.x == 1) {
        while (*head_cache + transport::shm::kSimpleFifoSteps <
               step + transport::shm::kSimpleFifoSliceSteps) {
            *head_cache = transport::shm::load_step(head);
        }
    }
}

template <int NRanks, typename T>
__device__ inline void wait_recv_ready(
    transport::shm::ShmFifoArgs<T> args, int channel, std::uint64_t step,
    std::uint64_t* tail_cache) {
    int prev = (args.rank + NRanks - 1) % NRanks;
    int edge = transport::shm::ring_edge_index(prev, args.rank, NRanks);
    std::uint64_t* tail =
        args.steps + transport::shm::step_idx(1, channel, edge);
    if (threadIdx.x == 0) {
        while (*tail_cache < step + transport::shm::kSimpleFifoSliceSteps) {
            *tail_cache = transport::shm::load_step(tail);
        }
    }
}

template <int NRanks, typename T>
__device__ inline void post_send_ready(
    transport::shm::ShmFifoArgs<T> args, int channel, std::uint64_t step,
    bool data_stored) {
    int edge = transport::shm::ring_edge_index(args.rank,
                                               (args.rank + 1) % NRanks, NRanks);
    std::uint64_t* tail =
        args.steps + transport::shm::step_idx(1, channel, edge);
    if (threadIdx.x == blockDim.x - 1) {
        if (data_stored) {
            __threadfence_system();
        }
        transport::shm::store_step(tail, step + transport::shm::kSimpleFifoSliceSteps);
    }
}

template <int NRanks, typename T>
__device__ inline void post_recv_credit(
    transport::shm::ShmFifoArgs<T> args, int channel, std::uint64_t step) {
    int prev = (args.rank + NRanks - 1) % NRanks;
    int edge = transport::shm::ring_edge_index(prev, args.rank, NRanks);
    std::uint64_t* head =
        args.steps + transport::shm::step_idx(0, channel, edge);
    if (threadIdx.x == blockDim.x - 2) {
        transport::shm::store_step(head, step + transport::shm::kSimpleFifoSliceSteps);
    }
}

template <int NRanks, typename T, RedOp kRedOp>
__device__ inline void direct_send(
    transport::shm::ShmFifoArgs<T> args, int channel, const float* src,
    std::size_t nelem, std::uint64_t* send_step,
    std::uint64_t* send_head_cache, int nworkers) {
    std::size_t slice_size = transport::shm::slice_elems(nelem, args.step_elems);
    std::size_t slice_offset = 0;
    for (int slice = 0;
         slice < transport::shm::kSimpleFifoChunkSteps / transport::shm::kSimpleFifoSliceSteps;
         ++slice) {
        std::size_t work =
            slice_offset < nelem
                ? transport::shm::nelem(slice_size, nelem, slice_offset)
                : 0;
        wait_send_credit<NRanks, T>(args, channel, *send_step, send_head_cache);
        __syncthreads();
        float* dst = args.send_fifo[channel] +
                     ((*send_step % transport::shm::kSimpleFifoSteps) *
                      args.slot_elems);
        if (work != 0) {
            copy_worker(src + slice_offset, dst, work, nworkers);
        }
        __syncthreads();
        post_send_ready<NRanks, T>(args, channel, *send_step, work != 0);
        *send_step += transport::shm::kSimpleFifoSliceSteps;
        slice_offset += slice_size;
    }
}

template <int NRanks, typename T, RedOp kRedOp>
__device__ inline void recv_reduce_send(
    transport::shm::ShmFifoArgs<T> args, int channel, const float* local,
    std::size_t nelem, std::uint64_t* recv_step, std::uint64_t* send_step,
    std::uint64_t* recv_tail_cache, std::uint64_t* send_head_cache,
    int nworkers) {
    std::size_t slice_size = transport::shm::slice_elems(nelem, args.step_elems);
    std::size_t slice_offset = 0;
    for (int slice = 0;
         slice < transport::shm::kSimpleFifoChunkSteps / transport::shm::kSimpleFifoSliceSteps;
         ++slice) {
        std::size_t work =
            slice_offset < nelem
                ? transport::shm::nelem(slice_size, nelem, slice_offset)
                : 0;
        wait_recv_ready<NRanks, T>(args, channel, *recv_step, recv_tail_cache);
        wait_send_credit<NRanks, T>(args, channel, *send_step, send_head_cache);
        __syncthreads();
        const float* recv = args.recv_fifo[channel] +
                            ((*recv_step % transport::shm::kSimpleFifoSteps) *
                             args.slot_elems);
        float* dst = args.send_fifo[channel] +
                     ((*send_step % transport::shm::kSimpleFifoSteps) *
                      args.slot_elems);
        if (work != 0) {
            reduce_volatile_worker<NRanks, T, kRedOp>(local + slice_offset, recv,
                                                     dst, work, nworkers);
        }
        __syncthreads();
        post_recv_credit<NRanks, T>(args, channel, *recv_step);
        post_send_ready<NRanks, T>(args, channel, *send_step, work != 0);
        *recv_step += transport::shm::kSimpleFifoSliceSteps;
        *send_step += transport::shm::kSimpleFifoSliceSteps;
        slice_offset += slice_size;
    }
}

template <int NRanks, typename T, RedOp kRedOp>
__device__ inline void recv_reduce_copy_send(
    transport::shm::ShmFifoArgs<T> args, int channel, const float* local,
    float* out, std::size_t nelem, std::uint64_t* recv_step,
    std::uint64_t* send_step, std::uint64_t* recv_tail_cache,
    std::uint64_t* send_head_cache, int nworkers) {
    std::size_t slice_size = transport::shm::slice_elems(nelem, args.step_elems);
    std::size_t slice_offset = 0;
    for (int slice = 0;
         slice < transport::shm::kSimpleFifoChunkSteps / transport::shm::kSimpleFifoSliceSteps;
         ++slice) {
        std::size_t work =
            slice_offset < nelem
                ? transport::shm::nelem(slice_size, nelem, slice_offset)
                : 0;
        wait_recv_ready<NRanks, T>(args, channel, *recv_step, recv_tail_cache);
        wait_send_credit<NRanks, T>(args, channel, *send_step, send_head_cache);
        __syncthreads();
        const float* recv = args.recv_fifo[channel] +
                            ((*recv_step % transport::shm::kSimpleFifoSteps) *
                             args.slot_elems);
        float* dst = args.send_fifo[channel] +
                     ((*send_step % transport::shm::kSimpleFifoSteps) *
                      args.slot_elems);
        if (work != 0) {
            reduce_broadcast_volatile_worker<NRanks, T, kRedOp>(
                local + slice_offset, recv, out + slice_offset, dst, work,
                nworkers);
        }
        __syncthreads();
        post_recv_credit<NRanks, T>(args, channel, *recv_step);
        post_send_ready<NRanks, T>(args, channel, *send_step, work != 0);
        *recv_step += transport::shm::kSimpleFifoSliceSteps;
        *send_step += transport::shm::kSimpleFifoSliceSteps;
        slice_offset += slice_size;
    }
}

template <int NRanks, typename T, RedOp kRedOp>
__device__ inline void recv_copy_send(
    transport::shm::ShmFifoArgs<T> args, int channel, float* out,
    std::size_t nelem, std::uint64_t* recv_step, std::uint64_t* send_step,
    std::uint64_t* recv_tail_cache, std::uint64_t* send_head_cache,
    int nworkers) {
    std::size_t slice_size = transport::shm::slice_elems(nelem, args.step_elems);
    std::size_t slice_offset = 0;
    for (int slice = 0;
         slice < transport::shm::kSimpleFifoChunkSteps / transport::shm::kSimpleFifoSliceSteps;
         ++slice) {
        std::size_t work =
            slice_offset < nelem
                ? transport::shm::nelem(slice_size, nelem, slice_offset)
                : 0;
        wait_recv_ready<NRanks, T>(args, channel, *recv_step, recv_tail_cache);
        wait_send_credit<NRanks, T>(args, channel, *send_step, send_head_cache);
        __syncthreads();
        const float* recv = args.recv_fifo[channel] +
                            ((*recv_step % transport::shm::kSimpleFifoSteps) *
                             args.slot_elems);
        float* dst = args.send_fifo[channel] +
                     ((*send_step % transport::shm::kSimpleFifoSteps) *
                      args.slot_elems);
        if (work != 0) {
            copy_broadcast_volatile_worker<NRanks, T, kRedOp>(
                recv, out + slice_offset, dst, work, nworkers);
        }
        __syncthreads();
        post_recv_credit<NRanks, T>(args, channel, *recv_step);
        post_send_ready<NRanks, T>(args, channel, *send_step, work != 0);
        *recv_step += transport::shm::kSimpleFifoSliceSteps;
        *send_step += transport::shm::kSimpleFifoSliceSteps;
        slice_offset += slice_size;
    }
}

template <int NRanks, typename T, RedOp kRedOp>
__device__ inline void direct_recv(
    transport::shm::ShmFifoArgs<T> args, int channel, float* out,
    std::size_t nelem, std::uint64_t* recv_step,
    std::uint64_t* recv_tail_cache, int nworkers) {
    std::size_t slice_size = transport::shm::slice_elems(nelem, args.step_elems);
    std::size_t slice_offset = 0;
    for (int slice = 0;
         slice < transport::shm::kSimpleFifoChunkSteps / transport::shm::kSimpleFifoSliceSteps;
         ++slice) {
        std::size_t work =
            slice_offset < nelem
                ? transport::shm::nelem(slice_size, nelem, slice_offset)
                : 0;
        wait_recv_ready<NRanks, T>(args, channel, *recv_step, recv_tail_cache);
        __syncthreads();
        const float* recv = args.recv_fifo[channel] +
                            ((*recv_step % transport::shm::kSimpleFifoSteps) *
                             args.slot_elems);
        if (work != 0) {
            copy_volatile_worker(recv, out + slice_offset, work, nworkers);
        }
        __syncthreads();
        post_recv_credit<NRanks, T>(args, channel, *recv_step);
        *recv_step += transport::shm::kSimpleFifoSliceSteps;
        slice_offset += slice_size;
    }
}

// Ring + Simple 协议主 kernel。
//   NRanks / T / RedOp 为编译期参数，内循环按 ring 位置展开，无虚调用。
//   base_steps 跨迭代持久化（对齐 NCCL conn->step），避免每轮重置 step counter。
//   nworkers = blockDim - 32：留 3 个专用线程做 wait/post，其余做数据搬运。
template <int NRanks, typename T, RedOp kRedOp>
__global__ __launch_bounds__(NANO_NCCL_BLOCK_THREADS, 1) void ring_simple_kernel(
    transport::shm::ShmFifoArgs<T> args) {
    int channel = blockIdx.x;
    if (channel >= kChannels) {
        return;
    }
    int nworkers = blockDim.x >= 3 * 32 ? blockDim.x - 32 : blockDim.x;
    std::size_t part_offset = 0;
    std::size_t part_count = 0;
    std::size_t chunk_count = 0;
    transport::shm::cbd_part(args.count, channel, &part_offset, &part_count,
                             &chunk_count);
    if (part_count == 0 || chunk_count == 0) {
        return;
    }

    __shared__ std::uint64_t s_base_step;
    if (threadIdx.x == 0) {
        s_base_step = args.base_steps[channel];
    }
    __syncthreads();
    std::uint64_t base_step = s_base_step;
    std::uint64_t send_step = base_step;
    std::uint64_t recv_step = base_step;
    std::uint64_t send_head_cache = base_step;
    std::uint64_t recv_tail_cache = base_step;
    int ring_ix = args.rank;

    for (std::size_t elem_offset = 0; elem_offset < part_count;) {
        std::size_t rem_count = part_count - elem_offset;
        std::size_t loop_chunk = chunk_count;
        if (rem_count < NRanks * loop_chunk) {
            loop_chunk = transport::shm::align_up(
                transport::shm::div_up(rem_count, NRanks), 16 / sizeof(float));
        }
        std::size_t loop_count = NRanks * loop_chunk;

        int chunk = (ring_ix + NRanks - 1) % NRanks;
        std::size_t chunk_offset = static_cast<std::size_t>(chunk) * loop_chunk;
        std::size_t offset = part_offset + elem_offset + chunk_offset;
        std::size_t work = transport::shm::nelem(loop_chunk, rem_count, chunk_offset);
        direct_send<NRanks, T, kRedOp>(args, channel, args.input + offset, work,
                                      &send_step, &send_head_cache, nworkers);

        for (int j = 2; j < NRanks; ++j) {
            chunk = (ring_ix + NRanks - j) % NRanks;
            chunk_offset = static_cast<std::size_t>(chunk) * loop_chunk;
            offset = part_offset + elem_offset + chunk_offset;
            work = transport::shm::nelem(loop_chunk, rem_count, chunk_offset);
            recv_reduce_send<NRanks, T, kRedOp>(
                args, channel, args.input + offset, work, &recv_step,
                &send_step, &recv_tail_cache, &send_head_cache, nworkers);
        }

        chunk = ring_ix;
        chunk_offset = static_cast<std::size_t>(chunk) * loop_chunk;
        offset = part_offset + elem_offset + chunk_offset;
        work = transport::shm::nelem(loop_chunk, rem_count, chunk_offset);
        recv_reduce_copy_send<NRanks, T, kRedOp>(
            args, channel, args.input + offset, args.output + offset, work,
            &recv_step, &send_step, &recv_tail_cache, &send_head_cache,
            nworkers);

        for (int j = 1; j < NRanks - 1; ++j) {
            chunk = (ring_ix + NRanks - j) % NRanks;
            chunk_offset = static_cast<std::size_t>(chunk) * loop_chunk;
            offset = part_offset + elem_offset + chunk_offset;
            work = transport::shm::nelem(loop_chunk, rem_count, chunk_offset);
            recv_copy_send<NRanks, T, kRedOp>(
                args, channel, args.output + offset, work, &recv_step,
                &send_step, &recv_tail_cache, &send_head_cache, nworkers);
        }

        chunk = (ring_ix + 1) % NRanks;
        chunk_offset = static_cast<std::size_t>(chunk) * loop_chunk;
        offset = part_offset + elem_offset + chunk_offset;
        work = transport::shm::nelem(loop_chunk, rem_count, chunk_offset);
        direct_recv<NRanks, T, kRedOp>(args, channel, args.output + offset, work,
                                      &recv_step, &recv_tail_cache, nworkers);

        elem_offset += loop_count;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        args.base_steps[channel] = send_step;
    }
}

}  // namespace nano_nccl::kernels
