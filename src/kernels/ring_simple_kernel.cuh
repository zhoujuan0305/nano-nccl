#pragma once

#include "nano_nccl/traits.h"
#include "transport/shm/shm_fifo.h"
#include "transport/shm/shm_step.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

namespace nano_nccl {

template <>
struct RedOpTraits<RedOp::Sum, __half> {
    static __device__ __forceinline__ __half apply(__half a, __half b) {
        return __hadd(a, b);
    }
};

template <>
struct RedOpTraits<RedOp::Sum, __nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 apply(__nv_bfloat16 a,
                                                          __nv_bfloat16 b) {
        return __hadd(a, b);
    }
};

template <>
struct RedOpTraits<RedOp::Avg, __half> {
    static __device__ __forceinline__ __half apply(__half a, __half b) {
        return __hadd(a, b);
    }
};

template <>
struct RedOpTraits<RedOp::Avg, __nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 apply(__nv_bfloat16 a,
                                                           __nv_bfloat16 b) {
        return __hadd(a, b);
    }
};

}  // namespace nano_nccl

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

__device__ __forceinline__ uint4 load_volatile_uint4(const void* ptr) {
    uint4 value;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(global_address(ptr))
                 : "memory");
    return value;
}

__device__ inline bool aligned_vec4(const float* a, const float* b,
                                    const float* c, std::size_t count) {
    unsigned long long addr =
        reinterpret_cast<unsigned long long>(a) |
        reinterpret_cast<unsigned long long>(b) |
        reinterpret_cast<unsigned long long>(c);
    return (count % 4 == 0) && ((addr & 0xfULL) == 0);
}

template <typename T>
__device__ inline bool aligned_packed16(const T* a, const T* b, const T* c,
                                        std::size_t count) {
    unsigned long long address = reinterpret_cast<unsigned long long>(a) |
                                 reinterpret_cast<unsigned long long>(b) |
                                 reinterpret_cast<unsigned long long>(c);
    return count % 8 == 0 && (address & 0xfULL) == 0;
}

template <typename T>
inline constexpr bool kIsPacked16 =
    std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;

__device__ __forceinline__ __half2 half2_from_bits(std::uint32_t bits) {
    __half2_raw raw;
    raw.x = static_cast<unsigned short>(bits);
    raw.y = static_cast<unsigned short>(bits >> 16);
    return __half2(raw);
}

__device__ __forceinline__ std::uint32_t half2_to_bits(__half2 value) {
    __half2_raw raw = static_cast<__half2_raw>(value);
    return static_cast<std::uint32_t>(raw.x) |
           (static_cast<std::uint32_t>(raw.y) << 16);
}

__device__ __forceinline__ __nv_bfloat162 bfloat162_from_bits(std::uint32_t bits) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(bits);
    raw.y = static_cast<unsigned short>(bits >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ std::uint32_t bfloat162_to_bits(__nv_bfloat162 value) {
    __nv_bfloat162_raw raw = static_cast<__nv_bfloat162_raw>(value);
    return static_cast<std::uint32_t>(raw.x) |
           (static_cast<std::uint32_t>(raw.y) << 16);
}

template <typename T, RedOp kRedOp>
struct Packed16Traits;

template <RedOp kRedOp>
struct Packed16Traits<__half, kRedOp> {
    static __device__ __forceinline__ std::uint32_t apply(std::uint32_t a,
                                                           std::uint32_t b) {
        __half2 lhs = half2_from_bits(a);
        __half2 rhs = half2_from_bits(b);
        if constexpr (kRedOp == RedOp::Sum || kRedOp == RedOp::Avg) {
            return half2_to_bits(__hadd2(lhs, rhs));
        }
        return half2_to_bits(__halves2half2(
            RedOpTraits<kRedOp, __half>::apply(__low2half(lhs), __low2half(rhs)),
            RedOpTraits<kRedOp, __half>::apply(__high2half(lhs), __high2half(rhs))));
    }
};

template <RedOp kRedOp>
struct Packed16Traits<__nv_bfloat16, kRedOp> {
    static __device__ __forceinline__ std::uint32_t apply(std::uint32_t a,
                                                           std::uint32_t b) {
        __nv_bfloat162 lhs = bfloat162_from_bits(a);
        __nv_bfloat162 rhs = bfloat162_from_bits(b);
        if constexpr (kRedOp == RedOp::Sum || kRedOp == RedOp::Avg) {
            return bfloat162_to_bits(__hadd2(lhs, rhs));
        }
        return bfloat162_to_bits(__halves2bfloat162(
            RedOpTraits<kRedOp, __nv_bfloat16>::apply(
                __low2bfloat16(lhs), __low2bfloat16(rhs)),
            RedOpTraits<kRedOp, __nv_bfloat16>::apply(
                __high2bfloat16(lhs), __high2bfloat16(rhs))));
    }
};

template <typename T, RedOp kRedOp>
__device__ __forceinline__ uint4 reduce_packed16(uint4 local, uint4 recv) {
    return make_uint4(Packed16Traits<T, kRedOp>::apply(local.x, recv.x),
                      Packed16Traits<T, kRedOp>::apply(local.y, recv.y),
                      Packed16Traits<T, kRedOp>::apply(local.z, recv.z),
                      Packed16Traits<T, kRedOp>::apply(local.w, recv.w));
}

template <typename T>
__device__ inline void copy_packed16_worker(const T* src, T* dst,
                                             std::size_t count, int nworkers) {
    if (threadIdx.x >= nworkers) return;
    std::size_t vec_count = count / 8;
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    for (std::size_t i = threadIdx.x; i < vec_count; i += nworkers) {
        dst4[i] = src4[i];
    }
}

template <typename T>
__device__ inline void copy_volatile_packed16_worker(
    const T* src, T* dst, std::size_t count, int nworkers) {
    if (threadIdx.x >= nworkers) return;
    std::size_t vec_count = count / 8;
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    for (std::size_t i = threadIdx.x; i < vec_count; i += nworkers) {
        dst4[i] = load_volatile_uint4(src + 8 * i);
    }
}

template <typename T, RedOp kRedOp>
__device__ inline void reduce_volatile_packed16_worker(
    const T* local, const T* recv, T* dst, std::size_t count, int nworkers) {
    if (threadIdx.x >= nworkers) return;
    std::size_t vec_count = count / 8;
    const uint4* local4 = reinterpret_cast<const uint4*>(local);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    for (std::size_t i = threadIdx.x; i < vec_count; i += nworkers) {
        dst4[i] = reduce_packed16<T, kRedOp>(local4[i],
                                              load_volatile_uint4(recv + 8 * i));
    }
}

template <typename T>
__device__ inline void copy_broadcast_volatile_packed16_worker(
    const T* src, T* dst0, T* dst1, std::size_t count, int nworkers) {
    if (threadIdx.x >= nworkers) return;
    std::size_t vec_count = count / 8;
    uint4* dst04 = reinterpret_cast<uint4*>(dst0);
    uint4* dst14 = reinterpret_cast<uint4*>(dst1);
    for (std::size_t i = threadIdx.x; i < vec_count; i += nworkers) {
        uint4 value = load_volatile_uint4(src + 8 * i);
        dst04[i] = value;
        dst14[i] = value;
    }
}

template <typename T, RedOp kRedOp>
__device__ inline void reduce_broadcast_volatile_packed16_worker(
    const T* local, const T* recv, T* dst0, T* dst1,
    std::size_t count, int nworkers) {
    if (threadIdx.x >= nworkers) return;
    std::size_t vec_count = count / 8;
    const uint4* local4 = reinterpret_cast<const uint4*>(local);
    uint4* dst04 = reinterpret_cast<uint4*>(dst0);
    uint4* dst14 = reinterpret_cast<uint4*>(dst1);
    for (std::size_t i = threadIdx.x; i < vec_count; i += nworkers) {
        uint4 value = reduce_packed16<T, kRedOp>(local4[i],
                                                  load_volatile_uint4(recv + 8 * i));
        dst04[i] = value;
        dst14[i] = value;
    }
}

__device__ inline void copy_float_worker(const float* src, float* dst,
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

__device__ inline void copy_volatile_float_worker(const float* src, float* dst,
                                                   std::size_t count,
                                                   int nworkers) {
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

template <typename T>
__device__ inline void copy_scalar_worker(const T* src, T* dst,
                                          std::size_t count, int nworkers) {
    if (threadIdx.x >= nworkers) return;
    volatile const T* src_v = reinterpret_cast<volatile const T*>(src);
    for (std::size_t i = threadIdx.x; i < count; i += nworkers) dst[i] = src_v[i];
}

template <typename T>
__device__ inline void copy_worker(const T* src, T* dst, std::size_t count,
                                   int nworkers) {
    if constexpr (std::is_same_v<T, float>) {
        copy_float_worker(src, dst, count, nworkers);
    } else if constexpr (kIsPacked16<T>) {
        if (aligned_packed16(src, dst, dst, count)) {
            copy_packed16_worker(src, dst, count, nworkers);
        } else {
            copy_scalar_worker(src, dst, count, nworkers);
        }
    } else {
        copy_scalar_worker(src, dst, count, nworkers);
    }
}

template <typename T>
__device__ inline void copy_volatile_worker(const T* src, T* dst,
                                             std::size_t count, int nworkers) {
    if constexpr (std::is_same_v<T, float>) {
        copy_volatile_float_worker(src, dst, count, nworkers);
    } else if constexpr (kIsPacked16<T>) {
        if (aligned_packed16(src, dst, dst, count)) {
            copy_volatile_packed16_worker(src, dst, count, nworkers);
        } else {
            copy_scalar_worker(src, dst, count, nworkers);
        }
    } else {
        copy_scalar_worker(src, dst, count, nworkers);
    }
}

template <typename T>
__device__ __forceinline__ T scale_avg(T value, float inverse_nranks) {
    return value * inverse_nranks;
}

template <>
__device__ __forceinline__ __half scale_avg(__half value, float inverse_nranks) {
    return __hmul(value, __float2half(inverse_nranks));
}

template <>
__device__ __forceinline__ __nv_bfloat16 scale_avg(
    __nv_bfloat16 value, float inverse_nranks) {
    return __hmul(value, __float2bfloat16(inverse_nranks));
}

template <typename T>
__device__ inline void scale_avg_worker(T* output, std::size_t count,
                                        float inverse_nranks, int nworkers) {
    if (threadIdx.x >= nworkers) return;
    for (std::size_t index = threadIdx.x; index < count; index += nworkers) {
        output[index] = scale_avg(output[index], inverse_nranks);
    }
}

// local + recv 规约后写 dst。reduce 通过 RedOpTraits，未来扩展 max/min 直接换 trait。
template <typename T, RedOp kRedOp>
__device__ inline void reduce_volatile_worker(const T* local, const T* recv,
                                              T* dst, std::size_t count,
                                              int nworkers) {
    if (threadIdx.x >= nworkers) return;
    if constexpr (std::is_same_v<T, float>) {
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
    } else if constexpr (kIsPacked16<T>) {
        if (aligned_packed16(local, recv, dst, count)) {
            reduce_volatile_packed16_worker<T, kRedOp>(local, recv, dst, count,
                                                        nworkers);
            return;
        }
    }
    volatile const T* recv_v = reinterpret_cast<volatile const T*>(recv);
    for (std::size_t i = threadIdx.x; i < count; i += nworkers) {
        dst[i] = RedOpTraits<kRedOp, T>::apply(local[i], recv_v[i]);
    }
}

template <typename T, RedOp kRedOp>
__device__ inline void copy_broadcast_volatile_worker(
    const T* src, T* dst0, T* dst1, std::size_t count,
    int nworkers) {
    if (threadIdx.x >= nworkers) return;
    if constexpr (std::is_same_v<T, float>) {
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
    } else if constexpr (kIsPacked16<T>) {
        if (aligned_packed16(src, dst0, dst1, count)) {
            copy_broadcast_volatile_packed16_worker(src, dst0, dst1, count,
                                                     nworkers);
            return;
        }
    }
    volatile const T* src_v = reinterpret_cast<volatile const T*>(src);
    for (std::size_t i = threadIdx.x; i < count; i += nworkers) {
        T v = src_v[i];
        dst0[i] = v;
        dst1[i] = v;
    }
}

template <typename T, RedOp kRedOp>
__device__ inline void reduce_broadcast_volatile_worker(
    const T* local, const T* recv, T* dst0, T* dst1,
    std::size_t count, int nworkers) {
    if (threadIdx.x >= nworkers) return;
    if constexpr (std::is_same_v<T, float>) {
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
    } else if constexpr (kIsPacked16<T>) {
        if (aligned_packed16(local, recv, dst0, count) &&
            aligned_packed16(local, recv, dst1, count)) {
            reduce_broadcast_volatile_packed16_worker<T, kRedOp>(
                local, recv, dst0, dst1, count, nworkers);
            return;
        }
    }
    volatile const T* recv_v = reinterpret_cast<volatile const T*>(recv);
    for (std::size_t i = threadIdx.x; i < count; i += nworkers) {
        T v = RedOpTraits<kRedOp, T>::apply(local[i], recv_v[i]);
        dst0[i] = v;
        dst1[i] = v;
    }
}

// 等待 send 方有可用 slot（credit）。专用 threadIdx==1 轮询，对齐 NCCL connStepCache。
template <typename T>
__device__ inline bool wait_send_credit(
    transport::SimpleChannelArgs<T> args, std::uint64_t step,
    std::uint64_t* head_cache, int* wait_status) {
    std::uint64_t* head = args.send_head;
    if (threadIdx.x == 0) *wait_status = 1;
    __syncthreads();
    if (threadIdx.x == 1) {
        if (args.abort != nullptr && transport::shm::load_abort(args.abort) != 0) {
            *wait_status = 0;
        }
        if (*head_cache + transport::shm::kSimpleFifoSteps <
                step + transport::shm::kSimpleFifoSliceSteps &&
            *wait_status != 0 && args.wait_observer != nullptr) {
            *args.wait_observer = 1;
            __threadfence_system();
        }
        while (*head_cache + transport::shm::kSimpleFifoSteps <
                   step + transport::shm::kSimpleFifoSliceSteps &&
               *wait_status != 0) {
            if (args.abort != nullptr && transport::shm::load_abort(args.abort) != 0) {
                *wait_status = 0;
                break;
            }
            *head_cache = transport::shm::load_step(head);
        }
        if (*wait_status != 0) *wait_status = 1;
    }
    __syncthreads();
    return *wait_status != 0;
}

template <typename T>
__device__ inline bool wait_recv_ready(
    transport::SimpleChannelArgs<T> args, std::uint64_t step,
    std::uint64_t* tail_cache, int* wait_status) {
    std::uint64_t* tail = args.recv_tail;
    if (threadIdx.x == 0) *wait_status = 1;
    __syncthreads();
    if (threadIdx.x == 0) {
        if (args.abort != nullptr && transport::shm::load_abort(args.abort) != 0) {
            *wait_status = 0;
        }
        if (*tail_cache < step + transport::shm::kSimpleFifoSliceSteps &&
            *wait_status != 0 && args.wait_observer != nullptr) {
            *args.wait_observer = 1;
            __threadfence_system();
        }
        while (*tail_cache < step + transport::shm::kSimpleFifoSliceSteps &&
               *wait_status != 0) {
            if (args.abort != nullptr && transport::shm::load_abort(args.abort) != 0) {
                *wait_status = 0;
                break;
            }
            *tail_cache = transport::shm::load_step(tail);
        }
        if (*wait_status != 0) *wait_status = 1;
    }
    __syncthreads();
    return *wait_status != 0;
}

__device__ inline void worker_barrier(int nworkers) {
    if (threadIdx.x < nworkers) {
        asm volatile("bar.sync 1, %0;" : : "r"(nworkers));
    }
}

template <typename T>
__device__ inline void post_send_ready(
    transport::SimpleChannelArgs<T> args, std::uint64_t step,
    std::size_t payload_bytes, bool data_stored) {
    std::uint64_t* tail = args.send_tail;
    if (threadIdx.x == blockDim.x - 1) {
        if (args.send_payload_bytes != nullptr) {
            args.send_payload_bytes[step % transport::shm::kSimpleFifoSteps] =
                static_cast<std::uint32_t>(payload_bytes);
        }
        if (data_stored || args.send_payload_bytes != nullptr) __threadfence_system();
        transport::shm::store_step(tail, step + transport::shm::kSimpleFifoSliceSteps);
    }
}

template <typename T>
__device__ inline void post_recv_credit(
    transport::SimpleChannelArgs<T> args, std::uint64_t step) {
    std::uint64_t* head = args.recv_head;
    if (threadIdx.x == blockDim.x - 2) {
        transport::shm::store_step(head, step + transport::shm::kSimpleFifoSliceSteps);
    }
}

template <typename T, RedOp kRedOp>
__device__ inline bool direct_send(
    transport::SimpleChannelArgs<T> args, const T* src,
    std::size_t nelem, std::uint64_t* send_step,
    std::uint64_t* send_head_cache, int nworkers, int* wait_status) {
    std::size_t slice_size =
        transport::shm::slice_elems<T>(nelem, args.step_elems);
    std::size_t slice_offset = 0;
    for (int slice = 0;
         slice < transport::shm::kSimpleFifoChunkSteps / transport::shm::kSimpleFifoSliceSteps;
         ++slice) {
        std::size_t work =
            slice_offset < nelem
                ? transport::shm::nelem(slice_size, nelem, slice_offset)
                : 0;
        if (!wait_send_credit<T>(args, *send_step, send_head_cache,
                                         wait_status)) return false;
        worker_barrier(nworkers);
        T* dst = args.send_fifo +
                     ((*send_step % transport::shm::kSimpleFifoSteps) *
                      args.slot_elems);
        if (work != 0) {
            copy_worker(src + slice_offset, dst, work, nworkers);
        }
        __syncthreads();
        post_send_ready<T>(args, *send_step, work * sizeof(T), work != 0);
        *send_step += transport::shm::kSimpleFifoSliceSteps;
        slice_offset += slice_size;
    }
    return true;
}

template <typename T, RedOp kRedOp>
__device__ inline bool recv_reduce_send(
    transport::SimpleChannelArgs<T> args, const T* local,
    std::size_t nelem, std::uint64_t* recv_step, std::uint64_t* send_step,
    std::uint64_t* recv_tail_cache, std::uint64_t* send_head_cache,
    int nworkers, int* wait_status) {
    std::size_t slice_size =
        transport::shm::slice_elems<T>(nelem, args.step_elems);
    std::size_t slice_offset = 0;
    for (int slice = 0;
         slice < transport::shm::kSimpleFifoChunkSteps / transport::shm::kSimpleFifoSliceSteps;
         ++slice) {
        std::size_t work =
            slice_offset < nelem
                ? transport::shm::nelem(slice_size, nelem, slice_offset)
                : 0;
        if (!wait_recv_ready<T>(args, *recv_step, recv_tail_cache,
                                        wait_status)) return false;
        if (!wait_send_credit<T>(args, *send_step, send_head_cache,
                                         wait_status)) return false;
        worker_barrier(nworkers);
        const T* recv = args.recv_fifo +
                            ((*recv_step % transport::shm::kSimpleFifoSteps) *
                             args.slot_elems);
        T* dst = args.send_fifo +
                     ((*send_step % transport::shm::kSimpleFifoSteps) *
                      args.slot_elems);
        if (work != 0) {
            reduce_volatile_worker<T, kRedOp>(local + slice_offset, recv,
                                                     dst, work, nworkers);
        }
        __syncthreads();
        post_recv_credit<T>(args, *recv_step);
        post_send_ready<T>(args, *send_step, work * sizeof(T), work != 0);
        *recv_step += transport::shm::kSimpleFifoSliceSteps;
        *send_step += transport::shm::kSimpleFifoSliceSteps;
        slice_offset += slice_size;
    }
    return true;
}

template <typename T, RedOp kRedOp>
__device__ inline bool recv_reduce_copy_send(
    transport::SimpleChannelArgs<T> args, const T* local,
    T* out, std::size_t nelem, std::uint64_t* recv_step,
    std::uint64_t* send_step, std::uint64_t* recv_tail_cache,
    std::uint64_t* send_head_cache, int nworkers, int* wait_status) {
    std::size_t slice_size =
        transport::shm::slice_elems<T>(nelem, args.step_elems);
    std::size_t slice_offset = 0;
    for (int slice = 0;
         slice < transport::shm::kSimpleFifoChunkSteps / transport::shm::kSimpleFifoSliceSteps;
         ++slice) {
        std::size_t work =
            slice_offset < nelem
                ? transport::shm::nelem(slice_size, nelem, slice_offset)
                : 0;
        if (!wait_recv_ready<T>(args, *recv_step, recv_tail_cache,
                                        wait_status)) return false;
        if (!wait_send_credit<T>(args, *send_step, send_head_cache,
                                         wait_status)) return false;
        worker_barrier(nworkers);
        const T* recv = args.recv_fifo +
                            ((*recv_step % transport::shm::kSimpleFifoSteps) *
                             args.slot_elems);
        T* dst = args.send_fifo +
                     ((*send_step % transport::shm::kSimpleFifoSteps) *
                      args.slot_elems);
        if (work != 0) {
            reduce_broadcast_volatile_worker<T, kRedOp>(
                local + slice_offset, recv, out + slice_offset, dst, work,
                nworkers);
        }
        __syncthreads();
        post_recv_credit<T>(args, *recv_step);
        post_send_ready<T>(args, *send_step, work * sizeof(T), work != 0);
        *recv_step += transport::shm::kSimpleFifoSliceSteps;
        *send_step += transport::shm::kSimpleFifoSliceSteps;
        slice_offset += slice_size;
    }
    return true;
}

template <typename T, RedOp kRedOp>
__device__ inline bool recv_copy_send(
    transport::SimpleChannelArgs<T> args, T* out,
    std::size_t nelem, std::uint64_t* recv_step, std::uint64_t* send_step,
    std::uint64_t* recv_tail_cache, std::uint64_t* send_head_cache,
    int nworkers, int* wait_status) {
    std::size_t slice_size =
        transport::shm::slice_elems<T>(nelem, args.step_elems);
    std::size_t slice_offset = 0;
    for (int slice = 0;
         slice < transport::shm::kSimpleFifoChunkSteps / transport::shm::kSimpleFifoSliceSteps;
         ++slice) {
        std::size_t work =
            slice_offset < nelem
                ? transport::shm::nelem(slice_size, nelem, slice_offset)
                : 0;
        if (!wait_recv_ready<T>(args, *recv_step, recv_tail_cache,
                                        wait_status)) return false;
        if (!wait_send_credit<T>(args, *send_step, send_head_cache,
                                         wait_status)) return false;
        worker_barrier(nworkers);
        const T* recv = args.recv_fifo +
                            ((*recv_step % transport::shm::kSimpleFifoSteps) *
                             args.slot_elems);
        T* dst = args.send_fifo +
                     ((*send_step % transport::shm::kSimpleFifoSteps) *
                      args.slot_elems);
        if (work != 0) {
            copy_broadcast_volatile_worker<T, kRedOp>(
                recv, out + slice_offset, dst, work, nworkers);
        }
        __syncthreads();
        post_recv_credit<T>(args, *recv_step);
        post_send_ready<T>(args, *send_step, work * sizeof(T), work != 0);
        *recv_step += transport::shm::kSimpleFifoSliceSteps;
        *send_step += transport::shm::kSimpleFifoSliceSteps;
        slice_offset += slice_size;
    }
    return true;
}

template <typename T, RedOp kRedOp>
__device__ inline bool direct_recv(
    transport::SimpleChannelArgs<T> args, T* out,
    std::size_t nelem, std::uint64_t* recv_step,
    std::uint64_t* recv_tail_cache, int nworkers, int* wait_status) {
    std::size_t slice_size =
        transport::shm::slice_elems<T>(nelem, args.step_elems);
    std::size_t slice_offset = 0;
    for (int slice = 0;
         slice < transport::shm::kSimpleFifoChunkSteps / transport::shm::kSimpleFifoSliceSteps;
         ++slice) {
        std::size_t work =
            slice_offset < nelem
                ? transport::shm::nelem(slice_size, nelem, slice_offset)
                : 0;
        if (!wait_recv_ready<T>(args, *recv_step, recv_tail_cache,
                                        wait_status)) return false;
        worker_barrier(nworkers);
        const T* recv = args.recv_fifo +
                            ((*recv_step % transport::shm::kSimpleFifoSteps) *
                             args.slot_elems);
        if (work != 0) {
            copy_volatile_worker(recv, out + slice_offset, work, nworkers);
        }
        __syncthreads();
        post_recv_credit<T>(args, *recv_step);
        *recv_step += transport::shm::kSimpleFifoSliceSteps;
        slice_offset += slice_size;
    }
    return true;
}

// Ring + Simple 协议主 kernel。
//   T / RedOp 为编译期参数；nranks 为运行时参数，内循环按 ring 位置展开。
//   base_steps 跨迭代持久化（对齐 NCCL conn->step），避免每轮重置 step counter。
//   nworkers = blockDim - 32：留 3 个专用线程做 wait/post，其余做数据搬运。
template <typename T, RedOp kRedOp>
__global__ __launch_bounds__(NANO_NCCL_BLOCK_THREADS, 1) void ring_simple_kernel(
    transport::SimpleFifoArgs<T> args, int nranks) {
    int channel = blockIdx.x;
    if (channel >= kChannels) {
        return;
    }
    transport::SimpleChannelArgs<T> channel_args{
        args.slot_elems,
        args.step_elems,
        args.send_fifo[channel],
        args.recv_fifo[channel],
        args.control.send_head[channel],
        args.control.recv_tail[channel],
        args.control.send_tail[channel],
        args.control.recv_head[channel],
        args.send_payload_bytes[channel],
        args.recv_payload_bytes[channel],
        args.abort,
        nullptr,
    };
    int nworkers = blockDim.x >= 3 * 32 ? blockDim.x - 32 : blockDim.x;
    std::size_t part_offset = 0;
    std::size_t part_count = 0;
    std::size_t chunk_count = 0;
    transport::shm::cbd_part<T>(args.count, channel, &part_offset, &part_count,
                                &chunk_count);
    if (part_count == 0 || chunk_count == 0) {
        return;
    }

    __shared__ std::uint64_t s_base_step;
    __shared__ int s_wait_status;
    if (threadIdx.x == 0) {
        s_base_step = args.control.base_steps[channel];
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
        std::size_t loop_chunk = transport::shm::simple_fifo_loop_chunk_elems<T>(
            chunk_count, args.slot_elems);
        if (rem_count < nranks * loop_chunk) {
            loop_chunk = transport::shm::align_up(
                transport::shm::div_up(rem_count, nranks),
                transport::shm::kSimpleFifoVectorBytes / sizeof(T));
        }
        std::size_t loop_count = nranks * loop_chunk;

        int chunk = (ring_ix + nranks - 1) % nranks;
        std::size_t chunk_offset = static_cast<std::size_t>(chunk) * loop_chunk;
        std::size_t offset = part_offset + elem_offset + chunk_offset;
        std::size_t work = transport::shm::nelem(loop_chunk, rem_count, chunk_offset);
        if (!direct_send<T, kRedOp>(channel_args, args.input + offset, work,
                                            &send_step, &send_head_cache, nworkers,
                                            &s_wait_status)) return;

        for (int j = 2; j < nranks; ++j) {
            chunk = (ring_ix + nranks - j) % nranks;
            chunk_offset = static_cast<std::size_t>(chunk) * loop_chunk;
            offset = part_offset + elem_offset + chunk_offset;
            work = transport::shm::nelem(loop_chunk, rem_count, chunk_offset);
            if (!recv_reduce_send<T, kRedOp>(
                channel_args, args.input + offset, work, &recv_step,
                &send_step, &recv_tail_cache, &send_head_cache, nworkers,
                &s_wait_status)) return;
        }

        chunk = ring_ix;
        chunk_offset = static_cast<std::size_t>(chunk) * loop_chunk;
        offset = part_offset + elem_offset + chunk_offset;
        work = transport::shm::nelem(loop_chunk, rem_count, chunk_offset);
        if (!recv_reduce_copy_send<T, kRedOp>(
            channel_args, args.input + offset, args.output + offset, work,
            &recv_step, &send_step, &recv_tail_cache, &send_head_cache,
            nworkers, &s_wait_status)) return;

        for (int j = 1; j < nranks - 1; ++j) {
            chunk = (ring_ix + nranks - j) % nranks;
            chunk_offset = static_cast<std::size_t>(chunk) * loop_chunk;
            offset = part_offset + elem_offset + chunk_offset;
            work = transport::shm::nelem(loop_chunk, rem_count, chunk_offset);
            if (!recv_copy_send<T, kRedOp>(
                channel_args, args.output + offset, work, &recv_step,
                &send_step, &recv_tail_cache, &send_head_cache, nworkers,
                &s_wait_status)) return;
        }

        chunk = (ring_ix + 1) % nranks;
        chunk_offset = static_cast<std::size_t>(chunk) * loop_chunk;
        offset = part_offset + elem_offset + chunk_offset;
        work = transport::shm::nelem(loop_chunk, rem_count, chunk_offset);
        if (!direct_recv<T, kRedOp>(channel_args, args.output + offset, work,
                                            &recv_step, &recv_tail_cache, nworkers,
                                            &s_wait_status)) return;

        elem_offset += loop_count;
    }

    if constexpr (kRedOp == RedOp::Avg) {
        scale_avg_worker(args.output + part_offset, part_count,
                         1.0f / static_cast<float>(nranks), nworkers);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        args.control.base_steps[channel] = send_step;
    }
}

}  // namespace nano_nccl::kernels
