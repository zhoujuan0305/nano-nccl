#include "kernels/ring_simple_kernel.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int kWorkers = 37;
constexpr std::size_t kGuard = 16;
constexpr std::size_t kMaxCount =
    static_cast<std::size_t>(kWorkers) * 8 * NANO_NCCL_FLOAT4_UNROLL + 32;
constexpr std::size_t kCapacity = kMaxCount + 2 * kGuard + 2;
constexpr float kGuardValue = -9137.25f;

void cuda_check(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " +
                                 cudaGetErrorString(status));
    }
}

template <typename T>
struct DeviceBuffer {
    T* raw = nullptr;
    std::vector<T> host = std::vector<T>(kCapacity);

    DeviceBuffer() { cuda_check(cudaMalloc(&raw, sizeof(T) * kCapacity), "cudaMalloc"); }
    ~DeviceBuffer() {
        if (raw != nullptr) {
            const cudaError_t status = cudaFree(raw);
            if (status != cudaSuccess)
                std::cerr << "cudaFree failed: " << cudaGetErrorString(status) << '\n';
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data(std::size_t offset) { return raw + kGuard + offset; }
    const T* data(std::size_t offset) const { return raw + kGuard + offset; }
    std::size_t index(std::size_t offset, std::size_t i = 0) const {
        return kGuard + offset + i;
    }
    void upload() {
        cuda_check(cudaMemcpy(raw, host.data(), sizeof(T) * kCapacity,
                              cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
    void download() {
        cuda_check(cudaMemcpy(host.data(), raw, sizeof(T) * kCapacity,
                              cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }
    void fill(T value) { std::fill(host.begin(), host.end(), value); }
};

template <typename T> T from_float(float x);
template <> float from_float(float x) { return x; }
template <> __half from_float(float x) { return __float2half(x); }
template <> __nv_bfloat16 from_float(float x) { return __float2bfloat16(x); }

template <typename T> float to_float(T x);
template <> float to_float(float x) { return x; }
template <> float to_float(__half x) { return __half2float(x); }
template <> float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
T host_reduce(nano_nccl::RedOp op, T a, T b, bool scale_avg,
              float inverse_nranks) {
    const float af = to_float(a);
    const float bf = to_float(b);
    if (op == nano_nccl::RedOp::Max) return from_float<T>(std::max(af, bf));
    if (op == nano_nccl::RedOp::Min) return from_float<T>(std::min(af, bf));
    T sum = from_float<T>(af + bf);
    if (op == nano_nccl::RedOp::Avg && scale_avg)
        return from_float<T>(to_float(sum) * inverse_nranks);
    return sum;
}

template <typename T>
struct Outputs {
    T* copy;
    T* copy_volatile;
    T* reduce[4];
    T* reduce_scaled_avg[4];
    T* broadcast0;
    T* broadcast1;
    T* broadcast_alias;
    T* reduce_broadcast0[4];
    T* reduce_broadcast1[4];
    T* reduce_broadcast_alias[4];
    T* local_into_dst[4];
    T* local_broadcast0[4];
    T* local_broadcast1[4];
    T* local_broadcast_alias[4];
};

template <typename T, nano_nccl::RedOp Op>
__device__ void run_redops(const T* local, const T* recv,
                           std::size_t count, int nworkers, float inv,
                           Outputs<T> out) {
    constexpr int oi = static_cast<int>(Op);
    nano_nccl::kernels::reduce_volatile_worker<T, Op>(
        local, recv, out.reduce[oi], count, nworkers);
    nano_nccl::kernels::reduce_volatile_worker<T, Op, true>(
        local, recv, out.reduce_scaled_avg[oi], count, nworkers, inv);
    nano_nccl::kernels::reduce_broadcast_volatile_worker<T, Op>(
        local, recv, out.reduce_broadcast0[oi], out.reduce_broadcast1[oi],
        count, inv, nworkers);
    nano_nccl::kernels::reduce_broadcast_volatile_worker<T, Op>(
        local, recv, out.reduce_broadcast_alias[oi],
        out.reduce_broadcast_alias[oi], count, inv, nworkers);
    // This helper updates its existing dst input in place by design.
    nano_nccl::kernels::reduce_local_into_dst<T, Op>(
        local, out.local_into_dst[oi], count, nworkers, inv);
    nano_nccl::kernels::reduce_local_broadcast_worker<T, Op>(
        local, out.local_broadcast0[oi], out.local_broadcast1[oi],
        count, inv, nworkers);
    nano_nccl::kernels::reduce_local_broadcast_worker<T, Op>(
        local, out.local_broadcast_alias[oi], out.local_broadcast_alias[oi],
        count, inv, nworkers);
}

template <typename T>
__global__ void vector_tail_kernel(const T* src, const T* local, const T* recv,
                                   std::size_t count, int nworkers,
                                   float inverse_nranks, Outputs<T> out) {
    nano_nccl::kernels::copy_worker(src, out.copy, count, nworkers);
    nano_nccl::kernels::copy_volatile_worker(src, out.copy_volatile, count,
                                              nworkers);
    nano_nccl::kernels::copy_broadcast_volatile_worker(
        src, out.broadcast0, out.broadcast1, count, nworkers);
    nano_nccl::kernels::copy_broadcast_volatile_worker(
        src, out.broadcast_alias, out.broadcast_alias, count, nworkers);
    run_redops<T, nano_nccl::RedOp::Sum>(local, recv, count, nworkers,
                                         inverse_nranks, out);
    run_redops<T, nano_nccl::RedOp::Avg>(local, recv, count, nworkers,
                                         inverse_nranks, out);
    run_redops<T, nano_nccl::RedOp::Max>(local, recv, count, nworkers,
                                         inverse_nranks, out);
    run_redops<T, nano_nccl::RedOp::Min>(local, recv, count, nworkers,
                                         inverse_nranks, out);
}

template <typename T>
__global__ void alignment_probe_kernel(const T* base, int* flags) {
    if (threadIdx.x != 0) return;
    if constexpr (std::is_same_v<T, float>) {
        flags[0] = nano_nccl::kernels::aligned_vec4(base, base, base);
        flags[1] = nano_nccl::kernels::aligned_vec4(base + 1, base, base);
        flags[2] = nano_nccl::kernels::aligned_vec4(base, base + 1, base);
        flags[3] = nano_nccl::kernels::aligned_vec4(base, base, base + 1);
    } else {
        flags[0] = nano_nccl::kernels::aligned_packed16(base, base, base);
        flags[1] = nano_nccl::kernels::aligned_packed16(base + 1, base, base);
        flags[2] = nano_nccl::kernels::aligned_packed16(base, base + 1, base);
        flags[3] = nano_nccl::kernels::aligned_packed16(base, base, base + 1);
    }
}

template <typename T>
struct NanOutputs {
    T* max_result;
    T* min_result;
    T* max_broadcast0;
    T* max_broadcast1;
    T* min_broadcast0;
    T* min_broadcast1;
};

template <typename T>
__global__ void nan_tail_kernel(const T* local, const T* recv,
                                std::size_t count, int nworkers,
                                NanOutputs<T> out) {
    nano_nccl::kernels::reduce_volatile_worker<T, nano_nccl::RedOp::Max>(
        local, recv, out.max_result, count, nworkers);
    nano_nccl::kernels::reduce_volatile_worker<T, nano_nccl::RedOp::Min>(
        local, recv, out.min_result, count, nworkers);
    nano_nccl::kernels::reduce_broadcast_volatile_worker<T, nano_nccl::RedOp::Max>(
        local, recv, out.max_broadcast0, out.max_broadcast1, count, 1.0f, nworkers);
    nano_nccl::kernels::reduce_broadcast_volatile_worker<T, nano_nccl::RedOp::Min>(
        local, recv, out.min_broadcast0, out.min_broadcast1, count, 1.0f, nworkers);
}

template <typename T>
bool same_value(T actual, T expected) {
    return to_float(actual) == to_float(expected);
}

template <typename T>
void prepare_input(DeviceBuffer<T>& b, std::size_t offset, std::size_t count,
                   float base) {
    b.fill(from_float<T>(kGuardValue));
    for (std::size_t i = 0; i < count; ++i)
        b.host[b.index(offset, i)] = from_float<T>(base + float(i % 11));
    b.upload();
}

template <typename T>
void prepare_output(DeviceBuffer<T>& b, std::size_t offset, std::size_t count,
                    float seed = kGuardValue) {
    b.fill(from_float<T>(kGuardValue));
    for (std::size_t i = 0; i < count; ++i)
        b.host[b.index(offset, i)] = from_float<T>(seed);
    b.upload();
}

template <typename T>
void verify_guard(const DeviceBuffer<T>& b, std::size_t offset,
                  std::size_t count, const char* label) {
    for (std::size_t i = 0; i < kCapacity; ++i) {
        if (i >= b.index(offset) && i < b.index(offset, count)) continue;
        if (!same_value(b.host[i], from_float<T>(kGuardValue)))
            throw std::runtime_error(std::string(label) + " modified a guard");
    }
}

template <typename T>
void verify_copy(DeviceBuffer<T>& b, std::size_t offset,
                 std::size_t count, float base, const char* label) {
    b.download();
    verify_guard(b, offset, count, label);
    for (std::size_t i = 0; i < count; ++i) {
        const T expected = from_float<T>(base + float(i % 11));
        if (!same_value(b.host[b.index(offset, i)], expected))
            throw std::runtime_error(std::string(label) + " data mismatch");
    }
}

struct CaseOffsets {
    std::size_t src;
    std::size_t local;
    std::size_t recv;
    std::size_t output0;
    std::size_t output1;
};

template <typename T>
void run_case(std::size_t count, CaseOffsets offsets) {
    DeviceBuffer<T> src, local, recv;
    DeviceBuffer<T> copy, copy_volatile, broadcast0, broadcast1, broadcast_alias;
    std::array<DeviceBuffer<T>, 4> reduce, reduce_scaled;
    std::array<DeviceBuffer<T>, 4> reduce_broadcast0, reduce_broadcast1;
    std::array<DeviceBuffer<T>, 4> reduce_broadcast_alias;
    std::array<DeviceBuffer<T>, 4> local_into_dst;
    std::array<DeviceBuffer<T>, 4> local_broadcast0, local_broadcast1;
    std::array<DeviceBuffer<T>, 4> local_broadcast_alias;

    prepare_input(src, offsets.src, count, 2.0f);
    prepare_input(local, offsets.local, count, 5.0f);
    prepare_input(recv, offsets.recv, count, 3.0f);
    prepare_output(copy, offsets.output0, count);
    prepare_output(copy_volatile, offsets.output0, count);
    prepare_output(broadcast0, offsets.output0, count);
    prepare_output(broadcast1, offsets.output1, count);
    prepare_output(broadcast_alias, offsets.output0, count);
    for (int op = 0; op < 4; ++op) {
        prepare_output(reduce[op], offsets.output0, count);
        prepare_output(reduce_scaled[op], offsets.output0, count);
        prepare_output(reduce_broadcast0[op], offsets.output0, count);
        prepare_output(reduce_broadcast1[op], offsets.output1, count);
        prepare_output(reduce_broadcast_alias[op], offsets.output0, count);
        prepare_input(local_into_dst[op], offsets.output0, count, 3.0f);
        prepare_output(local_broadcast0[op], offsets.output0, count);
        prepare_input(local_broadcast1[op], offsets.output1, count, 3.0f);
        prepare_input(local_broadcast_alias[op], offsets.output0, count, 3.0f);
    }

    Outputs<T> out{};
    out.copy = copy.data(offsets.output0);
    out.copy_volatile = copy_volatile.data(offsets.output0);
    out.broadcast0 = broadcast0.data(offsets.output0);
    out.broadcast1 = broadcast1.data(offsets.output1);
    out.broadcast_alias = broadcast_alias.data(offsets.output0);
    for (int op = 0; op < 4; ++op) {
        out.reduce[op] = reduce[op].data(offsets.output0);
        out.reduce_scaled_avg[op] = reduce_scaled[op].data(offsets.output0);
        out.reduce_broadcast0[op] = reduce_broadcast0[op].data(offsets.output0);
        out.reduce_broadcast1[op] = reduce_broadcast1[op].data(offsets.output1);
        out.reduce_broadcast_alias[op] = reduce_broadcast_alias[op].data(offsets.output0);
        out.local_into_dst[op] = local_into_dst[op].data(offsets.output0);
        out.local_broadcast0[op] = local_broadcast0[op].data(offsets.output0);
        out.local_broadcast1[op] = local_broadcast1[op].data(offsets.output1);
        out.local_broadcast_alias[op] = local_broadcast_alias[op].data(offsets.output0);
    }

    vector_tail_kernel<<<1, 64>>>(src.data(offsets.src), local.data(offsets.local),
                                  recv.data(offsets.recv), count, kWorkers, 0.25f, out);
    cuda_check(cudaGetLastError(), "vector_tail_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "vector_tail_kernel sync");

    verify_copy(copy, offsets.output0, count, 2.0f, "copy_worker");
    verify_copy(copy_volatile, offsets.output0, count, 2.0f, "copy_volatile_worker");
    verify_copy(broadcast0, offsets.output0, count, 2.0f, "broadcast dst0");
    verify_copy(broadcast1, offsets.output1, count, 2.0f, "broadcast dst1");
    verify_copy(broadcast_alias, offsets.output0, count, 2.0f, "broadcast dst0==dst1");

    constexpr std::array<nano_nccl::RedOp, 4> ops{
        nano_nccl::RedOp::Sum, nano_nccl::RedOp::Avg,
        nano_nccl::RedOp::Max, nano_nccl::RedOp::Min};
    for (int op = 0; op < 4; ++op) {
        const auto expected_for = [&](float a, float b) {
            return host_reduce(ops[op], from_float<T>(a), from_float<T>(b),
                               false, 0.25f);
        };
        reduce[op].download();
        reduce_scaled[op].download();
        reduce_broadcast0[op].download();
        reduce_broadcast1[op].download();
        reduce_broadcast_alias[op].download();
        local_into_dst[op].download();
        local_broadcast0[op].download();
        local_broadcast1[op].download();
        local_broadcast_alias[op].download();
        const std::string tag = "redop " + std::to_string(op);
        verify_guard(reduce[op], offsets.output0, count, "reduce_volatile_worker");
        verify_guard(reduce_scaled[op], offsets.output0, count, "scaled reduce");
        verify_guard(reduce_broadcast0[op], offsets.output0, count, "reduce broadcast0");
        verify_guard(reduce_broadcast1[op], offsets.output1, count, "reduce broadcast1");
        verify_guard(reduce_broadcast_alias[op], offsets.output0, count, "reduce broadcast alias");
        verify_guard(local_into_dst[op], offsets.output0, count, "reduce_local_into_dst");
        verify_guard(local_broadcast0[op], offsets.output0, count, "local broadcast0");
        verify_guard(local_broadcast1[op], offsets.output1, count, "local broadcast1");
        verify_guard(local_broadcast_alias[op], offsets.output0, count, "local broadcast alias");
        for (std::size_t i = 0; i < count; ++i) {
            const T unscaled = expected_for(5.0f + float(i % 11),
                                            3.0f + float(i % 11));
            const T scaled = host_reduce(ops[op], from_float<T>(5.0f + float(i % 11)),
                                         from_float<T>(3.0f + float(i % 11)),
                                         true, 0.25f);
            const T local_b = expected_for(5.0f + float(i % 11),
                                           3.0f + float(i % 11));
            if (!same_value(reduce[op].host[reduce[op].index(offsets.output0, i)], unscaled) ||
                !same_value(reduce_scaled[op].host[reduce_scaled[op].index(offsets.output0, i)], scaled) ||
                !same_value(reduce_broadcast0[op].host[reduce_broadcast0[op].index(offsets.output0, i)], scaled) ||
                !same_value(reduce_broadcast1[op].host[reduce_broadcast1[op].index(offsets.output1, i)], scaled) ||
                !same_value(reduce_broadcast_alias[op].host[reduce_broadcast_alias[op].index(offsets.output0, i)], scaled) ||
                !same_value(local_into_dst[op].host[local_into_dst[op].index(offsets.output0, i)], local_b) ||
                !same_value(local_broadcast0[op].host[local_broadcast0[op].index(offsets.output0, i)], scaled) ||
                !same_value(local_broadcast1[op].host[local_broadcast1[op].index(offsets.output1, i)], scaled) ||
                !same_value(local_broadcast_alias[op].host[local_broadcast_alias[op].index(offsets.output0, i)], scaled))
                throw std::runtime_error(tag + " output mismatch");
        }
    }
}

std::vector<std::size_t> test_counts() {
    std::vector<std::size_t> counts;
    for (std::size_t n = 0; n <= 17; ++n) counts.push_back(n);
    for (std::size_t width : {std::size_t{4}, std::size_t{8}}) {
        const std::size_t boundary = kWorkers * width * NANO_NCCL_FLOAT4_UNROLL;
        for (int delta = -1; delta <= 1; ++delta)
            counts.push_back(static_cast<std::size_t>(static_cast<long long>(boundary) + delta));
    }
    std::sort(counts.begin(), counts.end());
    counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
    return counts;
}

template <typename T>
void run_type(const char* name) {
    const std::vector<std::size_t> counts = test_counts();
    for (std::size_t count : counts) {
        run_case<T>(count, {0, 0, 0, 0, 0});
    }
    for (std::size_t count : {std::size_t{1}, std::size_t{5}, std::size_t{9},
                              std::size_t{17}, counts[counts.size() - 3],
                              counts[counts.size() - 2], counts[counts.size() - 1]})
        run_case<T>(count, {1, 1, 1, 1, 1});
    run_case<T>(17, {1, 0, 0, 0, 0});  // source only is misaligned
    run_case<T>(17, {0, 1, 0, 0, 0});  // local reduction source only is misaligned
    run_case<T>(17, {0, 0, 1, 0, 0});  // volatile receive only is misaligned
    run_case<T>(17, {0, 0, 0, 0, 1});  // second destination only is misaligned
    std::cout << name << " vector-tail cases passed (unroll="
              << NANO_NCCL_FLOAT4_UNROLL << ")\n";
}

template <typename T>
void test_alignment_predicate(const char* label) {
    DeviceBuffer<T> storage;
    int* device_flags = nullptr;
    cuda_check(cudaMalloc(&device_flags, sizeof(int) * 4), "cudaMalloc alignment flags");
    alignment_probe_kernel<<<1, 1>>>(storage.data(0), device_flags);
    cuda_check(cudaGetLastError(), "alignment_probe_kernel launch");
    std::array<int, 4> flags{};
    cuda_check(cudaMemcpy(flags.data(), device_flags, sizeof(flags),
                          cudaMemcpyDeviceToHost), "alignment flags D2H");
    cuda_check(cudaFree(device_flags), "cudaFree alignment flags");
    if (flags != std::array<int, 4>{1, 0, 0, 0})
        throw std::runtime_error(std::string(label) + " pointer alignment predicate mismatch");
}

template <typename T>
void test_nan_body_and_tail(const char* label) {
    const std::size_t vector_width = std::is_same_v<T, float> ? 4 : 8;
    const std::size_t count = vector_width + 1;
    DeviceBuffer<T> local, recv, max_result, min_result;
    DeviceBuffer<T> max_b0, max_b1, min_b0, min_b1;
    prepare_input(local, 0, count, 5.0f);
    prepare_input(recv, 0, count, 3.0f);
    local.host[local.index(0)] = from_float<T>(std::numeric_limits<float>::quiet_NaN());
    recv.host[recv.index(0, count - 1)] = from_float<T>(std::numeric_limits<float>::quiet_NaN());
    local.upload();
    recv.upload();
    prepare_output(max_result, 0, count);
    prepare_output(min_result, 0, count);
    prepare_output(max_b0, 0, count);
    prepare_output(max_b1, 0, count);
    prepare_output(min_b0, 0, count);
    prepare_output(min_b1, 0, count);
    NanOutputs<T> out{max_result.data(0), min_result.data(0), max_b0.data(0),
                      max_b1.data(0), min_b0.data(0), min_b1.data(0)};
    nan_tail_kernel<<<1, 64>>>(local.data(0), recv.data(0), count, kWorkers, out);
    cuda_check(cudaGetLastError(), "nan_tail_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "nan_tail_kernel sync");

    std::array<DeviceBuffer<T>*, 6> outputs{
        &max_result, &min_result, &max_b0, &max_b1, &min_b0, &min_b1};
    for (std::size_t output_index = 0; output_index < outputs.size(); ++output_index) {
        DeviceBuffer<T>& output = *outputs[output_index];
        output.download();
        verify_guard(output, 0, count, label);
        for (std::size_t i = 0; i < count; ++i) {
            const T value = output.host[output.index(0, i)];
            if (i == 0 || i == count - 1) {
                if (!std::isnan(to_float(value)))
                    throw std::runtime_error(std::string(label) +
                                             " failed to propagate operand NaN");
            } else {
                const bool is_max = output_index == 0 || output_index == 2 || output_index == 3;
                const T expected = from_float<T>(is_max
                    ? 5.0f + float(i % 11) : 3.0f + float(i % 11));
                if (!same_value(value, expected))
                    throw std::runtime_error(std::string(label) +
                                             " finite max/min mismatch");
            }
        }
    }
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status == cudaErrorNoDevice || (status == cudaSuccess && device_count == 0)) {
            std::cout << "SKIP: no CUDA device available\n";
            return 77;
        }
        cuda_check(status, "cudaGetDeviceCount");
        int current_device = 0;
        cuda_check(cudaGetDevice(&current_device), "cudaGetDevice");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, current_device),
                   "cudaGetDeviceProperties");
        run_type<float>("float");
        run_type<__half>("fp16");
        test_alignment_predicate<float>("float4");
        test_alignment_predicate<__half>("packed16 fp16");
        test_nan_body_and_tail<float>("float NaN");
        test_nan_body_and_tail<__half>("fp16 NaN");
        if (properties.major >= 8) {
            run_type<__nv_bfloat16>("bf16");
            test_alignment_predicate<__nv_bfloat16>("packed16 bf16");
            test_nan_body_and_tail<__nv_bfloat16>("bf16 NaN");
        } else {
            std::cout << "SKIP: BF16 worker cases require SM80+\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vector-tail test failed: " << error.what() << '\n';
        return 1;
    }
}
