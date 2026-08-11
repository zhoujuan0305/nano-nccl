#include "collective/all_reduce/ring_simple_geometry.h"
#include "transport/simple/geometry.h"
#include "transport/simple/protocol.h"
#include "transport/simple/step.h"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <type_traits>

#include <cuda_runtime.h>

namespace simple = nano_nccl::transport::simple;
namespace ring = nano_nccl::collective::all_reduce;

namespace {

constexpr bool empty_channel_geometry_is_zero() {
    std::size_t offset = 1;
    std::size_t count = 1;
    std::size_t chunk = 1;
    ring::cbd_part<float>(1, 1, &offset, &count, &chunk);
    return offset == 0 && count == 0 && chunk == 0;
}

constexpr bool non_aligned_channel_geometry_is_bounded() {
    std::size_t offset = 0;
    std::size_t count = 0;
    std::size_t chunk = 0;
    ring::cbd_part<float>(1025, 3, &offset, &count, &chunk);
    return offset == 768 && count == 257 && chunk == 128;
}

}  // namespace

static_assert(simple::kFifoSteps == 8);
static_assert(simple::kSliceSteps == 2);
static_assert(simple::kChunkSteps == 4);
static_assert(simple::kVectorBytes == 16);
static_assert(simple::step_elems<float>() ==
              simple::kFifoBytes / simple::kFifoSteps / sizeof(float));
static_assert(ring::simple_grain_elems<float>() == 128);
static_assert(ring::ring_edge_index(0, 1, 4) == 0);
static_assert(ring::ring_edge_index(0, 2, 4) == -1);
static_assert(std::is_standard_layout<simple::ControlArgs>::value);
static_assert(std::is_standard_layout<simple::FifoArgs<float>>::value);
static_assert(std::is_standard_layout<simple::ChannelArgs<float>>::value);
static_assert(sizeof(simple::ControlArgs) ==
              sizeof(void*) * (4 * nano_nccl::kChannels + 2));
static_assert(offsetof(simple::ControlArgs, send_head) == 0);
static_assert(offsetof(simple::ControlArgs, recv_tail) ==
              sizeof(void*) * nano_nccl::kChannels);
static_assert(offsetof(simple::ControlArgs, send_tail) ==
              sizeof(void*) * 2 * nano_nccl::kChannels);
static_assert(offsetof(simple::ControlArgs, recv_head) ==
              sizeof(void*) * 3 * nano_nccl::kChannels);
static_assert(offsetof(simple::ControlArgs, send_base_steps) ==
               sizeof(void*) * 4 * nano_nccl::kChannels);
static_assert(offsetof(simple::ControlArgs, recv_base_steps) ==
              sizeof(void*) * (4 * nano_nccl::kChannels + 1));
static_assert(offsetof(simple::FifoArgs<float>, rank) == 0);
static_assert(offsetof(simple::FifoArgs<float>, count) == 8);
static_assert(offsetof(simple::FifoArgs<float>, input) == 32);
static_assert(offsetof(simple::FifoArgs<float>, send_fifo) == 48);
static_assert(offsetof(simple::FifoArgs<float>, recv_fifo) ==
              48 + sizeof(void*) * nano_nccl::kChannels);
static_assert(offsetof(simple::FifoArgs<float>, send_payload_bytes) ==
              48 + sizeof(void*) * 2 * nano_nccl::kChannels);
static_assert(offsetof(simple::FifoArgs<float>, recv_payload_bytes) ==
              48 + sizeof(void*) * 3 * nano_nccl::kChannels);
static_assert(offsetof(simple::FifoArgs<float>, abort) ==
              48 + sizeof(void*) * 4 * nano_nccl::kChannels);
static_assert(offsetof(simple::FifoArgs<float>, control) ==
              56 + sizeof(void*) * 4 * nano_nccl::kChannels);
static_assert(offsetof(simple::ChannelArgs<float>, slot_elems) == 0);
static_assert(offsetof(simple::ChannelArgs<float>, send_fifo) == 16);
static_assert(offsetof(simple::ChannelArgs<float>, send_head) == 32);
static_assert(offsetof(simple::ChannelArgs<float>, send_payload_bytes) == 64);
static_assert(offsetof(simple::ChannelArgs<float>, abort) == 80);
static_assert(offsetof(simple::ChannelArgs<float>, wait_observer) == 88);
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
static_assert(offsetof(simple::FifoArgs<float>, turnaround) ==
              56 + sizeof(void*) * (8 * nano_nccl::kChannels + 2));
static_assert(sizeof(simple::FifoArgs<float>) ==
              56 + sizeof(void*) * (9 * nano_nccl::kChannels + 2));
static_assert(offsetof(simple::ChannelArgs<float>, turnaround) == 96);
static_assert(sizeof(simple::ChannelArgs<float>) == 104);
#else
static_assert(sizeof(simple::FifoArgs<float>) ==
              56 + sizeof(void*) * (8 * nano_nccl::kChannels + 2));
static_assert(sizeof(simple::ChannelArgs<float>) == 96);
#endif
static_assert(simple::slice_elems<float>(100, 1024) == 64);
static_assert(simple::slice_elems<float>(1025, 1024) == 516);
static_assert(empty_channel_geometry_is_zero());
static_assert(non_aligned_channel_geometry_is_bounded());
static_assert(ring::loop_chunk_elems<float>(64, 128) == 64);
static_assert(ring::loop_chunk_elems<float>(256, 128) == 128);
static_assert(ring::nelem(128, 100, 0) == 100);
static_assert(ring::nelem(128, 100, 64) == 36);
static_assert(ring::nelem(128, 100, 100) == 0);
static_assert(ring::nelem(128, 100, 101) == 0);

namespace {

constexpr std::uint32_t kPublishedPayload = 0x51a9c37dU;

__global__ void publish_payload(std::uint32_t* payload,
                                std::uint64_t* step) {
    if (threadIdx.x == 0) *payload = kPublishedPayload;
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1) {
        simple::store_step(step, simple::kSliceSteps);
    }
}

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

int run_publication_test() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice) return 77;
    if (!check_cuda(status, "cudaGetDeviceCount")) return 1;
    if (device_count == 0) return 77;
    if (!check_cuda(cudaSetDevice(0), "cudaSetDevice")) return 1;

    void* storage = nullptr;
    if (!check_cuda(cudaHostAlloc(&storage,
                                  sizeof(std::uint64_t) + sizeof(std::uint32_t),
                                  cudaHostAllocMapped),
                    "cudaHostAlloc")) return 1;
    auto* host_step = static_cast<std::uint64_t*>(storage);
    auto* host_payload = reinterpret_cast<std::uint32_t*>(host_step + 1);
    *host_step = 0;
    *host_payload = 0;

    std::uint64_t* device_step = nullptr;
    std::uint32_t* device_payload = nullptr;
    bool ok = check_cuda(cudaHostGetDevicePointer(&device_step, host_step, 0),
                         "cudaHostGetDevicePointer(step)") &&
              check_cuda(cudaHostGetDevicePointer(&device_payload, host_payload, 0),
                         "cudaHostGetDevicePointer(payload)");
    if (ok) {
        publish_payload<<<1, 32>>>(device_payload, device_step);
        ok = check_cuda(cudaGetLastError(), "publish_payload launch");
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(5);
    while (ok && __atomic_load_n(host_step, __ATOMIC_ACQUIRE) !=
                     simple::kSliceSteps &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    if (ok && __atomic_load_n(host_step, __ATOMIC_ACQUIRE) !=
                  simple::kSliceSteps) {
        std::fprintf(stderr, "timed out waiting for Simple publication\n");
        ok = false;
    }
    if (ok && *host_payload != kPublishedPayload) {
        std::fprintf(stderr, "system-scope publication did not expose payload\n");
        ok = false;
    }
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) ok = false;
    if (!check_cuda(cudaFreeHost(storage), "cudaFreeHost")) ok = false;
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    int result = run_publication_test();
    if (result != 0) return result;
    std::printf("simple_protocol=PASS\n");
    return 0;
}
