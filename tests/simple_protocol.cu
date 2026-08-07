#include "transport/simple_protocol.h"

#include <cstdio>

static_assert(nano_nccl::transport::kSimpleFifoSteps == 8);
static_assert(nano_nccl::transport::kSimpleFifoSliceSteps == 2);
static_assert(nano_nccl::transport::kSimpleFifoChunkSteps == 4);
static_assert(sizeof(nano_nccl::transport::SimpleChannelArgs<float>) <
              sizeof(nano_nccl::transport::SimpleFifoArgs<float>));
#if defined(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
static_assert(sizeof(nano_nccl::transport::SimpleChannelArgs<float>) == 104);
#else
static_assert(sizeof(nano_nccl::transport::SimpleChannelArgs<float>) == 96);
#endif

int main() {
    std::printf("simple_protocol=PASS\n");
    return 0;
}
