#include "transport/simple_protocol.h"

#include <cstdio>

static_assert(nano_nccl::transport::kSimpleFifoSteps == 8);
static_assert(nano_nccl::transport::kSimpleFifoSliceSteps == 2);
static_assert(nano_nccl::transport::kSimpleFifoChunkSteps == 4);
static_assert(sizeof(nano_nccl::transport::SimpleChannelArgs<float>) <
              sizeof(nano_nccl::transport::SimpleFifoArgs<float>));
static_assert(sizeof(nano_nccl::transport::SimpleChannelArgs<float>) == 96);

int main() {
    std::printf("simple_protocol=PASS\n");
    return 0;
}
