#pragma once

#include "nano_nccl/types.h"
#include "transport/simple/protocol.h"

#include <cstddef>
#include <cstdint>

namespace nano_nccl::transport::simple {

constexpr std::size_t kConnectionControlBytes =
    2 * static_cast<std::size_t>(kChannels) * sizeof(std::uint64_t);
constexpr std::size_t kConnectionDataOffset =
    ((kConnectionControlBytes + 255) / 256) * 256;
constexpr std::size_t kConnectionRegionBytes =
    kConnectionDataOffset +
    static_cast<std::size_t>(kChannels) * kFifoBytes;

inline std::uint64_t* connection_head(void* region, int channel) {
    return static_cast<std::uint64_t*>(region) + channel;
}

inline std::uint64_t* connection_tail(void* region, int channel) {
    return static_cast<std::uint64_t*>(region) + kChannels + channel;
}

inline void* connection_fifo(void* region, int channel) {
    auto* bytes = static_cast<std::uint8_t*>(region);
    return bytes + kConnectionDataOffset +
           static_cast<std::size_t>(channel) * kFifoBytes;
}

// Device-visible view consumed by the Simple kernel path. Allocation and
// backend lifetime remain transport-runtime concerns.
struct SimpleConnectionView {
    void* send_fifo[kChannels]{};
    const void* recv_fifo[kChannels]{};
    std::uint64_t* send_head[kChannels]{};
    std::uint64_t* send_tail[kChannels]{};
    std::uint64_t* recv_head[kChannels]{};
    std::uint64_t* recv_tail[kChannels]{};
};

}  // namespace nano_nccl::transport::simple
