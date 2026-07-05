#pragma once

namespace nano_nccl::transport {

// Transport 接口：为未来 P2P / NVLink / network 路径留 seam，当前仅 SHM FIFO 实现。
class Transport {
public:
    virtual ~Transport() = default;
};

}  // namespace nano_nccl::transport
