#pragma once

namespace nano_nccl::collective {

// Collective 接口：为未来 all_gather / reduce_scatter / broadcast 等留 seam。
class Collective {
public:
    virtual ~Collective() = default;
};

}  // namespace nano_nccl::collective
