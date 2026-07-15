#pragma once

#include "nano_nccl/types.h"

#include <cstddef>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace nano_nccl {

struct CommunicatorConfig {
    std::vector<int> devices;
    TransportKind transport = TransportKind::Auto;
};

struct CollectiveArgs {
    std::vector<const void*> send_buffers;
    std::vector<void*> recv_buffers;
    std::vector<cudaStream_t> streams;
    std::size_t count = 0;
    DType dtype = DType::Float;
    RedOp redop = RedOp::Sum;
};

class Communicator {
public:
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;
    Communicator(Communicator&&) noexcept;
    Communicator& operator=(Communicator&&) noexcept;
    ~Communicator();

    void all_reduce(const CollectiveArgs& args);
    void reduce_scatter(const CollectiveArgs& args);
    void all_gather(const CollectiveArgs& args);
    void check_async_error() const;
    int local_rank_count() const noexcept;
    int global_rank_count() const noexcept;

private:
    class Impl;
    explicit Communicator(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend std::unique_ptr<Communicator> create_communicator(
        const CommunicatorConfig& config);
};

std::unique_ptr<Communicator> create_communicator(
    const CommunicatorConfig& config);

}  // namespace nano_nccl
