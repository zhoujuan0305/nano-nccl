#pragma once

#include "nano_nccl/types.h"

#include <cstddef>
#include <memory>

#include <cuda_runtime.h>

namespace nano_nccl {

namespace collective::all_reduce {
class CommunicatorFactory;
}

struct CommunicatorConfig {
    int device = 0;
    TransportKind transport = TransportKind::Auto;
};

struct AllReduceArgs {
    const void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    cudaStream_t stream = nullptr;
    std::size_t count = 0;  // Input and output elements per rank.
    DType dtype = DType::Float;
    RedOp redop = RedOp::Sum;
};

struct ReduceScatterArgs {
    const void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    cudaStream_t stream = nullptr;
    // Output elements per rank; each input has recv_count * global ranks.
    std::size_t recv_count = 0;
    DType dtype = DType::Float;
    RedOp redop = RedOp::Sum;
};

struct AllGatherArgs {
    const void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    cudaStream_t stream = nullptr;
    // Input elements per rank; each output has send_count * global ranks.
    std::size_t send_count = 0;
    DType dtype = DType::Float;
};

class Communicator {
public:
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;
    Communicator(Communicator&&) noexcept;
    Communicator& operator=(Communicator&&) noexcept;
    ~Communicator();

    void all_reduce(const AllReduceArgs& args);
    void reduce_scatter(const ReduceScatterArgs& args);
    void all_gather(const AllGatherArgs& args);
    void check_async_error() const;
    int local_rank_count() const noexcept;
    int global_rank_count() const noexcept;
    TransportKind transport() const noexcept;
    // Returns the backend for source_global_rank -> (source_global_rank + 1) % nranks.
    TransportKind edge_transport(int source_global_rank) const;

private:
    class Impl;
    explicit Communicator(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend class collective::all_reduce::CommunicatorFactory;
};

}  // namespace nano_nccl
