#pragma once

#include "nano_nccl/types.h"

#include <cstddef>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace nano_nccl {

namespace collective::all_reduce {
class CommunicatorFactory;
}

struct CommunicatorConfig {
    std::vector<int> devices;
    TransportKind transport = TransportKind::Auto;
};

struct AllReduceArgs {
    std::vector<const void*> send_buffers;
    std::vector<void*> recv_buffers;
    std::vector<cudaStream_t> streams;
    std::size_t count = 0;  // Input and output elements per rank.
    DType dtype = DType::Float;
    RedOp redop = RedOp::Sum;
};

struct ReduceScatterArgs {
    std::vector<const void*> send_buffers;
    std::vector<void*> recv_buffers;
    std::vector<cudaStream_t> streams;
    // Output elements per rank; each input has recv_count * global ranks.
    std::size_t recv_count = 0;
    DType dtype = DType::Float;
    RedOp redop = RedOp::Sum;
};

struct AllGatherArgs {
    std::vector<const void*> send_buffers;
    std::vector<void*> recv_buffers;
    std::vector<cudaStream_t> streams;
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

private:
    class Impl;
    explicit Communicator(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend std::unique_ptr<Communicator> create_communicator(
        const CommunicatorConfig& config);
    friend class collective::all_reduce::CommunicatorFactory;
};

std::unique_ptr<Communicator> create_communicator(
    const CommunicatorConfig& config);

}  // namespace nano_nccl
