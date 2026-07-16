#pragma once

#include "nano_nccl/types.h"

#include <vector>

namespace nano_nccl::collective::all_reduce {

struct ProcessTopology {
    int global_rank_count;
    int local_rank_offset;
    std::vector<int> devices;
    std::vector<TransportKind> edge_kinds;
    bool distributed;
};

ProcessTopology make_single_process_topology(
    std::vector<int> devices, std::vector<TransportKind> edge_kinds);

void validate_process_topology(const ProcessTopology& topology);

bool is_local_global_rank(const ProcessTopology& topology, int global_rank);
int local_rank_for_global_rank(const ProcessTopology& topology, int global_rank);

}  // namespace nano_nccl::collective::all_reduce
