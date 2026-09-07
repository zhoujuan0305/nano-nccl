#include "nano_nccl/types.h"

#include <limits>
#include <vector>

namespace nano_nccl {

std::vector<std::size_t> make_sizes(std::size_t min_bytes,
                                    std::size_t max_bytes, int factor) {
    std::vector<std::size_t> sizes;
    if (min_bytes == 0 || max_bytes < min_bytes || factor < 2) return sizes;
    for (std::size_t size = min_bytes; size <= max_bytes; size *= factor) {
        sizes.push_back(size);
        if (size > std::numeric_limits<std::size_t>::max() /
                       static_cast<std::size_t>(factor)) {
            break;
        }
    }
    return sizes;
}

double algbw_gbs(std::size_t bytes, double time_us) {
    return time_us <= 0.0 ? 0.0 : static_cast<double>(bytes) / time_us / 1000.0;
}

double all_reduce_busbw_gbs(double algbw, int nranks) {
    return algbw * (2.0 * static_cast<double>(nranks - 1) /
                    static_cast<double>(nranks));
}

}  // namespace nano_nccl
