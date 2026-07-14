#include "nano_nccl/all_reduce.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool read_size_arg(int argc, char** argv, int* index, std::size_t* out) {
    if (*index + 1 >= argc) {
        return false;
    }
    *out = static_cast<std::size_t>(std::strtoull(argv[++(*index)], nullptr, 10));
    return true;
}

bool read_int_arg(int argc, char** argv, int* index, int* out) {
    if (*index + 1 >= argc) {
        return false;
    }
    *out = std::atoi(argv[++(*index)]);
    return true;
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--algo auto|ring_simple] "
                 "[--dtype float|fp16|bf16] "
                 "[-b bytes] [-e bytes] [-f factor] [-w warmup] [-n iters]\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    nano_nccl::BenchConfig config;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--algo") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 2;
            }
            config.algo = argv[++i];
        } else if (std::strcmp(argv[i], "--dtype") == 0) {
            if (i + 1 >= argc ||
                !nano_nccl::parse_dtype(argv[++i], &config.dtype)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "-b") == 0) {
            if (!read_size_arg(argc, argv, &i, &config.min_bytes)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "-e") == 0) {
            if (!read_size_arg(argc, argv, &i, &config.max_bytes)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "-f") == 0) {
            if (!read_int_arg(argc, argv, &i, &config.factor)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "-w") == 0) {
            if (!read_int_arg(argc, argv, &i, &config.warmup_iters)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "-n") == 0) {
            if (!read_int_arg(argc, argv, &i, &config.iters)) {
                usage(argv[0]);
                return 2;
            }
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    std::vector<nano_nccl::BenchResult> results;
    int rc = nano_nccl::run_all_reduce_bench(config, &results);

    std::printf("# nano-nccl all_reduce_bench\n");
    std::printf("# algo %s dtype %s nGpus %d warmup iters: %d iters: %d validation: 1\n",
                config.algo.c_str(), nano_nccl::dtype_name(config.dtype),
                nano_nccl::kRanks, config.warmup_iters, config.iters);
    std::printf("# %14s %8s %12s %12s %10s %10s %10s %8s %12s\n", "algo",
                "dtype", "size(B)", "count", "time(us)", "algbw", "busbw",
                "#wrong", "max_abs");
    for (const auto& result : results) {
        std::printf("%14s %8s %12zu %12zu %10.2f %10.2f %10.2f %8d %12.6g\n",
                    result.algo.c_str(), nano_nccl::dtype_name(result.dtype),
                    result.bytes, result.count, result.time_us, result.algbw,
                    result.busbw, result.wrong, result.max_abs_error);
    }

    if (rc != 0) {
        return rc;
    }
    for (const auto& result : results) {
        if (result.wrong != 0) {
            return 1;
        }
    }
    return 0;
}
