#pragma once

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace nano_nccl {

// GPU 数 / channel 数 / block 线程数为编译期常量，由 CMake 通过
// NANO_NCCL_NRANKS / NANO_NCCL_NCHANNELS / NANO_NCCL_BLOCK_THREADS 注入；
// 默认 4/4/512 对应当前单机 4 GPU 基线。宏兜底保证脱离 CMake 也能编译。
#ifndef NANO_NCCL_NRANKS
#define NANO_NCCL_NRANKS 4
#endif
constexpr int kRanks = NANO_NCCL_NRANKS;

#ifndef NANO_NCCL_NCHANNELS
#define NANO_NCCL_NCHANNELS 4
#endif
constexpr int kChannels = NANO_NCCL_NCHANNELS;

#ifndef NANO_NCCL_BLOCK_THREADS
#define NANO_NCCL_BLOCK_THREADS 512
#endif
constexpr int kBlockThreads = NANO_NCCL_BLOCK_THREADS;

// dtype 枚举与名称转换是公共接口；实际存储类型由 DTypeTraits 提供。
enum class DType { Float, Float16, BFloat16 };

inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float: return "float";
        case DType::Float16: return "fp16";
        case DType::BFloat16: return "bf16";
    }
    return "unknown";
}

inline bool parse_dtype(const char* text, DType* dtype) {
    if (std::strcmp(text, "float") == 0) { *dtype = DType::Float; return true; }
    if (std::strcmp(text, "fp16") == 0) { *dtype = DType::Float16; return true; }
    if (std::strcmp(text, "bf16") == 0) { *dtype = DType::BFloat16; return true; }
    return false;
}

enum class RedOp { Sum };

struct BenchConfig {
    std::string algo = "ring_simple";
    DType dtype = DType::Float;
    std::size_t min_bytes = 262144;
    std::size_t max_bytes = 67108864;
    int factor = 4;
    int warmup_iters = 2;
    int iters = 5;
    float epsilon = 0.0f;
};

struct BenchResult {
    std::string algo;
    DType dtype = DType::Float;
    std::size_t bytes = 0;
    std::size_t count = 0;
    double time_us = 0.0;
    double algbw = 0.0;
    double busbw = 0.0;
    int wrong = 0;
    float max_abs_error = 0.0f;
};

std::vector<std::size_t> make_sizes(std::size_t min_bytes,
                                    std::size_t max_bytes, int factor);

double algbw_gbs(std::size_t bytes, double time_us);
double all_reduce_busbw_gbs(double algbw, int nranks);

}  // namespace nano_nccl
