cmake_minimum_required(VERSION 3.18)
project(nano_nccl LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译期可配置参数：默认值对应当前单机 4 GPU / sm_61 基线。
# NRANKS/NCHANNELS/BLOCK_THREADS/FIFO_BUFF_BYTES 通过 compile definitions
# 注入代码；CUDA_ARCH 仅用于 -arch=sm_X，不作为宏传递。
set(NANO_NCCL_NRANKS 4 CACHE STRING "Number of GPU ranks")
set(NANO_NCCL_NCHANNELS 4 CACHE STRING "Number of channels")
set(NANO_NCCL_CUDA_ARCH 61 CACHE STRING "CUDA compute capability (e.g. 61 for sm_61)")
set(NANO_NCCL_BLOCK_THREADS 512 CACHE STRING "Threads per block")
set(NANO_NCCL_FIFO_BUFF_BYTES 33554432 CACHE STRING "FIFO buffer size in bytes")

find_package(CUDA REQUIRED)
find_library(NUMA_LIB numa REQUIRED)

add_subdirectory(src)

target_compile_definitions(nano_nccl PUBLIC
    NANO_NCCL_NRANKS=${NANO_NCCL_NRANKS}
    NANO_NCCL_NCHANNELS=${NANO_NCCL_NCHANNELS}
    NANO_NCCL_BLOCK_THREADS=${NANO_NCCL_BLOCK_THREADS}
    NANO_NCCL_FIFO_BUFF_BYTES=${NANO_NCCL_FIFO_BUFF_BYTES}
)

add_subdirectory(benchmarks)
add_subdirectory(tests)
