add_library(nano_nccl STATIC
    core/buffer.cu
    core/numa.cu
    transport/p2p/p2p_fifo.cu
    transport/p2p/p2p_step_counters.cu
    transport/p2p/p2p_topology.cu
    transport/shm/shm_fifo.cu
    collective/all_reduce/ring_simple.cu
)

target_include_directories(nano_nccl PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_options(nano_nccl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${NANO_NCCL_CUDA_ARCH} -O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

find_library(NVML_LIB nvidia-ml)
if(NOT NVML_LIB)
    find_file(NVML_LIB libnvidia-ml.so.1 REQUIRED
        PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
              /lib/${CMAKE_LIBRARY_ARCHITECTURE})
endif()
target_link_libraries(nano_nccl PUBLIC ${NUMA_LIB} cudart ${NVML_LIB})
