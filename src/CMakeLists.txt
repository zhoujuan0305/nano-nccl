add_library(nano_nccl STATIC
    core/buffer.cu
    core/numa.cu
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

target_link_libraries(nano_nccl PUBLIC ${NUMA_LIB} cudart)
