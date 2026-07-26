add_executable(nano_nccl_all_reduce_bench all_reduce_bench.cu)
target_link_libraries(nano_nccl_all_reduce_bench PRIVATE nano_nccl)

if(NANO_NCCL_ENABLE_BENCH_PROFILING)
    target_compile_definitions(nano_nccl_all_reduce_bench PRIVATE
        NANO_NCCL_ENABLE_BENCH_PROFILING=1)
    target_link_libraries(nano_nccl_all_reduce_bench PRIVATE CUDA::nvToolsExt)
endif()

if(NANO_NCCL_ENABLE_MPI)
    target_compile_definitions(nano_nccl_all_reduce_bench PRIVATE NANO_NCCL_ENABLE_MPI=1)
    target_include_directories(nano_nccl_all_reduce_bench PRIVATE ${MPI_C_INCLUDE_DIRS})
    target_link_libraries(nano_nccl_all_reduce_bench PRIVATE nano_nccl_mpi MPI::MPI_C)

    if(NANO_NCCL_ENABLE_RDMA)
        target_compile_definitions(nano_nccl_all_reduce_bench PRIVATE NANO_NCCL_ENABLE_RDMA=1)
        target_link_libraries(nano_nccl_all_reduce_bench PRIVATE ${IBVERBS_LIBRARY})
    endif()
endif()
