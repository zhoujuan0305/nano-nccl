add_executable(nano_nccl_smoke smoke.cu)
add_executable(nano_nccl_correctness correctness.cu)
target_link_libraries(nano_nccl_correctness PRIVATE nano_nccl)

add_executable(nano_nccl_public_api public_api.cu)
target_link_libraries(nano_nccl_public_api PRIVATE nano_nccl)

add_executable(nano_nccl_p2p_step_counters p2p_step_counters.cu)
target_link_libraries(nano_nccl_p2p_step_counters PRIVATE nano_nccl)

add_executable(nano_nccl_p2p_topology p2p_topology.cu)
target_link_libraries(nano_nccl_p2p_topology PRIVATE nano_nccl)

add_executable(nano_nccl_simple_protocol simple_protocol.cu)
target_link_libraries(nano_nccl_simple_protocol PRIVATE nano_nccl)

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
        NAME nano_nccl_communicator_bf16_validation_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/communicator_bf16_validation_static.py
                ${CMAKE_SOURCE_DIR}/src/collective/all_reduce/communicator.cu
    )
    add_test(
        NAME nano_nccl_avg_scale_fusion_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/avg_scale_fusion_static.py
                ${CMAKE_SOURCE_DIR}/src/kernels/ring_simple_kernel.cuh
    )
    add_test(
        NAME nano_nccl_bench_profiling_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/bench_profiling_static.py
                ${CMAKE_SOURCE_DIR}/CMakeLists.txt
                 ${CMAKE_SOURCE_DIR}/src/CMakeLists.txt
                 ${CMAKE_SOURCE_DIR}/benchmarks/CMakeLists.txt
                 ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
                 ${CMAKE_SOURCE_DIR}/src/collective/all_reduce/ring_simple.cu
                 ${CMAKE_SOURCE_DIR}/src/collective/all_reduce/bench_profiling.h
                 ${CMAKE_SOURCE_DIR}/benchmarks/all_reduce_bench.cu
    )
    add_test(
        NAME nano_nccl_rdma_transport_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/rdma_transport_static.py
                ${CMAKE_SOURCE_DIR}/include/nano_nccl/types.h
                ${CMAKE_SOURCE_DIR}/CMakeLists.txt
                ${CMAKE_SOURCE_DIR}/src/CMakeLists.txt
                ${CMAKE_SOURCE_DIR}/benchmarks/CMakeLists.txt
                ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_protocol.h
    )
    add_test(
        NAME nano_nccl_mpi_c_api_link_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/mpi_c_api_link_static.py
                ${CMAKE_SOURCE_DIR}/CMakeLists.txt
                ${CMAKE_SOURCE_DIR}/src/CMakeLists.txt
                ${CMAKE_SOURCE_DIR}/benchmarks/CMakeLists.txt
                ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    )
endif()

if(NANO_NCCL_ENABLE_MPI)
    add_executable(nano_nccl_mpi_correctness mpi_correctness.cu)
    target_include_directories(nano_nccl_mpi_correctness PRIVATE ${MPI_C_INCLUDE_DIRS})
    if(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION)
        target_link_libraries(nano_nccl_mpi_correctness PRIVATE
            nano_nccl_mpi_test_fault MPI::MPI_C)
    else()
        target_link_libraries(nano_nccl_mpi_correctness PRIVATE nano_nccl_mpi MPI::MPI_C)
    endif()

    add_executable(nano_nccl_mpi_bootstrap mpi_bootstrap.cu)
    target_include_directories(nano_nccl_mpi_bootstrap PRIVATE ${MPI_C_INCLUDE_DIRS})
    target_link_libraries(nano_nccl_mpi_bootstrap PRIVATE nano_nccl_mpi MPI::MPI_C)

    add_executable(nano_nccl_socket_protocol socket_protocol_test.cc)
    target_link_libraries(nano_nccl_socket_protocol PRIVATE nano_nccl_mpi)

    if(NANO_NCCL_ENABLE_RDMA)
        add_executable(nano_nccl_rdma_protocol rdma_protocol.cc)
        target_link_libraries(nano_nccl_rdma_protocol PRIVATE nano_nccl ${IBVERBS_LIBRARY})
        target_compile_definitions(nano_nccl_rdma_protocol PRIVATE NANO_NCCL_ENABLE_RDMA=1)
        add_test(NAME nano_nccl_rdma_protocol COMMAND nano_nccl_rdma_protocol)

        add_executable(nano_nccl_rdma_bootstrap rdma_bootstrap.cu)
        target_link_libraries(nano_nccl_rdma_bootstrap PRIVATE nano_nccl_mpi ${IBVERBS_LIBRARY})
        target_compile_definitions(nano_nccl_rdma_bootstrap PRIVATE NANO_NCCL_ENABLE_RDMA=1)
        add_test(NAME nano_nccl_rdma_bootstrap COMMAND nano_nccl_rdma_bootstrap)
        set_tests_properties(nano_nccl_rdma_bootstrap PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
