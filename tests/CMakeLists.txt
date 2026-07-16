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
endif()

if(NANO_NCCL_ENABLE_MPI)
    add_executable(nano_nccl_mpi_correctness mpi_correctness.cu)
    target_include_directories(nano_nccl_mpi_correctness PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    if(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION)
        target_link_libraries(nano_nccl_mpi_correctness PRIVATE
            nano_nccl_mpi_test_fault MPI::MPI_CXX)
    else()
        target_link_libraries(nano_nccl_mpi_correctness PRIVATE nano_nccl_mpi MPI::MPI_CXX)
    endif()

    add_executable(nano_nccl_mpi_bootstrap mpi_bootstrap.cu)
    target_include_directories(nano_nccl_mpi_bootstrap PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(nano_nccl_mpi_bootstrap PRIVATE nano_nccl_mpi MPI::MPI_CXX)

    add_executable(nano_nccl_socket_protocol socket_protocol_test.cc)
    target_link_libraries(nano_nccl_socket_protocol PRIVATE nano_nccl_mpi)
endif()
