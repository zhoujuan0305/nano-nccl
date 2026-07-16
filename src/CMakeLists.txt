set(nano_nccl_sources
    core/buffer.cu
    core/numa.cu
    transport/p2p/p2p_fifo.cu
    transport/p2p/p2p_step_counters.cu
    transport/p2p/p2p_topology.cu
    transport/shm/shm_fifo.cu
    transport/socket/socket_endpoint.cc
    transport/socket/socket_proxy.cc
    collective/all_reduce/topology.cc
    collective/all_reduce/communicator.cu
    collective/all_reduce/ring_simple.cu
)

find_library(NVML_LIB nvidia-ml)
if(NOT NVML_LIB)
    find_file(NVML_LIB libnvidia-ml.so.1 REQUIRED
        PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
              /lib/${CMAKE_LIBRARY_ARCHITECTURE})
endif()
find_package(Threads REQUIRED)

function(add_nano_nccl_library target)
    add_library(${target} STATIC ${nano_nccl_sources})
    target_include_directories(${target} PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${NANO_NCCL_CUDA_ARCH} -O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
    target_link_libraries(${target} PUBLIC ${NUMA_LIB} CUDA::cudart ${NVML_LIB} Threads::Threads)
endfunction()

add_nano_nccl_library(nano_nccl)

if(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION)
    add_nano_nccl_library(nano_nccl_test_fault)
    target_compile_definitions(nano_nccl_test_fault PRIVATE
        NANO_NCCL_SOCKET_TEST_FAULT_INJECTION=1
    )
endif()

if(NANO_NCCL_ENABLE_MPI)
    set(nano_nccl_mpi_include_dirs ${MPI_CXX_INCLUDE_DIRS})
    execute_process(
        COMMAND ${MPI_CXX_COMPILER} --showme:incdirs
        RESULT_VARIABLE nano_nccl_mpi_showme_status
        OUTPUT_VARIABLE nano_nccl_mpi_showme_include_dirs
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(nano_nccl_mpi_showme_status EQUAL 0 AND
       NOT nano_nccl_mpi_showme_include_dirs STREQUAL "")
        separate_arguments(nano_nccl_mpi_showme_include_dirs NATIVE_COMMAND
                           "${nano_nccl_mpi_showme_include_dirs}")
        list(APPEND nano_nccl_mpi_include_dirs
             ${nano_nccl_mpi_showme_include_dirs})
    endif()
    list(REMOVE_DUPLICATES nano_nccl_mpi_include_dirs)

    add_library(nano_nccl_mpi STATIC
        collective/all_reduce/mpi_communicator.cc
    )
    target_include_directories(nano_nccl_mpi PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${nano_nccl_mpi_include_dirs}
    )
    target_compile_options(nano_nccl_mpi PRIVATE -O3)
    target_link_libraries(nano_nccl_mpi PUBLIC nano_nccl MPI::MPI_CXX)

    if(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION)
        add_library(nano_nccl_mpi_test_fault STATIC
            collective/all_reduce/mpi_communicator.cc
        )
        target_include_directories(nano_nccl_mpi_test_fault PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
            ${nano_nccl_mpi_include_dirs}
        )
        target_compile_options(nano_nccl_mpi_test_fault PRIVATE -O3)
        target_link_libraries(nano_nccl_mpi_test_fault PUBLIC
            nano_nccl_test_fault MPI::MPI_CXX
        )
    endif()
endif()
