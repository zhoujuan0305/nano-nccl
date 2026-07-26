#!/usr/bin/env python3
import pathlib
import sys


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    paths = [pathlib.Path(arg) for arg in sys.argv[1:]]
    require(len(paths) == 4, "expected top-level, source, benchmark, and test CMake files")
    contents = {path: path.read_text() for path in paths}
    combined = "\n".join(contents.values())
    require("project(nano_nccl LANGUAGES C CXX CUDA)" in contents[paths[0]],
            "project must enable the C language for MPI::MPI_C")
    require("find_package(MPI REQUIRED COMPONENTS C)" in contents[paths[0]],
            "MPI discovery must require only the C component")
    require("MPI::MPI_C" in combined, "MPI consumers must link MPI::MPI_C")
    require("MPI::MPI_CXX" not in combined,
            "MPI C++ bindings are unavailable on supported Open MPI builds")
    require("MPI_CXX_" not in combined,
            "MPI consumers must use C-component include and compiler variables")
    source_contents = contents[paths[1]]
    require("target_compile_definitions(nano_nccl_mpi PUBLIC OMPI_SKIP_MPICXX)" in source_contents,
            "nano_nccl_mpi must publicly suppress Open MPI C++ bindings")
    require("target_compile_definitions(nano_nccl_mpi_test_fault PUBLIC OMPI_SKIP_MPICXX)" in source_contents,
            "nano_nccl_mpi_test_fault must publicly suppress Open MPI C++ bindings")


if __name__ == "__main__":
    main()
