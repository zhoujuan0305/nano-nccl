#!/usr/bin/env python3
"""Verify benchmark profiling is absent from the default build."""

from pathlib import Path
import sys


def require(source: str, fragment: str, description: str) -> None:
    if fragment not in source:
        raise AssertionError(f"{description}: missing {fragment!r}")


def function_body(source: str, signature: str) -> str:
    start = source.find(signature)
    if start == -1:
        raise AssertionError(f"missing function: {signature}")
    open_brace = source.find("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace:index + 1]
    raise AssertionError(f"unterminated function: {signature}")


def require_in_order(body: str, fragments: list[str]) -> None:
    position = 0
    for fragment in fragments:
        position = body.find(fragment, position)
        if position == -1:
            raise AssertionError(f"missing fragment: {fragment}")
        position += len(fragment)


def main() -> int:
    root_cmake = Path(sys.argv[1]).read_text()
    source_cmake = Path(sys.argv[2]).read_text()
    bench_cmake = Path(sys.argv[3]).read_text()
    tests_cmake = Path(sys.argv[4]).read_text()
    ring_source = Path(sys.argv[5]).read_text()
    profile_header_path = Path(sys.argv[6])
    benchmark_source = Path(sys.argv[7]).read_text()
    if not profile_header_path.exists():
        raise AssertionError("profiling header is missing")
    profile_header = profile_header_path.read_text()

    require(root_cmake,
            'option(NANO_NCCL_ENABLE_BENCH_PROFILING "Enable NVTX and CUDA profiler instrumentation in the all-reduce benchmark" OFF)',
            "profiling option must default to OFF")
    require(source_cmake, "if(NANO_NCCL_ENABLE_BENCH_PROFILING)\n    target_compile_definitions(nano_nccl PRIVATE\n        NANO_NCCL_ENABLE_BENCH_PROFILING=1)\n    target_link_libraries(nano_nccl PRIVATE CUDA::nvToolsExt)\nendif()",
            "nano_nccl profiling must be gated and link NVTX only when enabled")
    require(bench_cmake, "if(NANO_NCCL_ENABLE_BENCH_PROFILING)\n    target_compile_definitions(nano_nccl_all_reduce_bench PRIVATE\n        NANO_NCCL_ENABLE_BENCH_PROFILING=1)\n    target_link_libraries(nano_nccl_all_reduce_bench PRIVATE CUDA::nvToolsExt)\nendif()",
            "benchmark profiling must be gated and link NVTX only when enabled")
    require(tests_cmake, "NAME nano_nccl_bench_profiling_static",
            "profiling static test must be registered")
    require(profile_header, 'return "all_reduce size=" + std::to_string(bytes) + "B";',
            "size range label")
    require(profile_header,
            'return all_reduce_size_range_name(bytes) + " iteration=" + std::to_string(iteration);',
            "iteration range label")
    profiler_stop = function_body(profile_header, "void stop()")
    require_in_order(profiler_stop, [
        "cudaProfilerStop()",
        "CUDA_CHECK_THROW(status);",
    ])

    run_batch = function_body(ring_source, "void run_batch(std::size_t count, int iters, std::size_t bytes)")
    require_in_order(run_batch, [
        "for (int iteration = 0; iteration < iters; ++iteration)",
        "NvtxRange iteration_range(",
        "all_reduce_iteration_range_name(bytes, iteration)",
        "communicator_->all_reduce(make_args(count));",
    ])

    bench = function_body(ring_source, "int run_ring_simple_bench_typed")
    require_in_order(bench, [
        "for (int iteration = 0; iteration < config.warmup_iters; ++iteration)",
        "runner.run_once(count);",
        "ProfilerSession profiler;",
        "NvtxRange size_range(",
        "all_reduce_size_range_name(bytes)",
        "runner.run_batch(count, config.iters, bytes);",
        "profiler.stop();",
        "runner.verify(expected, epsilon, &max_abs_error);",
    ])

    mpi_bench = function_body(benchmark_source, "int run_mpi_bench_typed")
    require_in_order(mpi_bench, [
        "for (int iteration = 0; iteration < config.warmup_iters; ++iteration)",
        "launch_and_wait();",
        'MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(benchmark start)"',
        "ProfilerSession profiler;",
        "NvtxRange size_range(",
        "all_reduce_size_range_name(bytes)",
        "for (int iteration = 0; iteration < config.iters; ++iteration)",
        "NvtxRange iteration_range(",
        "all_reduce_iteration_range_name(bytes, iteration)",
        "launch_and_wait();",
        "profiler.stop();",
        "cudaMemcpy(host_output.data(), recv_buffers[local_rank], bytes,",
    ])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"benchmark profiling static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
