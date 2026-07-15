#!/usr/bin/env python3
"""Verify communicator cleanup fails closed before member destruction can run."""

from pathlib import Path
import sys


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
            raise AssertionError(f"missing cleanup check: {fragment}")
        position += len(fragment)


def main() -> int:
    source = Path(sys.argv[1]).read_text()
    if "#include <exception>" not in source:
        raise AssertionError("cleanup must include <exception> for std::terminate")

    helper = function_body(source, "void fail_stop_on_cuda_cleanup_error")
    for fragment in ("if (status == cudaSuccess)", "report_cuda_error_noexcept",
                     "std::terminate()"):
        if fragment not in helper:
            raise AssertionError(f"cleanup fail-stop helper lacks {fragment}")

    release = function_body(source, "void release_lifetime_tracking() noexcept")
    require_in_order(release, [
        'status = cudaSetDevice(rank);',
        'fail_stop_on_cuda_cleanup_error(status, "cudaSetDevice");',
        'status = cudaStreamSynchronize(fallback_streams_[rank]);',
        'fail_stop_on_cuda_cleanup_error(status, "cudaStreamSynchronize");',
        'status = cudaEventSynchronize(completion_events_[rank]);',
        'fail_stop_on_cuda_cleanup_error(status, "cudaEventSynchronize");',
        'status = cudaEventDestroy(completion_events_[rank]);',
        'fail_stop_on_cuda_cleanup_error(status, "cudaEventDestroy");',
    ])

    reset_destroy = function_body(source, "void destroy_noexcept() noexcept")
    require_in_order(reset_destroy, [
        'status = cudaSetDevice(rank);',
        'fail_stop_on_cuda_cleanup_error(status, "cudaSetDevice");',
        'status = cudaEventDestroy(events_[rank]);',
        'fail_stop_on_cuda_cleanup_error(status, "cudaEventDestroy");',
    ])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, IndexError) as error:
        print(f"communicator cleanup static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
