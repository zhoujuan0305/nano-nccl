#!/usr/bin/env python3
"""Static regression: all FIFO writers fence before send_tail publish."""

from pathlib import Path
import re
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
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


def require_in_order(body: str, fragments: list[str]) -> None:
    position = 0
    for fragment in fragments:
        position = body.find(fragment, position)
        if position == -1:
            raise AssertionError(f"missing fragment: {fragment}")
        position += len(fragment)


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: rdma_direct_publish_static.py "
            "<ring_simple_kernel.cuh> <shm_step.h>",
            file=sys.stderr,
        )
        return 2

    kernel = Path(sys.argv[1]).read_text()
    step = Path(sys.argv[2]).read_text()

    post = function_body(kernel, "__device__ inline void post_send_ready")
    require_in_order(
        post,
        [
            "bool data_stored, int nworkers)",
            "if (data_stored && threadIdx.x < nworkers)",
            "__threadfence_system()",
            "__syncthreads()",
            "if (threadIdx.x == blockDim.x - 1)",
            "__threadfence_system()",
            "transport::shm::store_step(tail,",
        ],
    )

    calls = re.findall(r"post_send_ready\s*<[^>]+>\s*\((.*?)\)\s*;", kernel, re.S)
    if not calls:
        raise AssertionError("no post_send_ready call sites found")
    for call in calls:
        if "nworkers" not in call:
            raise AssertionError(
                f"post_send_ready call missing nworkers: {call.strip()}"
            )

    if "st.release.sys.global.u64" not in step:
        raise AssertionError("shm_step.h missing st.release.sys.global.u64")
    if "ld.acquire.sys.global.u64" not in step:
        raise AssertionError("shm_step.h missing ld.acquire.sys.global.u64")
    if "__CUDA_ARCH__ >= 700" not in step and "__CUDA_ARCH__) && __CUDA_ARCH__ >= 700" not in step:
        if not re.search(r"__CUDA_ARCH__\s*>=\s*700", step):
            raise AssertionError("shm_step.h missing sm_70+ guard for release/acquire")

    print("rdma_direct_publish_static=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"rdma_direct_publish_static failed: {error}", file=sys.stderr)
        raise SystemExit(1)
