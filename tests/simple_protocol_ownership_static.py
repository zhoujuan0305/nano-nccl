#!/usr/bin/env python3
"""Verify Simple protocol ownership and dependency direction."""

from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: simple_protocol_ownership_static.py <source-root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    required = [
        root / "src/transport/simple/protocol.h",
        root / "src/transport/simple/step.h",
        root / "src/transport/simple/geometry.h",
        root / "src/collective/all_reduce/ring_simple_geometry.h",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise AssertionError(f"missing owned modules: {', '.join(missing)}")
    forbidden = [
        root / "src/transport/simple_protocol.h",
        root / "src/transport/shm/shm_step.h",
        root / "src/transport/shm/shm_fifo.cu",
    ]
    present = [str(path.relative_to(root)) for path in forbidden if path.exists()]
    if present:
        raise AssertionError(f"obsolete protocol files remain: {', '.join(present)}")
    kernel = (root / "src/kernels/ring_simple_kernel.cuh").read_text()
    if "transport::shm" in kernel or '"transport/shm/' in kernel:
        raise AssertionError("Ring kernel still depends on the SHM adapter")
    shm_fifo = (root / "src/transport/shm/shm_fifo.h").read_text()
    for leaked in ("slice_elems", "cbd_part", "load_step", "store_step"):
        if leaked in shm_fifo:
            raise AssertionError(f"SHM adapter still owns {leaked}")
    cmake = (root / "CMakeLists.txt").read_text()
    if "set(NANO_NCCL_CUDA_ARCH 70" not in cmake or "NANO_NCCL_CUDA_ARCH LESS 70" not in cmake:
        raise AssertionError("CMake does not enforce the SM70 architecture floor")
    documentation_contracts = {
        root / "README.md": {
            "the float/FP16 SM70+ floor": ("`float` and FP16 require SM70+",),
            "the BF16 SM80+ floor": ("BF16 requires SM80+",),
            "the CMake architecture default and rejection": (
                "`NANO_NCCL_CUDA_ARCH` | 70",
                "values below 70 are rejected",
            ),
            "Simple protocol ownership": (
                "`src/transport/simple/` owns",
                "Simple FIFO layout",
                "step ordering",
                "slice geometry",
            ),
            "SHM ownership": ("`src/transport/shm/` owns mapped host FIFO storage and control only",),
            "Ring scheduling ownership": (
                "`src/collective/all_reduce/ring_simple_geometry.h` owns Ring channel/rank work partitioning",
            ),
            "transport runtime lifecycle ownership": (
                "Transport runtime lifecycle and orchestration remain in `Communicator::Impl`",
            ),
        },
        root / "README.zh.md": {
            "the float/FP16 SM70+ floor": ("`float` 和 FP16 需要 SM70+",),
            "the BF16 SM80+ floor": ("BF16 需要 SM80+",),
            "the CMake architecture default and rejection": (
                "`NANO_NCCL_CUDA_ARCH` | 70",
                "低于 70 的值会被拒绝",
            ),
            "Simple protocol ownership": (
                "`src/transport/simple/` 负责 Simple FIFO layout",
                "step ordering",
                "slice geometry",
            ),
            "SHM ownership": ("`src/transport/shm/` 仅负责 mapped host FIFO storage 和 control",),
            "Ring scheduling ownership": (
                "`src/collective/all_reduce/ring_simple_geometry.h` 负责 Ring channel/rank work partitioning",
            ),
            "transport runtime lifecycle ownership": (
                "transport runtime lifecycle 与 orchestration 仍由 `Communicator::Impl` 管理",
            ),
        },
        root / "AGENTS.md": {
            "the float/FP16 SM70+ floor": ("`float` and FP16 require SM70+",),
            "the BF16 SM80+ floor": ("BF16 requires SM80+",),
            "the CMake architecture default and rejection": (
                "`NANO_NCCL_CUDA_ARCH` defaults to 70",
                "CMake rejects values below 70",
            ),
            "Simple protocol ownership": (
                "`src/transport/simple/` owns FIFO layout, step ordering, and slice geometry",
            ),
            "SHM ownership": ("`src/transport/shm/` owns mapped host FIFO storage/control only",),
            "Ring scheduling ownership": (
                "`src/collective/all_reduce/ring_simple_geometry.h` owns Ring channel/rank work partitioning",
            ),
            "transport runtime lifecycle ownership": (
                "Transport runtime lifecycle and orchestration remain in `Communicator::Impl`",
            ),
        },
    }
    for path, contracts in documentation_contracts.items():
        text = path.read_text()
        for obsolete in ("kSimpleFifoSteps", "kSimpleFifoSliceSteps"):
            if obsolete in text:
                raise AssertionError(f"{path.name} still uses obsolete {obsolete}")
        for description, phrases in contracts.items():
            if not all(phrase in text for phrase in phrases):
                raise AssertionError(f"{path.name} does not state {description}")
    print("simple_protocol_ownership=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"simple_protocol_ownership failed: {error}", file=sys.stderr)
        raise SystemExit(1)
