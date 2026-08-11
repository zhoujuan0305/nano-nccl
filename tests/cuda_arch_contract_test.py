#!/usr/bin/env python3
"""Reject CUDA architectures below the Simple protocol memory-model floor."""

from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: cuda_arch_contract_test.py <source-root>", file=sys.stderr)
        return 2

    source = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="nano-nccl-cuda-arch-") as temporary:
        temporary_root = Path(temporary)
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(temporary_root / "sm61"),
                "-DNANO_NCCL_CUDA_ARCH=61",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        valid_configure = subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(temporary_root / "default"),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if valid_configure.returncode != 0:
            raise AssertionError(
                f"default CUDA architecture configure failed:\n{valid_configure.stdout}"
            )
        valid_build = subprocess.run(
            [
                "cmake",
                "--build",
                str(temporary_root / "default"),
                "--target",
                "nano_nccl_simple_protocol",
                "-j2",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if valid_build.returncode != 0:
            raise AssertionError(
                "fresh default nano_nccl_simple_protocol build failed:\n"
                f"{valid_build.stdout}"
            )
    expected = "NANO_NCCL_CUDA_ARCH must be at least 70"
    if result.returncode == 0 or expected not in result.stdout:
        raise AssertionError(
            f"SM61 configure was not rejected with {expected!r}:\n{result.stdout}"
        )
    print("cuda_arch_contract=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"cuda_arch_contract failed: {error}", file=sys.stderr)
        raise SystemExit(1)
