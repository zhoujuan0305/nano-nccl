#!/usr/bin/env python3
"""Static contract: GPUDirect registration is a real RDMA module, not a comment."""

from pathlib import Path
import sys


def require(text: str, fragment: str, where: str) -> None:
    if fragment not in text:
        raise AssertionError(f"missing {where}: {fragment!r}")


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: rdma_gdr_static.py <rdma_gdr.h> <rdma_gdr.cc> [cmake...]",
              file=sys.stderr)
        return 2
    header = Path(sys.argv[1]).read_text()
    source = Path(sys.argv[2]).read_text()
    require(header, "enum class RdmaMemoryPlacement", "placement enum")
    require(header, "HostPin", "HostPin placement")
    require(header, "GpuDirect", "GpuDirect placement")
    require(header, "parse_rdma_memory_placement_env", "env parser")
    require(header, "class RdmaRegisteredMemory", "registration type")
    require(header, "register_device", "device registration")
    require(header, "register_host", "host registration")
    require(source, "NANO_NCCL_RDMA_GDR", "GDR env switch")
    require(source, "ibv_reg_mr", "verbs registration")
    for path in sys.argv[3:]:
        if Path(path).name != "CMakeLists.txt":
            continue
        text = Path(path).read_text()
        require(text, "rdma_gdr.cc", f"{Path(path)} compiles rdma_gdr.cc")
    for path in sys.argv[3:]:
        if Path(path).name != "communicator.cu":
            continue
        text = Path(path).read_text()
        require(text, "parse_rdma_memory_placement_env", "communicator GDR env")
        require(text, "RdmaMemoryPlacement::GpuDirect", "communicator GpuDirect branch")
        require(text, "register_device", "communicator register_device")
        require(text, "DeviceBuffer", "communicator device FIFO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
