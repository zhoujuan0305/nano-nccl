#!/usr/bin/env python3
"""Static regression for the RDMA transport public interface."""

from pathlib import Path
import sys


def require(text: str, fragment: str, where: str) -> None:
    if fragment not in text:
        raise AssertionError(f"missing {where}: {fragment!r}")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: rdma_transport_static.py <types.h> [cmake...]", file=sys.stderr)
        return 2
    types = Path(sys.argv[1]).read_text()

    require(types, "enum class TransportKind { Auto, Shm, P2p, Socket, Rdma, Mixed };",
            "TransportKind enum with Rdma")
    require(types, 'case TransportKind::Rdma: return "rdma";',
            "transport_name Rdma case")
    require(types,
            'if (std::strcmp(text, "rdma") == 0) { *transport = TransportKind::Rdma; return true; }',
            "parse_transport rdma branch")
    if len(sys.argv) >= 3:
        for path in sys.argv[2:]:
            # argv[2:] may include non-CMake paths (e.g. rdma_protocol.h); assert the
            # flag only against actual CMakeLists.txt files, validate others separately.
            if Path(path).name != "CMakeLists.txt":
                continue
            text = Path(path).read_text()
            where = Path(path).name
            require(text, "NANO_NCCL_ENABLE_RDMA", f"{where} CMake flag")

    rdma_header = Path(sys.argv[1]).parent.parent.parent / "src/transport/rdma/rdma_protocol.h"
    if rdma_header.exists():
        text = rdma_header.read_text()
        require(text, "struct RdmaPeerInfo", "rdma_protocol.h RdmaPeerInfo")
        require(text, "std::uint32_t qpn", "rdma_protocol.h qpn field")
        require(text, "std::uint32_t psn", "rdma_protocol.h psn field")
        require(text, "std::uint16_t port_lid", "rdma_protocol.h port_lid field")
        require(text, "std::uint16_t gid_index", "rdma_protocol.h gid_index field")
        require(text, "std::uint8_t  gid[16]", "rdma_protocol.h gid field")
        require(text, "static_assert(sizeof(RdmaPeerInfo) == 28)", "rdma_protocol.h size assert")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"rdma transport static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)