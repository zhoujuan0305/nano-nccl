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
        require(text, "struct RdmaCtsSlot", "rdma_protocol.h RdmaCtsSlot")
        require(text, "struct RdmaPeerInfo", "rdma_protocol.h RdmaPeerInfo")
        require(text, "std::uint32_t qpn", "rdma_protocol.h qpn field")
        require(text, "std::uint32_t psn", "rdma_protocol.h psn field")
        require(text, "std::uint16_t port_lid", "rdma_protocol.h port_lid field")
        require(text, "std::uint16_t gid_index", "rdma_protocol.h gid_index field")
        require(text, "std::uint8_t  gid[16]", "rdma_protocol.h gid field")
        require(text, "std::uint32_t active_mtu", "rdma_protocol.h active_mtu field")
        require(text, "negotiate_path_mtu", "rdma_protocol.h negotiate_path_mtu")
        require(text, "recv_fifo_addr", "rdma_protocol.h recv_fifo_addr field")
        require(text, "cts_fifo_addr", "rdma_protocol.h cts_fifo_addr field")
        require(text, "static_assert(sizeof(RdmaCtsSlot) == 32)",
                "rdma_protocol.h RdmaCtsSlot size assert")
        require(text, "static_assert(sizeof(RdmaPeerInfo) == 64)",
                "rdma_protocol.h RdmaPeerInfo size assert")

    # Communicator must wire WRITE+CTS bootstrap when the env plane is selected.
    for path in sys.argv[2:]:
        if Path(path).name != "communicator.cu":
            continue
        text = Path(path).read_text()
        where = "communicator.cu"
        if "parse_rdma_data_plane_env" not in text and "NANO_NCCL_RDMA_USE_WRITE" not in text:
            raise AssertionError(
                f"{where}: missing parse_rdma_data_plane_env or NANO_NCCL_RDMA_USE_WRITE")
        require(text, "cts_fifo_addr", f"{where} cts_fifo_addr assignment")
        require(text, "RdmaDataPlane::WriteCts", f"{where} WriteCts construction branch")
        require(text, "RdmaWriteTargets", f"{where} RdmaWriteTargets for send proxy")
        require(text, "RdmaCtsRemote", f"{where} RdmaCtsRemote for recv proxy")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"rdma transport static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)