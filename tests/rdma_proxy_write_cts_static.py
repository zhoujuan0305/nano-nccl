#!/usr/bin/env python3
"""Static regression: WRITE+CTS data-plane types and env parse hooks.

Requires RdmaDataPlane, parse_rdma_data_plane_env, RdmaWriteTargets,
RdmaCtsRemote, and NANO_NCCL_RDMA_USE_WRITE in the RDMA proxy sources.
"""

from pathlib import Path
import re
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: rdma_proxy_write_cts_static.py <rdma_proxy.h> [rdma_proxy.cc ...]",
            file=sys.stderr,
        )
        return 2
    text = "\n".join(Path(p).read_text() for p in sys.argv[1:])

    if "enum class RdmaDataPlane" not in text:
        raise AssertionError("missing enum class RdmaDataPlane")
    if not re.search(r"\bSendRecv\b", text):
        raise AssertionError("RdmaDataPlane missing SendRecv")
    if not re.search(r"\bWriteCts\b", text):
        raise AssertionError("RdmaDataPlane missing WriteCts")

    if "parse_rdma_data_plane_env" not in text:
        raise AssertionError("missing parse_rdma_data_plane_env")
    if "NANO_NCCL_RDMA_USE_WRITE" not in text:
        raise AssertionError("missing NANO_NCCL_RDMA_USE_WRITE")

    parse = re.search(
        r"parse_rdma_data_plane_env\(\)\s*(?:noexcept\s*)?\{(?P<body>.*?)\n\}",
        text,
        re.S,
    )
    if parse is None:
        raise AssertionError("could not locate parse_rdma_data_plane_env body")
    parse_body = parse.group("body")
    if "NANO_NCCL_RDMA_USE_WRITE" not in parse_body:
        raise AssertionError("parser must read NANO_NCCL_RDMA_USE_WRITE")
    if not re.search(r"RdmaDataPlane::SendRecv", parse_body):
        raise AssertionError("parser must return SendRecv for default/off")
    if not re.search(r"RdmaDataPlane::WriteCts", parse_body):
        raise AssertionError("parser must return WriteCts for on/1/true")
    if "must be 0 or 1" not in parse_body:
        raise AssertionError("parser must throw on invalid values")

    if "struct RdmaWriteTargets" not in text:
        raise AssertionError("missing struct RdmaWriteTargets")
    for field in (
        "remote_fifo_addr",
        "remote_fifo_rkey",
        "remote_fifo_bytes",
        "local_cts",
        "cts_slot_count",
        "local_cts_mr",
    ):
        if not re.search(rf"\b{field}\b", text):
            raise AssertionError(f"RdmaWriteTargets missing field {field}")

    if "struct RdmaCtsRemote" not in text:
        raise AssertionError("missing struct RdmaCtsRemote")
    for field in (
        "remote_cts_addr",
        "remote_cts_rkey",
        "cts_slot_count",
        "local_shadow",
        "local_shadow_mr",
        "local_recv_fifo_addr",
        "local_recv_fifo_rkey",
    ):
        if not re.search(rf"\b{field}\b", text):
            raise AssertionError(f"RdmaCtsRemote missing field {field}")

    # post_cts must prefer IBV_SEND_INLINE when QP max_inline_data fits CTS.
    post_cts = re.search(
        r"void\s+RdmaRecvProxy::post_cts\s*\([^)]*\)\s*\{(?P<body>.*)\n\}",
        text,
        re.S,
    )
    if post_cts is None:
        # brace-balanced fallback
        m = re.search(r"void\s+RdmaRecvProxy::post_cts\s*\([^)]*\)\s*\{", text)
        if m is None:
            raise AssertionError("could not locate RdmaRecvProxy::post_cts")
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        cts_body = text[start : i - 1]
    else:
        cts_body = post_cts.group("body")
    if "IBV_SEND_INLINE" not in cts_body:
        raise AssertionError("post_cts must use IBV_SEND_INLINE when QP allows")
    if "max_inline_data" not in cts_body:
        raise AssertionError("post_cts must consult qp max_inline_data")

    print("rdma_proxy_write_cts_static=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"rdma_proxy_write_cts_static failed: {error}", file=sys.stderr)
        raise SystemExit(1)
