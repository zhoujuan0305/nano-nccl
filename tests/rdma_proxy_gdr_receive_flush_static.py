#!/usr/bin/env python3
"""Static regression: GDR receive data is flushed before recv_tail publish."""

from pathlib import Path
import re
import sys


def function_body(text: str, signature: str) -> str:
    match = re.search(signature + r"\s*\{", text)
    if match is None:
        raise AssertionError(f"could not locate {signature}")
    start = match.end()
    depth = 1
    index = start
    while index < len(text) and depth > 0:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    if depth != 0:
        raise AssertionError(f"unbalanced body for {signature}")
    return text[start : index - 1]


def require_flush_before_publish(body: str, name: str) -> None:
    poll = body.find("ibv_poll_cq")
    flush = body.find("gdr_receive_flush_.flush()")
    publish = body.find("store_counter(control_.recv_tail")
    if poll < 0 or flush < 0 or publish < 0:
        raise AssertionError(f"{name} missing CQ poll, GDR flush, or recv_tail publish")
    if not poll < flush < publish:
        raise AssertionError(f"{name} must order CQ poll -> GDR flush -> recv_tail")


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: rdma_proxy_gdr_receive_flush_static.py "
            "<rdma_proxy.h> <rdma_proxy.cc> <communicator.cu> <rdma_gdr.cc>",
            file=sys.stderr,
        )
        return 2

    header = Path(sys.argv[1]).read_text()
    source = Path(sys.argv[2]).read_text()
    communicator = Path(sys.argv[3]).read_text()
    gdr = Path(sys.argv[4]).read_text()

    if "RdmaGdrReceiveFlush gdr_receive_flush_" not in header:
        raise AssertionError("recv proxy must own its GDR receive flush policy")

    require_flush_before_publish(
        function_body(source, r"bool RdmaRecvProxy::progress_send_recv\(\) noexcept"),
        "progress_send_recv",
    )
    require_flush_before_publish(
        function_body(source, r"bool RdmaRecvProxy::progress_write_cts\(\) noexcept"),
        "progress_write_cts",
    )

    if "RdmaGdrReceiveFlush::for_device" not in communicator:
        raise AssertionError("communicator must create a device-specific GDR receive flush")
    if communicator.count("gdr_receive_flush") < 3:
        raise AssertionError("communicator must pass the flush to both recv proxy data planes")
    if "backend=rdma/gdr receive flush failed" not in gdr:
        raise AssertionError("asynchronous receive flush errors must name the GDR backend")

    print("rdma_proxy_gdr_receive_flush_static=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"rdma_proxy_gdr_receive_flush_static failed: {error}", file=sys.stderr)
        raise SystemExit(1)
