#!/usr/bin/env python3
"""Static regression: RDMA send must elide 0-byte Simple slices without ibv_post_send."""

from pathlib import Path
import re
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: rdma_proxy_zero_byte_static.py <rdma_proxy.cc>",
            file=sys.stderr,
        )
        return 2
    text = Path(sys.argv[1]).read_text()
    send_run = re.search(
        r"void RdmaSendProxy::run\(\)[^{]*\{(?P<body>.*)\n\}\n\n"
        r"std::size_t RdmaRecvProxy::max_inflight",
        text,
        re.S,
    )
    if send_run is None:
        raise AssertionError("could not locate RdmaSendProxy::run")
    body = send_run.group("body")
    if "payload_bytes == 0" not in body and "payload_bytes==0" not in body:
        raise AssertionError("send run missing 0-byte payload branch")
    # Zero branch must advance send_head / count elides and must not be the only path.
    zero_idx = body.find("payload_bytes == 0")
    if zero_idx < 0:
        zero_idx = body.find("payload_bytes==0")
    # Inspect a window after the condition covering the elide continue path.
    window = body[zero_idx : zero_idx + 800]
    if "continue" not in window:
        raise AssertionError("0-byte branch must continue without posting")
    if "ibv_post_send" in window.split("continue", 1)[0]:
        raise AssertionError("0-byte branch still calls ibv_post_send")
    if "send_head" not in window:
        raise AssertionError("0-byte branch must advance send_head")
    if "zero_payload_posts_" not in window:
        raise AssertionError("0-byte branch must count zero_payload_posts_")
    if "ibv_post_send" not in body:
        raise AssertionError("send run missing ibv_post_send for non-zero path")
    print("rdma_proxy_zero_byte_static=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"rdma_proxy_zero_byte_static failed: {error}", file=sys.stderr)
        raise SystemExit(1)
