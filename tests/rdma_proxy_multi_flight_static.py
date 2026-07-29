#!/usr/bin/env python3
"""Static regression: RDMA proxies must multi-flight post before single-WC blocking."""

from pathlib import Path
import re
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: rdma_proxy_multi_flight_static.py <rdma_proxy.cc>", file=sys.stderr)
        return 2
    text = Path(sys.argv[1]).read_text()
    # Must track inflight / max depth rather than post-then-poll-one forever.
    if "max_inflight" not in text and "inflight_" not in text:
        raise AssertionError("rdma_proxy.cc missing multi-flight inflight accounting")
    # Forbid the old 1-flight pattern: post_send then immediate poll_cq(..., 1) in a
    # tight wait-for-one-completion block without an inflight budget.
    send_run = re.search(
        r"void RdmaSendProxy::run\(\)[^{]*\{(?P<body>.*)\n\}\n\nvoid RdmaRecvProxy",
        text,
        re.S,
    )
    if send_run is None:
        raise AssertionError("could not locate RdmaSendProxy::run")
    body = send_run.group("body")
    if "ibv_post_send" not in body:
        raise AssertionError("send run missing ibv_post_send")
    # After multi-flight, poll batch size must be > 1 capability.
    if "ibv_poll_cq" not in body:
        raise AssertionError("send run missing ibv_poll_cq")
    if re.search(r"while\s*\(\s*\(n\s*=\s*ibv_poll_cq\([^)]*,\s*1\s*,", body):
        # single-WC wait loop is OK only if posts can pile up first; require
        # an explicit post-while-under-budget loop marker.
        if "while" not in body or "inflight" not in body.lower():
            raise AssertionError("send path still looks 1-flight")
    print("rdma_proxy_multi_flight_static=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"rdma_proxy_multi_flight_static failed: {error}", file=sys.stderr)
        raise SystemExit(1)
