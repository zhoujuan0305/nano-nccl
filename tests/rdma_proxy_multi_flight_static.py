#!/usr/bin/env python3
"""Static regression: RDMA proxies multi-flight with selective SEND signaling."""

from pathlib import Path
import re
import sys


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(
            "usage: rdma_proxy_multi_flight_static.py <rdma_proxy.cc> [rdma_qp.cc]",
            file=sys.stderr,
        )
        return 2
    proxy_path = Path(sys.argv[1])
    text = proxy_path.read_text()
    # Must track inflight / max depth rather than post-then-poll-one forever.
    if "max_inflight" not in text and "inflight_" not in text:
        raise AssertionError("rdma_proxy.cc missing multi-flight inflight accounting")
    # Forbid the old 1-flight pattern: post_send then immediate poll_cq(..., 1) in a
    # tight wait-for-one-completion block without an inflight budget.
    send_run = re.search(
        r"void RdmaSendProxy::run\(\)[^{]*\{(?P<body>.*)\n\}\n\n"
        r"(?:void RdmaRecvProxy|std::size_t RdmaRecvProxy)",
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
    # H3: selective CQ signal — not every WQE via sq_sig_all.
    if "IBV_SEND_SIGNALED" not in body:
        raise AssertionError(
            "send path must set IBV_SEND_SIGNALED on a subset of WRs "
            "(sq_sig_all=0 selective signaling)"
        )
    if not re.search(r"send_flags\s*=\s*.*IBV_SEND_SIGNALED", body):
        raise AssertionError("send_flags must assign IBV_SEND_SIGNALED conditionally")
    # Must not hard-code every WR signaled.
    if re.search(r"send_flags\s*=\s*IBV_SEND_SIGNALED\s*;", body):
        raise AssertionError(
            "send_flags must not always be IBV_SEND_SIGNALED; use selective signal"
        )

    qp_path = Path(sys.argv[2]) if len(sys.argv) == 3 else proxy_path.parent / "rdma_qp.cc"
    if qp_path.is_file():
        qp_text = qp_path.read_text()
        if not re.search(r"sq_sig_all\s*=\s*0\s*;", qp_text):
            raise AssertionError(f"{qp_path.name} must set sq_sig_all = 0")
        if re.search(r"sq_sig_all\s*=\s*1\s*;", qp_text):
            raise AssertionError(f"{qp_path.name} must not set sq_sig_all = 1")
    print("rdma_proxy_multi_flight_static=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"rdma_proxy_multi_flight_static failed: {error}", file=sys.stderr)
        raise SystemExit(1)
