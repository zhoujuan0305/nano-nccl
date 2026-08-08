#!/usr/bin/env python3
"""Static regression: RDMA proxies multi-flight with selective data-path signaling.

Selective IBV_SEND_SIGNALED is required on data SEND and data WRITE posts.
CTS posts in the recv proxy may always signal; that path is out of scope here.
"""

from pathlib import Path
import re
import sys


def extract_fn_body(text: str, qualified_name: str) -> str:
    """Return the body of `(void|bool) qualified_name() ... { ... }` (brace-balanced)."""
    m = re.search(
        rf"(?:void|bool)\s+{re.escape(qualified_name)}\s*\(\)[^{{]*\{{", text
    )
    if m is None:
        raise AssertionError(f"could not locate {qualified_name}")
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise AssertionError(f"unbalanced braces in {qualified_name}")
    return text[start : i - 1]


def check_multi_flight_data_path(body: str, name: str) -> None:
    if "ibv_post_send" not in body:
        raise AssertionError(f"{name} missing ibv_post_send")
    if "ibv_poll_cq" not in body:
        raise AssertionError(f"{name} missing ibv_poll_cq")
    if "inflight" not in body.lower():
        raise AssertionError(f"{name} missing inflight accounting")
    if re.search(r"while\s*\(\s*\(n\s*=\s*ibv_poll_cq\([^)]*,\s*1\s*,", body):
        if "while" not in body or "inflight" not in body.lower():
            raise AssertionError(f"{name} still looks 1-flight")


def check_selective_signal(body: str, name: str) -> None:
    """Data-path posts must use selective SIGNAL, not always-on."""
    if "IBV_SEND_SIGNALED" not in body:
        raise AssertionError(
            f"{name} must set IBV_SEND_SIGNALED on a subset of WRs "
            "(sq_sig_all=0 selective signaling)"
        )
    if not re.search(r"send_flags\s*=\s*.*IBV_SEND_SIGNALED", body):
        raise AssertionError(
            f"{name}: send_flags must assign IBV_SEND_SIGNALED conditionally"
        )
    # Unconditional assignment is forbidden on the data path only.
    if re.search(r"send_flags\s*=\s*IBV_SEND_SIGNALED\s*;", body):
        raise AssertionError(
            f"{name}: send_flags must not always be IBV_SEND_SIGNALED; "
            "use selective signal"
        )


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

    # Analyze data-plane send paths only (not the thin run() dispatcher, and not
    # recv-proxy CTS posts which may always signal).
    for method in (
        "RdmaSendProxy::progress_send_recv",
        "RdmaSendProxy::progress_write_cts",
    ):
        body = extract_fn_body(text, method)
        check_multi_flight_data_path(body, method)
        check_selective_signal(body, method)

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
