#!/usr/bin/env python3
"""Static regression: RDMA send must host-bounce payload before ibv_post_send.

GPU writers publish FIFO slices via cudaHostAllocMapped memory. Posting
IBV_WR_SEND directly from fifo_.data lets the HCA DMA stale bytes; the
production path must CPU-observe the full payload (memcpy into a bounce
region) and post the SGE from that bounce buffer.
"""

from pathlib import Path
import re
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: rdma_proxy_send_visibility_static.py <rdma_proxy.cc>",
            file=sys.stderr,
        )
        return 2
    text = Path(sys.argv[1]).read_text()
    # Bound body to RdmaSendProxy::run only (next top-level symbol is
    # RdmaRecvProxy::max_inflight); do not scan recv pre_post which posts
    # SGE from fifo_.data by design.
    send_run = re.search(
        r"void RdmaSendProxy::run\(\)[^{]*\{(?P<body>.*)\n\}\n\n"
        r"std::size_t RdmaRecvProxy::max_inflight",
        text,
        re.S,
    )
    if send_run is None:
        raise AssertionError("could not locate RdmaSendProxy::run")
    body = send_run.group("body")
    if "ibv_post_send" not in body:
        raise AssertionError("send run missing ibv_post_send")

    # Require an explicit host payload copy before post (bounce buffer path).
    host_copy = re.search(
        r"\b(?:std::)?(?:memcpy|copy_n|copy)\s*\(",
        body,
    )
    if host_copy is None:
        raise AssertionError(
            "send path posts without host bounce copy "
            "(expected memcpy/copy of payload before ibv_post_send)"
        )

    # SGE must not address raw fifo_.data as the sole send source.
    # After the fix, sge.addr should reference bounce storage.
    sge_from_fifo = re.search(
        r"sge\.addr\s*=\s*reinterpret_cast<[^>]+>\(\s*"
        r"fifo_\.data\s*\+",
        body,
    )
    if sge_from_fifo is not None:
        raise AssertionError(
            "send SGE still addresses fifo_.data directly; "
            "post from registered bounce buffer after host copy"
        )

    print("rdma_proxy_send_visibility_static=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"rdma_proxy_send_visibility_static failed: {error}", file=sys.stderr)
        raise SystemExit(1)
