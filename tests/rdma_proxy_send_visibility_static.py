#!/usr/bin/env python3
"""Static regression: RDMA send has no host bounce path.

SEND and WriteCts always post SGE from the registered mapped FIFO
(fifo_.data + fifo_mr_->lkey). Visibility is the all-worker
__threadfence_system + release/acquire step contract — not a bounce copy.
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

    for needle, msg in (
        ("NANO_NCCL_RDMA_SEND_MODE", "bounce env NANO_NCCL_RDMA_SEND_MODE must be gone"),
        ("parse_send_mode_env", "parse_send_mode_env must be gone"),
        ("SendMode", "SendMode enum must be gone"),
        ("bounce_mr_", "bounce_mr_ must be gone"),
        ("bytes_bounced_", "bytes_bounced_ must be gone"),
        ("bounce_", "bounce_ buffer must be gone"),
    ):
        if needle in text:
            raise AssertionError(msg)

    send_recv = re.search(
        r"bool RdmaSendProxy::progress_send_recv\(\)[^{]*\{(?P<body>.*)\n\}\n\n"
        r"bool RdmaSendProxy::progress_write_cts",
        text,
        re.S,
    )
    if send_recv is None:
        raise AssertionError("could not locate RdmaSendProxy::progress_send_recv")
    write_cts = re.search(
        r"bool RdmaSendProxy::progress_write_cts\(\)[^{]*\{(?P<body>.*)\n\}\n\n"
        r"std::size_t RdmaRecvProxy::max_inflight",
        text,
        re.S,
    )
    if write_cts is None:
        raise AssertionError("could not locate RdmaSendProxy::progress_write_cts")

    for name, body in (
        ("progress_send_recv", send_recv.group("body")),
        ("progress_write_cts", write_cts.group("body")),
    ):
        if "ibv_post_send" not in body:
            raise AssertionError(f"{name} missing ibv_post_send")
        if re.search(r"\b(?:std::)?(?:memcpy|copy_n|copy)\s*\(", body):
            raise AssertionError(f"{name} must not host-copy payload (bounce gone)")
        if not re.search(r"sge\.lkey\s*=\s*fifo_mr_->lkey", body):
            raise AssertionError(f"{name} must assign sge.lkey = fifo_mr_->lkey")
        if not re.search(
            r"sge\.addr\s*=\s*reinterpret_cast<[^>]+>\(\s*fifo_\.data\s*\+",
            body,
        ):
            raise AssertionError(f"{name} must post SGE from fifo_.data")
        if re.search(r"sge\.lkey\s*=\s*bounce", body):
            raise AssertionError(f"{name} must not use bounce lkey")

    print("rdma_proxy_send_visibility_static=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"rdma_proxy_send_visibility_static failed: {error}", file=sys.stderr)
        raise SystemExit(1)
