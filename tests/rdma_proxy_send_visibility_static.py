#!/usr/bin/env python3
"""Static regression: RDMA send keeps bounce fallback; default is direct.

Default posts IBV_WR_SEND from the registered mapped FIFO (fifo_mr_), matching
NCCL host-pin style after multi-host stress. Bounce mode remains available via
NANO_NCCL_RDMA_SEND_MODE=bounce for visibility debugging.
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

    if "NANO_NCCL_RDMA_SEND_MODE" not in text:
        raise AssertionError(
            "expected NANO_NCCL_RDMA_SEND_MODE env for bounce|direct"
        )
    if not re.search(
        r"return\s+RdmaSendProxy::SendMode::Direct|"
        r"SendMode\s*::\s*Direct",
        text,
    ):
        raise AssertionError("expected Direct send mode in proxy source")
    # Default when env unset must be Direct.
    parse = re.search(
        r"parse_send_mode_env\(\)\s*\{(?P<body>.*?)\n\}",
        text,
        re.S,
    )
    if parse is None:
        raise AssertionError("could not locate parse_send_mode_env")
    parse_body = parse.group("body")
    if not re.search(
        r"env\s*==\s*nullptr.*?return\s+RdmaSendProxy::SendMode::Direct",
        parse_body,
        re.S,
    ):
        raise AssertionError("default send mode (env unset) must be Direct")

    if not re.search(r"\b(?:std::)?(?:memcpy|copy_n|copy)\s*\(", body):
        raise AssertionError("bounce path must retain host payload copy")
    if not re.search(r"sge\.lkey\s*=\s*bounce_mr_->lkey", body):
        raise AssertionError("bounce path must assign sge.lkey = bounce_mr_->lkey")
    if not re.search(r"sge\.lkey\s*=\s*fifo_mr_->lkey", body):
        raise AssertionError("direct path must assign sge.lkey = fifo_mr_->lkey")
    if not re.search(
        r"sge\.addr\s*=\s*reinterpret_cast<[^>]+>\(\s*fifo_\.data\s*\+",
        body,
    ):
        raise AssertionError("direct path must post SGE from fifo_.data")

    print("rdma_proxy_send_visibility_static=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"rdma_proxy_send_visibility_static failed: {error}", file=sys.stderr)
        raise SystemExit(1)
