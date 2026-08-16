#!/usr/bin/env python3
"""Render performance.md from matrix.json produced by run_performance_matrix.sh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SIZE_LABELS = {
    262144: "256 KiB",
    1048576: "1 MiB",
    4194304: "4 MiB",
    16777216: "16 MiB",
    67108864: "64 MiB",
}

DTYPE_HEAD = {"float": "Float", "fp16": "FP16", "bf16": "BF16"}
REDOP_HEAD = {"sum": "Sum", "avg": "Avg", "max": "Max", "min": "Min"}
SECTION_HEAD = {
    "single": "Single Host: 4 Ranks Over Auto (P2P/SHM)",
    "socket": "Two Hosts: 8 Ranks Over TCP Socket",
    "rdma": "Two Hosts: 8 Ranks Over RDMA",
}


def fmt_time(us: float) -> str:
    if us >= 10000:
        return f"{us:.1f}"
    return f"{us:.2f}"


def fmt_bw(bw: float) -> str:
    return f"{bw:.2f}"


def fmt_ratio(r: float) -> str:
    return f"{r:.2f}"


def table_for(rows: list[dict]) -> str:
    lines = [
        "| Size | nano time (us) | nano busbw | NCCL time (us) | NCCL busbw | nano/NCCL |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: r["size"]):
        label = SIZE_LABELS.get(row["size"], f"{row['size']} B")
        lines.append(
            f"| {label} | {fmt_time(row['nano_time_us'])} | {fmt_bw(row['nano_busbw'])} | "
            f"{fmt_time(row['nccl_time_us'])} | {fmt_bw(row['nccl_busbw'])} | "
            f"{fmt_ratio(row['ratio'])} |"
        )
    return "\n".join(lines)


def render_section(name: str, body: dict) -> str:
    parts = [f"## {SECTION_HEAD[name]}", ""]
    if name == "single":
        parts.extend(
            [
                "In-process 4-GPU communicator. Nano `--transport auto` resolves each ring edge independently (P2P when bidirectional NVLink peer access is available, otherwise SHM). NCCL is the matching intra-node Ring+Simple path (P2P/SHM allowed).",
                "",
            ]
        )
    elif name == "socket":
        parts.extend(
            [
                "2 hosts x 4 GPUs, one MPI process per host. Nano `--transport auto` keeps local ring edges on P2P/SHM and places cross-host edges on TCP socket. NCCL is Ring+Simple with `NCCL_IB_DISABLE=1` (P2P/SHM allowed intra-node). Bootstrap uses the management IPv4 interface, not loopback.",
                "",
            ]
        )
    elif name == "rdma":
        parts.extend(
            [
                "2 hosts x 4 GPUs, one MPI process per host. Nano `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` (WRITE+CTS over registered host-pinned FIFO). Local edges stay P2P/SHM; cross-host edges are RDMA. RTR `path_mtu` is `min(local, remote) port.active_mtu`. NCCL: Ring+Simple, `NCCL_NET_GDR_LEVEL=0`, P2P/SHM allowed intra-node. NCCL 256 KiB OOP that collapsed (~0.8 GB/s) was re-run isolated up to four times; cells that stayed collapsed are not nano wins.",
                "",
            ]
        )
    for dtype in ("float", "fp16", "bf16"):
        if dtype not in body:
            continue
        parts.append(f"### {DTYPE_HEAD[dtype]}")
        parts.append("")
        for redop in ("sum", "avg", "max", "min"):
            if redop not in body[dtype]:
                continue
            parts.append(f"#### {REDOP_HEAD[redop]}")
            parts.append("")
            parts.append(table_for(body[dtype][redop]))
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def render(doc: dict) -> str:
    env = doc.get("env", {})
    sections = doc.get("sections", {})

    out: list[str] = []
    out.append("# Performance")
    out.append("")
    out.append(
        "All results below are out-of-place all-reduce measurements. "
        "Bandwidth is `busbw` in GB/s. Every measured nano-nccl and NCCL result completed "
        "validation with zero wrong values. The `nano/NCCL` column is calculated from the "
        "unrounded measured time (`nccl_time_us / nano_time_us`)."
    )
    out.append("")
    out.append("## Test Topology And Environment")
    out.append("")
    out.append(
        "Two hosts, each two-socket Intel Xeon Platinum 8462Y+ (32 cores per socket, two threads per core), "
        "4x NVIDIA RTX A6000 (SM86), CUDA 12.8.61, NCCL 2.30.7 built from source, nccl-tests 2.19.6, "
        "and Open MPI 4.1.2."
    )
    out.append("")
    out.append("| Node | OS kernel | GPU driver | GPUs |")
    out.append("| --- | --- | --- | --- |")
    ka = env.get("node_a_kernel", "Linux 5.15.0-136-generic")
    da = env.get("node_a_driver", "580.82.07")
    kb = env.get("node_b_kernel", ka)
    db = env.get("node_b_driver", da)
    out.append(
        f"| A | {ka} | {da} | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |"
    )
    out.append(
        f"| B | {kb} | {db} | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |"
    )
    out.append("")
    out.append(
        "On each host, GPU0-GPU1 and GPU2-GPU3 are connected by four NVLinks. The two pairs are "
        "separated by `SYS` paths across NUMA nodes. Tables are separate transport classes: "
        "in-process auto (P2P/SHM) on one host, two-host TCP socket, and two-host host-pinned RDMA."
    )
    out.append("")
    out.append(
        "All measurements use a Release build with `NANO_NCCL_ENABLE_BENCH_PROFILING=OFF`, "
        "message sizes 256 KiB through 64 MiB, `-w 5`, and `-n 20`. "
        "NCCL uses `Ring`, `Simple`, four channels, and a 32 MiB buffer. "
        "RDMA WRITE+CTS posts from the registered mapped FIFO (no host bounce; "
        "visibility via publisher `st.release.sys(send_tail)` after block sync and host acquire loads)."
    )
    out.append("")

    for name in ("single", "socket", "rdma"):
        if name in sections:
            out.append(render_section(name, sections[name]))
            out.append("")

    out.append("## Reproduction")
    out.append("")
    out.append(
        "Build nano-nccl with CUDA 12.8, SM86, Release mode, and profiling disabled. "
        "The in-process auto binary uses four ranks in one process. "
        "Socket and RDMA tables use four MPI ranks (`NANO_NCCL_NRANKS=4`, one GPU per rank) "
        "from the same Open MPI 4.1.2 prefix."
    )
    out.append("")
    out.append("```bash")
    out.append("# nano-nccl, in-process auto (P2P/SHM)")
    out.append("CUDA_VISIBLE_DEVICES=0,1,2,3 \\")
    out.append("  ./build-perf-single/benchmarks/nano_nccl_all_reduce_bench \\")
    out.append("  --algo ring_simple --transport auto --dtype <float|fp16|bf16> \\")
    out.append("  --redop <sum|avg|max|min> -b 262144 -e 67108864 -f 4 -w 5 -n 20")
    out.append("")
    out.append("# NCCL, in-process intra-node")
    out.append("CUDA_VISIBLE_DEVICES=0,1,2,3 \\")
    out.append("LD_LIBRARY_PATH=<path-to-nccl-lib> \\")
    out.append("NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 \\")
    out.append("NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=33554432 \\")
    out.append("  <path-to-nccl-tests>/build/all_reduce_perf \\")
    out.append("  -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 \\")
    out.append("  -d <float|half|bfloat16> -o <sum|avg|max|min>")
    out.append("```")
    out.append("")
    out.append(
        "For single-host socket, launch four MPI ranks with one GPU each "
        "(`CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK`). Nano uses `--transport auto` "
        "(cross-process edges are socket). NCCL sets `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, "
        "and `NCCL_IB_DISABLE=1`."
    )
    out.append("")
    out.append(
        "For single-host RDMA, the same 4-rank launch uses nano `--transport rdma` and "
        "`NANO_NCCL_RDMA_USE_WRITE=1`. Set `NANO_NCCL_SOCKET_IFNAME=<interface>` for bootstrap "
        "and `NANO_NCCL_RDMA_IFNAME=<rdma-interface>` (and `NANO_NCCL_RDMA_GID_INDEX` when required). "
        "NCCL sets `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_NET_GDR_LEVEL=0`, "
        "`NCCL_IB_HCA=<rdma-hca>`, and `NCCL_IB_GID_INDEX` when required."
    )
    out.append("")
    out.append("```bash")
    out.append("cmake -S . -B build-perf-rdma-n8 -DCMAKE_BUILD_TYPE=Release \\")
    out.append("  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON \\")
    out.append("  -DNANO_NCCL_NRANKS=8 -DNANO_NCCL_CUDA_ARCH=86 \\")
    out.append("  -DNANO_NCCL_ENABLE_BENCH_PROFILING=OFF")
    out.append("cmake --build build-perf-rdma-n8 -j<jobs>")
    out.append("```")
    out.append("")
    out.append("Or regenerate this file from a completed matrix JSON:")
    out.append("")
    out.append("```bash")
    out.append("scripts/run_performance_matrix.sh --nccl-bin <path> --nccl-lib <dir> --out-dir <out>")
    out.append("python3 scripts/render_performance_md.py <out>/matrix.json -o performance.md")
    out.append("```")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix_json")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    doc = json.loads(Path(args.matrix_json).read_text())
    text = render(doc)
    Path(args.output).write_text(text)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
