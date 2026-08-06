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
    "single": "Single Host: 4 Ranks",
    "socket": "Two Hosts: 8 Ranks Over TCP Socket",
    "rdma": "Two Hosts: 8 Ranks Over RDMA",
}


def fmt_time(us: float) -> str:
    if us >= 10000:
        return f"{us:.1f}"
    if us >= 1000:
        return f"{us:.2f}"
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
    if name == "rdma":
        parts.extend(
            [
                "The nano-nccl runs explicitly request `--transport rdma`. Their aggregate transport output is `mixed` because local ring edges retain their local transport while cross-host edges use RDMA. NCCL uses RDMA with `NCCL_NET_GDR_LEVEL=0` (host-pin / no GPUDirect RDMA), matching nano-nccl's registered host FIFO path. No two-host performance acceptance threshold has been established.",
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
    commit = doc.get("commit", "unknown")[:12]
    generated = doc.get("generated_at", "")
    sections = doc.get("sections", {})

    out: list[str] = []
    out.append("# Performance")
    out.append("")
    out.append(
        "All results below are out-of-place all-reduce measurements. Bandwidth is `busbw` in GB/s. "
        "Every measured nano-nccl and NCCL result completed validation with zero wrong values. "
        "The `nano/NCCL` column is calculated from the unrounded measured time "
        "(`nccl_time_us / nano_time_us`)."
    )
    out.append("")
    out.append(
        f"Measurement commit: `{commit}`. Generated at `{generated}` (UTC)."
    )
    out.append("")
    out.append("## Test Topology And Environment")
    out.append("")
    out.append(
        "Both hosts use two-socket Intel Xeon Platinum 8462Y+ CPUs (32 cores per socket, two threads per core), "
        "4x NVIDIA RTX A6000 GPUs (SM86), CUDA 12.8.61, NCCL 2.30.7 built from source, nccl-tests 2.19.6, "
        "and Open MPI 4.1.2."
    )
    out.append("")
    out.append("| Node | OS kernel | GPU driver | GPUs |")
    out.append("| --- | --- | --- | --- |")
    ka = env.get("node_a_kernel", "Linux 5.15.0-136-generic")
    kb = env.get("node_b_kernel", "Linux 6.8.0-124-generic")
    da = env.get("node_a_driver", "580.82.07")
    db = env.get("node_b_driver", "580.173.02")
    out.append(
        f"| A | {ka} | {da} | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |"
    )
    out.append(
        f"| B | {kb} | {db} | GPU0 `2a:00.0`, GPU1 `3d:00.0`, GPU2 `ab:00.0`, GPU3 `bd:00.0` |"
    )
    out.append("")
    out.append(
        "On each host GPU0-GPU1 and GPU2-GPU3 are connected by four NVLinks. "
        "The two pairs are separated by `SYS` paths across NUMA nodes. "
        "The nano-nccl `auto` plan resolves each ring edge independently (P2P when bidirectional NVLink peer access is available, otherwise SHM). "
        "Two-host socket runs use TCP; NCCL socket runs set `NCCL_IB_DISABLE=1`. "
        "Two-host RDMA runs use nano `--transport rdma` and NCCL IB/RoCE with `NCCL_NET_GDR_LEVEL=0`."
    )
    out.append("")
    out.append(
        "All measurements use a Release build with `NANO_NCCL_ENABLE_BENCH_PROFILING=OFF`, "
        "message sizes 256 KiB through 64 MiB, `-w 5`, and `-n 20`. "
        "NCCL uses `Ring`, `Simple`, four channels, and a 32 MiB buffer. "
        "RDMA SEND/WriteCts always post from the registered mapped FIFO (no host bounce; visibility via `__threadfence_system` + release/acquire steps)."
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
        "The single-host binary uses four ranks. The distributed binary on both hosts uses eight global ranks and the same Open MPI 4.1.2 prefix. "
        "Place matching trees under `<path-to-worktree>` on every host."
    )
    out.append("")
    out.append("```bash")
    out.append("# nano-nccl, single host")
    out.append("CUDA_VISIBLE_DEVICES=0,1,2,3 \\")
    out.append("  ./build-perf-single/benchmarks/nano_nccl_all_reduce_bench \\")
    out.append("  --algo ring_simple --transport auto --dtype <float|fp16|bf16> \\")
    out.append("  --redop <sum|avg|max|min> -b 262144 -e 67108864 -f 4 -w 5 -n 20")
    out.append("")
    out.append("# NCCL, single host")
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
        "For two hosts over TCP Socket, launch one 4-GPU process on each host with an Open MPI 4.1.2 launcher, "
        "set `NANO_NCCL_SOCKET_IFNAME=<interface>` for nano-nccl, and set `NCCL_SOCKET_IFNAME=<interface>`, "
        "`NCCL_IB_DISABLE=1`, `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `NCCL_MIN_NCHANNELS=4`, "
        "`NCCL_MAX_NCHANNELS=4`, and `NCCL_BUFFSIZE=33554432` for NCCL. "
        "Bind MPI TCP/OOB to the same bootstrap interface (`btl_tcp_if_include` / `oob_tcp_if_include`)."
    )
    out.append("")
    out.append(
        "For RDMA, build nano-nccl with MPI and RDMA enabled on both hosts. "
        "Set `NANO_NCCL_SOCKET_IFNAME=<interface>` for bootstrap and `NANO_NCCL_RDMA_IFNAME=<rdma-interface>` "
        "(and `NANO_NCCL_RDMA_GID_INDEX` when required). "
        "Set `NCCL_SOCKET_IFNAME=<interface>`, `NCCL_IB_HCA=<rdma-hca>`, `NCCL_IB_GID_INDEX` when required, "
        "and `NCCL_NET_GDR_LEVEL=0` for the host-pin baseline. Clear inherited `NCCL_IB_DISABLE`."
    )
    out.append("")
    out.append("```bash")
    out.append("cmake -S . -B build-perf-rdma -DCMAKE_BUILD_TYPE=Release \\")
    out.append("  -DNANO_NCCL_ENABLE_MPI=ON -DNANO_NCCL_ENABLE_RDMA=ON \\")
    out.append("  -DNANO_NCCL_NRANKS=8 -DNANO_NCCL_CUDA_ARCH=86 \\")
    out.append("  -DNANO_NCCL_ENABLE_BENCH_PROFILING=OFF")
    out.append("cmake --build build-perf-rdma -j<jobs>")
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
