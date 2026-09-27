#!/usr/bin/env python3
"""Render performance.md from matrix.json produced by run_performance_matrix.sh."""

from __future__ import annotations

import argparse
import json
import math
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
    "rdma_gdr": "Two Hosts: 8 Ranks Over RDMA (GDR)",
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


def render_section(name: str, body: dict, matched_timing: bool) -> str:
    parts = [f"## {SECTION_HEAD[name]}", ""]
    if name == "single":
        parts.extend(
            [
                ("Four local MPI processes, one GPU per process. " if matched_timing
                 else "In-process 4-GPU communicator. ") +
                "Nano `--transport auto` resolves each ring edge independently (P2P when bidirectional NVLink peer access is available, otherwise SHM). NCCL is the matching intra-node Ring+Simple path (P2P/SHM allowed).",
                "",
            ]
        )
    elif name == "socket":
        parts.extend(
            [
                ("2 hosts x 4 GPUs, one MPI process per GPU. " if matched_timing
                 else "2 hosts x 4 GPUs, one MPI process per host. ") +
                "Nano `--transport auto` keeps local ring edges on P2P/SHM and places cross-host edges on TCP socket. NCCL is Ring+Simple with `NCCL_IB_DISABLE=1` (P2P/SHM allowed intra-node). Bootstrap uses the management IPv4 interface, not loopback.",
                "",
            ]
        )
    elif name == "rdma":
        parts.extend(
            [
                ("2 hosts x 4 GPUs, one MPI process per GPU. " if matched_timing
                 else "2 hosts x 4 GPUs, one MPI process per host. ") +
                "Nano `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` (WRITE+CTS over registered host-pinned FIFO). Local edges stay P2P/SHM; cross-host edges are RDMA. RTR `path_mtu` is `min(local, remote) port.active_mtu`. NCCL: Ring+Simple, `NCCL_NET_GDR_LEVEL=0`, P2P/SHM allowed intra-node.",
                "",
            ]
        )
    elif name == "rdma_gdr":
        ratios = [
            row["ratio"]
            for dtype in ("float", "fp16", "bf16")
            for redop in ("sum", "avg", "max", "min")
            for row in body.get(dtype, {}).get(redop, [])
        ]
        parts.extend(
            [
                "Same 2x4 topology as the host-pinned RDMA table. Nano `--transport rdma` with `NANO_NCCL_RDMA_USE_WRITE=1` and `NANO_NCCL_RDMA_GDR=1` (WRITE+CTS from a registered GPU FIFO; host proxy still posts). After a receive CQE, nano flushes third-party GDR writes before publishing `recv_tail` when the device lacks native GPU/RDMA write ordering. NCCL: Ring+Simple with `NCCL_NET_GDR_LEVEL=SYS`; debug logs confirmed `/GDRDMA` on the cross-host edges. Each cell is the median of five complete, alternating-order repetitions; no isolated retries replace matrix samples. This is host-proxy GDR, not GPU-initiated IBGDA.",
                "",
            ]
        )
        if ratios:
            geomean = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
            parts.extend(
                [
                    f"Across {len(ratios)} dtype/op/size cells, nano/NCCL busbw geomean is {geomean:.3f}; the minimum cell is {min(ratios):.3f}, with {sum(ratio < 0.90 for ratio in ratios)} cells below 0.90.",
                    "",
                    "No accepted GDR nano baseline exists, so the formal 3% baseline-regression gate is not adjudicated. Cells below NCCL remain `Unknown` performance gaps until controlled causal experiments explain them. The existing packed FP16/BF16 `max`/`min` single-NaN propagation failure is unchanged in GDR=0 and GDR=1; these tables validate their ordinary benchmark inputs but do not establish complete dtype/redop contract correctness.",
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
    timing = env.get("benchmark_timing")
    matched_timing = bool(timing) and all((
        timing.get("nano_wait_mode") == "query",
        timing.get("nccl_tests_blocking_coll") == 2,
        timing.get("nccl_tests_average") == 3,
        timing.get("nccl_tests_agg_iters") == 1,
        env.get("rank_layout", "").startswith("one MPI process per GPU"),
    ))

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
    if matched_timing:
        out.append(
            "Benchmark timing contract: nano waits with `--wait-mode "
            f"{timing.get('nano_wait_mode', 'unknown')}` after each collective; "
            "nccl-tests uses `-z "
            f"{timing.get('nccl_tests_blocking_coll', 'unknown')} -a "
            f"{timing.get('nccl_tests_average', 'unknown')} -m "
            f"{timing.get('nccl_tests_agg_iters', 'unknown')}`. Both use "
            f"`-w {timing.get('warmup_iterations', 'unknown')} -n "
            f"{timing.get('timed_iterations', 'unknown')}`. This means per-collective "
            "completion and maximum-rank aggregation, with one collective per timing sample."
        )
    else:
        out.append(
            "**This matrix does not record the matched timing and rank-layout "
            "contract.** Do not use its ratios for formal nano/NCCL performance "
            "acceptance."
        )
    out.append("")
    if matched_timing:
        out.append(f"Rank-to-GPU mapping: {env.get('rank_device_map', 'not recorded')}.")
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
        + ("four-process auto (P2P/SHM) on one host" if matched_timing
           else "in-process auto (P2P/SHM) on one host")
        + ", two-host TCP socket and host-pinned RDMA"
        + (", and two-host RDMA with GPUDirect (device FIFO)." if "rdma_gdr" in sections else ".")
    )
    out.append("")
    measurement = (
        "All measurements use a Release build with `NANO_NCCL_ENABLE_BENCH_PROFILING=OFF`, "
        "message sizes 256 KiB through 64 MiB, `-w 5`, and `-n 20`. "
    )
    if matched_timing:
        measurement += (
            "Nano uses `--wait-mode query`; nccl-tests uses `-z 2 -a 3 -m 1` for "
            "per-collective completion, maximum-rank aggregation, and one collective per timing sample. "
        )
    measurement += (
        "NCCL uses `Ring`, `Simple`, four channels, and a 32 MiB buffer. "
    )
    if "rdma_gdr" in sections:
        measurement += (
            "RDMA WRITE+CTS posts from the registered mapped FIFO (no host bounce; "
            "publication via publisher `st.release.sys(send_tail)` after block sync and host acquire loads). "
            "The GDR table reports the median of five full-matrix repetitions for each implementation."
        )
    out.append(measurement)
    out.append("")

    for name in ("single", "socket", "rdma", "rdma_gdr"):
        if name in sections:
            out.append(render_section(name, sections[name], matched_timing))
            out.append("")

    out.append("## Reproduction")
    out.append("")
    if not matched_timing:
        out.append(
            "The commands below use the corrected timing contract. They do not "
            "reproduce the historical timing values in this metadata-free matrix."
        )
        out.append("")
    out.append(
        "Build nano-nccl with CUDA 12.8, SM86, Release mode, and profiling disabled. "
        "Run one MPI process per GPU for both implementations: four local ranks "
        "for the single-host section or four ranks per host with a global "
        "`NANO_NCCL_NRANKS=8` for the two-host sections. Build with MPI and use "
        "the same Open MPI 4.1.2 prefix."
    )
    out.append("")
    out.append("```bash")
    out.append("# nano-nccl, four local MPI ranks over auto (P2P/SHM)")
    out.append("CUDA_VISIBLE_DEVICES=0,1,2,3 \\")
    out.append("  mpirun --bind-to none -np 4 ./build-perf-single/benchmarks/nano_nccl_all_reduce_bench \\")
    out.append("  --algo ring_simple --transport auto --dtype <float|fp16|bf16> \\")
    out.append("  --redop <sum|avg|max|min> -b 262144 -e 67108864 -f 4 -w 5 -n 20 --wait-mode query")
    out.append("")
    out.append("# NCCL, four local MPI ranks, one GPU per process")
    out.append("CUDA_VISIBLE_DEVICES=0,1,2,3 \\")
    out.append("LD_LIBRARY_PATH=<path-to-nccl-lib> \\")
    out.append("NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 \\")
    out.append("NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=33554432 \\")
    out.append("  mpirun --bind-to none -np 4 <path-to-nccl-tests>/build/all_reduce_perf \\")
    out.append("  -b 262144 -e 67108864 -f 4 -g 1 -w 5 -n 20 -z 2 -a 3 -m 1 \\")
    out.append("  -d <float|half|bfloat16> -o <sum|avg|max|min>")
    out.append("```")
    out.append("")
    out.append(
        "For the socket runs, launch four MPI processes with four visible GPUs per host. "
        "Nano uses `--transport auto` "
        "(cross-host edges are socket). NCCL sets `NCCL_IB_DISABLE=1`; local P2P/SHM remain enabled."
    )
    out.append("")
    out.append(
        "For host-pinned RDMA, use nano `--transport rdma` with "
        "`NANO_NCCL_RDMA_USE_WRITE=1`. Set `NANO_NCCL_SOCKET_IFNAME=<interface>` for bootstrap "
        "and `NANO_NCCL_RDMA_IFNAME=<rdma-interface>` (and `NANO_NCCL_RDMA_GID_INDEX` when required). "
        "NCCL sets `NCCL_NET_GDR_LEVEL=0`, "
        "`NCCL_IB_HCA=<rdma-hca>`, and `NCCL_IB_GID_INDEX` when required."
    )
    out.append("")
    out.append(
        "For the two-host GDR matrix, build both hosts for eight ranks and launch four MPI processes "
        "per host with four visible GPUs. Add `NANO_NCCL_RDMA_GDR=1` for nano and use "
        "`NCCL_NET_GDR_LEVEL=SYS` for NCCL. Run `nano_nccl_rdma_gdr` before the matrix; nano's "
        "explicit GDR request fails instead of falling back. The benchmark prints the aggregate "
        "transport and each directed edge's backend and RDMA FIFO placement (`gdr`, `host-pinned`, "
        "or `n/a`). Verify `/GDRDMA` in NCCL debug output."
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
