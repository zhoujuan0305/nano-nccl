#!/usr/bin/env python3
"""Run matched two-host, eight-GPU nano-nccl vs nccl-tests GDR trials.

The runner records commands/configuration and one raw log set per repetition.
It never builds binaries and dry-run validates/prints the plan without MPI/GPU use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DTYPE_MAP = {"float": "float", "fp16": "half", "bf16": "bfloat16"}
COLLECTIVES = ("allreduce", "reducescatter", "allgather")
REDOPS = ("sum", "avg", "max", "min")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host-a", required=True)
    p.add_argument("--host-b", required=True)
    p.add_argument("--mpi-prefix", required=True, help="Open MPI installation prefix")
    for c in COLLECTIVES:
        p.add_argument(f"--nano-{c}", required=True, help=f"nano {c} benchmark executable")
        p.add_argument(f"--nccl-{c}", required=True, help=f"nccl-tests {c}_perf executable")
    p.add_argument("--nccl-lib", required=True, help="directory containing the intended libnccl")
    p.add_argument("--socket-if", required=True, help="MPI/nano TCP bootstrap interface")
    p.add_argument("--nccl-socket-if", help="NCCL bootstrap interface (default: --rdma-if)")
    p.add_argument("--rdma-if", required=True, help="NANO_NCCL_RDMA_IFNAME")
    p.add_argument("--rdma-hca", required=True, help="NCCL_IB_HCA")
    p.add_argument("--gid-index", required=True, type=int)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--dtypes", default="float,fp16,bf16",
                   help="comma list: float,fp16,bf16")
    p.add_argument("--collectives", default=",".join(COLLECTIVES),
                   help="comma list: allreduce,reducescatter,allgather")
    p.add_argument("--redops", default=",".join(REDOPS),
                   help="comma list used by AllReduce/ReduceScatter")
    p.add_argument("--min-bytes", type=int, default=262144)
    p.add_argument("--max-bytes", type=int, default=67108864)
    p.add_argument("--factor", type=int, default=4)
    p.add_argument("-w", "--warmup", type=int, default=5)
    p.add_argument("-n", "--iterations", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--cuda-visible-devices", default="0,1,2,3")
    p.add_argument("--timeout-seconds", type=int, default=240)
    p.add_argument("--dry-run", action="store_true")
    return p


def csv_values(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{label} must be a non-empty comma list without duplicates")
    invalid = [item for item in result if item not in allowed]
    if invalid:
        raise ValueError(f"unsupported {label}: {', '.join(invalid)}")
    return result


def redact(text: str, sensitive: list[str]) -> str:
    for value, placeholder in sensitive:
        if value:
            text = text.replace(value, placeholder)
    return text


def env_for(args: argparse.Namespace, kind: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(Path(args.mpi_prefix) / "bin") + os.pathsep + env.get("PATH", "")
    mpi_lib = str(Path(args.mpi_prefix) / "lib")
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(x for x in (mpi_lib, args.nccl_lib, old_ld) if x)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["NANO_NCCL_SOCKET_IFNAME"] = args.socket_if
    env["NANO_NCCL_RDMA_IFNAME"] = args.rdma_if
    env["NANO_NCCL_RDMA_GID_INDEX"] = str(args.gid_index)
    env["NANO_NCCL_RDMA_GDR"] = "1"
    env["NANO_NCCL_RDMA_USE_WRITE"] = "1"
    env["NCCL_SOCKET_IFNAME"] = args.nccl_socket_if or args.rdma_if
    env["NCCL_IB_HCA"] = args.rdma_hca
    env["NCCL_IB_GID_INDEX"] = str(args.gid_index)
    env["NCCL_NET_GDR_LEVEL"] = "SYS"
    env["NCCL_NET_GDR_READ"] = "1"
    env["NCCL_IB_DISABLE"] = "0"
    env["NCCL_CUMEM_ENABLE"] = "0"
    env["NCCL_ALGO"] = "Ring"
    env["NCCL_PROTO"] = "Simple"
    env["NCCL_MIN_NCHANNELS"] = "4"
    env["NCCL_MAX_NCHANNELS"] = "4"
    env["NCCL_BUFFSIZE"] = "33554432"
    env["NCCL_P2P_DISABLE"] = "0"
    env["NCCL_SHM_DISABLE"] = "0"
    if kind == "nccl":
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
    return env


def command_for(args: argparse.Namespace, collective: str, dtype: str,
                redop: str | None, kind: str) -> list[str]:
    binary = getattr(args, f"{kind}_{collective}")
    mpi = str(Path(args.mpi_prefix) / "bin" / "mpirun")
    exported = (["PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                 "NANO_NCCL_SOCKET_IFNAME", "NANO_NCCL_RDMA_IFNAME",
                 "NANO_NCCL_RDMA_GID_INDEX", "NANO_NCCL_RDMA_GDR",
                 "NANO_NCCL_RDMA_USE_WRITE"] if kind == "nano" else
                ["PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                 "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
                 "NCCL_NET_GDR_LEVEL", "NCCL_NET_GDR_READ", "NCCL_IB_DISABLE",
                 "NCCL_CUMEM_ENABLE", "NCCL_ALGO",
                 "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
                 "NCCL_BUFFSIZE", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
                 "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS"])
    exports = [item for key in exported for item in ("-x", key)]
    common_mpi = [mpi, "--prefix", args.mpi_prefix, "--bind-to", "none",
                  "--mca", "pml", "ob1", "--mca", "btl", "self,vader,tcp",
                  "--mca", "btl_tcp_if_include", args.socket_if,
                  "--mca", "oob_tcp_if_include", args.socket_if,
                  "--host", f"{args.host_a}:4", "-np", "4", *exports]
    second = [":", "--host", f"{args.host_b}:4", "-np", "4", *exports]
    if kind == "nano":
        bench = [binary]
        if collective == "allreduce":
            bench += ["--algo", "ring_simple"]
        else:
            bench += ["--collective", collective]
        bench += ["--transport", "rdma", "--wait-mode", "query", "--dtype", dtype,
                 "-b", str(args.min_bytes), "-e", str(args.max_bytes),
                 "-f", str(args.factor), "-w", str(args.warmup),
                 "-n", str(args.iterations)]
        if redop is not None:
            bench += ["--redop", redop]
    else:
        bench = [binary, "-b", str(args.min_bytes), "-e", str(args.max_bytes),
                 "-f", str(args.factor), "-g", "1", "-w", str(args.warmup),
                 "-n", str(args.iterations), "-d", DTYPE_MAP[dtype],
                 "-z", "2", "-a", "3", "-m", "1"]
        if redop is not None:
            bench += ["-o", redop]
    return common_mpi + bench + second + bench


def parse_rows(path: Path, kind: str, dtype: str, redop: str | None) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        try:
            if kind == "nano":
                # nano row: algo dtype redop transport size count time algbw busbw wrong max_abs
                if len(fields) < 11 or fields[0] != "ring_simple":
                    continue
                if fields[1] != dtype or (redop is not None and fields[2] != redop):
                    continue
                size, time_us, busbw, wrong = int(fields[4]), float(fields[6]), float(fields[8]), int(fields[9])
            else:
                # nccl-tests row: size count type redop root time algbw busbw wrong
                if len(fields) < 9 or fields[2] != DTYPE_MAP[dtype]:
                    continue
                if redop is not None and fields[3] != redop:
                    continue
                size, time_us, busbw, wrong = int(fields[0]), float(fields[5]), float(fields[7]), int(fields[8])
        except (ValueError, IndexError):
            continue
        rows[size] = {"time_us": time_us, "busbw_gbs": busbw, "wrong": wrong}
    return rows


def verify_gdr(log: Path, kind: str) -> bool:
    text = log.read_text(errors="replace")
    if kind == "nano":
        return text.count("transport=rdma rdma_memory=gdr") == 2 and "wait_mode=query" in text
    # On this topology each direction has multiple GDRDMA channels.
    return text.count("/GDRDMA") >= 16 and "NCCL WARN" not in text and "Blocking Enabled" in text


def expected_sizes(args: argparse.Namespace) -> list[int]:
    sizes = []
    size = args.min_bytes
    while size <= args.max_bytes:
        sizes.append(size)
        size *= args.factor
    return sizes


def check_gpu_health(args: argparse.Namespace) -> None:
    for host in (args.host_a, args.host_b):
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                   "timeout 15s nvidia-smi --query-gpu=index,gpu_recovery_action --format=csv,noheader"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=25)
        lines = result.stdout.splitlines()
        if result.returncode or len(lines) != 4 or any(not line.strip().endswith("None") for line in lines):
            raise RuntimeError(f"GPU recovery state is not healthy on {host}: {result.stdout} {result.stderr}")


def binary_digests(args: argparse.Namespace, collectives: list[str]) -> dict[str, dict[str, str]]:
    digests = {}
    for kind in ("nano", "nccl"):
        for collective in collectives:
            name = f"{kind}_{collective}"
            path = Path(getattr(args, name))
            local = hashlib.sha256(path.read_bytes()).hexdigest()
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                 args.host_b, "sha256sum -- " + shlex.quote(str(path))],
                capture_output=True, text=True, timeout=30)
            if result.returncode or not result.stdout.strip():
                raise RuntimeError(f"cannot fingerprint {name} on host B: {result.stderr}")
            remote = result.stdout.split()[0]
            digests[name] = {"host_a_sha256": local, "host_b_sha256": remote}
    return digests


def main() -> int:
    args = parser().parse_args()
    try:
        dtypes = csv_values(args.dtypes, tuple(DTYPE_MAP), "dtypes")
        collectives = csv_values(args.collectives, COLLECTIVES, "collectives")
        redops = csv_values(args.redops, REDOPS, "redops")
        if args.repeats < 1 or args.warmup < 0 or args.iterations < 1:
            raise ValueError("repeats/iterations must be positive and warmup non-negative")
        if args.min_bytes <= 0 or args.max_bytes < args.min_bytes or args.factor <= 1:
            raise ValueError("invalid message size range/factor")
        if args.host_a == args.host_b:
            raise ValueError("--host-a and --host-b must identify different hosts")
        if len(args.cuda_visible_devices.split(",")) != 4:
            raise ValueError("each host must expose exactly four CUDA devices")
        if args.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be positive")
    except ValueError as exc:
        parser().error(str(exc))

    mpi = Path(args.mpi_prefix) / "bin" / "mpirun"
    if not args.dry_run:
        for name in ("nano", "nccl"):
            for c in collectives:
                binary = Path(getattr(args, f"{name}_{c}"))
                if not binary.is_file() or not os.access(binary, os.X_OK):
                    parser().error(f"{name} {c} binary is not executable: {binary}")
                setattr(args, f"{name}_{c}", str(binary.resolve()))
        if not mpi.is_file() or not os.access(mpi, os.X_OK):
            parser().error(f"mpirun is not executable: {mpi}")
        if not Path(args.nccl_lib).is_dir():
            parser().error("--nccl-lib must be an existing directory")

    cases = [(c, d, r if c in ("allreduce", "reducescatter") else None)
             for c in collectives for d in dtypes
             for r in (redops if c in ("allreduce", "reducescatter") else [None])]
    secrets = [(args.host_a, "<host-a>"), (args.host_b, "<host-b>"),
               (args.socket_if, "<socket-if>"),
               (args.nccl_socket_if or args.rdma_if, "<nccl-socket-if>"),
               (args.rdma_if, "<rdma-if>"),
               (args.rdma_hca, "<rdma-hca>"), (str(Path(args.mpi_prefix)), "<mpi-prefix>"),
               (str(Path(args.nccl_lib)), "<nccl-lib>")]
    secrets += [(getattr(args, f"{kind}_{c}"), f"<{kind}-{c}-binary>")
                for kind in ("nano", "nccl") for c in collectives]
    plan = []
    for rep in range(1, args.repeats + 1):
        for c, d, r in cases:
            for kind in (("nano", "nccl") if rep % 2 else ("nccl", "nano")):
                cmd = command_for(args, c, d, r, kind)
                plan.append({"repeat": rep, "collective": c, "dtype": d,
                             "redop": r, "kind": kind,
                             "command": redact(shlex.join(cmd), secrets)})
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "hosts": ["<host-a>", "<host-b>"], "mpi_prefix": "<mpi-prefix>",
            "binaries": {f"{k}_{c}": "<benchmark-binary>" for k in ("nano", "nccl") for c in collectives},
            "nccl_lib": "<nccl-lib>", "socket_if": "<socket-if>",
            "nccl_socket_if": "<nccl-socket-if>",
            "rdma_if": "<rdma-if>", "rdma_hca": "<rdma-hca>",
            "gid_index": args.gid_index, "cuda_visible_devices": args.cuda_visible_devices,
            "collectives": collectives, "dtypes": dtypes, "redops": redops,
            "message_bytes": [args.min_bytes, args.max_bytes, args.factor],
            "warmup": args.warmup, "iterations": args.iterations,
            "repeats": args.repeats,
            "nano_environment": {"NANO_NCCL_RDMA_GDR": "1", "NANO_NCCL_RDMA_USE_WRITE": "1",
                                 "wait_mode": "query"},
            "nccl_environment": {"NCCL_NET_GDR_LEVEL": "SYS", "NCCL_NET_GDR_READ": "1",
                                 "NCCL_IB_DISABLE": "0", "NCCL_CUMEM_ENABLE": "0",
                                 "NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple",
                                 "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4",
                                 "NCCL_BUFFSIZE": "33554432"},
            "nccl_tests": {"-z": 2, "-a": 3, "-m": 1, "-g": 1},
            "mpi_ranks_per_host": 4,
            "rank_device_map": "ranks 0-3 host A visible GPUs 0-3; ranks 4-7 host B visible GPUs 0-3",
            "buffer_semantics": "out-of-place",
            "dry_run": args.dry_run,
        },
        "planned_commands": plan,
    }
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    config["configuration"]["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["configuration"]["binary_sha256"] = binary_digests(args, collectives)
    repository = str(Path(__file__).resolve().parents[1])
    local_head = subprocess.check_output(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        text=True).strip()
    remote_head = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         args.host_b, "git -C " + shlex.quote(repository) + " rev-parse HEAD"],
        text=True, timeout=20).strip()
    config["configuration"]["repository_head"] = {
        "host_a": local_head, "host_b": remote_head}
    if local_head != remote_head:
        raise RuntimeError("repository HEAD differs between hosts")

    root = args.out_dir
    logdir = root / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    (root / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    round_manifest = (root / "rounds.jsonl").open("w")
    samples: list[dict[str, Any]] = []
    failures: list[str] = []
    sizes = expected_sizes(args)
    check_gpu_health(args)
    for item in plan:
        rep, c, d, r, kind = (item["repeat"], item["collective"], item["dtype"],
                              item["redop"], item["kind"])
        tag = f"rep{rep:02d}_{c}_{d}_{r or 'none'}_{kind}"
        stdout_path, stderr_path = logdir / f"{tag}.stdout.log", logdir / f"{tag}.stderr.log"
        env = env_for(args, kind)
        command = command_for(args, c, d, r, kind)
        exported = {key: env[key] for key in (
            "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "NANO_NCCL_SOCKET_IFNAME",
            "NANO_NCCL_RDMA_IFNAME", "NANO_NCCL_RDMA_GID_INDEX",
            "NANO_NCCL_RDMA_GDR", "NANO_NCCL_RDMA_USE_WRITE", "NCCL_SOCKET_IFNAME",
            "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_NET_GDR_LEVEL",
            "NCCL_NET_GDR_READ", "NCCL_IB_DISABLE", "NCCL_CUMEM_ENABLE",
            "NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS",
            "NCCL_MAX_NCHANNELS", "NCCL_BUFFSIZE", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS")
            if key in env}
        safe_env = {key: redact(value, secrets) for key, value in exported.items()}
        round_manifest.write(json.dumps({"repeat": rep, "collective": c, "dtype": d,
            "redop": r, "kind": kind, "command": item["command"],
            "environment": safe_env}, sort_keys=True) + "\n")
        round_manifest.flush()
        try:
            with stdout_path.open("w") as out, stderr_path.open("w") as err:
                proc = subprocess.run(command, env=env, stdout=out, stderr=err,
                                      text=True, timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            failures.append(f"{tag}: timed out after {args.timeout_seconds}s")
            break
        log_text = stdout_path.read_text(errors="replace") + "\n" + stderr_path.read_text(errors="replace")
        # Combined per-invocation log makes GDR evidence and raw output easy to inspect.
        (logdir / f"{tag}.log").write_text(log_text)
        if proc.returncode != 0:
            failures.append(f"{tag}: exit={proc.returncode}")
            break
        try:
            check_gpu_health(args)
        except RuntimeError as exc:
            failures.append(f"{tag}: {exc}")
            break
        if not verify_gdr(logdir / f"{tag}.log", kind):
            failures.append(f"{tag}: GDR evidence not found")
            break
        parsed = parse_rows(stdout_path, kind, d, r)
        if sorted(parsed) != sizes:
            failures.append(f"{tag}: incomplete sizes: got {sorted(parsed)}, expected {sizes}")
            break
        for size, values in parsed.items():
            if values["wrong"] != 0:
                failures.append(f"{tag}: wrong={values['wrong']} at {size} bytes")
                break
            if values["time_us"] <= 0 or values["busbw_gbs"] <= 0:
                failures.append(f"{tag}: non-positive timing/bandwidth at {size} bytes")
                break
            if kind == "nano":
                samples.append({"repeat": rep, "collective": c, "dtype": d,
                                "redop": r, "size_bytes": size,
                                "nano_time_us": values["time_us"],
                                "nano_busbw_gbs": values["busbw_gbs"],
                                "nano_wrong": values["wrong"]})
            else:
                samples.append({"repeat": rep, "collective": c, "dtype": d,
                                "redop": r, "size_bytes": size,
                                "nccl_time_us": values["time_us"],
                                "nccl_busbw_gbs": values["busbw_gbs"],
                                "nccl_wrong": values["wrong"]})
        if failures:
            break

    round_manifest.close()
    # Pair same-repeat size/semantics observations, then report medians over repeats.
    keys = sorted({(x["collective"], x["dtype"], x["redop"], x["size_bytes"]) for x in samples},
                  key=lambda x: (x[0], x[1], str(x[2]), x[3]))
    aggregate = []
    for c, d, r, size in keys:
        pairs = []
        for rep in range(1, args.repeats + 1):
            n = next((x for x in samples if x["repeat"] == rep and x["collective"] == c and
                      x["dtype"] == d and x["redop"] == r and x["size_bytes"] == size and "nano_time_us" in x), None)
            q = next((x for x in samples if x["repeat"] == rep and x["collective"] == c and
                      x["dtype"] == d and x["redop"] == r and x["size_bytes"] == size and "nccl_time_us" in x), None)
            if n and q:
                pairs.append((n, q))
        if len(pairs) != args.repeats:
            failures.append(f"incomplete matched repeats: {c}/{d}/{r}/{size}")
            continue
        record: dict[str, Any] = {"collective": c, "dtype": d, "redop": r, "size_bytes": size,
                                  "matched_repeats": len(pairs)}
        for key in ("nano_time_us", "nano_busbw_gbs", "nccl_time_us", "nccl_busbw_gbs"):
            record[key + "_median"] = statistics.median(pair[0 if key.startswith("nano") else 1][key]
                                                          for pair in pairs)
        record["nano_over_nccl_busbw_ratio_from_time_medians"] = (
            record["nccl_time_us_median"] / record["nano_time_us_median"])
        record["paired_time_ratio_median"] = statistics.median(
            q["nccl_time_us"] / n["nano_time_us"] for n, q in pairs if n["nano_time_us"] > 0)
        record["nano_wrong"] = 0
        record["nccl_wrong"] = 0
        aggregate.append(record)
    result = {"run_config": config["configuration"], "samples": samples,
              "median_by_case_size": aggregate, "failures": failures}
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"completed: {root}; matched median rows={len(aggregate)}; failures={len(failures)}")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
