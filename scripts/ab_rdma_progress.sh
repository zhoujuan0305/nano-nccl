#!/usr/bin/env bash
# A/B nano RDMA progress modes (shared engine vs dedicated per-proxy threads).
# Calls compare_rdma_no_gdr.sh N times per mode; writes CSV + best/worst/median.
# Host/interface values come from the environment only — never hard-code them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT="${SCRIPT_DIR}/compare_rdma_no_gdr.sh"

NANO_BIN=""
NCCL_BIN=""
NCCL_LIB=""
OUT_DIR=""
RUNS=3
HELP=0

usage() {
    cat <<'EOF'
Usage: ab_rdma_progress.sh --nano-bin PATH --nccl-bin PATH --nccl-lib DIR --out-dir DIR [options]

Options:
  --nano-bin PATH     nano_nccl_all_reduce_bench binary (required)
  --nccl-bin PATH     nccl-tests all_reduce_perf binary (required)
  --nccl-lib DIR      directory containing libnccl.so (required)
  --out-dir DIR       output directory for CSV/logs (required)
  --runs N            repeats per mode (default: 3)
  -h, --help          show this help

Required environment (live run; forwarded to compare_rdma_no_gdr.sh):
  COMPARE_HOST_A
  COMPARE_HOST_B
  NANO_NCCL_SOCKET_IFNAME
  NANO_NCCL_RDMA_IFNAME
  NCCL_SOCKET_IFNAME
  NCCL_IB_HCA

Optional environment:
  NANO_NCCL_RDMA_GID_INDEX
  NANO_NCCL_RDMA_USE_WRITE
  NCCL_IB_GID_INDEX
  CUDA_VISIBLE_DEVICES

Modes:
  NANO_NCCL_RDMA_SHARED_PROGRESS=1  shared progress engine (default)
  NANO_NCCL_RDMA_SHARED_PROGRESS=0  dedicated per-proxy threads

Outputs under --out-dir:
  busbw.csv                 mode,run,size_bytes,nano_busbw,nccl_busbw,ratio
  shared_run<k>/            per-run compare logs
  dedicated_run<k>/
  summary_stats.md          best/worst/median nano_busbw and ratio per size×mode
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nano-bin)
            [[ $# -ge 2 ]] || die "--nano-bin needs a value"
            NANO_BIN="$2"
            shift 2
            ;;
        --nccl-bin)
            [[ $# -ge 2 ]] || die "--nccl-bin needs a value"
            NCCL_BIN="$2"
            shift 2
            ;;
        --nccl-lib)
            [[ $# -ge 2 ]] || die "--nccl-lib needs a value"
            NCCL_LIB="$2"
            shift 2
            ;;
        --out-dir)
            [[ $# -ge 2 ]] || die "--out-dir needs a value"
            OUT_DIR="$2"
            shift 2
            ;;
        --runs)
            [[ $# -ge 2 ]] || die "--runs needs a value"
            RUNS="$2"
            shift 2
            ;;
        -h|--help)
            HELP=1
            shift
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

if [[ "${HELP}" -eq 1 ]]; then
    usage
    exit 0
fi

[[ -n "${NANO_BIN}" ]] || die "--nano-bin is required"
[[ -n "${NCCL_BIN}" ]] || die "--nccl-bin is required"
[[ -n "${NCCL_LIB}" ]] || die "--nccl-lib is required"
[[ -n "${OUT_DIR}" ]] || die "--out-dir is required"
[[ -x "${COMPARE_SCRIPT}" || -f "${COMPARE_SCRIPT}" ]] || \
    die "compare script not found: ${COMPARE_SCRIPT}"
[[ "${RUNS}" =~ ^[1-9][0-9]*$ ]] || die "--runs must be a positive integer"

for name in \
    COMPARE_HOST_A \
    COMPARE_HOST_B \
    NANO_NCCL_SOCKET_IFNAME \
    NANO_NCCL_RDMA_IFNAME \
    NCCL_SOCKET_IFNAME \
    NCCL_IB_HCA
do
    if [[ -z "${!name:-}" ]]; then
        die "missing required env: ${name}"
    fi
done

mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/busbw.csv"
STATS="${OUT_DIR}/summary_stats.md"
echo "mode,run,size_bytes,nano_busbw,nccl_busbw,ratio" >"${CSV}"

SIZES=(262144 1048576 4194304 16777216 67108864)

parse_summary_row() {
    # summary.md: | 256 KiB | 12.34 | 56.78 | 0.217 |
    local summary="$1"
    local label="$2"
    awk -v label="${label}" '
        $0 ~ "^\\| " label " \\|" {
            gsub(/\|/, " ");
            # fields: size_label parts may span; busbw values are last three numbers
            n = 0
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^[0-9]+(\.[0-9]+)?$/) {
                    vals[++n] = $i
                }
            }
            if (n >= 3) {
                printf "%s %s %s\n", vals[n-2], vals[n-1], vals[n]
                exit 0
            }
        }
    ' "${summary}"
}

label_for_size() {
    case "$1" in
        262144) echo "256 KiB" ;;
        1048576) echo "1 MiB" ;;
        4194304) echo "4 MiB" ;;
        16777216) echo "16 MiB" ;;
        67108864) echo "64 MiB" ;;
        *) echo "$1 B" ;;
    esac
}

run_mode() {
    local mode_name="$1"
    local shared_val="$2"
    local run
    for ((run = 1; run <= RUNS; ++run)); do
        local run_dir="${OUT_DIR}/${mode_name}_run${run}"
        mkdir -p "${run_dir}"
        echo "==> ${mode_name} run ${run}/${RUNS}"
        NANO_NCCL_RDMA_SHARED_PROGRESS="${shared_val}" \
            bash "${COMPARE_SCRIPT}" \
                --nano-bin "${NANO_BIN}" \
                --nccl-bin "${NCCL_BIN}" \
                --nccl-lib "${NCCL_LIB}" \
                --out-dir "${run_dir}"
        local summary="${run_dir}/summary.md"
        [[ -f "${summary}" ]] || die "missing summary: ${summary}"
        local size label nano nccl ratio
        for size in "${SIZES[@]}"; do
            label="$(label_for_size "${size}")"
            read -r nano nccl ratio < <(parse_summary_row "${summary}" "${label}")
            [[ -n "${nano:-}" && -n "${nccl:-}" && -n "${ratio:-}" ]] || \
                die "failed to parse ${label} from ${summary}"
            printf "%s,%d,%d,%s,%s,%s\n" \
                "${mode_name}" "${run}" "${size}" "${nano}" "${nccl}" "${ratio}" \
                >>"${CSV}"
        done
    done
}

run_mode "shared" "1"
run_mode "dedicated" "0"

# best / worst / median nano_busbw and ratio per (mode, size)
{
    echo "# RDMA progress A/B stats (nano_busbw, ratio)"
    echo
    echo "runs_per_mode=${RUNS}"
    echo
    echo "| mode | size | nano_best | nano_worst | nano_median | ratio_best | ratio_worst | ratio_median |"
    echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    python3 - "${CSV}" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
rows = defaultdict(list)
with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["mode"], int(row["size_bytes"]))
        rows[key].append((float(row["nano_busbw"]), float(row["ratio"])))

sizes = [262144, 1048576, 4194304, 16777216, 67108864]
labels = {
    262144: "256 KiB",
    1048576: "1 MiB",
    4194304: "4 MiB",
    16777216: "16 MiB",
    67108864: "64 MiB",
}
for mode in ("shared", "dedicated"):
    for size in sizes:
        vals = rows.get((mode, size), [])
        if not vals:
            continue
        nanos = sorted(v[0] for v in vals)
        ratios = sorted(v[1] for v in vals)
        print(
            f"| {mode} | {labels[size]} | "
            f"{max(nanos):.2f} | {min(nanos):.2f} | {statistics.median(nanos):.2f} | "
            f"{max(ratios):.3f} | {min(ratios):.3f} | {statistics.median(ratios):.3f} |"
        )
PY
} | tee "${STATS}"

echo
echo "wrote ${CSV}"
echo "wrote ${STATS}"
