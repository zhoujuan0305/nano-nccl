#!/usr/bin/env bash
# Capture RDMA host-proxy timeline dumps from a 2-host × 4-GPU WRITE+CTS run.
# Host/interface/GID values come from the environment only — never hard-code them.
# Parses format_v2 per-event metrics (primary); v1 stage spin lines are ignored.
set -euo pipefail

DEFAULT_OUT_DIR="${TMPDIR:-/tmp}/nano-nccl-rdma-timeline"
NANO_BIN=""
OUT_DIR="${DEFAULT_OUT_DIR}"
WARMUP=2
ITERS=5
DRY_RUN=0
HELP=0
SIZES=(262144 1048576)

usage() {
    cat <<'EOF'
Usage: capture_rdma_proxy_timeline.sh --nano-bin PATH [options]

Options:
  --nano-bin PATH     nano_nccl_all_reduce_bench binary (required unless --dry-run)
  --out-dir DIR       log directory (default: ${TMPDIR:-/tmp}/nano-nccl-rdma-timeline)
  --warmup N          benchmark -w (default: 2)
  --iters N           benchmark -n (default: 5)
  --dry-run           print planned commands and env presence; do not execute
  -h, --help          show this help

Required environment (live run):
  COMPARE_HOST_A              MPI host A
  COMPARE_HOST_B              MPI host B
  NANO_NCCL_SOCKET_IFNAME     nano bootstrap socket interface
  NANO_NCCL_RDMA_IFNAME       nano RDMA interface

Optional environment:
  NANO_NCCL_RDMA_GID_INDEX    forwarded when set
  NANO_NCCL_RDMA_USE_WRITE    default 1 for this capture
  NANO_NCCL_RDMA_PROXY_TIMELINE  default 1 (requires timeline-ON binary)
  CUDA_VISIBLE_DEVICES        default 0,1,2,3
  LD_LIBRARY_PATH             library path for ranks (set clean Open MPI lib)

Sizes: 256 KiB and 1 MiB (one mpirun each). Stderr timeline blocks land in
  <out-dir>/size_<bytes>.err  (and stdout in size_<bytes>.log).

Primary parse target is format_v2:
  # rdma_proxy_timeline_v2 ...
  <metric> count sum_ns avg_ns min_ns max_ns
  hist <metric> bin0 ... bin11
  depth_hist <name> c0 c1 c2 c3 c4   # bins 0,1,2,3,>=4

Exit codes:
  0  success (v2 timeline blocks present, nano #wrong == 0 when parseable)
  1  usage/env/run failure
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nano-bin)
            [[ $# -ge 2 ]] || die "--nano-bin needs a value"
            NANO_BIN="$2"
            shift 2
            ;;
        --out-dir)
            [[ $# -ge 2 ]] || die "--out-dir needs a value"
            OUT_DIR="$2"
            shift 2
            ;;
        --warmup)
            [[ $# -ge 2 ]] || die "--warmup needs a value"
            WARMUP="$2"
            shift 2
            ;;
        --iters)
            [[ $# -ge 2 ]] || die "--iters needs a value"
            ITERS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
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

export NANO_NCCL_RDMA_USE_WRITE="${NANO_NCCL_RDMA_USE_WRITE:-1}"
export NANO_NCCL_RDMA_PROXY_TIMELINE="${NANO_NCCL_RDMA_PROXY_TIMELINE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

check_live_env() {
    local missing=()
    local name
    for name in \
        COMPARE_HOST_A \
        COMPARE_HOST_B \
        NANO_NCCL_SOCKET_IFNAME \
        NANO_NCCL_RDMA_IFNAME
    do
        if [[ -z "${!name:-}" ]]; then
            missing+=("${name}")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        die "missing required env: ${missing[*]}"
    fi
}

print_planned() {
    echo "# planned timeline capture (host/interface values not printed)"
    echo "out_dir=${OUT_DIR}"
    echo "nano_bin=${NANO_BIN:-<unset>}"
    echo "warmup=${WARMUP}"
    echo "iters=${ITERS}"
    echo "sizes=${SIZES[*]}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "has_COMPARE_HOST_A=$([[ -n "${COMPARE_HOST_A:-}" ]] && echo 1 || echo 0)"
    echo "has_COMPARE_HOST_B=$([[ -n "${COMPARE_HOST_B:-}" ]] && echo 1 || echo 0)"
    echo "has_NANO_NCCL_SOCKET_IFNAME=$([[ -n "${NANO_NCCL_SOCKET_IFNAME:-}" ]] && echo 1 || echo 0)"
    echo "has_NANO_NCCL_RDMA_IFNAME=$([[ -n "${NANO_NCCL_RDMA_IFNAME:-}" ]] && echo 1 || echo 0)"
    echo "has_NANO_NCCL_RDMA_GID_INDEX=$([[ -n "${NANO_NCCL_RDMA_GID_INDEX:-}" ]] && echo 1 || echo 0)"
    echo "NANO_NCCL_RDMA_USE_WRITE=${NANO_NCCL_RDMA_USE_WRITE}"
    echo "NANO_NCCL_RDMA_PROXY_TIMELINE=${NANO_NCCL_RDMA_PROXY_TIMELINE}"
    echo "nano_flags: --algo ring_simple --transport rdma --dtype float --redop sum -b <size> -e <size> -f 2 -w ${WARMUP} -n ${ITERS}"
    echo "parse_target=format_v2"
}

run_one_size() {
    local size="$1"
    local log="${OUT_DIR}/size_${size}.log"
    local err="${OUT_DIR}/size_${size}.err"
    local -a nano_x=(-x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH
        -x NANO_NCCL_SOCKET_IFNAME -x NANO_NCCL_RDMA_IFNAME
        -x NANO_NCCL_RDMA_USE_WRITE -x NANO_NCCL_RDMA_PROXY_TIMELINE)
    if [[ -n "${NANO_NCCL_RDMA_GID_INDEX:-}" ]]; then
        nano_x+=(-x NANO_NCCL_RDMA_GID_INDEX)
    fi
    local -a nano_args=(
        --algo ring_simple --transport rdma --dtype float --redop sum
        -b "${size}" -e "${size}" -f 2 -w "${WARMUP}" -n "${ITERS}"
    )
    local -a mca=()
    if [[ -n "${NANO_NCCL_SOCKET_IFNAME:-}" ]]; then
        mca+=(--mca btl_tcp_if_include "${NANO_NCCL_SOCKET_IFNAME}")
        mca+=(--mca oob_tcp_if_include "${NANO_NCCL_SOCKET_IFNAME}")
        mca+=(--mca btl_openib_warn_no_device_params_found 0)
    fi
    mpirun "${mca[@]}" \
        --host "${COMPARE_HOST_A}:1" -np 1 "${nano_x[@]}" \
        "${NANO_BIN}" "${nano_args[@]}" : \
        --host "${COMPARE_HOST_B}:1" -np 1 "${nano_x[@]}" \
        "${NANO_BIN}" "${nano_args[@]}" \
        >"${log}" 2>"${err}"
}

extract_timeline_v2() {
    local err="$1"
    local out="$2"
    grep -E '^# rdma_proxy_timeline_v2 |^# metric |^# hist metric |^# depth_hist name |^(send_ready_to_post|send_inter_post|send_post_to_cq|recv_cq_to_publish|recv_inter_publish|cts_ready_to_post|cts_inter_post|send_tail_arm_to_post|send_cts_arm_to_post|send_tail_to_cts_ready|send_post_to_next_tail) |^hist (send_ready_to_post|send_inter_post|send_post_to_cq|recv_cq_to_publish|recv_inter_publish|cts_ready_to_post|cts_inter_post|send_tail_arm_to_post|send_cts_arm_to_post|send_tail_to_cts_ready|send_post_to_next_tail) |^depth_hist (inflight_at_post|free_slots_at_post|inflight_at_cqe) ' \
        "${err}" >"${out}" || true
}

# Aggregate v2 event lines and hist bins across all proxy dumps.
# Prints metric count sum_ns avg_ns min_ns max_ns plus hist totals and
# approximate p50/p90 from cumulative bin mass (bin upper edges in ns).
summarize_events_v2() {
    local timeline="$1"
    local summary="$2"
    awk '
        BEGIN {
            metrics[1] = "send_ready_to_post"
            metrics[2] = "send_inter_post"
            metrics[3] = "send_post_to_cq"
            metrics[4] = "recv_cq_to_publish"
            metrics[5] = "recv_inter_publish"
            metrics[6] = "cts_ready_to_post"
            metrics[7] = "cts_inter_post"
            metrics[8] = "send_tail_arm_to_post"
            metrics[9] = "send_cts_arm_to_post"
            metrics[10] = "send_tail_to_cts_ready"
            metrics[11] = "send_post_to_next_tail"
            nmetrics = 11
            # hist bin upper-exclusive edges (ns); bin11 is +inf
            edge[0] = 250; edge[1] = 500; edge[2] = 1000; edge[3] = 2000
            edge[4] = 5000; edge[5] = 8000; edge[6] = 10000; edge[7] = 12000
            edge[8] = 15000; edge[9] = 20000; edge[10] = 30000
            nbins = 12
            depth[1] = "inflight_at_post"
            depth[2] = "free_slots_at_post"
            depth[3] = "inflight_at_cqe"
            ndepth = 3
            dnbins = 5
        }
        $1 ~ /^(send_ready_to_post|send_inter_post|send_post_to_cq|recv_cq_to_publish|recv_inter_publish|cts_ready_to_post|cts_inter_post|send_tail_arm_to_post|send_cts_arm_to_post|send_tail_to_cts_ready|send_post_to_next_tail)$/ && NF >= 6 {
            m = $1
            count[m] += $2 + 0
            sum[m] += $3 + 0
            c = $2 + 0
            if (c > 0) {
                mn = $5 + 0
                mx = $6 + 0
                if (!(m in minv) || mn < minv[m]) minv[m] = mn
                if (!(m in maxv) || mx > maxv[m]) maxv[m] = mx
            }
            blocks[m]++
            next
        }
        $1 == "hist" && NF >= 14 {
            m = $2
            for (b = 0; b < nbins; ++b) {
                hist[m, b] += $(b + 3) + 0
            }
            next
        }
        $1 == "depth_hist" && NF >= 7 {
            m = $2
            for (b = 0; b < dnbins; ++b) {
                dhist[m, b] += $(b + 3) + 0
            }
            next
        }
        function percentile_from_hist(m, pct,    total, target, acc, b, mid) {
            total = 0
            for (b = 0; b < nbins; ++b) total += hist[m, b] + 0
            if (total <= 0) return -1
            target = total * pct
            if (target < 1) target = 1
            acc = 0
            for (b = 0; b < nbins; ++b) {
                acc += hist[m, b] + 0
                if (acc >= target) {
                    if (b == 0) return int(edge[0] / 2)
                    if (b == nbins - 1) return edge[nbins - 2]
                    mid = int((edge[b - 1] + edge[b]) / 2)
                    return mid
                }
            }
            return edge[nbins - 2]
        }
        END {
            print "# v2 aggregated across all proxy dumps in this size run"
            print "# metric count sum_ns avg_ns min_ns max_ns n_blocks p50_ns_approx p90_ns_approx"
            for (i = 1; i <= nmetrics; ++i) {
                m = metrics[i]
                c = count[m] + 0
                s = sum[m] + 0
                avg = (c == 0) ? 0 : int(s / c)
                mn = (c == 0 || !(m in minv)) ? 0 : minv[m]
                mx = (c == 0 || !(m in maxv)) ? 0 : maxv[m]
                p50 = percentile_from_hist(m, 0.50)
                p90 = percentile_from_hist(m, 0.90)
                if (p50 < 0) p50 = 0
                if (p90 < 0) p90 = 0
                printf "%s %d %d %d %d %d %d %d %d\n", \
                    m, c, s, avg, mn, mx, blocks[m] + 0, p50, p90
            }
            print "# hist metric bin0..bin11 (edges ns: 250 500 1k 2k 5k 8k 10k 12k 15k 20k 30k +inf)"
            for (i = 1; i <= nmetrics; ++i) {
                m = metrics[i]
                printf "hist %s", m
                for (b = 0; b < nbins; ++b) printf " %d", hist[m, b] + 0
                printf "\n"
            }
            print "# depth_hist name c0 c1 c2 c3 c4 (bins 0,1,2,3,>=4)"
            for (i = 1; i <= ndepth; ++i) {
                m = depth[i]
                printf "depth_hist %s", m
                for (b = 0; b < dnbins; ++b) printf " %d", dhist[m, b] + 0
                printf "\n"
            }
        }
    ' "${timeline}" | tee "${summary}"
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_planned
    exit 0
fi

[[ -n "${NANO_BIN}" ]] || die "--nano-bin is required"
check_live_env
require_cmd mpirun
[[ -f "${NANO_BIN}" ]] || die "nano-bin not found: ${NANO_BIN}"
[[ -x "${NANO_BIN}" ]] || die "nano-bin not executable: ${NANO_BIN}"

mkdir -p "${OUT_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
echo "capture_start_utc=${STAMP}" >"${OUT_DIR}/meta.txt"
echo "warmup=${WARMUP}" >>"${OUT_DIR}/meta.txt"
echo "iters=${ITERS}" >>"${OUT_DIR}/meta.txt"
echo "nano_bin=${NANO_BIN}" >>"${OUT_DIR}/meta.txt"
echo "NANO_NCCL_RDMA_USE_WRITE=${NANO_NCCL_RDMA_USE_WRITE}" >>"${OUT_DIR}/meta.txt"
echo "NANO_NCCL_RDMA_PROXY_TIMELINE=${NANO_NCCL_RDMA_PROXY_TIMELINE}" >>"${OUT_DIR}/meta.txt"
echo "parse_target=format_v2" >>"${OUT_DIR}/meta.txt"

any_fail=0
for size in "${SIZES[@]}"; do
    echo "running size=${size} ..."
    if ! run_one_size "${size}"; then
        echo "error: mpirun failed for size=${size}" >&2
        any_fail=1
        continue
    fi
    extract_timeline_v2 "${OUT_DIR}/size_${size}.err" "${OUT_DIR}/size_${size}.timeline_v2.txt"
    if [[ ! -s "${OUT_DIR}/size_${size}.timeline_v2.txt" ]]; then
        echo "error: no v2 timeline blocks in ${OUT_DIR}/size_${size}.err" >&2
        any_fail=1
        continue
    fi
    if ! grep -q '^# rdma_proxy_timeline_v2 ' "${OUT_DIR}/size_${size}.timeline_v2.txt"; then
        echo "error: missing rdma_proxy_timeline_v2 headers for size=${size}" >&2
        any_fail=1
        continue
    fi
    summarize_events_v2 "${OUT_DIR}/size_${size}.timeline_v2.txt" \
        "${OUT_DIR}/size_${size}.event_summary.txt"
    if grep -qE 'ring_simple' "${OUT_DIR}/size_${size}.log" 2>/dev/null; then
        if awk '
            $1 == "ring_simple" && NF >= 10 {
                if (($10 + 0) != 0) exit 1
            }
        ' "${OUT_DIR}/size_${size}.log"; then
            :
        else
            echo "error: #wrong != 0 for size=${size}" >&2
            any_fail=1
        fi
    fi
done

if [[ "${any_fail}" -ne 0 ]]; then
    die "one or more sizes failed; see ${OUT_DIR}"
fi

echo "capture complete: ${OUT_DIR}"
