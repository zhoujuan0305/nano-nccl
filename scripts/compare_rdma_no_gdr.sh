#!/usr/bin/env bash
# Differential nano-nccl vs NCCL (GDR=0) all_reduce busbw on 2-host × 4-GPU RDMA.
# Host/interface/GID values come from the environment only — never hard-code them.
set -euo pipefail

DEFAULT_OUT_DIR="/data02/zhiyuanzhou/apptainer/tmp/opencode/no-gdr-gap"
NANO_BIN=""
NCCL_BIN=""
NCCL_LIB=""
OUT_DIR="${DEFAULT_OUT_DIR}"
DRY_RUN=0
HELP=0
PARSE_NANO_LOG=""
PARSE_NCCL_LOG=""

usage() {
    cat <<'EOF'
Usage: compare_rdma_no_gdr.sh --nano-bin PATH --nccl-bin PATH --nccl-lib DIR [options]

Options:
  --nano-bin PATH     nano_nccl_all_reduce_bench binary (required unless --parse-*)
  --nccl-bin PATH     nccl-tests all_reduce_perf binary (required unless --parse-*)
  --nccl-lib DIR      directory containing libnccl.so (required unless --parse-*)
  --out-dir DIR       log/summary directory (default: /data02/.../no-gdr-gap)
  --parse-nano LOG    skip runs; parse existing nano log
  --parse-nccl LOG    skip runs; parse existing nccl log
  --dry-run           print planned commands and env presence; do not execute benches
  -h, --help          show this help

Required environment (live run):
  COMPARE_HOST_A              MPI host A
  COMPARE_HOST_B              MPI host B
  NANO_NCCL_SOCKET_IFNAME     nano bootstrap socket interface
  NANO_NCCL_RDMA_IFNAME       nano RDMA interface
  NCCL_SOCKET_IFNAME          NCCL OOB socket interface
  NCCL_IB_HCA                 NCCL IB/RoCE HCA name

Optional environment:
  NANO_NCCL_RDMA_GID_INDEX    forwarded to nano MPI ranks when set
  NCCL_IB_GID_INDEX           forwarded to NCCL ranks when set
  CUDA_VISIBLE_DEVICES        default 0,1,2,3
  LD_LIBRARY_PATH             extra library path prefix (NCCL lib is prepended)

Exit codes:
  0  success (all nano #wrong == 0, table printed)
  1  usage/env/run/parse failure or nano #wrong != 0
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
        --parse-nano)
            [[ $# -ge 2 ]] || die "--parse-nano needs a value"
            PARSE_NANO_LOG="$2"
            shift 2
            ;;
        --parse-nccl)
            [[ $# -ge 2 ]] || die "--parse-nccl needs a value"
            PARSE_NCCL_LOG="$2"
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

PARSE_ONLY=0
if [[ -n "${PARSE_NANO_LOG}" || -n "${PARSE_NCCL_LOG}" ]]; then
    PARSE_ONLY=1
    [[ -n "${PARSE_NANO_LOG}" && -n "${PARSE_NCCL_LOG}" ]] || \
        die "--parse-nano and --parse-nccl must be used together"
fi

if [[ "${PARSE_ONLY}" -eq 0 && "${DRY_RUN}" -eq 0 ]]; then
    [[ -n "${NANO_BIN}" ]] || die "--nano-bin is required"
    [[ -n "${NCCL_BIN}" ]] || die "--nccl-bin is required"
    [[ -n "${NCCL_LIB}" ]] || die "--nccl-lib is required"
fi

mkdir -p "${OUT_DIR}"
NANO_LOG="${OUT_DIR}/nano.log"
NANO_ERR="${OUT_DIR}/nano.err"
NCCL_LOG="${OUT_DIR}/nccl.log"
NCCL_ERR="${OUT_DIR}/nccl.err"
SUMMARY="${OUT_DIR}/summary.md"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

check_live_env() {
    local missing=()
    local name
    for name in \
        COMPARE_HOST_A \
        COMPARE_HOST_B \
        NANO_NCCL_SOCKET_IFNAME \
        NANO_NCCL_RDMA_IFNAME \
        NCCL_SOCKET_IFNAME \
        NCCL_IB_HCA
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
    echo "# planned compare (host/interface values not printed)"
    echo "out_dir=${OUT_DIR}"
    echo "nano_bin=${NANO_BIN:-<unset>}"
    echo "nccl_bin=${NCCL_BIN:-<unset>}"
    echo "nccl_lib=${NCCL_LIB:-<unset>}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "has_COMPARE_HOST_A=$([[ -n "${COMPARE_HOST_A:-}" ]] && echo 1 || echo 0)"
    echo "has_COMPARE_HOST_B=$([[ -n "${COMPARE_HOST_B:-}" ]] && echo 1 || echo 0)"
    echo "has_NANO_NCCL_SOCKET_IFNAME=$([[ -n "${NANO_NCCL_SOCKET_IFNAME:-}" ]] && echo 1 || echo 0)"
    echo "has_NANO_NCCL_RDMA_IFNAME=$([[ -n "${NANO_NCCL_RDMA_IFNAME:-}" ]] && echo 1 || echo 0)"
    echo "has_NANO_NCCL_RDMA_GID_INDEX=$([[ -n "${NANO_NCCL_RDMA_GID_INDEX:-}" ]] && echo 1 || echo 0)"
    echo "has_NCCL_SOCKET_IFNAME=$([[ -n "${NCCL_SOCKET_IFNAME:-}" ]] && echo 1 || echo 0)"
    echo "has_NCCL_IB_HCA=$([[ -n "${NCCL_IB_HCA:-}" ]] && echo 1 || echo 0)"
    echo "has_NCCL_IB_GID_INDEX=$([[ -n "${NCCL_IB_GID_INDEX:-}" ]] && echo 1 || echo 0)"
    echo "nano_flags: --algo ring_simple --transport rdma --dtype float --redop sum -b 262144 -e 67108864 -f 4 -w 5 -n 20"
    echo "nccl_flags: NCCL_NET_GDR_LEVEL=0 NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=33554432 -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 -d float -o sum"
}

run_nano() {
    local -a nano_x=(-x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH
        -x NANO_NCCL_SOCKET_IFNAME -x NANO_NCCL_RDMA_IFNAME)
    if [[ -n "${NANO_NCCL_RDMA_GID_INDEX:-}" ]]; then
        nano_x+=(-x NANO_NCCL_RDMA_GID_INDEX)
    fi
    local -a nano_args=(
        --algo ring_simple --transport rdma --dtype float --redop sum
        -b 262144 -e 67108864 -f 4 -w 5 -n 20
    )
    # Bind MPI TCP/OOB to the bootstrap interface so Open MPI does not pick a
    # non-routable NIC (can hang with connect() EINPROGRESS).
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
        >"${NANO_LOG}" 2>"${NANO_ERR}"
}

run_nccl() {
    local -a nccl_x=(-x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH
        -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA
        -x NCCL_ALGO -x NCCL_PROTO
        -x NCCL_MIN_NCHANNELS -x NCCL_MAX_NCHANNELS -x NCCL_BUFFSIZE
        -x NCCL_NET_GDR_LEVEL)
    if [[ -n "${NCCL_IB_GID_INDEX:-}" ]]; then
        nccl_x+=(-x NCCL_IB_GID_INDEX)
    fi
    export NCCL_ALGO=Ring
    export NCCL_PROTO=Simple
    export NCCL_MIN_NCHANNELS=4
    export NCCL_MAX_NCHANNELS=4
    export NCCL_BUFFSIZE=33554432
    export NCCL_NET_GDR_LEVEL=0
    local -a nccl_args=(
        -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 -d float -o sum
    )
    local -a mca=()
    if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
        mca+=(--mca btl_tcp_if_include "${NCCL_SOCKET_IFNAME}")
        mca+=(--mca oob_tcp_if_include "${NCCL_SOCKET_IFNAME}")
        mca+=(--mca btl_openib_warn_no_device_params_found 0)
    fi
    env -u NCCL_IB_DISABLE mpirun "${mca[@]}" \
        --host "${COMPARE_HOST_A}:1" -np 1 "${nccl_x[@]}" \
        "${NCCL_BIN}" "${nccl_args[@]}" : \
        --host "${COMPARE_HOST_B}:1" -np 1 "${nccl_x[@]}" \
        "${NCCL_BIN}" "${nccl_args[@]}" \
        >"${NCCL_LOG}" 2>"${NCCL_ERR}"
}

# nano: ring_simple float sum mixed SIZE COUNT time algbw busbw wrong max_abs
parse_nano() {
    local log="$1"
    awk '
        BEGIN { n = 0 }
        /^[[:space:]]*#/ { next }
        NF >= 10 && $1 == "ring_simple" {
            size = $5 + 0
            busbw = $9 + 0
            wrong = $10 + 0
            sizes[++n] = size
            bus[size] = busbw
            wr[size] = wrong
        }
        END {
            for (i = 1; i <= n; ++i) {
                s = sizes[i]
                printf "%d %.6f %d\n", s, bus[s], wr[s]
            }
        }
    ' "${log}"
}

# nccl-tests out-of-place: size count type redop root time algbw busbw #wrong ...
parse_nccl() {
    local log="$1"
    awk '
        BEGIN { n = 0 }
        /^[[:space:]]*#/ { next }
        NF >= 9 && $3 == "float" && $4 == "sum" {
            size = $1 + 0
            busbw = $8 + 0
            wrong = $9 + 0
            sizes[++n] = size
            bus[size] = busbw
            wr[size] = wrong
        }
        END {
            for (i = 1; i <= n; ++i) {
                s = sizes[i]
                printf "%d %.6f %d\n", s, bus[s], wr[s]
            }
        }
    ' "${log}"
}

human_size() {
    local bytes="$1"
    case "${bytes}" in
        262144) echo "256 KiB" ;;
        1048576) echo "1 MiB" ;;
        4194304) echo "4 MiB" ;;
        16777216) echo "16 MiB" ;;
        67108864) echo "64 MiB" ;;
        *) echo "${bytes} B" ;;
    esac
}

write_summary() {
    local nano_tsv="$1"
    local nccl_tsv="$2"
    local out="$3"
    local any_wrong=0
    local size n_bus n_wr c_bus ratio label
    local -A nano_bus nano_wr nccl_bus
    local tmp

    while read -r size n_bus n_wr; do
        [[ -n "${size:-}" ]] || continue
        nano_bus["${size}"]="${n_bus}"
        nano_wr["${size}"]="${n_wr}"
        if [[ "${n_wr}" -ne 0 ]]; then
            any_wrong=1
        fi
    done <"${nano_tsv}"

    while read -r size c_bus _; do
        [[ -n "${size:-}" ]] || continue
        nccl_bus["${size}"]="${c_bus}"
    done <"${nccl_tsv}"

    tmp="$(mktemp)"
    {
        echo "| size | nano_busbw | nccl_busbw | ratio |"
        echo "| ---: | ---: | ---: | ---: |"
        for size in 262144 1048576 4194304 16777216 67108864; do
            if [[ -z "${nano_bus[${size}]+x}" || -z "${nccl_bus[${size}]+x}" ]]; then
                rm -f "${tmp}"
                die "missing busbw for size ${size} in parsed logs"
            fi
            n_bus="${nano_bus[${size}]}"
            c_bus="${nccl_bus[${size}]}"
            ratio="$(awk -v n="${n_bus}" -v c="${c_bus}" 'BEGIN {
                if (c == 0) { print "nan"; exit }
                printf "%.3f", n / c
            }')"
            label="$(human_size "${size}")"
            printf "| %s | %.2f | %.2f | %s |\n" "${label}" "${n_bus}" "${c_bus}" "${ratio}"
        done
    } >"${tmp}"

    cat "${tmp}" | tee "${out}"
    rm -f "${tmp}"

    if [[ "${any_wrong}" -ne 0 ]]; then
        die "nano log has #wrong != 0"
    fi
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_planned
    exit 0
fi

if [[ "${PARSE_ONLY}" -eq 1 ]]; then
    [[ -f "${PARSE_NANO_LOG}" ]] || die "nano log not found: ${PARSE_NANO_LOG}"
    [[ -f "${PARSE_NCCL_LOG}" ]] || die "nccl log not found: ${PARSE_NCCL_LOG}"
    NANO_LOG="${PARSE_NANO_LOG}"
    NCCL_LOG="${PARSE_NCCL_LOG}"
else
    check_live_env
    require_cmd mpirun
    [[ -f "${NANO_BIN}" ]] || die "nano-bin not found: ${NANO_BIN}"
    [[ -f "${NCCL_BIN}" ]] || die "nccl-bin not found: ${NCCL_BIN}"
    [[ -d "${NCCL_LIB}" ]] || die "nccl-lib not a directory: ${NCCL_LIB}"
    export LD_LIBRARY_PATH="${NCCL_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    run_nano
    run_nccl
fi

NANO_TSV="$(mktemp)"
NCCL_TSV="$(mktemp)"
trap 'rm -f "${NANO_TSV}" "${NCCL_TSV}"' EXIT

parse_nano "${NANO_LOG}" >"${NANO_TSV}"
parse_nccl "${NCCL_LOG}" >"${NCCL_TSV}"

[[ -s "${NANO_TSV}" ]] || die "failed to parse nano busbw from ${NANO_LOG}"
[[ -s "${NCCL_TSV}" ]] || die "failed to parse nccl busbw from ${NCCL_LOG}"

write_summary "${NANO_TSV}" "${NCCL_TSV}" "${SUMMARY}"
