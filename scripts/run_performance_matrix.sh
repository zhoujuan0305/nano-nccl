#!/usr/bin/env bash
# Performance matrix helper. Published performance.md is single-host only
# (in-process auto + 4-rank socket + 4-rank RDMA). The socket/rdma sections
# below still launch two-host MPI pairs; do not regenerate performance.md
# from those sections without rewriting this script.
# Hosts and interfaces come from the environment only — never hard-code secrets.
set -euo pipefail

DEFAULT_OUT="/data02/zhiyuanzhou/apptainer/tmp/opencode/perf-matrix"
OUT_DIR="${DEFAULT_OUT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SINGLE_BIN="${ROOT}/build-perf-single/benchmarks/nano_nccl_all_reduce_bench"
RDMA_BIN="${ROOT}/build-perf-rdma/benchmarks/nano_nccl_all_reduce_bench"
NCCL_BIN=""
NCCL_LIB=""
SECTIONS="single,socket,rdma"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: run_performance_matrix.sh --nccl-bin PATH --nccl-lib DIR [options]

Options:
  --out-dir DIR       output directory for logs + matrix.json
  --single-bin PATH   nano single-host bench (default: build-perf-single/...)
  --rdma-bin PATH     nano MPI/RDMA bench (default: build-perf-rdma/...)
  --sections LIST     comma list: single,socket,rdma (default: all)
  --dry-run           print plan only
  -h, --help

Required env (live multi-host):
  COMPARE_HOST_A COMPARE_HOST_B
  NANO_NCCL_SOCKET_IFNAME NCCL_SOCKET_IFNAME
  NANO_NCCL_RDMA_IFNAME NCCL_IB_HCA   (rdma section)
Optional:
  NANO_NCCL_RDMA_GID_INDEX NCCL_IB_GID_INDEX
  CUDA_VISIBLE_DEVICES (default 0,1,2,3)
EOF
}

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --single-bin) SINGLE_BIN="$2"; shift 2 ;;
        --rdma-bin) RDMA_BIN="$2"; shift 2 ;;
        --nccl-bin) NCCL_BIN="$2"; shift 2 ;;
        --nccl-lib) NCCL_LIB="$2"; shift 2 ;;
        --sections) SECTIONS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[[ -n "${NCCL_BIN}" && -x "${NCCL_BIN}" ]] || die "need --nccl-bin"
[[ -n "${NCCL_LIB}" && -d "${NCCL_LIB}" ]] || die "need --nccl-lib"
command -v mpirun >/dev/null || die "mpirun not in PATH"
command -v python3 >/dev/null || die "python3 required"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PATH="${PATH}"
# Prefer the Open MPI 4.1.2 prefix used for dual-host builds. ~/.local/openmpi can
# shadow it with an incompatible libevent and break mpirun symbol lookup.
MPI_LIB_HINT=""
if [[ -d "${HOME}/opt/openmpi-4.1.2/lib" ]]; then
    MPI_LIB_HINT="${HOME}/opt/openmpi-4.1.2/lib"
elif [[ -d "${HOME}/.local/openmpi/lib" ]]; then
    MPI_LIB_HINT="${HOME}/.local/openmpi/lib"
fi
# MPI libs before NCCL so mpirun resolves against the intended Open MPI build.
export LD_LIBRARY_PATH="${MPI_LIB_HINT:+$MPI_LIB_HINT:}${NCCL_LIB}:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "${OUT_DIR}/logs"
RAW_JSONL="${OUT_DIR}/cells.jsonl"
: >"${RAW_JSONL}"

DTYPES=(float fp16 bf16)
REDOPS=(sum avg max min)

nccl_dtype() {
    case "$1" in
        float) echo float ;;
        fp16) echo half ;;
        bf16) echo bfloat16 ;;
        *) die "bad dtype $1" ;;
    esac
}

mca_args() {
    local ifn="$1"
    echo --mca btl_tcp_if_include "${ifn}" --mca oob_tcp_if_include "${ifn}" --mca btl_openib_warn_no_device_params_found 0
}

parse_nano_line() {
    # stdout: size time_us busbw wrong
    awk '
      /^[[:space:]]*#/ { next }
      NF >= 10 && $1 == "ring_simple" {
        printf "%s %s %s %s\n", $5, $7, $9, $10
      }
    '
}

parse_nccl_lines() {
    # out-of-place columns: size count type redop root time algbw busbw wrong
    local dtype="$1" redop="$2"
    awk -v dt="$dtype" -v ro="$redop" '
      /^[[:space:]]*#/ { next }
      NF >= 9 && $3 == dt && $4 == ro {
        printf "%s %s %s %s\n", $1, $6, $8, $9
      }
    '
}

append_cell() {
    local section="$1" dtype="$2" redop="$3" size="$4"
    local nt="$5" nb="$6" ct="$7" cb="$8" nw="$9" cw="$10"
    python3 - "$RAW_JSONL" "$section" "$dtype" "$redop" "$size" "$nt" "$nb" "$ct" "$cb" "$nw" "$cw" <<'PY'
import json, sys
path, section, dtype, redop, size, nt, nb, ct, cb, nw, cw = sys.argv[1:]
size=int(float(size)); nt=float(nt); nb=float(nb); ct=float(ct); cb=float(cb)
nw=int(float(nw)); cw=int(float(cw))
ratio = (ct / nt) if nt > 0 else 0.0
rec = {
  "section": section, "dtype": dtype, "redop": redop, "size": size,
  "nano_time_us": nt, "nano_busbw": nb, "nccl_time_us": ct, "nccl_busbw": cb,
  "nano_wrong": nw, "nccl_wrong": cw, "ratio": ratio,
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(rec) + "\n")
print(f"OK {section} {dtype}/{redop} size={size} nano_bw={nb:.3f} nccl_bw={cb:.3f} ratio={ratio:.3f} wrong={nw}/{cw}")
PY
}

run_single_pair() {
    local dtype="$1" redop="$2"
    local tag="single_${dtype}_${redop}"
    local nano_log="${OUT_DIR}/logs/${tag}_nano.log"
    local nccl_log="${OUT_DIR}/logs/${tag}_nccl.log"
    local nano_err="${OUT_DIR}/logs/${tag}_nano.err"
    local nccl_err="${OUT_DIR}/logs/${tag}_nccl.err"
    local ndt
    ndt="$(nccl_dtype "${dtype}")"

    echo "==> single ${dtype}/${redop}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then return 0; fi
    [[ -x "${SINGLE_BIN}" ]] || die "missing single bin ${SINGLE_BIN}"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      "${SINGLE_BIN}" --algo ring_simple --transport auto \
      --dtype "${dtype}" --redop "${redop}" \
      -b 262144 -e 67108864 -f 4 -w 5 -n 20 \
      >"${nano_log}" 2>"${nano_err}" || die "nano single failed ${tag}: $(tail -5 "${nano_err}")"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    NCCL_ALGO=Ring NCCL_PROTO=Simple \
    NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=33554432 \
      "${NCCL_BIN}" -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 \
      -d "${ndt}" -o "${redop}" \
      >"${nccl_log}" 2>"${nccl_err}" || die "nccl single failed ${tag}"

    # join by size
    python3 - "$nano_log" "$nccl_log" "$dtype" "$redop" "$RAW_JSONL" <<'PY'
import json, sys
from collections import OrderedDict
nano_log, nccl_log, dtype, redop, out = sys.argv[1:]
def parse_nano(path):
    d={}
    for line in open(path):
        if line.lstrip().startswith('#') or not line.strip(): continue
        p=line.split()
        if len(p)<10 or p[0]!='ring_simple': continue
        d[int(p[4])]=(float(p[6]), float(p[8]), int(p[9]))
    return d
def parse_nccl(path, dt, ro):
    d={}
    for line in open(path):
        if line.lstrip().startswith('#') or not line.strip(): continue
        p=line.split()
        if len(p)<9: continue
        if p[2]!=dt or p[3]!=ro: continue
        try: size=int(p[0])
        except: continue
        d[size]=(float(p[5]), float(p[7]), int(p[8]))
    return d
ndt={'float':'float','fp16':'half','bf16':'bfloat16'}[dtype]
n=parse_nano(nano_log); c=parse_nccl(nccl_log, ndt, redop)
sizes=sorted(set(n)&set(c))
if not sizes: raise SystemExit(f'no overlapping sizes for single {dtype}/{redop}')
with open(out,'a') as f:
    for s in sizes:
        nt,nb,nw=n[s]; ct,cb,cw=c[s]
        if nw!=0 or cw!=0:
            raise SystemExit(f'wrong!=0 single {dtype}/{redop} size={s} nano={nw} nccl={cw}')
        ratio=ct/nt if nt else 0.0
        rec=dict(section='single',dtype=dtype,redop=redop,size=s,
                 nano_time_us=nt,nano_busbw=nb,nccl_time_us=ct,nccl_busbw=cb,
                 nano_wrong=nw,nccl_wrong=cw,ratio=ratio)
        f.write(json.dumps(rec)+'\n')
        print(f"OK single {dtype}/{redop} size={s} nano_bw={nb:.3f} nccl_bw={cb:.3f} ratio={ratio:.3f}")
PY
}

run_mpi_pair() {
    local section="$1" dtype="$2" redop="$3" transport="$4"
    local tag="${section}_${dtype}_${redop}"
    local nano_log="${OUT_DIR}/logs/${tag}_nano.log"
    local nccl_log="${OUT_DIR}/logs/${tag}_nccl.log"
    local nano_err="${OUT_DIR}/logs/${tag}_nano.err"
    local nccl_err="${OUT_DIR}/logs/${tag}_nccl.err"
    local sock_if="${NANO_NCCL_SOCKET_IFNAME:?set NANO_NCCL_SOCKET_IFNAME}"
    local host_a="${COMPARE_HOST_A:?set COMPARE_HOST_A}"
    local host_b="${COMPARE_HOST_B:?set COMPARE_HOST_B}"
    local ndt
    ndt="$(python3 -c "print({'float':'float','fp16':'half','bf16':'bfloat16'}['${dtype}'])")"

    echo "==> ${section} ${dtype}/${redop} transport=${transport}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then return 0; fi
    [[ -x "${RDMA_BIN}" ]] || die "missing rdma/mpi bin ${RDMA_BIN}"
    [[ -n "${NCCL_SOCKET_IFNAME:-}" ]] || die "set NCCL_SOCKET_IFNAME"

    local -a mca
    # shellcheck disable=SC2207
    mca=($(mca_args "${sock_if}"))

    local -a nano_x=(-x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH
        -x NANO_NCCL_SOCKET_IFNAME)
    local -a nano_env=(NANO_NCCL_SOCKET_IFNAME="${sock_if}")
    if [[ "${transport}" == "rdma" ]]; then
        [[ -n "${NANO_NCCL_RDMA_IFNAME:-}" ]] || die "set NANO_NCCL_RDMA_IFNAME"
        nano_x+=(-x NANO_NCCL_RDMA_IFNAME)
        nano_env+=(NANO_NCCL_RDMA_IFNAME="${NANO_NCCL_RDMA_IFNAME}")
        if [[ -n "${NANO_NCCL_RDMA_GID_INDEX:-}" ]]; then
            nano_x+=(-x NANO_NCCL_RDMA_GID_INDEX)
            nano_env+=(NANO_NCCL_RDMA_GID_INDEX="${NANO_NCCL_RDMA_GID_INDEX}")
        fi
        if [[ -n "${NANO_NCCL_RDMA_USE_WRITE:-}" ]]; then
            nano_x+=(-x NANO_NCCL_RDMA_USE_WRITE)
            nano_env+=(NANO_NCCL_RDMA_USE_WRITE="${NANO_NCCL_RDMA_USE_WRITE}")
        fi
    fi

    env "${nano_env[@]}" \
    mpirun "${mca[@]}" \
      --host "${host_a}:1" -np 1 "${nano_x[@]}" \
      "${RDMA_BIN}" --algo ring_simple --transport "${transport}" \
      --dtype "${dtype}" --redop "${redop}" \
      -b 262144 -e 67108864 -f 4 -w 5 -n 20 : \
      --host "${host_b}:1" -np 1 "${nano_x[@]}" \
      "${RDMA_BIN}" --algo ring_simple --transport "${transport}" \
      --dtype "${dtype}" --redop "${redop}" \
      -b 262144 -e 67108864 -f 4 -w 5 -n 20 \
      >"${nano_log}" 2>"${nano_err}" || die "nano ${tag} failed: $(tail -8 "${nano_err}")"

    local -a nccl_x=(-x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH
        -x NCCL_SOCKET_IFNAME -x NCCL_ALGO -x NCCL_PROTO
        -x NCCL_MIN_NCHANNELS -x NCCL_MAX_NCHANNELS -x NCCL_BUFFSIZE)
    local -a nccl_env=(
        NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
        NCCL_ALGO=Ring NCCL_PROTO=Simple
        NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=33554432
    )
    local -a nccl_mca
    # shellcheck disable=SC2207
    nccl_mca=($(mca_args "${NCCL_SOCKET_IFNAME}"))

    if [[ "${section}" == "socket" ]]; then
        nccl_env+=(NCCL_IB_DISABLE=1)
        nccl_x+=(-x NCCL_IB_DISABLE)
        env "${nccl_env[@]}" \
        mpirun "${nccl_mca[@]}" \
          --host "${host_a}:1" -np 1 "${nccl_x[@]}" \
          "${NCCL_BIN}" -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 -d "${ndt}" -o "${redop}" : \
          --host "${host_b}:1" -np 1 "${nccl_x[@]}" \
          "${NCCL_BIN}" -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 -d "${ndt}" -o "${redop}" \
          >"${nccl_log}" 2>"${nccl_err}" || die "nccl socket ${tag} failed"
    else
        [[ -n "${NCCL_IB_HCA:-}" ]] || die "set NCCL_IB_HCA"
        nccl_env+=(NCCL_IB_HCA="${NCCL_IB_HCA}" NCCL_NET_GDR_LEVEL=0)
        nccl_x+=(-x NCCL_IB_HCA -x NCCL_NET_GDR_LEVEL)
        if [[ -n "${NCCL_IB_GID_INDEX:-}" ]]; then
            nccl_env+=(NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}")
            nccl_x+=(-x NCCL_IB_GID_INDEX)
        fi
        env -u NCCL_IB_DISABLE "${nccl_env[@]}" \
        mpirun "${nccl_mca[@]}" \
          --host "${host_a}:1" -np 1 "${nccl_x[@]}" \
          "${NCCL_BIN}" -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 -d "${ndt}" -o "${redop}" : \
          --host "${host_b}:1" -np 1 "${nccl_x[@]}" \
          "${NCCL_BIN}" -b 262144 -e 67108864 -f 4 -g 4 -w 5 -n 20 -d "${ndt}" -o "${redop}" \
          >"${nccl_log}" 2>"${nccl_err}" || die "nccl rdma ${tag} failed"
    fi

    python3 - "$nano_log" "$nccl_log" "$section" "$dtype" "$redop" "$RAW_JSONL" <<'PY'
import json, sys
nano_log, nccl_log, section, dtype, redop, out = sys.argv[1:]
def parse_nano(path):
    d={}
    for line in open(path):
        if line.lstrip().startswith('#') or not line.strip(): continue
        p=line.split()
        if len(p)<10 or p[0]!='ring_simple': continue
        d[int(p[4])]=(float(p[6]), float(p[8]), int(p[9]))
    return d
def parse_nccl(path, dt, ro):
    d={}
    for line in open(path):
        if line.lstrip().startswith('#') or not line.strip(): continue
        p=line.split()
        if len(p)<9: continue
        if p[2]!=dt or p[3]!=ro: continue
        try: size=int(p[0])
        except: continue
        d[size]=(float(p[5]), float(p[7]), int(p[8]))
    return d
ndt={'float':'float','fp16':'half','bf16':'bfloat16'}[dtype]
n=parse_nano(nano_log); c=parse_nccl(nccl_log, ndt, redop)
sizes=sorted(set(n)&set(c))
if not sizes: raise SystemExit(f'no sizes {section} {dtype}/{redop}')
with open(out,'a') as f:
    for s in sizes:
        nt,nb,nw=n[s]; ct,cb,cw=c[s]
        if nw!=0 or cw!=0:
            raise SystemExit(f'wrong!=0 {section} {dtype}/{redop} size={s} nano={nw} nccl={cw}')
        ratio=ct/nt if nt else 0.0
        rec=dict(section=section,dtype=dtype,redop=redop,size=s,
                 nano_time_us=nt,nano_busbw=nb,nccl_time_us=ct,nccl_busbw=cb,
                 nano_wrong=nw,nccl_wrong=cw,ratio=ratio)
        f.write(json.dumps(rec)+'\n')
        print(f"OK {section} {dtype}/{redop} size={s} nano_bw={nb:.3f} nccl_bw={cb:.3f} ratio={ratio:.3f}")
PY
}

IFS=',' read -r -a SEC_ARR <<<"${SECTIONS}"
for sec in "${SEC_ARR[@]}"; do
    case "${sec}" in
        single)
            for d in "${DTYPES[@]}"; do
                for r in "${REDOPS[@]}"; do run_single_pair "$d" "$r"; done
            done
            ;;
        socket)
            # MPI multi-host default is socket on cross-process edges; CLI has no
            # --transport socket (only auto|shm|p2p|rdma).
            for d in "${DTYPES[@]}"; do
                for r in "${REDOPS[@]}"; do run_mpi_pair socket "$d" "$r" auto; done
            done
            ;;
        rdma)
            for d in "${DTYPES[@]}"; do
                for r in "${REDOPS[@]}"; do run_mpi_pair rdma "$d" "$r" rdma; done
            done
            ;;
        *) die "unknown section ${sec}" ;;
    esac
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "dry-run complete"
    exit 0
fi

NODE_A_KERNEL="$(uname -r)"
NODE_A_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
NODE_B_KERNEL=""
NODE_B_DRIVER=""
if [[ -n "${COMPARE_HOST_B:-}" ]]; then
    NODE_B_KERNEL="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${COMPARE_HOST_B}" 'uname -r' 2>/dev/null || true)"
    NODE_B_DRIVER="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${COMPARE_HOST_B}" 'nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1' 2>/dev/null | tr -d ' ' || true)"
fi

# Assemble matrix.json
python3 - "$RAW_JSONL" "${OUT_DIR}/matrix.json" "${ROOT}" \
  "${NODE_A_KERNEL}" "${NODE_A_DRIVER}" "${NODE_B_KERNEL}" "${NODE_B_DRIVER}" <<'PY'
import json, subprocess, sys, datetime
from pathlib import Path
raw, out, root, ka, da, kb, db = sys.argv[1:]
cells=[]
for line in Path(raw).read_text().splitlines():
    if line.strip(): cells.append(json.loads(line))
try:
    commit=subprocess.check_output(['git','-C',root,'rev-parse','HEAD'], text=True).strip()
except Exception:
    commit='unknown'
sections={}
for c in cells:
    sec=sections.setdefault(c['section'], {})
    dt=sec.setdefault(c['dtype'], {})
    ro=dt.setdefault(c['redop'], [])
    ro.append({k:c[k] for k in (
        'size','nano_time_us','nano_busbw','nccl_time_us','nccl_busbw','ratio',
        'nano_wrong','nccl_wrong')})
for sec in sections.values():
    for dt in sec.values():
        for ro, rows in dt.items():
            rows.sort(key=lambda r: r['size'])
doc={
  'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'commit': commit,
  'env': {
    'cuda': '12.8.61',
    'nccl': '2.30.7',
    'nccl_tests': '2.19.6',
    'mpi': '4.1.2',
    'rdma_nccl_gdr_level': '0',
    'node_a_kernel': f'Linux {ka}' if ka and not ka.startswith('Linux') else (ka or 'unknown'),
    'node_b_kernel': f'Linux {kb}' if kb and not kb.startswith('Linux') else (kb or 'unknown'),
    'node_a_driver': da or 'unknown',
    'node_b_driver': db or 'unknown',
  },
  'sections': sections,
}
Path(out).write_text(json.dumps(doc, indent=2) + '\n')
print(f'wrote {out} cells={len(cells)}')
PY

echo "matrix complete: ${OUT_DIR}/matrix.json"
