# shellcheck shell=bash
# Copyright (C) 2026 Lu Verissimo.
# Part of an independent performance study of AutoDock-GPU. Not affiliated with
# or endorsed by Scripps Research or the AutoDock-GPU developers.
# GPL-2.0-or-later. See LICENSE.
#
# Shared helpers for scripts/phase0.sh. Source this file; do not execute it.
#   DRY_RUN=1  print GPU, build and profiler commands instead of running them.
#              Path checks and run-directory setup still run for real.

DRY_RUN=${DRY_RUN:-0}
NUMWI=${NUMWI:-128}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# Toolkit layout: /usr/local/cuda (NVIDIA installer) or distro packaging (e.g. Lambda's
# nvidia-cuda-toolkit: headers in /usr/include, libs in /usr/lib/x86_64-linux-gnu).
if [ -z "${GPU_INCLUDE_PATH:-}" ]; then
  if [ -e "$CUDA_HOME/include/cuda_runtime_api.h" ]; then GPU_INCLUDE_PATH=$CUDA_HOME/include
  elif [ -e /usr/include/cuda_runtime_api.h ]; then GPU_INCLUDE_PATH=/usr/include; fi
fi
if [ -z "${GPU_LIBRARY_PATH:-}" ]; then
  if [ -e "$CUDA_HOME/lib64/libcudart.so" ]; then GPU_LIBRARY_PATH=$CUDA_HOME/lib64
  elif [ -e /usr/lib/x86_64-linux-gnu/libcudart.so ]; then GPU_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu; fi
fi
export GPU_INCLUDE_PATH GPU_LIBRARY_PATH
# Nsight tools installed from NVIDIA's apt repo live under /opt/nvidia, not on PATH.
for _d in /opt/nvidia/nsight-compute/*/ /opt/nvidia/nsight-systems/*/bin; do
  [ -d "$_d" ] && PATH="$_d:$PATH"
done; unset _d; export PATH
ROOT=$(git rev-parse --show-toplevel)

# --- GPU pinning -------------------------------------------------------------
# Headline (timed/profiled) work runs on GPU 0 only, one job at a time.
# GPUs 1-3 may be used for exploratory/correctness runs with EXPLORE=1; their
# output folders are prefixed "explore_" and must never feed headline numbers.
GPU_ID=${GPU_ID:-0}
EXPLORE=${EXPLORE:-0}
if [ "$EXPLORE" = 1 ]; then
  case $GPU_ID in 1|2|3) ;; *) echo "EXPLORE=1 requires GPU_ID in 1..3" >&2; exit 1 ;; esac
elif [ "$GPU_ID" != 0 ]; then
  echo "headline runs are pinned to GPU 0; use EXPLORE=1 GPU_ID=<1..3> for other GPUs" >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES=$GPU_ID   # the pinned GPU becomes visible device 0
# Applies to the CUDA builds; NVIDIA's OpenCL ICD is also expected to honor it,
# which is verified per run from the device line AutoDock-GPU prints (see gpu.txt).

# --- session hard stop ---------------------------------------------------------
# preflight writes the session start time; every later stage refuses to start, and
# stops launching new jobs, once HARD_STOP_MIN has elapsed since then.
HARD_STOP_MIN=${HARD_STOP_MIN:-120}
RESULTS_PHASE=${RESULTS_PHASE:-phase0}   # results/<phase>/ (phase1.sh sets phase1)
SESSION_FILE=$ROOT/results/phase0/.session_start   # one session clock per instance session
session_min() {
  [ -f "$SESSION_FILE" ] || { echo 0; return; }
  echo $(( ($(date +%s) - $(cat "$SESSION_FILE")) / 60 ))
}
session_left() { echo $(( HARD_STOP_MIN - $(session_min) )); }
check_session() {  # check_session <minutes needed to start>
  [ "$DRY_RUN" = 1 ] && return 0
  [ -f "$SESSION_FILE" ] || die "no session start recorded; run 'scripts/phase0.sh preflight' first"
  local left; left=$(session_left)
  [ "$left" -ge "${1:-1}" ] || die "hard stop: $(session_min) of $HARD_STOP_MIN min used, ${left} left (< ${1:-1} needed). Ask before extending (HARD_STOP_MIN=...)."
  log "session: $(session_min) of $HARD_STOP_MIN min used"
}

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# run <cmd...>: execute, or print under DRY_RUN. Commands are logged either way.
run() {
  printf '+ %s\n' "$(printf '%q ' "$@")" >> "${OUT:-/dev/null}/commands.txt" 2>/dev/null || true
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY: %s\n' "$(printf '%q ' "$@")"
  else
    "$@"
  fi
}

# new_outdir <stage>: create results/phase0/[explore_]<stage>_gpu<N>_<host>_<utc>,
# set OUT, and record which physical GPU this stage used.
new_outdir() {
  local pre=""; [ "$EXPLORE" = 1 ] && pre=explore_
  OUT=$ROOT/results/$RESULTS_PHASE/${pre}${1}_gpu${GPU_ID}_$(hostname -s)_$(date -u +%Y%m%dT%H%M%SZ)
  [ "$DRY_RUN" = 1 ] && OUT=${OUT}_dryrun
  mkdir -p "$OUT"
  : > "$OUT/commands.txt"
  log "output: $OUT"
  gpu_record start
}

# gpu_record <tag>: physical GPU identity + state, appended to $OUT/gpu.txt
gpu_record() {
  {
    echo "=== $1 $(date -u +%FT%TZ)  GPU_ID=$GPU_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES EXPLORE=$EXPLORE"
    nvidia-smi -i "$GPU_ID" --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,pstate --format=csv 2>&1
    echo "--- other processes on this GPU (should be none):"
    nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
  } >> "$OUT/gpu.txt" 2>&1 || true
}

# setup_case <case> <workdir>: symlink the shipped inputs into a run dir. The .fld
# of some cases (3tmn, 7cpa) names rec.*.map while the files are <case>_rec.*.map,
# so both spellings are linked. Inputs are never modified. Sets FLD and LIG.
# Fails if any map the .fld names cannot be resolved.
setup_case() {
  local c=$1 wd=$2 src=$ROOT/input/$1/derived f m
  [ -d "$src" ] || die "no input dir $src"
  mkdir -p "$wd"
  for f in "$src"/*; do ln -sf "$f" "$wd/"; done
  for m in "$src/${c}"_rec.*.map; do
    [ -e "$m" ] && ln -sf "$m" "$wd/${m##*/"${c}"_}"
  done
  XRAY=""
  FLD=$(ls "$wd"/*.maps.fld 2>/dev/null | head -1)
  LIG=$wd/${c}_ligand.pdbqt
  [ -n "$FLD" ] || die "$c: no .maps.fld"
  [ -e "$LIG" ] || die "$c: no ligand $LIG"
  check_maps "$c" "$wd"
}

# check_maps <label> <workdir>: every grid map ("variable" line) in $FLD must exist.
# "coord" lines (.maps.xyz) are not read by AutoDock-GPU and are absent for some inputs.
check_maps() {
  local c=$1 wd=$2 missing=0 mf
  for mf in $(awk '$1=="variable"' "$FLD" | grep -o 'file=[^ ]*' | cut -d= -f2); do
    [ -e "$wd/$mf" ] || { log "$c: map $mf named in $(basename "$FLD") not found"; missing=1; }
  done
  [ $missing = 0 ] || die "$c: unresolved map files"
}

# build <cuda|ocl|cuda_lineinfo>: build and copy the binary to $BIN/adgpu_<kind>_<N>wi.
# Both paths write bin/autodock_gpu_<N>wi, so each build is copied out right away.
build() {
  local kind=$1 dst=$BIN/adgpu_${1}_${NUMWI}wi
  mkdir -p "$BIN"
  rm -f "$ROOT/bin/autodock_gpu_${NUMWI}wi"
  # -lineinfo only adds source-line tables for ncu's source view; nvcc docs say it
  # does not change code generation. Register/smem use of both builds is recorded.
  # TENSOR=ON: upstream WMMA reduction path (USE_NVTENSOR, sm_80, NUMWI >= 64).
  case $kind in
    cuda)                 ( cd "$ROOT" && run make DEVICE=CUDA NUMWI="$NUMWI" TARGETS=80 ) ;;
    cuda_lineinfo)        ( cd "$ROOT" && run make DEVICE=CUDA NUMWI="$NUMWI" TARGETS=80 "NVCC=nvcc -lineinfo" ) ;;
    cuda_tensor)          ( cd "$ROOT" && run make DEVICE=CUDA NUMWI="$NUMWI" TARGETS=80 TENSOR=ON ) ;;
    cuda_tensor_lineinfo) ( cd "$ROOT" && run make DEVICE=CUDA NUMWI="$NUMWI" TARGETS=80 TENSOR=ON "NVCC=nvcc -lineinfo" ) ;;
    ocl)                  ( cd "$ROOT" && run make DEVICE=OCLGPU NUMWI="$NUMWI" ) ;;
    *) die "unknown build kind $kind" ;;
  esac
  run cp "$ROOT/bin/autodock_gpu_${NUMWI}wi" "$dst"
  if [ "$kind" != ocl ]; then
    run cuobjdump -res-usage "$dst" > "${dst}.res-usage.txt" 2>&1 || true
  fi
}

# setup_set42 <pdb> <workdir>: one complex of diogomart/AD-GPU_set_of_42 ($SET42_DIR).
# Maps ship gzipped; they are decompressed into the work dir (the clone is untouched).
# Ligand = rand-0.pdbqt (randomized start), reference = flex-xray.pdbqt (for RMSD).
setup_set42() {
  local c=$1 wd=$2 src=$SET42_DIR/data/$1 f
  [ -d "$src" ] || die "no set-of-42 dir $src"
  mkdir -p "$wd"
  for f in "$src"/*; do
    case $f in
      *.map.gz) [ -e "$wd/$(basename "$f" .gz)" ] || gunzip -c "$f" > "$wd/$(basename "$f" .gz)" ;;
      *) ln -sf "$f" "$wd/" ;;
    esac
  done
  FLD=$wd/protein.maps.fld; LIG=$wd/rand-0.pdbqt; XRAY=$wd/flex-xray.pdbqt
  [ -e "$LIG" ] && [ -e "$XRAY" ] || die "$c: missing rand-0/flex-xray pdbqt"
  check_maps "$c" "$wd"
}

# adgpu_args <seed> <nrun> <resnam>: the docking command line used everywhere.
# If XRAY is set, RMSD is reported against it (--xraylfile).
adgpu_args() {
  # --devnum 1 = first visible device = the pinned GPU (CUDA_VISIBLE_DEVICES)
  # ADGPU_EXTRA: extra docking options (e.g. the fixed-work settings of phase1.sh)
  # shellcheck disable=SC2086
  echo --devnum 1 --ffile "$FLD" --lfile "$LIG" --nrun "$2" --seed "$1" --resnam "$3" ${XRAY:+--xraylfile "$XRAY"} ${ADGPU_EXTRA:-}
}

# copy_hint: how to get results off the box. Run the scp line on YOUR machine, in the
# terminal you normally ssh from (BOX_IP = the instance's public IP).
copy_hint() {
  log "copy off the box: on the box run"
  log "  tar czf ~/${RESULTS_PHASE}_results.tgz --exclude=bin --exclude=work -C $ROOT/results $RESULTS_PHASE"
  log "then on your machine:  scp USER@BOX_IP:~/${RESULTS_PHASE}_results.tgz ~/"
}
