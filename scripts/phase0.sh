#!/usr/bin/env bash
# Copyright (C) 2026 Lu Verissimo.
# Part of an independent performance study of AutoDock-GPU. Not affiliated with
# or endorsed by Scripps Research or the AutoDock-GPU developers.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. See LICENSE.
#
# Phase 0 driver. Run from anywhere inside the repo on the GPU box:
#   scripts/phase0.sh preflight   #  5 min  exact GPU/driver/CUDA/commit record, topology; starts session clock
#   scripts/phase0.sh build       # 20 min  default CUDA, TENSOR=ON, both +lineinfo (OpenCL: BUILDS+=ocl)
#   scripts/phase0.sh baseline    # 35 min  5 shipped complexes x {default, TENSOR=ON} x REPS
#   scripts/phase0.sh nsys        # 10 min  timeline + kernel table per case and build
#   scripts/phase0.sh ncu         # 30 min  targeted ncu --set full
#   scripts/phase0.sh set42       # only with an approved extension of the hard stop
# Session hard stop: HARD_STOP_MIN=120 from preflight; stages refuse to start past it.
# Timed/profiled work is pinned to GPU 0 and runs sequentially. EXPLORE=1 GPU_ID=1..3
# runs a stage on another GPU, output prefixed explore_ (never used for headline numbers).
# Add DRY_RUN=1 to print the commands without running GPU work.
# Each stage writes results/phase0/<stage>_<host>_<utc>/; copy it off afterwards.
set -euo pipefail
# shellcheck source=scripts/lib.sh
. "$(dirname "$0")/lib.sh"

SEED=${SEED:-42}
NRUN=${NRUN:-100}
REPS=${REPS:-5}
CASES=${CASES:-"1ac8 1stp 3ce3 3tmn 7cpa"}   # all 5 shipped complexes
PATHS=${PATHS:-"cuda cuda_tensor"}           # add "ocl" for the OpenCL comparison
BUILDS=${BUILDS:-"cuda cuda_tensor cuda_lineinfo cuda_tensor_lineinfo"}
RUN_TIMEOUT=${RUN_TIMEOUT:-600}              # per docking run (s)
DEADLINE_MIN=${DEADLINE_MIN:-35}             # baseline/set42: no new runs after this
NSYS_CASES=${NSYS_CASES:-"1stp 3ce3 7cpa"}
NSYS_PATHS=${NSYS_PATHS:-"cuda cuda_tensor"}
# ncu plan entries: case:topK:build. topK = top K kernels by nsys GPU time for that build.
NCU_PLAN=${NCU_PLAN:-"3ce3:3:cuda 3ce3:1:cuda_tensor 7cpa:1:cuda 7cpa:1:cuda_tensor"}
NCU_TIMEOUT=${NCU_TIMEOUT:-300}              # per ncu invocation (s); 6 x 5 min = 30 min worst case
NCU_SKIP=${NCU_SKIP:-10}                     # launches skipped per kernel (warm-up)
NCU_COUNT=${NCU_COUNT:-2}                    # launches profiled per kernel
SET42_URL=${SET42_URL:-https://github.com/diogomart/AD-GPU_set_of_42.git}
SET42_COMMIT=${SET42_COMMIT:-e7b7eada7b3c46e58e09165fbea9f62fdf646138}  # 2022-08-18, pinned
SET42_DIR=${SET42_DIR:-$ROOT/../AD-GPU_set_of_42}
SET42_CASES=${SET42_CASES:-all}
SET42_REPS=${SET42_REPS:-3}
BIN=$ROOT/results/phase0/bin                 # shared by all stages; not committed
PY=${PY:-python3}
# Used when no nsys kernel table exists yet (and in dry runs).
FALLBACK_KERNELS="gpu_gradient_minAD_kernel gpu_gen_and_eval_newpops_kernel gpu_sum_evals_kernel gpu_calc_initpop_kernel"

stage=${1:-}
T0=$(date +%s)
mins() { echo $(( ($(date +%s) - T0) / 60 )); }
# stop launching when the stage budget or the session hard stop is reached
past_deadline() {
  [ "$(mins)" -ge "$DEADLINE_MIN" ] && return 0
  [ "$DRY_RUN" != 1 ] && [ "$(session_left)" -le 0 ] && return 0
  return 1
}

need_bin() {
  [ "$DRY_RUN" = 1 ] && return 0
  [ -x "$BIN/adgpu_${1}_${NUMWI}wi" ] || die "missing $BIN/adgpu_${1}_${NUMWI}wi; run the build stage first"
}

latest_nsys_csv() {  # latest_nsys_csv <case> <build>
  ls -t "$ROOT"/results/phase0/nsys_*/"$1_$2"_kern_cuda_gpu_kern_sum.csv 2>/dev/null | head -1 || true
}

ncu_kill_flag() {  # newer ncu: --kill on|off; older: --kill yes|no
  local h; h=$(ncu --help 2>/dev/null | grep -A4 -- '--kill' || true)
  if   grep -qi 'on' <<<"$h" && ! grep -qi 'yes' <<<"$h"; then echo "--kill on"
  elif grep -qi 'yes' <<<"$h"; then echo "--kill yes"
  else echo ""; fi
}

# dock <setup-fn> <case> <path> <rep> <outdir>: one timed docking run
dock() {
  local setup=$1 c=$2 path=$3 r=$4 dir=$5 name
  name=${c}_${path}_s${SEED}_r${r}
  log "$name ($(mins) min)"
  "$setup" "$c" "$OUT/work/$c"
  # shellcheck disable=SC2046
  ( cd "$OUT/work/$c" && run timeout "$RUN_TIMEOUT" /usr/bin/time -v \
      "$BIN/adgpu_${path}_${NUMWI}wi" $(adgpu_args "$SEED" "$NRUN" "$dir/$name") ) \
    > "$dir/$name.stdout" 2> "$dir/$name.time" \
    || echo "FAILED: $name (exit $?)" | tee -a "$OUT/failures.txt"
  # AutoDock-GPU can exit 0 after a failed setup (e.g. ligand parse errors)
  if grep -qE "The job was not successful|Error in setup of Job" "$dir/$name.stdout" 2>/dev/null; then
    echo "FAILED: $name (job not successful, exit 0)" | tee -a "$OUT/failures.txt"
  fi
  # which device AutoDock-GPU says it used (checks the pinning, incl. OpenCL)
  { printf '%s: ' "$name"; grep -i -m2 'device' "$dir/$name.stdout" | tr '\n' ' '; echo; } >> "$OUT/run_devices.txt" 2>/dev/null || true
}

# run_matrix <setup-fn> <cases> <reps>: reps outermost, builds alternate within a rep,
# so slow drift (thermals, clocks) hits all builds equally.
run_matrix() {
  local setup=$1 cases=$2 reps=$3 r c p
  mkdir -p "$OUT/runs"
  for p in $PATHS; do need_bin "$p"; done
  for r in $(seq 1 "$reps"); do
    for c in $cases; do
      for p in $PATHS; do
        if past_deadline; then
          echo "DEADLINE: stage ${DEADLINE_MIN} min or session hard stop reached; stopped before $c $p r$r" | tee -a "$OUT/failures.txt" >&2
          return 0
        fi
        dock "$setup" "$c" "$p" "$r" "$OUT/runs"
      done
    done
  done
}

case $stage in
# ---------------------------------------------------------------------------
preflight)
  if [ "$DRY_RUN" != 1 ] && { [ ! -f "$SESSION_FILE" ] || [ "${RESET_SESSION:-0}" = 1 ]; }; then
    mkdir -p "$(dirname "$SESSION_FILE")"; date +%s > "$SESSION_FILE"
  fi
  new_outdir preflight
  {
    echo "## exact record (also in hardware.txt)"
    echo "session_start_utc: $(date -u -d @"$(cat "$SESSION_FILE" 2>/dev/null || date +%s)" +%FT%TZ)  hard_stop_min: $HARD_STOP_MIN"
    echo "--- all GPUs"; nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id,uuid --format=csv
    echo "--- headline GPU (index $GPU_ID)"; nvidia-smi -i "$GPU_ID" --query-gpu=index,name,uuid,pci.bus_id,pci.device_id,vbios_version,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,ecc.mode.current,compute_mode --format=csv
    echo "--- nvidia-smi header (driver / driver CUDA version)"; nvidia-smi | head -4
    echo "--- topology (nvidia-smi topo -m)"; nvidia-smi topo -m
    echo "--- nvcc";   nvcc --version | tail -2
    echo "--- cuda version.json"; cat "$CUDA_HOME/version.json" 2>/dev/null | head -12
    echo "--- toolkit packages"; dpkg -l 2>/dev/null | grep -iE "nvidia-cuda-toolkit|cuda-nvcc|nsight" | awk '{print $2, $3}'
    echo "--- build paths: GPU_INCLUDE_PATH=$GPU_INCLUDE_PATH GPU_LIBRARY_PATH=$GPU_LIBRARY_PATH"
    echo "--- git"; echo "head: $(git -C "$ROOT" rev-parse HEAD)"
    echo "upstream_base: $(git -C "$ROOT" merge-base HEAD upstream/develop 2>/dev/null || echo n/a)"
    echo "adgpu_version_string: $(cd "$ROOT" && ./version_string.sh 2>/dev/null)"
    echo "submodules:"; git -C "$ROOT" submodule status
    echo "git_status:"; git -C "$ROOT" status --short
    echo "--- gcc";    gcc --version | head -1
    echo "--- nsys";   nsys --version 2>&1
    echo "--- ncu";    ncu --version 2>&1 | tail -1
    echo "--- ncu kill flag: $(ncu_kill_flag)"
    echo "--- sudo -n: $(sudo -n true 2>&1 && echo ok)"
    echo "--- RmProfilingAdminOnly"; grep -i RmProfilingAdminOnly /proc/driver/nvidia/params 2>&1
    echo "--- OpenCL ICDs"; ls -l /etc/OpenCL/vendors/ 2>&1
    echo "--- uname";  uname -a
    echo "--- cpu"; lscpu | grep -E 'Model name|^CPU\(s\)'
    echo "--- disk"; df -h "$ROOT"
    echo "--- nvidia-smi"; nvidia-smi
    echo "--- clocks/perf"; nvidia-smi -q -d CLOCK,PERFORMANCE,POWER
    echo "--- full nvidia-smi -q"; nvidia-smi -q
  } > "$OUT/env.txt" 2>&1 || true
  sed -n '/^## exact/,/^--- gcc/p' "$OUT/env.txt" > "$OUT/hardware.txt"
  [ "$DRY_RUN" = 1 ] && echo "DRY: environment probe written to $OUT/env.txt"
  nvidia-smi topo -m > "$OUT/topo.txt" 2>&1 || true
  cat "$OUT/hardware.txt"
  grep -E '^--- (ncu|nsys|sudo|ncu kill|RmProf)' -A1 "$OUT/env.txt"
  log "session clock started; hard stop at $HARD_STOP_MIN min"; copy_hint ;;
# ---------------------------------------------------------------------------
build)
  check_session 20
  new_outdir build
  ( cd "$ROOT" && run git submodule update --init wmma_extension ) > "$OUT/submodule.log" 2>&1 \
    || die "wmma_extension submodule init failed (needed for TENSOR=ON); see $OUT/submodule.log"
  git -C "$ROOT" submodule status > "$OUT/submodule_status.txt" 2>&1 || true
  for k in $BUILDS; do
    log "build $k ($(mins) min)"
    build "$k" > "$OUT/build_${k}.log" 2>&1 || die "build $k failed; see $OUT/build_${k}.log"
  done
  run cp "$BIN"/*.res-usage.txt "$OUT/" 2>/dev/null || true
  for k in $BUILDS; do  # registers / smem per kernel from ptxas -v
    grep -E "Compiling entry function|Used [0-9]+ registers" "$OUT/build_${k}.log" > "$OUT/ptxas_${k}.txt" 2>/dev/null || true
  done
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
baseline)
  check_session 10
  new_outdir baseline
  run_matrix setup_case "$CASES" "$REPS"
  run "$PY" "$ROOT/scripts/summarize_runs.py" "$OUT/runs" | tee "$OUT/summary.tsv"
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
set42)
  check_session 15
  new_outdir set42
  if [ ! -d "$SET42_DIR/.git" ]; then
    run git clone "$SET42_URL" "$SET42_DIR"
  fi
  run git -C "$SET42_DIR" checkout -q "$SET42_COMMIT"
  { echo "url: $SET42_URL"; echo "commit: $(git -C "$SET42_DIR" rev-parse HEAD 2>/dev/null || echo "$SET42_COMMIT")"; } > "$OUT/set42_source.txt"
  run cp "$SET42_DIR/ligand_properties.csv" "$OUT/" 2>/dev/null || true
  if [ "$SET42_CASES" = all ]; then
    if [ -d "$SET42_DIR/data" ]; then
      SET42_CASES=$(cd "$SET42_DIR/data" && ls -d -- */ | tr -d / | tr '\n' ' ')
    else
      SET42_CASES="1ac8 1kzk"   # dry run without a clone
    fi
  fi
  if [ "$DRY_RUN" = 1 ] && [ ! -d "$SET42_DIR/data" ]; then
    echo "DRY: would run set42 cases: $SET42_CASES"
  else
    run_matrix setup_set42 "$SET42_CASES" "$SET42_REPS"
    run "$PY" "$ROOT/scripts/summarize_runs.py" "$OUT/runs" | tee "$OUT/summary.tsv"
  fi
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
nsys)
  check_session 10
  new_outdir nsys
  for p in $NSYS_PATHS; do need_bin "$p"; done
  for c in $NSYS_CASES; do
    setup_case "$c" "$OUT/work/$c"
    for p in $NSYS_PATHS; do
      if [ "$DRY_RUN" != 1 ] && [ "$(session_left)" -le 0 ]; then
        echo "HARD STOP: not starting nsys $c $p" | tee -a "$OUT/failures.txt" >&2; break 2
      fi
      rep=$OUT/${c}_${p}
      log "nsys $c $p ($(mins) min)"
      # shellcheck disable=SC2046
      ( cd "$OUT/work/$c" && run timeout "$RUN_TIMEOUT" nsys profile \
          --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true \
          -o "$rep" "$BIN/adgpu_${p}_${NUMWI}wi" $(adgpu_args "$SEED" "$NRUN" "${rep}_run") ) \
        > "${rep}_nsys.stdout" 2>&1 || echo "FAILED: nsys $c $p" | tee -a "$OUT/failures.txt"
      for r in cuda_gpu_kern_sum cuda_api_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum; do
        run nsys stats --report "$r" --format csv --force-export=true \
            --output "${rep}_kern" "${rep}.nsys-rep" > /dev/null 2>> "${rep}_stats.err" || true
      done
      run "$PY" "$ROOT/scripts/nsys_top_kernels.py" "${rep}_kern_cuda_gpu_kern_sum.csv" 10 \
        | tee "${rep}_top_kernels.txt" || true
    done
  done
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
ncu)
  check_session 10
  new_outdir ncu
  NCU=$(command -v ncu || echo ncu)
  if [ "$DRY_RUN" = 1 ]; then KILL="--kill on"; else KILL=$(ncu_kill_flag); fi
  for item in $NCU_PLAN; do
    IFS=: read -r c k b <<<"$item"
    nb=${b}_lineinfo; need_bin "$nb"
    setup_case "$c" "$OUT/work/$c"
    csv=$(latest_nsys_csv "$c" "$b")
    if [ -n "$csv" ]; then
      kernels=$("$PY" "$ROOT/scripts/nsys_top_kernels.py" "$csv" "$k" --names)
      log "$c/$b: top $k kernels from $csv"
    else
      kernels=$(echo "$FALLBACK_KERNELS" | tr ' ' '\n' | head -"$k")
      log "$c/$b: no nsys table found, using fallback kernel list"
    fi
    for kn in $kernels; do
      inst=99
      [ -n "$csv" ] && inst=$("$PY" "$ROOT/scripts/nsys_top_kernels.py" "$csv" 0 --instances "$kn")
      skip=$NCU_SKIP; cnt=$NCU_COUNT
      [ "$inst" -le $((skip + cnt)) ] && skip=$(( inst > cnt ? inst - cnt : 0 ))
      [ "$inst" -lt "$cnt" ] && cnt=$inst
      if [ "$DRY_RUN" != 1 ] && [ "$(session_left)" -lt $(( NCU_TIMEOUT / 60 + 1 )) ]; then
        echo "HARD STOP: not starting ncu $c $b $kn ($(session_left) min left)" | tee -a "$OUT/failures.txt" >&2
        break 2
      fi
      rep=$OUT/${c}_${b}_${kn}
      log "ncu $c $b $kn skip=$skip count=$cnt ($(mins) min)"
      # shellcheck disable=SC2046,SC2086
      # sudo resets the environment, so the pinning is passed explicitly; inside the
      # pinned view the GPU is device 0, hence --devices 0.
      ( cd "$OUT/work/$c" && run sudo env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" timeout "$NCU_TIMEOUT" "$NCU" \
          --devices 0 --set full --import-source yes --target-processes all \
          -k "regex:${kn}" --launch-skip "$skip" --launch-count "$cnt" $KILL \
          -f -o "$rep" "$BIN/adgpu_${nb}_${NUMWI}wi" $(adgpu_args "$SEED" "$NRUN" "${rep}_run") ) \
        > "${rep}.ncu.log" 2>&1 || echo "FAILED/TIMEOUT: ncu $c $b $kn (see ${rep}.ncu.log)" | tee -a "$OUT/failures.txt"
      [ "$DRY_RUN" = 1 ] || sudo chown -R "$(id -u):$(id -g)" "$OUT"
      # Nsight Compute >= 2026.x writes compressed .ncu-repz; older versions .ncu-rep
      rf=""; for ext in ncu-repz ncu-rep; do [ -f "${rep}.${ext}" ] && rf="${rep}.${ext}" && break; done
      [ "$DRY_RUN" = 1 ] && rf="${rep}.ncu-repz"
      if [ -n "$rf" ]; then
        run "$NCU" -i "$rf" --page details > "${rep}_details.txt" 2>&1 || true
        run "$NCU" -i "$rf" --page details --csv > "${rep}_details.csv" 2>&1 || true
        run "$NCU" -i "$rf" --page raw --csv > "${rep}_raw.csv" 2>&1 || true
        run "$NCU" -i "$rf" --page source --csv --print-source sass > "${rep}_source_sass.csv" 2>&1 || true
        # CUDA-source-correlated view (needs -lineinfo + --import-source); option spelling
        # differs between ncu versions, so try both and keep whichever succeeds.
        run "$NCU" -i "$rf" --page source --csv --print-source cuda,sass > "${rep}_source_cuda_sass.csv" 2>&1 \
          || run "$NCU" -i "$rf" --page source --csv --print-source sass,cuda > "${rep}_source_cuda_sass.csv" 2>&1 || true
        run "$NCU" -i "$rf" --page source --csv --print-source cuda > "${rep}_source_cuda.csv" 2>&1 || true
        run "$PY" "$ROOT/scripts/ncu_summary.py" "${rep}_raw.csv" >> "$OUT/ncu_summary.txt" || true
      else
        echo "NO REPORT: ncu $c $b $kn (see ${rep}.ncu.log)" | tee -a "$OUT/failures.txt"
      fi
    done
  done
  if [ -f "$OUT/ncu_summary.txt" ]; then cat "$OUT/ncu_summary.txt"; fi
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
*) sed -n '11,23p' "$0"; exit 2 ;;
esac
