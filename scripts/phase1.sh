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
# Phase 1 session driver: build every variant from its own branch (git worktrees), then
#   scripts/phase1.sh preflight   # session clock, tool versions, branch SHAs
#   scripts/phase1.sh build       # all variants (+ -lineinfo builds for ncu)
#   scripts/phase1.sh gate        # correctness gate: bit-identical dlg fix1 vs TENSOR=ON; determinism
#   scripts/phase1.sh e2e         # GPU 0: default settings, end-to-end, all variants
#   scripts/phase1.sh fixedwork   # GPU 0: fixed work (--autostop 0 --heuristics 0 --nev N)
#   scripts/phase1.sh nsys        # GPU 0: kernel table per variant, fixed work
#   scripts/phase1.sh ncu         # GPU 0: source-level ncu on minAD per variant
#   [EXPLORE=1 GPU_ID=1] DIST_VARIANT=v [DIST_SET=main|null] scripts/phase1.sh dist
#                                 # seed distributions (GPUs 1-3 on an 8-GPU box; GPU 0 after timing on 1 GPU)
#   [EXPLORE=1 GPU_ID=1] scripts/phase1.sh sanitize   # compute-sanitizer (never on the ceiling variant)
#   scripts/phase1.sh set42       # optional, GPU 0: AD-GPU_set_of_42, reference vs best variant
#   scripts/phase1.sh compare     # CPU only: gate verdicts, distributions, speedup tables
# Confirmatory session (owner decision 2026-11-04: stop optimizing, confirm):
#   scripts/phase1.sh confirm_fw  # 7cpa fixed work, CF_REPS (>=10) interleaved reps, wall CI
#   scripts/phase1.sh confirm_nsys # same under nsys: minAD time per Meval with CI
#   scripts/phase1.sh nonident    # where does fix #1's non-identity come from? (dlg comparisons)
#   scripts/phase1.sh racecheck_li # racecheck on -lineinfo builds, hazards mapped to source lines
# DRY_RUN=1 prints commands. All output goes to results/phase1/<stage>_gpu<N>_<host>_<utc>/.
set -euo pipefail
export RESULTS_PHASE=phase1
# shellcheck source=scripts/lib.sh
. "$(dirname "$0")/lib.sh"

SEED=${SEED:-42}
NRUN=${NRUN:-100}
CASES=${CASES:-"1stp 3ce3 3tmn 7cpa"}          # shipped complexes that dock (1ac8 fails upstream)
REPS=${REPS:-5}
FW_REPS=${FW_REPS:-3}
FW_ARGS=${FW_ARGS:-"--autostop 0 --heuristics 0 --nev 3000000"}
RUN_TIMEOUT=${RUN_TIMEOUT:-900}
DEADLINE_MIN=${DEADLINE_MIN:-30}
GATE_SEEDS=${GATE_SEEDS:-"42 7 123"}
# Seed design (amended pre-registration 2026-11-04): DIST_N seeds per complex, DIST_N_7CPA for 7cpa.
# DIST_SET=main uses seeds 1..n; DIST_SET=null uses the disjoint block n+1..2n and labels the
# runs <variant>_null (null calibration of the reference).
DIST_N=${DIST_N:-48}
DIST_N_7CPA=${DIST_N_7CPA:-24}
DIST_SET=${DIST_SET:-main}
DIST_VARIANT=${DIST_VARIANT:-base_tensor}
if [ "$DIST_SET" = null ]; then DIST_LABEL=${DIST_LABEL:-${DIST_VARIANT}_null}; else DIST_LABEL=${DIST_LABEL:-$DIST_VARIANT}; fi
dist_n() { if [ "$1" = 7cpa ]; then echo "$DIST_N_7CPA"; else echo "$DIST_N"; fi; }
REF=${REF:-base_tensor}
NCU_VARIANTS=${NCU_VARIANTS:-"base_tensor fix1_tensor fix12_shfl"}
NCU_CASES=${NCU_CASES:-"7cpa 3ce3"}
NCU_TIMEOUT=${NCU_TIMEOUT:-300}
NCU_SKIP=${NCU_SKIP:-10}
NCU_COUNT=${NCU_COUNT:-2}
SAN_VARIANTS=${SAN_VARIANTS:-"base_tensor fix1_tensor fix12_shfl"}
SAN_ARGS=${SAN_ARGS:-"--autostop 0 --heuristics 0 --nev 30000"}
SET42_PATHS=${SET42_PATHS:-"base_tensor fix1_tensor"}
SET42_URL=${SET42_URL:-https://github.com/diogomart/AD-GPU_set_of_42.git}
SET42_COMMIT=${SET42_COMMIT:-e7b7eada7b3c46e58e09165fbea9f62fdf646138}
SET42_DIR=${SET42_DIR:-$ROOT/../AD-GPU_set_of_42}
WT=${WT:-$ROOT/../adgpu-wt}
BIN=$ROOT/results/phase1/bin
PY=${PY:-python3}

# name|git ref|make args|extra nvcc flags. Edit here (or set VARIANTS) to change the matrix.
VARIANTS=${VARIANTS:-"\
base_tensor|ce6eed1b07d6e7094dfa4a7f6719c39953f6616a|TENSOR=ON|
fix1_tensor|origin/teardown/phase1-fix1-intra-segmented|TENSOR=ON|
fix1_legacy_tensor|origin/teardown/phase1-fix1-intra-segmented|TENSOR=ON|-DINTRA_ATOMICS_LEGACY
ceiling_tensor|origin/teardown/phase1-ceiling-noatomic|TENSOR=ON|
fix12_shfl|origin/teardown/phase1-fix2-shfl-reduce|SHFLREDUCE=ON|
fix2only_shfl|origin/teardown/phase1-fix2-shfl-reduce|SHFLREDUCE=ON|-DINTRA_ATOMICS_LEGACY"}
ALL_PATHS=$(echo "$VARIANTS" | cut -d'|' -f1 | tr '\n' ' ')
PATHS=${PATHS:-$ALL_PATHS}
CF_CASE=${CF_CASE:-7cpa}
CF_REPS=${CF_REPS:-10}
CF_PATHS=${CF_PATHS:-"base_tensor fix1_tensor fix12_shfl fix2only_shfl ceiling_tensor"}
RC_VARIANTS=${RC_VARIANTS:-"base_tensor fix1_tensor fix12_shfl"}

stage=${1:-}
T0=$(date +%s)
mins() { echo $(( ($(date +%s) - T0) / 60 )); }
past_deadline() {
  [ "$(mins)" -ge "$DEADLINE_MIN" ] && return 0
  [ "$DRY_RUN" != 1 ] && [ "$(session_left)" -le 0 ] && return 0
  return 1
}
need_bin() {
  [ "$DRY_RUN" = 1 ] && return 0
  [ -x "$BIN/adgpu_${1}_${NUMWI}wi" ] || die "missing $BIN/adgpu_${1}_${NUMWI}wi; run the build stage first"
}

# dock <setup-fn> <case> <variant> <seed> <rep> <dir> [<label>]: one timed docking job.
# <label> (default: variant) is the build tag in the output names.
dock() {
  local setup=$1 c=$2 v=$3 s=$4 r=$5 dir=$6 name
  name=${c}_${7:-$v}_s${s}_r${r}
  log "$name ($(mins) min)"
  "$setup" "$c" "$OUT/work/$c"
  # shellcheck disable=SC2046
  ( cd "$OUT/work/$c" && run timeout "$RUN_TIMEOUT" /usr/bin/time -v \
      "$BIN/adgpu_${v}_${NUMWI}wi" $(adgpu_args "$s" "$NRUN" "$dir/$name") ) \
    > "$dir/$name.stdout" 2> "$dir/$name.time" \
    || echo "FAILED: $name (exit $?)" | tee -a "$OUT/failures.txt"
  if grep -qE "The job was not successful|Error in setup of Job" "$dir/$name.stdout" 2>/dev/null; then
    echo "FAILED: $name (job not successful, exit 0)" | tee -a "$OUT/failures.txt"
  fi
}

# matrix <setup-fn> <cases> <reps> <seed>: reps outermost, variants alternate within a rep
matrix() {
  local setup=$1 cases=$2 reps=$3 s=$4 r c v
  mkdir -p "$OUT/runs"
  for v in $PATHS; do need_bin "$v"; done
  for r in $(seq 1 "$reps"); do
    for c in $cases; do
      for v in $PATHS; do
        if past_deadline; then
          echo "DEADLINE: stopped before $c $v r$r" | tee -a "$OUT/failures.txt" >&2; return 0
        fi
        dock "$setup" "$c" "$v" "$s" "$r" "$OUT/runs"
      done
    done
  done
}

case $stage in
# ---------------------------------------------------------------------------
preflight)
  "$ROOT/scripts/phase0.sh" preflight
  new_outdir preflight_phase1
  {
    echo "--- compute-sanitizer"; compute-sanitizer --version 2>&1 | tail -2
    echo "--- variant refs"
    # never prompt for credentials on the box (the clone's token is removed after cloning)
    GIT_TERMINAL_PROMPT=0 git -C "$ROOT" fetch -q origin 2>&1 || echo "fetch skipped/failed (using refs from the clone)"
    echo "$VARIANTS" | while IFS='|' read -r n ref mk ex; do
      printf '%s|%s|%s|%s|%s\n' "$n" "$ref" "$(git -C "$ROOT" rev-parse --short=12 "$ref" 2>&1)" "$mk" "$ex"
    done
  } > "$OUT/phase1_env.txt" 2>&1
  cat "$OUT/phase1_env.txt"
  copy_hint ;;
# ---------------------------------------------------------------------------
build)
  check_session 15
  new_outdir build
  ( cd "$ROOT" && run git submodule update --init wmma_extension ) > "$OUT/submodule.log" 2>&1 \
    || die "wmma_extension init failed (TENSOR=ON needs it); see $OUT/submodule.log"
  mkdir -p "$BIN" "$WT"
  : > "$OUT/builds.txt"
  echo "$VARIANTS" | while IFS='|' read -r n ref mk ex; do
    [ -n "$n" ] || continue
    sha=$(git -C "$ROOT" rev-parse "$ref")
    w=$WT/$n
    if [ ! -d "$w" ]; then run git -C "$ROOT" worktree add -q --detach "$w" "$sha"; fi
    [ "$DRY_RUN" = 1 ] && mkdir -p "$w"
    run git -C "$w" checkout -q --detach "$sha"
    # the submodule is only a header library: share the main checkout's copy
    [ "$DRY_RUN" = 1 ] || { rm -rf "$w/wmma_extension"; ln -s "$ROOT/wmma_extension" "$w/wmma_extension"; }
    for li in "" "_lineinfo"; do
      if [ -n "$li" ] && ! grep -qw "$n" <<<"$NCU_VARIANTS"; then continue; fi
      flags="$ex"; [ -n "$li" ] && flags="$flags -lineinfo"
      log "build $n$li @ ${sha:0:12} ($(mins) min)"
      rm -f "$w/bin/autodock_gpu_${NUMWI}wi"
      # shellcheck disable=SC2086
      ( cd "$w" && run make DEVICE=CUDA NUMWI="$NUMWI" TARGETS=80 $mk "NVCC=nvcc $flags" ) \
        > "$OUT/build_${n}${li}.log" 2>&1 || die "build $n$li failed; see $OUT/build_${n}${li}.log"
      run cp "$w/bin/autodock_gpu_${NUMWI}wi" "$BIN/adgpu_${n}${li}_${NUMWI}wi"
      run cuobjdump -res-usage "$BIN/adgpu_${n}${li}_${NUMWI}wi" > "$OUT/res-usage_${n}${li}.txt" 2>&1 || true
      grep -E "Compiling entry function|Used [0-9]+ registers" "$OUT/build_${n}${li}.log" > "$OUT/ptxas_${n}${li}.txt" 2>/dev/null || true
      echo "$n$li|$ref|$sha|$mk|$flags" >> "$OUT/builds.txt"
    done
  done
  cat "$OUT/builds.txt"
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
gate)
  # Primary gate: fix1_tensor must produce byte-identical .dlg to base_tensor at fixed seeds.
  # Determinism check: the shuffle-reduction builds must reproduce themselves bit for bit.
  check_session 5
  new_outdir gate
  for s in $GATE_SEEDS; do
    for c in $CASES; do
      for v in base_tensor fix1_tensor; do
        mkdir -p "$OUT/$v"; need_bin "$v"; dock setup_case "$c" "$v" "$s" 1 "$OUT/$v"
      done
    done
  done
  for v in fix12_shfl fix2only_shfl; do
    for k in a b; do
      mkdir -p "$OUT/${v}_$k"; need_bin "$v"
      for c in $CASES; do dock setup_case "$c" "$v" "$SEED" 1 "$OUT/${v}_$k"; done
    done
  done
  {
    echo "=== PRIMARY GATE: fix1_tensor vs base_tensor (seeds $GATE_SEEDS)"
    run "$PY" "$ROOT/scripts/dlg_compare.py" "$OUT/base_tensor" "$OUT/fix1_tensor" || true
    for v in fix12_shfl fix2only_shfl; do
      echo; echo "=== DETERMINISM: $v run a vs run b (seed $SEED)"
      run "$PY" "$ROOT/scripts/dlg_compare.py" "$OUT/${v}_a" "$OUT/${v}_b" || true
    done
  } | tee "$OUT/gate_verdict.txt"
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
e2e)
  check_session 10
  new_outdir e2e
  matrix setup_case "$CASES" "$REPS" "$SEED"
  run "$PY" "$ROOT/scripts/summarize_runs.py" "$OUT/runs" --ref "$REF" | tee "$OUT/summary.tsv"
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
fixedwork)
  check_session 10
  new_outdir fixedwork
  echo "$FW_ARGS" > "$OUT/fixedwork_args.txt"
  ADGPU_EXTRA="$FW_ARGS" matrix setup_case "$CASES" "$FW_REPS" "$SEED"
  run "$PY" "$ROOT/scripts/summarize_runs.py" "$OUT/runs" --ref "$REF" | tee "$OUT/summary.tsv"
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
nsys)
  check_session 8
  new_outdir nsys
  for c in ${NSYS_CASES:-7cpa 3ce3}; do
    setup_case "$c" "$OUT/work/$c"
    for v in $PATHS; do
      need_bin "$v"
      if [ "$DRY_RUN" != 1 ] && [ "$(session_left)" -le 0 ]; then
        echo "HARD STOP: not starting nsys $c $v" | tee -a "$OUT/failures.txt" >&2; break 2
      fi
      rep=$OUT/${c}_${v}
      log "nsys $c $v ($(mins) min)"
      # shellcheck disable=SC2046
      ( cd "$OUT/work/$c" && run timeout "$RUN_TIMEOUT" nsys profile \
          --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true \
          -o "$rep" "$BIN/adgpu_${v}_${NUMWI}wi" $(ADGPU_EXTRA="$FW_ARGS" adgpu_args "$SEED" "$NRUN" "${rep}_run") ) \
        > "${rep}_nsys.stdout" 2>&1 || echo "FAILED: nsys $c $v" | tee -a "$OUT/failures.txt"
      run nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
          --output "${rep}_kern" "${rep}.nsys-rep" > /dev/null 2>> "${rep}_stats.err" || true
      run "$PY" "$ROOT/scripts/nsys_top_kernels.py" "${rep}_kern_cuda_gpu_kern_sum.csv" 5 \
        | tee "${rep}_top_kernels.txt" || true
    done
  done
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
ncu)
  check_session 10
  new_outdir ncu
  NCU=$(command -v ncu || echo ncu)
  KILL="--kill on"
  for c in $NCU_CASES; do
    setup_case "$c" "$OUT/work/$c"
    for v in $NCU_VARIANTS; do
      nb=${v}_lineinfo; need_bin "$nb"
      if [ "$DRY_RUN" != 1 ] && [ "$(session_left)" -lt $(( NCU_TIMEOUT / 60 + 1 )) ]; then
        echo "HARD STOP: not starting ncu $c $v" | tee -a "$OUT/failures.txt" >&2; break 2
      fi
      rep=$OUT/${c}_${v}_gpu_gradient_minAD_kernel
      log "ncu $c $v ($(mins) min)"
      # default settings (as in Phase 0) so the profile matches the Phase 0 reference
      # shellcheck disable=SC2046,SC2086
      ( cd "$OUT/work/$c" && run sudo env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" timeout "$NCU_TIMEOUT" "$NCU" \
          --devices 0 --set full --import-source yes --target-processes all \
          -k "regex:gpu_gradient_minAD_kernel" --launch-skip "$NCU_SKIP" --launch-count "$NCU_COUNT" $KILL \
          -f -o "$rep" "$BIN/adgpu_${nb}_${NUMWI}wi" $(adgpu_args "$SEED" "$NRUN" "${rep}_run") ) \
        > "${rep}.ncu.log" 2>&1 || echo "FAILED/TIMEOUT: ncu $c $v" | tee -a "$OUT/failures.txt"
      [ "$DRY_RUN" = 1 ] || sudo chown -R "$(id -u):$(id -g)" "$OUT"
      rf=""; for ext in ncu-repz ncu-rep; do [ -f "${rep}.${ext}" ] && rf="${rep}.${ext}" && break; done
      [ "$DRY_RUN" = 1 ] && rf="${rep}.ncu-repz"
      if [ -n "$rf" ]; then
        run "$NCU" -i "$rf" --page details > "${rep}_details.txt" 2>&1 || true
        run "$NCU" -i "$rf" --page raw --csv > "${rep}_raw.csv" 2>&1 || true
        run "$NCU" -i "$rf" --page source --csv --print-source sass > "${rep}_source_sass.csv" 2>&1 || true
        run "$NCU" -i "$rf" --page source --csv --print-source cuda,sass > "${rep}_source_cuda_sass.csv" 2>&1 || true
        run "$PY" "$ROOT/scripts/ncu_summary.py" "${rep}_raw.csv" >> "$OUT/ncu_summary.txt" || true
        run "$PY" "$ROOT/scripts/ncu_regions.py" "${rep}_source_cuda_sass.csv" > "${rep}_regions.txt" 2>&1 || true
      else
        echo "NO REPORT: ncu $c $v" | tee -a "$OUT/failures.txt"
      fi
    done
  done
  if [ -f "$OUT/ncu_summary.txt" ]; then cat "$OUT/ncu_summary.txt"; fi
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
dist)
  # correctness distributions (not timing): run on GPUs 1-3 with EXPLORE=1
  check_session 5
  new_outdir "dist_${DIST_LABEL}"
  echo "variant=$DIST_VARIANT label=$DIST_LABEL set=$DIST_SET n=$DIST_N n_7cpa=$DIST_N_7CPA cases=$CASES" > "$OUT/dist_args.txt"
  mkdir -p "$OUT/runs"; need_bin "$DIST_VARIANT"
  DEADLINE_MIN=${DIST_DEADLINE_MIN:-25}
  maxn=$DIST_N; [ "$DIST_N_7CPA" -gt "$maxn" ] && maxn=$DIST_N_7CPA
  for i in $(seq 1 "$maxn"); do          # seed index outermost: all complexes progress together
    for c in $CASES; do
      n=$(dist_n "$c"); [ "$i" -le "$n" ] || continue
      s=$i; [ "$DIST_SET" = null ] && s=$(( n + i ))
      if past_deadline; then echo "DEADLINE: stopped before $c s$s" | tee -a "$OUT/failures.txt" >&2; break 2; fi
      dock setup_case "$c" "$DIST_VARIANT" "$s" 1 "$OUT/runs" "$DIST_LABEL"
    done
  done
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
sanitize)
  check_session 5
  new_outdir sanitize
  setup_case 1stp "$OUT/work/1stp"
  for v in $SAN_VARIANTS; do
    case $v in *ceiling*) echo "SKIP: no sanitizer runs on the ceiling variant ($v)" | tee -a "$OUT/failures.txt"; continue ;; esac
    need_bin "$v"
    for tool in memcheck racecheck initcheck synccheck; do
      if past_deadline; then echo "DEADLINE: stopped before $v $tool" | tee -a "$OUT/failures.txt" >&2; break 2; fi
      log "compute-sanitizer $tool $v ($(mins) min)"
      # shellcheck disable=SC2046
      ( cd "$OUT/work/1stp" && run timeout 900 compute-sanitizer --tool "$tool" --print-limit 20 \
          "$BIN/adgpu_${v}_${NUMWI}wi" $(ADGPU_EXTRA="$SAN_ARGS" adgpu_args "$SEED" 2 "$OUT/${v}_${tool}") ) \
        > "$OUT/${v}_${tool}.log" 2>&1 || echo "NONZERO EXIT: $v $tool" | tee -a "$OUT/failures.txt"
      echo "$v $tool: $(grep -h "ERROR SUMMARY\|RACECHECK SUMMARY" "$OUT/${v}_${tool}.log" | tail -1)" | tee -a "$OUT/sanitize_summary.txt"
    done
  done
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
set42)
  check_session 15
  new_outdir set42
  [ -d "$SET42_DIR/.git" ] || run git clone -q "$SET42_URL" "$SET42_DIR"
  run git -C "$SET42_DIR" checkout -q "$SET42_COMMIT"
  if [ "$DRY_RUN" = 1 ] && [ ! -d "$SET42_DIR/data" ]; then
    echo "DRY: would run set42 ($SET42_PATHS) on all complexes of $SET42_DIR"
  else
    cases=$(cd "$SET42_DIR/data" && ls -d -- */ | tr -d / | tr '\n' ' ')
    PATHS=$SET42_PATHS matrix setup_set42 "$cases" "${SET42_REPS:-1}" "$SEED"
    run "$PY" "$ROOT/scripts/summarize_runs.py" "$OUT/runs" --ref "$REF" | tee "$OUT/summary.tsv"
  fi
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
compare)
  # CPU only, reads the latest folders: gate verdict, distributions vs reference.
  R=$ROOT/results/phase1
  g=$(ls -td "$R"/gate_* 2>/dev/null | head -1); [ -n "$g" ] && cat "$g/gate_verdict.txt"
  # explore_dist_* on an 8-GPU box (GPUs 1-3); dist_* when run on GPU 0 of a 1-GPU box
  dd=$(ls -d "$R"/explore_dist_*/runs "$R"/dist_*/runs 2>/dev/null || true)
  if [ -n "$dd" ]; then
    # shellcheck disable=SC2086
    run "$PY" "$ROOT/scripts/dist_compare.py" "$REF" $dd --calibrate-on "${CALIBRATE_ON:-criterion}" ${DIST_TOL:-}
  fi ;;
# ---------------------------------------------------------------------------
confirm_fw)
  check_session 10
  new_outdir confirm_fw
  echo "$FW_ARGS case=$CF_CASE reps=$CF_REPS paths=$CF_PATHS" > "$OUT/fixedwork_args.txt"
  DEADLINE_MIN=${CF_DEADLINE:-20} PATHS=$CF_PATHS ADGPU_EXTRA="$FW_ARGS" matrix setup_case "$CF_CASE" "$CF_REPS" "$SEED"
  run "$PY" "$ROOT/scripts/ci_speedup.py" wall "$OUT/runs" --case "$CF_CASE" --ref "$REF" | tee "$OUT/ci_wall.tsv"
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
confirm_nsys)
  check_session 10
  new_outdir confirm_nsys
  setup_case "$CF_CASE" "$OUT/work/$CF_CASE"
  for r in $(seq 1 "$CF_REPS"); do
    for v in $CF_PATHS; do
      need_bin "$v"
      if [ "$(mins)" -ge "${CFN_DEADLINE:-25}" ] || { [ "$DRY_RUN" != 1 ] && [ "$(session_left)" -le 3 ]; }; then
        echo "DEADLINE: stopped before $v r$r" | tee -a "$OUT/failures.txt" >&2; break 2
      fi
      rep=$OUT/${CF_CASE}_${v}_r${r}
      log "nsys $CF_CASE $v r$r ($(mins) min)"
      # shellcheck disable=SC2046
      ( cd "$OUT/work/$CF_CASE" && run timeout "$RUN_TIMEOUT" nsys profile --trace=cuda --sample=none --cpuctxsw=none \
          --force-overwrite=true -o "$rep" "$BIN/adgpu_${v}_${NUMWI}wi" $(ADGPU_EXTRA="$FW_ARGS" adgpu_args "$SEED" "$NRUN" "${rep}_run") ) \
        > "${rep}_nsys.stdout" 2>&1 || echo "FAILED: nsys $v r$r" | tee -a "$OUT/failures.txt"
      run nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true --output "${rep}_kern" "${rep}.nsys-rep" \
        > /dev/null 2>> "${rep}_stats.err" || true
      # keep the first rep's full report per variant; later reps keep only the CSV (archive size)
      [ "$r" = 1 ] || [ "$DRY_RUN" = 1 ] || rm -f "${rep}.nsys-rep" "${rep}.sqlite"
    done
  done
  run "$PY" "$ROOT/scripts/ci_speedup.py" nsys "$OUT" --case "$CF_CASE" --ref "$REF" | tee "$OUT/ci_minAD.tsv"
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
nonident)
  # Is fix #1's non-identity caused by (a) its own code change or (b) the TENSOR-path race?
  # Same seeds/cases as the gate. Each build twice (a/b) where repeatability matters.
  check_session 8
  new_outdir nonident
  for d in base_tensor:a base_tensor:b fix1_tensor:a fix1_tensor:b fix1_legacy_tensor:a \
           fix12_shfl:a fix12_shfl:b fix2only_shfl:a fix2only_shfl:b; do
    v=${d%%:*}; k=${d##*:}; need_bin "$v"; mkdir -p "$OUT/${v}_$k"
    for s in $GATE_SEEDS; do for c in $CASES; do
      if past_deadline; then echo "DEADLINE: stopped before $v $k $c s$s" | tee -a "$OUT/failures.txt" >&2; break 3; fi
      dock setup_case "$c" "$v" "$s" 1 "$OUT/${v}_$k"
    done; done
  done
  {
    cmp() { echo; echo "=== $1"; run "$PY" "$ROOT/scripts/dlg_compare.py" "$OUT/$2" "$OUT/$3" | grep -E "^SUMMARY|^DIFFERENT" || true; }
    cmp "baseline repeatability: base_tensor a vs b"            base_tensor_a base_tensor_b
    cmp "fix #1 repeatability: fix1_tensor a vs b"              fix1_tensor_a fix1_tensor_b
    cmp "gate repeat: base_tensor vs fix1_tensor"               base_tensor_a fix1_tensor_a
    cmp "control: base_tensor vs fix1 branch + INTRA_ATOMICS_LEGACY" base_tensor_a fix1_legacy_tensor_a
    cmp "race-free path: fix2only_shfl vs fix12_shfl (differ only by fix #1)" fix2only_shfl_a fix12_shfl_a
    cmp "repeatability: fix12_shfl a vs b"                      fix12_shfl_a fix12_shfl_b
    cmp "repeatability: fix2only_shfl a vs b"                   fix2only_shfl_a fix2only_shfl_b
  } | tee "$OUT/nonident_verdict.txt"
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
racecheck_li)
  check_session 5
  new_outdir racecheck_li
  setup_case 1stp "$OUT/work/1stp"
  for v in $RC_VARIANTS; do
    case $v in *ceiling*) continue ;; esac
    need_bin "${v}_lineinfo"
    log "racecheck (lineinfo) $v ($(mins) min)"
    # shellcheck disable=SC2046
    ( cd "$OUT/work/1stp" && run timeout 900 compute-sanitizer --tool racecheck --racecheck-report all \
        --show-backtrace device --print-limit 100 \
        "$BIN/adgpu_${v}_lineinfo_${NUMWI}wi" $(ADGPU_EXTRA="$SAN_ARGS" adgpu_args "$SEED" 2 "$OUT/${v}_racecheck") ) \
      > "$OUT/${v}_racecheck.log" 2>&1 || echo "NONZERO EXIT: $v racecheck" | tee -a "$OUT/failures.txt"
    {
      echo "== $v: $(grep -h "RACECHECK SUMMARY" "$OUT/${v}_racecheck.log" | tail -1 || true)"
      grep -oE "(Error|Warning): Race reported between [A-Za-z]+ access at [^ ]+ in [^ ]+:[0-9]+|and [A-Za-z]+ access at [^ ]+ in [^ ]+:[0-9]+|[A-Za-z_./-]+\.(cu|hpp|h):[0-9]+" "$OUT/${v}_racecheck.log" | sort | uniq -c | sort -rn | head -20 || true
    } | tee -a "$OUT/racecheck_lines.txt"
  done
  log "done in $(mins) min"; gpu_record end; copy_hint ;;
# ---------------------------------------------------------------------------
*) sed -n '11,32p' "$0"; exit 2 ;;
esac
