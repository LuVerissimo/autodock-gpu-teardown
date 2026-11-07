# Corrections

Corrections to published files, newest first. The original value is kept here; the git history keeps
the original file.

## 2026-11-07: initcheck totals in `data/phase1/compute_sanitizer_summary.txt`

**What was published.** The three `initcheck` lines (builds `base_tensor`, `fix1_tensor`, `fix12_shfl`)
read:

```
ERROR SUMMARY: 2080 errors were not printed. Use --print-limit option to adjust the number of printed errors
```

**Why it was wrong.** That line counts the reports *not printed*, not the total. The runs used
`--print-limit 20`, and each raw log has two summary lines; the summary step kept only the last one
(`scripts/phase1.sh`, `sanitize` stage: `grep ... | tail -1`).

**Correct value.** **2100 `initcheck` errors per build**, for each of the three builds: 20 printed plus
2080 not printed. Source: the raw line `ERROR SUMMARY: 2100 errors` in each `*_initcheck.log` of the
Phase 1 sanitize stage (line 211 of each log), in the private Phase 1 results archive, whose sha256
`cb702f07ebb5266d86190f14eb89334b5b799a8de9615287646885dd915659ba` was verified before reading. The
data file now records 2100.

**What does not change.**
- The README's statement that `initcheck` gives identical reports in all three builds holds: 2100 in each.
  No other file in this repository stated 2080.
- `racecheck`: the recorded lines are correct as published. The raw line is
  `RACECHECK SUMMARY: 6 hazards displayed (1 error, 5 warnings)` for `base_tensor` and `fix1_tensor`, and
  `0 hazards displayed` for `fix12_shfl`: 6 hazards, of which 1 is an error and 5 are warnings, not 12.
- `memcheck` and `synccheck`: 0 for all three builds, unchanged.
- `scripts/phase1.sh` is left as it ran. Anyone rerunning it should read the total from the
  `ERROR SUMMARY: <n> errors` line, not from the last summary line.

**How it was found.** An audit of the same summary logic in a later tool (it kept the last line, and
summed both numbers of the racecheck line into 12), then checked here against the archived raw logs.
