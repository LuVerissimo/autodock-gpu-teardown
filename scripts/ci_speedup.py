#!/usr/bin/env python3
# Copyright (C) 2026 Lu Verissimo. Independent performance study of AutoDock-GPU;
# not affiliated with or endorsed by Scripps Research. GPL-2.0-or-later, see LICENSE.
"""Speedup with bootstrap confidence intervals from repeated fixed-work runs.

  ci_speedup.py wall <runs_dir> --case 7cpa --ref base_tensor
      metric per run = wall seconds per million energy evaluations
      (stdout "Run time of entire job set" / .dlg "Number of energy evaluations performed")
  ci_speedup.py nsys <nsys_dir> --case 7cpa --ref base_tensor [--kernel gpu_gradient_minAD_kernel]
      metric per run = kernel milliseconds per million evaluations
      (<case>_<variant>_r<rep>_kern_cuda_gpu_kern_sum.csv and <case>_<variant>_r<rep>_run.dlg)
Speedup = median(ref) / median(variant); 95% percentile bootstrap CI resampling the reps of
each build independently (default 10000 resamples, seed 20261104)."""
import argparse, csv, glob, os, random, re, statistics

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("wall", "nsys")); ap.add_argument("dir")
ap.add_argument("--case", default="7cpa"); ap.add_argument("--ref", default="base_tensor")
ap.add_argument("--kernel", default="gpu_gradient_minAD_kernel")
ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=20261104)
a = ap.parse_args()

def evals(dlg):
    m = re.search(r"Number of energy evaluations performed:\s+(\d+)", open(dlg, errors="replace").read())
    return int(m.group(1)) if m else None

vals = {}
if a.mode == "wall":
    for out in glob.glob(os.path.join(a.dir, f"{a.case}_*.stdout")):
        m = re.match(rf"^{a.case}_(.+)_s\d+_r(\d+)$", os.path.basename(out)[:-7])
        txt = open(out, errors="replace").read()
        t = re.search(r"Run time of entire job set.*?:\s*([\d.]+)\s*sec", txt)
        ev = evals(out[:-7] + ".dlg") if os.path.exists(out[:-7] + ".dlg") else None
        if m and t and ev and "not successful" not in txt:
            vals.setdefault(m.group(1), []).append(float(t.group(1)) / ev * 1e6)
    unit = "s per Meval (wall)"
else:
    for f in glob.glob(os.path.join(a.dir, f"{a.case}_*_r*_kern_cuda_gpu_kern_sum.csv")):
        m = re.match(rf"^{a.case}_(.+)_r(\d+)_kern_cuda_gpu_kern_sum\.csv$", os.path.basename(f))
        dlg = os.path.join(a.dir, f"{a.case}_{m.group(1)}_r{m.group(2)}_run.dlg")
        if not (m and os.path.exists(dlg)):
            continue
        rows = [r for r in csv.reader(open(f)) if r]
        hi = next(i for i, r in enumerate(rows) if any("Name" in x for x in r)); h = rows[hi]
        it = next(i for i, x in enumerate(h) if x.replace(" ", "").lower().startswith("totaltime"))
        iname = next(i for i, x in enumerate(h) if "Name" in x)
        ms = sum(float(r[it]) for r in rows[hi + 1:] if len(r) > iname and r[iname].split("(")[0] == a.kernel) / 1e6
        ev = evals(dlg)
        if ms and ev:
            vals.setdefault(m.group(1), []).append(ms / ev * 1e6)
    unit = f"ms {a.kernel} per Meval"

if a.ref not in vals:
    raise SystemExit(f"no {a.ref} runs for {a.case} in {a.dir}")
rng = random.Random(a.seed); ref = vals[a.ref]
print(f"case {a.case}, metric: {unit}; speedup = median(ref)/median(variant), 95% bootstrap CI ({a.boot} resamples)")
print("variant\tn\tmedian\tmin\tmax\tspeedup\tCI95_lo\tCI95_hi")
for v in sorted(vals):
    x = vals[v]; s = statistics.median(ref) / statistics.median(x)
    bs = sorted(statistics.median(rng.choices(ref, k=len(ref))) / statistics.median(rng.choices(x, k=len(x)))
                for _ in range(a.boot))
    lo, hi = bs[int(0.025 * a.boot)], bs[int(0.975 * a.boot) - 1]
    print(f"{v}\t{len(x)}\t{statistics.median(x):.4f}\t{min(x):.4f}\t{max(x):.4f}\t{s:.3f}\t{lo:.3f}\t{hi:.3f}")
