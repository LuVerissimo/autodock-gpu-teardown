#!/usr/bin/env python3
# Copyright (C) 2026 Lu Verissimo. Independent performance study of AutoDock-GPU;
# not affiliated with or endorsed by Scripps Research. GPL-2.0-or-later, see LICENSE.
"""Summarize timed docking runs named <case>_<build>_s<seed>_r<rep>.{stdout,dlg}.

Per run:   job wall time ("Run time of entire job set", stdout), best energy, top-cluster
           size (clustering histogram) and reference RMSD of the best pose (RMSD table).
Per group: median/min/max job time over reps, distinct best energies, RMSD range.
Per case:  default CUDA vs TENSOR=ON ratio, and which is faster. The faster of the two
           is the reference that any later optimization is measured against.
Usage: summarize_runs.py <runs_dir> [--ref <build>]
With --ref, also prints each build's speedup over <build> per case: wall time and, from the
.dlg counters, time per million energy evaluations (the fixed-work view when work differs)."""
import argparse, pathlib, re, statistics, sys
from collections import defaultdict

NAME = re.compile(r"^(?P<case>[^_]+)_(?P<build>.+)_s(?P<seed>\d+)_r(?P<rep>\d+)$")

def counters(path):
    if not path.exists():
        return None, None
    txt = path.read_text(errors="replace")
    ev = re.search(r"Number of energy evaluations performed:\s+(\d+)", txt)
    gn = re.search(r"Number of generations used:\s+(\d+)", txt)
    return (int(ev.group(1)) if ev else None), (int(gn.group(1)) if gn else None)

def parse_dlg(path):
    e = nclus = rmsd = None
    if not path.exists():
        return e, nclus, rmsd
    txt = path.read_text(errors="replace")
    i = txt.find("CLUSTERING HISTOGRAM")
    if i >= 0:
        for line in txt[i:].splitlines():
            f = [x.strip() for x in line.split("|")]
            if len(f) >= 5 and f[0] == "1":
                e, nclus = float(f[1]), int(f[4]); break
    j = txt.find("RMSD TABLE")
    if j >= 0:
        for line in txt[j:].splitlines():
            f = line.split()
            if len(f) >= 6 and f[0] == "1" and f[1] == "1":
                rmsd = float(f[5]); break
    return e, nclus, rmsd

ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("--ref")
args = ap.parse_args()
d = pathlib.Path(args.runs)
rows, groups = [], defaultdict(list)
work = defaultdict(list)   # (case, build) -> [(job_s, evals, gens)]
for out in sorted(d.glob("*.stdout")):
    m = NAME.match(out.stem)
    if not m:
        continue
    txt = out.read_text(errors="replace")
    if re.search(r"The job was not successful|Error in setup of Job", txt):
        rows.append((out.stem + " FAILED(job not successful)", None, None, None, None))
        continue  # exit status may be 0, but nothing was docked: exclude from timings
    t = re.search(r"Run time of entire job set.*?:\s*([\d.]+)\s*sec", txt)
    t = float(t.group(1)) if t else None
    e, k, r = parse_dlg(d / f"{out.stem}.dlg")
    ev, gn = counters(d / f"{out.stem}.dlg")
    if t is not None and ev:
        work[(m["case"], m["build"])].append((t, ev, gn))
    rows.append((out.stem, t, e, k, r))
    groups[(m["case"], m["build"])].append((t, e, r))

na = lambda v: "NA" if v is None else v
print("# per run")
print("run\tjob_s\tbest_kcal\ttop_cluster_size\tref_rmsd_A")
for n, t, e, k, r in rows:
    print(f"{n}\t{na(t)}\t{na(e)}\t{na(k)}\t{na(r)}")

med = {}
print("\n# per case and build")
print("case\tbuild\tn\tjob_s_median\tjob_s_min\tjob_s_max\tbest_kcal_distinct\tref_rmsd_min..max")
for (c, b), v in sorted(groups.items()):
    ts = [t for t, _, _ in v if t is not None]
    es = sorted({e for _, e, _ in v if e is not None})
    rs = [r for _, _, r in v if r is not None]
    rr = f"{min(rs):.2f}..{max(rs):.2f}" if rs else "NA"
    if ts:
        med[(c, b)] = statistics.median(ts)
        print(f"{c}\t{b}\t{len(ts)}\t{med[(c, b)]:.3f}\t{min(ts):.3f}\t{max(ts):.3f}\t{es}\t{rr}")
    else:
        print(f"{c}\t{b}\t0\tNA\tNA\tNA\t{es}\t{rr}")

print("\n# default vs TENSOR=ON (ratio = median default / median tensor; >1 means TENSOR=ON faster)")
print("case\tdefault_s\ttensor_s\tratio\treference_build\treference_s")
for c in sorted({c for c, _ in med}):
    a, b = med.get((c, "cuda")), med.get((c, "cuda_tensor"))
    if a is None or b is None:
        continue
    ref = "cuda" if a <= b else "cuda_tensor"
    print(f"{c}\t{a:.3f}\t{b:.3f}\t{a / b:.3f}\t{ref}\t{min(a, b):.3f}")
if any(b == "ocl" for _, b in med):
    print("\n# OpenCL vs faster CUDA build (ratio = ocl / reference; >1 means CUDA faster)")
    for c in sorted({c for c, _ in med}):
        o = med.get((c, "ocl"))
        cands = [v for k, v in med.items() if k[0] == c and k[1] in ("cuda", "cuda_tensor")]
        if o is not None and cands:
            print(f"{c}\tocl_s={o:.3f}\tref_s={min(cands):.3f}\tratio={o / min(cands):.3f}")

if args.ref:
    print(f"\n# speedup vs {args.ref} (median over reps; >1 = faster than {args.ref})")
    print("case\tbuild\tjob_s\tref_job_s\twall_speedup\tevals_median\tgens_median\ts_per_Meval\tref_s_per_Meval\tper_eval_speedup")
    for (c, b) in sorted(work):
        if b == args.ref or (c, args.ref) not in work:
            continue
        w, rw = work[(c, b)], work[(c, args.ref)]
        js, rjs = statistics.median(x[0] for x in w), statistics.median(x[0] for x in rw)
        pe = statistics.median(x[0] / x[1] * 1e6 for x in w)
        rpe = statistics.median(x[0] / x[1] * 1e6 for x in rw)
        print(f"{c}\t{b}\t{js:.3f}\t{rjs:.3f}\t{rjs / js:.3f}\t{statistics.median(x[1] for x in w):.0f}\t"
              f"{statistics.median(x[2] or 0 for x in w):.0f}\t{pe:.4f}\t{rpe:.4f}\t{rpe / pe:.3f}")
