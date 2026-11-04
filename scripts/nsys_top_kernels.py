#!/usr/bin/env python3
# Copyright (C) 2026 Lu Verissimo. Independent performance study of AutoDock-GPU;
# not affiliated with or endorsed by Scripps Research. GPL-2.0-or-later, see LICENSE.
"""Rank kernels from an 'nsys stats --report cuda_gpu_kern_sum --format csv' export.
  nsys_top_kernels.py <csv> <K>                  table of top K by total GPU time
  nsys_top_kernels.py <csv> <K> --names          bare kernel names (no params), one per line
  nsys_top_kernels.py <csv> 0 --instances <name> launch count of one kernel"""
import csv, sys

def col(header, *keys):
    for i, h in enumerate(header):
        hn = h.lower().replace(" ", "")
        if any(k in hn for k in keys):
            return i
    raise SystemExit(f"column {keys} not found in {header}")

path, k = sys.argv[1], int(sys.argv[2])
mode = sys.argv[3] if len(sys.argv) > 3 else "--table"
with open(path, newline="") as f:
    rows = [r for r in csv.reader(f) if r]
# nsys may print notices before the header; find the header row
hi = next(i for i, r in enumerate(rows) if any("Name" in c for c in r))
header, data = rows[hi], rows[hi + 1:]
it, ip, ii, ia, iname = (col(header, "totaltime"), col(header, "time(%)"),
                         col(header, "instances"), col(header, "avg"), col(header, "name"))
recs = []
for r in data:
    try:
        recs.append((float(r[it]), float(r[ip]), int(float(r[ii])), float(r[ia]),
                     r[iname].split("(")[0].strip()))
    except (ValueError, IndexError):
        continue
recs.sort(reverse=True)
if mode == "--names":
    print("\n".join(n for *_, n in recs[:k]))
elif mode == "--instances":
    print(next((inst for _, _, inst, _, n in recs if n == sys.argv[4]), 0))
else:
    print(f"{'time_%':>7} {'total_ms':>10} {'launches':>8} {'avg_us':>10}  kernel")
    for tot, pct, inst, avg, n in recs[:k]:
        print(f"{pct:7.2f} {tot/1e6:10.2f} {inst:8d} {avg/1e3:10.1f}  {n}")
