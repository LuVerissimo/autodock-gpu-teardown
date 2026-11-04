#!/usr/bin/env python3
# Copyright (C) 2026 Lu Verissimo. Independent performance study of AutoDock-GPU;
# not affiliated with or endorsed by Scripps Research. GPL-2.0-or-later, see LICENSE.
"""Compact per-launch summary from 'ncu -i <rep> --page raw --csv'. Prints the metrics
used in the teardown; a metric absent from this ncu version prints NA. All percentages
are ncu's own percent-of-peak for the profiled GPU. Usage: ncu_summary.py <raw.csv>"""
import csv, re, sys

M = [  # (label, candidate raw metric names)
    ("duration_us",          ["gpu__time_duration.sum"]),
    ("sm_throughput_%",      ["sm__throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("dram_throughput_%",    ["dram__throughput.avg.pct_of_peak_sustained_elapsed",
                              "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("l2_throughput_%",      ["lts__throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("l1tex_throughput_%",   ["l1tex__throughput.avg.pct_of_peak_sustained_active"]),
    ("achieved_occupancy_%", ["sm__warps_active.avg.pct_of_peak_sustained_active"]),
    ("theoretical_occ_%",    ["sm__maximum_warps_per_active_cycle_pct"]),
    ("registers/thread",     ["launch__registers_per_thread"]),
    ("occ_limit_registers",  ["launch__occupancy_limit_registers"]),
    ("occ_limit_smem",       ["launch__occupancy_limit_shared_mem"]),
    ("dyn_smem/block",       ["launch__shared_mem_per_block_dynamic"]),
    ("static_smem/block",    ["launch__shared_mem_per_block_static"]),
    ("block_size",           ["launch__block_size"]),
    ("grid_size",            ["launch__grid_size"]),
    ("smem_ld_bank_confl",   ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"]),
    ("smem_st_bank_confl",   ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"]),
    ("local_ld_sectors",     ["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"]),
    ("issue_active_%",       ["sm__inst_issued.avg.pct_of_peak_sustained_active",
                              "smsp__issue_active.avg.pct_of_peak_sustained_active"]),
]
STALL = re.compile(r"smsp__average_warps?_issue_stalled_(\w+?)_per_issue_active\.ratio$")

with open(sys.argv[1], newline="") as f:
    rows = [r for r in csv.reader(f) if r]
hi = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
if hi is None:
    sys.exit(f"{sys.argv[1]}: no 'Kernel Name' header (ncu error? see the .ncu.log)")
h = rows[hi]
data = [r for r in rows[hi + 1:] if len(r) == len(h)]
units = data.pop(0) if data and not data[0][h.index("ID")].strip() else [""] * len(h)
idx = {n: i for i, n in enumerate(h)}
for r in data:
    print(f"== {r[idx['Kernel Name']]}  (ID {r[idx['ID']]}) file={sys.argv[1]}")
    for label, names in M:
        n = next((n for n in names if n in idx), None)
        v = r[idx[n]] if n else "NA"
        if n == "gpu__time_duration.sum" and units[idx[n]].strip() in ("nsecond", "ns"):
            v = f"{float(v.replace(',', ''))/1e3:.1f}"
        elif n == "gpu__time_duration.sum" and units[idx[n]].strip() in ("msecond", "ms"):
            v = f"{float(v.replace(',', ''))*1e3:.1f}"
        u = units[idx[n]].strip() if n else ""
        if n and u and n != "gpu__time_duration.sum" and u not in ("%",):
            v = f"{v} {u}"
        print(f"   {label:22s} {v}")
    st = []
    for n, i in idx.items():
        m = STALL.search(n)
        if m and m.group(1) not in ("not_issued",):
            try: st.append((float(r[i].replace(",", "")), m.group(1)))
            except ValueError: pass
    st.sort(reverse=True)
    print("   top stalls (cycles/issued inst): " + ", ".join(f"{s}={v:.2f}" for v, s in st[:6]))
