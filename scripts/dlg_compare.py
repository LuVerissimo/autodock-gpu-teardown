#!/usr/bin/env python3
# Copyright (C) 2026 Lu Verissimo. Independent performance study of AutoDock-GPU;
# not affiliated with or endorsed by Scripps Research. GPL-2.0-or-later, see LICENSE.
"""Bit-identity gate for AutoDock-GPU .dlg files.

Usage: dlg_compare.py <dir_a> <dir_b> [<name_glob>]
Pairs files by name after removing the build tag (the name part between case and _s<seed>),
e.g. 7cpa_base_tensor_s42_r1.dlg <-> 7cpa_fix1_tensor_s42_r1.dlg.

Ignored lines (and only these): the command-line echo, wall-clock timings and the version
string. File paths are compared by basename. Everything else (poses, energies, RMSD tables,
clusters, evaluation and generation counts) must match byte for byte.
Exit status 0 if all pairs are IDENTICAL, 1 otherwise."""
import fnmatch, os, re, sys

IGNORE = re.compile(r"^(Program call in command line was:|Time taken for this run:|"
                    r"Run time of entire job set|Idle time|AutoDock-GPU version|Run time \S+ sec)")
PATHLINE = re.compile(r"^(Ligand file:|Grid fld file:|Flexres file:|Reference file:)\s+(\S+)")
NAME = re.compile(r"^(?P<case>[^_]+)_(?P<build>.+)_(?P<tail>s\d+_r\d+\.dlg)$")

def norm(path):
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if IGNORE.search(line.strip()):
                continue
            m = PATHLINE.match(line)
            if m:
                line = f"{m.group(1)} {os.path.basename(m.group(2))}"
            out.append(line)
    return out

def index(d, glob):
    idx = {}
    for f in os.listdir(d):
        m = NAME.match(f)
        if m and fnmatch.fnmatch(f, glob):
            idx[(m["case"], m["tail"])] = os.path.join(d, f)
    return idx

a_dir, b_dir = sys.argv[1], sys.argv[2]
glob = sys.argv[3] if len(sys.argv) > 3 else "*.dlg"
A, B = index(a_dir, glob), index(b_dir, glob)
keys = sorted(set(A) & set(B))
if not keys:
    sys.exit(f"no matching .dlg pairs between {a_dir} and {b_dir}")
bad = 0
for k in keys:
    la, lb = norm(A[k]), norm(B[k])
    if la == lb:
        print(f"IDENTICAL  {os.path.basename(A[k])} == {os.path.basename(B[k])}  ({len(la)} lines compared)")
        continue
    bad += 1
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    print(f"DIFFERENT  {os.path.basename(A[k])} vs {os.path.basename(B[k])}: "
          f"{len(diffs)} differing lines, line counts {len(la)}/{len(lb)}")
    for i, x, y in diffs[:5]:
        print(f"    line {i}:\n      a: {x[:110]}\n      b: {y[:110]}")
missing = sorted(set(A) ^ set(B))
for k in missing:
    print(f"UNPAIRED   {k}")
print(f"\nSUMMARY: {len(keys) - bad}/{len(keys)} identical, {bad} different, {len(missing)} unpaired")
sys.exit(0 if bad == 0 and not missing else 1)
