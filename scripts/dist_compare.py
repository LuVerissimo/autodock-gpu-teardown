#!/usr/bin/env python3
# Copyright (C) 2026 Lu Verissimo. Independent performance study of AutoDock-GPU;
# not affiliated with or endorsed by Scripps Research. GPL-2.0-or-later, see LICENSE.
"""Compare docking-result distributions over seeds between a reference build and variants.

Usage: dist_compare.py <ref_build> <runs_dir> [<runs_dir> ...] [options]
Run files: <case>_<build>_s<seed>_r<rep>.dlg. Per seed (one job of nrun LGA runs): the job's best
energy (rank-1 cluster lowest energy) and that pose's reference RMSD.

Criteria per complex, variant vs reference (pre-registered 2026-11-04, owner-approved):
  PRIMARY    95% percentile-bootstrap CI of median(variant) - median(reference) lies entirely
             inside [-tol, +tol] (default tol 0.10 kcal/mol; 10000 resamples, fixed RNG seed).
  SECONDARY  two-sample KS p >= 0.05; |mean difference| <= 0.10 kcal/mol;
             |difference in #seeds with RMSD < 2 A| <= 3.
Null calibration, PER COMPLEX (amended pre-registration, 2026-11-04): <ref>_null is the reference
binary on a disjoint seed set. A complex is INTERPRETABLE only if the null passes the calibration
criteria on it; for a non-interpretable complex no equivalence claim is made (in particular 7cpa).
--calibrate-on criterion (option C, owner-selected 2026-11-04, default): interpretable = the null
  passes the PRIMARY criterion on that complex; each SECONDARY criterion is applied to a complex only
  if the null passes that criterion there (criteria the null fails are reported and not used there).
--calibrate-on primary: interpretable = null passes the primary; all secondaries always applied.
--calibrate-on all: interpretable = null passes every criterion.
Fail rule per variant, over interpretable complexes only (counted per criterion):
  any criterion failing on >= 2 complexes -> FAIL; exactly one failure -> INVESTIGATE; none -> PASS.
A variant verdict needs >= --min-interpretable (default 3) interpretable complexes, otherwise it
is NOT INTERPRETABLE."""
import argparse, math, os, random, re, statistics
from collections import defaultdict

NAME = re.compile(r"^(?P<case>[^_]+)_(?P<build>.+)_s(?P<seed>\d+)_r(?P<rep>\d+)\.dlg$")

def best(path):
    e = r = None
    txt = open(path, errors="replace").read()
    i = txt.find("CLUSTERING HISTOGRAM")
    for line in txt[i:].splitlines() if i >= 0 else []:
        f = [x.strip() for x in line.split("|")]
        if len(f) >= 5 and f[0] == "1":
            e = float(f[1]); break
    j = txt.find("RMSD TABLE")
    for line in txt[j:].splitlines() if j >= 0 else []:
        f = line.split()
        if len(f) >= 6 and f[0] == "1" and f[1] == "1":
            r = float(f[5]); break
    return e, r

def ks(a, b):
    a, b = sorted(a), sorted(b); n, m = len(a), len(b); i = j = 0; d = 0.0
    while i < n and j < m:
        x = min(a[i], b[j])
        while i < n and a[i] <= x: i += 1
        while j < m and b[j] <= x: j += 1
        d = max(d, abs(i / n - j / m))
    en = math.sqrt(n * m / (n + m)); lam = (en + 0.12 + 0.11 / en) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, min(1.0, max(0.0, p))

def mwu(a, b):
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b]); ranks = {}; i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]: j += 1
        for k in range(i, j + 1): ranks.setdefault(k, (i + j) / 2 + 1)
        i = j + 1
    r1 = sum(ranks[k] for k, (v, g) in enumerate(allv) if g == 0)
    n1, n2 = len(a), len(b); u = r1 - n1 * (n1 + 1) / 2; mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) or 1e-12
    z = (u - mu) / sd
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("dirs", nargs="+")
ap.add_argument("--tol-median", type=float, default=0.10)
ap.add_argument("--tol-mean", type=float, default=0.10)
ap.add_argument("--ks-alpha", type=float, default=0.05)
ap.add_argument("--tol-success", type=int, default=3)
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--boot-seed", type=int, default=20261104)
ap.add_argument("--min-interpretable", type=int, default=3)
ap.add_argument("--calibrate-on", choices=("criterion", "primary", "all"), default="criterion")
a = ap.parse_args()
data = defaultdict(dict)   # (case, build) -> seed -> (e, r)
for d in a.dirs:
    for f in os.listdir(d):
        m = NAME.match(f)
        if m:
            e, r = best(os.path.join(d, f))
            if e is not None:
                data[(m["case"], m["build"])][int(m["seed"])] = (e, r)
def boot_ci(ref, var, n, seed):
    rng = random.Random(seed); d = []
    for _ in range(n):
        d.append(statistics.median(rng.choices(var, k=len(var))) - statistics.median(rng.choices(ref, k=len(ref))))
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n) - 1]

print(f"PRIMARY: 95% bootstrap CI of median difference within +/-{a.tol_median} kcal/mol "
      f"({a.boot} resamples, seed {a.boot_seed})")
print(f"SECONDARY: KS p>={a.ks_alpha}; |dmean|<={a.tol_mean} kcal/mol; |dsuccess(RMSD<2A)|<={a.tol_success} seeds")
print(f"CALIBRATION: per complex on '{a.calibrate_on}' criteria; interpretable only if <ref>_null passes them")
print(f"FAIL RULE (interpretable complexes only): a criterion failing on >=2 -> FAIL; one failure -> INVESTIGATE; "
      f"verdict needs >={a.min_interpretable} interpretable complexes")
print("\ncase\tbuild\tn\tmedian\tmean\tsd\tmin\tmax\tmedRMSD\tsuccess")
for (c, b), v in sorted(data.items()):
    es = [x[0] for x in v.values()]; rs = [x[1] for x in v.values() if x[1] is not None]
    print(f"{c}\t{b}\t{len(es)}\t{statistics.median(es):.3f}\t{statistics.mean(es):.3f}\t"
          f"{(statistics.stdev(es) if len(es) > 1 else 0):.3f}\t{min(es):.2f}\t{max(es):.2f}\t"
          f"{(statistics.median(rs) if rs else float('nan')):.2f}\t{sum(r < 2 for r in rs)}")

CRIT = ("median_CI", "KS", "mean", "success")
fails = {}   # variant -> {criterion: [cases]}
print("\ncase\tvariant\tn_ref\tn_var\tdmedian\tCI95_lo\tCI95_hi\tdmean\tKS_D\tKS_p\tMWU_p\tdsuccess\tfailed")
for (c, b), v in sorted(data.items()):
    if b == a.ref or (c, a.ref) not in data:
        continue
    ref = data[(c, a.ref)]
    ea = [x[0] for x in ref.values()]; eb = [x[0] for x in v.values()]
    dmed = statistics.median(eb) - statistics.median(ea); dmean = statistics.mean(eb) - statistics.mean(ea)
    lo, hi = boot_ci(ea, eb, a.boot, a.boot_seed)
    d, p = ks(ea, eb); pu = mwu(ea, eb)
    ds = sum(x[1] < 2 for x in v.values() if x[1] is not None) - sum(x[1] < 2 for x in ref.values() if x[1] is not None)
    bad = []
    if not (lo >= -a.tol_median and hi <= a.tol_median): bad.append("median_CI")
    if p < a.ks_alpha: bad.append("KS")
    if abs(dmean) > a.tol_mean: bad.append("mean")
    if abs(ds) > a.tol_success: bad.append("success")
    for k in bad:
        fails.setdefault(b, {}).setdefault(k, []).append(c)
    fails.setdefault(b, {})
    print(f"{c}\t{b}\t{len(ea)}\t{len(eb)}\t{dmed:+.3f}\t{lo:+.3f}\t{hi:+.3f}\t{dmean:+.3f}\t{d:.3f}\t{p:.3f}\t{pu:.3f}\t{ds:+d}\t{','.join(bad) or '-'}")

nullb = f"{a.ref}_null"
percase = {}   # (variant, case) -> failed criteria
for b, fv in fails.items():
    for k, cs in fv.items():
        for c in cs:
            percase.setdefault((b, c), []).append(k)
cases = sorted({c for (c, b) in data if b == a.ref})
print()
if not fails:
    print("no variant/reference pairs found\nOVERALL: FAIL (0 comparisons)")
    raise SystemExit(1)
interp = []
unused = {}   # case -> secondary criteria not applied there (option C)
for c in cases:
    if (c, nullb) not in data:
        print(f"CALIBRATION {c}: NOT RUN -> not interpretable"); continue
    allbad = percase.get((nullb, c), [])
    bad = allbad if a.calibrate_on == "all" else [k for k in allbad if k == "median_CI"]
    if a.calibrate_on == "criterion":
        unused[c] = [k for k in allbad if k != "median_CI"]
        note = f" (secondary criteria NOT USED on {c}, null failed them: {unused[c]})" if unused[c] else ""
    else:
        note = f" (null also failed secondary: {[k for k in allbad if k not in bad]})" if set(allbad) - set(bad) else ""
    print(f"CALIBRATION {c}: {'PASS -> interpretable' if not bad else 'FAIL ' + str(bad) + ' -> NOT interpretable, no equivalence claim'}{note}")
    if not bad:
        interp.append(c)
print(f"interpretable complexes: {interp} ({len(interp)})")
for b in sorted(x for x in fails if x != nullb):
    fv = {}
    for c in interp:
        for k in percase.get((b, c), []):
            if k in unused.get(c, []):
                continue   # option C: criterion not calibrated on this complex
            fv.setdefault(k, []).append(c)
    excluded = {c: percase[(b, c)] for c in cases if c not in interp and (b, c) in percase}
    if len(interp) < a.min_interpretable:
        v = f"NOT INTERPRETABLE (only {len(interp)} interpretable complexes)"
    elif any(len(cs) >= 2 for cs in fv.values()):
        v = "FAIL"
    elif any(len(cs) == 1 for cs in fv.values()):
        v = "INVESTIGATE"
    else:
        v = "PASS"
    print(f"VERDICT {b}: {v}  failures on interpretable complexes: {fv or '-'}"
          + (f"  (not judged, non-interpretable: {excluded})" if excluded else ""))
