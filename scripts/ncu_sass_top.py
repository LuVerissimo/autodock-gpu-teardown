#!/usr/bin/env python3
# Copyright (C) 2026 Lu Verissimo. Independent performance study of AutoDock-GPU;
# not affiliated with or endorsed by Scripps Research. GPL-2.0-or-later, see LICENSE.
"""Aggregate ncu SASS-level source export by opcode and top instruction."""
import csv, sys, re
from collections import defaultdict
rows=list(csv.reader(open(sys.argv[1], newline='')))
h=rows[1]; data=[r for r in rows[2:] if len(r)==len(h)]
ix={n:i for i,n in enumerate(h)}
def f(r,n):
    try: return float(r[ix[n]].replace(',','') or 0)
    except: return 0.0
tot=lambda n: sum(f(r,n) for r in data)
S='Warp Stall Sampling (All Samples)'; B='stall_barrier'; X='L1 Wavefronts Shared Excessive'; W='L1 Wavefronts Shared'
T={k:tot(k) for k in (S,B,X,W,'stall_long_sb','stall_mio','stall_wait','L2 Theoretical Sectors Global Excessive')}
print("instructions:",len(data)); [print(f"  total {k}: {v:,.0f}") for k,v in T.items()]
op=lambda r: re.sub(r'^@!?U?P\w+\s+','',r[ix['Source']].strip()).split(' ')[0]
for key,lab in ((S,'all stall samples'),(B,'barrier stall'),(X,'shared excessive wavefronts'),('L2 Theoretical Sectors Global Excessive','global excessive sectors')):
    agg=defaultdict(float)
    for r in data: agg[op(r).split('.')[0]]+=f(r,key)
    t=sum(agg.values()) or 1
    print(f"\n== by opcode: {lab}"); [print(f"  {k:10s} {v/t*100:5.1f}%") for k,v in sorted(agg.items(),key=lambda x:-x[1])[:6]]
    top=sorted(data,key=lambda r:-f(r,key))[:8]
    print(f"   top instructions ({lab}):")
    for r in top: print(f"    {r[ix['Address']]:>6} {f(r,key)/t*100:5.1f}%  {r[ix['Source']].strip()[:70]}")
