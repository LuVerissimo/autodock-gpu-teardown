#!/usr/bin/env python3
# Copyright (C) 2026 Lu Verissimo. Independent performance study of AutoDock-GPU;
# not affiliated with or endorsed by Scripps Research. GPL-2.0-or-later, see LICENSE.
"""Attribute ncu stall samples, shared/global excess traffic to CUDA source lines and top instructions (input: ncu --page source --csv --print-source cuda,sass)."""
import csv, sys, os
from collections import defaultdict
rows=list(csv.reader(open(sys.argv[1],newline='')))
h=next(r for r in rows if r and r[0]=='Line No'); ix={}
for i,n in enumerate(h):
    ix.setdefault(n,i)
M={'stall':'Warp Stall Sampling (All Samples)','barrier':'stall_barrier','shx':'L1 Wavefronts Shared Excessive','gx':'L2 Theoretical Sectors Global Excessive','long_sb':'stall_long_sb','mio':'stall_mio'}
def v(r,k):
    try: return float(r[ix[M[k]]].replace(',','') or 0)
    except: return 0.0
ins=[]; cur=None; fpath=None
for r in rows:
    if not r: continue
    if r[0]=='File Path': fpath=os.path.basename(r[1]); continue
    if r[0] in('Function Name','Line No') or len(r)!=len(h): continue
    if r[0]: cur=(fpath,int(r[0]),r[1].strip()); continue
    if r[2].startswith('0x') and cur: ins.append(dict(addr=int(r[2],16),sass=r[3].strip(),loc=cur,**{k:v(r,k) for k in M}))
tot={k:sum(i[k] for i in ins) for k in M}
print("instructions(2 launches):",len(ins)," totals:",{k:f"{x:,.0f}" for k,x in tot.items()})
proj=('calcMergeEneGra.cu','kernel_ad.cu','kernels.cu','calcenergy.cu','auxiliary_genetic.cu')
byaddr=sorted({i['addr']:i for i in ins}.values(),key=lambda i:i['addr'])
pos={i['addr']:n for n,i in enumerate(byaddr)}
def ctx(a):
    n=pos[a]; b=next((byaddr[k]['loc'] for k in range(n,-1,-1) if byaddr[k]['loc'][0] in proj),None)
    f=next((byaddr[k]['loc'] for k in range(n,len(byaddr)) if byaddr[k]['loc'][0] in proj),None)
    return b,f
for k,lab in (('shx','shared excessive wavefronts'),('barrier','barrier stall'),('stall','all stall samples'),('gx','global excessive sectors')):
    agg=defaultdict(float)
    for i in ins: agg[i['loc'][:2]]+=i[k]
    print(f"\n=== by source line: {lab}  (total {tot[k]:,.0f})")
    for (f,l),x in sorted(agg.items(),key=lambda t:-t[1])[:10]:
        src=next(i['loc'][2] for i in ins if i['loc'][:2]==(f,l))
        print(f"  {x/tot[k]*100:5.1f}%  {f}:{l}  {src[:80]}")
    agg2=defaultdict(float); meta={}
    for i in ins: agg2[i['addr']]+=i[k]; meta[i['addr']]=i
    print(f"  top instructions with project context (before -> after):")
    for a,x in sorted(agg2.items(),key=lambda t:-t[1])[:6]:
        b,f=ctx(a); i=meta[a]
        print(f"   {x/tot[k]*100:5.1f}% {i['sass'][:34]:34s} @{i['loc'][0]}:{i['loc'][1]}  ctx {b[0]}:{b[1]} -> {f[0]}:{f[1]}")
