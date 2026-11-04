#!/usr/bin/env python3
# Copyright (C) 2026 Lu Verissimo. Independent performance study of AutoDock-GPU;
# not affiliated with or endorsed by Scripps Research. GPL-2.0-or-later, see LICENSE.
"""Split ncu stall samples of gpu_gradient_minAD_kernel by code region of calcMergeEneGra.cu (one entry per SASS address)."""
import csv, sys, os
from collections import defaultdict
rows=list(csv.reader(open(sys.argv[1],newline='')))
h=next(r for r in rows if r and r[0]=='Line No'); ix={}
for i,n in enumerate(h): ix.setdefault(n,i)
S=ix['Warp Stall Sampling (All Samples)']; B=ix['stall_barrier']
# one entry per address (inlined instructions can be listed under several lines; keep the
# deepest PROJECT line seen for that address, else the header)
proj=('calcMergeEneGra.cu','kernel_ad.cu','kernels.cu','calcenergy.cu','auxiliary_genetic.cu')
seen={}; cur=None; fp=None
for r in rows:
    if not r: continue
    if r[0]=='File Path': fp=os.path.basename(r[1]); continue
    if r[0] in('Function Name','Line No') or len(r)!=len(h): continue
    if r[0]: cur=(fp,int(r[0])); continue
    if r[2].startswith('0x'):
        a=r[2]; s=float(r[S] or 0); b=float(r[B] or 0)
        if a not in seen or (cur[0] in proj and seen[a][0][0] not in proj): seen[a]=(cur,s,b)
        # metrics per address are identical across duplicate listings; launches are separate addresses? no: same addr both launches
# NOTE: both launches share addresses; keep max-merge per address within a launch is not separable here.
def region(f,l):
    if f=='calcMergeEneGra.cu':
        if l<128: return 'setup/init (CMEG <128)'
        if l<192: return 'rotations (CMEG 128-191)'
        if l<416: return 'intermolecular / grid interp (CMEG 192-415)'
        if l<599: return 'intramolecular pair loop (CMEG 416-598)'
        if l<720: return 'reductions (CMEG 599-719)'
        return 'torque/rotbond gradients (CMEG >=720)'
    if f in('kernels.cu','tcec.hpp','common.hpp','functions.hpp','sm_80_tf32.hpp','mma.hpp','wmma_extension.hpp'): return 'tensor reduction (kernels.cu/tcec)'
    if f=='kernel_ad.cu': return 'ADADELTA loop (kernel_ad.cu)'
    return 'other: '+f
agg=defaultdict(lambda:[0.0,0.0])
for (loc,s,b) in seen.values():
    k=region(*loc); agg[k][0]+=s; agg[k][1]+=b
ts=sum(v[0] for v in agg.values()); tb=sum(v[1] for v in agg.values())
print(f"unique addresses: {len(seen)}  stall samples {ts:,.0f}  barrier {tb:,.0f}")
for k,(s,b) in sorted(agg.items(),key=lambda t:-t[1][0]):
    print(f"  {s/ts*100:5.1f}% of stalls  {b/tb*100:5.1f}% of barrier   {k}")
