# AutoDock-GPU on an NVIDIA A100: a performance teardown

*Independent study by Lu Verissimo, November 2026. Not affiliated with, reviewed by, or endorsed by Scripps
Research, the Forli lab, or the AutoDock-GPU developers. AutoDock is a trademark of the Scripps Research
Institute. All numbers below were measured in this study unless stated otherwise; the files behind each one are
listed in [Reproduce](#reproduce).*

## Summary

- **Headline.** On the shipped 7cpa complex (17 rotatable bonds), with a fixed amount of docking work, two kernel
  changes make AutoDock-GPU **1.139x faster in wall time per energy evaluation** (10 interleaved repetitions, 95%
  bootstrap CI 1.138-1.140). The comparison is against the faster stock build (`TENSOR=ON`) on one
  A100-SXM4-40GB.
- **End to end.** At default settings the same changes make a 7cpa docking job **1.125x faster** (5 repetitions,
  one seed). The search takes a different path in each build, so this is the realistic view, not a controlled
  comparison. **Small ligands slow down:** 1stp is 0.953x end to end.
- **Scope of the intervals.** The confidence intervals cover run-to-run variation on one instance. They do not
  include instance-to-instance variation.
- **Dominant kernel.** `gpu_gradient_minAD_kernel` (the ADADELTA local search) is **1.151x** faster per evaluation
  (CI 1.149-1.153). The study had set 1.15x on this kernel as the threshold for continuing to optimize. The interval
  contains 1.15, so this was a **toss-up**, and optimization stopped there.
- **Targets missed.** The goals were at least 1.3x end-to-end or a 20% cut in kernel time. The result is 1.125x
  end-to-end and a 13.1% cut in kernel time.
- **Correctness.** On three of the four shipped complexes, docking results showed **no difference detected at this
  sensitivity**. The fourth (7cpa) could not be judged at the sample size used.
- **Byproducts.**
  - A racecheck-reported shared-memory hazard in the stock tensor-core path.
  - Measured repeatability limits of AutoDock-GPU on GPU.
  - On this hardware, OpenCL is faster than the default CUDA build.

## Setup

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, GPU 0 only, ECC on, 400 W limit |
| Software | driver 580.105.08; CUDA 12.8 (nvcc 12.8.93); gcc 11.4.0; Nsight Systems 2026.3.2; Nsight Compute 2026.3.1; compute-sanitizer 2025.1.0 |
| Code | AutoDock-GPU `develop` at `be06a135` (v1.6), built with `make DEVICE=CUDA NUMWI=128 TARGETS=80`, plus `TENSOR=ON` |
| Inputs | the shipped complexes 1stp, 3ce3, 3tmn, 7cpa, and the 42-complex set `diogomart/AD-GPU_set_of_42` at `e7b7ead` |
| Runs | 100 LGA runs per docking job, one job at a time, three sessions on three separate cloud instances |

Ratios are only compared within a session.

## Where the time goes

- **The faster stock build is `TENSOR=ON`.** It replaces a block reduction with a tensor-core reduction (upstream
  since 2024, based on Schieffer's KTH thesis). It beats the default CUDA build on all four shipped complexes
  (1.10-1.40x end-to-end) and on 40 of the 42-set (median 1.146x). Every speedup below is against `TENSOR=ON`.
- **One kernel dominates.** `gpu_gradient_minAD_kernel` takes 96-99.5% of GPU kernel time on every complex and
  build.
- **Profile of that kernel (7cpa, Nsight Compute).**
  - SM throughput 45%. L1/shared memory 82%, the busiest unit. DRAM 0.03%.
  - Occupancy 48%, out of 50% possible (limited by 64 registers per thread).
  - Top stall: threads waiting at barriers.
  - The kernel is limited by synchronization and shared memory, not memory bandwidth.
- **Source-level attribution.**
  - About 78% of the excess shared-memory traffic comes from one place: per-pair atomic updates of
    `gradient[atom1_id]` in the intramolecular pair loop (`calcMergeEneGra.cu:588-590`). The pair list is ordered
    by first atom, so up to 32 threads of a warp hit the same address.
  - Stall time splits roughly into the pair loop 37%, torque gradients 18%, the rotations loop 16.5%, and the
    tensor reduction 16%.

## What changed, and what it bought

- **Fix #1** sums each run of equal first-atom IDs inside the warp (a segmented shuffle scan) and issues one atomic
  per run instead of one per pair.
- **Fix #2** replaces the tensor-core reduction with a warp-shuffle block reduction: one barrier, fixed summation
  order.
- **Ceiling** is a deliberately wrong build that drops the atomics in the pair loop. It is used only to time the
  upper bound.

| 7cpa, fixed work, vs `TENSOR=ON` | Wall time per evaluation (95% CI) | minAD time per evaluation (95% CI) |
|---|---|---|
| Fix #1 | 1.059x (1.058-1.060) | 1.062x (1.061-1.064) |
| Fix #2 alone | 1.050x (1.048-1.051) | 1.053x (1.051-1.055) |
| **Fix #1 + #2** | **1.139x (1.138-1.140)** | **1.151x (1.149-1.153)** |
| Ceiling (wrong output) | 1.153x (1.152-1.154) | 1.189x (1.187-1.192) |

Fixed work = `--autostop 0 --heuristics 0 --nev 3000000`. 10 interleaved repetitions per variant; speedup =
ratio of medians, with a percentile bootstrap CI.

**A method lesson: time the ceiling.** The profile pointed at the pair-loop atomics as the dominant waste: about
78% of excess shared-memory traffic. Alongside the fixes, we built a ceiling variant that removes those atomics
entirely and gives wrong results. It was only 1.153x faster in wall time, and 1.189x on the kernel. The biggest
line in the profile was worth roughly 19% of kernel time, and the correct fixes recovered most of it (1.139x of
1.153x wall).

A dominant counter is not the same as a dominant cost. Because the ceiling was built at the same time as the fixes,
not before them, its answer arrived only once the engineering was done. Timing a deliberately wrong upper bound
first, before investing in a fix, would have set expectations earlier, at the cost of about one build and one run.

**It depends on the ligand.**
- On the smaller shipped ligands (fixed work), fix #1 alone is 0.93-1.00x, i.e. no faster or slower. Fix #2 alone
  is 1.10-1.18x, and both together 1.09-1.12x.
- Across the 42-set, fix #1 alone is neutral per evaluation (geometric mean 1.001). It is faster on large ligands
  (3er5 1.19x, 2vaa 1.18x).

**End to end at default settings** (5 repetitions, one seed): fix #1 + #2 gives 7cpa 1.125x, 3ce3 1.147x,
3tmn 1.035x and 1stp 0.953x. Short dockings are dominated by a fixed setup cost of about 0.35 s per job.

**Kernel profile after both fixes** (7cpa, one launch, clocks locked; the search trajectories differ):
76.5 -> 64.7 ms per launch, SM throughput 45% -> 68%, shared-memory store conflicts 271M -> 22M.

## Correctness

- **Statistical equivalence (pre-registered before the data).**
  - Per build and complex: the best binding energy per seed, over 48 seeds (24 for 7cpa).
  - Primary criterion: the 95% bootstrap CI of the median difference from `TENSOR=ON` lies within ±0.10 kcal/mol.
  - Secondary criteria: KS test, mean difference, and the number of seeds whose best pose is within 2 Å of the
    crystal pose.
  - Calibration: each criterion is checked first on `TENSOR=ON` against itself on disjoint seeds, per complex.
  - **Result:** on 1stp, 3ce3 and 3tmn, **no difference detected at this sensitivity** for fix #1, fix #2 alone,
    or both.
  - **7cpa could not be judged.** `TENSOR=ON` against itself already spans ±0.385 kcal/mol at 24 seeds, so no
    equivalence statement is made for 7cpa. Descriptively: median best energy -13.69 vs -13.72 kcal/mol;
    best pose within 2 Å in 17/24 vs 19/24 seeds.
- **Byte identity.** Fix #1 was designed to leave results unchanged, since the gradients it touches are integers.
  It does not: 0/12 fixed-seed jobs matched the baseline byte for byte.
  - It changes results even between two race-free builds that differ only by fix #1. Ten of those pairs were jobs
    where each build repeats itself, and all ten differed.
  - The mechanism is not established. A likely but untested explanation is that the restructured loop changes how
    the compiler orders the floating-point energy sums.
- **Repeatability.** With a fixed seed, `TENSOR=ON` gave best energy/RMSD identical in every repeated job; files
  byte-identical 10/12. Fix #1 + #2 likewise had files byte-identical 10/12. The default CUDA build's best energy
  varied on 21 of 42 complexes, and OpenCL's on 4 of 4.
- **compute-sanitizer** (short workload):
  - memcheck and synccheck: 0 errors on the baseline, fix #1, and fix #1 + #2.
  - initcheck: identical reports in all three builds. They are not introduced by the changes.
  - racecheck: a **racecheck-reported shared-memory hazard**. The code shows no `__syncwarp()` between the writes
    (`kernels.cu:165`) and the reads (`tcec.hpp:139`, in the bundled wmma_extension). No effect on results was
    observed. The line mapping covers the baseline only. Fix #2 avoids the hazard by replacing that reduction
    (0 hazards reported for the fix #1 + #2 build; fix #2 alone was not racechecked).

## A note on determinism policy

AutoDock-GPU on a GPU is not byte-reproducible at a fixed seed in any build measured here. A likely but untested
explanation: floating-point sums assembled with atomics can arrive in a different order from run to run, and the
genetic-algorithm search amplifies last-bit differences into different trajectories.

This study therefore did not use byte identity as its correctness gate for performance changes. It used
equivalence of result distributions over many seeds, calibrated against the baseline's own variation, with
thresholds and a fail rule fixed before the data came in.

For users: compare docking outcomes across seeds rather than between single runs. Where repeatability matters,
`TENSOR=ON` was among the most stable builds here (identical best energy and RMSD), though even its files were not
always byte-identical.

## Other observations

- **OpenCL vs CUDA.** On the A100, the OpenCL build was 10-23% faster than the default CUDA build. Against
  `TENSOR=ON` it was within about 2% on 3ce3, 3tmn and 7cpa; on 1stp, OpenCL was 6% slower than `TENSOR=ON`. This
  is consistent in direction with upstream issue #239.
- **Measuring docking speed.** At default settings, end-to-end time mixes kernel speed with search convergence.
  On 2xy9, `TENSOR=ON` ran 1.42x longer than the default build but found a better pose. Speed claims here use
  fixed work.
- **Fixed overhead.** CUDA setup is a median of 0.345 s per job (379 runs). 35 of the 42 complexes finish in under
  3 s, which limits end-to-end gains on short dockings.
- **Shipped example.** The shipped 1ac8 example was not used: the current version rejects its ligand file at input
  parsing.

## Limitations

- **Scope.** One GPU model; three sessions on different instances; `NUMWI=128` only; one seed for timing runs.
- **Equivalence coverage.** Equivalence was established on 3 of 4 shipped complexes, and not on 7cpa, the headline
  complex.
- **Race location.** The racecheck line mapping covers the baseline only.
- **Not evaluated.** No AMD/HIP comparison. An open upstream pull request (#320) ports the CUDA path to HIP; it was
  not evaluated.

## Reproduce

**Code changes.** [`patches/`](patches/) holds plain diffs against upstream AutoDock-GPU `develop` at `be06a135`.
Fix #1 and the ceiling diff were each checked with `git apply --check` on a clean checkout. Fix #2 was checked on
top of fix #1, which it requires. Applying fix #1 then fix #2 reproduces the measured fix #1 + #2 sources exactly.

```bash
git clone https://github.com/forlilab/AutoDock-GPU.git && cd AutoDock-GPU && git checkout be06a135
git submodule update --init wmma_extension                       # needed for TENSOR=ON
git apply ../autodock-gpu-teardown/patches/fix1-intra-segmented-atomics.diff                     # fix #1
make DEVICE=CUDA NUMWI=128 TARGETS=80 TENSOR=ON                  # fix #1 build (as measured)
cp bin/autodock_gpu_128wi ../autodock_gpu_128wi_fix1             # keep the fix #1 binary
rm -f kernels.o bin/autodock_gpu_128wi bin/adgpu_analysis        # clean: upstream's Makefile has no clean recipe
git apply ../autodock-gpu-teardown/patches/fix2-shfl-block-reduction.diff                        # + fix #2
make DEVICE=CUDA NUMWI=128 TARGETS=80 SHFLREDUCE=ON              # fix #1 + #2 build (as measured)
```

- The baseline is the unpatched checkout built with `TENSOR=ON`.
- Each build writes `bin/autodock_gpu_128wi`; copy it out before building the next variant, and clean between
  variants as above.
- Fix #2 alone is the fix #1 + #2 tree built with `make ... SHFLREDUCE=ON "NVCC=nvcc -DINTRA_ATOMICS_LEGACY"`.
- The ceiling diff applies to a clean checkout and produces wrong docking results by design.

**Measurement.** [`scripts/`](scripts/) holds the session driver (`phase1.sh`) and the analysis tools (summaries,
bootstrap CIs, byte comparison, seed-distribution equivalence, ncu attribution). `phase1.sh` builds variants from
named git refs set in its `VARIANTS` table; point them at your patched checkouts.

**Data.** [`data/`](data/) holds the tables and text exports behind every number above, regenerated from the raw
outputs with these scripts (index in [`data/README.md`](data/README.md)). The raw docking logs (1,739 `.dlg` files,
directory paths reduced to file names) are attached to the release as `autodock-gpu-teardown-raw-dlg.tar.xz`, with
SHA-256 `28b3a2ff672cd0a98717e7c1b892e5b410719c1608755376cc4adaab8140c24c`. Profiler reports are not published.

## License

- Report text and data (`README.md`, `data/`, the release tarball): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- `scripts/`: GPL-2.0-or-later.
- `patches/`: modify AutoDock-GPU files and carry those files' licenses (GPL-2.0-or-later / LGPL-2.1-or-later).
- See [NOTICE.md](NOTICE.md).
