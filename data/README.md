# Data index

All files were regenerated from the raw session outputs with the scripts in `../scripts/`. Host-specific folder names
were removed. Timings are medians of repeated jobs on GPU 0 of an A100-SXM4-40GB; ratios are only compared within a
session.

| File | Content |
|---|---|
| `phase0/baseline_cuda_vs_tensor.tsv` | default CUDA vs TENSOR=ON, 4 shipped complexes, 5 reps (headline baseline run) |
| `phase0/opencl_comparison.tsv` | default CUDA vs TENSOR=ON vs OpenCL, 5 reps |
| `phase0/set42_cuda_vs_tensor.tsv` | 42-complex set, default CUDA vs TENSOR=ON, 2-3 reps |
| `phase0/nsys_kernel_tables.txt` | kernel time breakdown (Nsight Systems) |
| `phase0/ncu_summary.txt` | per-launch Nsight Compute metrics, minAD and gen_and_eval kernels |
| `phase0/minAD_source_attribution.txt` | stalls and excess memory traffic by source line and code region |
| `phase1/end_to_end_default_settings.tsv` | all variants, default settings, 5 reps |
| `phase1/fixed_work_3reps.tsv` | all variants, fixed work, 3 reps (preliminary) |
| `phase1/set42_base_vs_fix1.tsv` | 42-complex set, TENSOR=ON vs fix #1, 1 rep |
| `phase1/seed_distributions_equivalence.txt` | pre-registered equivalence test over seeds (calibration and verdicts) |
| `phase1/gate_byte_identity.txt` | byte comparison of .dlg outputs, fix #1 vs baseline, and repeatability |
| `phase1/compute_sanitizer_summary.txt` | memcheck/racecheck/initcheck/synccheck summaries (initcheck totals corrected 2026-11-07, see `../CORRECTIONS.md`) |
| `confirm/fixed_work_7cpa_wall_ci.tsv` | headline: 10 reps, wall time per evaluation, bootstrap CI |
| `confirm/fixed_work_7cpa_minAD_ci.tsv` | 10 reps, minAD kernel time per evaluation, bootstrap CI |
| `confirm/nonidentity_diagnosis.txt` | where fix #1's non-identity comes from |
| `confirm/racecheck_baseline_lines.txt` | racecheck hazard locations, baseline build |
