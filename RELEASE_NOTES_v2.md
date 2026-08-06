# Release notes — v2, 2026-07

## Summary

Version 2 revises the analysis code of the accompanying paper (*Computers & Geosciences*,
CAGEO-D-26-00556) and supersedes version 1. The two substantive changes are:

- the importance weights are computed as joint density ratios `w = f(m)/g(m)` evaluated
  at the sampled points (v1 evaluated the densities at the kernel centers and averaged
  per-dimension ratios; results derived from the v1 weight cell should not be used);
- the estimator now pools the naive and proposal batches with a defensive mixture
  g' = alpha*f + (1-alpha)*g (alpha = prior-sample share), which bounds the weights by
  1/alpha and stabilizes the effective sample size.

All results of the revised paper are produced with this version.

## What v2 contains

**Canonical workflow**
- `notebooks/Final_Script_after_IS_Simulation_CORRECTED.ipynb` — end-to-end weighting and
  certification analysis.
- `notebooks/revision_tools.py` — shared implementation: `is_weights_correct`,
  `defensive_mixture_weights`, `variance_report` (Var[h] vs Var[p_hat]), `n_required`
  (Chebyshev / Wilson / CLT), data loaders.
- `notebooks/check_corrected_weights.py` — 5-second sanity check of the weight
  computation and the mixture estimator.

**Replication & stress-testing**
- `notebooks/seed_study.py` — 25 independent replicates × batch sizes L ∈ {100, 300, 500}
  against the 9,993-run reference pool; outputs `seed_study_results.csv`.
- `notebooks/stress_tests.py` — t1–t9: certificate coverage, bound comparisons, the
  reference-free weight-sanity identity E_f[g'/f] = 1, split-pool evaluation, SNIS bias,
  pilot feasibility, bandwidth/alpha sweeps, PC-selection stability, and the CI-first
  certification analysis.
- `notebooks/task8_rare_regime.py` — re-threshold experiment at p = 1.0×10⁻³ (feasibility
  follows the binomial design rule; certified interval covered the reference in 16/16
  replayed replicates; the basis for scoping the method to p ~ 10⁻²).
- `notebooks/synth_rerun_final.py` — the Section-3 synthetic example re-executed with
  fixed seeds.

**Headline numbers (v2)**
- Certification guarantee: never violated in 25 replicates (67 replicate-configurations)
  and 6 fresh out-of-sample MRST validation batches (299/300 fresh runs completed).
- Typical certification-cost advantage at the studied margin: ≈1.6× median (≈1.5–2×
  band; up to ≈3× for favourable single batches, quoted only as anchors).
- Combined estimate for the anchor configuration: p_hat = 0.032 (SNIS, ESS 365/600),
  reference pool value 0.0271.

**Retained from v1**
- `notebooks/Final_Script_after_IS_Simulation.ipynb` (v1) is retained for archival
  completeness with a note in its first cell; its weight cell is superseded.
- `figures/Fig_5.png` (v1 running-mean comparison) is superseded by
  `figures/Fig4_corrected.png`.

## Provenance

All v2 numbers were reproduced independently in a clean environment before this release
(bit-identical reconstruction of all 67 seed-study replicates; digit-identical
reproduction of the sanity-check outputs).
