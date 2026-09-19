---
name: detector-pipeline
description: Implement or modify the fidnn detection pipeline in src/fidnn/detect — robust normaliser, SVDD / one-class SVM fitting, ν/γ selection, D1/D2/D3/D4 variants, score-CDF mapping, threshold calibration, windowed alarms, and the output-only and feature-space baselines — without leaking fault data (C2) or picking thresholds post hoc (C3). Use for any code that fits, selects, calibrates or scores a detector.
argument-hint: <component>
---

# Detector pipeline

Load `.claude/references/features-and-detectors.md` and the "Splits" and "Calibration" parts of
`.claude/references/evaluation-protocol.md`.

## Data-flow rule (C2, C3) — make it structural

Every fit/select/calibrate function takes **clean splits only**, by type or by explicit argument name:

| Step | Allowed input |
|---|---|
| Normaliser median/IQR, IQR-floor drop list | `clean_fit` |
| SVDD / baseline fit | `clean_fit` |
| ν/γ selection (min clean-val FPR s.t. SV-fraction band) | `clean_fit` + `clean_cal` |
| Per-tap empirical CDF (D3) | `clean_cal` |
| Thresholds `τ_α`, α ∈ {5 %, 1 %, 0.1 %} (binomial upper bound at 0.1 %) | `clean_cal` |
| FPR measurement | `clean_test` |
| TPR, AUROC, AUPR | fault splits — **scoring only** |

Prefer signatures like `fit(clean_fit: CleanFeatures)` and have fault feature containers be a distinct
type, so passing one is a type error rather than a review finding. Never select anything by test AUROC.

## Implementation notes

- SVDD = `OneClassSVM(kernel="rbf", nu, gamma)`; score `-decision_function(x)`.
- `γ_median = 1 / median‖xᵢ − xⱼ‖²` on a seeded clean subsample; grid `{"scale"} ∪ {k·γ_median}`.
- Nystroem + `SGDOneClassSVM` only if M-0's SVDD measurement says exact fitting is too slow; if used,
  measure and report the gap vs exact on M2 once.
- D3 combiners: max, mean, clean-FPR-weighted mean, one-class combiner on the score vector — all after
  CDF mapping. First-alarm tap = first tap in registry order over its own threshold.
- Windowed m-of-n for (n, m) ∈ {(1,1), (8,3), (32,8)}; per-inference always reported too.
- Feature-space baselines consume **the same normalised features as D2**. LOF uses `novelty=True`.
- **C1:** no covariance estimator, no whitening, no inverse covariance anywhere in this package. The
  edit hook will block banned names; structural equivalents (`np.cov` + `inv`) are on you.
- Persist fitted normaliser (incl. drop list), detector and thresholds under `artifacts/detectors/`
  with a sidecar (commit, config hash, seed, versions, device). Detector size in KB is an overhead
  metric — make it measurable from the serialised file.

## Tests

Leakage: passing a fault container to any fit/calibrate path fails · normaliser stats unchanged after
scoring faults · calibrated FPR on `clean_test` within 0.3×–3× nominal on M1 (the M-4 sanity gate) ·
determinism for a fixed seed.

M-4 runs on M1 and produces **no thesis number** — say so in any output it generates.
