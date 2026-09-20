# fidnn.detect

Normalisation, SVDD detectors, calibration and baselines. SPEC §5.3, §6, §8. The
`/detector-pipeline` skill has the procedure.

| File | Role |
|---|---|
| `features.py` | `CleanFeatures` / `FaultFeatures` containers, Block E fusion, long parquet → (N, K, 29) |
| `normalise.py` | `(x − median)/(IQR + δ)` from `clean_fit`, with the persisted drop list |
| `svdd.py` | `OneClassSVM(kernel="rbf")`, γ_median, and the clean-only ν/γ selection |
| `variants.py` | D1 (per tap), D2 (early fusion), D3 (late fusion + first-alarm tap) |
| `calibrate.py` | τ_α from `clean_cal`, Clopper-Pearson bound, m-of-n windows |
| `baselines.py` | Output-only scores and the feature-space baselines (§6.4) |
| `run.py` | `fidnn fit`, and the `docs/M4_sanity.md` record |

## C2 and C3 are structural, not conventions

`CleanFeatures` and `FaultFeatures` are distinct types, and every fit, selection and calibration
path takes the clean one. Passing fault features raises `TypeError` — a test, not a review finding.
Nothing that touches a fault score feeds back into a fit, a hyperparameter or a threshold:

| Step | Input |
|---|---|
| Normaliser median/IQR and drop list | `clean_fit` |
| SVDD fit, baseline fit | `clean_fit` |
| ν/γ selection (min clean-cal FPR subject to the SV band) | `clean_fit` + `clean_cal` |
| D3 per-tap CDFs, tap weights, per-tap thresholds | `clean_cal` |
| τ_α, α ∈ {5 %, 1 %, 0.1 %} | `clean_cal` |
| FPR measurement | `clean_test` |
| TPR, AUROC, AUPR | fault splits — scoring only |

## Decisions

- **Selection rule.** SPEC §6.3 says "minimise clean-validation FPR subject to a support-vector
  stability band". Implemented as: fit on `clean_fit`, set τ at the (1−α) quantile of the
  `clean_fit` scores, measure FPR on `clean_cal`, keep pairs whose SV fraction lies in
  `SV_BAND × ν`, and take the lowest FPR (ties broken by |SV fraction − ν|). If no pair holds the
  band, the closest one is used — visible in the sidecar rather than silently.
- **Score direction** is uniform: higher means more anomalous, for SVDD and for every baseline, so
  thresholds and metrics read the same way everywhere.
- **Block E** compares each tap with the previous tap in registry order; the first tap gets neutral
  values (ratio 1, difference 0) rather than being dropped.
- **PCA baseline** is reconstruction error, with the scaling flag off. Turning it on and then
  measuring Euclidean distance would reintroduce exactly the metric C1 excludes.
- **D4 (Deep SVDD)** is not built: it is a C0 addition, wanted only if D2/D3 leave headroom (§6.2).

## Invariants

- No covariance estimator, no inverse covariance, no whitening anywhere in this package (C1). The
  grep guard in `tests/test_c1_guard.py` is the enforcement; the edit hook blocks the names.
- M-4 runs on M1 and produces **no thesis number**; the generated record says so on its face.
- The fitted bundle is persisted with its size in KB, because detector size is an overhead metric
  reported next to detection quality (C5).
