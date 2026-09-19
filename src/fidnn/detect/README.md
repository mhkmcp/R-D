# fidnn.detect

**Status: empty.** Needed for M-4 (sanity check on M1) and M-5. SPEC §5.3, §6, §8. The
`/detector-pipeline` skill has the procedure.

## To build

| Piece | Contract |
|---|---|
| Normaliser | `(x − median)/(IQR + δ)` from `clean_fit`. Drop features under an IQR floor and persist the drop list |
| Block E | 3 cross-tap features added at fusion (L2 ratio, entropy ratio, saturation difference vs previous tap) → 29/tap |
| SVDD | `OneClassSVM(kernel="rbf")`, score = `-decision_function`. Use Nystroem + `SGDOneClassSVM` only if M-0 shows exact fitting is too slow |
| ν/γ selection | ν ∈ {0.001 … 0.1}, γ ∈ {scale} ∪ {k·γ_median}. Pick the minimum clean-val FPR within the SV-fraction band |
| D1 / D2 / D3 / D4 | Per-tap / early fusion / late fusion via clean-cal CDFs (max, mean, weighted, one-class) / Deep SVDD (extension) |
| Calibration | τ_α = (1−α) quantile of `clean_cal`, α ∈ {5, 1, 0.1} %, binomial upper bound at 0.1 %. Windowed m-of-n (1,1), (8,3), (32,8) |
| Baselines | Output-only (MSP, entropy, margin, energy). IF, LOF(`novelty=True`), KDE, PCA recon, AE recon on D2 features |
| Persistence | Normaliser + detector + thresholds → `artifacts/detectors/` with sidecar |

## Invariants

- **C2:** every fit/select/calibrate function accepts clean splits only. Make fault features a
  distinct type so passing them is a type error.
- **C3:** no threshold or hyperparameter is chosen from fault or test data.
- **C1:** no covariance, inverse covariance, whitening or `EllipticEnvelope`-style estimator.
- First-alarm tap (H4) = first tap in registry order whose D3 score exceeds its own threshold.
