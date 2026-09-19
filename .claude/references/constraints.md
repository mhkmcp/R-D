# Hard constraints (SPEC §0)

| # | Constraint | Violated when |
|---|---|---|
| **C0** | `new_thesis_contract.pdf` governs. Where SPEC and contract conflict, the contract wins and SPEC is amended. Anything the contract does not require is an *addition* and is marked as one. | A contract commitment is cut before an addition; SPEC text is used to override the contract |
| **C1** | **Mahalanobis distance is excluded from the entire project** — not a detector, baseline, feature-space metric or whitening step. | Any construct in the banned list below appears in code, configs, experiments, or the thesis *as a method* |
| **C2** | Detectors are trained on **clean executions only**. No fault-injected sample touches detector fitting or threshold selection. | A fault record (any split) reaches a normaliser fit, SVDD fit, ν/γ selection, CDF mapping or threshold |
| **C3** | Thresholds are calibrated on `clean_cal` at a target FPR fixed **a priori** (α ∈ {5 %, 1 %, 0.1 %}). | Threshold chosen to maximise F1/TPR, or chosen on any test split |
| **C4** | Every reported number carries a **seed count and dispersion estimate** (≥ 5 seeds, mean ± 95 % CI, §9.5). | A single-run number, or a number without n and CI |
| **C5** | Detection quality and **runtime/memory overhead** are reported together, never separately. | A detection table with no overhead columns (or companion table), or vice versa |

## C1 — banned constructs

Must not appear in code, configs, notebooks, or thesis methods:

- `scipy.spatial.distance.mahalanobis`
- `sklearn.covariance.EllipticEnvelope`, `MinCovDet`, `EmpiricalCovariance(...).mahalanobis`
- `LedoitWolf`, `ShrunkCovariance`, `GraphicalLasso`, or any shrinkage/covariance estimator feeding a
  distance
- Any explicit `(x-μ)ᵀ Σ⁻¹ (x-μ)` — including Cholesky-solve and pseudo-inverse variants
  (`np.linalg.inv(np.cov(...))`, `torch.linalg.pinv(cov)`, `cholesky` + `solve_triangular` on a
  feature covariance)
- Lee et al. (2018) "Mahalanobis OOD score" and its class-conditional variants
- **Covariance / ZCA whitening** of features or inputs (full-covariance whitening + Euclidean = Mahalanobis
  under another name). This includes GCN + ZCA in older CIFAR pipelines.
- Also watch for: PCA *with* `whiten=True` followed by Euclidean distance (equivalent to Mahalanobis
  in the retained subspace). PCA reconstruction error (unwhitened) is permitted.

**Permitted substitutes:** per-feature robust standardisation `(x − median)/(IQR + δ)`, RBF-kernel
SVDD / ν-OCSVM, Gaussian KDE, Local Outlier Factor (novelty mode), Isolation Forest, PCA
reconstruction error, shallow autoencoder reconstruction error. Per-channel mean/std input
normalisation is *diagonal* and permitted.

Docs may *name* the banned constructs in order to ban them. The only code allowed to spell them out is
the C1 grep-guard test (`tests/*c1*`). Enforced by `.claude/hooks/c1_guard.sh` and SPEC §11.4's test.

## C0 — additions (cuttable) vs commitments (never cut)

**Commitments:** `bf_w` and `bf_b` coverage; both §4.3 tracks (S and H); M2 and M3; all eight contract
metrics (§9.1); overhead as a first-class result.

**Additions**, with cut cost:

| Addition | Cut cost |
|---|---|
| §3.3 reinvestments (extra seeds, extended taps on both models, full INT8 parity) | Forgoes planned gains only |
| L2 adaptive attacker (§10) | None to the contract; first real reinvestment candidate |
| `sa0`/`sa1`, `rnd_val` | None |
| `bf_act` | None (contract scopes to weights/biases) |
| Deep SVDD (D4) | None |
| **INT8 arm** | **Effectively uncuttable** — the only external anchor (BFA/DeepHammer/TBT/ProFlip are all INT8). M2's INT8 is never cut |
| C1 itself | Not the contract's; a separate standing requirement — not cuttable either |

## Cut order under schedule/compute pressure (§4.2, §13)

1. §3.3 reinvestments (extended taps, extra seeds beyond 5)
2. Repetitions 100 → 50 per grid cell
3. L2 adaptive attacker
4. `sa` / `rnd_val` / `bf_act`
5. D4
6. M3's INT8 arm — **never M2's**

**Never cut:** bit strata (they keep §9.1 honest), `bf_w`/`bf_b`, Track S or H, M2/M3, M2 INT8.
Any §9.6 ablation is cut only after all of the above.

## Hypotheses and rejection criteria (§1)

| ID | Claim | Rejected if |
|---|---|---|
| H1 | Multi-layer SVDD beats output-only scores at low FPR | Fused TPR@1%FPR − best output-only: paired-bootstrap 95 % CI includes 0. **Separately on Track S and Track H; Track S is the discriminating test** |
| H2 | Fusing K > 1 layers beats the best single layer | Fused TPR@1%FPR ≤ best per-layer within CI |
| H3 | K ≤ 3 taps, ≤ 32 features/tap retains ≥ 95 % of full TPR@1%FPR | Retention < 95 % on **both** M2 and M3 |
| H4 | First-alarm layer correlates with injected layer | Spearman ρ not significantly > 0 |

**Design targets, never rejection criteria:** ≥ 5-point TPR@1%FPR gain (H1); < 10 % added latency for
the H3 minimal config, read against M2. Missing a target = *confirmed hypothesis with a deployment
caveat*, not a failure.

**Scope wording is mandatory:** H1–H4 are claims about *convolutional image classifiers on 32×32
inputs*. Never write them as claims about DNNs in general.
