# Features and detectors (SPEC §5, §6)

## Taps (§5.1) — `src/fidnn/taps/registry.py`

Hooks: `module.register_forward_hook(hook, always_call=True)`. Each tap has a stable string ID.

| Model | Default | Extended |
|---|---|---|
| M1 ResNet-8 (smoke test, no thesis result) | 6 | — |
| **M2 ResNet-20** | 6: `stem`, `stage1–3`, `pooled`, `logits` | 13: stem, 9 block outputs `s{s}b{b}`, `s3b3_pre` (pre-addition), pooled, logits |
| **M3 VGG-11-BN** | 7: `block1–5`, `pooled`, `logits` | 10: 8 conv taps `conv{b}_{k}`, pooled, logits |

- Every model must expose **≥ 5 usable taps** (tested from the registry).
- ResNet taps record `pre_add` — post-addition taps see `F(x) + x`, whose skip path can mask a fault
  inside `F`. That is the mechanism H4 measures, not a bug.
- Extended sets are **planned** on both models (§3.3). Headline results use default sets unless the K
  sweep says otherwise; **record the tap set with every number**.

## Per-tap descriptor (§5.2) — `src/fidnn/taps/features.py`

Per sample, single-pass, reduction-only, on the activation's device (no host copy).

| Block | Features | n |
|---|---|---|
| A — shape | mean, std, min, max, q01, q25, q50, q75, q99 | 9 |
| B — sparsity/saturation | zero fraction, fraction > clean p99.9, NaN count, Inf count | 4 |
| C — energy | L1/√n, L2/√n, L∞, L∞/L2, energy Gini over channels (conv) / units (dense) | 5 |
| D-conv | channel-energy entropy, top-1/2/4 channel shares, mean & std of per-channel spatial means, mean & std of per-channel spatial stds | 8 |
| D-dense | unit-energy entropy, top-1/2/4 unit shares; 4 spatial slots zero-padded | 4 (+4 pad) |
| E — cross-layer (at fusion) | L2 ratio to previous tap, entropy ratio, saturation-ratio difference | 3 |

Operative width **29 per tap** (26 from A–D + 3 from E). Pooled/logit vectors take the D-conv path as
C channels on a 1×1 grid. **No v2.0 model produces a dense tap**; D-dense is tested on synthetic
`(N, D)` tensors and exercised by the §9.6(3) downgrade. See README "Known inconsistencies" on 22 vs 26.

Descriptors are permutation-invariant over channels and positions → width independent of layer width
(what makes H3 answerable).

## Normalisation (§5.3)

`z = (x − median) / (IQR + δ)`, median/IQR from **`clean_fit` only**, frozen. Features with IQR below a
floor are dropped at fit time and **the drop list is persisted**. **No covariance whitening (C1).** The
RBF `gamma` absorbs remaining isotropic scale.

## SVDD (§6.1)

- RBF-kernel SVDD ≡ ν-one-class SVM → `sklearn.svm.OneClassSVM(kernel="rbf", nu=…, gamma=…)`;
  state the SVDD interpretation in the thesis.
- Score `s(x) = −decision_function(x)`; `> 0` ⇒ outside boundary.
- Large sets: `make_pipeline(Nystroem(gamma, n_components), SGDOneClassSVM(nu))`. **Measure** whether
  it is needed (`clean_fit` = 6k; 13-tap fused ≈ 377-d) and, if used, report the approximation gap vs
  exact OCSVM **once, on M2** (not M1).

## Variants (§6.2) — the study's independent variable

| ID | Variant |
|---|---|
| D1 | One SVDD per tap, every tap — per-layer baseline (H2) and propagation map (H4) |
| D2 | Early fusion: concatenate normalised features of all taps → one SVDD |
| D3 | Late fusion: per-tap SVDDs, scores mapped through each tap's **clean-cal empirical CDF**, combined by max / mean / clean-FPR-weighted mean / one-class combiner. First tap over threshold = **first-alarm layer** (H4) |
| D4 | Deep SVDD *(extension)*: no bias terms, unbounded activations, no learnable output scale, centre `c` fixed from an initial forward pass. Only if D2/D3 leave headroom |

## Hyperparameters (§6.3) — clean data only

- `ν ∈ {0.001, 0.005, 0.01, 0.05, 0.1}`
- `gamma ∈ {"scale"} ∪ {k·γ_median : k ∈ {0.1, 0.25, 0.5, 1, 2, 4}}`, `γ_median = 1/median‖xᵢ−xⱼ‖²` on a
  clean subsample.
- Select the pair **minimising clean-validation FPR subject to a support-vector-fraction stability
  band**, then verify on a clean held-out split. **Selecting by test AUROC is a C2 violation.**

## Baselines (§6.4)

- **Output-only:** max softmax probability, softmax entropy, logit margin, energy score.
- **Feature-space, on the same features as D2:** Isolation Forest, LOF (`novelty=True`), Gaussian KDE,
  PCA reconstruction error, shallow autoencoder reconstruction error.
- **Systems context (not a competitor):** parameter checksum/hash — perfect on persistent faults, blind
  to transient ones, fixed cost.
- **Attack-side reference:** L1 BFA curve vs published (M-1b).
- **Excluded by C1:** Mahalanobis OOD score (Lee et al. 2018), `EllipticEnvelope`, `MinCovDet`. The
  thesis methods paragraph (§14(4)) pre-empts the "where is Mahalanobis?" question rather than
  answering it defensively.
