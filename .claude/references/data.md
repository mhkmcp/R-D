# Data — CIFAR-10 only (SPEC §3.2, Dataset.md)

**One dataset, final.** M4 (Speech Commands) and M5 (UNSW-NB15) are *withdrawn*, not deferred.
Reinstating either is a pure C0 addition. No torchaudio, no tabular pipeline.

## Two sources, one dataset

| Role | Source |
|---|---|
| Training data (→ `train` + `clean_fit`/`clean_cal`/`clean_test`) | Kaggle `train.7z` (50,000 PNG, `1.png`…`50000.png`) + `trainLabels.csv` |
| Labelled held-out probe pool | Canonical `torchvision.datasets.CIFAR10(train=False)` — 10,000 labelled images |
| Kaggle `test.7z` | **Never downloaded.** 300,000 images, 290,000 are dummies, no labels. Useless for the §4.3 taxonomy |
| Confounder set | CIFAR-10-C, Zenodo `10.5281/zenodo.2535967`, CC BY 4.0 — 15 types × 5 severities |

Download (after accepting competition rules in the browser):

```bash
kaggle competitions download cifar-10 -f train.7z        -p data/cifar10
kaggle competitions download cifar-10 -f trainLabels.csv -p data/cifar10
7z x data/cifar10/train.7z -o data/cifar10/
```

Never `kaggle competitions download cifar-10` without `-f` (pulls `test.7z`). Credentials live in
`~/.kaggle/` — never read or print them. A leaderboard submission is allowed **once**, at M-1, as an
independent accuracy check; it produces no thesis number.

## Splits

Stratified per class (10 % each), seeded, **persisted**:

- `train` 40,000 (80 % of Kaggle 50k) — **classifier training only**, never a detector record
- `clean_fit` 6,000 / `clean_cal` 2,000 / `clean_test` 2,000 — 60/20/20 of the other 10,000 Kaggle
  images, which the classifier never sees (amended at M-1: detector records must all be unseen)
- Fault probe pool — canonical 10,000 test images, disjoint from all four

## Preprocessing

- Per-channel mean/std normalisation (≈ mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470, 0.2435,
  0.2616)`), **fitted on `train` only**, frozen.
- **No GCN / ZCA whitening** — C1 violation. Assert it in a test.
- Augmentation (random crop 32 with pad 4, horizontal flip) is **training only**. Detector feature
  extraction uses unaugmented images.
- INT8 PTQ calibration: a subsample of `clean_fit` only.

## Integrity checks — failing tests, not comments

1. **Source disjointness:** zero pixel-hash collisions, Kaggle train vs canonical test. If it fails,
   fall back to canonical for both and drop the Kaggle packaging.
2. **Label agreement:** `trainLabels.csv` agrees with canonical training labels on a sampled subset
   (catches mis-ordered 7z extraction).
3. Channel stats fitted on `train` only; `train` disjoint from every detector split.
4. No full-covariance transform anywhere in the input pipeline.

Known limit (not a C2 violation): ~3.3 % of CIFAR-10 test images have a *near*-duplicate in train
(Barz & Denzler 2020). Pixel hashing catches exact duplicates only. Mitigation if it ever matters:
re-run the headline with the published near-duplicate list removed from the probe pool.

## Accuracy gate (M-1)

| Model | Reference | Tolerance |
|---|---|---|
| M2 ResNet-20 | **91.25 %** (He et al. 2016) | within 1 pt |
| M3 VGG-11-BN | ≈ 92 % — **must be pinned to a specific cited source** in `artifacts/models/manifest.json` | within 1 pt |

## Provenance (§11.3)

Sidecar for CIFAR runs records SHA-256 of `train.7z` and `trainLabels.csv`, the torchvision archive
hash, and the disjointness-check result. All checksums also go in `artifacts/models/manifest.json`.

## Licensing

CIFAR-10: no formal licence, cite Krizhevsky (2009); Kaggle copies under competition rules, no
redistribution. CIFAR-10-C: CC BY 4.0, cite Hendrycks & Dietterich (2019). One provenance sentence in
the thesis: parent 80M Tiny Images was withdrawn in 2020; CIFAR-10's ten hand-verified classes are not
affected.
