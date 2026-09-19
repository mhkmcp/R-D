---
name: cifar-data
description: Acquire, assemble and verify the CIFAR-10 data for fidnn — Kaggle train.7z + trainLabels.csv for training, canonical torchvision test set as the labelled probe pool, CIFAR-10-C as the confounder suite — and implement src/fidnn/data (stratified persisted splits, per-channel normalisation fitted on the classifier train split, source-disjointness and label-agreement tests, provenance hashes). Use for anything that downloads, loads, splits or preprocesses data.
argument-hint: "[download | splits | verify | loader]"
---

# CIFAR-10 data

Load `.claude/references/data.md`. The full acquisition checklist is `docs/Dataset.md` §13.

## Hard rules

- **Never fetch `test.7z`** and never run `kaggle competitions download cifar-10` without `-f` (a hook
  blocks both). Only `train.7z` and `trainLabels.csv` come from Kaggle.
- Never read or echo `~/.kaggle/**` or `.env*` (denied). If Kaggle auth fails, tell the user to accept
  the competition rules in the browser and configure credentials themselves.
- `kaggle competitions submit` is at most **once**, at M-1, and needs the user's approval.
- Raw data goes under `data/` (git-ignored — add it to `.gitignore` if missing), never `artifacts/`.
- **No GCN, no ZCA, no covariance of pixels** (C1). Per-channel mean/std only, fitted on `train` (the 40k classifier split).

## Loader implementation (`src/fidnn/data/`)

1. Kaggle images: decode `1.png`…`50000.png`, join labels from `trainLabels.csv` by id — do not rely on
   directory listing order. Assert 50,000 images, all 32×32×3, 5,000 per class.
2. Canonical test: `torchvision.datasets.CIFAR10(root="data/cifar10-canonical", train=False,
   download=True)`. Assert 10,000, 1,000 per class.
3. Splits: stratified `train` 40k (classifier only) + `clean_fit`/`clean_cal`/`clean_test` 6k/2k/2k
   from the other 10k Kaggle images, fixed seed; **persist index files** (`artifacts/splits/`) and
   load them thereafter — never re-derive.
4. Normalisation constants computed from `train` only, persisted, then frozen.
5. Augmentation (pad-4 random crop, horizontal flip) only in the training loader. Feature extraction
   loaders are unaugmented.
6. CIFAR-10-C: loaded as clean-input shift only; expose corruption type, family
   (noise/blur/weather/digital) and severity as columns. Never combine with fault injection.
7. Provenance: SHA-256 of `train.7z`, `trainLabels.csv`, the torchvision archive; the disjointness
   result. Written into the sidecar and `artifacts/models/manifest.json`.

## Tests (failing, not comments)

- Source disjointness: hash every Kaggle training image's pixel bytes and every canonical test image's;
  intersection must be empty. On failure: fall back to canonical for both splits, and tell the user.
- Label agreement on a sampled subset vs canonical `train=True` labels (match by pixel hash).
- Normalisation stats equal those recomputed from `train` indices and differ from full-50k stats.
- No full-covariance transform in the pipeline (assert the transform list contains only the allowed
  types).
- Splits: disjoint, stratified, stable across runs for the same seed.

Report counts and hashes from the actual download, not from the documents.
