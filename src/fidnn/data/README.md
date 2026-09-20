# fidnn.data

CIFAR-10 assembly for every model. SPEC §3.2, `docs/Dataset.md`. The `/cifar-data` skill has the
procedure.

| File | Role |
|---|---|
| `cifar10.py` | Kaggle download (CLI, `-f` per file), 7z extraction, canonical loader, pixel hashes, label agreement, stratified splits, channel stats |
| `prepare.py` | `fidnn data prepare`: integrity checks as hard failures, then persisted splits + stats |
| `loader.py` | `Batches`: whole dataset as a uint8 tensor on-device, normalised batches, on-device crop+flip |

## Pipeline

```bash
fidnn data download   # data/cifar10/kaggle/{train.7z,trainLabels.csv} + torchvision archive
fidnn data prepare    # artifacts/data/cifar10_splits.parquet + cifar10.json
```

| Source | Role |
|---|---|
| Kaggle `train.7z` + `trainLabels.csv` | 50k images → `train` (40k, classifier) and `clean_fit`/`clean_cal`/`clean_test` (6k/2k/2k, detector) |
| `torchvision CIFAR10(train=False)` | 10k labelled probe pool |
| `torchvision CIFAR10(train=True)` | Only used as the reference for the label-agreement check |

`cifar10.json` records normalisation stats, split counts, both integrity results, the SHA-256 of all
three archives and a provenance sidecar.

## Design

- **Kaggle via the CLI, one `-f` per file.** `KAGGLE_FILES` is the allowlist, so `test.7z` cannot be
  fetched. Single-file downloads sometimes arrive zip-wrapped; `download_kaggle` unwraps them.
- **Labels are joined by id**, never by directory order. The decoded 50k array is cached as
  `train.npz` and the extracted PNGs are deleted.
- **Integrity failures raise.** Any Kaggle-train/canonical-test pixel-hash collision, or any label
  mismatch in the 5,000-image sample, stops `prepare`. The SPEC fallback (canonical for both) is a
  decision for a human, not an automatic switch.
- **Normalisation** is per-channel mean/std from `train` only: diagonal scaling, so no C1 issue.
- **Augmentation** (zero-pad 4, random crop, horizontal flip) runs on-device per batch through two
  `gather`s. It is training-only; `Batches(augment=False)` is what feature extraction uses.
- **Two-stage stratified split** (SPEC §3.2, amended at M-1): 80 % → `train`, then 60/20/20 of the
  held-out 10k → `clean_*`. No detector record comes from an image the classifier trained on.

## Invariants

- No GCN, ZCA or pixel covariance (C1).
- Raw data stays in `data/` (git-ignored). Prepared outputs go under `artifacts/data/`.
- Split files are loaded, never re-derived.
