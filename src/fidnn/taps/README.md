# fidnn.taps

Where activations are observed, and how they become fixed-width features. SPEC §5.1–§5.2.

| File | Role |
|---|---|
| `registry.py` | Stable tap ID → module path, `default` and `extended` sets per model |
| `hooks.py` | `TapMonitor`: attaches forward hooks, collects raw outputs or features |
| `features.py` | `tap_features`: per-sample descriptor, Blocks A–D (26 features) |
| `extract.py` | `fidnn extract`: saturation thresholds and clean features → `artifacts/features/` |

## Registry

| Model | Default | Extended |
|---|---|---|
| m1 | stem, stage1–3, pooled, logits | same as default |
| m2 | stem, stage1–3, pooled, logits (6) | stem, `s{1..3}b{1..3}`, `s3b3_pre`, pooled, logits (13) |
| m3 | block1–5, pooled, logits (7) | `conv{b}_{k}` × 8, pooled, logits (10) |

Paths point at `Tap()` modules, so the same registry works for FP32 and FX-converted INT8 models.
`TapInfo.pre_add` marks the one pre-addition ResNet tap in the extended set, which gives the H4
analysis a pre/post pair at the deepest block.

## Monitor

- `mode="capture"` stores each tap's output (dequantised for INT8). This measures the cost of
  observation alone, so §9.4 can separate hook cost from feature cost.
- `mode="features"` stores `tap_features(...)` per tap.
- Hooks use `always_call=True`, so a fault that raises in a later module does not drop observations
  already made.
- `remove()` (or leaving the `with` block) detaches every hook, and the model is bare again.
- `calls` counts hook firings. The hook-coverage test asserts exactly one firing per tap per forward.

## Features

Blocks A (shape, 9), B (sparsity/saturation, 4), C (energy, 5), D (structure, 8) → **26**. Block E
(3 cross-tap ratios) is added at fusion time in `fidnn.detect`, which gives the SPEC's 29.

- Per sample, reduction-only, on the activation's own device. Nothing is copied to the host.
- Rank-2 activations (pooled, logits) are treated as C channels on a 1×1 grid and take the D-conv
  path.
- `dense=True` switches to D-dense: 4 structure features, 4 spatial slots zero-padded. No v2.0 model
  produces a dense tap. The path is kept for synthetic tests and the §9.6(3) downgrade.
- `sat_threshold` is the clean p99.9 per tap, fitted by `extract.saturation_thresholds` on a
  `clean_fit` subsample (`configs/taps/extract.yaml: sat_sample`) — an exact quantile over every
  clean_fit activation would be ~10^8 values per tap. The default `inf` saturates nothing and is
  only for timing.
- **C1:** no covariance-based feature here or downstream.

## Extraction (`extract.py`, M-3)

`fidnn extract m2 --precision int8 --tap-set extended` writes one row per (sample, tap) for
`clean_fit`/`clean_cal`/`clean_test` to `artifacts/features/{model}_{precision}_{tap_set}_seed{n}_clean.parquet`,
with a sidecar carrying the tap list, feature names and the fitted thresholds.

Fault-side features come from the same hooks during the M-2 sweep: `fidnn inject m2 --tap-set
extended` attaches a monitor to the **fault** instance and writes `*_fault_features.parquet`, keyed
by `(injection_id, probe_index)` so it joins onto the outcome labels.

**Batch size is part of the provenance.** Conv reductions are not associative, so extracting at a
different batch size moves features by ~1e-4 relative. Determinism holds per batching, and the batch
size is recorded in the sidecar.

## Tests owed (SPEC §11.4)

Hook coverage (each tap fires exactly once per forward; removing hooks restores baseline latency) ·
tap budget ≥ 5 per model · D-conv non-zero on every conv tap · D-dense on synthetic `(N, D)` ·
determinism (same input gives the same vector).
