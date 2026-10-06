# fidnn.deploy

`fidnn export` writes a self-contained bundle: one ONNX graph holding the classifier and the D2
monitor, plus a runner that needs only `onnxruntime`, `numpy` and `Pillow`. This is a C0
addition. It is presentation and deployment only and produces **no thesis number**.

| File | Role |
|---|---|
| `graph.py` | `MonitoredClassifier`: uint8 NHWC → normalisation → model → tap descriptors → Block E → robust normaliser → D2 RBF-SVDD score → alarm per α |
| `export.py` | Builds the graph from fitted artifacts, exports ONNX, checks parity, measures overhead, writes the bundle and zip |
| `bundle/` | Files copied verbatim into every bundle: `run.py`, the recipient's `README.md`, `requirements.txt` |

Output: `artifacts/deploy/{model}_{precision}_{tap_set}_seed{n}/` and a `.zip` of it. Install the
toolchain with `uv sync --group deploy`.

## Contract

- **Nothing is fitted here (C2, C3).** The constants come from these sources:
  - channel stats: the `train` split;
  - saturation levels: the extract sidecar (`clean_fit`);
  - normaliser and SVDD: the `fidnn fit` bundle;
  - τ_α: the `_calibration.parquet` (`clean_cal`).
- **The SVDD is written out as kernel arithmetic:**
  `score = −(Σ dual_coef·exp(−γ‖z − sv‖²) + intercept)`. That is `−decision_function` (SPEC
  §6.1), with no covariance (C1).
- **Parity is a gate, not a hope.** Export fails unless ONNX Runtime reproduces the fidnn alarm
  decision on every `clean_test` image at every α, and the score to within `SCORE_TOL`. The
  residual is FP32 kernel rounding.
- **Overhead travels with detection (C5).** The manifest records single-image CPU latency of the
  classifier alone and of the monitored graph, both under ORT.
- **FP32 only.** FX-quantised INT8 modules have no faithful ONNX translation. The exported monitor
  would not match the features it was calibrated on.

## Decisions

- `graph.tap_features` is a twin of `taps.features.tap_features`, not a reuse of it.
  `torch.quantile` and data-dependent indexing don't export, so it uses sorted rows with static
  interpolation indices instead. `tests/test_deploy.py` holds the two to parity.
- Entropy uses `xlogy(p, p + ε)`. The literal `p·log(p + ε)` exports to a graph that ONNX Runtime
  evaluates as NaN at p = 0, which happens on dead stem channels.
- Tap outputs are captured by swapping each `Tap` identity for a recorder. The model's own
  `forward` is untouched, so the same code covers ResNet and VGG.
- The m-of-n window is stateful, so it lives in `run.py`, not in the graph.
