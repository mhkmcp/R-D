# fidnn.deploy

`fidnn export` writes a portable bundle with:
- one ONNX graph holding the classifier and the D2 monitor;
- classifier-only versions of that model for smaller devices;
- a browser UI that runs the ONNX graph with ONNX Runtime Web;
- a runner that needs only `onnxruntime`, `numpy` and `Pillow`.

This is a C0 addition. It is presentation and deployment only and produces **no thesis number**.

| File | Role |
|---|---|
| `graph.py` | `MonitoredClassifier`: uint8 NHWC → normalisation → model → tap descriptors → Block E → robust normaliser → D2 RBF-SVDD score → alarm per α |
| `formats.py` | Classifier-only exports: ONNX FP32, static INT8 QDQ, ORT format, TFLite through an isolated `uv` env |
| `tflite_convert.py` | onnx2tf → TFLite FP32/FP16/full INT8, a C header, and per-file accuracy. Runs **outside** the project env |
| `export.py` | Builds everything from fitted artifacts and gates on parity. Records accuracy, size and hash per file, the weight byte-offset table and overhead |
| `bundle/` | Copied verbatim into every bundle: `run.py`, the recipient's `README.md`, `requirements.txt`, `ui/index.html` |

Output: `artifacts/deploy/{model}_{precision}_{tap_set}_seed{n}/` and a `.zip` of it.
- Install with `uv sync --group deploy --group ui`. `uv sync --group deploy` alone uninstalls
  Gradio.
- The first TFLite run downloads TensorFlow into uv's cache, about 1 GB; `--no-tflite` skips it.
- ONNX Runtime Web is downloaded once, pinned, into `artifacts/cache/onnxruntime-web-*`.

## Contract

- **Nothing is fitted here (C2, C3).** The constants come from these sources:
  - channel stats: the `train` split;
  - saturation levels: the extract sidecar (`clean_fit`);
  - normaliser and SVDD: the `fidnn fit` bundle;
  - τ_α: the `_calibration.parquet` (`clean_cal`);
  - INT8 and TFLite calibration: a `clean_fit` subsample, as in `fidnn quantize`.
- **The SVDD is written out as kernel arithmetic:**
  `score = −(Σ dual_coef·exp(−γ‖z − sv‖²) + intercept)`. That is `−decision_function` (SPEC
  §6.1), with no covariance (C1).
- **Parity is a gate, not a hope.** Export fails unless ONNX Runtime reproduces the fidnn alarm
  decision on every `clean_test` image at every α, and the score to within `SCORE_TOL`. The
  residual is FP32 kernel rounding.
- **Every file carries its own measurement.** The manifest's `formats` gives accuracy on
  `clean_test` for each file, measured in the runtime that file targets. That is ORT for
  `.onnx`/`.ort` and LiteRT for `.tflite`.
- **Overhead travels with detection (C5).** The manifest records single-image CPU latency of the
  classifier alone and of the monitored graph, both under ORT.
- **Only `model.onnx` / `model.ort` carry the monitor.** The classifier-only formats exist for
  devices that can't afford it.
- **Calibration matches the model it runs on.** Re-quantising `model.onnx` would quantise the
  monitor's features and silently invalidate τ_α. A quantised monitor needs a fresh
  `extract → fit` on the quantised model.

## Decisions

- **Twin descriptor.** `graph.tap_features` is a twin of `taps.features.tap_features`, not a
  reuse of it. `torch.quantile` and data-dependent indexing don't export, so it uses sorted rows
  with static interpolation indices instead. `tests/test_deploy.py` holds the two to parity.
- **Entropy uses `xlogy(p, p + ε)`.** The literal `p·log(p + ε)` exports to a graph that ONNX
  Runtime evaluates as NaN at p = 0, which happens on dead stem channels.
- **Tap capture.** Tap outputs are captured by swapping each `Tap` identity for a recorder. The
  model's own `forward` is untouched.
- **INT8 ONNX skips symbolic shape inference** (`quant_pre_process(skip_symbolic_shape=True)`).
  It fails on the `Range` node of the option-A shortcut under a dynamic batch.
- **TFLite starts from a separate graph:** NCHW float input, static batch 1.
  - With a dynamic batch, onnx2tf can't convert the shortcut's `Range`.
  - With an NHWC input, it inserts a second transpose and breaks full-INT8 calibration.
  - From NCHW it produces a native NHWC TFLite input.
  - onnx2tf's built-in sample download is a pickled `.npy` that NumPy 2 refuses to load, so it
    is replaced with the clean calibration images.
- **TFLite never enters the project env.** TensorFlow pins its own protobuf and numpy, so
  conversion runs under `uv run --isolated` with pinned versions.
- **Browser fault simulation** flips bytes at the offsets in `manifest.weights`. Those are the
  positions of each FP32 weight's raw data inside `model.onnx`, recorded only when the bytes
  occur exactly once.
- **The m-of-n window is stateful**, so it lives in `run.py` and the UI, not in the graph.
