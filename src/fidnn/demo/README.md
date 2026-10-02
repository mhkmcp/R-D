# fidnn.demo

A plain-language demo for non-specialists: flip bits in the INT8 model, see the answer change (or
not), and see whether the fitted detectors notice. This is a C0 addition. It is presentation only
and produces **no thesis number**.

| File | Role |
|---|---|
| `session.py` | `DemoSession`: flips, exact restore, live evaluation with the fitted detectors. No UI dependency |
| `app.py` | `fidnn ui`: the Gradio front end (`uv sync --group ui`) |
| `export.py` | `fidnn ui --export out.json`: precomputed scenarios and result tables for the static page |

## Contract

- **Nothing is fitted or calibrated here (C2, C3).** The detectors are the `fidnn fit` bundle
  (`artifacts/detectors/{stem}.joblib`). τ_α comes from its `_calibration.parquet`, and saturation
  thresholds come from the extract sidecar. The demo only scores.
- **Two instances (SPEC §4.4).** The cache is built from the clean instance. Flips go to the fault
  instance only, inside `injected_flips`. After every evaluation the session checks
  `state_checksum` and raises if the clean state did not come back. `reset()` clears the flip list;
  the model itself was already restored.
- **Live results are the experiment's results.** Evaluation runs `fidnn.cache.SuffixRunner`, which
  is bit-exact on INT8. `tests/test_demo.py` checks it against a full forward pass. The UI is
  therefore INT8 only.
- **CRASH means what §4.3 says.** The chosen picture is always judged together with 31 fixed other
  probes (`BATCH` = the sweep's `probes_per_injection`). "Accuracy collapses to chance" is an
  injection-level property, and on one picture alone every wrong answer would read as a crash.
- **Tracks S and H are never merged on screen.** The plain-language names are "damage hidden /
  answer unchanged" (S) and "answer wrong or broken" (H).
- Non-finite features, from a blown-up forward, score +∞, which is always over threshold. The fit
  pipeline never sees them.
- The Gradio app holds **one** shared session. It is a local, single-user tool and must not be
  exposed as a multi-user service.

## Shareable page

`fidnn ui --export demo.json` writes the following, deterministic for a seed:
- 24 gallery pictures, as 32×32 PNG data URIs written without an image library;
- 3 buckets × 5 strata × budgets {1, 16} × 2 draws of precomputed scenarios;
- the M-5 headline rows;
- the M-0 CPU overhead rows, so detection is never shown without its cost (C5).

The published page embeds that JSON and states the seed count while `enough_seeds` is false (C4).
