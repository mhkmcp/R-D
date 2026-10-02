# fidnn.cache

Clean probe-pool cache and suffix execution for the M-2 sweep. This is a C0 addition: harness speed
only, and the method is unchanged. The sweep's outputs are the same as with a full forward pass. For
INT8 they are bit-identical; `tests/test_cache.py` enforces that.

| File | Role |
|---|---|
| `graph.py` | `traced` (FX graph of either precision), `cut_points`, `Resume.start/span/starts` |
| `store.py` | `build` (one clean pass), `save` / `load` / `ensure` under `artifacts/cache/` |
| `suffix.py` | `SuffixRunner.run(layers, idx)` → fault logits plus per-tap features |

## How it works

- A **cut point** is an FX node whose output is the only live value at that position, so the rest of
  the graph depends on everything before it through that one tensor alone. On ResNet these are the
  stem and every block output, never a node inside a residual block. On VGG they are every layer.
- `build` runs the **clean** instance once over the probe pool. It stores the tensors at each cut
  point that some stateful module resumes from: INT8 `int_repr` plus per-tensor qparams, or FP32
  values. It also stores the clean per-tap features (`taps.hooks.observe`, the function `TapMonitor`
  uses) and the clean logits.
- For an injection, `SuffixRunner` resumes the **fault** instance's graph at the last cut point
  before the first faulted module (`fx.Interpreter.run(initial_env=…)`). Taps upstream of that
  point take their features from the cache.
- **Early exit.** Once every faulted module has run, the first later cut point whose value equals
  the clean one proves everything downstream is clean. The cached logits and features are returned
  from there. Checking only after the *last* faulted module matters: a multi-flip injection can span
  layers.
- An injection with no applied flips (stuck-at where every bit already holds the value) runs nothing.
- A forward pass that raises still propagates and is labelled CRASH by the sweep. `runner.partial`
  holds the taps observed before it raised.

## Invariants

- The cache comes from the clean instance only. The runner executes the fault instance (SPEC §4.4).
- The cache key covers the model `state_checksum`, a hash of the probe tensor, the cut points, the
  tap list, the saturation thresholds and the batch size. `load` refuses any mismatch and `ensure`
  rebuilds.
- **INT8 is exact.** qnnpack integer kernels do not depend on batch size, so the default
  `inject --exec` is `suffix` for INT8.
- **FP32 is not exact.** The cache is built at batch 500, the sweep runs batch 32, and FP32 GEMMs
  differ in the last bits across batch sizes. Measured on M2: ≤ 1.3e-5 on logits, only where cached
  clean logits are reused. FP32 therefore defaults to `full`. `--exec suffix` remains available for
  exploratory runs.
- §9.4 overhead and M-0 benchmarks never use this. They measure the deployed detector on a full
  forward pass.
- `artifacts/cache/` is git-ignored and can be regenerated, at about 1 GB per M2 INT8 model with the
  10k probe pool. `cache.json` is its sidecar.

## Persisted INT8 model

`fidnn quantize m2 --seed 0` writes `artifacts/models/m2_seed0_int8.pt`, an INT8 `state_dict` with
a sidecar holding the checksum and the calib hash. `models.checkpoint.load(..., "int8")` then
builds an uncalibrated FX skeleton and loads the state into it instead of re-running PTQ. It
refuses a sidecar checksum mismatch, or a calibration set different from the one the caller passes.
