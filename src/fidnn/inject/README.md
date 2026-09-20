# fidnn.inject

Bit-level fault injection. SPEC §4. The `/fault-injection` skill covers extending it.

## `bitflip.py`

| Function | Does |
|---|---|
| `flip_bits_(t, flat_idx, bits)` | XOR bit `bits[i]` of element `flat_idx[i]` in place. FP32 via an int32 view, int8 directly. Self-inverse |
| `flip_quantized_weight_(module, …)` | Quantised conv/linear: unpack → XOR int8 repr → requantise with same qparams → repack |
| `flip_quantized_bias_(module, …)` | Same, for the FP32 bias qnnpack keeps in the packed params |
| `injected(t, idx, bits)` | Context manager: flip on entry, restore in `finally` (reverse order) |
| `state_checksum(model)` | blake2b over the full `state_dict()`, covering parameters **and buffers** |

| File | Role |
|---|---|
| `targets.py` | Injectable tensors per model/precision and their early/middle/late buckets |
| `plan.py` | §4.2 grid: bit strata × buckets × budgets × repetitions → one row per flip |
| `engine.py` | Applies one injection to a live model and restores it in `finally` |
| `taxonomy.py` | MASKED / DEGRADED / SDC / CRASH and their tracks |
| `sweep.py` | Runs a plan over probe batches, labels outcomes, checksums periodically |
| `run.py` / `report.py` | `fidnn inject`, and the `docs/M2_taxonomy.md` exit record |

## Why it's built this way

- **In place + XOR back, never copy.** A per-injection `deepcopy`/`state_dict` reload would dominate
  the sweep. XOR is its own inverse, so restore is exact.
- **Sequential XOR** in `flip_bits_`: the same element can be hit twice in one multi-flip, and a
  vectorised scatter would drop one of the flips.
- **Sign bit mask** is negative (`-(1 << (w-1))`) because the views are signed integers.
- **Checksum over `state_dict()`**, not `parameters()`: `bf_bn` hits BN `running_mean`/`running_var`,
  which are buffers.
- **INT8 repack** is a per-injection fixed cost, paid twice (flip and restore). M-0 measures it.
- **INT8 `bf_b`:** qnnpack stores quantised-module biases in FP32, so bias flips use `flip_bits_` on
  FP32 bits, not int8 bits.

## Decisions SPEC §4 leaves open (flagged, not silently taken)

- **INT8 bit strata.** §4.2 says "all 8 positions, sign/MSB separated" without grouping them. Read as
  five strata — sign {7}, MSB {6}, {5,4}, {3,2}, {1,0} — so both precisions have the same cell count
  and the §4.2 estimate of ≈4,500 injections per arm holds.
- **VGG buckets.** "Five conv blocks folded into thirds": blocks 1–2 early, 3 middle, 4–5 and the
  classifier late.
- **ε (§4.3 MASKED vs DEGRADED)** = `configs/inject/grid.yaml: epsilon`, an L-inf logit shift, fixed
  a priori at 0.01 and recorded in every sidecar.
- **"Accuracy collapse to chance" (CRASH)** is an injection-level property: accuracy over that
  injection's own probes at or below 1/C while the clean model is above it. It labels every probe of
  the injection.
- **`rnd_val`** XORs a uniformly random mask, which is a uniformly random replacement value, so whole
  value corruption reuses the exactly restorable flip path.
- **`bf_act`** (transient activation faults) is not built yet; it is a C0 addition, and needs a
  forward hook rather than the parameter path.

## Known INT8 effects

- **`state_checksum` cannot hash the packed params' `repr`.** A quantised `Linear` exposes its packed
  weight and bias as a tuple in `state_dict()`; that repr is truncated (it would hide small flips)
  and changes on any repack even when values are identical (it would cry fault after a clean
  restore). The digest recurses into the tuple and hashes tensor bytes plus qparams instead.

## Invariants

- Never inject into the model instance that produced clean training features.
- Tests: FP32/INT8 round trip, FP32 bit-30 known value, restore after an exception with a BN buffer
  flipped, grid coverage per cell, stuck-at only flipping bits that differ, and the labeller on
  hand-built logits for each of the four outcomes.
- `bf_bn` exists in FP32 only: PTQ folds BatchNorm into the convolution, so the INT8 model has no
  running statistics to corrupt.
- The sweep aborts loudly (`RestoreError`) if a periodic checksum does not match the clean state.
