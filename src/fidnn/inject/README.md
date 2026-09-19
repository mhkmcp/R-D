# fidnn.inject

Bit-level fault injection. SPEC §4. The `/fault-injection` skill covers extending it.

## `bitflip.py`

| Function | Does |
|---|---|
| `flip_bits_(t, flat_idx, bits)` | XOR bit `bits[i]` of element `flat_idx[i]` in place. FP32 via an int32 view, int8 directly. Self-inverse |
| `flip_quantized_weight_(module, …)` | Quantised conv/linear: unpack → XOR int8 repr → requantise with same qparams → repack |
| `injected(t, idx, bits)` | Context manager: flip on entry, restore in `finally` (reverse order) |
| `state_checksum(model)` | blake2b over the full `state_dict()`, covering parameters **and buffers** |

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

## Still to build (M-2)

- Injection planner over the §4.2 grid: stratified bit strata, layer buckets, budgets {1, 4, 16},
  100 reps × 32 probes. Reduced grid for `sa*`, `rnd_val`, `bf_act`, `bf_bn`.
- Replay records → `artifacts/faults/*.parquet`:
  `{model, precision, seed, layer, flat_index, bit, count, mode, …}`.
- Outcome labeller: `CRASH` → `SDC` → `DEGRADED` → `MASKED`, with a separate track column.
- Periodic checksum assertion in the sweep loop.

## Invariants

- Never inject into the model instance that produced clean training features.
- Tests: FP32/INT8 round trip, FP32 bit-30 known value, restore after an exception with a BN buffer
  flipped.
