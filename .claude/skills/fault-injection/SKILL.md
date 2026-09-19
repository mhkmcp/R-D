---
name: fault-injection
description: Implement or extend fault injection in src/fidnn/inject (bit flips in weights, biases, BN buffers, stuck-at, random-value, activation faults; the injection planner; replay records; outcome taxonomy labelling) following SPEC §4. Use when adding a fault mode, writing the §4.2 grid planner, the MASKED/DEGRADED/SDC/CRASH labeller, or anything that mutates model state during a sweep.
argument-hint: <fault mode or component>
---

# Fault injection

Load `.claude/references/fault-injection.md` first. Existing primitives live in
`src/fidnn/inject/bitflip.py` — reuse them rather than writing new bit twiddling:

- `flip_bits_(t, flat_idx, bits)` — FP32 via int32 view, int8 direct; self-inverse.
- `flip_quantized_weight_(module, flat_idx, bits)` — unpack → XOR → requantise → repack.
- `injected(t, flat_idx, bits)` — context manager, restores in `finally`, reverse order.
- `state_checksum(model)` — covers parameters **and buffers**.

## Non-negotiables (SPEC §4.4)

1. **In place + XOR back.** No `copy.deepcopy(model)`, no `load_state_dict(clean_sd)` per injection.
2. **Restore in `finally`.** Every mutation happens inside `injected(...)` or an equivalent
   try/finally.
3. **Periodic checksum** over `state_dict()` every N injections; mismatch aborts the sweep loudly.
4. **Separate model instances.** The instance that produces clean training features is never passed to
   injection code. Make this structural (different objects), not a convention.
5. **Replayable record** per injection → `artifacts/faults/*.parquet` with at least
   `{model, precision, seed, layer, flat_index, bit, count, mode}`; add `tap_set`, `attacker`
   (L0/L1/L2), `bit_stratum`, `layer_bucket`, `injection_id`. Written by code, never by hand.
6. **Bit strata are stratified**, never uniform over 0–31. Use the §4.2 strata exactly.
7. `bf_bn` targets `running_mean`/`running_var` and is tagged separately from `bf_w`. `bf_act` is
   transient (a forward hook for one inference), reported separately.
8. INT8 `bf_b`: biases are FP32 under qnnpack — flip FP32 bits.

## Taxonomy labeller (§4.3)

Needs clean logits and the ground-truth label per probe. `CRASH` if any NaN/Inf in output (check
first, so NaN never compares as "unchanged"), else `SDC` if top-1 changed, else `DEGRADED` if the logit
shift ≥ ε, else `MASKED`. ε is a config value recorded in the sidecar. Emit the track (S/H) as its own
column so nothing downstream merges them.

## Tests to add or keep green

Round trip (FP32 + INT8, bit-exact) · FP32 bit-30 known value · restore after a raised exception
mid-probe (checksum equals clean, **with a BN buffer flipped**) · planner covers every
(bucket × stratum × budget) cell with the configured repetitions · labeller on hand-built logits for
each of the four outcomes.

Run `uv run pytest -q` and `uv run ruff check src tests` before reporting done. Any sweep beyond a smoke
test waits for M-0 (and M-1b for L1-generated faults).
