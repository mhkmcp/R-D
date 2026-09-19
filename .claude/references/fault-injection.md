# Fault injection (SPEC §2, §4)

## Threat model (§2)

- Goal: silent data corruption. Capability: bounded bit flips in parameter memory (Rowhammer, laser,
  glitching). **Persistent** faults are primary; **transient** secondary.
- Attacker levels: **L0** blind random; **L1** white-box BFA progressive bit search; **L2**
  detector-aware (knows taps + detector, keeps every SVDD score under threshold) — §10.
- Out of scope: control-flow faults, input faults, faults in the detector itself, supply chain,
  training-time backdoors.

## Fault modes (§4.1)

| Code | Fault | Grid |
|---|---|---|
| `bf_w` | Bit flip(s) in a weight | **Full** (co-primary) |
| `bf_b` | Bit flip(s) in a bias | **Full** (co-primary — never a reduced arm) |
| `bf_bn` | Flip in BN buffers `running_mean` / `running_var` (tag within `bf_w` family) | Reduced; **reported separately from `bf_w`** |
| `sa0` / `sa1` | Stuck-at on a parameter bit | Reduced |
| `rnd_val` | Whole-parameter random value | Reduced |
| `bf_act` | Transient flip in an activation (secondary) | Reduced; **never mixed into main tables** |

BN affine `weight`/`bias` are ordinary `bf_w`/`bf_b` targets. A `running_var` flip driving it
negative → NaN via `sqrt` → `CRASH`; it must not flatter Track H.

INT8: qnnpack keeps quantised-module biases in FP32, so `bf_b` on INT8 flips FP32 bias bits.

## Grid (§4.2)

- **Bit strata** — FP32: sign (31) · exponent MSBs (30–27) · remaining exponent (26–23) · high mantissa
  (22–16) · low mantissa (15–0). INT8: all 8 positions, sign/MSB separated. **Stratified, never
  uniform. Never cut.**
- **Layer:** every injectable layer, bucketed early / middle / late (ResNet-20: the 3 stages; VGG-11-BN:
  5 conv blocks folded into thirds).
- **Flip budget** N ∈ {1, 4, 16} (L0/L1).
- **Repetitions:** 100 injections per (model, layer bucket, bit stratum, budget) cell.
- **Probes per injection:** 32 inputs. Statistical unit = (injection, input) pair.
- **Reduced grid** (`sa*`, `rnd_val`, `bf_act`, `bf_bn`): late bucket only; strata {exponent-MSB,
  low-mantissa}; budgets {1, 16}.
- **Model coverage:** M2 and M3, full grid, FP32 **and** INT8.
- Record the per-layer bias parameter count alongside `bf_b` results (much denser pool coverage).
- ≈ 4,500 injections × 32 probes ≈ 144k forwards per model/mode/precision/seed.
- **Track S population gate:** if `MASKED` + `DEGRADED` < ~15 %, escalate to ResNet-56 **before** the
  full sweep.

## Outcome taxonomy (§4.3) — label every run by effect

| Label | Definition | Track |
|---|---|---|
| `MASKED` | Top-1 unchanged, logit shift < ε | **S** |
| `DEGRADED` | Top-1 unchanged, margin/confidence shift ≥ ε | **S** |
| `SDC` | Top-1 changed, no numerical failure | **H** |
| `CRASH` | NaN/Inf in output, or collapse to chance | **H** |

- Tracks are **co-primary and never merged** into one "fault" class. H1 is evaluated on each; Track S is
  the discriminating one.
- Break out `CRASH` within H (a NaN detector is trivial) and `MASKED` vs `DEGRADED` within S.
- A Track S alarm is a **detection**, never reported as a false positive.
- Labelling needs ground truth for every probe → probes come from the canonical labelled test set only.

## Implementation rules (§4.4) — see `src/fidnn/inject/bitflip.py`

- FP32 via `tensor.view(torch.int32)`, INT8 via `int8`; XOR with a bit mask.
- **In place, XOR-ed back** after the probe batch. Never `deepcopy` / `state_dict` copy per injection.
- Restore in a **`finally`** block (`injected(...)` context manager) so an exception cannot leave the
  model corrupted. Multi-flip restore applies flips in reverse order.
- Quantised conv/linear: unpack → XOR int8 repr → requantise with same qparams → repack.
- **Periodic checksum over `state_dict()`** (parameters **and buffers**) every N injections, asserting
  return to clean state (`state_checksum`). `model.parameters()` misses `bf_bn` targets.
- Every injection is a replayable record written to `artifacts/faults/*.parquet`:
  `{model, precision, seed, layer, flat_index, bit, count, mode}`.
- **No fault is ever applied to the model instance used to generate clean training features.**

## L1 / L2 (§6.4, §10)

- L1 = faithful BFA reimplementation. **M-1b gate:** its flip-budget-vs-accuracy curve on INT8
  ResNet-20/CIFAR-10 must match published figures *before* it generates the fault population.
- L2 = greedy constrained search: rank (param, bit) by loss gain, filter by monitor score. Report attack
  success rate and required flip budget **with vs without monitor**. Run on M2 INT8 first. Honest
  framing: an increased required budget is a real gain even if the detector is evadable.
