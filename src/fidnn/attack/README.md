# fidnn.attack

Attacker levels above L0. SPEC §2, §6.4, §10. L1 is built (M-1b); L2 is owed at M-8.

| File | Role |
|---|---|
| `bfa.py` | L1: BFA progressive bit search on INT8 weights (Rakin et al. 2019) |
| `run.py` | `fidnn attack bfa`, and the `docs/M1b_bfa.md` validation record |

| Attacker | Contract |
|---|---|
| L0 | Uniform random parameters/bits. That is the planner in `fidnn.inject`; no code here |
| L1 | Faithful BFA: rank bits by gradient, evaluate the shortlist, flip the best, repeat |
| L2 | Detector-aware greedy search: candidates ranked by loss gain, filtered to keep every monitored SVDD score under its threshold (**M-8**) |

## How L1 works here

1. **Gradient ranking needs a float twin.** Autograd does not run through quantised kernels, so
   `surrogate()` fuses a copy of the FP32 model (`fuse_fx`, the same fusion the INT8 conversion
   used, so module paths address both) and copies the dequantised int8 weights into it.
2. **In-layer search.** Per layer, the first-order gain of flipping bit *b* of weight *i* is
   `grad · Δw`, where `Δw = (flipped_int8 − int8) × scale` — per-channel scale under qnnpack. Only
   loss-increasing flips survive, and the top `candidates_per_layer` go to the next step.
3. **Cross-layer search.** Every shortlisted candidate is applied to the *real INT8 model*, scored
   by the actual loss, and restored. The single best flip across all layers is kept.
4. Repeat until the flip budget, or until accuracy drops below the target.

Flips stay applied: the threat model is persistent corruption (§2). `replay()` re-applies a
recorded attack to a clean model, so any run reproduces exactly.

## M-1b gate (SPEC §12)

`fidnn attack bfa --model m2` runs 5 trials and writes `docs/M1b_bfa.md`, comparing our N_flip
(flips to push top-1 below 11 %) against **Rakin et al. 2019, Table 2** — ResNet-20, Nq = 8:
`[7, 10, 10, 12, 17]`. The gate passes when our median lands inside that range. Known differences
that are not by themselves bugs: our INT8 is per-channel PTQ through qnnpack (theirs is per-layer),
and our baseline trains on the 40k `train` split (§3.2).

## Invariants

- Every flip goes through `fidnn.inject`, so restore, checksum and record guarantees hold.
- The attacker sees only its sample batch (128 images, as in the paper) — never the labels of the
  evaluation set it is scored on.
- L1 generates no fault population before the M-1b gate passes; L2 runs on M2 INT8 first (§10).
