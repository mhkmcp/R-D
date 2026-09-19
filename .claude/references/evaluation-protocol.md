# Evaluation protocol (SPEC §7–§10)

## Splits (§7)

| Split | Content | Used for |
|---|---|---|
| `clean_fit` | 60 % of the 10k classifier-held-out Kaggle images (6k) | Normaliser + SVDD fit |
| `clean_cal` | 20 % | Thresholds, score-CDF mapping, ν/γ selection |
| `clean_test` | 20 % | FPR measurement |
| `fault_dev` | Random 30 % of injection **instances** across the full grid, excluding `fault_gen` configs | Dev / sanity only |
| `fault_test` | Disjoint 70 %, full grid (all buckets, strata, budgets, modes) + all L1/L2 attacks | **Headline** |
| `fault_gen` | Tagged subset of `fault_test`: configs never sampled into dev (stratum {sign}, budget {4}) | Unseen-config generalisation, **separate row** |

Split over injection **instances**, not grid axes. Clean and fault records share the same held-out
input pool (canonical CIFAR-10 test set).

**Leakage guard** — fails the results build on any of:
- a fault record ID in any fitting or calibration set;
- a `fault_test` instance ID in `fault_dev`;
- a `fault_gen` config tuple anywhere in `fault_dev`;
- `fault_test` missing any layer bucket, bit stratum or budget from the grid;
- any pixel-hash collision Kaggle train ↔ canonical test.

## Calibration (§8)

- `τ_α` = (1 − α) empirical quantile of `clean_cal` scores, α ∈ {5 %, 1 %, 0.1 %}. Per-tap for D3.
- At α = 0.1 %: finite-sample correction (upper binomial confidence bound on the quantile).
- Two regimes, always both: **per-inference** and **windowed m-of-n**, (n, m) ∈ {(1,1), (8,3), (32,8)}.
  Per-inference numbers always accompany windowed ones.

## Metrics (§9.1)

Headline: **TPR@1%FPR** and **TPR@0.1%FPR**, **Track S and Track H separately**, never aggregated.

| Contract term | Reported as |
|---|---|
| detection rate | TPR at calibrated threshold, per track |
| false-positive rate | measured FPR on `clean_test` — must land near α; a large gap is a calibration failure, reported as such |
| precision / recall / F1 | at calibrated threshold, per track |
| ROC-AUC | AUROC **plus AUPR** (classes unbalanced) |
| runtime and memory overhead | §9.4 (C5: always alongside) |

**Mandatory breakdowns — aggregate-only tables are not acceptable:** per outcome category
(`MASKED`/`DEGRADED`/`SDC`/`CRASH`), per bit stratum, per layer bucket, per flip budget, per fault mode
(`bf_bn` separate from `bf_w`), per precision (FP32/INT8), per attacker level (L0/L1/L2). Also record
model, tap set, detector variant, seed count.

**One externally-anchored table:** M2 INT8 under L1 — flips to chance accuracy and accuracy-vs-budget,
beside published figures. Presented as a *sanity check on the attack*, not a detector claim.

## Localisation (§9.2, H4)

Spearman ρ (injected layer index vs first-alarm tap index); confusion matrix injected bucket vs
first-alarm bucket; mean propagation depth. **Separately for M2 and M3 — the difference is the result.**
Use the `pre_add` flag to check whether residual first alarms concentrate on pre-addition taps.

## Latency (§9.3)

Detection latency in inferences (windowed) and inferences before the first SDC — does the monitor fire
before visible harm? Needs its own figure.

## Overhead (§9.4)

| Metric | Method |
|---|---|
| Added latency (ms and %) | Median of ≥ 1000 timed runs, warm-up discarded, batch ∈ {1, 32}, MPS and CPU, `torch.mps.synchronize()` before each timestamp |
| Peak memory (MB) | `torch.mps.current_allocated_memory()` + `tracemalloc` delta, monitor on vs off |
| Detector size (KB) | Serialised support vectors + normaliser |
| Throughput | Sustained samples/s, on vs off |
| Feature-extraction share | Hook cost vs scoring cost |

- Batch 1 is the operating point; batch 32 is context.
- **Absolute ms first, percentage second, baseline model stated with every percentage.** Percentages
  from earlier spec revisions are not comparable. H3's < 10 % target is read against M2.
- INT8 tabulated separately from FP32 (different kernels) with the reason inline.
- Per detector variant and per K.
- Fixed machine specified: chip, memory, macOS, torch version. Edge-class hardware is the *right*
  platform (§2) — state once, don't hedge.

## Statistics (§9.5)

- ≥ 5 seeds per configuration (model init where applicable, fault sampling, detector fit).
- Mean ± 95 % CI.
- Variant comparisons: **paired bootstrap, 2000 resamples** over the shared evaluation set; report the
  CI of the *difference*. AUROC differences: DeLong or bootstrap.
- CI of a difference includes 0 ⇒ write **"no detected difference"**, never a win.

## Ablations (§9.6)

1. **K sweep** — greedy forward selection over extended sets (13 on M2, 10 on M3) by clean-validation
   criteria; TPR/overhead Pareto front → H3.
2. **Tap position** — early / middle / late / logits-only.
3. **Structure** — drop each of Blocks A–E on M2 and M3; plus the **D-conv → D-dense downgrade** (zero
   the 4 per-channel spatial stats) and re-measure TPR@1%FPR. Replaces v1.1's cross-dataset contrast.
   Indirect evidence only — never written up as a substitute for a second modality.
4. **Precision** FP32 vs INT8 on M2 and M3 — runs at **M-5**, not M-7.
5. **Training budget** — clean fit n ∈ {500, 1k, 2k, 6k}.
6. **Kernel** — RBF vs polynomial vs linear.
7. **Transfer** — fit on M2, apply to M3 (expected to fail; same-data, different-topology).
8. **Input-shift confounder (CIFAR-10-C)** — clean inputs only, never combined with faults. Mandatory:
   **per corruption type** (15) and **per family** (noise/blur/weather/digital), plus fault-slope vs
   severity-slope side by side. If it is a novelty detector, the thesis says so.

## Adaptive attacker (§10)

See `fault-injection.md`. Report success rate and required flip budget with vs without monitor, in
units comparable to the §9.1 external table.
