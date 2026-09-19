# fidnn.eval

**Status: empty.** Needed for M-5 onward. SPEC §7, §9. The `/report-results` skill covers reporting
rules.

## To build

| Piece | Contract |
|---|---|
| Fault splits | `fault_dev` 30 % / `fault_test` 70 % by **injection instance**. `fault_gen` = the {sign stratum, budget 4} configs, never in dev |
| Leakage guard | Fails the build on: fault IDs in fit/cal, test IDs in dev, `fault_gen` configs in dev, `fault_test` missing any grid cell, Kaggle/canonical hash collision |
| Metrics | TPR@1 % and @0.1 % FPR, measured FPR, precision/recall/F1, AUROC + AUPR. **Per track** (S, H) |
| Breakdowns | category × stratum × layer bucket × budget × mode (`bf_bn` separate) × precision × attacker |
| Statistics | ≥ 5 seeds, mean ± 95 % CI. Paired bootstrap (2000) on differences, DeLong/bootstrap for AUROC |
| Localisation | Spearman ρ, bucket confusion matrix, propagation depth. Separate for M2 and M3 |
| Overhead | Latency (ms, then %), peak memory, detector KB, throughput, hook vs scoring share. Batch 1 primary, CPU |
| Output | `artifacts/results/*.parquet` + sidecar, the only source for thesis tables |

## Invariants

- Tracks S and H are never merged.
- A CI that includes 0 is reported as "no detected difference".
- All analysis lives here. Notebooks only plot.
