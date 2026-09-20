# fidnn.eval

Fault splits, the leakage guard, metrics and the results build. SPEC §7, §9. The `/report-results`
skill covers the reporting rules.

| File | Role |
|---|---|
| `splits.py` | `fault_dev` / `fault_test` by injection **instance**; `fault_gen` tagged configs |
| `leakage.py` | The guard that fails a results build instead of reporting a leaked number |
| `metrics.py` | TPR at calibrated thresholds, precision/recall/F1, AUROC + AUPR, per track |
| `stats.py` | Bootstrap CIs, paired-bootstrap differences, the ≥ 5-seed rule |
| `localisation.py` | H4: Spearman ρ, bucket confusion, propagation depth, pre-add share |
| `run.py` / `report.py` | `fidnn eval`, `fidnn report` → `docs/M5_results.md` |

## What the guard refuses to build

`leakage.assert_clean` raises before any table is written when: an injection instance is in both
`fault_dev` and `fault_test`; a `fault_gen` configuration (stratum `sign`, budget 4) appears in
`fault_dev`; `fault_test` is missing any bucket, stratum or budget of the grid; a fault record id
turns up in a clean split; or the Kaggle/canonical pixel-hash check found collisions.

## Reporting rules this package enforces

- **Tracks S and H never merge.** Every metric row carries its track.
- **No aggregate-only table.** `metrics.breakdowns` produces the mandatory per-category, per-stratum,
  per-bucket, per-budget, per-mode, per-precision and per-attacker rows.
- **`fault_gen` is its own row**, never folded into the headline.
- **A CI that includes 0 is "no detected difference"**, never a win — `stats.paired_difference`
  returns that verdict string, and the report prints it verbatim.
- **Fewer than 5 seeds is not reportable** (C4): the report prints a warning banner instead of
  quietly showing a single-run number.
- **Detection is read with overhead** (C5): the results page links the overhead record rather than
  standing alone.

## Decisions

- **Spearman is rank-based**, so H4 correlates the injected bucket index directly with the
  first-alarm tap index; no scaling between the two axes is needed or applied.
- **Output-only baselines** are scored from the per-probe scores recorded during the sweep and the
  matching clean scores from extraction, so both sides see the same probes and the same threshold
  rule.
