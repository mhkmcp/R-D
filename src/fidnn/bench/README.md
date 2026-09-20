# fidnn.bench

M-0 throughput calibration. SPEC §12 (gate), §9.4 (timing protocol), §4.2 and §3.3 (what gets sized).

| File | Role |
|---|---|
| `throughput.py` | Measures unit costs → `artifacts/m0/{inference,grad,injection,svdd}.parquet` + `sidecar.json` |
| `budget.py` | `Plan` × `Assumptions` × measurements → hours ledger; `plan_budget` picks cuts or reinvestments |
| `report.py` | Renders `docs/M0_throughput.md`, the M-0 exit record, plus `artifacts/m0/budget.json` |

Run: `make m0` (full, ≥1000 timed runs per cell) · `make m0-quick` (100 runs, output stays under
`artifacts/m0_quick/`) · `make m0-report` (re-derive the verdict without re-measuring).
Settings live in `configs/bench/m0.yaml`.

## What is measured

- **Inference matrix:** model {m1, m2, m3} × {CPU FP32, CPU INT8, MPS FP32} × batch {1, 32} × hooks
  {off, capture, features} × tap set {default, extended}. INT8 × MPS rows are written as `n/a`: there
  are no quantised MPS kernels. Timing follows §9.4: warm-up discarded, median of ≥ 1000 runs, and
  `torch.mps.synchronize()` before every timestamp.
- **Gradient steps** (batch 128, FP32, CPU and MPS). `fwd_bwd` sizes the L1 BFA ranking;
  `train_step` sizes M-1 training.
- **Injection fixed costs** (CPU): FP32 flip+restore, INT8 unpack/repack per quantised layer, and
  the `state_dict` checksum. Each injection is one batch of 32 probes, so these sit directly beside
  the forward cost.
- **SVDD cost:** exact `OneClassSVM` (RBF) over n × d × ν, plus Nystroem(500) + `SGDOneClassSVM` at
  the largest n (§6.1 says measure, don't assume).

## Why the measurements are synthetic

Throughput does not depend on weight values, so M-0 uses random-init models, N(0, 1) inputs, and
INT8 calibrated on random batches. That keeps M-0 ahead of M-1 and the Kaggle download, as §12
requires. The exception is SVDD cost, which does depend on the data. There, `heavy` synthetic
features (Student-t df=3, per-feature scales over two decades) stand in for robust-standardised real
features, as the pessimistic case. The report says to re-check at M-4 on real features.

## Budget model

- `Plan` holds only the knobs the SPEC allows to move: seeds, repetitions, extended taps, L2 scope,
  INT8 parity, and M3's INT8 arm. Bit strata, the `bf_w`/`bf_b` full grids, and M2's INT8 arm are
  deliberately **not** fields, so no cut can reach them. `_check_invariants` asserts this.
- `Assumptions` holds every workload volume the SPEC does not pin: L1 attack count, L2 cost factor,
  epochs, hyperparameter grid size, ablation fit count, and the slack factor. They are printed in the
  report so each ledger line can be challenged.
- **Reinvestment** (baseline fits): walk §3.3 in priority order and take the fullest affordable
  variant of each item. An unaffordable item is skipped, not treated as a stop, because skipping it
  takes nothing from a higher priority. L2 has an "M2 INT8 only" variant because §10 says to run it
  there first.
- **Cuts** (baseline doesn't fit): reinvestments are already absent, then repetitions 100 → 50,
  then M3's INT8 arm. If it still doesn't fit, the verdict says so and escalates. There are no
  further cuts.
- Reported-grade work is costed on **CPU** (§11.3). Training is costed on `Assumptions.train_device`.
  The ledger is serial wall-clock, so actual calendar time is lower if CPU and MPS work overlap.

## Invariants

- The report is generated; never hand-edit `docs/M0_throughput.md`. Change code or config and re-run.
- A torch upgrade invalidates the measurements. torch is pinned, and an upgrade means re-running M-0.
- Quick-mode output never goes to `docs/`: it is flagged in the report and kept under
  `artifacts/m0_quick/`.
