# fidnn — agent and developer guide

Master's thesis codebase: detect bit-flip faults in CNN weights/biases from multi-layer internal
activations with SVDD. `docs/SPEC.md` is the specification; `docs/new_thesis_contract.pdf` overrides it.

## Architecture

```
                      configs/*.yaml ──────────────────────────────┐
                                                                   ▼
 Kaggle train.7z ─┐            train (40k, classifier)      fidnn.cli
 canonical test ──┼─► data ──► clean_fit / clean_cal /    data | train | extract | inject
 CIFAR-10-C ──────┘            clean_test / probe pool    fit | calibrate | eval | report | bench
                                    │                              │
                                    ▼                              │
            models (M1 ResNet-8, M2 ResNet-20, M3 VGG-11-BN; FP32 + INT8 PTQ)
                 │  Tap() identity modules at every observation point
        ┌────────┴──────────────────────┐
        ▼ clean instance                ▼ separate instance
   taps: TapMonitor hooks          inject: in-place XOR bit flips ◄── attack (L1 BFA, L2 adaptive)
   → per-tap 26-d descriptor       restore in finally, checksum         │
        │                               │                               │
        │                               ▼                               │
        │                       taps (same hooks) → fault features      │
        ▼                               │                               │
   detect: median/IQR normaliser → SVDD (D1 per-tap, D2 fused,          │
           D3 late fusion, D4 deep) → thresholds from clean_cal         │
        │                               │                               │
        └──────────────► eval ◄─────────┘◄──────────────────────────────┘
               taxonomy (MASKED/DEGRADED/SDC/CRASH → Track S / Track H),
               metrics, bootstrap CIs, overhead
                              │
                              ▼
          artifacts/{models,features,faults,detectors,results}/*.parquet + sidecar.json
                              │
                              ▼
                  notebooks/ (figures only)  →  thesis
   bench: M-0 throughput calibration across models × devices × precisions × taps
```

Each package under `src/fidnn/` has a `README.md` with its contract and invariants — read it before
editing that package.

## Commands

```bash
uv sync                                  # install (Python 3.11, torch pinned)
uv run pytest -q                         # tests
uv run ruff check src tests              # lint (line length 100)
uv run python -m fidnn bench m0 --quick  # M-0 smoke run
uv run python -m fidnn data download     # M-1: Kaggle train.7z + labels, canonical archive
uv run python -m fidnn data prepare      # M-1: integrity checks, splits, train stats
uv run python -m fidnn train m2          # M-1: train on MPS, evaluate FP32+INT8 on CPU
uv run python -m fidnn inject m2 --mode bf_w  # M-2: injection sweep + taxonomy
uv run python -m fidnn attack bfa --model m2  # M-1b: BFA validation vs published N_flip
uv run python -m fidnn extract m2             # M-3: clean per-tap features
uv run python -m fidnn fit m1                 # M-4: detectors, calibration, sanity gate
make reproduce                                # M-5: eval + headline tables
uv run python -m fidnn overhead m2            # M-6: latency/memory per K
uv run python -m fidnn attack l2 --model m2   # M-8: adaptive attacker, with vs without
uv run python -m fidnn <cmd> --help
```

## Rules

- **Before implementing anything**, find the SPEC clause for it (`/spec-check`). No clause → it is an
  addition; say so.
- **C1:** no Mahalanobis, covariance estimators, inverse covariance, ZCA or whitened PCA — anywhere.
  Hooks block the names; structural equivalents are on you. Substitutes: median/IQR, RBF SVDD, KDE,
  LOF, Isolation Forest, reconstruction error.
- **C2:** fault data never reaches any fit, hyperparameter selection, CDF mapping or threshold.
- **C3:** thresholds = (1−α) quantile of `clean_cal`, α ∈ {5 %, 1 %, 0.1 %} fixed in advance.
- **C4/C5:** every reported number has ≥ 5 seeds + 95 % CI, and overhead is reported with it.
- Track S (`MASKED`∪`DEGRADED`) and Track H (`SDC`∪`CRASH`) are never merged.
- Bit positions are sampled by stratum, never uniformly. Bit strata are never cut for compute.
- Injection mutates in place, restores in `finally`, checksums `state_dict()` (buffers included).
  The clean-feature model instance is never injected.
- Reported numbers come from CPU runs; MPS is exploratory only. Every artifact gets a sidecar JSON.
- Analysis logic in `src/`; notebooks only plot from `artifacts/results/*.parquet`.
- Never edit `artifacts/**` by hand, never edit the contract PDF; SPEC/Dataset edits need approval.
- Never download Kaggle `test.7z`; never read `~/.kaggle/` or `.env*`.
- Check the milestone gate before long runs (`/milestone-gate`): M-0 before any sweep, M-1b before
  generating faults, M-2 Track S ≥ ~15 % before H1 work.

## Code style

- Comments minimal: one-line docstrings, no restating the code. Design rationale goes in the package
  `README.md`, not in docstrings.
- Where a docstring is needed, point to the parent/library method instead of re-explaining it
  (e.g. "See `nn.Module.register_forward_hook`").
- Cite the SPEC section (`SPEC §4.4`) instead of paraphrasing it.
- New package → add its `README.md` (for developers/agents) in the same change. Root `Readme.md` is
  for users only.

## Claude tooling

- Skills: `/spec-check`, `/c1-audit`, `/fault-injection`, `/detector-pipeline`, `/cifar-data`,
  `/milestone-gate`, `/report-results`.
- Agents: `spec-auditor` (after code changes), `results-reviewer` (before results go in the thesis),
  `experiment-runner` (long runs), `examiner` (defence prep).
- SPEC digests: `.claude/references/` — load these instead of the full SPEC.
