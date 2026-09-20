# Thesis draft — status and artifact bindings

Chapters of *Fault Injection Detection in Deep Neural Networks Using Multi-Layer Internal States
and Support Vector Data Description*. Prose lives here; **every number comes from an artifact**, and
each results table below names the file that fills it. Nothing in these chapters is typed in by
hand from a terminal — if a number has no artifact, it is not in the draft.

| # | Chapter | State | Fills from |
|---|---|---|---|
| 1 | [Introduction](01-introduction.md) | **written** | — |
| 2 | [Threat model](02-threat-model.md) | **written** | — |
| 3 | [Method](03-method.md) | **written** | — |
| 4 | [Experimental design](04-experimental-design.md) | **written**, except the measured grid cost | `docs/M0_throughput.md` ✅ |
| 5 | [Results](05-results.md) | **skeleton** — every table bound, none populated | `artifacts/results/*.parquet` ⏳ |
| 6 | [Adaptive attacker](06-adaptive-attacker.md) | **skeleton** | `artifacts/attacks/l2_*.parquet` ⏳ |
| 7 | [Threats to validity](07-threats-to-validity.md) | **written** | — |
| 8 | [Deployment note](08-deployment-note.md) | **written**, costs pending | `artifacts/overhead/*.parquet` ⏳ |
| 9 | [Contract traceability](09-contract-traceability.md) | **written** | — |
| 10 | [Dataset scope note](10-dataset-scope-note.md) | **written** | — |

✅ = artifact exists · ⏳ = milestone not yet run

## What is blocked, and by what

M-0 (throughput) is measured and its numbers are in chapter 4. Everything else waits on the CIFAR-10
training data: without it there are no trained models, so no fault population, no features, no
detectors and no results. The gates in SPEC §12 are measurements, and none of them can be reported
from anything but a real run:

| Gate | Chapter that needs it |
|---|---|
| M-1 accuracy within 1 pt of the pinned references | 4 (models), 5 (every arm) |
| M-1b BFA flip budget consistent with Rakin et al. (2019) | 5 (external anchor), 6 |
| M-2 Track S share ≥ ~15 % | 5 (the discriminating half of H1) |
| M-4 calibrated FPR within 0.3×–3× nominal | 5 (calibration honesty) |
| M-5 headline tables | 5 |
| M-6 overhead | 5 (C5 pairing), 8 |
| M-8 adaptive attacker | 6 |

## Rebuilding the numbers

```bash
make reproduce          # eval + headline tables from cached features and detectors
```

Each results section carries the command that regenerates its artifact. A chapter that cites a
number with no matching sidecar is a bug in the draft, not a formatting question.

## Conventions in these chapters

- **Tracks S and H are never merged.** Every detection claim states its track.
- **A confidence interval that includes zero is written "no detected difference"** — never as a win.
- **Absolute milliseconds first, percentage second**, with the baseline model named (SPEC §9.4).
- **Scope is stated, not implied away:** claims are about convolutional image classifiers on 32×32
  inputs at the scale evaluated here. Chapter 7 states the bounds; chapter 1 does not overreach them.
