# fidnn — User manual

This manual covers how to install fidnn, try the demo, run the full study, and read what it
produces. The [README](../Readme.md) is the one-page summary. [SPEC.md](SPEC.md) is the technical
specification. Each `src/fidnn/*/README.md` documents the internals of one package.

- [1. What fidnn does](#1-what-fidnn-does)
- [2. Install](#2-install)
- [3. The demo](#3-the-demo)
- [4. Running the full study](#4-running-the-full-study)
- [5. Command reference](#5-command-reference)
- [6. Where outputs go](#6-where-outputs-go)
- [7. Reading the results](#7-reading-the-results)
- [8. Current status and known issues](#8-current-status-and-known-issues)
- [9. Troubleshooting](#9-troubleshooting)
- [10. Glossary](#10-glossary)

---

## 1. What fidnn does

A neural network's knowledge is stored as millions of numbers (weights) in memory. A hardware fault
or a deliberate attack such as Rowhammer can flip a single bit of one of those numbers. Sometimes
nothing visible happens. Sometimes the network starts giving wrong answers without raising any error.

fidnn tests one idea: **watch the network's internal layers, not just its final answer**, and raise
an alarm when they stop looking normal. It:

1. trains CIFAR-10 image classifiers (M1 ResNet-8, M2 ResNet-20, M3 VGG-11-BN), in FP32 and 8-bit
   (INT8);
2. flips bits in their weights in a controlled sweep and records what each flip did;
3. learns what "normal" internal activity looks like from clean images only, using Support Vector
   Data Description (SVDD);
4. measures how many faults the detectors catch at a fixed false-alarm rate, compares them with
   detectors that only look at the final answer, and measures what the monitoring costs in time and
   memory.

---

## 2. Install

Requirements: macOS on Apple silicon or Linux, Python 3.11, [uv](https://docs.astral.sh/uv/), and
about 400 MB of disk for CIFAR-10 (plus about 3 GB for CIFAR-10-C, and about 1 GB per INT8 model for
the sweep cache).

```bash
git clone git@github.com:mhkmcp/R-D.git && cd R-D
uv sync                    # core install
uv sync --group ui         # adds Gradio, needed only for the demo
uv run fidnn --help
```

Every command below can be written as `uv run fidnn …` or `uv run python -m fidnn …`. They are the
same command.

---

## 3. The demo

The demo shows the idea to someone who has never used a terminal: pick a picture, damage the model's
memory, and see whether the answer changes and whether the detector notices. It uses plain words
("Break the model's memory", "Wrong answer, silently") instead of thesis terms.

There are two versions:

| | Local app | Shareable page |
|---|---|---|
| Runs | on your machine, live on the real model | in any browser, no install |
| Model | the actual INT8 M2 network | 60 precomputed fault scenarios |
| Use it for | exploring, live demonstrations | sending to supervisors or examiners |

### 3.1 Local app

**Before the first start**, the demo needs prepared data, a trained M2 model, clean features and
fitted INT8 detectors. If you have run the study ([§4](#4-running-the-full-study)) you already have
these. If not, start the app anyway: it opens a page listing the exact commands still missing, in
order, instead of crashing.

```bash
uv sync --group ui
uv run fidnn quantize m2   # once: saves the INT8 model, so later starts are fast
uv run fidnn ui            # opens http://127.0.0.1:7860
```

Options: `--port 8000` picks another port. `--model`, `--tap-set` and `--seed` select a different
fitted detector bundle (default `m2`, `default`, `0`). The live app is INT8 only.

The page has three columns.

**Step 1 — Pick a picture.** Click one of 24 CIFAR-10 pictures, or press **Random picture**.

**Step 2 — Damage the memory.**

| Control | Choices | Meaning |
|---|---|---|
| Where | Early / Middle / Late layers | which part of the network the flipped weights are in |
| Which bit | Sign bit — drastic … Lowest bits — subtle | which bit of each 8-bit weight is flipped |
| How many flips | 1, 4, 16 | how many weights are damaged at once |

- **Flip bits** applies that damage. Flips accumulate until you reset.
- **Surprise me** applies 4 drastic flips in a random place.
- **Reset model** removes all damage. The model returns to exactly its clean state, and the app
  checks this with a checksum.
- The list under the buttons shows each flipped weight, with its real value before and after.
- **Advanced: flip one exact bit** lets you choose the layer, weight index and bit (0 = lowest).

**Step 3 — What happened.**

- **How often may the detector cry wolf on a healthy model?** This sets the false-alarm rate: 5 %
  (sensitive), 1 % (balanced) or 0.1 % (strict). A stricter setting raises fewer false alarms and
  also misses more real faults.
- **Model's answer:** the answer before → after, the true label, and the top three guesses now.
- **Detector:** 🟢 *Looks normal* or 🔴 *Fault detected*.
- **What the fault did:**

  | On screen | Meaning | Group |
  |---|---|---|
  | No effect | answer and confidence unchanged | looks harmless from the outside |
  | Shaken, same answer | same answer, but the internal numbers moved | looks harmless from the outside |
  | Wrong answer, silently | a different answer, with no error raised | visibly harmful |
  | Broken | the output on this batch is unusable | visibly harmful |

- **How unusual each layer looks:** one bar per monitored layer. A red bar above 0 means that layer
  crossed its own alarm line. The text below the chart names the first layer to look abnormal.
- **Try this damage on 200 pictures** reports accuracy before and after, and the share flagged by
  the detector. Faults that leave the answer unchanged and faults that make it wrong are reported in
  separate rows.
- **Advanced: scores and checksum** shows the raw per-layer scores and thresholds, the fused (D2)
  score against its threshold, the output-only (MSP) score, and the model checksum next to the clean
  checksum.

Each click takes about 7–31 ms on an Apple M5. A picture is always judged together with 31 fixed
other pictures. "Broken" is defined over a batch (accuracy falls to chance), so a single picture
alone cannot be judged broken.

The app holds one shared model in memory. It is for one person on one machine. Do not expose it on
a network as a multi-user service.

### 3.2 Shareable page

The shareable page has no live model. It plays back scenarios precomputed from the real model.

```bash
uv run fidnn ui --export demo.json
```

The JSON contains:

- 24 gallery pictures;
- 60 fault scenarios (3 layer groups × 5 bit levels × 1 or 16 flips × 2 draws). Each flip shows the
  weight's 8 bits before and after, with the flipped bit marked;
- the headline detection results from `artifacts/results/`;
- the CPU overhead numbers from `artifacts/m0/`, because detection is never shown without its cost.

The output is the same for the same seed. The current page, built from this file, is
[Bit-Flip Detector Lab](https://claude.ai/artifact/HuN2XkBdVpDy97xCpt6rnX). Its second tab, "Results
so far", shows how often each detector catches a fault next to what monitoring costs. The page is
**private**: nobody else can open it until you share it from the page's Share menu. It has not yet
been checked visually at phone width or in dark mode.

### 3.3 What viewers will see with the current results

The current results come from **one seed**. The page shows a banner saying the thesis needs at least
five. With these results the inner-layer detector rarely fires. Across 60 scenarios × 24 pictures,
it flagged 1 pair at the 1 % false-alarm rate. On visibly harmful faults, the output-only detectors
catch far more: MSP about 30 %, D2 about 3.5 %. The page states this directly and does not hide it.

---

## 4. Running the full study

### 4.1 Get the data (one time)

CIFAR-10 training images come from Kaggle.

1. Accept the [competition rules](https://www.kaggle.com/competitions/cifar-10/data) in a browser.
2. Create an API token under Kaggle → Settings → API. Save it as `~/.kaggle/kaggle.json` and run
   `chmod 600` on it, or export it as `KAGGLE_API_TOKEN`.

Only `train.7z` and `trainLabels.csv` are fetched. **Never download `test.7z`**: most of its images
are dummies and none are labelled. The labelled test images come from torchvision instead, and the
classifier never trains on them.

### 4.2 The pipeline, in order

Each step needs the outputs of the steps before it. Timings are for an Apple M5. Run
`fidnn bench m0` to measure your own machine.

| # | Command | Milestone | Produces | Takes |
|---|---|---|---|---|
| 0 | `uv run fidnn bench m0` | M-0 | `artifacts/m0/`, `docs/M0_throughput.md` | `--quick` for a smoke run |
| 1 | `uv run fidnn data download` | M-1 | raw CIFAR-10 | ~120 MB + ~163 MB |
| 2 | `uv run fidnn data prepare` | M-1 | `artifacts/data/cifar10_splits.parquet` | minutes |
| 3 | `uv run fidnn train m2` | M-1 | `artifacts/models/m2_seed0.pt` | ~30 min (M3 ~70 min) |
| 4 | `uv run fidnn attack bfa --model m2` | M-1b | `artifacts/attacks/`, `docs/M1b_bfa.md` | |
| 5 | `uv run fidnn quantize m2` | — | `artifacts/models/m2_seed0_int8.pt` | |
| 6 | `uv run fidnn inject m2 --precision int8 --tap-set default` | M-2 | `artifacts/faults/`, fault features, `docs/M2_taxonomy.md` | longest step |
| 7 | `uv run fidnn extract m2 --precision int8` | M-3 | `artifacts/features/…_clean.parquet` | |
| 8 | `uv run fidnn fit m2 --precision int8` | M-4 | `artifacts/detectors/`, `docs/M4_sanity.md` | |
| 9 | `make reproduce` | M-5 | `artifacts/results/`, `docs/M5_results.md` | |
| 10 | `uv run fidnn overhead m2` | M-6 | `artifacts/overhead/` | |
| 11 | `uv run fidnn attack l2 --model m2` | M-8 | `artifacts/attacks/l2_*`, `docs/M8_adaptive.md` | |

`make reproduce` runs `fidnn eval m2 --precision int8` followed by `fidnn report`.

**Milestone gates.** Each step checks its own exit criterion and says plainly when it fails, rather
than passing bad results downstream:

- M-0: the measured cost of the planned grid fits the compute budget. Run this before any sweep.
- M-1: test accuracy is within 1 point of the published reference.
- M-1b: the BFA attacker's flip budget matches Rakin et al. (2019). Run this before generating
  faults.
- M-2: at least about 15 % of faults are "hidden" (Track S), otherwise there is nothing subtle to
  detect.
- M-4: the measured false-positive rate is within 0.3×–3× the nominal rate.

**More seeds.** Every reported number needs at least 5 seeds. Repeat steps 3 and 5–10 with
`--seed 1`, `--seed 2`, and so on. `fidnn report` aggregates over every seed it finds in
`artifacts/results/`.

**More configurations.** Use `--precision fp32`, `--tap-set extended`, `--mode` (see
[§5](#5-command-reference)) or a different model (`m1`, `m3`). FP32 sweeps run a full forward pass
by default, and INT8 sweeps reuse a clean cache (`--exec suffix`). Both give the same outcomes;
INT8 is bit-identical.

**Smoke runs.** `--epochs`, `--reps` and `--trials` shrink a run for testing. Results from smoke
runs are not for reporting.

**Reported numbers come from CPU runs.** Training uses MPS (Apple GPU) by default, and the
evaluation that produces reported numbers runs on CPU. Treat MPS timings as exploratory only.

---

## 5. Command reference

Every command accepts `--help`. Options shared by most commands:

| Option | Default | Meaning |
|---|---|---|
| `model` | — | `m1` (ResNet-8, sanity model), `m2` (ResNet-20), `m3` (VGG-11-BN) |
| `--precision` | `fp32` | `fp32` or `int8` |
| `--tap-set` | `default` | which internal layers are monitored: `default` or `extended` |
| `--seed` | `0` | random seed; one seed = one independent replicate |
| `--config` | per command | YAML file under `configs/` |
| `--out` | per command | output directory under `artifacts/` |
| `--report-only` | off | rewrite the `docs/` summary from existing outputs without rerunning |

| Command | What it does | Notable options |
|---|---|---|
| `bench m0` | measures inference, injection, gradient and SVDD cost | `--quick` (100 runs per cell), `--only inference grad injection svdd`, `--budget-hours 250` |
| `data download` | fetches Kaggle train + torchvision test | |
| `data prepare` | integrity checks, persisted splits, normalisation statistics | |
| `train <model>` | trains and checks accuracy (FP32 and INT8) | `--device mps`, `--epochs` |
| `quantize <model>` | saves the INT8 model so later commands load it instead of re-quantising | |
| `inject <model>` | bit-flip sweep and outcome labels | `--mode bf_w\|bf_b\|bf_bn\|sa0\|sa1\|rnd_val`, `--tap-set`, `--exec full\|suffix`, `--reps` |
| `extract <model>` | clean per-layer features | `--tap-set` |
| `fit <model>` | fits the D1–D3 detectors on clean data, sets thresholds, runs the sanity gate | |
| `eval <model>` | scores every detector against the fault population | |
| `report` | headline tables and hypothesis verdicts across all seeds | `--results-dir`, `--out` |
| `overhead <model>` | latency, memory and throughput for each number of monitored layers | `--tap-set extended` (default) |
| `attack bfa` | BFA attacker, checked against published flip budgets | `--model`, `--trials` |
| `attack l2` | detector-aware attacker, run with and without the monitor | `--precision`, `--tap-set` |
| `ui` | the demo ([§3](#3-the-demo)) | `--port`, `--export FILE` |

Fault modes for `inject --mode`:

| Mode | Fault |
|---|---|
| `bf_w` | bit flip in a weight (the main study) |
| `bf_b` | bit flip in a bias |
| `bf_bn` | bit flip in a batch-norm buffer |
| `sa0` / `sa1` | a bit stuck at 0 / at 1 |
| `rnd_val` | a weight replaced by a random value |

The injection grid (`configs/inject/grid.yaml`) runs 100 injections per (layer group × bit level ×
flip count) cell, with 1, 4 and 16 flips and 32 probe images per injection. Bit positions are
sampled level by level (sign, top, high, middle, low), never uniformly, so the subtle bits are not
drowned out by the drastic ones.

---

## 6. Where outputs go

Files are named by a stem: `{model}_{precision}_{tap_set or mode}_seed{seed}`, for example
`m2_int8_default_seed0`.

| Directory | Contents |
|---|---|
| `artifacts/data/` | splits and dataset provenance |
| `artifacts/models/` | checkpoints (`*.pt`), INT8 models, `manifest.json` |
| `artifacts/attacks/` | BFA and L2 attacker results |
| `artifacts/faults/` | injection plans (`_records`) and labelled outcomes (`_outcomes`) |
| `artifacts/features/` | clean features, clean outputs, fault features |
| `artifacts/detectors/` | fitted detector bundles (`.joblib`) and calibration tables |
| `artifacts/results/` | per-seed evaluation: `_headline` and `_breakdowns` |
| `artifacts/overhead/`, `artifacts/m0/` | cost measurements |
| `artifacts/cache/` | sweep cache, git-ignored, rebuilt automatically when stale |
| `docs/M*_*.md` | human-readable summary written by each milestone |

Every artifact has a **JSON sidecar** next to it that records the code version, configuration, seed,
device and input hashes. Never edit anything under `artifacts/` by hand. Rerun the command instead.

Figures for the thesis are drawn by the notebooks in `notebooks/`, which only read
`artifacts/results/*.parquet`. [docs/thesis/README.md](thesis/README.md) lists which artifact fills
each thesis table.

---

## 7. Reading the results

### Outcomes and tracks

Every (injection, probe image) pair gets one label:

| Label | Meaning | Track |
|---|---|---|
| `MASKED` | output unchanged (logits moved by less than 0.01) | **S** — hidden |
| `DEGRADED` | logits moved, same answer | **S** — hidden |
| `SDC` | silent data corruption: a different answer, no error | **H** — harmful |
| `CRASH` | output unusable, or accuracy at chance on the batch | **H** — harmful |

Track S and Track H are **always reported separately**. They answer different questions: can the
monitor see damage that the output hides (S), and does it catch damage that already matters (H)?

### Detectors

| Name | What it watches |
|---|---|
| D1 | each monitored layer separately, one SVDD per layer |
| D2 | all monitored layers fused into one SVDD |
| D3 (`_max`, `_mean`, `_fpr_weighted`) | D1's per-layer scores combined afterwards |
| D4 | deep SVDD; in the SPEC, not implemented yet |
| `msp`, `entropy`, `margin`, `energy` | the final answer only (baselines) |

All detectors are fitted on clean images only. No fault data is used for fitting, tuning or setting
thresholds.

### Thresholds and metrics

- **α** is the false-alarm rate, fixed in advance at 5 %, 1 % or 0.1 %. The threshold is the
  (1 − α) quantile of detector scores on held-out clean images (`clean_cal`).
- **TPR** is the share of faulty pairs flagged. **Measured FPR** is the share of clean images
  flagged, and should be close to α.
- `mean`, `ci_low` and `ci_high` are averages over seeds with a 95 % bootstrap confidence interval.
  `enough_seeds = False` means fewer than 5 seeds, and the number is not reportable yet.
- Detection quality is always read together with its overhead (`docs/M0_throughput.md`,
  `artifacts/overhead/`). A detector that catches more but costs ten times as much is a different
  result.

The hypothesis verdicts in `docs/M5_results.md` compare D2 with the best output-only baseline on each
track and α, using a paired bootstrap.

### Two worked examples

Both examples are real outputs of the M2 INT8 model (seed 0) with the fused detector D2 at α = 1 %
(threshold τ = −0.0789). A D2 score **above** τ raises an alarm.

**Example 1: a false alarm.** The model is clean, with no bits flipped.

| | |
|---|---|
| Picture | #101, a cat |
| Answer | cat, the correct answer |
| D2 score | −0.0769, just above τ = −0.0789 → 🔴 alarm |
| Per-layer (D1) | no single layer crossed its own alarm line |

Nothing is wrong with the model, so this alarm is false. It is the price of α = 1 %: the threshold
is set so that about 1 in 100 clean images is flagged. Across the 500 demo images on the clean model,
7 were flagged (1.4 %), which is close to the 1 % target. A score that only just crosses τ while no
single layer looks abnormal is typical of a false alarm.

**Example 2: a true detection.** Four sign bits are flipped in the late layers:

| Layer | Weight | Bit | Value before → after |
|---|---|---|---|
| `layer3.1.conv2` | #1586 | 7 (sign) | −0.1078 → +0.4034 |
| `layer3.2.conv2` | #33606 | 7 (sign) | −0.1590 → +0.7258 |
| `layer3.2.conv1` | #34703 | 7 (sign) | −0.0309 → +0.1340 |
| `fc` | #70 | 7 (sign) | −0.2703 → +1.4600 |

| | |
|---|---|
| Picture | #40, a deer |
| Answer | dog → horse: the answer changed with no error raised (`SDC`, Track H) |
| D2 score | −0.0519, well above τ = −0.0789 → 🔴 alarm |
| Per-layer (D1) | `logits` scores 96.3 against its line of 27.0; it is the first layer to look abnormal |

The model really is damaged and its answer really changed, so this alarm is correct. (The clean
model already got this picture wrong, calling it a dog; what matters for `SDC` is that the answer
*changed*.) On this damage, overall accuracy on 256 pictures stayed at 91.4 %. Only 1 of the 256
answers changed, so the fault is almost entirely hidden. The detector flagged 9.8 % of the pictures,
against 1.4 % for the clean model.

Pictures #40 and #101 are not in the demo's 24-picture gallery. To reproduce both examples, run:

```python
import numpy as np
from fidnn.demo.session import DemoSession

s = DemoSession.from_artifacts("m2", "int8", "default", 0)
ev = s.evaluate(s.batch_for(101))                      # Example 1: clean model
print(ev.d2[0], ev.taus[0.01], ev.alarm(0.01)[0])
for layer, i in [("layer3.1.conv2", 1586), ("layer3.2.conv2", 33606),
                 ("layer3.2.conv1", 34703), ("fc", 70)]:
    s.add_flip(layer, i, 7)                            # Example 2: four sign-bit flips
ev = s.evaluate(s.batch_for(40))
print(ev.labels[0], ev.d2[0], ev.alarm(0.01)[0])
s.reset()
```

In the live app, the same four flips can be applied with **Advanced: flip one exact bit**, and then
judged on any gallery picture.

---

## 8. Current status and known issues

- **Done, one seed each:** M-0 throughput, M-1 M2 training, M-1b BFA validation, M-2 INT8 `bf_w`
  sweep, M-3 features, M-4 detectors (M1 FP32 and M2 INT8), M-5 evaluation (M2 INT8), and a first M-6
  overhead run (M2 FP32, extended taps).
- **Not reportable yet:** every M-5 number comes from seed 0 only. The thesis requires at least 5.
- **Headline so far (1 seed, not reportable):** on Track H at α = 1 %, D2 catches about 3.5 % of
  faulty pairs and MSP about 30 %. On Track S, all detectors are near α. In
  [M5_results.md](M5_results.md) this currently reads as H1 not supported.
- **Bug — `precision` column in the results tables.** In
  `artifacts/results/*_headline.parquet`, the detector's classification precision (`tp / (tp + fp)`)
  and the model precision (`int8`) are both written to a single `precision` column, and the model
  precision overwrites the metric. `fidnn eval` needs a fix (for example, rename one of the two)
  before those tables go in the thesis. The demo does not use this column.
- **Shareable page:** private, and not yet checked at phone width or in dark mode.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `fidnn ui` shows "Almost there" | Run the listed commands in order, then start `fidnn ui` again |
| `fidnn ui` fails to import gradio | `uv sync --group ui` |
| `fidnn ui` takes long to start | `uv run fidnn quantize m2` once |
| Kaggle download refuses | accept the competition rules in a browser; check the token and its `chmod 600` |
| `eval`: "no fault features" | rerun `inject` with `--tap-set default` (or the tap set you are evaluating) |
| `report`: "no *_headline.parquet" | run `fidnn eval` first |
| `load` refuses a checksum or calibration mismatch | the INT8 model is stale: rerun `fidnn quantize` |
| A gate says FAIL | read the matching `docs/M*_*.md`; do not run later milestones on top of it |
| Disk filling up | `artifacts/cache/` can be deleted safely and rebuilds when needed |
| Port 7860 already in use | `uv run fidnn ui --port 7861` |

---

## 10. Glossary

| Term | Meaning |
|---|---|
| Bit flip | one stored 0 becomes 1, or the reverse, in a weight's binary representation |
| INT8 / FP32 | weights stored as 8-bit integers / 32-bit floating-point numbers |
| Tap | an observation point on an internal layer, where features are read |
| Probe | a clean test image used to judge a damaged model |
| SVDD | Support Vector Data Description: learns a boundary around normal data and scores how far outside it a new point falls |
| BFA | Bit-Flip Attack (Rakin et al., 2019): an attacker that searches for the most damaging bits |
| L2 attacker | an attacker that knows about the monitor and tries to stay under it |
| Seed | one independent repetition of training, sampling and fitting |
| Sidecar | the JSON file next to every artifact that records how it was made |
| Overhead | extra latency, memory and throughput cost of monitoring |
