# fidnn — Fault Injection Detection in Deep Neural Networks Using Multi-Layer Internal States and Support Vector Data Description

Detects bit-flip faults in a neural network's stored weights and biases (for example from Rowhammer
or fault-injection hardware) by watching the network's internal activations at several layers, and
flagging inferences that fall outside a learned model of normal behaviour (Support Vector Data
Description, SVDD).

It is the software for a master's thesis. It evaluates CIFAR-10 image classifiers (ResNet-20 and
VGG-11-BN, in FP32 and INT8) and compares multi-layer monitoring against per-layer and output-only
monitoring, reporting both detection quality and runtime/memory overhead.

> **Status:** the full pipeline is implemented and tested, from data preparation through to the
> results tables. **One measurement exists so far** — the throughput calibration in
> [`docs/M0_throughput.md`](docs/M0_throughput.md). Everything downstream needs the CIFAR-10
> training data, which requires a Kaggle account (see [Data](#data)).

## Requirements

- macOS on Apple silicon, or Linux, with Python 3.11
- [uv](https://docs.astral.sh/uv/)
- ~400 MB disk for CIFAR-10, plus ~3 GB if you run the CIFAR-10-C confounder study

## Install

```bash
git clone git@github.com:mhkmcp/R-D.git && cd R-D   # or https://github.com/mhkmcp/R-D.git
uv sync
uv run fidnn --help
```

## Commands

| Command | What it does |
|---|---|
| `fidnn bench m0 [--quick]` | Measures inference, injection and detector cost on this machine |
| `fidnn data download` / `prepare` | Downloads CIFAR-10, checks it, and creates the data splits |
| `fidnn train m1\|m2\|m3` | Trains a classifier and checks its accuracy against the published reference |
| `fidnn inject m2 [--precision int8] [--mode bf_w] [--tap-set default]` | Runs a bit-flip sweep and labels each outcome |
| `fidnn extract m2 [--tap-set extended]` | Extracts internal-layer features from clean inferences |
| `fidnn attack bfa --model m2` | Runs the BFA attacker and checks it against published results |
| `fidnn attack l2 --model m2` | Runs the detector-aware attacker, with and without the monitor |
| `fidnn fit m1` | Trains SVDD detectors on clean features and sets alarm thresholds |
| `fidnn eval m2` / `fidnn report` | Scores detectors against the faults and writes the results tables |
| `fidnn overhead m2` | Measures what monitoring costs in latency, memory and throughput |

Configuration lives in `configs/`. Outputs are written to `artifacts/` as Parquet files, each with a
JSON sidecar recording the code version, configuration, seed and device. Summary pages are written to
`docs/`.

## Data

CIFAR-10 training images come from Kaggle. Before the first download:

1. Accept the [competition rules](https://www.kaggle.com/competitions/cifar-10/data) in your browser.
2. Create an API token under Kaggle → Settings → API, and either save it as
   `~/.kaggle/kaggle.json` (`chmod 600`) or export it as `KAGGLE_API_TOKEN`.

Only `train.7z` and `trainLabels.csv` are fetched from Kaggle. Do not download `test.7z`: most of its
images are dummies and none are labelled. The labelled test set — used as the pool the faults are
probed with — comes from torchvision instead, so the classifier never trains on it.

## Running the study

```bash
uv run fidnn data download          # ~120 MB from Kaggle + ~163 MB from torchvision
uv run fidnn data prepare           # integrity checks, then the persisted splits

uv run fidnn train m2               # ResNet-20, ~30 min on Apple silicon
uv run fidnn train m3               # VGG-11-BN, ~70 min

uv run fidnn attack bfa --model m2  # checks the attacker against published flip budgets
uv run fidnn inject m2 --precision int8 --tap-set default   # the fault sweep
uv run fidnn extract m2 --precision int8                    # clean features

uv run fidnn fit m1                 # detectors + calibration (sanity model)
make reproduce                      # evaluation + headline tables
uv run fidnn overhead m2            # what the monitoring costs
```

Each step writes a summary page under `docs/` and checks its own milestone gate: training checks
accuracy against the published reference, the attacker checks its flip budget against the
literature, the sweep checks that enough faults are subtle enough to be worth detecting, and the
detectors check that measured false-positive rates match the calibrated ones. A step that fails its
gate says so rather than passing results downstream.

Timings are for an Apple M5; `fidnn bench m0` measures your own machine.

## Documentation

- [`docs/thesis/`](docs/thesis/README.md): the thesis draft, and which artifact fills each table
- [`docs/SPEC.md`](docs/SPEC.md): full technical and experimental specification
- [`docs/Dataset.md`](docs/Dataset.md): dataset choice, download, splits and licensing
- [`docs/M0_throughput.md`](docs/M0_throughput.md): measured cost of the design on this machine
- `src/fidnn/*/README.md`: contract and design notes for each package

## Development

```bash
uv run pytest -q                 # 218 tests
uv run ruff check src tests      # lint
```

## Citation and licences

CIFAR-10: Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*. CIFAR-10-C:
Hendrycks & Dietterich (2019), CC BY 4.0. The BFA attacker follows Rakin, He & Fan (2019),
*Bit-Flip Attack: Crushing Neural Network with Progressive Bit Search*, ICCV.
