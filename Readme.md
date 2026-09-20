# fidnn — Fault Injection Detection in Deep Neural Networks

Detects bit-flip faults in a neural network's stored weights and biases (for example from Rowhammer
or fault-injection hardware) by watching the network's internal activations at several layers, and
flagging inferences that fall outside a learned model of normal behaviour (Support Vector Data
Description, SVDD).

It is the software for a master's thesis. It evaluates CIFAR-10 image classifiers (ResNet-20 and
VGG-11-BN, in FP32 and INT8) and compares multi-layer monitoring against per-layer and output-only
monitoring, reporting both detection quality and runtime/memory overhead.

> **Status:** early development. The throughput benchmark, data preparation, model training and
> fault-injection sweeps are available; detection and evaluation commands are not yet implemented.

## Requirements

- macOS on Apple silicon, or Linux, with Python 3.11
- [uv](https://docs.astral.sh/uv/)
- ~1 GB disk for CIFAR-10 and CIFAR-10-C

## Install

```bash
git clone <repo-url> && cd R-D
uv sync
```

## Usage

```bash
uv run fidnn --help
```

| Command | What it does | Available |
|---|---|---|
| `fidnn bench m0 [--quick]` | Measures inference, injection and detector cost on this machine | ✅ |
| `fidnn data download` / `prepare` | Downloads CIFAR-10, checks it, and creates the data splits | ✅ |
| `fidnn train m1\|m2\|m3` | Trains a classifier and checks its accuracy against the published reference | ✅ |
| `fidnn extract` | Extracts internal-layer features from clean inferences | planned |
| `fidnn inject m2 [--precision int8] [--mode bf_w]` | Runs a bit-flip sweep and labels each outcome | ✅ |
| `fidnn attack bfa --model m2` | Runs the BFA attacker and checks it against published results | ✅ |
| `fidnn fit` / `calibrate` | Trains SVDD detectors on clean features and sets alarm thresholds | planned |
| `fidnn eval` / `report` | Evaluates detection and overhead, writes tables and figures | planned |

Configuration lives in `configs/`. Results are written to `artifacts/` as Parquet files, each with a
JSON sidecar recording the code version, configuration, seed and device.

### Benchmark example

```bash
uv run fidnn bench m0 --quick     # a few minutes; full run without --quick
```

Writes measurements to `artifacts/m0/` and a summary to `docs/M0_throughput.md`.

## Data

CIFAR-10 training images come from Kaggle. Before the first download:

1. Accept the [competition rules](https://www.kaggle.com/competitions/cifar-10/data) in your browser.
2. Create an API token under Kaggle → Settings → API, and either save it as
   `~/.kaggle/kaggle.json` (`chmod 600`) or export it as `KAGGLE_API_TOKEN`.

Then:

```bash
uv run fidnn data download   # ~120 MB from Kaggle + ~163 MB from torchvision
uv run fidnn data prepare    # checks the data and writes the splits
uv run fidnn train m2        # ResNet-20; about an hour on Apple silicon
```

The labelled test set is downloaded automatically through torchvision. More detail is in
[`docs/Dataset.md`](docs/Dataset.md) §4.

Only `train.7z` and `trainLabels.csv` are needed from Kaggle. Do not download `test.7z`: most of its
images are dummies and none are labelled.

## Documentation

- [`docs/SPEC.md`](docs/SPEC.md): full technical and experimental specification
- [`docs/Dataset.md`](docs/Dataset.md): dataset choice, download, splits and licensing

## Citation and licences

CIFAR-10: Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*. CIFAR-10-C:
Hendrycks & Dietterich (2019), CC BY 4.0.
