# fidnn — Fault Injection Detection in Deep Neural Networks

Detects bit-flip faults in a neural network's stored weights and biases (for example from Rowhammer
or fault-injection hardware) by watching the network's internal activations at several layers, and
flagging inferences that fall outside a learned model of normal behaviour (Support Vector Data
Description, SVDD).

It is the software for a master's thesis. It evaluates CIFAR-10 image classifiers (ResNet-20 and
VGG-11-BN, in FP32 and INT8) and compares multi-layer monitoring against per-layer and output-only
monitoring, reporting both detection quality and runtime/memory overhead.

> **Status:** early development. Model definitions, fault injection primitives, activation taps and
> the throughput benchmark (`fidnn bench m0`) are available; training, detection and evaluation
> commands are not yet implemented.

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
| `fidnn extract` | Extracts internal-layer features from clean inferences | planned |
| `fidnn inject` | Runs bit-flip fault-injection sweeps | planned |
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

CIFAR-10 training images come from Kaggle (you need a Kaggle account and must accept the
[competition rules](https://www.kaggle.com/competitions/cifar-10/data)). The labelled test set is
downloaded automatically through torchvision. Step-by-step instructions are in
[`docs/Dataset.md`](docs/Dataset.md) §4.

Only `train.7z` and `trainLabels.csv` are needed from Kaggle. Do not download `test.7z`: most of its
images are dummies and none are labelled.

## Documentation

- [`docs/SPEC.md`](docs/SPEC.md): full technical and experimental specification
- [`docs/Dataset.md`](docs/Dataset.md): dataset choice, download, splits and licensing

## Citation and licences

CIFAR-10: Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*. CIFAR-10-C:
Hendrycks & Dietterich (2019), CC BY 4.0.
