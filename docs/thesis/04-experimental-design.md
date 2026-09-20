# 4. Experimental design

## 4.1 Models

| ID | Model | Precision | Role |
|---|---|---|---|
| M1 | ResNet-8, 2-class subset | FP32 | pipeline smoke test — **never cited as a result** |
| M2 | ResNet-20 (CIFAR variant) | FP32 + INT8 | primary: full grid, all ablations, all baselines |
| M3 | VGG-11-BN (plain, no skip connections) | FP32 + INT8 | primary: architecture contrast |

M2 and M3 consume identical inputs and differ in whether skip connections exist, which is what makes
the H4 propagation comparison a statement about topology rather than about two unrelated networks.

INT8 post-training quantisation is included for two reasons. A flip in an 8-bit weight has a bounded
and qualitatively different effect from a flip in a floating-point exponent, and the detector must be
shown to work in both regimes. And the attack literature this study anchors against — BFA,
DeepHammer, TBT, ProFlip — operates on 8-bit networks, so the INT8 arm is where flip budgets are
comparable to published ones. Quantised kernels run on CPU only, which costs nothing for correctness
because CPU is already the reporting device, but INT8 and FP32 overhead are tabulated separately
because the kernels differ.

Each model's clean accuracy must be within one point of its pinned published reference before any
fault experiment runs.

## 4.2 Data and splits

CIFAR-10, assembled from two sources so that the fault probes come from a pool the classifier never
trained on:

| Split | Source | n |
|---|---|---|
| `train` | stratified 80 % of the Kaggle training images | 40,000 |
| `clean_fit` | 60 % of the remaining 10,000 | 6,000 |
| `clean_cal` | 20 % of the remaining 10,000 | 2,000 |
| `clean_test` | 20 % of the remaining 10,000 | 2,000 |
| fault probe pool | the canonical labelled test set | 10,000 |

No detector record — clean or fault — comes from an image the classifier was trained on. An earlier
revision of this design carved the clean detector splits from the classifier's own training images,
which would have fitted the detector on memorised activations and tested it on unseen ones: an
input-distribution difference between fitting and evaluation, introduced by the experimental setup
rather than by any fault. Holding 10,000 images out of training removes it, at the cost of a
classifier trained on 40k rather than 50k and a calibration set capped at 6,000.

Two integrity checks run as failing tests: zero pixel-hash collisions between the training source and
the probe pool, and label agreement between the two sources on a sampled subset.

## 4.3 Fault grid

Injections are stratified rather than uniform, because sampling bit positions uniformly makes the
problem artificially easy — exponent-MSB flips dominate and are trivially detectable.

- **Bit strata.** FP32: sign, exponent MSBs, remaining exponent, high mantissa, low mantissa. INT8:
  five strata over the eight positions with sign and MSB separated. **Never traded away for compute.**
- **Layer buckets.** Every injectable layer, reported in early / middle / late thirds.
- **Flip budgets.** 1, 4 and 16 flips per injection, bracketing the budgets reported in the attack
  literature.
- **Repetitions.** 100 injections per (bucket × stratum × budget) cell, 32 probe inputs each. The
  statistical unit is the (injection, input) pair.
- **Fault modes.** Full grid for weight and bias flips — the contract names them together, so neither
  is a reduced arm. Stuck-at, whole-value corruption, BatchNorm-buffer flips and transient activation
  flips run a reduced grid and are reported separately.

BatchNorm buffers deserve a note: they are not parameters, but they are equally resident in writable
memory and equally corruptible, and a flip that drives a running variance negative produces NaN
through the square root. Those land in `CRASH`, and they are tagged separately so they cannot flatter
Track H by inflating a category that a NaN check would catch for free.

## 4.4 Measured cost of the design

The grid was sized against measurement, not assumption. On the reporting machine — Apple M5, 10
cores, 16 GB, macOS 26.6.2, PyTorch 2.14.0 — the throughput calibration confirms the grid as
specified: the unmodified workload costs **≈89 hours** with slack against a 250-hour budget, so no
repetition, arm or model is cut and every bit stratum runs
([`M0_throughput.md`](../M0_throughput.md)).

Two measurements from that calibration shape the chapters that follow:

- **Feature extraction, not the forward pass, dominates monitored inference.** On ResNet-20 at batch 1,
  monitoring with the six default taps costs 2.24 ms against a 1.34 ms bare forward pass in FP32
  (+67 %), and against 1.14 ms in INT8 (+96 %). The INT8 forward is faster, so the floating-point
  descriptor arithmetic is a proportionally larger share. This is the number H3's design target is
  read against, and it says plainly that the unoptimised descriptor at K = 6 does not meet a 10 %
  budget.
- **Exact SVDD fitting is affordable at this scale.** The slowest exact fit at n = 6,000 is 0.41 s at
  377 dimensions, so the approximate kernel path is not needed for cost, and the approximation gap it
  would have required measuring does not arise.

## 4.5 Evaluation protocol

Fault records are split by injection **instance**, not by grid axis: 30 % development, 70 % held out
for the headline, with a reserved subset of configurations (sign stratum, budget 4) never sampled
into development and reported as its own row for unseen-configuration generalisation.

A leakage guard runs before any results build and fails it on: an instance in both splits, a reserved
configuration appearing in development, a grid cell missing from the held-out split, a fault record in
a clean split, or a pixel-hash collision between data sources.

Headline metrics are TPR at 1 % and 0.1 % calibrated FPR, per track, with precision, recall, F1,
AUROC and AUPR beside them, and the measured false-positive rate on clean test data — a large gap
between measured and nominal is a calibration failure and is reported as one. Every number carries at
least five seeds and a 95 % confidence interval; variant comparisons use a paired bootstrap over the
shared evaluation set and report the interval of the *difference*. Breakdowns per outcome category,
bit stratum, layer bucket, flip budget, fault mode, precision and attacker level are mandatory:
an aggregate-only table would hide exactly the structure this design was built to expose.
