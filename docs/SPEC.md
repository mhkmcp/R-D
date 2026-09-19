# SPEC — Fault Injection Detection in Deep Neural Networks Using Multi-Layer Internal States and Support Vector Data Description

**Status:** Draft v2.0 · **Date:** 2026-09-19 · **Derived from:** `new_thesis_contract.pdf`
**Dataset companion:** [`Dataset.md`](Dataset.md) v2.0 — dataset rationale, acquisition, licensing,
alternatives considered and revision history live there. This document keeps only the data
requirements the experiments depend on.

This document is the engineering and experimental specification for the thesis. It fixes the
threat model, the feature extraction contract, the detector formulation, the evaluation protocol,
and the deliverables, so that results are reproducible and the contract's claims are testable.

**Scope.** One dataset (CIFAR-10), two contrasting architectures (M2 residual, M3 plain) on identical
inputs, two precisions (FP32, INT8). The contract requires variation across *architectures* and
*fault models*, not datasets, so this satisfies it as written; the resulting bound on generality is
stated in §1 and §13.

---

## 0. Hard constraints

| # | Constraint | Rationale |
|---|---|---|
| **C0** | **`new_thesis_contract.pdf` governs.** Where this spec and the contract conflict, the contract wins and the spec is amended — not the other way round. Anything here that the contract does not require is an addition, and is marked as such so it can be cut without touching a commitment. | The contract is the agreed scope; this document only makes it executable. |
| C1 | **Mahalanobis distance is excluded from the entire project** — not as a detector, not as a baseline, not as a feature-space metric, not as a whitening step. | Explicit project requirement. See §6.4 for the substitutions used instead. |
| C2 | The detector is trained on **clean executions only**. No fault-injected sample may touch detector fitting or threshold selection. | One-class premise of the contract; prevents leakage of attack knowledge. |
| C3 | Decision thresholds are calibrated on a **clean validation split**, at a target false-positive rate fixed *a priori*. | Prevents the common overstatement of picking the threshold that maximises F1 on the test set. |
| C4 | Every reported number carries a seed count and a dispersion estimate (§9.5). | Single-run anomaly-detection numbers are not trustworthy. |
| C5 | Detection quality and **runtime/memory overhead** are reported together, never separately. | The contract requires overhead as a first-class result. |

### C1 in practice — banned constructs

The following must not appear in code, experiments, or the thesis text as a method:

- `scipy.spatial.distance.mahalanobis`, `sklearn.covariance.EllipticEnvelope`,
  `sklearn.covariance.MinCovDet`, `EmpiricalCovariance.mahalanobis`, any explicit
  `(x-μ)ᵀ Σ⁻¹ (x-μ)`, its Cholesky/pseudo-inverse variants, or shrinkage estimators feeding one.
- The Lee et al. (2018) "Mahalanobis OOD score" baseline and its class-conditional variants.
- Covariance/ZCA whitening of features (a full-covariance whitening followed by Euclidean
  distance is Mahalanobis distance under another name).

Permitted and used instead: per-feature robust standardisation (median/IQR), RBF-kernel SVDD,
kernel density estimation, Local Outlier Factor, Isolation Forest, and reconstruction error.
A short paragraph in the thesis states this exclusion and the substitutions, so a reader does not
assume the omission was an oversight.

> **Note on CIFAR-10 preprocessing.** The standard CIFAR-10 recipe is per-channel mean/std
> normalisation, which is *diagonal* scaling and therefore permitted. Some older CIFAR pipelines
> (notably pre-2015 ones) apply **GCN + ZCA whitening** — that is a full-covariance whitening and is
> a direct C1 violation. The input pipeline uses per-channel normalisation only, and §11.4 asserts it.

### C0 in practice — what is an addition

The contract is silent on these, so they are additions rather than commitments. They are listed here
so the cut order under schedule pressure (§13) is obvious and no commitment is ever cut first:

| Addition | Where | Cut cost |
|---|---|---|
| Adaptive attacker (L2) | §2, §10 | None to the contract; a planned §3.3 reinvestment, so cutting it forgoes a planned gain |
| `sa0`/`sa1`, `rnd_val` fault modes | §4.1 | None; `bf_w`/`bf_b` carry the contract's fault model |
| `bf_act` transient activation faults | §4.1 | None; the contract scopes faults to weights and biases |
| INT8 quantised arm | §3, §9.6(4) | **Effectively uncuttable.** The bit-flip literature attacks 8-bit quantised models, so INT8 is the project's *only* external anchor (§3.1). Cutting it leaves every number in the thesis internal |
| Deep SVDD (D4) | §6.2 | None; already marked an extension |
| Dataset and architectures (§3) | §3.1 | The contract names neither, requiring only "across architectures and fault models" |
| C1 (Mahalanobis exclusion) | §0 | Not the contract's; a separate standing project requirement |

Everything else in this document traces to a clause of the contract. §14 requires a traceability
table in the thesis making that mapping explicit.

---

## 1. Problem statement and hypotheses

An attacker flips bits in the stored parameters (weights, biases) of a deployed DNN. The effect of
a flip depends on which parameter, which bit position, and which layer is hit. Some flips change
the prediction dramatically; others perturb internal activations while leaving the output label
intact until they compound. Output-only monitors see only the last category too late, or not at all.

The thesis tests whether **internal states observed at several layers** separate clean execution
from fault-injected execution better than the output alone.

| ID | Hypothesis | Rejected if |
|---|---|---|
| **H1** | Multi-layer internal state features fed to an SVDD detect fault-injected inferences at a fixed low FPR better than output-only scores. | TPR@1%FPR of the fused detector does not exceed the best output-only baseline, paired bootstrap 95 % CI of the difference including 0. **Evaluated separately on Track S and Track H (§4.3)**; Track S is the discriminating test. |
| **H2** | Fusing K > 1 layers outperforms the best single layer. | Fused TPR@1%FPR ≤ best per-layer TPR@1%FPR within CI. |
| **H3** | A minimal configuration (K ≤ 3 layers, ≤ 32 features/layer) retains ≥ 95 % of the full-monitoring TPR@1%FPR. | Retention below 95 % on both primary architectures (M2 and M3). |
| **H4** | Per-layer SVDD scores localise the earliest layer where the anomaly is observable, and that layer correlates with the injected fault's layer index. | Spearman ρ between injected layer index and first-alarm layer index is not significantly > 0. |

H4 is the "how it propagates" direction from the contract; H3 is the "minimal monitoring
configuration" secondary objective.

**Scope of the hypotheses — stated here, not deferred to a caveat.** H1–H4 are claims about
**convolutional image classifiers on 32×32 inputs** (M2 residual, M3 plain; §3). The evidence varies
architecture, fault mode, bit position, layer and precision — the variation the contract requires —
but **not modality**: no audio, tabular or sequence model is evaluated (§3.3). Every statement of
these hypotheses in the thesis carries that scope in its own wording, rather than relying on a
threats-to-validity section to walk back an over-broad claim made earlier. §13 records the bound.

**Rejection criteria track the contract, not a stricter private bar.** The contract asks for *"a
measurable advantage"* (H1) and configurations that *"preserve high detection accuracy"* (H3), so
both are decided by statistical significance and by accuracy retention — not by a margin threshold or
a latency budget the contract never sets. Two quantities remain as **pre-registered design targets**,
reported prominently but never used to declare a hypothesis failed:

- a ≥ 5-point TPR@1%FPR gain for H1, the margin at which the advantage is practically meaningful;
- < 10 % added inference latency for the H3 minimal configuration (§9.4), the budget at which the
  monitor is deployable on the §2 edge target.

A result that is statistically solid but misses a target is a *confirmed hypothesis with a
deployment caveat*, and must be written up that way rather than as a failure.

---

## 2. Threat model

**Attacker goal.** Cause silent data corruption (SDC): wrong predictions without an obvious crash.

**Attacker capability.** Can flip a bounded number of bits in the parameter memory of a deployed
model — the Rowhammer / laser-injection / bus-glitch style adversary. Flips are assumed
**persistent** in the primary setting (the corrupted weight stays corrupted for subsequent
inferences) and **transient** in a secondary setting (single-inference corruption).

**Deployment setting.** Two concrete settings are assumed, and they are the same two the bit-flip
literature demonstrates against:

- **Embedded / edge vision inference** — an on-device image classifier on a smart camera, drone, or
  ADAS-class SoC. Physical fault injection (laser, clock/voltage glitching) is realistic rather than
  hypothetical here, and it is what makes the overhead budget in §9.4 binding: a monitor that only
  fits on a datacentre GPU does not defend this threat model.
- **Shared-tenancy commodity hardware** — the DeepHammer setting, where a co-located process flips
  DRAM bits in a victim model's parameter pages via Rowhammer. No physical access required.

Overhead is therefore measured on edge-class hardware (§9.4) by design, not by accident of available
equipment.

**Attacker knowledge.** Three levels, all evaluated:
- *L0 blind* — random parameters, random bit positions.
- *L1 model-aware* — white-box access to weights and gradients; chooses flips by a progressive
  bit-search ranking (BFA-style) to maximise loss with a minimal flip budget.
- *L2 detector-aware (adaptive)* — additionally knows the monitored layers and the detector, and
  searches for flips that cause misclassification while keeping every monitored SVDD score below
  its threshold. Analysed in §10.

**Out of scope.** Faults in the inference engine's control flow, in input data, in the detector's
own parameters; software supply-chain compromise; training-time backdoors. These are named as
non-goals in the thesis so the contribution boundary is explicit.

**Defender capability.** Read access to activations at chosen layers at inference time; a clean
calibration dataset; a fixed compute budget for monitoring.

---

## 3. Target models and dataset

**One dataset — CIFAR-10**, handled per §3.2. Acquisition, provenance, licensing and the selection
rationale live in [`Dataset.md`](Dataset.md).

| ID | Dataset | Model | Precision | Role |
|---|---|---|---|---|
| M1 | CIFAR-10, 2-class subset | ResNet-8 | FP32 | Pipeline smoke test — produces **no** thesis result |
| **M2** | **CIFAR-10, 10-class** | **ResNet-20** (CIFAR variant; He et al. 2016) | FP32 + INT8 PTQ | **Primary** — full §4.2 grid, all ablations, all baselines, literature comparison |
| **M3** | **CIFAR-10, 10-class** | **VGG-11-BN** (plain, no skip connections) | FP32 + INT8 PTQ | **Primary** — architecture contrast, plain vs residual, identical inputs |

M2 and M3 are mandatory and carry the thesis; both run the full §4.2 grid in both precisions. M1
exercises the pipeline end to end and is never cited as a result. Baseline clean accuracy for each
model is recorded in `artifacts/models/manifest.json` and must be within 1 point of the published
reference before any fault experiment runs.

**M2 vs M3 is the contract's architecture variation.** ResNet-20 and VGG-11-BN consume identical
inputs and differ only in whether skip connections exist, so the H4 propagation comparison isolates
topology.

INT8 post-training quantisation is included for two reasons. Bit flips in an 8-bit weight have a
bounded, qualitatively different effect from a flip in an FP32 exponent, and the detector must be
shown to work in both regimes. And **BFA, DeepHammer, TBT and ProFlip all attack 8-bit quantised
networks**, so the INT8 arm is where this thesis's flip budgets are comparable to published numbers —
the project's only external anchor (§0, C0 table).

**Constraint:** PyTorch quantized kernels run on CPU backends (qnnpack / x86), **not** on MPS. Since
§11.3 already fixes CPU as the reporting device this costs nothing for correctness, but INT8 and FP32
overhead figures are tabulated separately (§9.4) because the kernel implementations differ.

### 3.1 Dataset rationale

What the design requires of the dataset:

- **≥ 5 usable taps per model** (§5.1) — H2/H3/H4 are hypotheses about layers.
- **Channel-structured activations** — every tap is a conv tap, so all 29 §5.2 features are populated.
- **A severity-graded clean-input shift suite** for §9.6(8) — CIFAR-10-C.
- **Comparability to the bit-flip attack literature** — BFA and DeepHammer report ResNet-20 and
  VGG-class networks on CIFAR-10, so the L1 attacker can be validated against published curves (M-1b)
  before it generates the fault population.
- **Few enough classes to keep Track S (§4.3) populated** — more classes pack decision boundaries
  closer and empty it. The M-2 gate (§12) verifies this rather than assuming it.

Why CIFAR-10 meets these and the alternatives do not, and the standard objections (small and
saturated, 32×32 unrealistic, one dataset only), are answered in Dataset.md §9–§11.

**Division of labour.**

| Job | M2 ResNet-20 | M3 VGG-11-BN |
|---|---|---|
| Full §4.2 injection grid, all fault modes | ✅ FP32 + INT8 | ✅ FP32 + INT8 |
| All ablations and baselines | ✅ | Where the ablation is architecture-sensitive |
| §9.6(8) confounder evidence (CIFAR-10-C) | ✅ | ✅ |
| Architecture contrast (H4), inputs held constant | ✅ residual | ✅ plain |
| Feature-block contribution (§9.6 item 3) | ✅ by Block-D ablation | ✅ |
| Comparison to BFA / DeepHammer / TBT | ✅ (INT8 arm) | ✅ (INT8 arm) |
| Deployment narrative | Embedded vision / Rowhammer | Embedded vision / Rowhammer |

### 3.2 CIFAR-10 data handling

**Sources.**

| Role | Source |
|---|---|
| Training data | Kaggle `train.7z` + `trainLabels.csv` — 50,000 labelled images |
| Labelled held-out pool | Canonical `torchvision.datasets.CIFAR10(train=False)` — 10,000 labelled test images |
| Confounder set | CIFAR-10-C — 15 corruption types × 5 severities over the same 10,000 images |

The Kaggle `test.7z` is **not used**: 290,000 of its 300,000 images are dummies and none are labelled,
so it cannot supply the per-probe ground truth §4.3 needs (Dataset.md §3.1). A one-shot Kaggle
leaderboard submission may be made at M-1 as an independent accuracy check; it produces no thesis
number.

**Integrity checks (required tests, §11.4).** Zero pixel-hash collisions between the Kaggle training
images and the canonical test images — an overlap would put probe images into the classifier's
training set or `clean_fit`, a C2 violation; if it fails, use the canonical distribution for both. `trainLabels.csv` agrees with the
canonical training labels on a sampled subset.

**Splits.** The §7 partition is carved as:

| Split | Source |
|---|---|
| `train` (40,000) | Stratified 80 % of the 50,000 Kaggle training images — **classifier training only**, never a detector record |
| `clean_fit` (6,000) / `clean_cal` (2,000) / `clean_test` (2,000) | Stratified 60/20/20 split of the remaining 10,000 Kaggle images, which the classifier never sees |
| Fault probe pool | The **canonical labelled 10,000-image test set** — disjoint from all of the above |

All splits are seeded and persisted. Every detector record — clean or fault — comes from an image the
classifier was not trained on, so detection cannot be attributed to a seen-versus-unseen activation
shift (§7). *(Amended at M-1: the earlier table carved `clean_*` from the classifier's own training
images, which would have fitted the detector on memorised activations and tested it on unseen ones.)*

**Preprocessing.**

- Per-channel mean/std normalisation with the standard CIFAR-10 constants
  (mean ≈ `(0.4914, 0.4822, 0.4465)`, std ≈ `(0.2470, 0.2435, 0.2616)`), **fitted on `train`
  only** and frozen. Per C1, **no GCN/ZCA whitening** (§0).
- Training augmentation: random crop 32×32 with 4-pixel padding, random horizontal flip. Augmentation
  is **training only** — clean feature extraction for the detector uses unaugmented images, so the
  SVDD models the deployed inference distribution rather than the training distribution.
- INT8 PTQ calibration uses a subsample of `clean_fit` only.

**Accuracy targets (M-1 gate, §12).** ResNet-20 on CIFAR-10: **91.25 %** (He et al. 2016, 8.75 %
error). VGG-11-BN on CIFAR-10: **≈ 92 %**; the exact reference figure must be pinned to a specific
cited source in `manifest.json` before M-1 is signed off, since VGG-on-CIFAR numbers vary across
reimplementations in a way the ResNet figure does not.

**Confounder set.** CIFAR-10-C (Hendrycks & Dietterich 2019) is the §9.6(8) evidence. Corruptions are
applied to *clean* inputs only; they are never combined with fault injection.

### 3.3 What the single-dataset design costs, and what it frees

The history of the change is in Dataset.md §10. What remains binding here is which clauses stand in
for the withdrawn arms, and how the freed compute is spent.

| Lost | Substitute |
|---|---|
| Modality generality | **None** — stated as a bound in §1 and §13 |
| Dense-tap contrast | Block-D downgrade on conv taps (§9.6(3)) |
| Independent confounder corroboration | Per-type and per-family CIFAR-10-C reporting (§9.6(8)) |
| Second external accuracy reference | One-shot Kaggle leaderboard submission at M-1 (§3.2) |

**Reinvestment priority** — reinvested in depth on M2/M3, not banked:

1. **More seeds** (§9.5) — tightens every confidence interval, including H1's paired bootstrap.
2. **The extended tap sets** on both models (§5.1) — more range for H2 and H3.
3. **The L2 adaptive attacker** (§10).
4. **Full INT8 parity**, so every FP32 result has an INT8 counterpart.

This is a plan, not a guarantee: M-0 (§12) measures throughput and sizes how much of it the schedule
can absorb.

---

## 4. Fault injection

### 4.1 Fault models

| Code | Fault | Parameters |
|---|---|---|
| `bf_w` | Single/multi bit flip in a weight | layer, tensor index, bit position, count |
| `bf_b` | Single/multi bit flip in a bias | same |
| `sa0`/`sa1` | Stuck-at-0 / stuck-at-1 on a parameter bit | layer, index, bit |
| `rnd_val` | Whole-parameter random value corruption | layer, index |
| `bf_act` *(secondary)* | Transient bit flip in an activation tensor | layer, element, bit |

**`bf_w` and `bf_b` are co-primary**, because the contract names them together in both places it
describes the fault model — *"adversarial bit flips in model weights or biases"* and *"bit flips in
weights and biases"*. Biases are not a reduced arm here even though bias tensors are far smaller than
weight tensors. `sa0`/`sa1` and `rnd_val` are supersets the contract permits but does not require.
`bf_act` covers the transient-fault case and is reported separately, not mixed into the main tables.

> **BatchNorm note, new for the CIFAR arms.** ResNet-20 and VGG-11-BN carry BatchNorm affine
> parameters (`weight`, `bias`) and *buffers* (`running_mean`, `running_var`). The affine parameters
> are in scope for `bf_w`/`bf_b` like any other. The buffers are **not parameters** but are equally
> resident in writable memory and equally corruptible — a flip in `running_var` is a realistic and
> nastily effective fault. They are injected under a distinct tag `bf_bn` within the `bf_w` family,
> and results are broken out separately in §9.1, because merging them into the weight numbers would
> conflate two mechanisms. A flip in `running_var` that drives it negative produces NaN through the
> `sqrt`, which lands in `CRASH` and must not be allowed to flatter Track H.

### 4.2 Injection grid

The grid below is sized for laptop-class compute (§9.4). Where a quantity was reduced from the
statistically ideal value, the reduction and its cost are recorded rather than silently applied.

- **Bit position.** FP32: stratified over sign (31), exponent MSBs (30–27), remaining exponent
  (26–23), high mantissa (22–16), low mantissa (15–0). INT8: all 8 positions, sign/MSB separated.
  Stratification is mandatory — sampling bit positions uniformly makes the problem artificially easy,
  because exponent-MSB flips dominate and are trivially detectable. **Bit strata are never traded
  away for compute** (see §12 verification); they are what keeps §9.1 honest.
- **Layer.** Every injectable layer, bucketed into early / middle / late thirds for reporting. For
  ResNet-20 the buckets align with the three stages (16/32/64 channels), which makes the H4
  propagation story readable; for VGG-11-BN they align with the five conv blocks folded into thirds.
- **Flip budget.** N ∈ {1, 4, 16} per inference for L0/L1 — the range of the original
  {1, 2, 4, 8, 16} with the interpolating points dropped. This range brackets the budgets reported in
  the BFA/DeepHammer literature, which is what makes the §9.1 per-budget breakdown comparable.
- **Targeting.** L0 uniform random; L1 gradient-ranked progressive bit search (BFA).
- **Repetitions.** 100 independent injections per (model, layer bucket, bit stratum, budget) cell.
- **Probe samples per injection.** 32. The detection unit is per-inference, so the statistical sample
  is the *(injection, input)* pair — at equal cost, many injections × few inputs each covers fault
  diversity better than few injections × many inputs, because fault configuration is the dimension
  results are stratified over in §9.1.
- **Fault-mode coverage.** Full grid for **`bf_w` and `bf_b`** (§4.1 — the contract names weights and
  biases together, so neither is a reduced arm). `sa0`/`sa1`, `rnd_val`, `bf_act` and `bf_bn` run a
  reduced grid: late layer bucket, 2 bit strata (exponent-MSB and low-mantissa, i.e. the obvious and
  the subtle extremes), budgets {1, 16}. Note that a bias tensor holds far fewer parameters than a
  weight tensor, so the `bf_b` full grid re-samples a smaller target pool — record the per-layer bias
  parameter count alongside the results, since coverage of that pool is much denser than for weights
  and that changes how the two arms compare.
- **Model coverage.** M2 and M3 both run the **full grid in both FP32 and INT8**.

Net: ≈ 4,500 injections × 32 probes ≈ 144k forward passes per model / fault mode / precision / seed,
and two full-grid fault modes per model rather than one.

**Compute caveat — the live budget risk.** ResNet-20 costs ≈41 MFLOP/image and M2/M3 each run the
full grid in two precisions; the §3.3 reinvestments consume any slack. **M-0 (§12) must
re-measure throughput before any sweep is launched**, and if the budget is short the cut order is:
reinvestments from §3.3 first (extended taps, extra seeds — they are gains, not commitments), then
repetitions 100 → 50, then the INT8 arm of M3 — **never M2's INT8 arm**, which is the project's only
external anchor (§0). Bit strata are never cut.

**Track S population check.** ResNet-20's 270k parameters give enough redundancy that faults should
skew less catastrophic and Track S should be well populated. This is an expectation, not a
measurement. If the combined `MASKED` + `DEGRADED` share falls below ~15 %, escalate to ResNet-56 and
re-run. Check this before generating the full sweep, not after: without a populated Track S, H1's
discriminating test cannot be run at all.

### 4.3 Fault outcome taxonomy — required before any detection metric

Each injected run is labelled by its *effect*, not just by the fact that a fault occurred:

| Label | Definition |
|---|---|
| `MASKED` | Top-1 prediction unchanged on the evaluation input, logits shift < ε |
| `DEGRADED` | Prediction unchanged but confidence/logit margin changes ≥ ε |
| `SDC` | Top-1 prediction changed, no numerical failure |
| `CRASH` | NaN/Inf in output, or accuracy collapse to chance |

**Detection metrics are computed over two co-primary tracks, never merged into one "fault" class:**

| Track | Categories | What it measures | Why it is primary |
|---|---|---|---|
| **Track S — subtle** | `MASKED` ∪ `DEGRADED` | Faults that did **not** change the top-1 prediction | The contract's motivating case: *"others induce smaller internal deviations that remain unnoticed"* and *"enable earlier fault detection, before effects are clearly visible at the output"*. Output-only monitors are near-blind here, so this is where multi-layer monitoring must prove its advantage. |
| **Track H — harm-relevant** | `SDC` ∪ `CRASH` | Faults that produced a wrong or broken output | Establishes that the monitor catches faults that actually cause damage. |

**Both tracks are reported for every headline result, and H1's comparison against the output-only
baselines (§6.4) is evaluated separately on each.** Running H1 on Track H alone would be a weak test
of the contract's claim: output-only scores do well on faults that already changed the output, so the
comparison that actually discriminates the hypothesis is the Track S one.

Conflating the four categories into a single "fault" class is the most common way this class of
result is inflated, and the thesis must not do it. Within Track H, `CRASH` is additionally broken out
because a NaN detector is trivial and would otherwise flatter the headline numbers — and, per §4.1,
BatchNorm `running_var` flips are a new and prolific source of NaN in the CIFAR arms specifically.
Within Track S, `MASKED` and `DEGRADED` are broken out because they differ in how much internal
signal is available: `DEGRADED` has a measurable logit-margin shift, `MASKED` by definition does not.

On Track S a flagged fault is not a false positive — a fault really is present — but the deployment
reading differs from Track H: a Track S alarm is an early warning that a corrupted parameter exists
before it has caused visible harm, which is precisely the operational value the contract claims for
internal-state monitoring. Track S detection rates are therefore reported as detections, not as
false alarms, in every table.

### 4.4 Implementation

Bit manipulation via `tensor.view(torch.int32)` (FP32) or `torch.int8` and XOR with a bit mask.

**Injection is performed in place and XOR-ed back** after the probe batch completes — the parameter
tensor is never deep-copied per injection. XOR is its own inverse (already a required test in
§11.4), so the restore is exact and free. On unified memory a per-injection `state_dict` copy would
dominate the entire sweep; with thousands of injections this is the difference between minutes and
hours. The restore runs in a `finally` block so an exception mid-probe cannot leave the model
corrupted for the next injection, and a periodic parameter checksum (every N injections) asserts the
model has returned to its clean state — silent restore failure would contaminate every subsequent
result and is otherwise very hard to notice.

**The checksum must cover buffers, not only parameters** (§4.1): `bf_bn` writes to
`running_mean`/`running_var`, which `model.parameters()` does not enumerate. Checksum over
`state_dict()` instead, or the one fault mode most likely to corrupt state permanently is the one the
guard does not watch.

Every injection is recorded as a reproducible record
`{model, precision, seed, layer, flat_index, bit, count, mode}` written to
`artifacts/faults/*.parquet`, so any run can be replayed exactly. No fault is ever applied to the
model instance used to generate clean training features.

---

## 5. Internal-state feature extraction

### 5.1 Tap points

Taps are registered with `module.register_forward_hook(hook, always_call=True)`
(`always_call=True` so a fault that raises inside a later module does not silently drop earlier
observations). Tap set per model is declared in config, defaulting to:

- post-activation output of each block group (ResNet stage; VGG conv block),
- the pooled embedding preceding the classifier,
- the pre-softmax logits.

| Model | Default taps | Extended taps |
|---|---|---|
| M1 ResNet-8 | 6 | — |
| **M2 ResNet-20** | **6** (stem, stage1–3, pooled, logits) | up to 13 (all 9 residual blocks) |
| **M3 VGG-11-BN** | **7** (5 conv blocks, pooled, logits) | up to 10 (all 8 conv layers) |

Every model must clear the ≥ 5 usable taps required by §3.1 before it enters the study. A tap registry
maps each tap to a stable string ID so configurations are comparable across models.

**Every tap is a 2-D conv tap** (or the pooled/logit vectors derived from one), so §5.2's dense
variant has no model exercising it.

The extended tap sets give the H3 K-sweep (§9.6(1)) somewhere to search, and are planned for both
models (§3.3). Headline results use the default sets unless the sweep shows otherwise, and the set
used is recorded with every number.

### 5.2 Per-tap feature vector (the monitoring contract)

For a tap producing activation `A ∈ R^{N×C×H×W}` (2-D conv — ResNet, VGG), `R^{N×C×T}` (1-D conv) or
`R^{N×D}` (dense), the monitor computes a fixed-length descriptor **per sample**. All statistics are
single-pass and reduction-only — no tensor is copied to host memory, which is what keeps overhead
bounded.

The contract is specified for all three activation shapes and stays that way: it is what makes the
descriptor architecture-independent, and what lets a later study add a non-conv model without
redesigning the monitor. **Only the 2-D conv path is exercised by a model.**

**Block A — distributional shape (9 features)**
mean, std, min, max, and quantiles q01, q25, q50, q75, q99 of the activation values.

**Block B — sparsity and saturation (4)**
fraction of exact zeros (ReLU dead ratio), fraction above the clean 99.9th percentile
(saturation ratio), count of NaN, count of Inf.

**Block C — energy (5)**
L1/√n, L2/√n, L∞, ratio L∞/L2 (peakiness), and an energy Gini coefficient computed over **channels**
for conv taps and over **units** for dense taps — the same split Block D makes, so no feature in this
block is undefined for a dense tap.

**Block D — structure.** Two variants, selected by tap rank, not by model:

- **D-conv (8) — any tap with a channel axis**, including 1-D convolutional taps: Shannon entropy of
  the normalised channel-energy distribution; top-1, top-2, top-4 channel energy shares; mean and std
  of per-channel spatial means; mean and std of per-channel spatial stds.
- **D-dense (4) — dense/MLP taps**: Shannon entropy of the normalised unit-activation energy
  distribution; top-1, top-2, top-4 unit energy shares. The four spatial statistics have no analogue
  and are zero-padded. **No model produces a dense tap** — the variant is retained in the
  contract for shape-completeness and tested on synthetic tensors (§11.4), not by a model.

**Block E — cross-layer (3, computed at fusion time)**
ratio of this tap's L2 to the previous tap's L2; ratio of its entropy to the previous tap's; and
the difference of its saturation ratio from the previous tap's.

Total: **29 features per conv tap, 22 per dense tap** (9+4+5+4), fixed-width and
architecture-independent, with zero-padding keeping widths comparable across taps. Every tap is a
conv tap, so **29 is the operative width throughout**.

Whether the method depends on convolutional structure is tested by ablation: §9.6 item 3 **zeroes
D-conv's channel-structure features on M2/M3 taps** and re-measures.

**A note on ResNet taps specifically.** Tapping the *output* of a residual block observes
`F(x) + x`, which means the skip path can carry a clean signal that partially masks a fault injected
inside `F`. That is not a flaw in the tap placement — it is the mechanism H4 exists to measure, and it
is exactly why the plain-vs-residual contrast against VGG-11-BN is informative. The tap registry
additionally records, for each ResNet tap, whether it sits pre- or post-addition, so the H4 analysis
can distinguish the two.

**Design rationale.** Descriptors are deliberately permutation-invariant over channels and spatial
positions. This makes a monitor trained on one input distribution transfer across batch shapes and
keeps the feature dimension independent of layer width, which is what makes the "minimal
configuration" question (H3) answerable at all.

### 5.3 Normalisation

Per-feature **robust standardisation**: `z = (x − median) / (IQR + δ)`, with median and IQR
estimated on the clean training split only and frozen thereafter. Features with IQR below a floor
are dropped at fit time and the drop list is persisted.

Per C1, no covariance-based whitening is applied. The RBF kernel's `gamma` absorbs the remaining
isotropic scale; feature-wise scale differences are handled by the standardisation above.

---

## 6. Detectors

### 6.1 SVDD formulation

SVDD fits the minimum-volume hypersphere (centre `a`, radius `R`) enclosing the clean training
features in a kernel-induced space:

```
min_{R,a,ξ}  R² + (1/(νn)) Σ ξ_i
s.t.         ‖φ(x_i) − a‖² ≤ R² + ξ_i ,  ξ_i ≥ 0
```

`ν ∈ (0,1]` upper-bounds the fraction of training points outside the sphere and lower-bounds the
support-vector fraction. For the RBF kernel (`‖φ(x)‖` constant) SVDD and the ν-one-class SVM are
equivalent, so the implementation uses `sklearn.svm.OneClassSVM(kernel="rbf", nu=…, gamma=…)`, with
the SVDD interpretation stated in the thesis. Anomaly score:

```
s(x) = −decision_function(x)        # > 0 ⇒ outside the calibrated boundary
```

For training sets above ~50 k samples, `make_pipeline(Nystroem(gamma=…, n_components=…),
SGDOneClassSVM(nu=…))` is used; the approximation gap versus exact `OneClassSVM` is measured on
**M2** and reported once, not silently assumed to be zero. It is measured on M2 rather than M1
because it is a reported number, and M1 produces no thesis result (§3).

> Whether this path is needed at all is a measurement, not an assumption. `clean_fit` on CIFAR-10 is
> 6,000 samples — small enough that exact `OneClassSVM` should fit (M-0 measured it) — but the extended 13-tap fused
> feature vector (13 × 29 ≈ 377 dimensions) is wide, and §3.3 makes the extended set the planned
> configuration rather than a sweep-only one. Measure before assuming either way.

### 6.2 Detector variants (all four are built; they are the study's independent variable)

| ID | Variant | Description |
|---|---|---|
| **D1** | Single-layer SVDD | One SVDD on one tap. Run for every tap — gives the per-layer baseline for H2 and the propagation map for H4. |
| **D2** | Early-fusion SVDD | Concatenate normalised features from all monitored taps → one SVDD. The contract's "joint representation". |
| **D3** | Late-fusion of per-layer SVDDs | Per-tap SVDDs; combine scores by (a) max, (b) mean, (c) clean-FPR-weighted mean, (d) a one-class combiner over the score vector. Scores are made comparable by mapping each through its own clean-validation empirical CDF before combination. |
| **D4** | Deep SVDD *(extension)* | Small MLP encoder trained with the one-class objective on the fused features. Mandatory safeguards against hypersphere collapse: no bias terms, unbounded activations, no learnable output scaling; centre `c` fixed from an initial forward pass. Attempted only if D2/D3 leave headroom. |

D3 is the variant that yields localisation: the first tap index whose calibrated score exceeds
threshold is the "first-alarm layer" used in H4.

### 6.3 Hyperparameters and their selection

`ν ∈ {0.001, 0.005, 0.01, 0.05, 0.1}`; `gamma ∈ {"scale"} ∪ {k · γ_median}` for
`k ∈ {0.1, 0.25, 0.5, 1, 2, 4}`, where `γ_median = 1 / median‖x_i − x_j‖²` estimated on a clean
subsample (median heuristic). **Selection uses clean data only**: choose the pair minimising
clean-validation FPR subject to a stability criterion (support-vector fraction within a target
band), then verify on a clean held-out split. Selecting `ν`/`gamma` by test-set AUROC is a C2
violation and is not permitted.

### 6.4 Baselines

Output-only:
- Maximum softmax probability, softmax entropy, logit margin, energy score.

Feature-space, same features as D2 (so the comparison isolates the detector, not the features):
- Isolation Forest, Local Outlier Factor (novelty mode), Gaussian KDE, PCA-reconstruction error,
  and a shallow autoencoder reconstruction error.

Systems baselines (context, not competitors):
- Parameter checksum / hash over weights — perfect recall on persistent faults, zero coverage on
  transient activation faults, and a fixed per-inference or per-epoch cost. Included to position
  internal-state monitoring honestly: it is for the cases a checksum cannot cover (transient faults,
  faults outside checksum scope, and situations where re-hashing every inference is too expensive).

Attack-side reference — the project's only external anchor:
- The **L1 (BFA) attacker is validated against published results** before it is used to generate
  faults: its flip-budget-versus-accuracy curve on INT8 ResNet-20 / CIFAR-10 is compared to the
  literature and the comparison is reported as a figure. A reimplementation that collapses the model
  in wildly fewer or wildly more flips than published is a reimplementation bug, and catching that
  before the detection sweep is the whole value of having a comparable setting.

**Excluded by C1:** the Mahalanobis OOD score and `EllipticEnvelope` / `MinCovDet`
(robust-covariance). KDE, LOF and Isolation Forest fill the density-baseline role instead.

**C1 needs its most active guarding here.** The Lee et al. (2018) class-conditional Mahalanobis score
is *the* canonical OOD/anomaly baseline on CIFAR-10: a vision-literate examiner will ask for it by
name, and every tutorial implementation a contributor might copy includes it. The thesis paragraph
required by §14(4) must pre-empt the question rather than answer it defensively, and the §11.4
grep-based guard against banned constructs is not optional.

---

## 7. Data splits and leakage control

| Split | Content | Used for |
|---|---|---|
| `clean_fit` | 60 % of the 10,000 classifier-held-out Kaggle images (6,000) | Fitting normaliser + SVDD |
| `clean_cal` | 20 % (2,000) | Threshold calibration, score CDF mapping, hyperparameter selection |
| `clean_test` | 20 % (2,000) | FPR measurement |
| `fault_dev` | Random 30 % of injection **instances**, drawn across the full §4.2 grid, excluding the reserved configurations below | Development, sanity checks only |
| `fault_test` | The disjoint remaining 70 % of instances, spanning the **full grid** — all 3 layer buckets, all 5 bit strata, all budgets, all fault modes — plus all L1/L2 attacks | Headline results |
| `fault_gen` | A tagged subset **of `fault_test`**: the configurations deliberately never sampled into `fault_dev` (bit stratum {sign}, budget {4}) | Unseen-configuration generalisation, reported separately |

**The dev/test split is over injection instances, not over grid axes.** An earlier design held out
whole axes — middle layer bucket, specific strata — which made `fault_test` a narrow slice and left
the per-layer-bucket and per-stratum breakdowns required by §9.1 impossible to compute. It also
conflicted with the contract's *"varied across layers and bit positions to cover both obvious and
subtle error patterns"*: headline numbers must be measured over that full variation, not over one
third of it.

Generalisation to unseen fault configurations is a real claim, so it keeps its own evidence —
`fault_gen` — rather than being bought at the cost of the headline's coverage. It is reported as a
separate row, never merged into the headline number.

Input samples underlying clean and fault records are all held out from classifier training (§3.2):
`clean_*` from the 10,000 Kaggle images outside `train`, fault probes from the canonical test set, for
every model in the study — so detection cannot be attributed to a seen-versus-unseen input shift.
Clean inference on the probe pool itself is also recorded, for the §4.3 taxonomy.

An automated leakage check runs before every results build and fails it on any of:
- a fault record ID appearing in any fitting or calibration set;
- a `fault_test` instance ID appearing in `fault_dev`;
- a `fault_gen` configuration tuple appearing anywhere in `fault_dev`;
- `fault_test` missing any layer bucket, bit stratum or budget present in the §4.2 grid — the guard
  that stops the narrow-slice failure from recurring silently;
- **any pixel-hash collision between the Kaggle CIFAR-10 training images and the canonical CIFAR-10
  test images** (§3.2) — the guard specific to this dataset's two-source assembly.

---

## 8. Calibration

Thresholds come from `clean_cal` only: `τ_α` = the (1 − α) empirical quantile of clean scores, for
α ∈ {5 %, 1 %, 0.1 %}. Per-tap thresholds for D3 use the same procedure per tap. A finite-sample
correction (upper binomial confidence bound on the quantile) is applied at α = 0.1 %, since a naive
empirical quantile at that level on a few thousand samples is noisy.

Two operating regimes are reported throughout:
- **Per-inference** — one decision per forward pass.
- **Windowed** — alarm if ≥ m of the last n inferences exceed τ (m-of-n). This matches the
  persistent-fault threat model, where the attacker's corruption affects every subsequent inference,
  and it trades detection latency for a far lower effective FPR. Reported as (n, m) ∈ {(1,1),
  (8,3), (32,8)}.

---

## 9. Evaluation

### 9.1 Detection metrics

Headline: **TPR at FPR = 1 %** and **TPR at FPR = 0.1 %** (thresholds from §8), reported for
**Track S and Track H separately** (§4.3). Neither track is aggregated into the other.

The contract enumerates the quantities to report — *"detection rate, false-positive rate, precision,
recall, F1-score, and ROC-AUC, alongside runtime and memory overhead"*. All eight appear, mapped as:

| Contract term | Here |
|---|---|
| detection rate | TPR at the calibrated threshold (per track) |
| false-positive rate | measured FPR on `clean_test`, which must land near nominal α — a large gap is a calibration failure and is reported as such |
| precision / recall / F1-score | at the calibrated threshold, per track |
| ROC-AUC | AUROC, with AUPR alongside it because the fault/clean ratio is not balanced |
| runtime and memory overhead | §9.4 |

Every result is broken out *per fault category* (`MASKED` / `DEGRADED` / `SDC` / `CRASH`), *per bit
stratum*, *per layer bucket*, *per flip budget*, *per fault mode* (with `bf_bn` separated from `bf_w`,
§4.1), *per precision* (FP32 / INT8), and *per attacker level* (L0/L1/L2). Aggregate-only tables are
not acceptable: they hide the results that matter, namely performance on subtle, low-bit-position,
single-flip faults, which is the regime the contract is about.

**One externally-anchored table.** For M2 INT8 under the L1 attacker, report the
attacker-side quantities the bit-flip literature reports — flips required to drive top-1 accuracy to
chance, and accuracy-versus-flip-budget — alongside the published figures for the same architecture
and dataset. This is the thesis's only cross-paper comparison and it must be presented as a
*sanity check on the attack implementation*, not as a claim about the detector.

### 9.2 Localisation metrics (H4)

Spearman ρ between injected layer index and first-alarm tap index; confusion matrix of injected
layer bucket vs first-alarm bucket; mean "propagation depth" = number of taps between injection and
first alarm.

**Reported separately for M2 (residual) and M3 (plain), and the difference between them is the
result.** Skip connections give a fault an alternate propagation path, so ρ and mean propagation
depth should differ measurably between the two. Because M2 and M3 consume identical inputs (§3), any
difference is attributable to topology rather than to representation. The pre-/post-addition tap
flag (§5.2) is used to check whether the
residual model's first alarms concentrate on pre-addition taps.

### 9.3 Latency metrics

Detection latency in inferences (windowed mode) and the number of inferences before the first SDC —
i.e. does the monitor fire *before* the fault causes a visible wrong answer? This is the contract's
"earlier fault detection" claim and needs its own figure, not a sentence.

### 9.4 Overhead metrics

| Metric | How measured |
|---|---|
| Added latency (ms and %) | Median over ≥ 1000 timed runs, warm-up discarded, batch ∈ {1, 32}, on **Apple MPS and laptop CPU**, `torch.mps.synchronize()` before each timestamp |
| Peak memory (MB) | `torch.mps.current_allocated_memory()` and `tracemalloc` delta, monitor on vs off |
| Detector size (KB) | Serialised support vectors + normaliser |
| Throughput (samples/s) | Sustained, monitor on vs off |
| Feature-extraction share | Hook cost vs scoring cost, split out |

Measured on one fixed machine, specified in the thesis: chip, memory, macOS version, torch version.
MPS requires `torch.mps.synchronize()` in place of CUDA event synchronisation, and
`torch.cuda.max_memory_allocated` has no MPS equivalent — use the two sources above. Batch = 1 is the
operating point that matters for the §2 deployment setting; batch = 32 is reported for context only.

**Edge-class hardware is the right measurement platform here, not a limitation to apologise for.**
The threat model (§2) assumes embedded vision inference, so an overhead budget measured on a
laptop-class CPU and integrated GPU is closer to the deployment reality than a datacentre-GPU number
would be. State this once, in these terms, and do not hedge it elsewhere.

**Percentage overhead is a ratio, and the denominator is the model.** Monitoring cost is roughly
proportional to activation volume, while the baseline it is divided by is the model's forward cost —
so the same monitor reports a smaller percentage on a heavier model. That is arithmetic, not an
improvement, and it means overhead percentages from earlier revisions of this spec (measured against
a much cheaper model) are **not comparable** to v2.0's. Report absolute milliseconds first and the
percentage second, and state the baseline model with every percentage. H3's < 10 % design target
(§1) is read against M2 specifically.

The **INT8 arm** uses different kernels from the FP32 arm, so its latency and memory figures are
tabulated separately with the reason given inline — an unexplained discontinuity between the two would
read as a measurement error.

Overhead is reported per detector variant and per K (number of taps), since that is the trade-off
H3 turns on.

### 9.5 Statistical protocol

5 seeds minimum per configuration (model init where applicable, fault sampling, detector fit).
Report mean ± 95 % CI. Comparisons between detector variants use a **paired bootstrap** (2000
resamples) over the shared evaluation set, reporting the CI of the difference. AUROC differences use
DeLong or bootstrap CIs. A difference whose CI includes 0 is described as "no detected difference",
never as a win.

### 9.6 Ablations

1. **K sweep** — 1 → all taps, greedy forward selection by clean-validation criteria, reporting the
   TPR/overhead Pareto front. This is the direct answer to H3. Run over the **extended** tap sets
   (§5.1) so the search has range: 13 taps on M2, 10 on M3.
2. **Tap position** — early-only vs middle-only vs late-only vs logits-only. On M2 these map onto the
   three ResNet stages, which makes the result readable as a statement about network depth.
3. **Structure ablation** — drop each of Blocks A–E in turn, on both M2 and M3, and report which
   blocks carry signal. **Block D gets a second, sharper treatment**, because it is where the
   "does this method need convolutional structure?" question lives: run a **D-conv → D-dense
   downgrade**, zeroing the four per-channel spatial statistics so conv taps carry the same 22-feature
   descriptor a dense tap would (§5.2), and re-measure TPR@1%FPR. A large drop says the method depends
   on channel-spatial structure and would transfer poorly to dense architectures; a small one says the
   descriptor's power is in Blocks A–C and the method is more portable than it looks. It varies the
   feature set alone. Report it on both models — agreement between a residual and a plain network
   strengthens whichever way it lands.
4. **Precision** — FP32 vs INT8, on M2 and M3. A headline-adjacent result rather than a
   side-ablation, since INT8 is the regime the attack literature operates in (§3) and the only arm
   carrying external comparability.
5. **Training budget** — clean fitting samples ∈ {500, 1k, 2k, 6k}; establishes the minimum
   calibration data a deployer needs. Upper bound is 6k because `clean_fit` is 60 % of the 10,000
   images held out from classifier training (§3.2).
6. **Kernel sensitivity** — RBF vs polynomial vs linear SVDD.
7. **Transfer** — detector fitted on M2 applied to M3 taps (expected to fail; documents that the
   monitor is model-specific, which is a deployment cost worth stating). Because M2 and M3 share
   inputs, a failure isolates architecture-specificity from input-distribution-specificity.
8. **Input-shift confounder** — score clean-but-shifted inputs to quantify how much of the detector's
   signal is "fault" versus "anything unusual". A monitor that fires equally on both is a novelty
   detector, not a fault detector, and the thesis must say so if that is what the data shows.

   **Source: CIFAR-10-C** (§3.2), on both M2 and M3. It is the only evidence for the thesis's central
   caveat (§3.3), so two reporting requirements are mandatory:

   - **Report per corruption type, never only the aggregate.** A monitor that fires on
     `gaussian_noise` but not on `fog` is telling you which §5.2 blocks are doing the work, and
     collapsing to one number discards the internal replication.
   - **Report by corruption family** — noise, blur, weather, digital — and state whether the
     detector's response is consistent across families or driven by one. A response confined to a
     single family is a materially different finding from a uniform one, and only the family
     breakdown distinguishes them.

   Severity is the axis that carries the argument: a fault detector's score should rise sharply with
   injected faults and weakly with corruption severity. Report the two slopes side by side.

---

## 10. Adaptive attacker (L2)

An evaluation whose attacker is unaware of the defence overstates the defence. L2 formulates flip
selection as a constrained search: maximise task loss subject to every monitored SVDD score staying
below its threshold, solved greedily over candidate (parameter, bit) pairs ranked by loss gain and
filtered by monitor score. Reported outcome: the attacker's success rate and required flip budget
with and without the monitor present. An increase in required budget is a real, quantifiable
security gain even when the detector is evadable — that framing is the honest one and should be the
thesis's conclusion for this section.

**The dataset choice strengthens this section without extra work.** Because L1 is a validated BFA reimplementation
(§6.4) with a flip budget comparable to published figures, "the monitor raises the required budget
from X to Y flips" is now a statement in units a reader of that literature already has intuition for.
Run L2 on M2 INT8 first for this reason.

---

## 11. Implementation

### 11.1 Stack

Python 3.11 · PyTorch 2.x + torchvision (CIFAR-10 loading, canonical test split) · scikit-learn 1.7 ·
NumPy · pandas/pyarrow · `py7zr` or system `7z` for the Kaggle archives · Hydra or plain YAML configs
· pytest. No dependency on a fault-injection
framework is assumed; injection is ~200 lines of local code (§4.4) and stays auditable.

### 11.2 Layout

```
R-D/
├── docs/SPEC.md
├── configs/
│   ├── data/cifar10.yaml
│   ├── model/{resnet8,resnet20,vgg11bn}.yaml
│   ├── faults/{bf_w,bf_b,bf_bn,sa,rnd_val,bf_act}.yaml
│   ├── taps/<model>.yaml            # tap registry per model, default + extended sets
│   └── detector/{d1_single,d2_fused,d3_late,d4_deepsvdd}.yaml
├── src/fidnn/
│   ├── data/          # cifar pipeline only: kaggle 7z + canonical test, normalise, CIFAR-10-C
│   ├── models/        # builders, checkpoints, quantisation
│   ├── inject/        # bit manipulation, injection planner, replay
│   ├── taps/          # forward hooks, feature blocks A–E (D-conv/D-dense), registry
│   ├── detect/        # normaliser, svdd, fusion, calibration, baselines
│   ├── attack/        # L1 BFA progressive bit search, L2 detector-aware search
│   ├── eval/          # metrics, taxonomy labelling, bootstrap, overhead
│   └── cli.py         # extract | inject | fit | calibrate | eval | report
├── tests/             # incl. bit-flip correctness, leakage guard, determinism, C1 grep guard
├── artifacts/         # models/, features/, faults/, detectors/, results/ (parquet + json)
└── notebooks/         # figure generation only, no analysis logic
```

Analysis logic lives in `src/`, never in notebooks; notebooks read `artifacts/results/*.parquet` and
emit figures. This keeps every number in the thesis traceable to a committed code path.

### 11.3 Reproducibility

Every artifact carries a sidecar JSON with: git commit, config hash, seed, library versions, **and
the device it was produced on**. `make reproduce` regenerates all headline tables from cached
features.

**Device policy.** `torch.use_deterministic_algorithms(True)` has coverage gaps on MPS. All reported
experiments and every §9.4 overhead measurement therefore run on **CPU**, where determinism holds and
these models are small enough for it to be affordable; MPS is used for exploratory sweeps only. The
device field in the sidecar is what distinguishes the two, so a reported number produced on MPS is
detectable after the fact rather than silently mixed in. Where non-determinism remains, it is
recorded and absorbed into the seed variance (§9.5).

**Data provenance.** The sidecar additionally records, for CIFAR-10 runs, the SHA-256 of
`train.7z` and `trainLabels.csv`, the torchvision CIFAR-10 archive hash, and the result of the
pixel-hash disjointness check (§3.2). Assembling a dataset from two sources is a reproducibility
liability unless it is recorded, and this is how it is recorded.

### 11.4 Required tests

- Bit-flip round-trip: flipping bit *b* twice restores the original tensor, bit-exactly, for FP32
  and INT8.
- Known-value test: flipping FP32 bit 30 of a known float yields the expected value.
- **In-place restore (§4.4):** after a simulated exception mid-probe, the model's `state_dict`
  checksum — **parameters and buffers** — equals its clean checksum. This guards the one failure mode
  that would silently contaminate every subsequent injection rather than producing a visible error.
- **CIFAR-10 source disjointness (§3.2):** zero pixel-hash collisions between the Kaggle training
  images and the canonical labelled test images. A failing test, not a comment.
- **CIFAR-10 label agreement:** for a sampled subset, `trainLabels.csv` agrees with the canonical
  training labels for the same images — catches a mis-ordered or mis-indexed 7z extraction, which is
  otherwise silent and corrupts everything downstream.
- **No ZCA (C1, §0):** the input pipeline applies per-channel normalisation only; assert no
  full-covariance transform is present.
- **C1 grep guard:** the banned constructs listed in §0 do not appear anywhere in `src/`. A test, so
  it fails CI rather than relying on review.
- Hook coverage: every configured tap fires exactly once per forward pass; removing hooks restores
  baseline latency to within noise.
- **Tap budget:** every model in §3 exposes ≥ 5 usable taps, asserted from the tap registry.
- **Feature-block population:** D-conv is non-zero for all conv taps (ResNet, VGG) — an all-zero block
  means a silently misapplied variant. **D-dense is tested on synthetic `R^{N×D}` tensors**, not on a
  model: no model produces a dense tap (§5.2), but the code path stays live because §9.6(3)'s
  Block-D downgrade exercises the same 22-feature width, and an untested path would make that
  ablation's result untrustworthy.
- Leakage guard (§7) as a failing test, not a comment — including the assertions that the CIFAR-10
  channel statistics were fit on `train` only and that `train` is disjoint from every detector split.
- Feature determinism: identical input ⇒ identical feature vector across runs.
- Overhead harness sanity: monitor-off path is statistically indistinguishable from the unmodified
  model.

---

## 12. Milestones

| # | Milestone | Exit criterion |
|---|---|---|
| **M-0** | **Throughput calibration** | Measured samples/s for M1/M2/M3 on MPS and CPU, FP32 and INT8, hooks on and off; §4.2 grid sizes confirmed or revised **before** any sweep is launched, and the §3.3 reinvestment plan sized against the measurement |
| M-1 | Models trained, accuracy verified | `manifest.json` within 1 pt of reference: ResNet-20 CIFAR-10 **91.25 %**, VGG-11-BN CIFAR-10 **≈92 %** (exact reference pinned in manifest). CIFAR-10 source-disjointness and label-agreement tests passing. **Optional one-shot Kaggle leaderboard submission** taken as an independent check (§3.2) — now the project's only external accuracy cross-check (§3.3) |
| M-1b | **L1 attacker validated** | BFA reimplementation's flip-budget-vs-accuracy curve on INT8 ResNet-20 / CIFAR-10 is consistent with published figures; discrepancy explained or the implementation fixed. Gate on this *before* generating the fault population. |
| M-2 | Injection engine + taxonomy labelling | Bit-flip, buffer-checksum and in-place-restore tests pass; taxonomy distribution plotted for M2 and M3; **Track S (`MASKED` + `DEGRADED`) share ≥ ~15 %** — a hard gate, since H1's discriminating test runs on that track |
| M-3 | Tap/feature pipeline | Features extracted for M1–M3, default and extended tap sets; tap-budget, block-population and determinism tests pass |
| M-4 | D1 + D2 on M1 — **sanity gate, produces no thesis number** | Calibrated FPR within 0.3×–3× of nominal; pipeline yields a plausible TPR@1%FPR on both tracks. First *reported* detector numbers come from M-5 on M2. |
| M-5 | Full evaluation on M2/M3 (FP32 + INT8), all baselines | Headline tables generated by `make reproduce` |
| M-6 | Overhead study | Pareto front (TPR vs latency) for the K sweep, on CPU and MPS, FP32 and INT8 |
| M-7 | Ablations + localisation (H4) | §9.6 items 1–6 and 8 complete, including CIFAR-10-C **per type and per family** (§9.6(8)), the Block-D downgrade (§9.6(3)), and the M2/M3 residual-vs-plain propagation comparison |
| M-8 | Adaptive attacker (L2) | Success-rate/budget table with and without monitor, in flip-budget units comparable to §9.1's external table |
| M-9 | Writing, threats to validity | Draft complete |

**M-0 is a gate, not a formality.** Every grid size in §4.2 was originally sized against a much
cheaper model, and M2/M3 each run two precisions. Launching the full sweep before re-measuring risks
discovering mid-thesis that the budget was wrong by an order of magnitude. In v2.0 M-0 has a second
job: sizing how much of the §3.3 reinvestment (extra seeds, extended taps, L2) the freed budget
actually buys — that is a measurement, not an assumption.

**M-1b is a gate too.** External comparability is the whole justification for this dataset (§3.1),
and in a single-dataset design it is the *only* external anchor. An unvalidated BFA reimplementation
delivers none of it, and validating it after the detection sweep means re-running the sweep.

M-1…M-5 are the minimum viable thesis: they settle H1 and H2. M-6/M-7 settle H3, M-7 settles H4,
M-8 is what raises the work above a purely empirical detection study.

---

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Faults are trivially detectable (exponent-MSB flips dominate) | Results look strong but say nothing | Bit-strata stratification (§4.2) and per-stratum reporting (§9.1) are mandatory, not optional |
| Detector is really a generic novelty detector | Claim of *fault* detection unsupported — **the thesis's central caveat** | §9.6(8) measures it directly on CIFAR-10-C. The per-type and per-family breakdowns are **mandatory, not presentational** (§3.3). Report the finding either way |
| INT8 faults too subtle to detect | H1 fails in the quantised regime — **and this is now the comparability arm** | Report FP32/INT8 separately; a negative INT8 result is a legitimate, publishable finding, but it would cost the §3.1 rationale, so it must not be discovered late (§9.6 item 4 runs at M-5, not M-7) |
| SVDD does not scale to fused high-dimensional features | Fitting time blows up | Nystroem + `SGDOneClassSVM` path (§6.1), with the approximation gap measured. Extended 13-tap fusion is ~377-d, and §3.3 makes the extended set the *planned* configuration rather than a sweep-only one, so this path is likelier to be exercised than it was |
| Overhead exceeds any plausible deployment budget | H3 fails | K sweep finds the Pareto front; report the minimum viable K honestly even if it is "all taps" |
| Persistent-fault setting makes windowed detection near-trivial | Overstated results | Per-inference numbers are reported alongside windowed ones, always |
| **§4.2 grid does not fit — two models × two precisions × full grid** | Schedule overrun, or a silently truncated grid | M-0 gate re-measures before any sweep. Cut order: §3.3 reinvestments first (they are gains, not commitments), then repetitions 100 → 50, then M3's INT8 arm — **never M2's**. **Bit strata are never cut** — they keep §9.1 honest |
| **BFA reimplementation is wrong** | The external comparison is worthless, and the fault population it generates is unrepresentative. **This is the only external anchor the project has** | M-1b gate validates the flip-budget curve against published figures *before* the detection sweep runs |
| **Reviewer demands the Mahalanobis OOD baseline** | C1 reads as an oversight on the dataset where that baseline is canonical | §6.4: pre-empt in the §14(4) methods paragraph rather than answering defensively; §11.4 grep guard prevents accidental reintroduction |
| **CIFAR-10 read as a toy / saturated benchmark** | Perceived lack of ambition | Dataset.md §9, objection 1: the object of study is the detector, not the classifier, and saturation sharpens the M-1 gate. No claim is made about ImageNet-scale networks |
| **No evidence the method generalises beyond vision CNNs** | **H1–H4 are bounded to one modality**, and nothing in the design bounds it further | **Stated, never implied away.** The hypotheses in §1 are worded as claims about vision CNNs on 32×32 inputs; the thesis says plainly that audio, tabular and sequence models are untested here. §9.6(3)'s Block-D downgrade gives *indirect* evidence about dependence on convolutional structure. Indirect is not the same as a second modality, and must not be written as if it were |
| **No evidence the method scales beyond 32×32 / ~300k-parameter models** | Generality of H1–H4 further bounded, on a second axis | Stated as a limit, not implied away. Nothing in this design bounds the *scale* claim and no such claim is made |
| **"One dataset" read as insufficient rigour** | Reviewer objection to the whole design | The contract requires variation across *architectures and fault models*, and M2/M3 × 4 fault modes × 5 bit strata × 2 precisions delivers it. The choice is argued in Dataset.md §9–§10 and its cost accounted in §3.3 — a decision on the record, not an omission |
| **Both models fail M-1 for the same systematic reason** | A pipeline error affecting M2 and M3 equally passes the accuracy gate unnoticed | Take the one-shot Kaggle leaderboard submission at M-1 (§3.2) — it is the only remaining independent accuracy check, and it is cheap |
| **Freed budget is banked rather than reinvested** | The single-dataset design's main compensating benefit never materialises; the thesis is simply smaller | §3.3 fixes the reinvestment priority *before* the budget appears, and M-0 sizes it. Reinvestment is planned work, not spare capacity |
| **CIFAR-10 train/test near-duplicates (~3.3 %, Barz & Denzler 2020)** | Probe pool marginally less independent of `clean_fit` than the split table implies; not a C2 violation, since no record is in two splits, but the pixel-hash guard catches exact duplicates only | Stated as a dataset-level limit, not implied away (`Dataset.md` §5.2). If the effect is ever suspected of mattering, re-run the headline with the published near-duplicate list removed from the probe pool |
| Schedule slip | Scope must shrink | Cut in the C0 order (§0): §3.3 reinvestments first (extended taps, extra seeds — planned gains, not commitments), then the adaptive attacker (§10), then `sa`/`rnd_val`/`bf_act`, then D4 — before any §9.6 ablation. `bf_w`/`bf_b` coverage, both §4.3 tracks, and M2/M3 are contract commitments and are never cut. **The INT8 arm of M2 is effectively uncuttable** (§0): it is the project's only external anchor |

---

## 14. Deliverables

1. Reproducible codebase per §11, with tests and `make reproduce`.
2. Result artifacts (parquet + JSON) for every table and figure in the thesis.
3. Thesis chapters: threat model, method, evaluation, adaptive-attacker analysis, threats to validity.
4. A methods paragraph stating the C1 exclusion and the substituted baselines — written to pre-empt
   the Mahalanobis question on CIFAR-10, where that baseline is canonical (§6.4).
5. A short deployment note: which layers to monitor, what it costs, and what it does not cover
   (transient input faults, control-flow faults, detector-aware attackers beyond budget X, models
   above the scale evaluated here, **and non-convolutional or non-vision architectures, which this
   study does not evaluate at all** — §3.3).
6. **A contract traceability table** mapping each clause of `new_thesis_contract.pdf` — problem
   statement, objective, secondary objective, each proposed technical direction, and each named
   evaluation metric — to the section and result that discharges it. Additions (§0, C0) are listed
   separately so the distinction between commitment and extra is visible to an examiner.
7. **A dataset-scope note** stating the move to CIFAR-10 alone, what it removed, what substitutes for
   each loss (§3.3) and what the freed budget bought (source: Dataset.md §10–§11). **Pair it with the
   §13 modality bound**: the note explains the decision, the bound states its consequence.
