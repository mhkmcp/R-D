# SPEC — Fault Injection Detection in Deep Neural Networks Using Multi-Layer Internal States and Support Vector Data Description

**Status:** Draft v1.0 · **Date:** 2026-09-14 · **Derived from:** `new_thesis_contract.pdf`

This document is the engineering and experimental specification for the thesis. It fixes the
threat model, the feature extraction contract, the detector formulation, the evaluation protocol,
and the deliverables, so that results are reproducible and the contract's claims are testable.

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

### C0 in practice — what is an addition

The contract is silent on these, so they are additions rather than commitments. They are listed here
so the cut order under schedule pressure (§13) is obvious and no commitment is ever cut first:

| Addition | Where | Cut cost |
|---|---|---|
| Adaptive attacker (L2) | §2, §10 | None to the contract — cut first |
| `sa0`/`sa1`, `rnd_val` fault modes | §4.1 | None; `bf_w`/`bf_b` carry the contract's fault model |
| `bf_act` transient activation faults | §4.1 | None; the contract scopes faults to weights and biases |
| INT8 quantised arm | §3, §9.6(4) | None, but it is the deployment-relevant regime for KWS |
| Deep SVDD (D4) | §6.2 | None; already marked an extension |
| Datasets and architectures (§3) | §3.1 | The contract names neither, requiring only "across architectures and fault models" |
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

**Deployment setting.** The assumed target is **edge / embedded inference**: an always-on keyword
spotter on a microcontroller or phone DSP, or an inline network sensor. This is where physical and
Rowhammer-class fault injection is realistic rather than hypothetical, and it is what makes the
overhead budget in §9.4 binding — a monitor that only fits on a datacentre GPU does not defend the
threat model this thesis assumes. Overhead is therefore measured on edge-class hardware (§9.4) by
design, not by accident of available equipment.

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

## 3. Target models and datasets

| ID | Dataset | Model | Precision | Role |
|---|---|---|---|---|
| M1 | Speech Commands v2, 2-class subset | DS-CNN-S | FP32 | Pipeline smoke test — produces **no** thesis result |
| **M2** | **Speech Commands v2, 12-class** | **DS-CNN** (plain; MLPerf Tiny reference) | FP32 + INT8 PTQ | **Primary** — full §4.2 grid, all ablations, all baselines |
| M3 | Speech Commands v2, 12-class | **M5-Net** (residual 1-D CNN on raw waveform; Dai et al. 2017) | FP32 | Architecture contrast — plain vs residual |
| **M4** | **UNSW-NB15**, 10-class `attack_cat` | Residual MLP, 8–10 layers | FP32 | **Secondary** — modality contrast + security narrative |

> Naming: "M5-Net" is the Dai et al. raw-waveform CNN, renamed here so it does not collide with the
> model IDs M1–M4 or the milestone IDs M-1…M-9 in §12.

M2 and M3 are mandatory and carry the thesis. M4 is mandatory but runs a confirmatory subset of the
grid (§4.2). M1 exercises the pipeline end to end and is never cited as a result. Baseline clean
accuracy for each model is recorded in `artifacts/models/manifest.json` and must be within 1 point of
the published reference before any fault experiment runs.

INT8 post-training quantisation is included because bit flips in an 8-bit weight have a bounded,
qualitatively different effect from a flip in an FP32 exponent — the detector must be shown to work
in both regimes rather than only in the easy FP32-exponent case. Keyword spotting is a natural home
for this arm: deployed KWS models are almost always quantised, so the INT8 result is the deployment-
relevant one rather than a curiosity. **Constraint:** PyTorch quantized kernels run on CPU backends
(qnnpack / x86), **not** on MPS, so the INT8 arm is CPU-only and its overhead numbers are not
comparable to the MPS FP32 numbers (§9.4).

### 3.1 Dataset rationale

The selection constraints were: no image datasets, laptop-class compute (Apple MPS / CPU), and two
datasets evaluated deeply. The binding requirement is **not** modality — it is structure:

- **H2/H3/H4 are hypotheses about layers.** A 3–4 layer model cannot support a multi-layer fusion
  claim or a propagation result. Every model here provides ≥ 5 usable taps.
- **§5.2's feature blocks assume channel structure.** Conv taps populate all 29 features; dense taps
  populate 22 (§5.2). This is why a shallow tabular MLP is unsuitable as the *primary* dataset, and
  why the NIDS model is deliberately deepened (§3.2).

**Why keyword spotting is primary.** Always-on KWS runs on embedded microcontrollers and phone DSPs —
precisely the deployment class where Rowhammer, glitching and laser fault injection are realistic
rather than hypothetical (§2). MLPerf Tiny includes KWS as a reference benchmark with a published
DS-CNN accuracy (**92.2 %** on the 12-class task, measured on a 1,000-sample test subset; the
benchmark's quality target is 90 %), preserving a recognised point of validation. The input
is a 49×10 MFCC frame and DS-CNN-S is ≈25k parameters, roughly an order of magnitude cheaper than a
CIFAR-class model, which is what makes the §4.2 grid fit the hardware. Twelve classes keeps a healthy
`MASKED`/`DEGRADED` population (§4.3): high class counts pack decision boundaries together and push
nearly every injection into `SDC`, erasing the subtle-fault regime this thesis is about.

**Why DS-CNN + M5-Net.** Plain versus residual topology is the contrast **H4** needs: skip
connections give a fault an alternate propagation path, so the relationship between injected layer
and first-alarm tap should differ measurably between the two.

**A fair objection, stated here rather than left for the defense.** MFCC features are 2-D and DS-CNN
applies 2-D convolutions, so DS-CNN can reasonably be called an image model in disguise. M5-Net is
the mitigation: it consumes **raw 16 kHz waveform with 1-D convolutions**, so at least one primary
architecture involves no 2-D processing at any point. Claims about generality rest on M3 and M4, not
on M2 alone.

**Why UNSW-NB15 is secondary.** An attacker flipping bits to blind an intrusion detector is a
coherent, self-motivated attack story rather than a contrived one. It also provides genuine modality
contrast: it tests whether the method needs convolutional structure at all, and a degraded result
there is a legitimate finding that tells a deployer which §5.2 feature blocks actually carry signal.
Collapsing the 10-class `attack_cat` prediction to Normal-vs-Attack costs nothing and yields the
operationally meaningful binary metric from the same sweep, so both are reported.

**Division of labour.** Speech Commands carries the full grid and the §9.6(8) confounder ablation;
UNSW-NB15 carries modality contrast and the security narrative. Neither dataset is asked to do a job
it cannot do.

**Cost of excluding images, stated plainly.** The bit-flip attack literature (BFA, DeepHammer, TBT,
ProFlip) is almost entirely vision, so this thesis has **no direct numerical comparison to prior work**
on flip budgets or attack success rates. This is recorded as a threat to validity in §13 rather than
quietly omitted.

### 3.2 UNSW-NB15 model and data handling

- **Model.** 8–10 layer residual MLP over one-hot categoricals (`proto`, `service`, `state`, with
  rare-category bucketing) concatenated to standardised numerics; taps at each residual block. Depth
  beyond the 2–4 layers typical in the NIDS literature is **required by the research question**, not
  chosen for accuracy. The thesis must show depth costs nothing by validating accuracy against
  published shallow baselines on both the 10-class and collapsed-binary views.
- **Fallback.** If the reduced D-dense feature set (§5.2) proves substantially weaker, an
  FT-Transformer restores full structure — per-feature token embeddings give "spatial" = feature
  tokens and "channels" = embedding dimensions, so D-conv applies unchanged. This is the documented
  recovery path, not the first build.
- **Preprocessing.** Drop `id` and the binary `label` column when training `attack_cat`; deduplicate
  records; **fit the scaler on the training split only** — a classic leakage route on this dataset
  and a direct C2 violation. The §7 leakage guard asserts this.
- **Class imbalance.** Worms and Shellcode have very few records. Fault probes are drawn from a
  **class-balanced probe set** so that §4.3 SDC rates are not confounded by prior class frequency;
  per-class analysis for the rare categories is flagged as underpowered in §13.
- **Confounder ablation.** No corruption suite exists for this dataset. §9.6(8) is run with graded
  numeric-feature perturbation at 5 severities, labelled explicitly as *constructed* and therefore
  weaker evidence than Speech Commands' native background-noise SNR sweep, which remains the primary
  evidence for that ablation.

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

### 4.2 Injection grid

The grid below is sized for laptop-class compute (§9.4). Where a quantity was reduced from the
statistically ideal value, the reduction and its cost are recorded rather than silently applied.

- **Bit position.** FP32: stratified over sign (31), exponent MSBs (30–27), remaining exponent
  (26–23), high mantissa (22–16), low mantissa (15–0). INT8: all 8 positions, sign/MSB separated.
  Stratification is mandatory — sampling bit positions uniformly makes the problem artificially easy,
  because exponent-MSB flips dominate and are trivially detectable. **Bit strata are never traded
  away for compute** (see §12 verification); they are what keeps §9.1 honest.
- **Layer.** Every injectable layer, bucketed into early / middle / late thirds for reporting.
- **Flip budget.** N ∈ {1, 4, 16} per inference for L0/L1 — the range of the original
  {1, 2, 4, 8, 16} with the interpolating points dropped.
- **Targeting.** L0 uniform random; L1 gradient-ranked progressive bit search.
- **Repetitions.** 100 independent injections per (model, layer bucket, bit stratum, budget) cell.
- **Probe samples per injection.** 32. The detection unit is per-inference, so the statistical sample
  is the *(injection, input)* pair — at equal cost, many injections × few inputs each covers fault
  diversity better than few injections × many inputs, because fault configuration is the dimension
  results are stratified over in §9.1.
- **Fault-mode coverage.** Full grid for **`bf_w` and `bf_b`** (§4.1 — the contract names weights and
  biases together, so neither is a reduced arm). `sa0`/`sa1`, `rnd_val` and `bf_act` run a reduced
  grid: late layer bucket, 2 bit strata (exponent-MSB and low-mantissa, i.e. the obvious and the
  subtle extremes), budgets {1, 16}. Note that a bias tensor holds far fewer parameters than a weight
  tensor, so the `bf_b` full grid re-samples a smaller target pool — record the per-layer bias
  parameter count alongside the results, since coverage of that pool is much denser than for weights
  and that changes how the two arms compare.
- **Model coverage.** M2 and M3 run the full grid; M4 (UNSW-NB15) runs the confirmatory subset —
  `bf_w` and `bf_b` full grid, other fault modes omitted.

Net: ≈ 4,500 injections × 32 probes ≈ 144k forward passes per model / fault mode / seed, and two
full-grid fault modes per model rather than one.

**Small-model caveat.** DS-CNN-S has ≈25k parameters and correspondingly little redundancy, so faults
may skew catastrophic and starve **Track S** (§4.3) — the track carrying the contract's central
claim. If the combined `MASKED` + `DEGRADED` share falls below ~15%, switch M2 to DS-CNN-L (≈400k
parameters) and re-run. Check this before generating the full sweep, not after: without a populated
Track S, H1's discriminating test cannot be run at all.

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
because a NaN detector is trivial and would otherwise flatter the headline numbers. Within Track S,
`MASKED` and `DEGRADED` are broken out because they differ in how much internal signal is available:
`DEGRADED` has a measurable logit-margin shift, `MASKED` by definition does not.

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

Every injection is recorded as a reproducible record
`{model, seed, layer, flat_index, bit, count, mode}` written to `artifacts/faults/*.parquet`, so any
run can be replayed exactly. No fault is ever applied to the model instance used to generate clean
training features.

---

## 5. Internal-state feature extraction

### 5.1 Tap points

Taps are registered with `module.register_forward_hook(hook, always_call=True)`
(`always_call=True` so a fault that raises inside a later module does not silently drop earlier
observations). Tap set per model is declared in config, defaulting to:

- post-activation output of each block group (DS-CNN depthwise-separable block; M5-Net residual
  stage; MLP residual block),
- the pooled embedding preceding the classifier,
- the pre-softmax logits.

This yields 5–6 taps for M2, 5–7 for M3, and 8–10 for M4 (one per residual block). Every model must
clear the ≥ 5 usable taps required by §3.1 before it enters the study. A tap registry maps each tap
to a stable string ID so configurations are comparable across models.

### 5.2 Per-tap feature vector (the monitoring contract)

For a tap producing activation `A ∈ R^{N×C×H×W}` (2-D conv, DS-CNN), `R^{N×C×T}` (1-D conv, M5-Net)
or `R^{N×D}` (dense, M4), the monitor computes a fixed-length descriptor **per sample**. All statistics are single-pass and reduction-only — no tensor is copied
to host memory, which is what keeps overhead bounded.

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
  of per-channel spatial means; mean and std of per-channel spatial stds. For M5-Net's 1-D taps,
  "channels" are filter channels and the "spatial" axis is time, so the block applies unchanged.
- **D-dense (4) — dense/MLP taps** (M4): Shannon entropy of the normalised unit-activation energy
  distribution; top-1, top-2, top-4 unit energy shares. The four spatial statistics have no analogue
  and are zero-padded.

**Block E — cross-layer (3, computed at fusion time)**
ratio of this tap's L2 to the previous tap's L2; ratio of its entropy to the previous tap's; and
the difference of its saturation ratio from the previous tap's.

Total: **29 features per conv tap, 22 per dense tap** (9+4+5+4), fixed-width and
architecture-independent, with zero-padding keeping widths comparable across taps. Losing 4 of 29
features on dense taps is a reduction, not a redesign — but whether that reduction costs detection
accuracy is itself a result, reported as the modality-contrast finding for M4 (§9.6 item 3).

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

**Excluded by C1:** the Mahalanobis OOD score and `EllipticEnvelope` / `MinCovDet`
(robust-covariance). KDE, LOF and Isolation Forest fill the density-baseline role instead.

This exclusion needs more active guarding on **M4 (UNSW-NB15)** than anywhere else in the project:
robust-covariance outlier detection is a *standard, expected* baseline in the tabular
anomaly-detection literature, so it is the one place a reviewer may ask for it by name and a
well-meaning implementer may add it by reflex. The answer is the §0 rationale plus the KDE/LOF
substitutes, not an exception.

---

## 7. Data splits and leakage control

| Split | Content | Used for |
|---|---|---|
| `clean_fit` | 60 % of clean inference records | Fitting normaliser + SVDD |
| `clean_cal` | 20 % | Threshold calibration, score CDF mapping, hyperparameter selection |
| `clean_test` | 20 % | FPR measurement |
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

Input samples underlying clean and fault records are drawn from the same held-out pool — the same
Speech Commands test clips for M2/M3, the same class-balanced UNSW-NB15 probe records for M4 — so
detection cannot be attributed to input distribution shift.

An automated leakage check runs before every results build and fails it on any of:
- a fault record ID appearing in any fitting or calibration set;
- a `fault_test` instance ID appearing in `fault_dev`;
- a `fault_gen` configuration tuple appearing anywhere in `fault_dev`;
- `fault_test` missing any layer bucket, bit stratum or budget present in the §4.2 grid — the guard
  that stops the narrow-slice failure from recurring silently.

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
stratum*, *per layer bucket*, *per flip budget*, *per fault mode*, and *per attacker level*
(L0/L1/L2). Aggregate-only tables are not acceptable: they hide the results that matter, namely
performance on subtle, low-bit-position, single-flip faults, which is the regime the contract is
about.

For **M4**, results are reported twice from the same sweep: over the 10-class `attack_cat` task, and
over the Normal-vs-Attack collapse of the same predictions. The collapse is free — it relabels
existing predictions rather than re-running anything — and it is the operationally meaningful view
for an intrusion detector, so omitting it would be a missed result rather than a saved cost.

### 9.2 Localisation metrics (H4)

Spearman ρ between injected layer index and first-alarm tap index; confusion matrix of injected
layer bucket vs first-alarm bucket; mean "propagation depth" = number of taps between injection and
first alarm.

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
The threat model (§2) assumes embedded inference, so an overhead budget measured on a laptop-class
CPU and integrated GPU is closer to the deployment reality than a datacentre-GPU number would be.
State this once, in these terms, and do not hedge it elsewhere.

The **INT8 arm is CPU-only** (§3) because PyTorch quantized kernels do not run on MPS. Its latency
and memory figures are therefore not comparable to the MPS FP32 figures and must be tabulated
separately, with the reason given inline — an unexplained discontinuity between the two would read
as a measurement error.

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
   TPR/overhead Pareto front. This is the direct answer to H3.
2. **Tap position** — early-only vs middle-only vs late-only vs logits-only.
3. **Feature-block ablation / modality contrast** — drop each of Blocks A–E in turn on M2. Then
   compare against M4, whose dense taps already run without the four spatial statistics (D-dense,
   §5.2): this isolates whether the method depends on convolutional structure, and tells a deployer
   which blocks actually carry signal. A large M2–M4 gap concentrated in Block D is a finding, not a
   failure.
4. **Precision** — FP32 (MPS) vs INT8 (CPU-only, §9.4), on M2.
5. **Training budget** — clean fitting samples ∈ {500, 2k, 10k, 50k}; establishes the minimum
   calibration data a deployer needs.
6. **Kernel sensitivity** — RBF vs polynomial vs linear SVDD.
7. **Transfer** — detector fitted on M2 applied to M3 taps (expected to fail; documents that the
   monitor is model-specific, which is a deployment cost worth stating).
8. **Input-shift confounder** — score clean-but-shifted inputs to quantify how much of the detector's
   signal is "fault" versus "anything unusual". A monitor that fires equally on both is a novelty
   detector, not a fault detector, and the thesis must say so if that is what the data shows.
   - **M2/M3 (primary evidence):** Speech Commands' own `_background_noise_` set mixed at 5
     controlled SNR levels. Severity-graded shift is native to the benchmark, so it needs no
     defending.
   - **M4 (supporting only):** graded numeric-feature perturbation at 5 severities, labelled
     explicitly as *constructed*. Weaker evidence, and reported as such — no corruption suite exists
     for UNSW-NB15.

---

## 10. Adaptive attacker (L2)

An evaluation whose attacker is unaware of the defence overstates the defence. L2 formulates flip
selection as a constrained search: maximise task loss subject to every monitored SVDD score staying
below its threshold, solved greedily over candidate (parameter, bit) pairs ranked by loss gain and
filtered by monitor score. Reported outcome: the attacker's success rate and required flip budget
with and without the monitor present. An increase in required budget is a real, quantifiable
security gain even when the detector is evadable — that framing is the honest one and should be the
thesis's conclusion for this section.

---

## 11. Implementation

### 11.1 Stack

Python 3.11 · PyTorch 2.x + torchaudio (MFCC / waveform loading) · scikit-learn 1.7 · NumPy ·
pandas/pyarrow · Hydra or plain YAML configs · pytest. No dependency on a fault-injection framework
is assumed; injection is ~200 lines of local code (§4.4) and stays auditable.

### 11.2 Layout

```
HK-2026/
├── SPEC.md
├── configs/
│   ├── data/{speech_commands,unsw_nb15}.yaml
│   ├── model/{dscnn_s,dscnn,m5net,nids_mlp}.yaml
│   ├── faults/{bf_w,bf_b,sa,rnd_val,bf_act}.yaml
│   ├── taps/<model>.yaml            # tap registry per model
│   └── detector/{d1_single,d2_fused,d3_late,d4_deepsvdd}.yaml
├── src/fidnn/
│   ├── data/          # audio pipeline (MFCC, SNR mixing), tabular pipeline (encode, scale)
│   ├── models/        # builders, checkpoints, quantisation
│   ├── inject/        # bit manipulation, injection planner, replay
│   ├── taps/          # forward hooks, feature blocks A–E (D-conv/D-dense), registry
│   ├── detect/        # normaliser, svdd, fusion, calibration, baselines
│   ├── eval/          # metrics, taxonomy labelling, bootstrap, overhead
│   └── cli.py         # extract | inject | fit | calibrate | eval | report
├── tests/             # incl. bit-flip correctness, leakage guard, determinism
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

### 11.4 Required tests

- Bit-flip round-trip: flipping bit *b* twice restores the original tensor, bit-exactly, for FP32
  and INT8.
- Known-value test: flipping FP32 bit 30 of a known float yields the expected value.
- **In-place restore (§4.4):** after a simulated exception mid-probe, the model's parameter checksum
  equals its clean checksum. This guards the one failure mode that would silently contaminate every
  subsequent injection rather than producing a visible error.
- Hook coverage: every configured tap fires exactly once per forward pass; removing hooks restores
  baseline latency to within noise.
- **Tap budget:** every model in §3 exposes ≥ 5 usable taps, asserted from the tap registry.
- **Feature-block population:** D-conv is non-zero for all conv taps (DS-CNN, M5-Net); D-dense is
  non-zero for all MLP taps (M4). An all-zero block means a silently misapplied variant.
- Leakage guard (§7) as a failing test, not a comment — including the assertion that the M4 tabular
  scaler and categorical encoder were fit on `clean_fit` only.
- Feature determinism: identical input ⇒ identical feature vector across runs.
- Overhead harness sanity: monitor-off path is statistically indistinguishable from the unmodified
  model.

---

## 12. Milestones

| # | Milestone | Exit criterion |
|---|---|---|
| **M-0** | **Throughput calibration** | Measured samples/s for M2/M3/M4 on MPS and CPU, hooks on and off; §4.2 grid sizes confirmed or revised **before** any sweep is launched |
| M-1 | Models trained, accuracy verified | `manifest.json` within 1 pt of reference: DS-CNN 12-class **92.2 %** (MLPerf Tiny reference), M5-Net comparable, M4 matching published UNSW-NB15 baselines on both the 10-class and binary views |
| M-2 | Injection engine + taxonomy labelling | Bit-flip and in-place-restore tests pass; taxonomy distribution plotted for M2 and M4; **Track S (`MASKED` + `DEGRADED`) share ≥ ~15%** — a hard gate, since H1's discriminating test runs on that track |
| M-3 | Tap/feature pipeline | Features extracted for M1–M4; tap-budget, block-population and determinism tests pass |
| M-4 | D1 + D2 on M1 — **sanity gate, produces no thesis number** | Calibrated FPR within 0.3×–3× of nominal; pipeline yields a plausible TPR@1%FPR on both tracks. First *reported* detector numbers come from M-5 on M2. |
| M-5 | Full evaluation on M2/M3 + confirmatory M4, all baselines | Headline tables generated by `make reproduce` |
| M-6 | Overhead study | Pareto front (TPR vs latency) for the K sweep, on CPU and MPS, FP32 and INT8 |
| M-7 | Ablations + localisation (H4) | §9.6 items 1–6 and 8 complete, including the M2–M4 modality contrast |
| M-8 | Adaptive attacker (L2) | Success-rate/budget table with and without monitor |
| M-9 | Writing, threats to validity | Draft complete |

**M-0 is a gate, not a formality.** Every grid size in §4.2 rests on a throughput estimate rather
than a measurement on this machine; launching the full sweep before checking it risks discovering
mid-thesis that the budget was wrong by an order of magnitude.

M-1…M-5 are the minimum viable thesis: they settle H1 and H2. M-6/M-7 settle H3, M-7 settles H4,
M-8 is what raises the work above a purely empirical detection study.

---

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Faults are trivially detectable (exponent-MSB flips dominate) | Results look strong but say nothing | Bit-strata stratification (§4.2) and per-stratum reporting (§9.1) are mandatory, not optional |
| Detector is really a generic novelty detector | Claim of *fault* detection unsupported | Ablation §9.6(8) measures it directly; report the finding either way |
| INT8 faults too subtle to detect | H1 fails in the quantised regime | Report FP32/INT8 separately; a negative INT8 result is a legitimate, publishable finding |
| SVDD does not scale to fused high-dimensional features | Fitting time blows up | Nystroem + `SGDOneClassSVM` path (§6.1), with the approximation gap measured |
| Overhead exceeds any plausible deployment budget | H3 fails | K sweep finds the Pareto front; report the minimum viable K honestly even if it is "all taps" |
| Persistent-fault setting makes windowed detection near-trivial | Overstated results | Per-inference numbers are reported alongside windowed ones, always |
| **No comparability to the bit-flip literature** (BFA, DeepHammer, TBT are all vision) | Cannot compare flip budgets or attack success rates to prior work | Named as a threat to validity in the thesis; comparisons are made *within* this study (detector variants, baselines) rather than across papers |
| **Throughput estimate wrong on this machine** | §4.2 grid does not fit the schedule | M-0 gate measures it first; if short, cut repetitions 100 → 50 before cutting bit strata — strata keep §9.1 honest, repetitions only widen CIs |
| **DS-CNN-S too small — Track S collapses** | The contract's central claim becomes untestable: H1's discriminating comparison against output-only baselines has no subtle faults to run on | Hard gate at M-2 (§12); escalate to DS-CNN-L (≈400k params) |
| **M4 dense taps (D-dense) too weak** | Secondary dataset yields nothing | It is a *finding* first (§9.6 item 3), not a failure; FT-Transformer is the documented recovery path (§3.2) |
| **UNSW-NB15 label quality; official split known to flatter results** | M4 accuracy and SDC rates optimistic | Deduplicate; report the split used; treat M4 as confirmatory, never as the headline |
| **Rare classes (Worms, Shellcode) underpowered** | Per-class M4 analysis not trustworthy | Class-balanced probe set (§3.2); rare-class results explicitly marked underpowered |
| **Non-standard MLP depth reads as gratuitous** | Reviewer objection to M4 | §3.2 justifies depth from the research question and validates that accuracy matches published shallow baselines |
| Schedule slip | Scope must shrink | Cut in the C0 order (§0): additions first — adaptive attacker (§10), then `sa`/`rnd_val`/`bf_act`, then D4 — before any §9.6 ablation. `bf_w`/`bf_b` coverage, both §4.3 tracks, and M2/M3 are contract commitments and are never cut |

---

## 14. Deliverables

1. Reproducible codebase per §11, with tests and `make reproduce`.
2. Result artifacts (parquet + JSON) for every table and figure in the thesis.
3. Thesis chapters: threat model, method, evaluation, adaptive-attacker analysis, threats to validity.
4. A methods paragraph stating the C1 exclusion and the substituted baselines.
5. A short deployment note: which layers to monitor, what it costs, and what it does not cover
   (transient input faults, control-flow faults, detector-aware attackers beyond budget X).
6. **A contract traceability table** mapping each clause of `new_thesis_contract.pdf` — problem
   statement, objective, secondary objective, each proposed technical direction, and each named
   evaluation metric — to the section and result that discharges it. Additions (§0, C0) are listed
   separately so the distinction between commitment and extra is visible to an examiner.
