# Dataset

**Companion to:** [`SPEC.md`](SPEC.md) §3 (Target models and datasets) · **Governed by:** `new_thesis_contract.pdf`
**Status:** Draft v2.0 · **Date:** 2026-09-19 · **Supersedes:** v1.1 (three datasets)

This document records the dataset the thesis uses, where to obtain it, how it is split and
preprocessed, and — at length, because it is the question an examiner will ask — why this one, why
only this one, and what the single-dataset design costs.

The contract names no dataset. It requires only that the study evaluate *"across architectures and
fault models"* and that faults be injected *"as bit flips in weights and biases, varied across layers
and bit positions"*. Dataset choice is therefore an **addition** under [`SPEC.md`](SPEC.md) §0 (C0),
and it is justified here rather than assumed.

Note what the contract does **and does not** ask for: variation across *architectures* and *fault
models*, not across datasets. A single-dataset design with two contrasting architectures satisfies the
clause as written. That is the formal basis for this revision, and §10 records what it nonetheless
costs in scientific reach.

> **Change in v2.0 — CIFAR-10 is the only dataset.** v1.1 carried three: CIFAR-10 (primary),
> Speech Commands (audio modality contrast, M4) and UNSW-NB15 (dense-tap contrast, M5). **M4 and M5
> are withdrawn.** The thesis now runs entirely on CIFAR-10, across the model lineup M1–M3.
> §10 records exactly what that removes and what it frees, because a reader comparing revisions is
> entitled to see this as a decision rather than a drift.
>
> ✅ **[`SPEC.md`](SPEC.md) v2.0 is aligned with this document.** The single-dataset decision is
> final and carried through both: M4 and M5 are removed from the model table, the clauses that
> depended on them are re-specified rather than deleted, and §13 now carries the modality bound as an
> explicit limit. §10.3 records where each affected clause landed.

---

## 1. Selection criteria

Constraints fixed before any dataset was considered:

| # | Constraint | Origin |
|---|---|---|
| S2 | **Laptop-class compute** — Apple MPS / CPU, no CUDA workstation. | Available hardware |
| S3 | **One dataset, evaluated exhaustively** rather than several evaluated partially. | Supervisory scope (revised in v2.0) |

Withdrawn:

| # | Constraint | Status |
|---|---|---|
| ~~S1~~ | ~~**No image datasets.**~~ | **Withdrawn 2026-09-19** ([`SPEC.md`](SPEC.md) §0). Recorded, not deleted — §11.1. |

Four criteria follow from the thesis itself, and they decided the outcome:

| # | Requirement | Why it binds |
|---|---|---|
| S4 | **≥ 5 usable tap points per model.** | H2, H3 and H4 are hypotheses about *layers*. A 3–4 layer network cannot support a multi-layer fusion claim or a fault-propagation result at all. |
| S5 | **Channel-structured activations.** | [`SPEC.md`](SPEC.md) §5.2 Block D derives 8 of 29 per-tap features from channel structure (channel-energy entropy, top-k channel shares, per-channel spatial statistics). Conv taps populate all 29. |
| S6 | **A severity-graded clean-input shift set.** | [`SPEC.md`](SPEC.md) §9.6(8) asks whether the detector responds to *faults* or to *anything unusual*. Without graded shifted-but-clean inputs, that question cannot be answered, and the thesis cannot rule out that it has built a novelty detector. |
| **S7** | **External comparability to the bit-flip attack literature.** | v1.0 recorded as its single largest threat to validity that the thesis would have *no direct numerical comparison to prior work* on flip budgets or attack success rates. Only a dataset the attack literature actually uses can retire that threat. |

With S3 narrowed to one dataset, **all four must be satisfied by the same dataset simultaneously.**
That is a much harder test than v1.1's — where S6 could be met by one dataset and S7 by another — and
CIFAR-10 is the only candidate considered that passes all four alone (§11).

---

## 2. What the dataset is

60,000 32×32 colour images in 10 mutually exclusive classes, 6,000 per class, curated with
human-verified labels by Alex Krizhevsky, Vinod Nair and Geoffrey Hinton from the 80 Million Tiny
Images collection.

| Property | Value |
|---|---|
| Released | 2009 |
| Size | 60,000 images — 50,000 train / 10,000 test |
| Format | 32×32 RGB |
| Classes | `airplane` `automobile` `bird` `cat` `deer` `dog` `frog` `horse` `ship` `truck` |
| Balance | Exactly 6,000 images per class; 5,000 train + 1,000 test each |
| Task used here | The standard 10-class classification task |
| Licence | No formal licence; freely distributed for research. Kaggle copies are governed by the competition rules. |
| Reference | Krizhevsky, A. (2009), *Learning Multiple Layers of Features from Tiny Images*, Technical Report, University of Toronto |

**Models it carries** ([`SPEC.md`](SPEC.md) §3):

| ID | Model | Precision | Role |
|---|---|---|---|
| M1 | ResNet-8, 2-class subset | FP32 | Pipeline smoke test — produces **no** thesis result |
| **M2** | **ResNet-20** (CIFAR variant; He et al. 2016) | FP32 + INT8 PTQ | **Primary** — full §4.2 grid, all ablations, all baselines, literature comparison |
| **M3** | **VGG-11-BN** (plain, no skip connections) | FP32 + INT8 PTQ | **Primary** — architecture contrast, plain vs residual, identical inputs |

M2 and M3 are the thesis. Because both consume identical inputs and differ only in whether skip
connections exist, the H4 propagation comparison isolates the variable it intends to — which is a
*stronger* contrast than v1.0's, where topology and input representation varied together.

---

## 3. Source packaging — and the one non-obvious fact

The Kaggle *CIFAR-10 — Object Recognition in Images* competition
(<https://www.kaggle.com/competitions/cifar-10/data>) is a playground/knowledge competition scored on
plain classification accuracy. It ships:

| File | Contents | Approx. size |
|---|---|---|
| `train.7z` | 50,000 PNG images, 32×32 RGB, named `1.png` … `50000.png` | ~120 MB |
| `trainLabels.csv` | `id,label` over those 50,000 ids | < 1 MB |
| `test.7z` | **300,000** PNG images, `1.png` … `300000.png` | ~600 MB |
| `sampleSubmission.csv` | `id,label` submission template | small |

### 3.1 The test archive is unusable for this thesis

Only **10,000** of the 300,000 images in `test.7z` are the real CIFAR-10 test set. The remaining
**290,000 are dummies**, added by the competition organisers to deter hand-labelling, and Kaggle
silently ignores predictions on them when scoring. **No labels are published for any of the 300,000.**

This is disqualifying, for two independent reasons:

- [`SPEC.md`](SPEC.md) §4.3's outcome taxonomy needs ground truth for *every* probe —
  `MASKED` / `DEGRADED` / `SDC` are defined by whether top-1 changed **and** what the correct answer
  was. A leaderboard score returns one aggregate number per submission and supplies none of that.
- [`SPEC.md`](SPEC.md) §7 requires a labelled held-out pool **shared between clean and fault
  records**, so that detection cannot be attributed to input distribution shift.

This matters more in v2.0 than it did in v1.1: with no second dataset, there is no other labelled pool
anywhere in the project to fall back on.

### 3.2 Resolution — two sources, one dataset

| Role | Source |
|---|---|
| Training data | Kaggle `train.7z` + `trainLabels.csv` — 50,000 labelled images |
| Labelled held-out pool | **Canonical** CIFAR-10 distribution (`torchvision.datasets.CIFAR10(train=False)`), which ships the same 10,000 test images *with* labels |
| `test.7z` | **Not downloaded.** ~600 MB of which 97 % is unusable. |

A Kaggle leaderboard submission may optionally be made **once**, at M-1, as an independent check that
the M2 accuracy gate is not the product of a split error. It produces no thesis number.

This is still one dataset — the two sources are two packagings of the same 60,000 images — but the
assembly must be verified rather than assumed (§5.1).

---

## 4. Download

Kaggle competition rules must be accepted in the browser first ("I Understand and Accept" on the
competition page) — the API will not serve the files otherwise. Verify with
`kaggle competitions list --group entered`. Credentials go in `~/.kaggle/kaggle.json`
(`{"username":"…","key":"…"}`, `chmod 600`) or in `KAGGLE_USERNAME` / `KAGGLE_KEY`.

```bash
pip install kaggle

# Training images and labels only — test.7z is deliberately not fetched (§3.1)
kaggle competitions download cifar-10 -f train.7z        -p data/cifar10
kaggle competitions download cifar-10 -f trainLabels.csv -p data/cifar10

# Single-file downloads may arrive wrapped in a .zip; unwrap if so, then extract the 7z
#   brew install p7zip
7z x data/cifar10/train.7z -o data/cifar10/
```

```python
# Canonical labelled held-out pool (§3.2)
import torchvision
torchvision.datasets.CIFAR10(root="data/cifar10-canonical", train=False, download=True)
```

- Kaggle competition data page: <https://www.kaggle.com/competitions/cifar-10/data>
- Canonical distribution and the technical report: <https://www.cs.toronto.edu/~kriz/cifar.html>
  (`cifar-10-python.tar.gz`, ~163 MB — record its checksum in `manifest.json`)
- CIFAR-10-C confounder suite: Zenodo [`10.5281/zenodo.2535967`](https://zenodo.org/records/2535967)

---

## 5. Splits

| [`SPEC.md`](SPEC.md) §7 split | Source | Share |
|---|---|---|
| `train` | Stratified 80 % of the 50,000 Kaggle training images — classifier training only | 40,000 |
| `clean_fit` | Stratified 60 % of the remaining 10,000 Kaggle images | 6,000 |
| `clean_cal` | Same 10,000, disjoint | 2,000 |
| `clean_test` | Same 10,000, disjoint | 2,000 |
| Fault probe pool | The **canonical labelled 10,000-image test set** — disjoint from all of the above | 10,000 |

Stratification is per class, so every partition keeps the dataset's native 10 % per-class balance.
All splits are seeded and persisted. No detector record — clean or fault — comes from an image the
classifier was trained on, so detection cannot be attributed to a seen-versus-unseen activation shift.

*Amended at M-1.* The v2.0 table carved `clean_*` from all 50,000 training images, which are also the
classifier's training data: the detector would have been fitted on memorised activations and tested
on unseen ones — the input shift [`SPEC.md`](SPEC.md) §7 forbids. Holding 10,000 images out of
classifier training fixes that. The costs: the classifier trains on 40k rather than 50k (expected
ResNet-20 accuracy ≈ 90.5–91 %, still inside the 1-point M-1 gate), and `clean_fit` shrinks from
30,000 to 6,000, which caps the [`SPEC.md`](SPEC.md) §9.6(5) training-budget ablation at 6k.

### 5.1 Two integrity checks the assembly depends on

**Source disjointness.** The Kaggle training images are a repackaging of the canonical 50,000-image
training set, so *in principle* the canonical test set is disjoint from them. [`SPEC.md`](SPEC.md)
§11.4 asserts this as a **failing test**: zero pixel-hash collisions between the Kaggle training
images and the canonical test images. A silent overlap would put probe-pool images into `train` or `clean_fit`
and inflate every detection number through a route invisible in the results — a direct C2 violation.
If the check fails, the fallback is to use the canonical distribution for both splits and drop the
Kaggle packaging entirely.

**Label agreement.** For a sampled subset, `trainLabels.csv` must agree with the canonical training
labels for the same images. This catches a mis-ordered or mis-indexed 7z extraction — which corrupts
everything downstream and produces no error of its own.

### 5.2 A known limit of the hash check

Pixel-hashing catches *exact* duplicates only. Barz & Denzler (2020) found that roughly **3.3 %** of
CIFAR-10 test images have a **near**-duplicate in the training set — a property of the dataset itself,
present in every study that uses it, not an artefact of this two-source assembly. It is not a C2
violation (no record appears in two of our splits) but it does mean the probe pool is marginally less
independent of `clean_fit` than the split table implies. Recorded as a dataset-level limit in
[`SPEC.md`](SPEC.md) §13; the mitigation, if the effect is ever suspected of mattering, is to re-run
the headline with the published near-duplicate list removed from the probe pool.

---

## 6. Preprocessing

- **Per-channel mean/std normalisation** with the standard constants (mean ≈ `(0.4914, 0.4822,
  0.4465)`, std ≈ `(0.2470, 0.2435, 0.2616)`), **fitted on `train` only** and frozen.
- **No GCN + ZCA whitening.** Some older CIFAR pipelines apply it; full-covariance whitening followed
  by a Euclidean metric *is* Mahalanobis distance under another name, and is therefore a direct
  **C1** violation ([`SPEC.md`](SPEC.md) §0). Per-channel scaling is diagonal and permitted.
  [`SPEC.md`](SPEC.md) §11.4 asserts the absence of any full-covariance transform.
- **Training augmentation:** random crop to 32×32 with 4-pixel padding, random horizontal flip.
  Augmentation is **training only** — clean feature extraction for the detector uses unaugmented
  images, so the SVDD models the deployed inference distribution rather than the training one.
- **INT8 PTQ calibration** uses a subsample of `clean_fit` only.

---

## 7. Accuracy targets

| Model | Reference | Source |
|---|---|---|
| M2 ResNet-20 | **91.25 %** (8.75 % error) | He et al. (2016) |
| M3 VGG-11-BN | **≈ 92 %** | Varies across reimplementations — the exact figure **must be pinned to a specific cited source** in `artifacts/models/manifest.json` before M-1 is signed off |

Both must land within 1 point of reference before any fault experiment runs
([`SPEC.md`](SPEC.md) §12, M-1). The asymmetry is deliberate: ResNet-20's CIFAR number is unambiguous
in a way VGG-on-CIFAR numbers are not, so the latter gets a named source rather than a folk figure.

**These are now the only external accuracy references in the project.** v1.1 had a second, independent
check from a different benchmark family (MLPerf Tiny's 92.2 % DS-CNN). With M4 withdrawn, a systematic
pipeline error that happened to affect both CIFAR models equally would pass M-1 unnoticed. The
mitigation is the optional one-shot Kaggle leaderboard submission (§3.2) — it is now worth actually
doing rather than merely permitted.

---

## 8. Confounder set — CIFAR-10-C

Hendrycks & Dietterich (ICLR 2019), Zenodo `10.5281/zenodo.2535967`, **CC BY 4.0**: **15 corruption
types × 5 severity levels** applied to the same 10,000 test images.

This is the [`SPEC.md`](SPEC.md) §9.6(8) evidence — the ablation that decides whether this thesis has
built a *fault* detector or merely a novelty detector. Corruptions are applied to **clean inputs
only** and are never combined with fault injection; the question being asked is what the detector does
when the input is unusual but the weights are intact.

**It is now the sole evidence for that question**, where v1.1 had a second, independent measurement
from a different modality (§10.1). Two mitigations follow, and both should be taken:

1. **Report per-corruption-type results, not only the aggregate.** 15 corruption types at 5 severities
   is itself a spread of evidence; collapsing it to one number throws away the internal replication
   that partly substitutes for the lost second suite.
2. **Split the corruption types by family** — noise, blur, weather, digital — and check whether the
   detector's response is consistent across families or driven by one. A detector that fires only on
   the noise family is telling you something different from one that fires on all four.

---

## 9. Why this dataset

**Direct comparability to the bit-flip attack literature (S7).** This is the reason that decided it.
The attack literature is almost entirely CIFAR-10 and ImageNet: BFA (Rakin et al., ICCV 2019) and
DeepHammer (Yao et al., USENIX Security 2020) both report on ResNet-20 and VGG-class networks trained
on CIFAR-10; TBT (CVPR 2020) and ProFlip (ICCV 2021) use the same setting for targeted attacks.
Adopting CIFAR-10 + ResNet-20 + **INT8** lets the L1 attacker ([`SPEC.md`](SPEC.md) §2) be run as a
faithful BFA reimplementation, its flip-budget-to-accuracy-collapse curve checked against the
published one (**M-1b**, a gate), and only then used to generate the fault population this thesis
detects. *A defender evaluated against a calibrated, externally-validated attacker is a materially
stronger result than one evaluated against an attacker only this thesis has ever run.*

This argument runs through the **INT8** arm specifically — the published attacks target 8-bit
quantised networks. That promotes INT8 from a curiosity to the comparability arm, and is why
[`SPEC.md`](SPEC.md) §0 marks it an expensive cut despite being a C0 addition. **In a single-dataset
design the INT8 arm is effectively uncuttable**: it is the only remaining route to any external
number.

**Depth (S4).** ResNet-20 gives 9 residual blocks across 3 stages, plus stem, pooled embedding and
logits — 6 taps by default, up to 13 if every block is tapped. H2/H3/H4 have room to breathe.

**A clean architecture contrast — and it survives v2.0 intact.** ResNet-20 and VGG-11-BN consume
identical inputs and differ only in whether skip connections exist. This is the contrast the contract
actually asks for (*"across architectures"*), and it is untouched by dropping M4/M5. It is the reason
a single-dataset design remains contract-compliant.

**A native, citable, severity-graded shift suite (S6).** CIFAR-10-C is standard, published and
peer-reviewed — no constructed perturbation to justify, which matters far more now that it stands
alone.

**Ten classes preserves the subtle-fault regime.** [`SPEC.md`](SPEC.md) §4.3 splits outcomes into
Track S (`MASKED` ∪ `DEGRADED` — no change to top-1) and Track H (`SDC` ∪ `CRASH`), and **Track S is
where the contract's claim lives**: *"others induce smaller internal deviations that remain
unnoticed."* As class count rises, decision boundaries pack closer, any perturbation is likelier to
flip top-1, and Track S empties out. Ten classes keeps it populated; a 1000-class task would quietly
delete the phenomenon the thesis exists to study.

**Redundancy is adequate for Track S.** At ~270k parameters ResNet-20 has roughly 10× DS-CNN-S's
capacity, so single low-mantissa flips are likelier to be absorbed. The M-2 gate (Track S share
≥ ~15 %) stays in place to verify this rather than assume it.

**Cost (S2).** 32×32×3 inputs and ≈41 MFLOP/image keep the [`SPEC.md`](SPEC.md) §4.2 grid on a laptop.
M2/M3 each run two precisions, so the M-0 throughput gate still binds — but with M4 and M5 withdrawn
the total project budget falls substantially (§10.2).

**Two fair objections, recorded here rather than left for the defense.**

1. *"CIFAR-10 is small, saturated and much-studied."* True, and irrelevant. The object of study is a
   **detector over internal states under parameter corruption**, not a classifier competing on
   accuracy. Saturation is an *advantage*: reference accuracies are unambiguous, so the M-1 gate is a
   sharp instrument — and the attack literature this thesis compares against lives here.
2. *"32×32 images are not a realistic deployment input."* The threat model does not depend on
   resolution; it depends on parameters living in writable memory. DeepHammer demonstrated exactly
   this attack against exactly this class of model. Where resolution *would* matter is a claim about
   scaling to ImageNet-class networks — and no such claim is made. [`SPEC.md`](SPEC.md) §13 records it
   as a bound on generality, stated plainly rather than implied away.

---

## 10. What the single-dataset design costs — and what it frees

This section exists because "you only evaluated on one dataset" is the *most* predictable examination
question about v2.0, and it deserves a prepared answer rather than an improvised one.

### 10.1 What is lost

| Lost with M4/M5 | Consequence | Honest position to take |
|---|---|---|
| **Modality generality** | H1–H4 become claims about **vision CNNs on 32×32 inputs**, not about DNNs generally. Nothing in the results bounds behaviour on audio, tabular or sequence models. | State the scope in the hypothesis wording itself, not only in §13. A narrower claim that is fully supported beats a broad claim resting on two confirmatory subsets. |
| **The dense-tap contrast (S5)** | Every tap in the project is now a conv tap. [`SPEC.md`](SPEC.md) §5.2's 22-feature D-dense variant is never exercised, and §9.6(3) — *which feature blocks actually carry signal* — loses its natural experiment. | Recover it **by ablation instead of by dataset**: zero out Block D's channel features on conv taps and re-measure. It answers the same question, on the same data, and is cheaper. This is a genuine substitute, not a fig leaf. |
| **Independent confounder corroboration** | CIFAR-10-C is the sole §9.6(8) evidence. If it says the detector is partly a novelty detector, there is no second suite to check that against. | §8's two mitigations: per-corruption-type reporting and per-family consistency. Internal replication across 15 types partly substitutes for one external cross-check. |
| **A second external accuracy reference** | MLPerf Tiny's 92.2 % DS-CNN check is gone; M-1 now validates against CIFAR references only. | Take the optional one-shot Kaggle leaderboard submission (§3.2). It is cheap and independent. |
| **The embedded deployment narrative** | v1.0's always-on-KWS-on-MCU story was the sharpest justification for the §2 threat model and made §9.4's laptop-class overhead numbers a feature. CIFAR-10's story (smart cameras, drones, ADAS SoCs; DeepHammer shared-tenancy) is real but is a *class* of deployment, not a named product. | Argue it from DeepHammer, which demonstrated this attack on this class of model. Weaker rhetorically, not weaker technically. |
| **The security-control narrative** | M5's "attacker blinds an intrusion detector" was a self-motivating attack story. | Nothing replaces it. Do not pretend otherwise. |

### 10.2 What is freed

The M4 and M5 grids, their training, their tap plumbing and their separate preprocessing paths all
disappear. That budget should be **reinvested in depth on M2/M3 rather than banked**, and the
candidates in priority order are:

1. **More seeds.** [`SPEC.md`](SPEC.md) §9.5 requires a dispersion estimate on every number; more seeds
   tighten every confidence interval in the thesis, including H1's paired bootstrap.
2. **The extended 13-tap configuration** on both models, rather than only the default 6.
   H2 (fusion beats best single layer) and H3 (minimal configuration) both get more to say.
3. **The L2 adaptive attacker** ([`SPEC.md`](SPEC.md) §10) — the addition that raises the work above a
   purely empirical detection study, and previously the first thing on the cut list.
4. **Full INT8 parity**, so every FP32 result has an INT8 counterpart rather than a subset.

### 10.3 How `SPEC.md` was reconciled

Dropping M4 and M5 was not a `Dataset.md`-local change. [`SPEC.md`](SPEC.md) **v2.0** carries the same
decision; this table records where each affected clause landed, so a reader can audit that nothing was
silently deleted rather than re-specified.

| Clause | What it said under v1.1 | Where it landed in SPEC v2.0 |
|---|---|---|
| §1 hypotheses | Scope unstated | **Scope paragraph added**: H1–H4 are claims about vision CNNs on 32×32 inputs, worded that way in the thesis rather than walked back in a caveat |
| §3 model table | M4, M5 **mandatory** | Removed; lineup is M1–M3, both primaries running the full grid in both precisions |
| §3.1 | "Why Speech Commands is retained", "Why UNSW-NB15 is retained" | Rewritten around M2/M3; new objection 3 answers "you evaluated on one dataset" directly |
| §3.3, §3.4 | Full handling specs for both datasets | Replaced by **§3.3 "What the single-dataset design costs, and what it frees"** — the loss/substitute table and the reinvestment priority |
| §4.2 | Confirmatory-subset tier for M4/M5 | Tier removed; every model runs the full grid. Cut order rewritten — M2's INT8 arm is never cut |
| §5.1 | Tap table listed DS-CNN and MLP taps | CIFAR models only; extended tap sets **promoted to planned** rather than sweep-only |
| §5.2 | D-dense described as M5's variant | Retained for shape-completeness, explicitly exercised by no model; tested on synthetic tensors (§11.4) |
| §7 | Split sourcing named Speech Commands clips and UNSW-NB15 records | Single CIFAR-10 held-out pool for every model |
| §9.6(3) | Dense-tap contrast as a cross-dataset comparison | Re-specified as the **D-conv → D-dense downgrade** on M2/M3 — varies the feature set alone, where the old design confounded feature set, architecture, modality and dataset |
| §9.6(8) | "corroborated by M4's SNR sweep" | CIFAR-10-C alone, with **per-type and per-family reporting made mandatory** and the severity-slope comparison added |
| §11.1/§11.2 | torchaudio, audio and tabular pipelines, five model configs | CIFAR pipeline only |
| §11.4 | Block-population test asserted D-dense on M5 taps | D-dense tested on synthetic tensors — the path stays live because §9.6(3) exercises the same width |
| §12 | M-1 gated on DS-CNN and UNSW-NB15 baselines | CIFAR references only; Kaggle leaderboard check promoted; M-0 additionally sizes the §3.3 reinvestment |
| §13 | "M4 and M5 bound the *modality* claim" | **Replaced.** The bound is now stated as an unmitigated limit, with the Block-D downgrade named as *indirect* evidence that must not be written up as a substitute for a second modality |

The §13 row was the important one: under v1.1 it claimed the modality bound was covered. That claim
would have been an active misstatement in v2.0, not merely an omission — which is why it was
rewritten rather than dropped.

---

## 11. Alternatives considered and rejected

Recorded because "why not the obvious dataset?" is a predictable examination question.

| Candidate | Why rejected |
|---|---|
| **Speech Commands v2** (v1.0's primary, v1.1's M4) | Fails S7 — no bit-flip attack paper reports on it, so no flip budget or attack-success figure is comparable. Its native SNR sweep (S6) and MLPerf Tiny reference were real strengths; §10.1 records losing them. |
| **UNSW-NB15** (v1.1's M5) | A shallow-by-convention tabular task: fails S4 without a deliberately deepened model, and weakens S5 (22 features per tap instead of 29). Fails S6 outright — no corruption suite exists, so §9.6(8) would rest on constructed perturbation. Fails S7. |
| **MNIST / LeNet-5** | Fails S4 — too shallow to support a multi-layer claim — and its feature distribution is tight enough to make detection look trivially easy. |
| **Shallow tabular NIDS** (CIC-IDS2017, NSL-KDD) | Fails S4, S5, S6 and S7 together. |
| **UCI HAR** (wearable sensors) | Viable and cheap, but fails S6 and S7. |
| **ESC-50 / UrbanSound8K** | Too few clips to fit an SVDD comfortably; fails S6 and S7. |
| **CIFAR-100** | Passes S4/S5/S6 and is partly present in the attack literature, but 100 classes packs decision boundaries tightly enough to starve **Track S** (§9) — the regime the contract's claim lives in. Rejected on the taxonomy, not on cost. |
| **Tiny-ImageNet** | Same Track S concern at 200 classes, and ~10× the sweep cost against S2. |
| **Full ImageNet** | Excluded by S2 — 50–100× the sweep cost. It is also where the *scale* claim would have to be made, and no such claim is made (§9, objection 2). |
| **A text / transformer task** | Deep enough, but tokenisation and embedding layers complicate the semantics of a weight bit flip, and the models are far above the S2 budget. Fails S6 and S7. |

**Only CIFAR-10 passes S4, S5, S6 and S7 simultaneously.** Under v1.1's three-dataset design that was
a convenience; under v2.0's S3 it is the whole selection argument, since no second dataset is
available to cover a gap the first one leaves.

### 11.1 The withdrawn S1 constraint, recorded not deleted

v1.0 of this document carried a standing project constraint excluding image datasets, which made
CIFAR-10 ineligible and forced keyword spotting into the primary slot. **That constraint was withdrawn
on 2026-09-19** ([`SPEC.md`](SPEC.md) §0, "Superseded constraint"). It is recorded rather than deleted
because the v1.0 rationale was written to satisfy it, and a reader comparing revisions is entitled to
see that the change was a decision.

What withdrawing S1 bought: v1.0 recorded as its single largest scientific cost that **all comparisons
would be internal** — detector variants against each other and against the [`SPEC.md`](SPEC.md) §6.4
baselines, with no direct numerical comparison to prior work on flip budgets, because the bit-flip
literature is almost entirely vision. That threat is retired, and **M-1b** (validate the BFA
reimplementation *before* generating the fault population) is the gate that makes the benefit real.

What it cost: the deployment argument (§10.1), and a sharper **C1** problem. On audio, omitting the
Mahalanobis OOD baseline was unremarkable. On CIFAR-10 it is the single most expected baseline in the
vision anomaly-detection literature, so C1 now risks reading as an oversight rather than a constraint.
The mitigation is pre-emptive, not defensive: [`SPEC.md`](SPEC.md) §6.4 states the exclusion and its
substitutions in the methods text, and §11.4's grep guard prevents accidental reintroduction.

---

## 12. Licensing and ethics

| Artefact | Licence | Obligations |
|---|---|---|
| CIFAR-10 | No formal licence; freely distributed for research | Cite Krizhevsky (2009). Kaggle copies are governed by the competition rules — academic use is within terms; no redistribution of the archives occurs here. |
| CIFAR-10-C | CC BY 4.0 | Cite Hendrycks & Dietterich (2019). |

**No personal data requiring ethics review.** CIFAR-10 is 32×32 images of ten common object and animal
categories, with no identifiable individuals as the subject of any class.

**One provenance note, stated rather than omitted.** CIFAR-10 was curated from the 80 Million Tiny
Images collection, which its authors **withdrew in 2020** after offensive labels and slurs were found
among its automatically-harvested WordNet noun classes. CIFAR-10's ten classes are common objects and
animals, hand-verified by the curators, so that specific problem does not carry over — but the
provenance is worth one sentence in the thesis rather than a discovery by a reader.

---

## 13. Acquisition checklist

Run before M-1 ([`SPEC.md`](SPEC.md) §12). **Verify these counts against the actual download rather
than against this document** — the figures here are from published descriptions.

**Acquisition**

- [ ] Kaggle competition rules accepted; `kaggle competitions list --group entered` confirms it
- [ ] `train.7z` + `trainLabels.csv` downloaded and extracted; **`test.7z` deliberately not fetched** (§3.1)
- [ ] Image count matches 50,000; all 32×32 RGB; `trainLabels.csv` covers ids 1…50000 over the 10 classes
- [ ] Per-class counts confirm 5,000 each — a stratification precondition, not a formality
- [ ] Canonical distribution downloaded for the labelled 10,000-image held-out pool; checksum recorded
- [ ] CIFAR-10-C downloaded; 15 corruptions × 5 severities present over 10,000 images each
- [ ] All checksums recorded in `artifacts/models/manifest.json`

**Integrity — failing tests, not comments**

- [ ] **Source disjointness** — zero pixel-hash collisions, Kaggle train vs canonical test (§5.1)
- [ ] **Label agreement** — `trainLabels.csv` agrees with canonical labels on a sampled subset (§5.1)
- [ ] **No GCN/ZCA anywhere in the input pipeline** (C1) — asserted, not reviewed by eye (§6)
- [ ] Channel statistics fitted on `train` only — leakage guard passing

**Splits and models**

- [ ] `train` / `clean_fit` / `clean_cal` / `clean_test` stratified split seeded and persisted; `train` disjoint from every detector split; probe pool disjoint
- [ ] VGG-11-BN reference accuracy pinned to a specific cited source in `manifest.json` (§7)
- [ ] Optional one-shot Kaggle leaderboard submission made as an independent M-1 check (§3.2, §7)

**Reconciliation**

- [x] [`SPEC.md`](SPEC.md) brought to v2.0 — all clauses in §10.3 updated, including §13's
      modality-bound row and §1's hypothesis scope

---

## Sources

- [CIFAR-10 / CIFAR-100 — Krizhevsky (2009), canonical distribution](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Kaggle — CIFAR-10: Object Recognition in Images (data)](https://www.kaggle.com/competitions/cifar-10/data)
- [Deep Residual Learning for Image Recognition (arXiv:1512.03385)](https://arxiv.org/abs/1512.03385) — ResNet-20 CIFAR-10 reference accuracy
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (arXiv:1409.1556)](https://arxiv.org/abs/1409.1556) — VGG
- [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (arXiv:1903.12261)](https://arxiv.org/abs/1903.12261) — CIFAR-10-C
- [CIFAR-10-C on Zenodo (10.5281/zenodo.2535967)](https://zenodo.org/records/2535967)
- [Do We Train on Test Data? Purging CIFAR of Near-Duplicates (arXiv:1902.00423)](https://arxiv.org/abs/1902.00423)
- [Bit-Flip Attack: Crushing Neural Network with Progressive Bit Search (arXiv:1903.12269)](https://arxiv.org/abs/1903.12269) — BFA
- [DeepHammer: Depleting the Intelligence of Deep Neural Networks through Targeted Chain of Bit Flips (USENIX Security 2020)](https://www.usenix.org/conference/usenixsecurity20/presentation/yao)
- [Targeted Bit Trojan / TBT (arXiv:1909.05193)](https://arxiv.org/abs/1909.05193)
- [Kaggle CLI documentation](https://github.com/Kaggle/kaggle-api)
