# Datasets

**Companion to:** [`SPEC.md`](SPEC.md) §3 (Target models and datasets) · **Governed by:** `new_thesis_contract.pdf`
**Status:** Draft v1.0 · **Date:** 2026-09-14

This document records which datasets the thesis uses, where to obtain them, how they are split and
preprocessed, and — at length, because it is the question an examiner will ask — why these two and
not the obvious alternatives.

The contract names no dataset. It requires only that the study evaluate *"across architectures and
fault models"* and that faults be injected *"as bit flips in weights and biases, varied across layers
and bit positions"*. Dataset choice is therefore an **addition** under [`SPEC.md`](SPEC.md) §0 (C0),
and it is justified here rather than assumed.

---

## 1. Selection criteria

Three constraints were fixed before any dataset was considered:

| # | Constraint | Origin |
|---|---|---|
| S1 | **No image datasets.** | Project requirement |
| S2 | **Laptop-class compute** — Apple MPS / CPU, no CUDA workstation. | Available hardware |
| S3 | **Two datasets, evaluated deeply** rather than four evaluated shallowly. | Supervisory scope |

Two further criteria follow from the thesis itself, and they did more to decide the outcome than S1–S3:

| # | Requirement | Why it binds |
|---|---|---|
| S4 | **≥ 5 usable tap points per model.** | H2, H3 and H4 are hypotheses about *layers*. A 3–4 layer network cannot support a multi-layer fusion claim or a fault-propagation result at all. This is what disqualifies the shallow tabular MLPs that dominate the NIDS literature from being the *primary* dataset. |
| S5 | **Channel-structured activations.** | [`SPEC.md`](SPEC.md) §5.2 Block D derives 8 of 29 per-tap features from channel structure (channel-energy entropy, top-k channel shares, per-channel spatial statistics). Dense taps fall back to a 4-feature variant and 22 features total. |
| S6 | **A severity-graded clean-input shift set.** | [`SPEC.md`](SPEC.md) §9.6(8) asks whether the detector responds to *faults* or to *anything unusual*. Without graded shifted-but-clean inputs, that question cannot be answered, and the thesis cannot rule out that it has built a novelty detector. |

S6 is the criterion most often overlooked, and it is decisive: a dataset with no natural shift suite
forces you to construct one and then defend it.

---

## 2. Primary dataset — Google Speech Commands v2

Used by **M1, M2, M3** ([`SPEC.md`](SPEC.md) §3). M2 is the primary model and carries the full
injection grid, every ablation and every baseline.

### 2.1 What it is

One-second recordings of single spoken English words, collected by Pete Warden at Google and
released for keyword-spotting research.

| Property | Value |
|---|---|
| Release | v0.02, April 2018 |
| Size | 105,829 utterances, 35 words |
| Format | 16-bit PCM mono WAV, 16 kHz, 1 second |
| Task used here | The standard **12-class** formulation |
| Classes | `yes no up down left right on off stop go` + `_silence_` + `_unknown_` |
| Licence | **CC BY 4.0** — free to use and redistribute with attribution |
| Reference | Warden, P. (2018), *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*, [arXiv:1804.03209](https://arxiv.org/abs/1804.03209) |

The 12-class task keeps ten fixed command words, folds the remaining 25 words into `_unknown_`, and
adds `_silence_` drawn from the dataset's own background-noise recordings. This is the formulation
MLPerf Tiny benchmarks, which is why it is used here rather than the raw 35-class task.

### 2.2 Download

```bash
# Training + validation archive (~2.3 GB)
curl -O http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

# Official held-out test archive, distributed separately
curl -O http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
```

- Landing page / paper: <https://arxiv.org/abs/1804.03209>
- Loader: `torchaudio.datasets.SPEECHCOMMANDS` (downloads the same archive; see
  [torchaudio docs](https://docs.pytorch.org/audio/master/_modules/torchaudio/datasets/speechcommands.html))
- Mirrors, if the TensorFlow host is unavailable: [Hugging Face `google/speech_commands`](https://huggingface.co/datasets/google/speech_commands)

### 2.3 Splits

Use the **official** split, not a random one. The main archive ships `validation_list.txt` and
`testing_list.txt`, and the test archive is the canonical held-out set.

**This matters for C2 (no leakage), not just for convention.** The official lists are assigned by
hashing the *speaker ID*, so a given speaker's recordings never straddle two splits. A random
per-file split would put the same speaker in both `clean_fit` and `clean_test`, and the SVDD would
be partly memorising speaker timbre rather than modelling clean layer-wise behaviour — inflating
detection numbers through a route that is invisible in the results.

The [`SPEC.md`](SPEC.md) §7 `clean_fit` / `clean_cal` / `clean_test` partition is carved from the
official train / validation / test sets respectively.

### 2.4 Preprocessing

- **M2 (DS-CNN):** 49 × 10 MFCC frame — 40 ms window, 20 ms stride, 10 MFCC coefficients — matching
  the MLPerf Tiny reference pipeline.
- **M3 (M5-Net):** raw 16 kHz waveform, 16,000 samples, no spectral transform at all.
- `_silence_` clips are generated by sampling one-second crops from `_background_noise_`.
- Standard augmentation (time shift ±100 ms, background noise mixing) during **training only**;
  clean feature extraction for the detector uses unaugmented audio, so the SVDD models the
  deployed inference distribution rather than the training distribution.

### 2.5 Accuracy target

The MLPerf Tiny reference DS-CNN reaches **92.2 %** on the 12-class task, measured on a 1,000-sample
subset of the test set; the benchmark's quality target is 90 %. `artifacts/models/manifest.json` must
land within 1 point of 92.2 % before any fault experiment runs ([`SPEC.md`](SPEC.md) §12, M-1).

### 2.6 Why this dataset

**The deployment story matches the threat model exactly.** Always-on keyword spotting runs on
embedded microcontrollers and phone DSPs. That is precisely the hardware class where Rowhammer,
clock/voltage glitching and laser fault injection are realistic rather than hypothetical — the
adversary [`SPEC.md`](SPEC.md) §2 assumes. On a datacentre-served model, the same threat model would
need an argument; here it needs none. It also makes the laptop-class overhead measurements in §9.4 a
*feature*: they are edge-class numbers for an edge-class deployment, not a compromise.

**It is a recognised benchmark with a published reference number.** MLPerf Tiny's KWS task gives an
external accuracy figure to validate the implementation against, which matters because the thesis
otherwise has no point of external comparison (§5 below).

**It is cheap enough that the injection grid fits the hardware.** A 49 × 10 input and a ~25k-parameter
DS-CNN is roughly an order of magnitude below a CIFAR-class model. The [`SPEC.md`](SPEC.md) §4.2 grid
— ~4,500 injections × 32 probes × two full-grid fault modes × 5 seeds — is minutes per sweep rather
than hours.

**Twelve classes preserves the subtle-fault regime.** This is a non-obvious but central point.
[`SPEC.md`](SPEC.md) §4.3 splits fault outcomes into Track S (`MASKED` ∪ `DEGRADED` — no change to
top-1) and Track H (`SDC` ∪ `CRASH`), and **Track S is where the contract's claim lives**: *"others
induce smaller internal deviations that remain unnoticed."* As class count rises, decision boundaries
pack closer together, any perturbation is likelier to flip top-1, and Track S empties out. A
10–12 class task keeps it populated. A 1000-class task would quietly delete the phenomenon the thesis
exists to study.

**It supplies the §9.6(8) confounder set for free.** The dataset ships `_background_noise_`, and
mixing it at controlled SNR is already standard practice in the benchmark pipeline. Severity-graded
clean-input shift is therefore *native* — no constructed perturbation to justify.

**One honest objection.** MFCC features are two-dimensional and DS-CNN applies 2-D convolutions, so
M2 can fairly be called an image model in disguise. M3 (M5-Net) is the answer: it consumes raw
waveform through 1-D convolutions, with no 2-D processing anywhere. Generality claims rest on M3 and
M4, not on M2 alone. This objection is recorded in [`SPEC.md`](SPEC.md) §3.1 rather than left for the
defense.

---

## 3. Secondary dataset — UNSW-NB15

Used by **M4** ([`SPEC.md`](SPEC.md) §3), which runs a confirmatory subset of the grid: `bf_w` and
`bf_b` full grid, other fault modes omitted.

### 3.1 What it is

Network flow records generated by the Australian Centre for Cyber Security (ACCS) at UNSW Canberra,
using the IXIA PerfectStorm tool to synthesise realistic modern traffic mixed with contemporary
attacks.

| Property | Value |
|---|---|
| Released | 2015 |
| Raw size | 2,540,044 records across four CSV files |
| Partitioned set | 175,341 training / 82,332 testing records |
| Task used here | 10-class `attack_cat`, plus a free Normal-vs-Attack collapse |
| Classes | Normal + Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms |
| Features | Mixed numeric and categorical (`proto`, `service`, `state`) |
| Licence | Free for **academic research in perpetuity**; commercial use requires agreement from the authors |
| Reference | Moustafa, N. & Slay, J. (2015), *UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems*, MilCIS |

### 3.2 Download

- Official project page: <https://research.unsw.edu.au/projects/unsw-nb15-dataset> — links to pcap,
  BRO, Argus and CSV source files plus the partitioned train/test CSVs
- IEEE DataPort mirror: DOI [`10.21227/8vf7-s525`](https://ieee-dataport.org/documents/unswnb15-dataset)
- Use **`UNSW_NB15_training-set.csv`** and **`UNSW_NB15_testing-set.csv`** (the partitioned set), not
  the four raw CSVs — the partitioned set is what published baselines report against.

**Citation obligation.** The authors require citing the foundational papers as a condition of use —
Moustafa & Slay (2015), Moustafa & Slay (2016), Moustafa et al. (2017) ×2, and Sarhan et al. (2020).
The full list is on the project page and must appear in the thesis bibliography. Contact for
licensing questions: Dr. Nour Moustafa, UNSW Canberra.

### 3.3 Preprocessing

Per [`SPEC.md`](SPEC.md) §3.2:

- Drop `id`; drop the binary `label` column when training the 10-class `attack_cat` head.
- One-hot encode `proto`, `service`, `state` with rare-category bucketing.
- **Deduplicate records** — the dataset contains duplicates that inflate accuracy if left in.
- **Fit the scaler and the categorical encoder on the training split only.** This is the classic
  leakage route on this dataset and a direct C2 violation; the [`SPEC.md`](SPEC.md) §7 leakage guard
  asserts it as a failing test.
- Fault probes are drawn from a **class-balanced probe set**, so §4.3 outcome rates are not
  confounded by prior class frequency.

### 3.4 Known limitations, stated up front

- **Class imbalance is severe.** Worms and Shellcode have very few records; per-class analysis for
  them is underpowered and is marked as such in [`SPEC.md`](SPEC.md) §13.
- **The official partitioned split is widely regarded as optimistic** relative to a temporally or
  subject-disjoint split. M4 is therefore treated as *confirmatory only* and never carries a headline
  result.
- **Label quality has been questioned** in the literature. Same mitigation: M4 confirms, M2 leads.
- **No corruption suite exists.** The §9.6(8) confounder ablation on M4 uses constructed graded
  numeric-feature perturbation at 5 severities, explicitly labelled as constructed and treated as
  weaker evidence than Speech Commands' native noise sweep.

### 3.5 Why this dataset

**The attack narrative is self-consistent.** An adversary flipping bits to blind an intrusion
detector is a coherently motivated attack, not a contrived scenario bolted onto a generic classifier.
The model being attacked *is* a security control, which makes the threat model argue for itself.

**It provides genuine modality contrast.** M4's dense taps run without Block D's four spatial
statistics — 22 features per tap instead of 29. That makes the M2-vs-M4 comparison a direct test of
whether the method depends on convolutional structure at all, and tells a deployer which feature
blocks actually carry signal ([`SPEC.md`](SPEC.md) §9.6 item 3). A degraded M4 result is a *finding*,
not a failure.

**The binary view is free.** Collapsing 10-class `attack_cat` predictions to Normal-vs-Attack
relabels existing predictions rather than re-running anything, and it is the operationally meaningful
metric for an IDS. Both are reported from the same sweep.

**One deliberate deviation.** The NIDS literature standard is a 2–4 layer MLP. That fails S4, so M4
uses an 8–10 layer residual MLP. Depth here is **required by the research question**, not chosen for
accuracy — and the thesis must show it costs nothing by validating against published shallow
baselines on both the 10-class and binary views ([`SPEC.md`](SPEC.md) §3.2). Left unexplained, the
depth would read as gratuitous.

---

## 4. Division of labour

Neither dataset is asked to do a job it cannot do:

| Job | Speech Commands (M2/M3) | UNSW-NB15 (M4) |
|---|---|---|
| Full §4.2 injection grid | ✅ all fault modes | Confirmatory: `bf_w`, `bf_b` only |
| All ablations and baselines | ✅ | Modality contrast only |
| §9.6(8) confounder evidence | ✅ native noise SNR sweep | Supporting, constructed |
| Architecture contrast (H4) | ✅ plain vs residual | — |
| Modality contrast (S5) | — | ✅ conv vs dense taps |
| Security deployment narrative | Edge/embedded | ✅ inline network sensor |
| Non-2-D processing evidence | ✅ via M3 raw waveform | ✅ tabular |

---

## 5. Alternatives considered and rejected

Recorded because "why not the obvious dataset?" is a predictable examination question.

| Candidate | Why rejected |
|---|---|
| **CIFAR-10 / ResNet** | Excluded by S1. Would otherwise have been primary: cheap, deep, and CIFAR-10-C gives the §9.6(8) shift suite directly. |
| **MNIST / LeNet-5** | Excluded by S1, and fails S4 regardless — too shallow to support a multi-layer claim, and its feature distribution is tight enough to make detection look trivially easy. |
| **Shallow tabular NIDS as primary** (CIC-IDS2017, NSL-KDD, UNSW-NB15 with a standard MLP) | Fails S4 (2–4 layers) and weakens S5. Retained as *secondary* with a deliberately deepened model instead. |
| **UCI HAR** (wearable sensors) | Viable and cheap, but weaker narrative than UNSW-NB15 and no security framing. Would have been the lower-risk secondary choice; the trade was narrative strength for a Block-D-free feature variant. |
| **ESC-50 / UrbanSound8K** | Too few clips to fit an SVDD comfortably, and still spectrogram-CNN — no contrast against M2. |
| **Tiny-ImageNet / CIFAR-100** | Excluded by S1; also confound resolution with class count. |
| **Full ImageNet** | Excluded by S1 and by S2 — 50–100× the sweep cost. |
| **A text / transformer task** | Deep enough, but tokenisation and embedding layers complicate the semantics of a weight bit flip, and the models are far above the S2 budget. |

### The cost of S1, stated plainly

The bit-flip attack literature — BFA, DeepHammer, TBT, ProFlip — is almost entirely vision. Excluding
images means the thesis has **no direct numerical comparison to prior work** on flip budgets or attack
success rates. All comparisons are therefore *internal*: detector variants against each other and
against the baselines in [`SPEC.md`](SPEC.md) §6.4. This is recorded as a threat to validity
([`SPEC.md`](SPEC.md) §13) rather than quietly omitted, and it is the single largest scientific cost
the dataset choice incurs.

---

## 6. Licensing and ethics summary

| Dataset | Licence | Obligations |
|---|---|---|
| Speech Commands v2 | CC BY 4.0 | Attribute Warden (2018). Redistribution permitted. |
| UNSW-NB15 | Academic research, in perpetuity; commercial use by agreement | Cite the five required papers. Do not redistribute. Thesis use is academic and within terms. |

Neither dataset contains personal data requiring ethics review: Speech Commands is consented
volunteer speech with no speaker identities published, and UNSW-NB15 is synthetic traffic generated
in a cyber range rather than captured from live users.

---

## 7. Acquisition checklist

Run before M-1 ([`SPEC.md`](SPEC.md) §12). **Verify these counts against the actual download rather
than against this document** — the figures here are from published descriptions.

- [ ] Both Speech Commands archives downloaded; checksums recorded in `artifacts/models/manifest.json`
- [ ] Official `validation_list.txt` / `testing_list.txt` split reproduced; speaker-disjointness
      asserted as a test
- [ ] 12-class mapping built: 10 words + `_unknown_` fold + `_silence_` from `_background_noise_`
- [ ] Utterance count matches 105,829; sample rate 16 kHz throughout
- [ ] UNSW-NB15 partitioned CSVs downloaded; row counts match 175,341 / 82,332
- [ ] Column inventory recorded, `id` and `label` dropped for the `attack_cat` head
- [ ] Duplicates removed; per-class counts logged, rare classes flagged
- [ ] Scaler/encoder fitted on train split only — leakage guard passing
- [ ] Five UNSW-NB15 citations added to the bibliography

---

## Sources

- [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition (arXiv:1804.03209)](https://arxiv.org/abs/1804.03209)
- [Google Speech Commands on Hugging Face](https://huggingface.co/datasets/google/speech_commands)
- [torchaudio SPEECHCOMMANDS loader](https://docs.pytorch.org/audio/master/_modules/torchaudio/datasets/speechcommands.html)
- [MLPerf Tiny Benchmark (arXiv:2106.07597)](https://ar5iv.labs.arxiv.org/html/2106.07597)
- [MLCommons Tiny benchmark repository](https://github.com/mlcommons/tiny)
- [The UNSW-NB15 Dataset — UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [UNSW-NB15 on IEEE DataPort](https://ieee-dataport.org/documents/unswnb15-dataset)
- [Very Deep Convolutional Neural Networks for Raw Waveforms (arXiv:1610.00087)](https://arxiv.org/abs/1610.00087)
