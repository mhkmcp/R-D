# 9. Contract traceability

*SPEC §14(6). Every clause of the thesis contract, mapped to what discharges it. Additions are listed
separately, so the line between what was promised and what was added is visible.*

## 9.1 Problem statement

| Contract clause | Discharged by | Status |
|---|---|---|
| "adversarial bit flips in model weights **or biases**" | Weight and bias flips are co-primary, both on the full grid — neither is a reduced arm (§4.1) | design fixed |
| "can affect internal calculations and cause the model to make incorrect predictions" | `SDC` and `CRASH` categories, Track H (§4.3) | design fixed |
| "impact depends on the specific parameter, bit position, and network layer affected" | Stratified grid over bit strata × layer buckets × budgets, with mandatory per-axis breakdowns (§4.2, §9.1) | design fixed; numbers ⏳ 5.7 |
| "Some faults substantially change the final prediction, whereas others induce smaller internal deviations that remain unnoticed" | The taxonomy's two tracks; Track S is exactly the "remain unnoticed" case (§4.3) | design fixed; numbers ⏳ 5.2 |
| "Many existing approaches rely mainly on the final output" | Output-only baselines — max softmax, entropy, margin, energy (§6.4) | implemented; numbers ⏳ 5.4 |
| "a fault can influence internal states across multiple layers as it propagates" | H4 localisation: first-alarm tap, propagation depth, residual vs plain (§9.2) | implemented; numbers ⏳ 5.10 |
| "enable earlier fault detection, before effects are clearly visible at the output" | Track S detection reported **as detection**, not as false alarm; detection latency in inferences before the first SDC (§9.3) | design fixed; numbers ⏳ 5.4 |

## 9.2 Objective

| Contract clause | Discharged by | Status |
|---|---|---|
| "determine whether multi-layer internal states can reliably detect fault-injection attacks" | **H1**, evaluated separately per track with a pre-registered rejection criterion (§1) | ⏳ 5.4 |
| "model normal layer-wise behavior" | Per-tap descriptor + robust normaliser fitted on clean data only (§5.2, §5.3) | implemented |
| "use SVDD-based anomaly detection to flag deviations" | RBF SVDD in ν-one-class form; D1/D2/D3 variants (§6.1, §6.2) | implemented |
| "evaluating detection performance **and overhead**" | Detection and overhead reported together, never separately (C5, §9.4) | implemented; numbers ⏳ 5.11 |
| "across architectures and fault models" | ResNet-20 vs VGG-11-BN; weight, bias, BatchNorm-buffer, stuck-at, whole-value and transient activation faults | implemented; numbers ⏳ 5.7 |
| *Secondary:* "whether fused multi-layer monitoring outperforms per-layer monitoring" | **H2**: D2/D3 against the best D1 tap (§6.2) | ⏳ 5.5 |
| *Secondary:* "identify minimal monitoring configurations that preserve high detection accuracy" | **H3**: K sweep and the TPR-vs-latency front (§9.6(1)) | ⏳ 5.12 |

## 9.3 Proposed technical directions

| Contract clause | Discharged by | Status |
|---|---|---|
| "extract compact, informative features from selected internal layers" | 26 features per tap from Blocks A–D, plus 3 cross-tap features at fusion — fixed width, permutation-invariant, independent of layer width (§5.2) | implemented |
| "during both normal and fault-injected inference" | The same hooks produce clean features during extraction and fault features during the injection sweep | implemented |
| "faults introduced primarily as bit flips in weights and biases" | Both on the full grid (§4.1) | implemented |
| "varied across layers and bit positions to cover both obvious and subtle error patterns" | Layer buckets × five bit strata, stratified and never cut; per-stratum reporting is what keeps this honest (§4.2) | implemented |
| "SVDD trained on clean executions" | C2 enforced structurally: clean and fault features are distinct types, so fitting on faults is a type error | implemented |
| "(1) fuse multi-layer features into a joint representation" | **D2** early fusion, and **D3** late fusion over per-tap CDFs | implemented |
| "(2) train per-layer SVDD models to localize where anomalies first emerge and how they propagate" | **D1** per-tap detectors; first-alarm tap and propagation depth (§9.2) | implemented; numbers ⏳ 5.10 |
| "Evaluation will span multiple fault-injection scenarios and models" | M2 and M3, FP32 and INT8, six fault modes, three attacker levels | implemented |

## 9.4 Named evaluation metrics

Every metric the contract names, and where it appears:

| Contract metric | Reported as | Status |
|---|---|---|
| detection rate | TPR at calibrated thresholds, per track | ⏳ 5.4 |
| false-positive rate | measured on clean test data against nominal α, with a binomial bound at 0.1 % | ⏳ 5.3 |
| precision | at the calibrated threshold, per track | ⏳ 5.4 |
| recall | at the calibrated threshold, per track | ⏳ 5.4 |
| F1-score | at the calibrated threshold, per track | ⏳ 5.4 |
| ROC-AUC | AUROC **plus AUPR**, since the classes are unbalanced | ⏳ 5.4 |
| runtime overhead | absolute ms then %, per tap count, batch, device and precision | ⏳ 5.11 |
| memory overhead | peak memory monitor-on vs monitor-off, plus detector size | ⏳ 5.11 |
| "whether multi-layer monitoring yields a measurable advantage over single-layer monitoring" | H2's paired-bootstrap difference, with "no detected difference" as a permitted outcome | ⏳ 5.5 |

## 9.5 Additions — not promised by the contract

Listed separately so an examiner can see what is extra rather than owed. Each could be cut without
breaking a commitment:

| Addition | Why it is here | Cut cost |
|---|---|---|
| L2 adaptive attacker (chapter 6) | A detector evaluated only against oblivious attackers answers an easier question | None to the contract |
| Stuck-at and whole-value fault modes | Supersets of the contract's bit flips | None |
| Transient activation faults | Covers the transient case the contract does not name | None |
| Deep SVDD (D4) | Only if the built variants leave headroom | None |
| Extended tap sets, extra seeds, full INT8 parity | Bought with budget freed by the single-dataset design (chapter 10) | Forgoes planned gains only |
| CIFAR-10-C confounder study | Tests the "generic novelty detector" threat directly (§7.1) | Would leave the central threat unmeasured |

Two things in this study are **not** cuttable despite not being contract clauses: the exclusion of the
covariance-based distance (a standing project constraint, chapter 3.6), and the quantised arm of the
primary model, which is the only point of external comparison the design has.
