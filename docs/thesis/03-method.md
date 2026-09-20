# 3. Method

## 3.1 Where the network is observed

Observation points ("taps") are fixed identity modules placed in the model at construction time, so
the same registry addresses both the FP32 model and its INT8 conversion. Each model exposes a
**default** set and, on the primary models, an **extended** set:

| Model | Default | Extended |
|---|---|---|
| ResNet-20 | 6: stem, three stage outputs, pooled, logits | 13: stem, nine block outputs, one pre-addition tap, pooled, logits |
| VGG-11-BN | 7: five block outputs, pooled, logits | 10: eight convolution outputs, pooled, logits |

One extended ResNet tap observes the residual branch **before** the skip addition. Post-addition taps
see `F(x) + x`, where the identity path can mask a perturbation inside `F`; the pre-addition tap sees
the branch alone. That difference is the mechanism H4 measures, and it is the reason the residual and
plain architectures are compared on identical inputs.

## 3.2 What is recorded per tap

Each tap produces a fixed-width descriptor per sample, computed in a single reduction pass on the
activation's own device:

| Block | Features | n |
|---|---|---|
| A — distributional shape | mean, std, min, max, five quantiles | 9 |
| B — sparsity and saturation | zero fraction, fraction above the clean p99.9, NaN count, Inf count | 4 |
| C — energy | L1/√n, L2/√n, L∞, peakiness, channel-energy Gini | 5 |
| D — structure | channel-energy entropy, top-1/2/4 channel shares, and the mean and std of per-channel spatial means and stds | 8 |
| E — cross-layer (added at fusion) | L2 ratio, entropy ratio and saturation difference against the previous tap | 3 |

**29 features per tap.** The descriptor is invariant to permutations of channels and spatial
positions, and its width does not depend on the layer's channel count. That property is what makes H3
answerable at all: monitoring cost per tap is constant, so "how many taps do we need" is a question
about K rather than about which layers happen to be wide.

The saturation level in Block B is the clean 99.9th percentile per tap, fitted on the classifier's
held-out calibration data and then frozen.

## 3.3 Normalisation

Features are standardised per dimension as `z = (x − median) / (IQR + δ)`, with the median and IQR
taken from clean fitting data only and then frozen. Features whose clean spread falls below a floor
are dropped at fit time and the drop list is persisted with the detector.

This is **diagonal scaling**. It rescales each feature independently and introduces no cross-feature
metric. The RBF kernel's `gamma` absorbs the remaining isotropic scale.

## 3.4 Detector

The detector is Support Vector Data Description with an RBF kernel, in its ν-one-class-SVM form. The
anomaly score is the negative decision function, so a positive score means the sample fell outside
the learned boundary, and higher always means more anomalous — for the SVDD and for every baseline,
so that thresholds read the same way everywhere.

Four variants are the study's independent variable:

| ID | Variant | What it answers |
|---|---|---|
| D1 | one SVDD per tap | the per-layer baseline for H2, and the propagation map for H4 |
| D2 | early fusion: all taps concatenated into one SVDD | the fused claim in H1 |
| D3 | late fusion: per-tap scores mapped through each tap's clean empirical CDF, then combined by max, mean or false-positive-weighted mean | fusion without a single high-dimensional fit; supplies the first-alarm layer |
| D4 | Deep SVDD | an extension, built only if D2 and D3 leave headroom |

Hyperparameters (ν, γ) are selected on **clean data only**: fit on the clean fitting split, measure
the false-positive rate on the clean calibration split, and keep the pair with the lowest rate whose
support-vector fraction stays within a stability band around ν. Selecting by fault scores or by test
AUROC would make the reported numbers meaningless, so the selection function takes no fault data —
this is enforced by the type system, not by review.

## 3.5 Calibration

Thresholds are the (1 − α) empirical quantile of the clean calibration scores, for α fixed in advance
at 5 %, 1 % and 0.1 %. At 0.1 %, on a few thousand clean samples, the quantile is a noisy estimate, so
a Clopper-Pearson upper bound on the false-positive rate is reported beside it: zero alarms in two
thousand samples is not "0 %".

Two alarm regimes are always reported together: per-inference, and windowed m-of-n over consecutive
inferences at (1,1), (8,3) and (32,8). Persistent faults make windowed detection easy, which is
exactly why the per-inference number never appears without it.

## 3.6 Baselines, and a note on an excluded one

Output-only baselines: maximum softmax probability, softmax entropy, logit margin, energy score.
Feature-space baselines, on the same normalised features as D2 so that the comparison isolates the
detector rather than the features: Isolation Forest, Local Outlier Factor in novelty mode, Gaussian
kernel density, PCA reconstruction error and a shallow autoencoder reconstruction error.

**On the distance this study excludes.** A reader working on CIFAR-10 anomaly detection will expect
the class-conditional distance of Lee et al. (2018) — the canonical OOD baseline on this dataset,
built from class-conditional means and a shared inverse covariance estimate. It is excluded here by a
standing constraint of the project, and the exclusion is structural rather than incidental: no
covariance estimator, no inverse covariance, no full-covariance whitening, and no whitened PCA
followed by a Euclidean metric appears anywhere in the codebase, because whitening followed by a
Euclidean metric is that same distance under another name. The density-baseline role it would play is
filled instead by kernel density estimation, Local Outlier Factor and Isolation Forest, all fitted on
the identical feature matrix, so the comparison the reader wants — "does the SVDD beat a
density-based alternative on these features?" — is answered, by three alternatives rather than one.
Per-channel input normalisation is diagonal and is therefore not affected by this exclusion. A
grep-based test over the source tree enforces the constraint on every commit.
