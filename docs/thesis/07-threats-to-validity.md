# 7. Threats to validity

Each threat below states what could make a conclusion wrong, what the design does about it, and — where
the answer is "nothing" — what the resulting bound on the claim is. Threats that are managed and
threats that are merely bounded are kept visibly apart, because a reader deciding how much to trust a
number needs to know which kind they are looking at.

## 7.1 Construct validity — is the thing measured the thing claimed?

**The detector may be a generic novelty detector rather than a fault detector.** This is the central
threat to the whole thesis. A monitor that fires on *anything* unusual — an unusual input, a
distribution shift, an unfamiliar lighting condition — would produce the same detection numbers
without supporting any claim about faults. The design measures this directly rather than arguing
about it: CIFAR-10-C is scored as a clean-input shift, never mixed with faults, broken down per
corruption type (15) and per family (4), with the fault-score slope and the severity slope shown side
by side. If the two slopes look alike, the honest reading is that a large part of what the monitor
detects is novelty, and the thesis says so. This breakdown is mandatory rather than presentational
precisely because it is the result most able to undermine the headline.

**Faults could be trivially detectable.** If exponent-MSB flips dominate the fault population, high
detection rates say little. Bit strata are sampled by stratum rather than uniformly, results are
reported per stratum, and strata are never cut for compute. A reader can therefore see the detection
rate on the subtle strata separately from the obvious ones.

**The two tracks measure different things and must not be merged.** Output-only baselines do well on
faults that already changed the output, so a merged "fault" class lets Track H carry the headline and
H1 becomes a claim about the easy half. Tracks are separate in every table, and H1 is evaluated
separately on each, with Track S named as the discriminating test.

**ε defines the MASKED/DEGRADED boundary and is a free parameter.** It is fixed a priori, recorded in
every artifact's sidecar, and never tuned against results. It nonetheless moves the Track S
composition: a smaller ε moves probes from MASKED to DEGRADED. The split within Track S is therefore
reported, not just the track total.

**Not every planned fault is an effective fault.** Some flips land on parameters whose value the
quantised representation cannot express differently, or on already-zero values, and change nothing.
Counting those as detected or undetected faults would bias both directions. The injection records
carry what was actually applied, so no-effect injections are identifiable.

## 7.2 Internal validity — could the numbers be right for the wrong reason?

**Leakage from fault data into fitting or calibration.** This would inflate every detection number
through a route invisible in the results. Three defences: clean and fault feature containers are
distinct types, so every fit, hyperparameter selection and threshold path rejects fault data as a type
error rather than as a review finding; the hyperparameter search takes no fault argument at all; and a
leakage guard fails the results build on fault records in clean splits, instances in two splits,
reserved configurations in the development split, or a missing grid cell.

**Threshold selection after seeing results.** Thresholds are the (1 − α) quantile of clean calibration
scores at α fixed in advance. No threshold is chosen to maximise F1 or TPR, and none is chosen on a
test split.

**Silent state corruption across the sweep.** If a restore ever failed, every subsequent injection
would be measured on a corrupted model. Injection is in-place and XOR-restored inside a `finally`
block, and a checksum over the full state — parameters **and** buffers, since BatchNorm statistics are
buffers — is asserted periodically; a mismatch aborts the sweep loudly rather than producing results.
During development this guard found a real defect in itself: for quantised linear layers the digest
was reading a truncated textual summary that both hid small flips and changed on every repack. It now
hashes tensor bytes structurally.

**Clean and fault records could differ by something other than the fault.** They are drawn from the
same held-out pool, and the classifier trained on neither, so a seen-versus-unseen difference cannot
be mistaken for a fault signal.

**Batch size changes features slightly.** Convolution reductions are not associative, so extracting at
a different batch size perturbs features by roughly 1e-4 relative. Determinism holds per batching, and
the extraction batch size is recorded in the sidecar; comparisons across configurations use the same
batching.

**Single-run numbers.** Every reported number carries at least five seeds and a confidence interval,
and the report refuses to present fewer as a result. Comparisons use a paired bootstrap on the shared
evaluation set, and an interval containing zero is written "no detected difference" rather than being
described as a win.

## 7.3 External validity — how far do the claims reach?

These are bounds, not managed risks. Nothing in the design removes them.

**One modality.** Every claim is about convolutional image classifiers on 32×32 inputs. Audio,
tabular and sequence models are not evaluated. The Block-D downgrade ablation gives *indirect*
evidence about how much the method leans on convolutional structure, by zeroing the per-channel
spatial statistics and re-measuring. Indirect evidence is not a second modality and is not written as
if it were.

**One scale.** The models evaluated span roughly 0.27 M to 9.8 M parameters. Nothing here bounds
behaviour at ImageNet scale, and no such claim is made.

**One dataset.** CIFAR-10 alone. The variation the contract asks for is delivered across
architectures, fault models, bit strata and precisions rather than across datasets. Chapter 10 states
what that choice removed and what substitutes for each loss.

**Near-duplicates inside CIFAR-10.** About 3.3 % of the canonical test images have a near-duplicate in
the training set. The pixel-hash guard catches exact duplicates only, so the probe pool is marginally
less independent of the calibration data than the split table implies. This is a dataset-level limit,
not a leak: no record appears in two splits.

**One machine.** Overhead is measured on a single edge-class machine, fully specified. Ratios between
configurations should transfer; absolute milliseconds will not.

**Persistent faults are primary.** Transient activation faults are a secondary arm on a reduced grid
and are reported apart from the main tables.

## 7.4 Conclusion validity — are the statistics sound?

**Unbalanced classes.** Fault and clean populations differ in size, so AUROC alone can flatter; AUPR
is reported beside it throughout.

**Quantile estimates at 0.1 %.** On two thousand clean samples, the 0.1 % quantile rests on a couple of
order statistics. A Clopper-Pearson upper bound accompanies the measured rate, so "zero false
positives" is never reported as a rate of zero.

**Windowed detection is easy for persistent faults.** A persistent fault is present on every
inference, so an m-of-n window has many chances to fire. Per-inference numbers are always reported
alongside windowed ones so the easier regime cannot stand alone.

**Multiplicity.** The mandatory breakdowns produce many comparisons, and some will look significant by
chance. Hypothesis verdicts rest on the pre-registered headline comparisons in chapter 1, not on the
breakdowns, which are descriptive.

## 7.5 Threats specific to the attack side

**The attacker reimplementation could be wrong.** If the L1 attacker is not faithful, the fault
population it generates is unrepresentative and the one external anchor this study has is worthless.
It is validated against the published flip-budget figures for the same architecture and bit width
*before* it generates any faults, and the comparison is reported as a figure. Known differences that
are not in themselves bugs — per-channel rather than per-layer quantisation, a slightly different
training set — are stated with it.

**The adaptive attacker is one attacker, not the worst case.** A monitor that survives this particular
detector-aware search is not thereby secure. The result is reported as the cost imposed on this
attacker: an increased flip budget is a real gain even where the monitor remains evadable, and a
trial the monitor blocked is not proof of security.

**The adaptive attacker knows the threshold exactly.** That is deliberate — it is the strong form of
the assumption — but it means the reported evasion cost is a lower bound on what an attacker with
partial knowledge would pay.

## 7.6 What would change the conclusions

Stated in advance, so that a negative result is legible rather than explained away:

- Track S detection no better than output-only scores within the confidence interval would reject H1
  on the discriminating track, whatever Track H shows.
- Fused monitoring no better than the best single tap would reject H2 and remove the motivation for
  multi-layer monitoring, leaving a simpler per-layer method.
- A Track S population below roughly 15 % of injections would mean the discriminating test cannot be
  run on this architecture at all, and the study escalates to a deeper network before drawing
  conclusions.
- CIFAR-10-C scores that track fault scores closely would mean the monitor is substantially a novelty
  detector, which is a finding to report rather than a failure to hide.
- Overhead far above the deployment budget would leave H1 and H2 intact but make the method
  impractical at the configurations tested — a confirmed hypothesis with a deployment caveat, which
  is not the same thing as a failure.
