# 5. Results

> **Status: skeleton.** Every table below is bound to the artifact that fills it and to the command
> that produces that artifact. No number is written here by hand. Sections marked ⏳ are waiting on a
> milestone that has not been run; the gate each one depends on is named.

## 5.1 Models and the accuracy gate ⏳

*Fills from* `artifacts/models/manifest.json` · `uv run fidnn train m2` / `m3`

Clean accuracy for each model and precision, against its pinned published reference, with the seed
count. A model outside one point of its reference does not proceed to any fault experiment, so this
table gates everything after it.

## 5.2 The fault population ⏳

*Fills from* `artifacts/faults/*_outcomes.parquet` · `docs/M2_taxonomy.md` · `uv run fidnn inject`

Outcome distribution per model, precision and fault mode: MASKED, DEGRADED, SDC and CRASH shares,
with the Track S share called out. **Gate:** Track S below roughly 15 % means H1's discriminating test
cannot be run on this architecture, and the study escalates to a deeper network rather than reporting
a Track S number from a near-empty population.

Broken out per bit stratum, layer bucket and flip budget, because the whole point of the stratified
grid is that these differ.

## 5.3 Calibration ⏳

*Fills from* `artifacts/detectors/*_calibration.parquet` · `uv run fidnn fit`

Measured false-positive rate on clean test data against each nominal α, per detector variant, with the
Clopper-Pearson bound at α = 0.1 %. A measured rate far from nominal is a calibration failure and is
reported as one rather than absorbed into the detection numbers.

## 5.4 Headline detection — H1 ⏳

*Fills from* `artifacts/results/*_headline.parquet` · `make reproduce`

TPR at 1 % and 0.1 % calibrated FPR, **Track S and Track H in separate rows**, for D1, D2, D3 and every
baseline, on M2 and M3 in both precisions. Each row carries its seed count and 95 % confidence
interval; the fused-versus-best-output-only difference carries a paired-bootstrap interval, and an
interval containing zero is written "no detected difference".

The Track S comparison is the one that decides H1. Track H is reported in full but is expected to be
the easier half, and is not the discriminating test.

## 5.5 Fusion versus single layers — H2 ⏳

*Fills from* the same artifacts, D1 rows against D2 and D3 rows.

Best single tap against fused, per track. H2 fails if fusion is no better than the best per-layer
detector within the interval — which would leave a simpler method doing the same work.

## 5.6 Unseen configurations ⏳

*Fills from* the `fault_gen` rows of `artifacts/results/*_headline.parquet`

Configurations deliberately never sampled into development: reported as their own row, never merged
into the headline, so that generalisation to unseen fault configurations is a separate claim rather
than a hidden component of the main number.

## 5.7 Mandatory breakdowns ⏳

*Fills from* `artifacts/results/*_breakdowns.parquet`

Per outcome category, bit stratum, layer bucket, flip budget, fault mode (BatchNorm buffers separate
from weights), precision, and attacker level. An aggregate-only version of this chapter would conceal
the structure the design exists to expose.

## 5.8 Precision: FP32 against INT8 ⏳

*Fills from* the precision axis of the same artifacts.

Reported here rather than among the later ablations, because the quantised arm is where this study's
flip budgets are comparable with published attacks, and a negative INT8 result would cost the dataset
rationale — it must not surface late.

## 5.9 External anchor: the L1 attacker ⏳

*Fills from* `artifacts/attacks/bfa_*.parquet` · `docs/M1b_bfa.md` · `uv run fidnn attack bfa`

Flips needed to drive M2 INT8 to chance accuracy, against the published figures for the same
architecture and bit width (Rakin et al. 2019, Table 2: 7, 10, 10, 12, 17 over five trials). Presented
as a sanity check on the attack implementation, **not** as a detector claim.

## 5.10 Localisation — H4 ⏳

*Fills from* the `localisation` block of `artifacts/results/*_eval.json`

Spearman ρ between injected layer and first-alarm tap, the injected-versus-alarm bucket confusion
matrix, and mean propagation depth — **separately for the residual and plain architectures, because
the difference between them is the result**. The pre-addition tap share tests whether residual first
alarms concentrate where the skip path would otherwise mask the perturbation.

## 5.11 Overhead — C5 ⏳

*Fills from* `artifacts/overhead/*_overhead.parquet` · `uv run fidnn overhead`

Added latency in absolute milliseconds first and percentage second with the baseline named, peak
memory, sustained throughput, detector size, and the split between hook cost and scoring cost — per
tap count, batch size, device and precision. Detection quality is not reported without these numbers;
they are one result, not two.

The M-0 calibration already indicates what to expect: monitoring six taps on ResNet-20 at batch 1
costs +67 % in FP32 and +96 % in INT8 over the bare forward pass, so the H3 design target of under
10 % is not met by the unoptimised descriptor at K = 6, and the K sweep in chapter 5.12 is where that
budget is either met or reported as missed.

## 5.12 Minimal configurations — H3 ⏳

*Fills from* the K sweep and the TPR-versus-latency front.

Greedy forward tap selection by a **clean** criterion, then the Pareto front of detection against
latency. H3 asks whether at most three taps retain 95 % of the full detection rate; missing the
latency target is a deployment caveat on a confirmed hypothesis, not a rejection.
