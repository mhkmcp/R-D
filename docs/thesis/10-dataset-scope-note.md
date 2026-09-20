# 10. Dataset scope note

*SPEC §14(7). What restricting the study to one dataset removed, what substitutes for each loss, and
what the freed budget bought. Paired with the modality bound in chapter 7.3: this note explains the
decision, that bound states its consequence.*

## 10.1 The decision

The study evaluates on **CIFAR-10 alone**. An earlier design spanned additional datasets and
modalities. The narrowing was deliberate, taken before results existed, and it is on the record here
rather than presented as though one dataset had always been the obvious choice.

The reasoning: the object of study is the *detector*, not the classifier. Variation that tests a
fault detector is variation in architecture, fault model, bit position, layer and precision — and all
of that is available within one dataset. Variation in dataset mostly tests the classifier. Spreading a
fixed compute budget across datasets would have bought less of the variation that matters and more of
the variation that does not.

CIFAR-10 in particular is where the bit-flip attack literature reports its numbers, which is what
makes this study's flip budgets comparable to published ones at all. That comparability is the only
external anchor the design has.

## 10.2 What it removed, and what substitutes

| Lost | Substitute | Honest assessment |
|---|---|---|
| Cross-dataset generalisation evidence | None. Chapter 7.3 states the bound rather than filling it | **No substitute.** A single-dataset study cannot speak to dataset transfer |
| Cross-modality evidence (audio, tabular, sequence) | The Block-D downgrade ablation: zero the per-channel spatial statistics and re-measure, which probes how much the method leans on convolutional structure | **Indirect only.** It is not a second modality and is never written as one |
| A second input distribution for confounder analysis | CIFAR-10-C: 15 corruption types × 5 severities, per type and per family | Strong substitute for *this* purpose — it tests input shift, which is the confounder that matters |
| Independent accuracy validation across datasets | The pinned published reference accuracy per model, plus an optional one-shot leaderboard submission | Adequate: it catches a pipeline error affecting both models equally |
| Evidence at larger input resolution and model scale | None | **No substitute.** Chapter 7.3 states the scale bound |

## 10.3 What the freed budget bought

The compute not spent on additional datasets was committed to planned work *before* the budget
appeared, so that it would be reinvested rather than quietly absorbed. Sized against the measured
throughput (chapter 4.4), the plan funds, in priority order:

1. **More seeds** — ten rather than the minimum five, tightening every confidence interval.
2. **Extended tap sets on both primary models** — 13 taps on the residual model, 10 on the plain one,
   making the extended configuration the planned one rather than a sweep-only curiosity.
3. **The adaptive attacker** on the quantised primary model, which is what raises the study above a
   purely empirical detection benchmark.
4. **Full precision parity** — both primary models run the complete grid in FP32 and INT8, rather than
   the quantised arm being a partial add-on.

The measured cost of the committed workload is ≈89 hours against a 250-hour budget, and the
reinvestment plan brings it to ≈230 hours: the freed budget is spent, not banked.

## 10.4 How this should be read

The single-dataset design is a scope decision with a stated cost, not an omission. Two claims follow
from it and both appear in chapter 7.3 as bounds rather than caveats: this study says nothing about
non-vision modalities, and nothing about scales beyond those evaluated. Where the design compensates —
architecture contrast, fault-model breadth, bit-strata coverage, precision parity, the confounder
study — it compensates in the dimensions that test a *fault detector*. Where it cannot compensate, it
says so.
