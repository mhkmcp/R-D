# 1. Introduction

## 1.1 The problem

Deep neural networks are deployed where wrong predictions carry real cost: on-device vision in
cameras, drones and driver-assistance hardware, and on shared commodity servers. In both settings an
attacker can reach the memory holding the model's parameters. Rowhammer flips DRAM bits from a
co-located process; laser injection and clock or voltage glitching flip them with physical access.
A flipped bit in a weight or a bias changes the arithmetic of one layer, and that change then
propagates through everything downstream.

What makes such faults hard to catch is that their effect is not uniform. It depends on which
parameter was hit, which bit within it, and which layer the parameter belongs to. A flip in a
floating-point exponent can destroy the prediction outright. A flip deep in a mantissa may leave the
top-1 prediction untouched while still perturbing the network's internal state. The first kind is
conspicuous. The second is the interesting one: the model keeps answering, and it keeps answering
with the same label, so anything watching only the output sees nothing wrong.

Most existing detection work watches exactly that output. This thesis asks whether the internal
states of several layers, taken together, separate clean execution from fault-injected execution —
and in particular whether they do so for the faults that never reach the output.

## 1.2 Hypotheses

The study is organised around four hypotheses, each with a rejection criterion fixed before any
result was produced (SPEC §1):

- **H1.** Multi-layer SVDD monitoring detects parameter faults better than output-only scores at low
  false-positive rates. *Rejected if* the difference in TPR at 1 % FPR has a paired-bootstrap 95 %
  confidence interval containing zero — evaluated **separately on Track S and Track H**, with Track S
  the discriminating test.
- **H2.** Fusing more than one layer beats the best single layer. *Rejected if* fused TPR@1%FPR is no
  better than the best per-layer detector within the confidence interval.
- **H3.** A small configuration — at most three taps, at most 32 features per tap — retains at least
  95 % of the full detection rate. *Rejected if* retention falls below 95 % on **both** primary models.
- **H4.** The layer where the alarm first fires correlates with the layer that was injected.
  *Rejected if* Spearman ρ is not significantly greater than zero.

H1 and H2 are the contract's core claims. H3 asks whether the method is deployable; H4 asks whether
it localises as well as detects.

## 1.3 Why the subtle faults are the point

The taxonomy this thesis uses (§4.3) labels every injected run by its *effect*, not by the fact that
an injection happened:

| Label | Definition | Track |
|---|---|---|
| `MASKED` | Top-1 unchanged, logit shift below ε | **S** |
| `DEGRADED` | Top-1 unchanged, confidence or margin shifted by at least ε | **S** |
| `SDC` | Top-1 changed, no numerical failure | **H** |
| `CRASH` | NaN/Inf in the output, or accuracy collapsed to chance | **H** |

Track S and Track H are reported separately throughout and are never merged into a single "fault"
class. The reason is methodological rather than presentational: output-only baselines are strong on
Track H almost by construction — those faults changed the output, which is what the baseline watches.
Merging the tracks lets Track H's easy wins carry the headline. The comparison that actually
discriminates H1 is the Track S one, where the output looks normal and only the internal state does
not.

A flagged Track S fault is a detection, not a false positive. A corrupted parameter really is
present; the monitor found it before it caused visible harm, which is precisely the operational value
claimed for internal-state monitoring.

## 1.4 Contributions

1. A multi-layer monitoring design: a fixed-width, permutation-invariant descriptor per observed
   layer, whose width is independent of that layer's channel count, and four detector variants over
   it (per-tap, early fusion, late fusion, and a deep extension).
2. An effect-based fault taxonomy that keeps subtle and harmful faults apart, and an evaluation
   protocol built around it.
3. A fault population generated over a stratified grid — bit strata, layer buckets, flip budgets,
   fault modes — in both FP32 and INT8, rather than a uniform sample over bit positions that would
   flatter the method.
4. Detection quality reported together with runtime and memory overhead, on edge-class hardware, as
   one result rather than two.
5. An adaptive attacker that knows the monitor and searches for flips that evade it, with the cost
   it pays reported in the same flip-budget units as the published attack literature.

## 1.5 Scope, stated up front

The hypotheses are claims about **convolutional image classifiers on 32×32 inputs**, at the parameter
scale evaluated here (≈0.27 M to ≈9.8 M parameters), under **persistent parameter faults**. Audio,
tabular and sequence models are not evaluated. Larger networks are not evaluated. Transient
activation faults are a secondary arm, reported apart from the main tables. Faults in the inference
engine's control flow, in the input data, and in the detector's own parameters are out of scope, as
are supply-chain compromise and training-time backdoors.

Chapter 7 states what follows from these bounds. They are limits of the study, not caveats to be
argued away.
