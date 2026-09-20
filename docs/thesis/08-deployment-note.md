# 8. Deployment note

*SPEC §14(5). Written for an engineer deciding whether to put this monitor in a product.*

## 8.1 What to monitor

Start with the default tap set: the stem, one tap per stage or block, the pooled representation and
the logits — six or seven observation points depending on the architecture. They are cheap to place:
each is an identity module in the model graph, so adding them changes no arithmetic and survives
quantised conversion under the same name.

The minimal configuration that H3 tests — at most three taps — comes out of the K sweep (chapter
5.12). Until those numbers exist, the honest advice is: monitor the default set, measure on your own
hardware, and cut using the measured Pareto front rather than intuition about which layers "should"
matter. ⏳

## 8.2 What it costs

Two components, and they behave differently:

- **Hook cost** — attaching to a tap and keeping its output. Close to free.
- **Feature cost** — computing the 29-wide descriptor per tap. This dominates.

From the throughput calibration on an Apple M5 at batch 1, ResNet-20 with six taps:

| Arm | Bare forward | Monitored (6 taps) | Added |
|---|---|---|---|
| CPU FP32 | 1.34 ms | 2.24 ms | +0.89 ms (+67 %) |
| CPU INT8 | 1.14 ms | 2.24 ms | +1.10 ms (+96 %) |

The quantised forward pass is faster while the descriptor arithmetic stays in floating point, so the
*relative* cost is higher exactly where the deployment case is strongest. Two levers exist: fewer taps
(the K sweep), and a cheaper Block A — the quantiles are the expensive part, and histogram-based
approximations are the obvious substitution. Neither is applied in the measured numbers above, which
are the unoptimised descriptor.

Detector size and memory sit beside these numbers in chapter 5.11. ⏳

## 8.3 What it does not cover

Stated plainly, because a monitor whose limits are unclear gets trusted for things it cannot do:

- **Transient input faults.** Corruption of the input rather than the parameters is out of scope.
- **Control-flow faults.** Faults in the inference engine itself, rather than in the model's
  parameters, are not modelled.
- **Faults in the detector.** The monitor's own parameters are assumed intact. A threat model where
  the attacker can corrupt the monitor needs a different design.
- **Detector-aware attackers beyond the measured budget.** Chapter 6 reports what an adaptive attacker
  pays; an attacker willing to spend more than that is not covered by any claim here.
- **Models above the scale evaluated.** Roughly 0.27 M to 9.8 M parameters on 32×32 inputs.
- **Non-convolutional and non-vision architectures.** Transformers, recurrent models, tabular and
  audio networks are **not evaluated at all**. The descriptor's structural block assumes channels and
  spatial positions; what it would mean elsewhere is untested.

## 8.4 Where a checksum is the better answer

If the faults you care about are persistent parameter corruption, and you can afford to re-hash the
parameters at the cadence that matters, **a checksum is simpler, exact, and has perfect recall**. It is
the right tool for that case and this method does not beat it there.

Internal-state monitoring earns its place when re-hashing every inference is too expensive, when the
faults include transient effects a periodic checksum can miss between hashes, or when you want the
alarm to carry information about *where* the deviation appeared — which a checksum, by construction,
does not provide.
