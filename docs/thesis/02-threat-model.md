# 2. Threat model

## 2.1 Goal and capability

The adversary wants **silent data corruption**: wrong predictions without an obvious failure. A crash
is a poor outcome for them, because a crash is noticed.

The capability assumed is a bounded number of bit flips in the memory holding the model's
parameters. This is the capability demonstrated by the Rowhammer and physical-injection literature,
not a hypothetical one. Flips are **persistent** in the primary setting — the corrupted weight stays
corrupted across subsequent inferences — and **transient** in a secondary setting, where a single
inference is affected. The persistent case is primary because it is what parameter-memory attacks
actually produce.

## 2.2 Deployment settings

Two settings are assumed, and they are the two the literature demonstrates against:

- **Embedded / edge inference.** An image classifier on a smart camera, drone or ADAS-class SoC.
  Physical fault injection is realistic here, and this setting is what makes the overhead budget
  binding: a monitor that only fits on a datacentre GPU does not defend it. Overhead is therefore
  measured on edge-class hardware by design, not by accident of what was available.
- **Shared-tenancy commodity hardware.** The DeepHammer setting: a co-located process flips DRAM bits
  in the victim's parameter pages. No physical access is required.

## 2.3 Attacker knowledge — three levels, all evaluated

| Level | Knows | Chooses flips by |
|---|---|---|
| **L0 — blind** | nothing | uniform random parameters, stratified random bit positions |
| **L1 — model-aware** | weights and gradients | progressive bit search (BFA): rank candidate bits by gradient, evaluate the shortlist, keep the best |
| **L2 — detector-aware** | additionally the monitored layers, the detector and its threshold | the same search, restricted to flips that keep every monitored score below its alarm threshold |

L0 establishes what faults look like when nobody is optimising. L1 is the attacker the published
flip-budget figures describe, which is what makes this study's budgets comparable to them. L2 is the
honest test of the monitor: an attacker who knows exactly what is watching and optimises against it.

L1 is validated against published results *before* it generates any fault population. An attacker
implementation that collapses a network in far fewer or far more flips than the literature reports is
a bug, and discovering that after the detection sweep would mean re-running the sweep.

## 2.4 Defender capability

The defender can read activations at chosen layers at inference time, has a clean calibration set,
and has a fixed compute budget for monitoring. The defender does **not** have labels at inference
time, does not know which parameter was hit, and cannot re-verify the whole parameter set on every
inference — that last point is what distinguishes this problem from a checksum.

A parameter checksum is included in the evaluation as **context, not as a competitor**. It has
perfect recall on persistent parameter faults and a fixed cost, and it is blind to transient
activation faults. Stating that plainly is more useful than pretending the comparison is close:
internal-state monitoring is for the cases a checksum does not cover, and for deployments where
re-hashing every inference is too expensive.

## 2.5 Out of scope

Faults in the inference engine's control flow; faults in the input data; faults in the detector's own
parameters; software supply-chain compromise; training-time backdoors. These are named here so the
contribution boundary is explicit rather than left to the reader to infer.
