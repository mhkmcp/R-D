# 6. The adaptive attacker

> **Status: skeleton.** Tables bind to `artifacts/attacks/l2_*.parquet` and `docs/M8_adaptive.md`,
> produced by `uv run fidnn attack l2 --model m2`. ⏳ not yet run.

## 6.1 Why this chapter exists

A detector evaluated only against attackers who ignore it answers an easier question than the one a
deployer asks. The L2 attacker knows the monitored layers, the detector and its threshold, and
searches for flips that damage the model while keeping every monitored score below the alarm. It is
run on the quantised primary model first, because that is where the flip budgets are comparable to
published attacks.

## 6.2 What the attacker does

The search is the same progressive bit search as L1, with one change: a candidate flip is only kept if
the monitored scores stay under the threshold. The unmonitored arm is the identical search with that
filter removed, so the two columns of the comparison differ in exactly one thing. The attacker sees
only its own sample batch, never the labels of the evaluation set it is scored on.

## 6.3 Results to fill ⏳

| Table | Content |
|---|---|
| 6.1 | Flips required to reach the target accuracy, with and without the monitor, per trial |
| 6.2 | Attack success rate within the flip budget, with and without |
| 6.3 | Candidates rejected by the monitor per accepted flip — how much the constraint narrows the search |
| 6.4 | Where the constrained attack flips bits, against where the unconstrained one does |

## 6.4 How this result will be written

Three framings are ruled out in advance, because each would overstate what the experiment shows.

**"The monitor stops the attack."** It does not, unless every trial fails, and even then the claim
would be about this attacker and this budget. What the experiment measures is a *cost*: the multiple
by which the required flip budget grows when the attacker must stay quiet.

**"Evadable, therefore useless."** An attacker forced from a handful of flips to many times that
number is materially constrained, because flip budget is the scarce resource in every physical and
Rowhammer-style attack this threat model describes. A higher budget is a real gain even against an
attacker who eventually succeeds.

**"The numbers are comparable to the published attack table."** They are comparable in *units* —
flips to chance accuracy on the same architecture and bit width — but the published figures are for
an unmonitored model. The unmonitored arm here is the right comparison for the monitor's effect, and
the published figures are the right comparison for whether the attack implementation is faithful.
Those are two different questions and the chapter keeps them apart.

## 6.5 The bound this chapter cannot escape

The attacker knows the threshold exactly, which is the strong form of the assumption and is
deliberate. It is also one search strategy among many: a different objective, a different candidate
ranking, or an attacker willing to accept partial damage might pay less. The reported cost is
therefore an upper bound on this attacker's efficiency and a lower bound on nothing. Chapter 7.5
states this alongside the other attack-side threats.
