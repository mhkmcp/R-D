# M-8 — Adaptive (L2) attacker

**Milestone:** M-8 (SPEC §12, §10) · **Model:** m2 int8 ·
**Monitor:** D2 at α=0.01, τ=-0.0789 · **Commit:** `a9142c22f4594e6cbbf6cfda866a2c5568445cb2-dirty`

The L2 attacker knows the taps and the detector, and only keeps flips that leave every monitored
score under the threshold. The unmonitored arm is the same search with the filter removed, so the
two columns differ in exactly one thing.

## Flip budget with and without the monitor

| target_accuracy | flips_unmonitored | flips_monitored | cost_multiplier | monitor_blocked_attack | rejected_candidates | trial |
|---|---|---|---|---|---|---|
| 11.00 | 6 | None | None | True | 1082 | 0 |
| 11.00 | 8 | None | None | True | 1293 | 1 |
| 11.00 | 13 | None | None | True | 0 | 2 |
| 11.00 | 7 | None | None | True | 829 | 3 |
| 11.00 | 5 | None | None | True | 1715 | 4 |

The monitor blocked the attack outright in 5 of 5 trials.

**Honest framing (§10).** A higher required budget is a real gain even when the monitor is
ultimately evadable; a blocked trial is not proof of security. Budgets are in the same units as the
M-1b table, so they compare with the published BFA figures.
