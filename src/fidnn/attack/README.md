# fidnn.attack

**Status: empty.** L1 is needed for M-1b, L2 for M-8. SPEC §2, §6.4, §10.

## To build

| Attacker | Contract |
|---|---|
| L0 | Uniform random parameters/bits. This is the planner in `fidnn.inject` and needs no code here |
| L1 | Faithful BFA (Rakin et al. 2019) progressive bit search on INT8: rank bits by gradient, flip the best, repeat. Uses `fidnn.inject` primitives for every flip |
| L2 | Detector-aware greedy search: candidates ranked by loss gain, filtered to keep every monitored SVDD score under its threshold. Reports success rate and flip budget with vs without the monitor |

## Invariants

- **M-1b gate:** L1's flip-budget-vs-accuracy curve on INT8 ResNet-20 must match published BFA
  figures **before** L1 generates any fault population. Save the comparison figure.
- Every flip goes through `fidnn.inject`, so restore and checksum guarantees still apply, and every
  attack emits the same replay records.
- Run L2 on M2 INT8 first.
