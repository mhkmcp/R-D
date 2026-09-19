# References

Condensed, section-cited digests of `docs/SPEC.md` (v2.0) and `docs/Dataset.md` (v2.0) for skills
and agents to load instead of the full documents.

**Precedence:** `docs/new_thesis_contract.pdf` > `docs/SPEC.md` > `docs/Dataset.md` > these files.
These digests never introduce a rule. If a digest and SPEC disagree, SPEC wins and the digest is
stale — fix the digest, never the other way round. Cite SPEC sections (`SPEC §x.y`), not these files,
in code comments, commit messages and thesis text.

| File | Covers | SPEC sections |
|---|---|---|
| [constraints.md](constraints.md) | Hard constraints C0–C5, the C1 banned/permitted list, additions and the cut order | §0, §13 |
| [fault-injection.md](fault-injection.md) | Fault modes, the §4.2 grid, bit strata, outcome taxonomy, injection implementation rules | §2, §4 |
| [features-and-detectors.md](features-and-detectors.md) | Tap sets, the per-tap descriptor, normalisation, SVDD, D1–D4, baselines | §5, §6 |
| [evaluation-protocol.md](evaluation-protocol.md) | Splits, leakage guard, calibration, metrics, breakdowns, statistics, overhead, ablations | §7, §8, §9, §10 |
| [data.md](data.md) | CIFAR-10 two-source assembly, splits, preprocessing, integrity checks, CIFAR-10-C | §3.2, Dataset.md |
| [milestones.md](milestones.md) | M-0 … M-9 exit criteria, required tests, reproducibility sidecar | §11, §12 |

## Known inconsistencies in the source documents

Recorded so nobody "fixes" code to match one side silently. Resolve in `docs/SPEC.md` (an `ask`
permission), then update the digest.

1. **Dense-tap width.** SPEC §5.2 says "29 per conv tap, 22 per dense tap (9+4+5+4)". 9+4+5+8+3 = 29
   includes Block E; 9+4+5+4 = 22 does not. `src/fidnn/taps/features.py` emits 26 (A–D) for both
   variants, zero-padding D-dense's 4 spatial features, with Block E's 3 added at fusion → 29 for both.
   The §9.6(3) "22-feature downgrade" is therefore 22 *informative* features inside a 26/29 width.
2. SPEC §3.1 repeats the "two fair objections" list verbatim (once as "Two fair objections", once as
   "Objections to the single-dataset design" with a third added).
3. SPEC §13 cites "§3.1a objection 3" (no §3.1a exists — it is §3.1) and "`Dataset.md` §2.4" (the
   near-duplicate note is Dataset.md §5.2).
