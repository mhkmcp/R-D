---
name: report-results
description: Turn fidnn experiment outputs (artifacts/results/*.parquet) into thesis-ready tables, figures and prose that follow SPEC §9 reporting rules — Track S and Track H separate, mandatory per-stratum/bucket/budget/mode/precision/attacker breakdowns, seeds and 95% CIs, paired-bootstrap comparisons, overhead alongside detection, and hypothesis verdicts against the §1 rejection criteria. Use when summarising results, writing a results section, making a table, or answering "did H1/H2/H3/H4 hold?".
argument-hint: "[hypothesis, table, or results path]"
---

# Report results

Load `.claude/references/evaluation-protocol.md` and the hypotheses section of
`.claude/references/constraints.md`.

## Where the numbers come from

Only from `artifacts/results/*.parquet` (+ sidecars) produced by `src/` code. Analysis code goes in
`src/fidnn/eval/`; notebooks only plot. Never type a number into a table by hand, never compute a
headline number in a notebook. Check each sidecar: **`device` must be `cpu`** for any reported number
(MPS = exploratory). Refuse to report M1 or M-4 outputs as thesis results.

## Checklist for every table or claim

- [ ] Track S and Track H in separate columns/tables; never a merged "fault" rate.
- [ ] `CRASH` broken out inside H; `MASKED` vs `DEGRADED` inside S.
- [ ] TPR@1%FPR and TPR@0.1%FPR; measured FPR on `clean_test` beside nominal α.
- [ ] Precision, recall, F1, AUROC **and AUPR** available (all eight contract metrics appear somewhere).
- [ ] Breakdowns exist: category, bit stratum, layer bucket, budget, fault mode (`bf_bn` ≠ `bf_w`),
      precision, attacker level. An aggregate-only table is not acceptable.
- [ ] Seed count n ≥ 5 and mean ± 95 % CI on every number (C4).
- [ ] Overhead in the same table or an explicitly paired one (C5): absolute ms first, % second, baseline
      model named, batch = 1 primary. INT8 separate from FP32 with the reason inline.
- [ ] Model, tap set (default/extended), detector variant, K, per-inference vs windowed (n, m) stated.
- [ ] `fault_gen` reported as its own row, never merged into the headline.

## Comparisons and verdicts

- Variant vs variant: paired bootstrap, 2000 resamples, CI of the **difference**. CI includes 0 →
  "no detected difference". Never "outperforms" on overlapping CIs.
- Hypotheses: apply the §1 rejection criterion exactly. H1 gets **two** verdicts (Track S, Track H);
  lead with Track S. Design targets (≥ 5-pt H1 gain; < 10 % latency for H3 on M2) are reported but
  never turn a verdict into "rejected" — use "confirmed, with a deployment caveat".
- H4: ρ and propagation depth **separately for M2 and M3**; the difference is the finding.
- §9.6(8): per corruption type and per family, with the fault-slope vs severity-slope comparison. If it
  looks like a novelty detector, write that.

## Wording

- Scope in the sentence itself: "for convolutional image classifiers on 32×32 inputs (CIFAR-10)".
- The M2-INT8 L1 literature table is "a sanity check on the attack implementation", not a detector
  claim.
- Block-D downgrade is *indirect* evidence about convolutional dependence, never a substitute for a
  second modality.
- Mahalanobis appears only as the stated C1 exclusion with the KDE/LOF/IF substitutes.

Output tables as Markdown (or LaTeX if asked) with a caption listing n seeds, CI method, device and
split.
