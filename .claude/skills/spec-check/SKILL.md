---
name: spec-check
description: Check a planned change, design choice, or experiment against docs/SPEC.md before implementing it — which SPEC clause it discharges, whether it is a contract commitment or a C0 addition, and whether it touches hard constraints C1–C5. Use before starting any new module, experiment, config, or scope change in this fidnn thesis project, or when the user asks "does the spec allow…", "is this in scope", or "which section covers…".
argument-hint: <change or question>
allowed-tools: Read Grep Glob
---

# Spec check

Map `$ARGUMENTS` onto the specification before any code is written.

## Steps

1. Read `.claude/references/constraints.md`, then the one reference that covers the area
   (index in `.claude/references/README.md`). Open `docs/SPEC.md` at the cited section only when the
   digest is not specific enough — SPEC is ~1,100 lines; grep for the `§` heading rather than reading
   it whole.
2. Answer, in this order:
   - **Clause.** Which SPEC section(s) require or permit this. If none does, say so plainly: it is an
     unrequested addition.
   - **Commitment or addition (C0).** Commitments trace to the contract; additions are cuttable. Place
     additions in the cut order.
   - **Constraint exposure.** For each of C1–C5, "not touched" or the specific risk:
     - C1 — does any step compute a covariance, whiten, or use a banned estimator?
     - C2 — can any fault record reach fitting, ν/γ selection, CDF mapping or thresholds?
     - C3 — is the threshold fixed a priori on `clean_cal`?
     - C4 — will outputs carry seeds and CIs?
     - C5 — is overhead reported alongside detection?
   - **Milestone.** Which gate (M-0 … M-9) it belongs to, and whether an earlier gate must pass first
     (M-0 before any sweep; M-1b before generating faults; M-2 Track S ≥ ~15 % before H1 work).
   - **Required tests** from SPEC §11.4 that the change must add or keep green.
3. If the request conflicts with SPEC, say which clause and stop. Do not work around it. SPEC and
   `docs/Dataset.md` edits need the user's approval (`ask` permission); the contract PDF is never edited.

## Output

A short verdict line (`in scope — §x.y`, `addition — cut tier n`, or `conflicts with §x.y`), then the
bullets above. No implementation.
