---
name: milestone-gate
description: Evaluate whether a fidnn milestone (M-0 throughput, M-1 accuracy, M-1b BFA validation, M-2 injection and Track S share, M-3 features, M-4 sanity, M-5 … M-9) has met its SPEC §12 exit criterion, using the actual artifacts and test results, and say what blocks the next milestone. Use when the user asks "can we start the sweep", "is M-x done", "what's next", or before launching any long-running experiment.
argument-hint: <M-0 | M-1 | M-1b | M-2 | … | next>
allowed-tools: Read Grep Glob Bash(uv run pytest *) Bash(uv run python -m fidnn *) Bash(git status *) Bash(git log *) Bash(shasum -a 256 *)
---

# Milestone gate

Load `.claude/references/milestones.md`. Target: `$ARGUMENTS` (`next` = the earliest milestone whose
criterion is not yet evidenced).

## Procedure

1. Quote the exit criterion for the milestone from the reference.
2. Collect **evidence from files, not from memory or chat history**: `artifacts/**` outputs and their
   sidecars, `artifacts/models/manifest.json`, test results from `uv run pytest -q`, and reports under
   `docs/` (e.g. `docs/M0_throughput.md`). Record the path for every piece of evidence.
3. Mark each sub-criterion ✅ met (with evidence path) · ❌ not met (with the measured value) ·
   ⚠️ no evidence found.
4. Check upstream gates are met too: nothing past M-0 without throughput numbers; no fault population
   before M-1b; no H1 work before M-2's Track S ≥ ~15 %.
5. Check sidecars: reported-grade outputs must have `device: cpu`, a git commit, config hash and seed.

## Milestone-specific checks

- **M-0:** every (model × device × precision × hooks × tap set) cell measured; project the §4.2 grid in
  hours from the measured throughput; if it does not fit, propose cuts **in the SPEC cut order**
  (`.claude/references/constraints.md`) — never bit strata, never M2 INT8.
- **M-1:** accuracies within 1 pt of 91.25 % (M2) and the pinned VGG source (M3); disjointness and
  label-agreement tests exist and pass; VGG reference actually pinned in the manifest.
- **M-1b:** a figure comparing the BFA curve to published numbers exists; discrepancies explained.
- **M-2:** taxonomy distribution for M2 and M3; Track S share numerically ≥ ~15 % (else: ResNet-56
  escalation per §4.2).
- **M-4:** clearly labelled as producing no thesis number.

## Output

Verdict line: `M-x: PASSED` / `M-x: BLOCKED — <reason>` / `M-x: NOT STARTED`, then the criterion
table, then the single next action. Do not launch long sweeps from this skill; recommend the command.
