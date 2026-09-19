---
name: c1-audit
description: Audit the fidnn repository for violations of SPEC constraint C1 (Mahalanobis distance excluded from the entire project) — banned estimators, explicit inverse-covariance distances, ZCA/covariance whitening, whitened PCA — across src, tests, configs, notebooks and thesis text. Use before a results build, before a commit touching detectors/features/data preprocessing, or when the user asks whether something is "C1-safe".
argument-hint: "[path — defaults to the whole repo]"
allowed-tools: Read Grep Glob Bash(uv run pytest *)
---

# C1 audit

Rules: `.claude/references/constraints.md` § "C1 — banned constructs". Scope: `$ARGUMENTS`, or the
repo root if empty. Skip `.venv/`, `artifacts/`, `.git/`.

## 1. Name-level scan

Grep case-insensitively for:

```
mahalanobis|EllipticEnvelope|MinCovDet|EmpiricalCovariance|LedoitWolf|ShrunkCovariance|OAS\b|GraphicalLasso|zca
```

Allowed hits: `docs/*.md`, `.claude/**` (these ban the constructs by name), and `tests/*c1*` (the
guard test). Any other hit is a violation.

## 2. Structure-level scan — the ones a name grep misses

Grep for these and **read each hit in context**:

| Pattern | Violation when |
|---|---|
| `np.cov`, `torch.cov`, `.cov(`, `covariance` | The covariance feeds a distance, a whitening transform, or an inverse |
| `linalg.inv`, `linalg.pinv`, `solve`, `cholesky`, `solve_triangular` | Applied to a feature covariance — that is `Σ⁻¹` |
| `whiten` (e.g. `PCA(whiten=True)`) | Output then goes into a Euclidean distance/kernel — Mahalanobis in the retained subspace |
| `eigh`/`svd` on a covariance followed by scaling by `1/sqrt(eigvals)` | Hand-rolled whitening |
| GCN / "global contrast normalization" in data code | Usually paired with ZCA in old CIFAR recipes |

Permitted and not violations: per-feature median/IQR scaling, per-channel mean/std input
normalisation, unwhitened PCA reconstruction error, KDE with an isotropic/diagonal bandwidth, LOF,
Isolation Forest, RBF SVDD.

## 3. Thesis-text check

In any `.md`/`.tex` outside `docs/SPEC.md`/`docs/Dataset.md`: Mahalanobis may be *named as excluded*
but never presented as a method or result. Verify the §14(4) methods paragraph (exclusion + KDE/LOF/IF
substitutes) exists once thesis text exists.

## 4. Guard test

If `tests/` contains a C1 guard test, run `uv run pytest -q tests -k c1`. If none exists, report that
SPEC §11.4 requires one.

## Report

`C1: clean` or `C1: N violations`, then one line per finding: `path:line — construct — why it is (or
is equivalent to) Mahalanobis — permitted substitute`. List allowed hits separately and briefly.
Do not edit files unless the user asks for the fixes.
