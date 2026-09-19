"""M-0 budget ledger: SPEC workload × measured unit costs → wall-clock hours (SPEC §4.2, §3.3, §12).

A `Plan` is the set of knobs the SPEC allows to move. Everything else in the grid — bit strata,
fault modes, flip budgets, both tracks, M2's INT8 arm — is not a field of `Plan` at all, so no
cut can reach it. Workload volumes the SPEC does not pin are named in `Assumptions` and printed
in the report, so each line of the ledger can be challenged and recomputed.

Reporting-device policy (§11.3): every fault/clean/CIFAR-10-C extraction, attack and overhead
line is costed on **CPU**. Model training is costed on the device named in `Assumptions`.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from fidnn.taps.registry import taps

PRIMARY = ("m2", "m3")
FEATURES_PER_TAP = 29  # §5.2 width including Block E


@dataclass(frozen=True)
class Assumptions:
    # §4.2 grid
    buckets: int = 3
    strata: int = 5            # never cut; not a Plan field
    flip_budgets: int = 3      # N ∈ {1, 4, 16}
    full_grid_modes: int = 2   # bf_w, bf_b
    reduced_modes: int = 5     # sa0, sa1, rnd_val, bf_act, bf_bn
    reduced_cells: int = 4     # 1 bucket × 2 strata × 2 budgets
    probes: int = 32
    checksum_every: int = 100
    # data volumes (§3.2, §9.6(8))
    clean_images: int = 60_000       # 50k Kaggle train (fit/cal/test) + 10k canonical probe pool
    cifar10c_images: int = 750_000   # 15 types × 5 severities × 10k
    # M-1 training — not pinned by SPEC; He et al. schedule ≈ 164 epochs, VGG commonly 200
    train_images: int = 50_000
    train_batch: int = 128
    epochs: dict = field(default_factory=lambda: {"m1": 30, "m2": 200, "m3": 200})
    train_device: str = "mps"
    # L1 BFA (§2, §6.4): per (model, precision, seed)
    l1_attacks: int = 100
    l1_flips: int = 16
    # L2 (§10): greedy search filtered by monitor score — assumed 3× an L1 run
    l2_factor: float = 3.0
    # detectors (§6.3, §9.6)
    hp_grid: int = 35                # 5 ν × 7 γ
    ablation_fits: int = 110         # K sweep ≈ 91 greedy fits + blocks 6 + position 4 + kernel 3 + budget 4
    svdd_n: int = 30_000             # clean_fit size
    # M-6 overhead study: 3 detector variants × K ∈ 1..K_ext × batch {1,32} × 3 device/precision arms
    overhead_runs: int = 1000
    overhead_variants: int = 3
    # M1 smoke test: 1 seed, FP32, repetitions reduced
    m1_reps: int = 10
    # orchestration, data loading, parquet I/O, reruns — applied to every line
    slack: float = 1.5


@dataclass(frozen=True)
class Plan:
    seeds: int = 5
    reps: int = 100
    extended_taps: bool = False
    l2: str = "none"           # "none" | "m2_int8" (§10: run it there first) | "all"
    int8_parity: bool = False
    m3_int8: bool = True

    def arms(self) -> list[tuple[str, str]]:
        out = [("m2", "fp32"), ("m2", "int8"), ("m3", "fp32")]  # M2 INT8 is unconditional
        if self.m3_int8:
            out.append(("m3", "int8"))
        return out


REINVESTMENTS = [  # §3.3 priority order; each entry lists its variants, fullest first
    [("More seeds (5 → 10)", lambda p: replace(p, seeds=10))],
    [("Extended tap sets (13 on M2, 10 on M3)", lambda p: replace(p, extended_taps=True))],
    [("L2 adaptive attacker on every arm", lambda p: replace(p, l2="all")),
     ("L2 adaptive attacker on M2 INT8 only (§10: run there first)",
      lambda p: replace(p, l2="m2_int8"))],
    [("Full INT8 parity", lambda p: replace(p, int8_parity=True))],
]
CUTS = [  # §4.2 / §13 cut order, applied only after all reinvestments are already absent
    ("Repetitions 100 → 50", lambda p: replace(p, reps=50)),
    ("Drop M3's INT8 arm", lambda p: replace(p, m3_int8=False)),
]


class Measurements:
    """Unit costs (seconds) looked up from the M-0 parquet tables."""

    def __init__(self, inference: pd.DataFrame, grad: pd.DataFrame, injection: pd.DataFrame,
                 svdd: pd.DataFrame):
        self.inf = inference[inference.status == "ok"]
        self.grad = grad
        self.inj = injection
        self.svdd = svdd

    def forward(self, model: str, precision: str, hooks: str, tap_set: str, batch: int = 32,
                device: str = "cpu") -> float:
        q = self.inf[(self.inf.model == model) & (self.inf.precision == precision)
                     & (self.inf.device == device) & (self.inf.batch == batch)
                     & (self.inf.hooks == hooks) & (self.inf.tap_set == tap_set)]
        if q.empty:
            raise KeyError((model, precision, hooks, tap_set, batch, device))
        return float(q.median_ms.iloc[0]) / 1e3

    def step(self, model: str, device: str, kind: str) -> float:
        q = self.grad[(self.grad.model == model) & (self.grad.device == device)
                      & (self.grad.kind == kind)]
        return float(q.median_ms.iloc[0]) / 1e3

    def injection_fixed(self, model: str, precision: str, checksum_every: int) -> float:
        q = self.inj[(self.inj.model == model) & (self.inj.precision == precision)]
        flip = q[q.kind == "flip_restore"].median_ms.mean()
        checksum = q[q.kind == "checksum"].median_ms.mean()
        return (flip + checksum / checksum_every) / 1e3

    def n_quant_layers(self, model: str) -> int:
        q = self.inj[(self.inj.model == model) & (self.inj.precision == "int8")
                     & (self.inj.kind == "flip_restore")]
        return q.layer.nunique()

    def svdd_fit(self, d: int, n: int) -> float:
        """Mean exact-OCSVM fit time over the measured ν span, pessimistic data shape,
        interpolated in d and extrapolated as n² beyond the largest measured n."""
        s = self.svdd[(self.svdd.method == "ocsvm_exact") & (self.svdd.data == "heavy")]
        measured = sorted(s.n.unique())
        n_ref = next((x for x in measured if x >= n), measured[-1])  # conservative: round n up
        at = s[s.n == n_ref].groupby("d").fit_s.mean()
        fit = float(np.interp(d, at.index.values, at.values))
        if d > at.index.max():
            fit *= d / at.index.max()
        return fit * max(1.0, n / n_ref) ** 2

    def svdd_score_rate(self, d: int) -> float:
        s = self.svdd[(self.svdd.method == "ocsvm_exact")]
        at = s[s.n == s.n.max()].groupby("d").score_per_s.min()
        return float(np.interp(d, at.index.values, at.values))


def _injections(a: Assumptions, reps: int) -> int:
    full = a.full_grid_modes * a.buckets * a.strata * a.flip_budgets * reps
    reduced = a.reduced_modes * a.reduced_cells * reps
    return full + reduced


def ledger(plan: Plan, m: Measurements, a: Assumptions = Assumptions()) -> pd.DataFrame:
    """One row per (workload, model, precision) with hours, before and after slack."""
    rows = []
    s = plan.seeds
    tap_set = "extended" if plan.extended_taps else "default"
    n_inj = _injections(a, plan.reps)

    def add(line: str, model: str, precision: str, seconds: float, detail: str) -> None:
        rows.append({"line": line, "model": model, "precision": precision,
                     "hours": seconds / 3600, "detail": detail})

    for model, precision in plan.arms():
        fwd = m.forward(model, precision, "features", tap_set)
        fixed = m.injection_fixed(model, precision, a.checksum_every)
        n_taps = len(taps(model, tap_set))
        add("Fault sweep (§4.2)", model, precision, s * n_inj * (fwd + fixed),
            f"{s} seeds × {n_inj:,} injections × ({fwd * 1e3:.1f} ms fwd+features "
            f"+ {fixed * 1e3:.2f} ms flip/restore/checksum)")
        add("Clean extraction", model, precision, s * a.clean_images / a.probes * fwd,
            f"{s} × {a.clean_images:,} images")

        # L1 BFA: per flip, one fwd+bwd on a batch of 128 (fake-quant for INT8, costed as FP32)
        # plus one candidate forward per injectable layer, then a probe batch.
        per_flip = (m.step(model, "cpu", "fwd_bwd")
                    + m.n_quant_layers(model) * m.forward(model, precision, "off", "-") * 4 + fwd)
        l1 = s * a.l1_attacks * a.l1_flips * per_flip
        add("L1 BFA attacks", model, precision, l1,
            f"{s} × {a.l1_attacks} attacks × {a.l1_flips} flips × {per_flip:.2f} s")
        if plan.l2 == "all" or (plan.l2 == "m2_int8" and (model, precision) == ("m2", "int8")):
            add("L2 adaptive attacks", model, precision, a.l2_factor * l1,
                f"{a.l2_factor}× L1")

        d2 = n_taps * FEATURES_PER_TAP
        fits = a.hp_grid * (n_taps * m.svdd_fit(FEATURES_PER_TAP, a.svdd_n)
                            + m.svdd_fit(d2, a.svdd_n))
        records = n_inj * a.probes + a.clean_images
        with_ablations = precision == "fp32" or plan.int8_parity
        if with_ablations:
            records += a.cifar10c_images
        scoring = records * (n_taps / m.svdd_score_rate(FEATURES_PER_TAP) + 1 / m.svdd_score_rate(d2))
        add("Detector fit + scoring", model, precision, s * (fits + scoring),
            f"{a.hp_grid}-point grid × ({n_taps} D1 + 1 D2 @ d={d2}), n={a.svdd_n:,}")

        if with_ablations:
            add("CIFAR-10-C confounder (§9.6(8))", model, precision,
                s * a.cifar10c_images / a.probes * fwd, f"{s} × {a.cifar10c_images:,} images")
            abl = s * a.ablation_fits * m.svdd_fit(d2, a.svdd_n)
            detail = f"{a.ablation_fits} D2-scale fits per seed"
            if not plan.extended_taps:  # K sweep (§9.6(1)) needs extended features: one seed
                ext = m.forward(model, precision, "features", "extended")
                abl += (n_inj + a.clean_images / a.probes) * ext
                detail += "; + 1-seed extended-tap extraction for the K sweep"
            add("Ablations (§9.6)", model, precision, abl, detail)

    for model in PRIMARY:
        steps = a.epochs[model] * a.train_images / a.train_batch
        add("M-1 training", model, "fp32", s * steps * m.step(model, a.train_device, "train_step"),
            f"{s} seeds × {a.epochs[model]} epochs on {a.train_device}")

    # M1 smoke test: one seed, FP32, few repetitions
    add("M1 smoke test", "m1", "fp32",
        a.epochs["m1"] * a.train_images / a.train_batch * m.step("m1", a.train_device, "train_step")
        + (_injections(a, a.m1_reps) + a.clean_images / a.probes)
        * m.forward("m1", "fp32", "features", "default"),
        f"1 seed, {a.m1_reps} reps")

    # M-6 overhead study (§9.4): upper bound — every K costed at the full extended set
    for model in PRIMARY:
        k_max = len(taps(model, "extended"))
        per_config = sum(m.forward(model, prec, "features", "extended", batch=b, device=dev)
                         for b in (1, 32) for dev, prec in (("cpu", "fp32"), ("cpu", "int8"),
                                                             ("mps", "fp32")))
        add("M-6 overhead study", model, "all",
            k_max * a.overhead_variants * a.overhead_runs * per_config * 1.1,
            f"K=1..{k_max} × {a.overhead_variants} variants × {a.overhead_runs} runs")

    df = pd.DataFrame(rows)
    df["hours_with_slack"] = df.hours * a.slack
    return df


def total(plan: Plan, m: Measurements, a: Assumptions = Assumptions()) -> float:
    return float(ledger(plan, m, a).hours_with_slack.sum())


@dataclass
class BudgetDecision:
    budget_hours: float
    plan: Plan
    total_hours: float
    baseline_hours: float
    fits: bool
    applied: list[str]      # reinvestments taken (or cuts applied)
    declined: list[tuple[str, float]]  # options that did not fit: (name, plan total if taken)
    mode: str               # "reinvest" | "cut" | "over"


def plan_budget(m: Measurements, budget_hours: float,
                a: Assumptions = Assumptions()) -> BudgetDecision:
    """Reinvest in §3.3 priority order, or cut in §4.2/§13 order.

    Reinvestments are taken in priority order: an item is only ever funded after every
    higher-priority item has been funded or found unaffordable. An unaffordable item is
    skipped rather than ending the search — skipping it takes nothing from a higher priority.
    Within an item, the fullest affordable variant is taken.
    """
    base = Plan()
    base_h = total(base, m, a)
    if base_h <= budget_hours:
        plan, applied, declined = base, [], []
        for variants in REINVESTMENTS:
            for name, apply in variants:
                cand = apply(plan)
                cost = total(cand, m, a)
                if cost <= budget_hours:
                    plan = cand
                    applied.append(name)
                    break
                declined.append((name, cost))
        decision = BudgetDecision(budget_hours, plan, total(plan, m, a), base_h, True,
                                  applied, declined, "reinvest")
    else:
        plan, applied = base, []
        for name, apply in CUTS:
            plan = apply(plan)
            applied.append(name)
            if total(plan, m, a) <= budget_hours:
                break
        t = total(plan, m, a)
        decision = BudgetDecision(budget_hours, plan, t, base_h, t <= budget_hours, applied, [],
                                  "cut" if t <= budget_hours else "over")
    _check_invariants(decision.plan, a)
    return decision


def _check_invariants(plan: Plan, a: Assumptions) -> None:
    assert a.strata == 5, "bit strata are never cut (§4.2)"
    assert ("m2", "int8") in plan.arms(), "M2's INT8 arm is never cut (§0)"
    assert a.full_grid_modes == 2, "bf_w and bf_b both run the full grid (§4.1)"
