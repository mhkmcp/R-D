"""M-0 budget ledger: cut order and reinvestment order follow SPEC §3.3, §4.2, §13."""

import pandas as pd
import pytest

from fidnn.bench.budget import CUTS, REINVESTMENTS, Measurements, Plan, ledger, plan_budget, total
from fidnn.taps.registry import taps


def _measurements(fwd_ms: float = 20.0) -> Measurements:
    inf = []
    for model in ("m1", "m2", "m3"):
        for device, precision in (("cpu", "fp32"), ("cpu", "int8"), ("mps", "fp32")):
            for batch in (1, 32):
                inf.append({"model": model, "device": device, "precision": precision, "batch": batch,
                                "hooks": "off", "tap_set": "-", "median_ms": fwd_ms / 2, "status": "ok"})
                for tap_set in ("default", "extended"):
                    k = len(taps(model, tap_set))
                    inf.append({"model": model, "device": device, "precision": precision,
                                    "batch": batch, "hooks": "features", "tap_set": tap_set,
                                    "median_ms": fwd_ms + k, "status": "ok"})
    grad = [{"model": m, "device": d, "kind": k, "median_ms": 300.0}
            for m in ("m1", "m2", "m3") for d in ("cpu", "mps") for k in ("fwd_bwd", "train_step")]
    inj = [{"model": m, "precision": p, "kind": k, "layer": f"l{i}", "median_ms": 0.5}
           for m in ("m1", "m2", "m3") for p in ("fp32", "int8")
           for k in ("flip_restore", "checksum") for i in range(10)]
    svdd = [{"method": "ocsvm_exact", "data": "heavy", "n": n, "d": d, "nu": 0.01, "fit_s": n / 1e3,
                 "score_per_s": 1e5} for n in (2000, 30000) for d in (29, 174, 377)]
    return Measurements(*map(pd.DataFrame, (inf, grad, inj, svdd)))


M = _measurements()


def test_baseline_has_every_spec_arm_and_line():
    lg = ledger(Plan(), M)
    arms = set(zip(lg[lg.line == "Fault sweep (§4.2)"].model,
                   lg[lg.line == "Fault sweep (§4.2)"].precision))
    assert arms == {("m2", "fp32"), ("m2", "int8"), ("m3", "fp32"), ("m3", "int8")}
    assert "L2 adaptive attacks" not in set(lg.line)


def test_generous_budget_funds_all_reinvestments_in_order():
    d = plan_budget(M, budget_hours=1e9)
    assert d.mode == "reinvest"
    assert d.applied == [variants[0][0] for variants in REINVESTMENTS]
    assert d.plan == Plan(seeds=10, extended_taps=True, l2="all", int8_parity=True)


def test_reinvestment_never_skips_a_fundable_higher_priority():
    base = total(Plan(), M)
    d = plan_budget(M, budget_hours=base * 1.05)
    order = [name for variants in REINVESTMENTS for name, _ in variants]
    idx = [order.index(a) for a in d.applied]
    assert idx == sorted(idx)
    for name, cost in d.declined:
        assert cost > d.budget_hours, name


@pytest.mark.parametrize("budget_fraction", [0.9, 0.6, 0.3, 0.01])
def test_cuts_follow_spec_order_and_never_touch_m2_int8(budget_fraction):
    d = plan_budget(M, budget_hours=total(Plan(), M) * budget_fraction)
    assert d.mode in {"cut", "over"}
    assert d.applied == [name for name, _ in CUTS][: len(d.applied)]
    assert ("m2", "int8") in d.plan.arms()
    assert d.plan.extended_taps is False and d.plan.l2 == "none" and d.plan.seeds == 5


def test_halving_reps_halves_the_fault_sweep():
    full = ledger(Plan(), M)
    half = ledger(Plan(reps=50), M)
    sweep = "Fault sweep (§4.2)"
    assert half[half.line == sweep].hours.sum() == pytest.approx(
        full[full.line == sweep].hours.sum() / 2)
