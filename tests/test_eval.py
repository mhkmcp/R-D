"""SPEC §7–§9: fault splits, leakage guard, metrics per track, bootstrap statistics."""

import numpy as np
import pandas as pd
import pytest

from fidnn.eval import localisation, metrics, splits, stats
from fidnn.eval.leakage import LeakageError, assert_clean, check
from fidnn.eval.report import write

STRATA = ["sign", "exp_msb", "mant_low"]
BUCKETS = ["early", "middle", "late"]


def make_outcomes(n_injections=90, probes=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_injections):
        stratum = STRATA[i % len(STRATA)]
        bucket = BUCKETS[i % len(BUCKETS)]
        budget = [1, 4, 16][i % 3]
        label = rng.choice(["MASKED", "DEGRADED", "SDC", "CRASH"], probes)
        rows.append(pd.DataFrame({
            "injection_id": i, "stratum": stratum, "bucket": bucket, "budget": budget,
            "mode": "bf_w", "attacker": "L0", "probe_index": np.arange(probes),
            "label": label, "track": np.where(np.isin(label, ["MASKED", "DEGRADED"]), "S", "H"),
        }))
    return pd.concat(rows, ignore_index=True)


def test_split_is_over_instances_not_probes():
    out = make_outcomes()
    out["fault_split"] = splits.assign(out, seed=0)
    per_injection = out.groupby("injection_id").fault_split.nunique()
    assert (per_injection == 1).all(), "an injection must land wholly in one split"
    assert set(out.fault_split) == {"fault_dev", "fault_test"}


def test_gen_configs_never_enter_dev():
    out = make_outcomes()
    out["fault_split"] = splits.assign(out, seed=0)
    dev = out[out.fault_split == "fault_dev"]
    assert not splits.is_gen_config(dev.stratum, dev.budget).any()
    gen = out[splits.gen_mask(out)]
    assert (gen.fault_split == "fault_test").all()      # fault_gen ⊂ fault_test


def test_dev_share_is_about_thirty_percent_of_eligible_instances():
    out = make_outcomes(n_injections=300)
    out["fault_split"] = splits.assign(out, seed=0)
    eligible = out[~splits.gen_mask(out)].injection_id.nunique()
    dev = out[out.fault_split == "fault_dev"].injection_id.nunique()
    assert abs(dev / eligible - splits.DEV_SHARE) < 0.02


def test_split_is_seeded():
    out = make_outcomes()
    a, b = splits.assign(out, seed=0), splits.assign(out, seed=0)
    assert a.equals(b) and not a.equals(splits.assign(out, seed=1))


def _cells(out):
    """The planned grid, as the injection records would report it."""
    return {tuple(r) for r in out[["bucket", "stratum", "budget"]].drop_duplicates().to_numpy()}


def test_leakage_guard_passes_a_clean_build():
    out = make_outcomes()
    out["fault_split"] = splits.assign(out, seed=0)
    assert check(out, {}, _cells(out), {"kaggle_vs_canonical_test_collisions": 0}) == []


def test_leakage_guard_catches_a_missing_grid_cell():
    """The cell must be compared against the plan; comparing the data with itself finds nothing."""
    out = make_outcomes()
    out["fault_split"] = splits.assign(out, seed=0)
    planned = _cells(out)
    trimmed = out[out.stratum != "mant_low"]
    problems = check(trimmed, {}, planned, None)
    assert any("missing" in p and "grid cells" in p for p in problems)
    assert check(trimmed, {}, _cells(trimmed), None) == []   # self-derived axes cannot catch it


def test_leakage_guard_catches_gen_configs_in_dev_and_pixel_collisions():
    out = make_outcomes()
    out["fault_split"] = "fault_test"
    out.loc[out.stratum == "sign", "fault_split"] = "fault_dev"   # a gen config in dev
    problems = check(out, {}, _cells(out), {"kaggle_vs_canonical_test_collisions": 3})
    assert any("fault_gen" in p for p in problems)
    assert any("pixel-hash collisions" in p for p in problems)
    with pytest.raises(LeakageError):
        assert_clean(out, {}, _cells(out), {"kaggle_vs_canonical_test_collisions": 3})


def test_leakage_guard_catches_an_instance_in_both_splits():
    out = make_outcomes()
    out["fault_split"] = splits.assign(out, seed=0)
    first = out.injection_id.iloc[0]
    out.loc[out.injection_id == first, "fault_split"] = ["fault_dev", "fault_test"] * 2
    assert any("both fault_dev and fault_test" in p for p in check(out, {}, _cells(out), None))


def test_metrics_separate_the_tracks():
    rng = np.random.default_rng(0)
    scored = make_outcomes(seed=1)
    scored["score"] = np.where(scored.track == "H", rng.normal(5, 1, len(scored)),
                               rng.normal(2, 1, len(scored)))
    clean = rng.normal(0, 1, 500)
    taus = {0.01: float(np.quantile(clean, 0.99))}
    table = metrics.by_track(scored, clean, taus)
    assert set(table.track) == {"S", "H"}
    h = table[table.track == "H"].tpr.iloc[0]
    s = table[table.track == "S"].tpr.iloc[0]
    assert h > s, "Track H should be easier than Track S on this fixture"
    assert table.auroc.between(0, 1).all() and table.aupr.between(0, 1).all()


def test_breakdowns_cover_every_mandatory_axis():
    rng = np.random.default_rng(0)
    scored = make_outcomes(seed=2).assign(precision="int8")
    scored["score"] = rng.normal(3, 1, len(scored))
    clean = rng.normal(0, 1, 300)
    table = metrics.breakdowns(scored, clean, {0.01: 2.0})
    for axis in ("label", "stratum", "bucket", "budget", "mode", "precision", "attacker"):
        assert axis in set(table.axis), axis
    assert set(table[table.axis == "label"].value) <= set(metrics.CATEGORIES)


def test_confusion_and_ranking_are_consistent():
    fault = np.array([3.0, 4.0, 5.0])
    clean = np.array([0.0, 1.0, 2.0])
    c = metrics.confusion(fault, clean, tau=2.5)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (3, 0, 0, 3)
    assert c["precision"] == 1.0 and c["recall"] == 1.0 and c["f1"] == 1.0
    assert metrics.ranking(fault, clean)["auroc"] == 1.0


def test_mean_ci_carries_n_and_dispersion():
    r = stats.mean_ci([0.5, 0.55, 0.6, 0.52, 0.58])
    assert r["n"] == 5 and r["ci_low"] < r["mean"] < r["ci_high"]


def test_paired_difference_reports_no_detected_difference_when_ci_spans_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 200)
    same = a + rng.normal(0, 0.1, 200)
    better = a + 0.3
    assert stats.paired_difference(a, same)["verdict"] == "no detected difference"
    win = stats.paired_difference(better, a)
    assert win["verdict"] == "difference" and win["ci_low"] > 0


def test_paired_difference_requires_matched_sets():
    with pytest.raises(ValueError, match="same evaluation set"):
        stats.paired_difference([1, 2, 3], [1, 2])


def test_seed_count_gate():
    assert not stats.seeds_ok([0, 1, 2])
    assert stats.seeds_ok(range(5))


def test_localisation_finds_a_planted_correlation():
    injected = np.repeat([0, 1, 2], 40)
    first_alarm = injected * 2 + np.random.default_rng(0).integers(0, 2, 120)
    r = localisation.spearman(injected, first_alarm)
    assert r["rho"] > 0.8 and r["greater_than_zero"]
    depth = localisation.propagation_depth(injected, first_alarm)
    assert depth["n"] == 120 and depth["mean_depth"] > 0


def test_localisation_reports_no_correlation_honestly():
    rng = np.random.default_rng(1)
    r = localisation.spearman(rng.integers(0, 3, 200), rng.integers(0, 6, 200))
    assert not r["greater_than_zero"]


def test_alarm_buckets_fold_taps_into_thirds():
    taps = [f"t{i}" for i in range(6)]
    assert localisation.tap_buckets(taps) == ["early", "early", "middle", "middle", "late", "late"]
    got = localisation.alarm_buckets(np.array([0, 5, -1]), taps)
    assert got.tolist() == ["early", "late", "none"]


def test_report_flags_too_few_seeds(tmp_path):
    rng = np.random.default_rng(0)
    head = pd.DataFrame({"model": "m2", "precision": "int8", "tap_set": "default",
                         "detector": "D2", "track": ["S", "H"], "alpha": [0.01, 0.01],
                         "tpr": [0.6, 0.9], "measured_fpr": 0.01, "auroc": 0.9, "aupr": 0.8,
                         "n_fault": 100, "seed": 0})
    head.to_parquet(tmp_path / "m2_headline.parquet", index=False)
    breaks = head.assign(axis="stratum", value="sign")
    breaks.to_parquet(tmp_path / "m2_breakdowns.parquet", index=False)
    (tmp_path / "m2_eval.json").write_text('{"seed": 0}')
    out = tmp_path / "M5_results.md"
    write(tmp_path, out)
    text = out.read_text()
    assert "Not yet reportable" in text and "≥ 5 seeds" in text.replace(" ", " ")
    assert "Track S" in text and "Track H" in text
    assert rng is not None
