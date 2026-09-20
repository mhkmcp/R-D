"""SPEC §9.4 overhead measurement and §10 adaptive attacker."""

import numpy as np
import pandas as pd
import pytest
import torch

from fidnn.attack import adaptive, bfa
from fidnn.bench.overhead import measure, pareto, throughput
from fidnn.detect.features import CleanFeatures, add_block_e
from fidnn.detect.variants import D2
from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import build
from fidnn.taps.features import FEATURE_NAMES
from fidnn.taps.registry import taps

TIMING = {"warmup": 1, "runs": 3}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build("m1").eval()


def test_overhead_rows_carry_absolute_ms_and_a_stated_baseline(model):
    df = measure(model, "m1", "default", "cpu", "fp32", (1,), (1, 3), TIMING, log=lambda _: None)
    assert {"model_only", "capture", "features"} <= set(df.stage)
    base = df[df.stage == "model_only"].median_ms.iloc[0]
    assert (df.baseline_ms == base).all()
    assert (df[df.stage == "model_only"].added_ms == 0).all()
    feats = df[df.stage == "features"]
    assert (feats.added_ms > 0).all(), "feature extraction is not free"
    np.testing.assert_allclose(feats.added_pct, 100 * feats.added_ms / base, rtol=1e-6)


def test_overhead_grows_with_the_number_of_taps(model):
    df = measure(model, "m1", "default", "cpu", "fp32", (1,), (1, 6), TIMING, log=lambda _: None)
    feats = df[df.stage == "features"].sort_values("k")
    assert feats.added_ms.iloc[-1] > feats.added_ms.iloc[0]


def test_memory_is_recorded_for_every_measured_row(model):
    df = measure(model, "m1", "default", "cpu", "fp32", (1,), (1,), TIMING, log=lambda _: None)
    measured = df[df.stage != "scoring"]
    assert measured.tracemalloc_peak_kb.notna().all()


def test_scoring_cost_is_measured_separately_when_a_detector_is_given(model):
    calls = []
    df = measure(model, "m1", "default", "cpu", "fp32", (1,), (2,), TIMING,
                 detector=lambda n: calls.append(n), log=lambda _: None)
    assert "scoring" in set(df.stage) and calls, "scoring must be timed apart from hook cost"


def test_throughput_is_samples_per_second(model):
    df = measure(model, "m1", "default", "cpu", "fp32", (32,), (1,), TIMING, log=lambda _: None)
    t = throughput(df)
    np.testing.assert_allclose(t.samples_per_s, t.batch / (t.median_ms / 1e3), rtol=1e-9)


def test_pareto_front_drops_dominated_points():
    latency = pd.DataFrame({"k": [1, 2, 3, 4], "added_ms": [1.0, 2.0, 3.0, 4.0]})
    tpr = pd.DataFrame({"k": [1, 2, 3, 4], "tpr": [0.50, 0.45, 0.80, 0.79]})
    front = pareto(latency, tpr)
    assert front.k.tolist() == [1, 3]      # k=2 is slower and worse; k=4 is slower and worse


@pytest.fixture(scope="module")
def int8_pair():
    torch.manual_seed(0)
    fp32 = build("m2").eval()
    return fp32, quantize_ptq(fp32, [torch.randn(16, 3, 32, 32)])


def _monitor(model, tau):
    torch.manual_seed(1)
    tap_list = taps("m2", "default")
    rng = np.random.default_rng(0)
    values = np.abs(rng.normal(1.0, 0.25, (200, len(tap_list), len(FEATURE_NAMES))))
    names = FEATURE_NAMES + ["e_l2_ratio", "e_entropy_ratio", "e_sat_diff"]
    clean = CleanFeatures(add_block_e(values, FEATURE_NAMES),
                          [t.tap_id for t in tap_list], names)
    d2 = D2(seed=0).fit(clean, clean)
    return adaptive.Monitor(model, tap_list, d2, tau=tau)


def test_unmonitored_arm_matches_the_l1_search(int8_pair):
    fp32, int8 = int8_pair
    torch.manual_seed(2)
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
    df = adaptive.attack(int8, fp32, x, y, budget=2, monitor=None, candidates_per_layer=2,
                         log=lambda _: None)
    assert len(df) == 2 and (df.loss_after > df.loss_before).all()
    assert set(df.attacker) == {"L1_unmonitored"}
    assert (df.rejected_candidates == 0).all()


def test_a_strict_monitor_blocks_every_candidate(int8_pair):
    fp32, int8 = int8_pair
    torch.manual_seed(3)
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    blocking = _monitor(int8, tau=-1e9)        # nothing can stay under this threshold
    df = adaptive.attack(int8, fp32, x, y, budget=2, monitor=blocking, candidates_per_layer=1,
                         log=lambda _: None)
    assert df.empty, "no flip should be accepted when the monitor rejects everything"


def test_a_permissive_monitor_lets_the_attack_through(int8_pair):
    fp32, int8 = int8_pair
    torch.manual_seed(4)
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    permissive = _monitor(int8, tau=1e9)
    df = adaptive.attack(int8, fp32, x, y, budget=1, monitor=permissive, candidates_per_layer=1,
                         log=lambda _: None)
    assert len(df) == 1 and df.attacker.iloc[0] == "L2"


def test_compare_reports_the_cost_of_the_monitor():
    without = pd.DataFrame({"step": [1, 2, 3], "accuracy": [50.0, 30.0, 9.0]})
    with_mon = pd.DataFrame({"step": [1, 2, 3, 4, 5], "accuracy": [60.0, 55.0, 40.0, 20.0, 10.0],
                             "rejected_candidates": [4, 3, 2, 1, 0]})
    c = adaptive.compare(with_mon, without, target=11.0)
    assert c["flips_unmonitored"] == 3 and c["flips_monitored"] == 5
    assert c["cost_multiplier"] == pytest.approx(5 / 3)
    assert not c["monitor_blocked_attack"] and c["rejected_candidates"] == 10


def test_compare_marks_an_attack_the_monitor_defeated():
    without = pd.DataFrame({"step": [1, 2], "accuracy": [40.0, 8.0]})
    with_mon = pd.DataFrame({"step": [1], "accuracy": [80.0], "rejected_candidates": [9]})
    c = adaptive.compare(with_mon, without, target=11.0)
    assert c["monitor_blocked_attack"] and c["flips_monitored"] is None


def test_bfa_and_adaptive_share_the_flip_record_shape(int8_pair):
    fp32, int8 = int8_pair
    torch.manual_seed(5)
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    l1 = bfa.attack(int8, fp32, x, y, budget=1, candidates_per_layer=1, log=lambda _: None)
    l2 = adaptive.attack(int8, fp32, x, y, budget=1, monitor=None, candidates_per_layer=1,
                         log=lambda _: None)
    assert {"step", "layer", "flat_index", "bit", "loss_before", "loss_after"} <= set(l1.columns)
    assert {"step", "layer", "flat_index", "bit", "loss_before", "loss_after"} <= set(l2.columns)
