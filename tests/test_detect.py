"""SPEC §5.3, §6, §8, §11.4: normaliser, SVDD variants, calibration, and the C2/C3 guards."""

import numpy as np
import pandas as pd
import pytest

from fidnn.detect.baselines import FeatureSpaceBaselines, output_only_scores
from fidnn.detect.calibrate import (
    fpr_upper_bound,
    measured_fpr,
    thresholds,
    windowed_alarm,
)
from fidnn.detect.features import BLOCK_E, WIDTH, CleanFeatures, FaultFeatures, add_block_e
from fidnn.detect.normalise import RobustNormaliser
from fidnn.detect.svdd import SV_BAND, gamma_median, select
from fidnn.detect.variants import D1, D2, D3
from fidnn.taps.features import FEATURE_NAMES

TAPS = ["stem", "stage1", "logits"]
NAMES = FEATURE_NAMES + BLOCK_E


def make(kind, n, shift=0.0, seed=0, spread=0.25):
    """A tight clean population: a heavy-tailed one puts the clean 99th percentile above the
    score RBF kernels saturate at, which is a property of the fixture, not of the detector."""
    rng = np.random.default_rng(seed)
    v = np.abs(rng.normal(1.0 + shift, spread, (n, len(TAPS), len(FEATURE_NAMES))))
    return kind(add_block_e(v, FEATURE_NAMES), list(TAPS), list(NAMES))


@pytest.fixture(scope="module")
def data():
    return {"fit": make(CleanFeatures, 300, seed=0), "cal": make(CleanFeatures, 200, seed=1),
            "test": make(CleanFeatures, 200, seed=2), "fault": make(FaultFeatures, 150, 4.0, 3)}


def test_block_e_adds_three_cross_tap_features():
    f = make(CleanFeatures, 5)
    assert f.values.shape == (5, 3, WIDTH) and WIDTH == 29
    assert f.names[-3:] == BLOCK_E
    first_tap = f.values[:, 0, :]
    assert np.allclose(first_tap[:, -3], 1.0)   # neutral ratio for the first tap
    assert np.allclose(first_tap[:, -1], 0.0)


def test_normaliser_uses_clean_fit_only_and_drops_flat_features(data):
    n = RobustNormaliser().fit(data["fit"])
    z = n.transform(data["fit"])
    assert np.abs(np.median(z[:, :, 0])) < 0.1     # centred on the clean median
    with pytest.raises(TypeError):
        RobustNormaliser().fit(data["fault"])

    flat = make(CleanFeatures, 50)
    flat.values[:, :, 3] = 7.0                     # no clean spread: must be dropped
    n2 = RobustNormaliser().fit(flat)
    assert not n2.keep[:, 3].any()
    assert (n2.transform(flat)[:, :, 3] == 0).all()
    assert any(f == flat.names[3] for _, f in n2.dropped(flat))


def test_normaliser_stats_do_not_change_when_scoring_faults(data):
    n = RobustNormaliser().fit(data["fit"])
    before = (n.median.copy(), n.iqr.copy(), n.keep.copy())
    n.transform(data["fault"])
    assert all(np.array_equal(a, b) for a, b in zip(before, (n.median, n.iqr, n.keep)))


@pytest.mark.parametrize("detector", ["D1", "D2"])
def test_fitting_on_fault_features_is_a_type_error(data, detector):
    cls = {"D1": D1, "D2": D2}[detector]
    with pytest.raises(TypeError, match="C2"):
        cls().fit(data["fault"], data["cal"])
    with pytest.raises(TypeError, match="C2"):
        cls().fit(data["fit"], data["fault"])


def test_d3_cdf_fit_rejects_fault_features(data):
    d1 = D1().fit(data["fit"], data["cal"])
    with pytest.raises(TypeError, match="C2"):
        D3(d1).fit(data["fault"])


def test_baselines_reject_fault_features(data):
    with pytest.raises(TypeError, match="C2"):
        FeatureSpaceBaselines().fit(data["fault"])


def test_selection_sees_only_clean_data_and_holds_the_sv_band(data):
    n = RobustNormaliser().fit(data["fit"])
    zf = n.transform(data["fit"]).reshape(len(data["fit"]), -1)
    zc = n.transform(data["cal"]).reshape(len(data["cal"]), -1)
    sel = select(zf, zc, alpha=0.01, seed=0)
    assert 0 < sel.nu <= 0.1 and sel.clean_cal_fpr >= 0
    assert SV_BAND[0] * sel.nu <= sel.sv_fraction <= SV_BAND[1] * sel.nu
    assert len(sel.trials) == 5 * 7    # 5 ν × (scale + 6 γ multiples)


def test_gamma_median_is_seeded_and_positive(data):
    x = data["fit"].fused()
    assert gamma_median(x, seed=0) == gamma_median(x, seed=0) > 0


def test_d2_separates_a_shifted_population(data):
    d2 = D2(seed=0).fit(data["fit"], data["cal"])
    tau = thresholds(d2.scores(data["cal"]))[0.01]
    fpr = (d2.scores(data["test"]) > tau).mean()
    tpr = (d2.scores(data["fault"]) > tau).mean()
    assert fpr < 0.1 and tpr > 0.5


def test_d1_gives_one_score_per_tap_and_d3_maps_them_to_cdf_values(data):
    d1 = D1(seed=0).fit(data["fit"], data["cal"])
    per_tap = d1.scores(data["test"])
    assert per_tap.shape == (len(data["test"]), len(TAPS))
    d3 = D3(d1, "max").fit(data["cal"])
    mapped = d3.map_scores(per_tap)
    assert mapped.min() >= 0 and mapped.max() <= 1
    assert np.all(d3.scores(data["test"]) == mapped.max(axis=1))


def test_d3_combiners_and_first_alarm_tap(data):
    d1 = D1(seed=0).fit(data["fit"], data["cal"])
    scores = {c: D3(d1, c).fit(data["cal"]).scores(data["fault"]) for c in
              ("max", "mean", "fpr_weighted")}
    assert all(s.shape == (len(data["fault"]),) for s in scores.values())
    assert (scores["max"] >= scores["mean"]).all()
    first = D3(d1, "max").fit(data["cal"]).first_alarm_tap(data["fault"])
    assert set(np.unique(first)) <= set(range(-1, len(TAPS)))
    with pytest.raises(ValueError, match="combiner"):
        D3(d1, "nonsense").fit(data["cal"]).scores(data["test"])


def test_thresholds_are_clean_cal_quantiles_fixed_a_priori():
    scores = np.arange(1000, dtype=float)
    t = thresholds(scores)
    assert t[0.05] < t[0.01] < t[0.001]
    assert np.isclose(t[0.01], np.quantile(scores, 0.99))


def test_measured_fpr_reports_a_binomial_upper_bound():
    scores = np.zeros(2000)
    m = measured_fpr(scores, tau=1.0)
    assert m["alarms"] == 0 and m["fpr"] == 0
    assert 0 < m["fpr_upper95"] < 0.01     # 0 of 2000 still is not "0 %"
    assert fpr_upper_bound(0, 2000) < fpr_upper_bound(5, 2000)


def test_windowed_alarm_is_m_of_n():
    alarms = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    assert windowed_alarm(alarms, 1, 1).tolist() == [bool(a) for a in alarms]
    got = windowed_alarm(alarms, 4, 2)   # windows ending at each index, inclusive
    assert got.tolist() == [False, False, False, True, False, True, True, False]


def test_output_only_scores_are_higher_for_less_confident_outputs():
    confident = np.array([[10.0, 0.0, 0.0]])
    unsure = np.array([[0.1, 0.0, 0.0]])
    c, u = output_only_scores(confident), output_only_scores(unsure)
    for k in ("msp", "entropy", "margin"):
        assert u[k][0] > c[k][0], k


def test_feature_space_baselines_score_every_sample(data):
    b = FeatureSpaceBaselines(seed=0).fit(data["fit"])
    scores = b.scores(data["fault"])
    assert set(scores) == {"isolation_forest", "lof", "kde", "pca_recon"}
    assert all(len(v) == len(data["fault"]) and np.isfinite(v).all() for v in scores.values())
    clean = b.scores(data["test"])
    assert scores["pca_recon"].mean() > clean["pca_recon"].mean()


def test_detectors_are_deterministic_for_a_seed(data):
    a = D2(seed=0).fit(data["fit"], data["cal"]).scores(data["test"])
    b = D2(seed=0).fit(data["fit"], data["cal"]).scores(data["test"])
    np.testing.assert_allclose(a, b)


def test_from_frame_requires_every_registry_tap():
    from fidnn.detect.features import from_frame
    df = pd.DataFrame({"index": [0, 1], "tap_id": ["stem", "stem"],
                       **{n: [0.5, 0.6] for n in FEATURE_NAMES}})
    with pytest.raises(ValueError, match="missing taps"):
        from_frame(df, TAPS, CleanFeatures)
