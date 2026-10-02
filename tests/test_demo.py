"""Demo session: exact restore, live results equal a full forward pass, export is deterministic."""

import copy
import json

import numpy as np
import pytest
import torch

from fidnn.demo import export
from fidnn.demo.session import BATCH, DemoSession
from fidnn.detect.features import BLOCK_E, CleanFeatures, add_block_e
from fidnn.detect.variants import D1, D2, D3
from fidnn.inject.bitflip import state_checksum
from fidnn.inject.engine import injected_flips
from fidnn.inject.plan import STRATA
from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import build
from fidnn.taps.features import FEATURE_NAMES
from fidnn.taps.registry import taps


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    torch.manual_seed(0)
    x = torch.randn(96, 3, 32, 32)
    clean = quantize_ptq(build("m2").eval(), [x[:32]])
    with torch.no_grad():
        y = clean(x).argmax(1).numpy()
    tap_list = taps("m2", "default")
    images = np.random.default_rng(0).integers(0, 256, (96, 32, 32, 3), dtype=np.uint8)
    s = DemoSession("m2", clean, copy.deepcopy(clean), x, y, {}, {}, tap_list,
                    {t.tap_id: 1.0 for t in tap_list}, tmp_path_factory.mktemp("cache"),
                    images=images)
    feats = CleanFeatures(add_block_e(np.asarray(s.runner.cache.features), FEATURE_NAMES),
                          s.tap_ids, FEATURE_NAMES + BLOCK_E)
    fit = CleanFeatures(feats.values[:64], feats.taps, feats.names)
    cal = CleanFeatures(feats.values[64:], feats.taps, feats.names)
    d1 = D1(alpha=0.05).fit(fit, cal)
    d2 = D2(alpha=0.05).fit(fit, cal)
    s.detectors = {"D1": d1, "D2": d2, "D3_max": D3(d1, "max").fit(cal)}
    s.taus = {0.05: float(np.quantile(d2.scores(cal), 0.95))}
    return s


def test_no_flips_is_masked_and_clean(session):
    session.reset()
    ev = session.evaluate(session.batch_for(3))
    assert len(ev.idx) == BATCH and ev.idx[0] == 3
    assert (ev.labels == "MASKED").all()
    assert np.array_equal(ev.clean_logits, ev.fault_logits)
    assert ev.checksum == session.clean_checksum


def test_live_result_equals_full_forward(session):
    session.reset()
    session.add_flip("fc", 7, 7)
    session.add_flips("late", "msb", 4, np.random.default_rng(1))
    idx = session.batch_for(0)
    ev = session.evaluate(idx)
    with torch.no_grad(), injected_flips(session.fault, session.flips, "bf_w"):
        full = session.fault(session.x[idx]).float().numpy()
    assert np.array_equal(full, ev.fault_logits)
    assert state_checksum(session.fault) == session.clean_checksum
    assert ev.d1.shape == (BATCH, len(session.tap_ids)) and ev.d2.shape == (BATCH,)


def test_add_flips_respects_bucket_and_stratum(session):
    session.reset()
    added = session.add_flips("early", "sign", 16, np.random.default_rng(0))
    assert len(added) == 16
    assert {f["bucket"] for f in added} == {"early"}
    assert {f["bit"] for f in added} == set(STRATA[8]["sign"])
    assert session.add_flip(added[0]["layer"], added[0]["flat_index"], added[0]["bit"]) == []
    session.reset()
    assert session.flips == []


def test_describe_flip_sign_bit_changes_sign_of_raw_value(session):
    d = session.describe_flip({"layer": "fc", "storage": "qweight", "tensor": "weight",
                               "flat_index": 0, "bit": 7})
    assert d["new_raw"] == d["old_raw"] ^ -128


def test_summary_keeps_tracks_apart(session):
    session.reset()
    session.add_flips("late", "sign", 16, np.random.default_rng(2))
    table = session.summary(session.evaluate(session.batch_for(0)), 0.05)
    assert list(table.group) == ["all probes", "track S", "track H"]
    assert table.n.iloc[1] + table.n.iloc[2] == BATCH
    session.reset()


def test_export_is_deterministic(session, tmp_path):
    a = export.export(session, tmp_path / "a.json", seed=0)
    b = export.export(session, tmp_path / "b.json", seed=0)
    for p in (a, b):
        p.pop("provenance")
    assert a == b
    assert len(a["images"]) == 24 and a["images"][0]["src"].startswith("data:image/png;base64,")
    n_strata = len(session.strata())
    assert len(a["scenarios"]) == 3 * n_strata * len(export.BUDGETS) * export.DRAWS
    assert json.loads((tmp_path / "a.json").read_text())["model"] == "m2"
    assert session.flips == []


def test_uploaded_image_is_exact(session):
    session.mean, session.std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    img = np.random.default_rng(3).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    i = session.add_image(img)
    assert i == len(session.y) - 1 and i >= session.n_pool
    with torch.no_grad():
        clean = session.clean(session.x[i:i + 1]).float().numpy()
    assert np.array_equal(clean, session.runner.cache.logits[i:i + 1])
    assert session.y[i] == clean.argmax()
    session.reset()
    session.add_flips("early", "msb", 4, np.random.default_rng(2))
    idx = session.batch_for(i)
    assert idx[0] == i and (idx[1:] < session.n_pool).all()
    ev = session.evaluate(idx)
    with torch.no_grad(), injected_flips(session.fault, session.flips, "bf_w"):
        full = session.fault(session.x[idx]).float().numpy()
    assert np.array_equal(full, ev.fault_logits)
    assert ev.checksum == session.clean_checksum
    session.reset()
