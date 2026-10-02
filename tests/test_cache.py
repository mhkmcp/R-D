"""Persisted INT8 model and suffix execution reproduce the full forward pass bit for bit."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from fidnn.cache import store
from fidnn.cache.graph import Resume, cut_points, stateful_modules, traced
from fidnn.cache.suffix import SuffixRunner
from fidnn.inject.bitflip import state_checksum
from fidnn.inject.engine import injected_flips
from fidnn.inject.plan import plan
from fidnn.inject.sweep import run as sweep_run
from fidnn.inject.targets import targets
from fidnn.models import checkpoint
from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import build
from fidnn.taps.hooks import TapMonitor
from fidnn.taps.registry import taps

N_PROBES = 96


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(1)
    return torch.randn(N_PROBES, 3, 32, 32)


def _int8(model_id: str, calib: torch.Tensor):
    torch.manual_seed(0)
    return quantize_ptq(build(model_id).eval(), [calib[:32]])


def _setup(model_id: str, x: torch.Tensor, tap_set: str = "extended"):
    clean = _int8(model_id, x)
    fault = copy.deepcopy(clean)
    tap_list = taps(model_id, tap_set)
    sat = {t.tap_id: 1.0 for t in tap_list}
    starts = Resume(clean).starts(stateful_modules(clean))
    cache = store.build(clean, x, starts, tap_list, sat, batch=40)
    return clean, fault, tap_list, sat, cache, SuffixRunner(fault, cache, tap_list, sat)


def _injections(model, model_id, mode, reps=2):
    p = plan(targets(model, model_id), mode, seed=0, reps=reps)
    return [g.to_dict("records") for _, g in p.groupby("injection_id")]


# --- persisted INT8 model -------------------------------------------------------------------

@pytest.fixture
def models_dir(tmp_path):
    torch.manual_seed(0)
    torch.save(build("m1").state_dict(), checkpoint.checkpoint_path("m1", 0, tmp_path))
    return tmp_path


def test_persisted_int8_matches_fresh_ptq(models_dir, x):
    calib = [x[:32], x[32:64]]
    record = checkpoint.save_int8("m1", 0, models_dir, calib)
    fresh = checkpoint.quantize_ptq(checkpoint._fp32("m1", 0, models_dir), calib)
    loaded = checkpoint.load("m1", 0, "int8", models_dir, calib)
    assert state_checksum(loaded) == state_checksum(fresh) == record["checksum"]
    with torch.no_grad():
        assert torch.equal(loaded(x), fresh(x))
    assert loaded is not checkpoint.load("m1", 0, "int8", models_dir)


def test_persisted_int8_refuses_other_calibration(models_dir, x):
    checkpoint.save_int8("m1", 0, models_dir, [x[:32]])
    with pytest.raises(checkpoint.StaleInt8Error):
        checkpoint.load("m1", 0, "int8", models_dir, [x[32:64]])


def test_persisted_int8_refuses_tampered_sidecar(models_dir, x):
    checkpoint.save_int8("m1", 0, models_dir, [x[:32]])
    side = checkpoint.int8_path("m1", 0, models_dir).with_suffix(".json")
    side.write_text(json.dumps(json.loads(side.read_text()) | {"checksum": "0" * 32}))
    with pytest.raises(checkpoint.StaleInt8Error):
        checkpoint.load("m1", 0, "int8", models_dir)


# --- cut points -----------------------------------------------------------------------------

def test_resnet_cuts_are_block_boundaries(x):
    cuts = cut_points(_int8("m2", x))
    assert {"tap_stem", "tap_pooled"} <= set(cuts)
    assert all(f"layer{s}_{b}_tap_out" in cuts for s in (1, 2, 3) for b in range(3))
    assert not [c for c in cuts if "conv" in c and c != "conv1"]  # nothing inside a block


def test_vgg_cuts_include_every_conv_tap():
    cuts = cut_points(traced(build("m3").eval()))
    assert all(f"block{b}_tap{k}" in cuts for b, k in [(1, 1), (3, 2), (5, 2)])


def test_resume_starts_before_first_faulted_layer(x):
    r = Resume(_int8("m2", x))
    assert r.start(["layer2.1.conv2"]) == "layer2_0_tap_out"
    assert r.start(["layer3.2.conv1", "layer2.1.conv2"]) == "layer2_0_tap_out"
    assert r.start(["fc"]) == "tap_pooled"
    assert r.start([]) is None


# --- suffix execution -----------------------------------------------------------------------

@pytest.mark.parametrize("model_id", ["m1", "m2", "m3"])
@pytest.mark.parametrize("mode", ["bf_w", "bf_b"])
def test_suffix_equals_full_forward_bitwise(x, model_id, mode):
    _, fault, tap_list, sat, _, runner = _setup(model_id, x)
    clean_sum = state_checksum(fault)
    rng = np.random.default_rng(0)
    with TapMonitor(fault, tap_list, mode="features", sat_thresholds=sat) as mon, torch.no_grad():
        for flips in _injections(fault, model_id, mode):
            idx = rng.choice(N_PROBES, 32, replace=False)
            with injected_flips(fault, flips, mode) as applied:
                full = fault(x[idx]).float().numpy()
                full_taps = dict(mon.outputs)
                logits, feats = runner.run({f["layer"] for f in applied}, idx)
            assert np.array_equal(full, logits)
            assert list(feats) == list(full_taps)
            for t, full_feat in full_taps.items():
                assert torch.equal(full_feat, feats[t]), t
    assert state_checksum(fault) == clean_sum
    assert runner.stats["early_exit"] > 0


def test_no_early_exit_between_two_faulted_layers(x):
    """A clean block output after the first fault says nothing about a fault further down."""
    _, fault, tap_list, sat, _, runner = _setup("m2", x)
    flips = [{"layer": "layer1.0.conv1", "tensor": "weight", "storage": "qweight",
              "flat_index": 0, "bit": 0},
             {"layer": "layer3.2.conv2", "tensor": "weight", "storage": "qweight",
              "flat_index": 5, "bit": 7}]
    idx = np.arange(32)
    with TapMonitor(fault, tap_list, mode="features", sat_thresholds=sat) as mon, \
            torch.no_grad(), injected_flips(fault, flips, "bf_w"):
        full = fault(x[idx]).float().numpy()
        logits, feats = runner.run(["layer1.0.conv1", "layer3.2.conv2"], idx)
        assert np.array_equal(full, logits)
        assert all(torch.equal(mon.outputs[t], feats[t]) for t in feats)


def test_nothing_applied_returns_the_clean_cache(x):
    _, _, _, _, cache, runner = _setup("m1", x, "default")
    idx = np.arange(8)
    logits, feats = runner.run([], idx)
    assert np.array_equal(logits, cache.logits[idx])
    assert torch.equal(feats["stem"], cache.tap_features("stem", idx))
    assert runner.stats["no_flip"] == 1


def test_forward_error_propagates_with_upstream_taps(x):
    _, fault, _, _, _, runner = _setup("m2", x, "default")

    def boom(*_):
        raise RuntimeError("fault broke the forward")

    handle = fault.get_submodule("layer2.0.conv1").register_forward_pre_hook(boom)
    try:
        with pytest.raises(RuntimeError):
            runner.run(["layer2.0.conv1"], np.arange(8))
    finally:
        handle.remove()
    assert {"stem", "stage1"} <= set(runner.partial) and "stage2" not in runner.partial


# --- persistence and the sweep -------------------------------------------------------------

def test_stale_cache_is_refused_and_rebuilt(tmp_path, x):
    clean, _, tap_list, sat, cache, _ = _setup("m1", x, "default")
    store.save(cache, tmp_path / "c")
    loaded = store.load(tmp_path / "c", cache.key)
    assert np.array_equal(loaded.features, cache.features)
    with pytest.raises(store.StaleCacheError):
        store.load(tmp_path / "c", cache.key | {"model_checksum": "other"})
    starts = cache.key["cuts"]
    rebuilt = store.ensure(tmp_path / "c", clean, x, starts, tap_list, {**sat, "stem": 2.0},
                           batch=40, log=lambda _: None)
    assert rebuilt.key["sat"]["stem"] == 2.0


def test_sweep_outputs_identical_full_and_suffix(x):
    clean, fault, tap_list, sat, _, runner = _setup("m2", x, "default")
    with torch.no_grad():
        clean_logits = clean(x).float().numpy()
    y = clean_logits.argmax(1)
    grid = plan(targets(fault, "m2"), "bf_w", seed=0, reps=2)
    args = {"epsilon": 0.01, "num_classes": 10, "seed": 0, "log": lambda _: None}

    full_feats: list[pd.DataFrame] = []
    with TapMonitor(fault, tap_list, mode="features", sat_thresholds=sat) as mon:
        full = sweep_run(fault, grid, x, y, clean_logits, monitor=mon, features_out=full_feats,
                         **args)
    suffix_feats: list[pd.DataFrame] = []
    suffix = sweep_run(fault, grid, x, y, clean_logits, features_out=suffix_feats, runner=runner,
                       **args)
    pd.testing.assert_frame_equal(full, suffix)
    pd.testing.assert_frame_equal(pd.concat(full_feats), pd.concat(suffix_feats))
