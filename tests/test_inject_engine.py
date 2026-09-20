"""SPEC §4.4: in-place injection with exact restore, across storages and fault modes."""

import numpy as np
import pytest
import torch

from fidnn.inject.bitflip import state_checksum
from fidnn.inject.engine import injected_flips
from fidnn.inject.plan import plan
from fidnn.inject.targets import targets
from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import build


@pytest.fixture(scope="module")
def m2():
    return build("m2").eval()


@pytest.fixture(scope="module")
def m2_int8(m2):
    return quantize_ptq(m2, [torch.randn(8, 3, 32, 32)])


def _flips(model, model_id, mode, reps=1, seed=0):
    p = plan(targets(model, model_id), mode, seed=seed, reps=reps)
    return [g.to_dict("records") for _, g in p.groupby("injection_id")]


@pytest.mark.parametrize("mode", ["bf_w", "bf_b", "bf_bn", "sa0", "sa1"])
def test_state_returns_to_clean_after_every_injection(m2, mode):
    clean = state_checksum(m2)
    for flips in _flips(m2, "m2", mode, reps=2):
        with injected_flips(m2, flips, mode):
            pass
        assert state_checksum(m2) == clean, mode


def test_injection_actually_changes_state(m2):
    clean = state_checksum(m2)
    changed = 0
    for flips in _flips(m2, "m2", "bf_w", reps=2):
        with injected_flips(m2, flips, "bf_w"):
            changed += state_checksum(m2) != clean
    assert changed == len(_flips(m2, "m2", "bf_w", reps=2))


def test_restore_runs_after_an_exception_mid_probe(m2):
    clean = state_checksum(m2)
    flips = _flips(m2, "m2", "bf_bn", reps=1)[0]
    with pytest.raises(RuntimeError), injected_flips(m2, flips, "bf_bn"):
        raise RuntimeError("probe blew up")
    assert state_checksum(m2) == clean


@pytest.mark.parametrize("mode", ["bf_w", "bf_b"])
def test_quantised_weights_and_biases_restore(m2_int8, mode):
    clean = state_checksum(m2_int8)
    for flips in _flips(m2_int8, "m2", mode, reps=2):
        with injected_flips(m2_int8, flips, mode) as applied:
            assert applied and state_checksum(m2_int8) != clean
        assert state_checksum(m2_int8) == clean


def test_checksum_sees_inside_quantised_linear_packed_params(m2_int8):
    """Regression: the packed tuple's repr is truncated and changes on any repack (§4.4 guard)."""
    fc = m2_int8.get_submodule("fc")
    clean = state_checksum(m2_int8)
    fc.set_weight_bias(fc.weight(), fc.bias().detach().clone())
    assert state_checksum(m2_int8) == clean, "repacking identical values must not look like a fault"
    from fidnn.inject.bitflip import flip_quantized_weight_
    flip_quantized_weight_(fc, [3], [0])             # lowest bit of one int8 weight
    assert state_checksum(m2_int8) != clean
    flip_quantized_weight_(fc, [3], [0])
    assert state_checksum(m2_int8) == clean


def test_stuck_at_only_flips_bits_that_differ(m2):
    t = m2.get_submodule("layer3.2.conv2").weight.data
    t.view(-1)[:4] = torch.tensor([1.0, 1.0, 1.0, 1.0])  # bit 0 of 1.0f is 0
    flips = [{"layer": "layer3.2.conv2", "tensor": "weight", "storage": "fp32",
              "flat_index": i, "bit": 0} for i in range(4)]
    with injected_flips(m2, flips, "sa0") as applied:
        assert applied == []                       # already 0: stuck-at-0 is a no-op
    with injected_flips(m2, flips, "sa1") as applied:
        assert len(applied) == 4
        assert (t.view(-1)[:4].view(torch.int32) & 1).all()


def test_one_injection_hitting_one_element_twice_still_restores(m2):
    flips = [{"layer": "conv1", "tensor": "weight", "storage": "fp32", "flat_index": 5, "bit": 3},
             {"layer": "conv1", "tensor": "weight", "storage": "fp32", "flat_index": 5, "bit": 20}]
    clean = state_checksum(m2)
    with injected_flips(m2, flips, "bf_w"):
        assert state_checksum(m2) != clean
    assert state_checksum(m2) == clean


def test_forward_still_runs_under_injection(m2):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    p = plan(targets(m2, "m2"), "bf_w", seed=0, reps=1)
    exponent = p[p.stratum == "exp_msb"]          # a stratum whose effect is always visible
    flips = exponent[exponent.injection_id == exponent.injection_id.iloc[0]].to_dict("records")
    with torch.no_grad():
        before = m2(x)
        with injected_flips(m2, flips, "bf_w"):
            during = m2(x)
        after = m2(x)
    assert torch.equal(before, after)
    assert not np.allclose(before.numpy(), during.numpy())
