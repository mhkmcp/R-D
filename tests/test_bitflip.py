"""SPEC §11.4: bit-flip round trip, known value, in-place restore (buffers included)."""

import pytest
import torch

from fidnn.inject.bitflip import flip_bits_, flip_quantized_weight_, injected, state_checksum
from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import build


@pytest.mark.parametrize("bit", range(32))
def test_fp32_round_trip_every_bit(bit):
    t = torch.randn(64)
    ref = t.clone()
    idx = list(range(0, 64, 7))
    flip_bits_(t, idx, [bit] * len(idx))
    assert not torch.equal(t.view(torch.int32), ref.view(torch.int32))
    flip_bits_(t, idx, [bit] * len(idx))
    assert torch.equal(t.view(torch.int32), ref.view(torch.int32))


@pytest.mark.parametrize("bit", range(8))
def test_int8_round_trip_every_bit(bit):
    t = torch.randint(-128, 128, (64,), dtype=torch.int8)
    ref = t.clone()
    flip_bits_(t, [3, 17], [bit, bit])
    assert not torch.equal(t, ref)
    flip_bits_(t, [3, 17], [bit, bit])
    assert torch.equal(t, ref)


def test_fp32_bit30_known_values():
    t = torch.tensor([1.0, 0.5])
    flip_bits_(t, [0, 1], [30, 30])
    assert torch.isinf(t[0]) and t[0] > 0
    assert t[1].item() == 2.0**127


def test_sign_bit():
    t = torch.tensor([3.0])
    flip_bits_(t, [0], [31])
    assert t.item() == -3.0
    i = torch.tensor([5], dtype=torch.int8)
    flip_bits_(i, [0], [7])
    assert i.item() == 5 - 128


def test_same_element_hit_twice_keeps_both_flips():
    t = torch.tensor([1.0])
    flip_bits_(t, [0, 0], [0, 1])
    assert t.view(torch.int32).item() == torch.tensor([1.0]).view(torch.int32).item() ^ 0b11


def test_bit_out_of_range():
    with pytest.raises(ValueError):
        flip_bits_(torch.zeros(2, dtype=torch.int8), [0], [8])


def test_restore_after_exception_covers_bn_buffer():
    model = build("m2")
    clean = state_checksum(model)
    running_var = model.layer2[0].bn1.running_var
    with pytest.raises(RuntimeError), injected(running_var, [3], [31]):
        assert state_checksum(model) != clean  # buffer flip is visible to the checksum
        raise RuntimeError("fault raised mid-probe")
    assert state_checksum(model) == clean


def test_quantized_weight_round_trip():
    q = quantize_ptq(build("m1"), [torch.randn(4, 3, 32, 32)])
    mod = dict(q.named_modules())["layer1.0.conv2"]
    clean = state_checksum(q)
    flip_quantized_weight_(mod, [5, 40], [7, 2])
    assert state_checksum(q) != clean
    flip_quantized_weight_(mod, [40, 5], [2, 7])
    assert state_checksum(q) == clean
