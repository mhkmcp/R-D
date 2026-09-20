"""Apply one planned injection to a live model and restore it (SPEC §4.4)."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from torch import nn

from fidnn.inject.bitflip import flip_bits_, flip_quantized_bias_, flip_quantized_weight_

STUCK_AT = {"sa0": 0, "sa1": 1}


def _fp32_tensor(model: nn.Module, layer: str, tensor: str) -> torch.Tensor:
    return getattr(model.get_submodule(layer), tensor).data


def _bit_is_set(value: int, bit: int) -> bool:
    return bool(value >> bit & 1)


def _stuck_at_subset(model: nn.Module, flips: Sequence[dict], stuck: int) -> list[dict]:
    """Stuck-at is a flip only where the bit does not already hold the stuck value."""
    keep = []
    for f in flips:
        if f["storage"] == "qweight":
            mod = model.get_submodule(f["layer"])
            cur = int(mod.weight().int_repr().view(-1)[f["flat_index"]].item()) & 0xFF
        else:
            t = (model.get_submodule(f["layer"]).bias() if f["storage"] == "qbias"
                 else _fp32_tensor(model, f["layer"], f["tensor"]))
            cur = int(t.view(-1)[f["flat_index"]].view(torch.int32).item()) & 0xFFFFFFFF
        if _bit_is_set(cur, f["bit"]) != bool(stuck):
            keep.append(f)
    return keep


def _apply(model: nn.Module, flips: Sequence[dict]) -> None:
    """Group by tensor so quantised modules are unpacked and repacked once per tensor."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for f in flips:
        groups.setdefault((f["layer"], f["tensor"], f["storage"]), []).append(f)
    for (layer, tensor, storage), fs in groups.items():
        idx = [int(f["flat_index"]) for f in fs]
        bits = [int(f["bit"]) for f in fs]
        mod = model.get_submodule(layer)
        if storage == "qweight":
            flip_quantized_weight_(mod, idx, bits)
        elif storage == "qbias":
            flip_quantized_bias_(mod, idx, bits)
        else:
            flip_bits_(_fp32_tensor(model, layer, tensor).view(-1), idx, bits)


@contextmanager
def injected_flips(model: nn.Module, flips: Sequence[dict], mode: str) -> Iterator[list[dict]]:
    """Apply an injection for the duration of the block; restore in `finally` (SPEC §4.4).

    Yields the flips actually applied, which for stuck-at faults is a subset of those planned.
    """
    applied = _stuck_at_subset(model, flips, STUCK_AT[mode]) if mode in STUCK_AT else list(flips)
    _apply(model, applied)
    try:
        yield applied
    finally:
        _apply(model, list(reversed(applied)))
