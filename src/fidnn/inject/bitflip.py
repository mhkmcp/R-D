"""In-place bit flips with exact XOR restore (SPEC §4.4)."""

import hashlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from torch import nn


def _mask(bit: int, width: int) -> int:
    """XOR mask for `bit` as a signed integer of `width` bits (the sign bit is negative)."""
    return -(1 << bit) if bit == width - 1 else 1 << bit


def flip_bits_(t: torch.Tensor, flat_idx: Sequence[int], bits: Sequence[int]) -> torch.Tensor:
    """Flip bit `bits[i]` of element `flat_idx[i]` of `t`, in place. Applying twice restores `t`."""
    if t.dtype == torch.float32:
        view, width = t.view(torch.int32).view(-1), 32
    elif t.dtype == torch.int8:
        view, width = t.view(-1), 8
    else:
        raise TypeError(f"unsupported dtype {t.dtype}")
    if not 0 <= max(bits) < width:
        raise ValueError(f"bit position out of range for {width}-bit elements")
    with torch.no_grad():
        for i, b in zip(flat_idx, bits):  # sequential: the same element may be hit twice
            view[i] ^= _mask(b, width)
    return t


def _requantise(q: torch.Tensor, int_repr: torch.Tensor) -> torch.Tensor:
    if q.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
        return torch._make_per_channel_quantized_tensor(
            int_repr, q.q_per_channel_scales(), q.q_per_channel_zero_points(), q.q_per_channel_axis())
    return torch._make_per_tensor_quantized_tensor(int_repr, q.q_scale(), q.q_zero_point())


def flip_quantized_weight_(module: nn.Module, flat_idx: Sequence[int], bits: Sequence[int]) -> None:
    """Flip int8 weight bits of a quantised conv/linear module (unpack → XOR → repack)."""
    w = module.weight()
    ir = w.int_repr().contiguous()
    flip_bits_(ir, flat_idx, bits)
    module.set_weight_bias(_requantise(w, ir), module.bias())


def flip_quantized_bias_(module: nn.Module, flat_idx: Sequence[int], bits: Sequence[int]) -> None:
    """Flip FP32 bias bits of a quantised conv/linear module; qnnpack keeps biases in FP32."""
    b = module.bias().detach().clone()
    flip_bits_(b, flat_idx, bits)
    module.set_weight_bias(module.weight(), b)


def _digest(h: "hashlib._Hash", name: str, v: object) -> None:
    """Hash tensor *bytes*, recursing into the tuples quantised Linear keeps its packed params in.

    Hashing `repr()` instead would be both blind (it truncates and rounds) and unstable: repacking
    a quantised Linear with identical values changes that repr.
    """
    if isinstance(v, torch.Tensor):
        h.update(name.encode())
        if v.is_quantized:
            qparams = (v.q_per_channel_scales() if v.qscheme() in
                       (torch.per_channel_affine, torch.per_channel_symmetric)
                       else torch.tensor([v.q_scale()]))
            _digest(h, f"{name}.qparams", qparams.float())
            v = v.int_repr()
        h.update(v.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(v, (tuple, list)):
        for i, item in enumerate(v):
            _digest(h, f"{name}.{i}", item)
    else:
        h.update(f"{name}={v!r}".encode())


def state_checksum(model: nn.Module) -> str:
    """Digest of the full `state_dict()` — parameters **and buffers** (BN running stats, §4.4)."""
    h = hashlib.blake2b(digest_size=16)
    for name, v in model.state_dict().items():
        _digest(h, name, v)
    return h.hexdigest()


@contextmanager
def injected(t: torch.Tensor, flat_idx: Sequence[int], bits: Sequence[int]) -> Iterator[None]:
    """Apply a flip for the duration of the block; restore in `finally`, even on exception."""
    flip_bits_(t, flat_idx, bits)
    try:
        yield
    finally:
        flip_bits_(t, list(reversed(flat_idx)), list(reversed(bits)))
