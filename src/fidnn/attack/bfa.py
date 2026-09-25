"""L1 attacker: BFA progressive bit search on INT8 weights (Rakin et al. 2019). SPEC §2, §6.4."""

import copy
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.ao.quantization.quantize_fx import fuse_fx

from fidnn.inject.bitflip import flip_quantized_weight_
from fidnn.inject.targets import is_quantized


@dataclass(frozen=True)
class Flip:
    layer: str
    flat_index: int
    bit: int


def surrogate(fp32: nn.Module, int8: nn.Module) -> tuple[nn.Module, dict[str, nn.Module]]:
    """Float twin of the INT8 model for gradient ranking: same fused graph, dequantised weights.

    Autograd does not run through quantised kernels, so BFA's ranking step needs this. See
    `fuse_fx`; module paths survive fusion, so they address both models.
    """
    twin = fuse_fx(copy.deepcopy(fp32).eval())
    convs = {}
    for name, mod in int8.named_modules():
        if not is_quantized(mod):
            continue
        target = twin.get_submodule(name)
        conv = target[0] if isinstance(target, nn.Sequential) else target
        with torch.no_grad():
            conv.weight.copy_(mod.weight().dequantize().reshape(conv.weight.shape))
            if mod.bias() is not None:
                conv.bias.copy_(mod.bias())
        convs[name] = conv
    return twin, convs


def element_scales(module: nn.Module) -> torch.Tensor:
    """Per-element quantisation scale, so an int8 step maps to a weight step."""
    w = module.weight()
    if w.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
        per_channel = w.q_per_channel_scales().float()
        return per_channel.repeat_interleave(w.numel() // len(per_channel))
    return torch.full((w.numel(),), w.q_scale(), dtype=torch.float32)


def flipped_values(q: torch.Tensor, bit: int) -> torch.Tensor:
    """int8 values after flipping `bit`, in two's-complement byte semantics."""
    u = (q.to(torch.int16) & 0xFF) ^ (1 << bit)
    return torch.where(u >= 128, u - 256, u)


def rank_candidates(module: nn.Module, grad: torch.Tensor, k: int) -> list[tuple[int, int, float]]:
    """Top-`k` (index, bit, first-order loss gain) for one layer; only loss-increasing flips."""
    q = module.weight().int_repr().flatten()
    scale = element_scales(module)
    best = []
    for bit in range(8):
        gain = grad * (flipped_values(q, bit) - q).float() * scale
        top = torch.topk(gain, min(k, gain.numel()))
        best += [(int(i), bit, float(v)) for v, i in zip(top.values, top.indices) if v > 0]
    return sorted(best, key=lambda c: -c[2])[:k]


@torch.no_grad()
def _loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cross_entropy(model(x), y))


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 500) -> float:
    correct = sum(int((model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum())
                  for i in range(0, len(y), batch))
    return 100.0 * correct / len(y)


def attack(int8: nn.Module, fp32: nn.Module, x: torch.Tensor, y: torch.Tensor, budget: int,
           candidates_per_layer: int = 10, eval_xy: tuple[torch.Tensor, torch.Tensor] | None = None,
           log=print, stop_below: float | None = None) -> pd.DataFrame:
    """Progressive bit search: rank by gradient, evaluate the shortlist, keep the best flip.

    Flips stay applied, as the threat model is persistent corruption (§2). The returned records
    replay the attack exactly.
    """
    layers = {n: m for n, m in int8.named_modules() if is_quantized(m)}
    rows = []
    for step in range(1, budget + 1):
        twin, convs = surrogate(fp32, int8)
        twin.zero_grad(set_to_none=True)
        print("x shape:", x.shape)
        # print("logits shape:", logits.shape)
        print("y shape:", y.shape)
        print("y:", y)
        print("y min/max:", y.min().item(), y.max().item())
        F.cross_entropy(twin(x), y).backward()
        before = _loss(int8, x, y)

        best, best_loss = None, before
        for name, module in layers.items():
            grad = convs[name].weight.grad
            if grad is None:
                continue
            for idx, bit, _ in rank_candidates(module, grad.detach().flatten(), candidates_per_layer):
                flip_quantized_weight_(module, [idx], [bit])
                loss = _loss(int8, x, y)
                flip_quantized_weight_(module, [idx], [bit])
                if loss > best_loss:
                    best, best_loss = Flip(name, idx, bit), loss
        if best is None:
            log(f"step {step}: no loss-increasing flip found, stopping")
            break

        flip_quantized_weight_(layers[best.layer], [best.flat_index], [best.bit])
        row = {"step": step, "attacker": "L1", "layer": best.layer,
               "flat_index": best.flat_index, "bit": best.bit,
               "loss_before": before, "loss_after": best_loss}
        if eval_xy is not None:
            row["accuracy"] = accuracy(int8, *eval_xy)
        rows.append(row)
        log(f"step {step}: {best.layer}[{best.flat_index}] bit {best.bit}, "
            f"loss {before:.3f} → {best_loss:.3f}"
            + (f", accuracy {row['accuracy']:.2f} %" if eval_xy is not None else ""))
        if stop_below is not None and row.get("accuracy", float("inf")) < stop_below:
            break
    return pd.DataFrame(rows)


def replay(int8: nn.Module, records: pd.DataFrame) -> None:
    """Re-apply a recorded attack to a clean INT8 model, in order."""
    layers = {n: m for n, m in int8.named_modules() if is_quantized(m)}
    for r in records.itertuples():
        flip_quantized_weight_(layers[r.layer], [int(r.flat_index)], [int(r.bit)])
