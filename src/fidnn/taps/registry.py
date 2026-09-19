"""Tap IDs → `Tap` module paths, default and extended sets per model (SPEC §5.1)."""

from dataclasses import dataclass

from fidnn.models.vgg import CFG as VGG_CFG


@dataclass(frozen=True)
class TapInfo:
    tap_id: str
    path: str
    pre_add: bool = False  # ResNet only: observes the residual branch before the skip addition


def _resnet_taps(n: int) -> tuple[list[TapInfo], list[TapInfo]]:
    stem = TapInfo("stem", "tap_stem")
    pooled = TapInfo("pooled", "tap_pooled")
    logits = TapInfo("logits", "tap_logits")
    stages = [TapInfo(f"stage{s}", f"layer{s}.{n - 1}.tap_out") for s in (1, 2, 3)]
    blocks = [
        TapInfo(f"s{s}b{b + 1}", f"layer{s}.{b}.tap_out") for s in (1, 2, 3) for b in range(n)
    ]
    default = [stem, *stages, pooled, logits]
    extended = [stem, *blocks]
    if n > 1:
        extended.append(TapInfo(f"s3b{n}_pre", f"layer3.{n - 1}.tap_pre", pre_add=True))
    extended += [pooled, logits]
    return default, extended


def _vgg_taps() -> tuple[list[TapInfo], list[TapInfo]]:
    pooled = TapInfo("pooled", "tap_pooled")
    logits = TapInfo("logits", "tap_logits")
    default = [TapInfo(f"block{b}", f"block{b}.tap_out") for b in range(1, 6)]
    convs = [
        TapInfo(f"conv{b}_{k}", f"block{b}.tap{k}")
        for b, block in enumerate(VGG_CFG, start=1)
        for k in range(1, len(block) + 1)
    ]
    return [*default, pooled, logits], [*convs, pooled, logits]


_M1 = _resnet_taps(1)
_M2 = _resnet_taps(3)
_M3 = _vgg_taps()

TAPS: dict[str, dict[str, list[TapInfo]]] = {
    "m1": {"default": _M1[0], "extended": _M1[0]},  # SPEC §5.1: M1 has no extended set
    "m2": {"default": _M2[0], "extended": _M2[1]},
    "m3": {"default": _M3[0], "extended": _M3[1]},
}


def taps(model_id: str, tap_set: str = "default") -> list[TapInfo]:
    return TAPS[model_id][tap_set]
