"""Injectable tensors per model and precision, with their layer buckets (SPEC §4.1, §4.2)."""

from dataclasses import dataclass

import torch
from torch import nn

BUCKETS = ("early", "middle", "late")

# SPEC §4.2: ResNet buckets are the three stages; VGG-11-BN folds its five conv blocks into thirds.
BUCKET_OF = {
    "resnet": {"conv1": "early", "layer1": "early", "layer2": "middle", "layer3": "late",
               "fc": "late"},
    "vgg": {"block1": "early", "block2": "early", "block3": "middle", "block4": "late",
            "block5": "late", "classifier": "late"},
}
FAMILY = {"m1": "resnet", "m2": "resnet", "m3": "vgg"}


@dataclass(frozen=True)
class Target:
    layer: str      # module path
    tensor: str     # weight | bias | running_mean | running_var
    storage: str    # fp32 | qweight (int8, packed) | qbias (fp32, packed)
    numel: int
    bucket: str

    @property
    def width(self) -> int:
        return 8 if self.storage == "qweight" else 32

    @property
    def family(self) -> str:
        return {"weight": "bf_w", "bias": "bf_b"}.get(self.tensor, "bf_bn")


def is_quantized(module: nn.Module) -> bool:
    return hasattr(module, "set_weight_bias") and callable(getattr(module, "weight", None))


def targets(model: nn.Module, model_id: str) -> list[Target]:
    """Every weight, bias and BN running-stat tensor, in module order."""
    buckets = BUCKET_OF[FAMILY[model_id]]
    out = []
    for name, mod in model.named_modules():
        bucket = buckets.get(name.split(".")[0])
        if bucket is None:
            continue
        if is_quantized(mod):
            out.append(Target(name, "weight", "qweight", mod.weight().numel(), bucket))
            if mod.bias() is not None:
                out.append(Target(name, "bias", "qbias", mod.bias().numel(), bucket))
        elif isinstance(mod, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            for t in ("weight", "bias", "running_mean", "running_var"):
                v = getattr(mod, t, None)
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    out.append(Target(name, t, "fp32", v.numel(), bucket))
    return out


def for_mode(all_targets: list[Target], mode: str) -> list[Target]:
    """`bf_w`/`bf_b`/`bf_bn` by tensor family; stuck-at and `rnd_val` hit weights and biases."""
    if mode in ("bf_w", "bf_b", "bf_bn"):
        return [t for t in all_targets if t.family == mode]
    return [t for t in all_targets if t.family in ("bf_w", "bf_b")]
