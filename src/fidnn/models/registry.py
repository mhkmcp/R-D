"""Model lineup (SPEC §3): M1 smoke test, M2 and M3 primary."""

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from fidnn.models.resnet_cifar import resnet8, resnet20
from fidnn.models.vgg import vgg11_bn


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    name: str
    num_classes: int
    builder: Callable[[int], nn.Module]
    thesis_result: bool


MODELS: dict[str, ModelSpec] = {
    "m1": ModelSpec("m1", "ResNet-8", 2, resnet8, thesis_result=False),
    "m2": ModelSpec("m2", "ResNet-20", 10, resnet20, thesis_result=True),
    "m3": ModelSpec("m3", "VGG-11-BN", 10, vgg11_bn, thesis_result=True),
}


def build(model_id: str) -> nn.Module:
    spec = MODELS[model_id]
    return spec.builder(spec.num_classes)
