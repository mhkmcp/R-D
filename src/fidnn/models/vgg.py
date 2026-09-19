"""VGG-11-BN for CIFAR-10 (M3), no skip connections; matches the pinned reference (models/README.md)."""

from collections import OrderedDict

from torch import nn

from fidnn.models.common import Tap

# channels per conv, grouped by pooling block
CFG = [[64], [128], [256, 256], [512, 512], [512, 512]]


class VGG11BN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        cin = 3
        for b, block in enumerate(CFG, start=1):
            layers = OrderedDict()
            for k, cout in enumerate(block, start=1):
                layers[f"conv{k}"] = nn.Conv2d(cin, cout, 3, 1, 1)
                layers[f"bn{k}"] = nn.BatchNorm2d(cout)
                layers[f"relu{k}"] = nn.ReLU()
                layers[f"tap{k}"] = Tap()
                cin = cout
            layers["pool"] = nn.MaxPool2d(2)
            layers["tap_out"] = Tap()
            setattr(self, f"block{b}", nn.Sequential(layers))
        self.flatten = nn.Flatten()
        self.tap_pooled = Tap()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self.tap_logits = Tap()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for b in range(1, 6):
            x = getattr(self, f"block{b}")(x)
        x = self.tap_pooled(self.flatten(x))  # spatial size is 1×1 after five 2× pools
        return self.tap_logits(self.classifier(x))


def vgg11_bn(num_classes: int = 10) -> VGG11BN:
    return VGG11BN(num_classes)
