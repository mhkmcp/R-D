from torch import nn


def Tap() -> nn.Identity:
    """Tap point for the monitor. See `nn.Identity`; must not be subclassed (models/README.md)."""
    return nn.Identity()
