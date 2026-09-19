import torch.nn as nn


def Tap() -> nn.Identity:  # noqa: N802 — used like a module class at construction sites
    """Tap point for the monitor. See `nn.Identity`; must not be subclassed (models/README.md)."""
    return nn.Identity()
