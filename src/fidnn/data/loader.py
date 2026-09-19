"""In-memory batch iterator with on-device augmentation (SPEC §3.2)."""

import math
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F


def random_crop_flip(x: torch.Tensor, pad: int, gen: torch.Generator) -> torch.Tensor:
    """Zero-pad by `pad`, random crop back to size, random horizontal flip — per sample."""
    b, c, h, w = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad))
    dy = torch.randint(0, 2 * pad + 1, (b,), generator=gen).to(dev)
    dx = torch.randint(0, 2 * pad + 1, (b,), generator=gen).to(dev)
    rows = (dy[:, None] + torch.arange(h, device=dev))[:, None, :, None]
    cols = (dx[:, None] + torch.arange(w, device=dev))[:, None, None, :]
    xp = xp.gather(2, rows.expand(b, c, h, w + 2 * pad)).gather(3, cols.expand(b, c, h, w))
    flip = (torch.rand(b, generator=gen) < 0.5).to(dev)[:, None, None, None]
    return torch.where(flip, xp.flip(3), xp)


class Batches:
    """Holds uint8 NHWC images on `device`; yields normalised (x, y) batches."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mean, std, batch_size: int,
                 device: str = "cpu", augment: bool = False, shuffle: bool = False, seed: int = 0):
        self.x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
        self.y = torch.from_numpy(y).to(device)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.batch_size, self.augment, self.shuffle = batch_size, augment, shuffle
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.y) / self.batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        n = len(self.y)
        order = torch.randperm(n, generator=self.gen) if self.shuffle else torch.arange(n)
        order = order.to(self.x.device)
        for i in range(0, n, self.batch_size):
            idx = order[i:i + self.batch_size]
            x = self.x[idx].float().div_(255)
            if self.augment:
                x = random_crop_flip(x, 4, self.gen)
            yield (x - self.mean) / self.std, self.y[idx]
