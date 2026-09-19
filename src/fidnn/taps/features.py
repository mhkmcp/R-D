"""Per-tap feature descriptor, Blocks A–D (SPEC §5.2)."""

import torch

QUANTILES = (0.01, 0.25, 0.5, 0.75, 0.99)
BLOCK_A = ["mean", "std", "min", "max", *(f"q{int(q * 100):02d}" for q in QUANTILES)]
BLOCK_B = ["zero_frac", "sat_frac", "nan_count", "inf_count"]
BLOCK_C = ["l1_rms", "l2_rms", "linf", "peakiness", "energy_gini"]
BLOCK_D = ["energy_entropy", "top1_share", "top2_share", "top4_share",
           "ch_mean_mean", "ch_mean_std", "ch_std_mean", "ch_std_std"]
FEATURE_NAMES = BLOCK_A + BLOCK_B + BLOCK_C + BLOCK_D
WIDTH = len(FEATURE_NAMES)  # 26
_EPS = 1e-12


def _gini(e: torch.Tensor) -> torch.Tensor:
    """Gini coefficient of non-negative energies along dim 1 (0 = uniform, →1 = concentrated)."""
    k = e.shape[1]
    s, _ = torch.sort(e, dim=1)
    idx = torch.arange(1, k + 1, device=e.device, dtype=e.dtype)
    total = s.sum(1)
    return (2 * (s * idx).sum(1)) / (k * total + _EPS) - (k + 1) / k


def tap_features(a: torch.Tensor, sat_threshold: float = float("inf"),
                 dense: bool = False) -> torch.Tensor:
    """Return an (N, 26) float32 descriptor for activation `a` of shape (N, C, H, W) or (N, D)."""
    if a.is_quantized:
        a = a.dequantize()
    a = a.float()
    n = a.shape[0]
    if a.dim() == 2 and not dense:
        a = a[:, :, None, None]
    flat = a.reshape(n, -1)
    numel = flat.shape[1]

    # Block A — distributional shape
    q = torch.quantile(flat, torch.tensor(QUANTILES, device=a.device), dim=1).T
    block_a = torch.cat([
        torch.stack([flat.mean(1), flat.std(1, correction=0), flat.amin(1), flat.amax(1)], 1),
        q,
    ], 1)

    # Block B — sparsity and saturation
    block_b = torch.stack([
        (flat == 0).float().mean(1),
        (flat > sat_threshold).float().mean(1),
        torch.isnan(flat).float().sum(1),
        torch.isinf(flat).float().sum(1),
    ], 1)

    # Block C — energy
    absf = flat.abs()
    l2 = flat.norm(dim=1)
    linf = absf.amax(1)
    energy = (a * a).reshape(n, a.shape[1], -1).sum(2) if not dense else flat * flat
    block_c = torch.stack([
        absf.sum(1) / numel**0.5,
        l2 / numel**0.5,
        linf,
        linf / (l2 + _EPS),
        _gini(energy),
    ], 1)

    # Block D — structure over the channel (or unit) energy distribution
    p = energy / (energy.sum(1, keepdim=True) + _EPS)
    entropy = -(p * torch.log(p + _EPS)).sum(1)
    top = torch.sort(p, dim=1, descending=True).values.cumsum(1)
    k = p.shape[1]
    tops = [top[:, min(t, k) - 1] for t in (1, 2, 4)]
    if dense:
        spatial = [torch.zeros(n, device=a.device)] * 4
    else:
        per_ch = a.reshape(n, a.shape[1], -1)
        ch_mean = per_ch.mean(2)
        ch_std = per_ch.std(2, correction=0)
        spatial = [ch_mean.mean(1), ch_mean.std(1, correction=0),
                   ch_std.mean(1), ch_std.std(1, correction=0)]
    block_d = torch.stack([entropy, *tops, *spatial], 1)

    return torch.cat([block_a, block_b, block_c, block_d], 1)
