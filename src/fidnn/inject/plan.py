"""§4.2 injection planner: stratified bit strata, layer buckets, budgets, repetitions."""

from dataclasses import asdict

import numpy as np
import pandas as pd

from fidnn.inject.targets import BUCKETS, Target, for_mode

# SPEC §4.2. FP32 strata are the SPEC's five; INT8 "all 8 positions, sign/MSB separated" is read as
# five strata too, so both precisions have the same cell count (inject/README.md).
STRATA: dict[int, dict[str, list[int]]] = {
    32: {"sign": [31], "exp_msb": [30, 29, 28, 27], "exp_low": [26, 25, 24, 23],
         "mant_high": list(range(22, 15, -1)), "mant_low": list(range(15, -1, -1))},
    8: {"sign": [7], "msb": [6], "high": [5, 4], "mid": [3, 2], "low": [1, 0]},
}
REDUCED_STRATA = {32: ("exp_msb", "mant_low"), 8: ("msb", "low")}
REDUCED_MODES = ("sa0", "sa1", "rnd_val", "bf_bn", "bf_act")
FULL_MODES = ("bf_w", "bf_b")


def width_of(targets: list[Target]) -> int:
    widths = {t.width for t in targets}
    if len(widths) != 1:
        raise ValueError(f"mixed storage widths in one mode: {widths}")
    return widths.pop()


def cells(mode: str, width: int, budgets: tuple[int, ...]) -> list[tuple[str, str, int]]:
    """(bucket, stratum, budget) cells; reduced modes run late bucket, 2 strata, budgets {1, 16}."""
    if mode in REDUCED_MODES:
        return [("late", s, b) for s in REDUCED_STRATA[width] for b in (1, 16)]
    return [(bucket, s, b) for bucket in BUCKETS for s in STRATA[width] for b in budgets]


def _pool(targets: list[Target], bucket: str) -> list[Target]:
    return [t for t in targets if t.bucket == bucket]


def plan(all_targets: list[Target], mode: str, seed: int, reps: int = 100,
         budgets: tuple[int, ...] = (1, 4, 16)) -> pd.DataFrame:
    """One row per flip; `injection_id` groups the flips applied together (SPEC §4.4 record)."""
    targets = for_mode(all_targets, mode)
    width = width_of(targets)
    rng = np.random.default_rng(seed)
    rows, injection_id = [], 0
    for bucket, stratum, budget in cells(mode, width, budgets):
        pool = _pool(targets, bucket)
        if not pool:
            continue
        sizes = np.array([t.numel for t in pool])
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        bits = STRATA[width][stratum]
        for _ in range(reps):
            # draw distinct (element, bit) pairs so two flips in one injection cannot cancel out
            picks = rng.choice(offsets[-1] * len(bits), size=budget, replace=False)
            for p in picks:
                flat, bit = divmod(int(p), len(bits))
                k = int(np.searchsorted(offsets, flat, side="right") - 1)
                t = pool[k]
                rows.append({"injection_id": injection_id, "mode": mode, "attacker": "L0",
                             "bucket": bucket, "stratum": stratum, "budget": budget,
                             **{f: v for f, v in asdict(t).items() if f != "bucket"},
                             "flat_index": flat - int(offsets[k]), "bit": bits[bit]})
            injection_id += 1
    return pd.DataFrame(rows)


def rnd_val_plan(all_targets: list[Target], seed: int, reps: int = 100) -> pd.DataFrame:
    """`rnd_val`: a uniformly random replacement value, expressed as the bits it differs in.

    XOR-ing a uniformly random mask gives a uniformly random result, so whole-value corruption
    reuses the flip path and stays exactly restorable (SPEC §4.4).
    """
    targets = for_mode(all_targets, "rnd_val")
    width = width_of(targets)
    rng = np.random.default_rng(seed)
    rows, injection_id = [], 0
    for bucket, _, budget in cells("rnd_val", width, ()):
        pool = _pool(targets, bucket)
        if not pool:
            continue
        sizes = np.array([t.numel for t in pool])
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        for _ in range(reps):
            for flat in rng.choice(offsets[-1], size=budget, replace=False):
                k = int(np.searchsorted(offsets, flat, side="right") - 1)
                t = pool[k]
                mask = rng.integers(1, 2**width)  # never 0: that would be a no-op corruption
                for bit in range(width):
                    if mask >> bit & 1:
                        rows.append({"injection_id": injection_id, "mode": "rnd_val",
                                     "attacker": "L0", "bucket": bucket, "stratum": "-",
                                     "budget": budget,
                                     **{f: v for f, v in asdict(t).items() if f != "bucket"},
                                     "flat_index": int(flat) - int(offsets[k]), "bit": bit})
            injection_id += 1
    return pd.DataFrame(rows)
