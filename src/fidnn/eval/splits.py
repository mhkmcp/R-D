"""Fault splits by injection instance, and the reserved `fault_gen` configs (SPEC §7)."""

import numpy as np
import pandas as pd

DEV_SHARE = 0.30
GEN_STRATA = ("sign",)   # configs deliberately never sampled into dev …
GEN_BUDGETS = (4,)       # … so `fault_gen` measures unseen-configuration generalisation


def is_gen_config(stratum: pd.Series, budget: pd.Series) -> pd.Series:
    return stratum.isin(GEN_STRATA) | budget.isin(GEN_BUDGETS)


def assign(outcomes: pd.DataFrame, seed: int = 0, dev_share: float = DEV_SHARE) -> pd.Series:
    """`fault_dev` / `fault_test` per injection instance; `fault_gen` is tagged separately.

    The split is over instances, not grid axes, so every cell appears in `fault_test` (§7).
    """
    per_injection = (outcomes[["injection_id", "stratum", "budget"]]
                     .drop_duplicates("injection_id").set_index("injection_id"))
    gen = is_gen_config(per_injection.stratum, per_injection.budget)
    eligible = per_injection.index[~gen].to_numpy()
    rng = np.random.default_rng(seed)
    dev = set(rng.choice(eligible, size=round(dev_share * len(eligible)), replace=False))
    split = pd.Series("fault_test", index=per_injection.index, name="fault_split")
    split[split.index.isin(dev)] = "fault_dev"
    return outcomes.injection_id.map(split)


def gen_mask(outcomes: pd.DataFrame) -> pd.Series:
    """`fault_gen` ⊂ `fault_test`: reported as its own row, never merged into the headline (§7)."""
    return is_gen_config(outcomes.stratum, outcomes.budget)
