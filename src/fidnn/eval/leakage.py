"""The leakage guard that must pass before any results build (SPEC §7, C2)."""

import pandas as pd

from fidnn.eval.splits import is_gen_config


class LeakageError(RuntimeError):
    """A results build that would report a leaked number fails here instead."""


def check(fault: pd.DataFrame, clean_splits: dict, grid_axes: dict[str, set],
          data_integrity: dict | None = None) -> list[str]:
    """Return the violations found; `assert_clean` turns them into a failure."""
    problems = []

    dev = set(fault.loc[fault.fault_split == "fault_dev", "injection_id"])
    test = set(fault.loc[fault.fault_split == "fault_test", "injection_id"])
    if dev & test:
        problems.append(f"{len(dev & test)} injection instances are in both fault_dev and "
                        "fault_test")

    in_dev = fault[fault.fault_split == "fault_dev"]
    if len(in_dev) and is_gen_config(in_dev.stratum, in_dev.budget).any():
        problems.append("fault_gen configurations (stratum sign, budget 4) appear in fault_dev")

    held = fault[fault.fault_split == "fault_test"]
    for axis, expected in grid_axes.items():
        missing = expected - set(held[axis].unique())
        if missing:
            problems.append(f"fault_test is missing {axis}: {sorted(missing)}")

    for name, ids in clean_splits.items():
        overlap = set(ids) & set(fault.get("record_id", pd.Series(dtype=object)))
        if overlap:
            problems.append(f"{len(overlap)} fault record ids appear in {name}")

    if data_integrity is not None:
        collisions = data_integrity.get("kaggle_vs_canonical_test_collisions")
        if collisions:
            problems.append(f"{collisions} pixel-hash collisions between Kaggle train and the "
                            "canonical test set")
    return problems


def assert_clean(*args, **kwargs) -> None:
    problems = check(*args, **kwargs)
    if problems:
        raise LeakageError("; ".join(problems))
