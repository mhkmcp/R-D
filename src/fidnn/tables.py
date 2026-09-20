"""Markdown tables for the milestone records; no tabulate dependency."""

import pandas as pd


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def cell(v: object) -> str:
        return f"{v:{floatfmt}}" if isinstance(v, float) else str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
