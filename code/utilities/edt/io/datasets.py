from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_curvature_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"k", "branch_a", "branch_b"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df
