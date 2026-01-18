import pdb

import numpy as np
import pandas as pd

def daily_aggregate(
    series: pd.Series,
    aggregations: list[str],
    max_missing_ratio: float,
) -> dict[str, float] | None:

    missing_ratio = series.isna().mean()
    if missing_ratio > max_missing_ratio:
        return None

    result = {}
    clean = series.dropna()

    for agg in aggregations:
        if agg == "mean":
            result["mean"] = series.mean()
        elif agg == "min":
            result["min"] = series.min()
        elif agg == "max":
            result["max"] = series.max()
        elif agg == "trend":
            # potrzeba co najmniej 2 punktów
            if len(clean) < 2:
                return None

            # indeks czasu: 0, 1, 2, ...
            t = np.arange(len(clean), dtype=float)
            y = clean.to_numpy(dtype=float)

            # regresja liniowa y = a*t + b
            a, _ = np.polyfit(t, y, 1)
            result["trend"] = float(a)
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

    if any(pd.isna(v) for v in result.values()):
        return None

    return result
