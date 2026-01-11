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
    for agg in aggregations:
        if agg == "mean":
            result["mean"] = series.mean()
        elif agg == "min":
            result["min"] = series.min()
        elif agg == "max":
            result["max"] = series.max()

    if any(pd.isna(v) for v in result.values()):
        return None

    return result
