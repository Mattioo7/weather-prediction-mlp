import numpy as np
import pandas as pd
from pathlib import Path
from .io import load_variable_csv
from .preprocessing import daily_aggregate
from .encoding import encode_wind_direction_deg


def build_dataset(cfg, verbose: bool | str = False):
    data_dir = Path(cfg.data_dir)

    debug = verbose == "debug"

    def log(msg):
        if verbose:
            print(msg)

    def debug_log(msg):
        if debug:
            print(msg)

    log("=== BUILD DATASET START ===")
    log(f"Data directory: {data_dir.resolve()}")
    log(f"Target variable: {cfg.target}")
    log(f"Window size (I): {cfg.window_size}, Skip day X: {cfg.skip_day}")
    log(f"Max missing ratio per day: {cfg.max_missing_ratio_per_day}")
    log(f"Input variables: {cfg.input_variables}")

    # -------------------- LOAD DATA --------------------
    data = {}
    for var in cfg.input_variables:
        path = data_dir / "train" / f"{var}_train.csv"
        log(f"Loading {path.name}")
        data[var] = load_variable_csv(path)
        debug_log(f"{var}: shape={data[var].shape}")

    # -------------------- CITIES --------------------
    cities = list(cfg.cities) if cfg.cities is not None else list(data[cfg.input_variables[0]].columns)
    log(f"Number of cities: {len(cities)}")

    X_rows = []
    Y_rows = []

    # -------------------- COUNTERS --------------------
    total_windows = 0
    rejected_windows = 0
    rejected_days = 0
    rejected_targets = 0

    # -------------------- MAIN LOOP --------------------
    for city_idx, city in enumerate(cities, start=1):
        log(f"\n--- City [{city_idx}/{len(cities)}]: {city} ---")

        city_series = {var: df[city] for var, df in data.items()}
        dates = city_series[cfg.target].index.normalize().unique()

        debug_log(f"Total days available: {len(dates)}")

        for i in range(cfg.window_size, len(dates) - 1):
            total_windows += 1

            day_I = dates[i - cfg.window_size : i]
            day_O = dates[i + 1] if cfg.skip_day else dates[i]

            features = []
            valid = True

            for d in day_I:
                debug_log(f"  Processing day I: {d.date()}")

                for var, aggs in cfg.aggregations.items():
                    day_data = city_series[var][d : d + pd.Timedelta("1D")]

                    agg = daily_aggregate(
                        day_data,
                        aggs,
                        cfg.max_missing_ratio_per_day,
                    )

                    if agg is None:
                        rejected_days += 1
                        valid = False
                        debug_log(
                            f"    REJECT day {d.date()} | var={var} | missing ratio too high or NaN"
                        )
                        break

                    if var == "wind_direction" and cfg.encode_wind_direction:
                        sin, cos = encode_wind_direction_deg(agg["mean"])
                        features.extend([sin, cos])
                    else:
                        features.extend(agg.values())

                if not valid:
                    break

            if not valid:
                rejected_windows += 1
                continue

            # -------------------- TARGET --------------------
            target_series = city_series[cfg.target][
                day_O : day_O + pd.Timedelta("1D")
            ]

            target_day = target_series.mean()

            if np.isnan(target_day):
                rejected_targets += 1
                rejected_windows += 1
                debug_log(
                    f"  REJECT target day {day_O.date()} | NaN target"
                )
                continue

            X_rows.append(features)
            Y_rows.append(target_day)

    # -------------------- FINAL ARRAYS --------------------
    X = np.asarray(X_rows, dtype=float)
    Y = np.asarray(Y_rows, dtype=float).reshape(-1, 1)

    # -------------------- SUMMARY --------------------
    log("\n=== BUILD DATASET SUMMARY ===")
    log(f"Total candidate windows: {total_windows}")
    log(f"Accepted samples: {len(X_rows)}")
    log(f"Rejected windows: {rejected_windows}")
    log(f"  - rejected due to day aggregation: {rejected_days}")
    log(f"  - rejected due to NaN target: {rejected_targets}")
    log(f"Final X shape: {X.shape}")
    log(f"Final Y shape: {Y.shape}")
    log("=== BUILD DATASET END ===")

    return X, Y
