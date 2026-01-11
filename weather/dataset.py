import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pdb

from .io import load_variable_csv
from .preprocessing import daily_aggregate
from .encoding import encode_wind_direction_deg


def build_dataset(cfg, verbose: bool | str = False):
    data_dir = Path(cfg.data_dir)

    debug = verbose == "debug"
    use_tqdm = debug is not True

    def log(msg):
        if verbose:
            print(msg)

    def debug_log(msg):
        if debug:
            print(msg)

    log("=== BUILD DATASET START ===")
    log(f"Data directory: {data_dir.resolve()}")
    log(f"Target variable: {cfg.target}")
    log(f"Window aggregation mode: {cfg.window_aggregation}")
    log(f"Window size (I): {cfg.window_size}, Skip day X: {cfg.skip_day}")
    log(f"Max missing ratio per day: {cfg.max_missing_ratio_per_day}")
    log(f"Input variables: {cfg.input_variables}")

    # -------------------- LOAD DATA --------------------
    data = {}
    for var in cfg.input_variables:
        path = data_dir / cfg.split / f"{var}_{cfg.split}.csv"
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

        day_iter = range(cfg.window_size, len(dates) - 1)

        if use_tqdm:
            day_iter = tqdm(day_iter, desc=f"{city} | windows")

        for i in day_iter:
            total_windows += 1

            day_I = dates[i - cfg.window_size : i]
            day_O = dates[i + 1] if cfg.skip_day else dates[i]

            debug_log(f"Processing window | days I: {[d.date() for d in day_I]} -> day O: {day_O.date()}")

            features = []
            valid = True

            # ==========================================================
            # FLATTEN WINDOW  (agregacja dnia + konkatenacja dni)
            # ==========================================================
            if cfg.window_aggregation == "flatten":
                for d in day_I:
                    debug_log(f"  Processing day I: {d.date()}")

                    for var, aggs in cfg.aggregations.items():
                        day_data = city_series[var][d : d + pd.Timedelta("1D")] # TODO: check slicing

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

            # ==========================================================
            # AGGREGATE WINDOW  (agregacja dnia + agregacja po dniach)
            # ==========================================================
            elif cfg.window_aggregation == "aggregate":
                day_feature_vectors = []

                for d in day_I:
                    debug_log(f"  Processing day I: {d.date()}")

                    day_features = []
                    for var, aggs in cfg.aggregations.items():
                        day_data = city_series[var][d: d + pd.Timedelta("1D")]  # TODO: check slicing

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
                            day_features.extend([sin, cos])
                        else:
                            day_features.extend(agg.values())

                    if not valid:
                        break

                    day_feature_vectors.append(day_features)

                if valid:
                    W = np.asarray(day_feature_vectors, dtype=float)
                    # pdb.set_trace()
                    features = W.mean(axis=0).tolist()

            else:
                raise ValueError(f"Unknown window_aggregation: {cfg.window_aggregation}")

            if not valid:
                rejected_windows += 1
                continue

            # ----------- TARGET (O) -----------
            target_series = city_series[cfg.target][
                day_O : day_O + pd.Timedelta("1D")
            ]

            target_day_max = target_series.max()
            target_day_mean = target_series.mean()

            if np.isnan(target_day_max):
                rejected_targets += 1
                rejected_windows += 1
                debug_log(f"  REJECT target {day_O.date()} | NaN")
                continue

            # ----------- TASK SELECTION -----------
            if cfg.target_mode == "regression":
                target_value = target_day_mean

            elif cfg.target_mode == "binary":
                target_value = int(target_day_max >= cfg.target_threshold)

            else:
                raise ValueError(f"Unknown target_mode: {cfg.target_mode}")

            X_rows.append(features)
            Y_rows.append(target_value)

    # -------------------- FINAL ARRAYS --------------------
    X = np.asarray(X_rows, dtype=float)

    if cfg.target_mode == "binary":
        Y = np.asarray(Y_rows, dtype=int).reshape(-1, 1)
    else:
        Y = np.asarray(Y_rows, dtype=float).reshape(-1, 1)

    # -------------------- SUMMARY --------------------
    log("\n=== BUILD DATASET SUMMARY ===")
    log(f"Window aggregation mode: {cfg.window_aggregation}")
    log(f"Total windows: {total_windows}")
    log(f"Accepted samples: {len(X_rows)}")
    log(f"Rejected windows: {rejected_windows}")
    log(f"Rejected days: {rejected_days}")
    log(f"Rejected targets: {rejected_targets}")
    log(f"X shape: {X.shape}")
    log(f"Y shape: {Y.shape}")
    log("=== BUILD DATASET END ===")

    return X, Y
