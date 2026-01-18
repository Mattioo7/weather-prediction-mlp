from dataclasses import dataclass
from typing import Literal


# =========================
# WEATHER
# =========================

Normalization = Literal["global", "per_city", "none"]
Split = Literal["train", "test"]
Target = Literal["temperature", "wind_speed"]
TargetMode = Literal["regression", "binary"]

@dataclass
class WeatherFixedParams:
    # --- zadanie ---
    target: Target
    target_mode: TargetMode
    target_threshold: float | None = None

    # --- dane ---
    data_dir: str = "../data"
    split: Split = "train"

    # --- okno czasowe ---
    skip_day: bool = True
    hours_per_day: int = 24

    # --- normalizacja ---
    normalization: Normalization = "global" # deprecated, use WeatherGridParams.normalization

    # --- kodowanie ---
    encode_wind_direction: bool = True
    include_city_coords: bool = False

    # --- braki danych ---
    max_missing_ratio_per_day: float = 0.3


WindowAggregation = Literal["flatten", "aggregate"]
Aggregation = Literal["mean", "min", "max", "trend"]
NormalizationType = Literal[
    "standardize",   # (X - mean) / std
    "minmax",        # [0, 1] (lub inny zakres)
    "l2",            # normalizacja wektorów
    "none",
]

@dataclass
class WeatherGridParams:
    window_aggregation: WindowAggregation | list[WindowAggregation] = "flatten"
    input_variables: tuple[str, ...] | list[tuple[str, ...]] = ()
    aggregations: dict[str, tuple[Aggregation, ...]] | list[dict[str, tuple[Aggregation, ...]]] | None = None
    cities: tuple[str, ...] | list[tuple[str, ...]] | None = None
    normalization: NormalizationType | list[NormalizationType] = "standardize"

    window_size: int | list[int] = 3


# =========================
# MLP
# =========================

Task = Literal["regression", "binary", "multiclass"]
Optimizer = Literal["sgd", "momentum", "adam"]

@dataclass
class MLPFixedParams:
    task: Task
    beta: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    adaptive_lr: bool = True
    lr_decay: float = 0.99


Activations = Literal["sigmoid", "identity", "relu", "leaky_relu", "gelu"]
Losses = Literal["mse", "mae", "huber", "binary_cross_entropy", "categorical_cross_entropy"]

@dataclass
class MLPGridParams:
    hidden_layers: tuple[int, ...] | list[tuple[int, ...]]
    loss: Losses | list[Losses]
    activation: Activations | list[Activations] = "gelu"

    learning_rate: float | list[float] = 0.01
    seed: int | list[int] = 42
    use_bias: bool | list[bool] = True
    optimizer: Optimizer | list[Optimizer] = "momentum"

# =========================
# FIT
# =========================

@dataclass
class FitFixedParams:
    verbose: bool = False
    log_every: int | None = None
    use_tqdm: bool = True
    one_hot_if_needed: bool = True
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 0.001

@dataclass
class FitGridParams:
    epochs: int | list[int] = 400
    batch_size: int | Literal["auto"] | None | list[int | Literal["auto"] | None] = "auto"

    shuffle: bool | list[bool] = False
    val_split: float | list[float] = 0.1


# =========================
# EXPERIMENT
# =========================

@dataclass
class Experiment:
    name: str

    weather_fixed: WeatherFixedParams
    weather_grid: WeatherGridParams

    mlp_fixed: MLPFixedParams
    mlp_grid: MLPGridParams

    fit_fixed: FitFixedParams
    fit_grid: FitGridParams
