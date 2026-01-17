from dataclasses import dataclass
from typing import Literal, Sequence


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
    window_size: int = 3
    skip_day: bool = True
    hours_per_day: int = 24

    # --- normalizacja ---
    normalization: Normalization = "global" # TODO: ???

    # --- kodowanie ---
    encode_wind_direction: bool = True
    include_city_coords: bool = False

    # --- braki danych ---
    max_missing_ratio_per_day: float = 0.3


WindowAggregation = Literal["flatten", "aggregate"]
Aggregation = Literal["mean", "min", "max"]

@dataclass
class WeatherGridParams:
    window_aggregation: WindowAggregation = "flatten"
    input_variables: Sequence[str] = ()
    aggregations: dict[str, Sequence[Aggregation]] | None = None
    cities: Sequence[str] | None = None


# =========================
# MLP
# =========================

Task = Literal["regression", "binary", "multiclass"]
Optimizer = Literal["sgd", "momentum", "adam"]

@dataclass
class MLPFixedParams:
    task: Task
    learning_rate: float = 0.01
    seed: int = 42
    use_bias: bool = True
    optimizer: Optimizer = "momentum"
    beta: float = 0.9  # for momentum and adam
    beta2: float = 0.999  # for adam
    eps: float = 1e-8  # for adam
    adaptive_lr: bool = False # TODO: true?
    lr_decay: float = 0.99


Activations = Literal["sigmoid", "identity", "softmax", "gelu"] # TODO: softmax?
Losses = Literal["mse", "mae", "huber", "binary_cross_entropy", "categorical_cross_entropy"]

@dataclass
class MLPGridParams:
    hidden_layers: Sequence[int]
    loss: Losses
    activation: Activations = "gelu"


# =========================
# FIT
# =========================

@dataclass
class FitFixedParams:
    shuffle: bool = False
    verbose: bool = False
    log_every: int | None = None
    use_tqdm: bool = True
    one_hot_if_needed: bool = True
    early_stopping: bool = True
    val_split: float = 0.1
    patience: int = 20
    min_delta: float = 0.001

@dataclass
class FitGridParams:
    epochs: int = 400
    batch_size: int | Literal["auto"] | None = "auto"


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
