from dataclasses import dataclass
from typing import Literal, Sequence

Aggregation = Literal["mean", "min", "max"]
Normalization = Literal["global", "per_city", "none"]
WindowAggregation = Literal["flatten", "aggregate"]
Split = Literal["train", "test"]
Target = Literal["temperature", "wind_speed"]

@dataclass
class WeatherConfig:
    data_dir: str
    split: Split

    # --- zadanie ---
    target: Target

    # --- okno czasowe ---
    window_size: int = 3          # liczba dni I
    skip_day: bool = True         # pomijanie X

    window_aggregation: WindowAggregation = "flatten"
    hours_per_day: int = 24

    # --- dane ---
    cities: Sequence[str] | None = None
    input_variables: Sequence[str] = (
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
    )

    # --- agregacja ---
    aggregations: dict[str, Sequence[Aggregation]] = None

    # --- kodowanie ---
    encode_wind_direction: bool = True
    include_city_coords: bool = False

    # --- normalizacja ---
    normalization: Normalization = "global"

    # --- braki danych ---
    max_missing_ratio_per_day: float = 0.3
