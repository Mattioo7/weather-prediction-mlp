import numpy as np

def encode_wind_direction_deg(deg: float) -> tuple[float, float]:
    rad = np.deg2rad(deg)
    return np.sin(rad), np.cos(rad)
