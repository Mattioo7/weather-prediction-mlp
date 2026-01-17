import numpy as np

def standardize_global(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_std = (X - mean) / std
    return X_std, mean, std

def minmax_scale_global(X, feature_range=(0.0, 1.0)):
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)

    scale = max_val - min_val
    scale[scale == 0] = 1.0

    X_scaled = (X - min_val) / scale
    low, high = feature_range
    X_scaled = X_scaled * (high - low) + low

    return X_scaled, min_val, max_val

def normalize_l2_global(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    X_norm = X / norms
    return X_norm, norms
