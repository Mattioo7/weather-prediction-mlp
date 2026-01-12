import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)

def d_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_true.size

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_pred - y_true))

def d_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sign(y_pred - y_true) / y_true.size

def huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    error = y_pred - y_true
    abs_error = np.abs(error)

    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic

    return np.mean(0.5 * quadratic**2 + delta * linear)

def d_huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
    error = y_pred - y_true
    grad = np.where(
        np.abs(error) <= delta,
        error,
        delta * np.sign(error),
    )
    return grad / y_true.size

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def d_binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_pred - y_true) / y_true.size

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_pred = np.clip(y_pred, eps, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def d_categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_pred - y_true) / y_true.shape[0]

LOSSES = {
    "mse": (mse, d_mse),
    "mae": (mae, d_mae),
    "huber": (huber, d_huber),
    "binary_cross_entropy": (binary_cross_entropy, d_binary_cross_entropy),
    "categorical_cross_entropy": (categorical_cross_entropy, d_categorical_cross_entropy),
}

if __name__ == "__main__":
    import numpy as np

    # Test A: 1D
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([0.8, 0.2, 0.9])

    loss = mse(y_true, y_pred)
    grad = d_mse(y_true, y_pred)

    print("=== Test A (1D) ===")
    print("MSE:", loss, "Expected: 0.03")
    print("Gradient:", grad)
    print("Expected grad:", [-0.13333333, 0.13333333, -0.06666667])

    # Test B: 2D
    y_true = np.array([[1.0, 2.0],
                       [3.0, 4.0]])
    y_pred = np.array([[1.1, 1.9],
                       [2.8, 4.2]])

    loss = mse(y_true, y_pred)
    grad = d_mse(y_true, y_pred)

    print("\n=== Test B (2D) ===")
    print("MSE:", loss, "Expected: 0.025")
    print("Gradient:\n", grad)
    print("Expected grad:\n", [[0.05, -0.05], [-0.10, 0.10]])
