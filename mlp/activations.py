from typing import Union
import numpy as np

ArrayLike = Union[float, np.ndarray]


def sigmoid(x: ArrayLike) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x: ArrayLike) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def identity(x: ArrayLike) -> np.ndarray:
    return np.asarray(x)


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x_shift)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def gelu(x: ArrayLike) -> np.ndarray:
    x = np.clip(x, -10.0, 10.0)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def gelu_derivative(x: ArrayLike) -> np.ndarray:
    x = np.clip(x, -10.0, 10.0)

    sqrt_2_pi = np.sqrt(2 / np.pi)
    x_cubed_term = 0.044715 * np.power(x, 3)
    tanh_arg = sqrt_2_pi * (x + x_cubed_term)
    tanh_val = np.tanh(tanh_arg)

    term1 = 0.5 * (1 + tanh_val)
    term2 = 0.5 * x * (1 - np.square(tanh_val))
    term3 = sqrt_2_pi * (1 + 3 * 0.044715 * np.square(x))

    derivative = term1 + term2 * term3
    return derivative


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "identity": identity,
    "softmax": softmax,
    "gelu": gelu,
}

ACTIVATIONS_DERIVATIVES = {
    "sigmoid": d_sigmoid,
    "identity": lambda x: 1,
    "gelu": gelu_derivative,
}


if __name__ == "__main__":
    x = np.array([-2.0, 0.0, 2.0])
    y = sigmoid(x)
    dy = d_sigmoid(x)

    print("Input:", x)
    print("Sigmoid:", y)
    print("Expected sigmoid:", [0.11920292, 0.5, 0.88079708])
    print("Difference:", y - np.array([0.11920292, 0.5, 0.88079708]))

    print("\nDerivative:", dy)
    print("Expected dSigmoid:", [0.10499359, 0.25, 0.10499359])
    print("Difference:", dy - np.array([0.10499359, 0.25, 0.10499359]))

    X = np.array([[1.0, 2.0, 3.0]])
    print("\nSoftmax row sums (should be 1):", softmax(X).sum(axis=1))