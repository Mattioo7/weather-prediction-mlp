from __future__ import annotations
from collections.abc import Callable
from typing import Literal

import numpy as np
from tqdm import trange
from tqdm import tqdm
import time

from .activations import sigmoid, d_sigmoid, identity, softmax, gelu, gelu_derivative
from .losses import LOSSES
from .utils import plot_loss, plot_predictions, plot_decision_boundary, plot_weight_evolution

TaskType = Literal["regression", "binary", "multiclass"]


class MLP:
    layer_sizes: list[int]
    n_layers: int
    learning_rate: float
    task: TaskType
    use_bias: bool

    # hidden layers
    activation: Callable[[np.ndarray], np.ndarray]
    d_activation: Callable[[np.ndarray], np.ndarray]

    # output layer
    out_activation: Callable[[np.ndarray], np.ndarray]
    d_out_activation: Callable[[np.ndarray], np.ndarray]

    # loss (on output after activation)
    loss_fn: Callable[[np.ndarray, np.ndarray], float]
    d_loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]

    W: list[np.ndarray]
    b: list[np.ndarray]

    Z: list[np.ndarray] | None  # pre-activation values
    A: list[np.ndarray] | None  # post-activation values

    loss_history: list[float]
    weight_history: list[list[float]]
    accuracy_history: list[float]

    def __init__(
            self,
            layer_sizes: list[int],
            task: TaskType,
            activation: str,
            learning_rate: float = 1e-2,
            seed: int | None = None,
            use_bias: bool = True,
            loss: str | None = None,
            momentum: bool = False,
            beta: float = 0.9,
            adaptive_lr: bool = False,
            lr_decay: float = 0.99
    ):
        assert len(layer_sizes) >= 2, "Provide at least [n_in, n_out]"
        assert task in ("regression", "binary", "multiclass"), \
            f"Invalid task: {task}. Allowed: 'regression', 'binary', 'multiclass'"
        assert activation in ("sigmoid", "gelu", "identity"), \
            f"Invalid activation function: {activation}. Allowed: 'sigmoid', 'gelu', 'identity'"

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.learning_rate = float(learning_rate)
        self.task = task

        self.loss_history = []
        self.weight_history = []
        self.accuracy_history = []

        if seed is not None:
            np.random.seed(seed)

        self.use_bias = use_bias

        # --- Initialize lists and buffers ---
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.Z: list[np.ndarray] | None = None
        self.A: list[np.ndarray] | None = None

        # --- Choose activation function for hidden layers ---
        if activation == "sigmoid":
            self.activation = sigmoid
            self.d_activation = d_sigmoid
        elif activation == "gelu":
            self.activation = gelu
            self.d_activation = gelu_derivative
        elif activation == "identity":
            self.activation = identity
            self.d_activation = lambda x: 1
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        # --- Choose output activation and loss function ---
        if task == "regression":
            self.out_activation = identity
            self.d_out_activation = identity
        elif task == "binary":
            self.out_activation = sigmoid
            self.d_out_activation = d_sigmoid
        elif task == "multiclass":
            self.out_activation = softmax
            self.d_out_activation = identity
        else:
            raise ValueError(f"Unknown task: {task}")

        # --- Choose loss function ---
        if loss is not None:
            assert loss in LOSSES, f"Unknown loss: {loss}. Available: {list(LOSSES.keys())}"
            self.loss_fn, self.d_loss_fn = LOSSES[loss]
        else:
            if task == "regression":
                self.loss_fn, self.d_loss_fn = LOSSES["mse"]
            elif task == "binary":
                self.loss_fn, self.d_loss_fn = LOSSES["binary_cross_entropy"]
            elif task == "multiclass":
                self.loss_fn, self.d_loss_fn = LOSSES["categorical_cross_entropy"]

        # --- Initialize weights ---
        for l in range(self.n_layers):
            n_in, n_out = layer_sizes[l], layer_sizes[l + 1]
            if activation in ("sigmoid", "identity"):
                scale = np.sqrt(1.0 / n_in)
            elif activation == "gelu":
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = 0.01
            W_l = np.random.randn(n_in, n_out) * scale

            b_l = np.zeros((1, n_out), dtype=float)
            self.W.append(W_l)
            self.b.append(b_l)

        self.momentum = momentum
        self.beta = beta
        self.adaptive_lr = adaptive_lr
        self.lr_decay = lr_decay

        # Initialize velocity buffers for momentum
        self.vW = [np.zeros_like(W) for W in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2 and X.shape[1] == self.layer_sizes[0], (
            f"X has shape {X.shape}, expected (m, {self.layer_sizes[0]})"
        )

        Z: list[np.ndarray] = [np.empty((0, 0))]  # placeholder for input layer; # pre-activation values
        A: list[np.ndarray] = [X]  # activations, starting with input; # post-activation values

        # hidden layers
        for layer in range(self.n_layers - 1):
            z = A[layer] @ self.W[layer] + (self.b[layer] if self.use_bias else 0)
            a = self.activation(z)
            Z.append(z)
            A.append(a)

        # output layer
        z_out = A[self.n_layers - 1] @ self.W[self.n_layers - 1] + (self.b[self.n_layers - 1] if self.use_bias else 0)
        a_out = self.out_activation(z_out)

        Z.append(z_out)
        A.append(a_out)

        self.Z, self.A = Z, A
        return a_out

    def backward(self, Y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        assert self.Z is not None and self.A is not None, "Call forward(X) first"
        Z, A, L = self.Z, self.A, self.n_layers

        # initialize gradients
        dW = [np.zeros_like(Wl) for Wl in self.W]
        db = [np.zeros_like(bl) for bl in self.b]

        Y_pred = A[L]  # post-activation output
        # if self.task == "binary":
        #     delta_next = (self.A[-1] - Y)
        # else:
        #     delta_next = self.d_loss_fn(Y, Y_pred)
        delta_next = self.d_loss_fn(Y, Y_pred) * 1.0

        # output layer gradients
        dW[L - 1] = A[L - 1].T @ delta_next
        db[L - 1] = np.sum(delta_next, axis=0, keepdims=True) if self.use_bias else np.zeros_like(db[L - 1])

        # hidden layers
        for layer in range(L - 2, -1, -1):
            delta_l = (delta_next @ self.W[layer + 1].T) * self.d_activation(Z[layer + 1])
            dW[layer] = A[layer].T @ delta_l
            db[layer] = np.sum(delta_l, axis=0, keepdims=True) if self.use_bias else np.zeros_like(db[layer])
            delta_next = delta_l

        return dW, db

    def step(self, learning_rate: float | None, grads: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
        eta = float(learning_rate) if learning_rate is not None else self.learning_rate
        dW, db = grads

        for l in range(self.n_layers):
            if self.momentum:
                self.vW[l] = self.beta * self.vW[l] + (1 - self.beta) * dW[l]
                self.vb[l] = self.beta * self.vb[l] + (1 - self.beta) * db[l]
                self.W[l] -= eta * self.vW[l]
                self.b[l] -= eta * self.vb[l]
            else:
                self.W[l] -= eta * dW[l]
                self.b[l] -= eta * db[l]

    def compute_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        Y_pred = self.forward(X)
        return float(self.loss_fn(Y, Y_pred))

    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        Y_pred = self.forward(X)
        if self.task == "regression":
            return Y_pred
        if self.task == "binary":
            if return_proba:
                return Y_pred
            return (Y_pred >= 0.5).astype(int)
        # multiclass
        if return_proba:
            return Y_pred
        return np.argmax(Y_pred, axis=1).reshape(-1, 1)

    @staticmethod
    def _to_one_hot_encoding(y: np.ndarray, n_classes: int) -> np.ndarray:
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        one_hot = np.zeros((y.shape[0], n_classes), dtype=float)
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1.0
        return one_hot

    def fit(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            learning_rate: float | None = None,
            epochs: int = 1000,
            batch_size: int | Literal["auto"] | None = "auto",
            shuffle: bool = True,
            verbose: bool = False,
            log_every: int | None = None,
            use_tqdm: bool = True,
            one_hot_if_needed: bool = True,
    ) -> tuple[list[float], list[list[float]], list[float]]:

        start_time = time.perf_counter()
        print(">>> Version 5 (mini-batch)...")

        if self.task == "regression":
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
        elif self.task == "binary":
            Y = Y.reshape(-1, 1)
        elif self.task == "multiclass":
            if one_hot_if_needed and (Y.ndim == 1 or (Y.ndim == 2 and Y.shape[1] == 1)):
                Y = self._to_one_hot_encoding(Y, self.layer_sizes[-1])

        history: list[float] = []
        accuracy_history: list[float] = []
        weight_history: list[list[float]] = []

        # --- Resolve effective batch size ---
        n_samples = X.shape[0]

        if batch_size == "auto":
            eff_batch_size = min(200, n_samples)
        elif batch_size is None:
            eff_batch_size = n_samples  # full batch (if 1 then SGD)
        else:
            eff_batch_size = int(batch_size)

        if log_every is None:
            log_every = max(1, epochs // 20)

        # --- Training loop ---
        epoch_iter = trange(epochs, desc="Training") if use_tqdm else range(epochs)
        for ep in epoch_iter:

            indices = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(indices)

            epoch_loss = 0.0

            for start in range(0, n_samples, eff_batch_size):
                end = start + eff_batch_size
                batch_idx = indices[start:end]

                X_batch = X[batch_idx]
                Y_batch = Y[batch_idx]

                Y_pred_batch = self.forward(X_batch)
                grads = self.backward(Y_batch)
                self.step(learning_rate, grads)

                epoch_loss += self.loss_fn(Y_batch, Y_pred_batch) * len(batch_idx)

            loss = epoch_loss / n_samples
            history.append(float(loss))

            # --- Accuracy (tylko dla klasyfikacji) ---
            if self.task in ["binary", "multiclass"]:
                Y_pred = self.forward(X)
                if self.task == "multiclass":
                    y_true = np.argmax(Y, axis=1)
                    y_pred = np.argmax(Y_pred, axis=1)
                else:  # binary
                    y_true = Y.ravel().astype(int) 
                    y_pred = (Y_pred >= 0.5).astype(int).ravel()
                acc = np.mean(y_true == y_pred)
                accuracy_history.append(acc)
            else:
                accuracy_history.append(np.nan)

            current_weight_norms = [float(np.linalg.norm(Wl)) for Wl in self.W]
            weight_history.append(current_weight_norms)

            cur_lr = float(learning_rate) if learning_rate is not None else self.learning_rate

            if use_tqdm:
                epoch_iter.set_postfix(
                    loss=f"{loss:.4f}",
                    acc=f"{accuracy_history[-1]:.4f}" if self.task != "regression" else "n/a",
                    lr=f"{cur_lr:.6g}",
                )

            if verbose and ((ep + 1) % log_every == 0 or (ep + 1) == epochs):
                if self.task != "regression":
                    msg = (
                        f"[epoch {ep + 1:5d}/{epochs}] "
                        f"loss={loss:.6f} acc={accuracy_history[-1]:.4f} lr={cur_lr:.6g}"
                    )
                else:
                    msg = (
                        f"[epoch {ep + 1:5d}/{epochs}] "
                        f"loss={loss:.6f} lr={cur_lr:.6g}"
                    )

                if use_tqdm:
                    tqdm.write(msg)
                else:
                    print(msg)

            if self.adaptive_lr:
                self.learning_rate *= self.lr_decay

        self.loss_history = history
        self.weight_history = weight_history
        self.accuracy_history = accuracy_history

        elapsed = time.perf_counter() - start_time
        print(f"Training finished in {elapsed:.2f} seconds")

        return history, weight_history, accuracy_history


# ------------------------- EXAMPLES -------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    rng = np.random.default_rng(1)

    # # === Regression ===
    # net_r = MLP(layer_sizes=[2, 16, 1], task="regression", activation="sigmoid", learning_rate=0.01, seed=0)
    # Xr = rng.normal(size=(512, 2))
    # Yr = (2 * Xr[:, :1] - 3 * Xr[:, 1:2]) + 0.05 * rng.normal(size=(512, 1))
    # print("=== Regression test ===")
    # print("Reg loss start:", net_r.compute_loss(Xr, Yr))
    # hist_r, weight_hist_r, _ = net_r.fit(Xr, Yr, epochs=2000, verbose=True)
    # print("Reg loss end:  ", hist_r[-1])
    # plot_loss(hist_r, title="Regression Training Loss")
    # Xr_preds = net_r.predict(Xr)
    # # plot_predictions(Yr, Xr_preds, title="Regression Predictions vs True")
    # # plot_weight_evolution(weight_hist_r, title="Regression Weight Evolution")
    # print("\n")

    # === Binary classification ===
    net_b = MLP(layer_sizes=[2, 8, 1], task="binary", activation="sigmoid", learning_rate=0.05, seed=0)
    Xb = rng.normal(size=(400, 2))
    yb = ((Xb[:, 0] * Xb[:, 1]) > 0).astype(int).reshape(-1, 1)  # XOR-like: sign of product

    print("=== Binary classification test ===")
    print("Bin loss start:", net_b.compute_loss(Xb, yb))
    hist_b, weight_hist_b, _ = net_b.fit(Xb, yb, epochs=100000, verbose=True)
    print("Bin loss end:  ", hist_b[-1])

    preds_b = net_b.predict(Xb)
    acc_b = np.mean(preds_b == yb)
    print(f"Binary classification accuracy: {acc_b:.4f}")
    # plot_loss(hist_b, title="Binary Classification Training Loss")
    # plot_decision_boundary(net_b, Xb, yb, title="Binary Classification Decision Boundary")
    # plot_weight_evolution(weight_hist_b, title="Binary Classification Weight Evolution")
