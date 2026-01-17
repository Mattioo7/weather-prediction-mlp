import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct


def load_dataset(path: str, verbose: bool = False):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=float)

    unique_vals = np.unique(y)

    # --- Heurystyka wykrywania typu zadania ---
    if len(unique_vals) <= 10 and np.all(np.equal(np.mod(unique_vals, 1), 0)):
        y = y.astype(int)
        if np.min(y) == 1:
            y = y - 1  # przesunięcie etykiet do [0..n_classes-1]
        elif set(unique_vals) == {-1, 1}:
            y = ((y + 1) // 2).astype(int)

        if verbose:
            print(f"[load_dataset] Detected CLASSIFICATION: unique labels {np.unique(y)}")
    else:
        y_mean, y_std = np.mean(y), np.std(y)
        y = (y - y_mean) / (y_std if y_std > 0 else 1)

        if verbose:
            print(f"[load_dataset] Detected REGRESSION: normalized target (mean={y_mean:.3f}, std={y_std:.3f})")

    # --- Ujednolicenie kształtu ---
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return X, y


def train_test_split(X, y, test_ratio=0.3, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_size = int(len(X) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy(y_true, y_pred):
    # Spłaszczenie tablic do 1D (eliminuje błędy broadcastingu)
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Wyrównanie typów (gdy np. są floaty zamiast intów)
    if y_true.dtype != y_pred.dtype:
        y_pred = y_pred.astype(y_true.dtype)

    # Oblicz dokładność
    return float(np.mean(y_true == y_pred))


def mse(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def plot_loss(losses, title="Training loss"):
    plt.plot(losses, label="Loss")
    plt.title(title)
    plt.xlabel("Epoka")
    plt.ylabel("Wartość")
    plt.grid(True)
    plt.legend("Funkcja straty")
    plt.show()


def plot_predictions(y_true, y_pred, title="Predictions vs True"):
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("True values")
    plt.ylabel("Predictions")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel("Cecha 1")
    plt.ylabel("Cecha 2")
    plt.grid(True)
    plt.show()


def plot_weight_evolution(weight_history, title="Weight Evolution"):
    weight_history_np = np.array(weight_history)
    num_layers = weight_history_np.shape[1]

    for i in range(num_layers):
        plt.plot(weight_history_np[:, i], label=f'Layer {i+1} Weights')

    plt.title(title)
    plt.xlabel("Epoka")
    plt.ylabel("L2 Norm of Weights")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_dataset(X, y, title="Dataset visualization"):
    if X.shape[1] == 1:
        plt.scatter(X, y, c='blue', s=20, alpha=0.7)
        plt.xlabel("X")
        plt.ylabel("y")
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.RdYlBu, edgecolor="k", s=25)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_accuracy(acc_history, title="Accuracy over epochs"):
    plt.plot(acc_history, label="Accuracy", color="green")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_weights_together(weight_history, title="Weight Norms Evolution (All Layers)"):
    weight_history_np = np.array(weight_history)
    plt.plot(weight_history_np)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("L2 Norm")
    plt.grid(True)
    plt.show()


def load_mnist_images(path: str) -> np.ndarray:
    """
    Wczytuje obrazy MNIST z pliku .idx3-ubyte.
    Zwraca tablicę (N, 784) znormalizowaną do [0, 1].
    """
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols).astype(np.float32) / 255.0
    return images

def load_mnist_labels(path: str) -> np.ndarray:
    """
    Wczytuje etykiety MNIST z pliku .idx1-ubyte.
    Zwraca tablicę (N, 1).
    """
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 1)
    return labels

def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.ravel(y_true).astype(int)
    y_score = np.ravel(y_score)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    comparisons = pos[:, None] > neg[None, :]
    ties = pos[:, None] == neg[None, :]

    return (comparisons.sum() + 0.5 * ties.sum()) / (len(pos) * len(neg))

def confusion_matrix_binary(y_true, y_pred):
    y_true = np.ravel(y_true).astype(int)
    y_pred = np.ravel(y_pred).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    return tp, fp, fn, tn

def precision(y_true, y_pred, eps=1e-8):
    tp, fp, _, _ = confusion_matrix_binary(y_true, y_pred)
    return tp / (tp + fp + eps)

def recall(y_true, y_pred, eps=1e-8):
    tp, _, fn, _ = confusion_matrix_binary(y_true, y_pred)
    return tp / (tp + fn + eps)

def f1_score(y_true, y_pred, eps=1e-8):
    p = precision(y_true, y_pred, eps)
    r = recall(y_true, y_pred, eps)
    return 2 * p * r / (p + r + eps)

def mae(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2_score(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1.0 - ss_res / (ss_tot + 1e-12)

def accuracy_within_tolerance(y_true, y_pred, tol: float):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    return float(np.mean(np.abs(y_pred - y_true) <= tol))

def threshold_predictions(y_score, threshold=0.5):
    return (np.ravel(y_score) >= threshold).astype(int).reshape(-1, 1)

def classification_report_binary(y_true, y_score, threshold=0.5):
    y_pred = threshold_predictions(y_score, threshold)

    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": auc(y_true, y_score),
    }

def regression_report(y_true, y_pred):
    return {
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

def roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.ravel(y_true).astype(int)
    y_score = np.ravel(y_score)

    # Sort by descending score
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    if P == 0 or N == 0:
        raise ValueError("ROC is undefined when only one class is present.")

    tpr = []
    fpr = []
    thresholds = []

    tp = fp = 0
    prev_score = None

    for yt, score in zip(y_true, y_score):
        if prev_score is None or score != prev_score:
            tpr.append(tp / P)
            fpr.append(fp / N)
            thresholds.append(score)
            prev_score = score

        if yt == 1:
            tp += 1
        else:
            fp += 1

    # Final point (1,1)
    tpr.append(1.0)
    fpr.append(1.0)
    thresholds.append(0.0)

    return np.array(fpr), np.array(tpr), np.array(thresholds)

def plot_roc_auc(y_true, y_score, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = auc(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
