from typing import Callable, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.data import test_functions as tfp


# ──────────────────────────────────────────────────────────────────────────────
# 1-D benchmark functions (built-in, for basic_1d group)
# ──────────────────────────────────────────────────────────────────────────────

def step_function(x: np.ndarray) -> np.ndarray:
    return np.where(x > -1, 3.0, 0.0)

def piecewise_linear_function(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    y[x < -5] = -2.0 * x[x < -5] + 1.0
    y[x >= -5] = 3.0
    y[x > 5] = 0.5 * x[x > 5] - 2.0
    return y

def multiple_steps_function(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    y[x > -5] = 2.0
    y[x > 0] = -1.0
    y[x > 5] = 3.0
    return y

def sine_with_discontinuity(x: np.ndarray) -> np.ndarray:
    y = np.sin(x)
    y[x > 2] += 2
    return y

def linear_with_varying_slopes(x: np.ndarray) -> np.ndarray:
    return np.where(x < -5, -0.5 * x,
           np.where(x < 0,   2 * x,
           np.where(x < 5,  -x + 5, 0.5 * x)))


# ──────────────────────────────────────────────────────────────────────────────
# Core data generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_data(
    num_samples: int = 100,
    function: str = "step_function",
    input_dim: int = 1,
    x_min: float = -10.0,
    x_max: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random inputs and evaluate the named function.

    Returns (X, y) as float32 tensors with shapes (N, input_dim) and (N, 1).
    """
    x = np.random.uniform(low=x_min, high=x_max, size=(num_samples, input_dim))

    if hasattr(tfp, function):
        y = getattr(tfp, function)(x)
    elif function in globals():
        y = globals()[function](x)
    else:
        raise ValueError(f"Unknown function: '{function}'")

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def load_image_data(
    image_path: str,
    num_samples: int,
):
    """Load image pixels as (coord, grayscale) pairs via SingleImageDataset.

    Returns (X, y, dataset) where X is (N, 2) coords in [-1, 1], y is (N, 1)
    luminance, and dataset is the SingleImageDataset (for full-grid reconstruction).
    """
    from src.data.image_dataset import SingleImageDataset
    dataset = SingleImageDataset(image_path, num_samples=num_samples)
    X = dataset.coords                      # (N, 2), already in [-1, 1]
    colors = dataset.colors                 # (N, 3), in [0, 1]
    y = (0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]).unsqueeze(1)
    return X, y, dataset


def prepare_experiment_data(func_name: str, group_cfg: dict):
    """
    Generate, scale, and return training data for one experiment group entry.

    Returns:
        X_train       — scaled input tensor (N, input_dim)
        y_train       — target tensor (N, 1)
        scaler        — fitted StandardScaler (for inverse-transforming plots)
        image_dataset — SingleImageDataset if this is an image group, else None
    """
    if "image_paths" in group_cfg:
        X_raw, y_train, image_dataset = load_image_data(func_name, group_cfg["num_samples"])
    else:
        X_raw, y_train = generate_data(
            num_samples=group_cfg["num_samples"],
            function=func_name,
            input_dim=group_cfg["input_dim"],
            x_min=group_cfg["x_min"],
            x_max=group_cfg["x_max"],
        )
        image_dataset = None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.numpy())
    X_train = torch.tensor(X_scaled, dtype=torch.float32)
    return X_train, y_train, scaler, image_dataset


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def create_model_with_shared_init(
    model_cls: Callable, *args, shared_seed: int = 42, **kwargs
) -> torch.nn.Module:
    """Seed all RNGs then instantiate model so initial weights are reproducible."""
    torch.manual_seed(shared_seed)
    np.random.seed(shared_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(shared_seed)
    return model_cls(*args, **kwargs)


def print_dnn_parameters(dnn: torch.nn.Module, showvalue: bool = False) -> None:
    print(f"\n{'='*20} Model: {dnn.name} {'='*20}")
    for name, module in dnn.named_modules():
        params = list(module.named_parameters(recurse=False))
        if params:
            print(f"\n[Layer: {name}] Type: {type(module).__name__}")
            for p_name, p in params:
                print(f"  - {p_name} (shape: {list(p.shape)})")
                if showvalue:
                    print(f"    {p.data}")
