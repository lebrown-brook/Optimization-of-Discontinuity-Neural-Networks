"""
Unified visualization module.

Public API used by the experiment runner:
    plot_loss_history(loss_history, save_path)
    plot_prediction(model, weights, X_train, y_train, save_path, scaler, x_min, x_max)
    plot_image_reconstruction(model, weights, image_dataset, save_path, scaler)
    plot_regions(model, weights, X_train, y_train, save_path, x_min, x_max, model_type, top_eps)
    plot_bab_iterations(model, history_log, X_train, y_train, get_iter_path, generations_per_iter)
    plot_bab_loss_history(history_log, generations_per_iter, config_name, save_path)

Grouped comparison plots (across variations):
    plot_combined_loss_histories(histories, save_path)
    tile_images(paths, save_path, nrows, ncols, row_labels, col_labels)

Legacy functions for post-hoc analysis (read a saved CSV):
    plot_grouped_bar_charts(df) — bar charts from backprop experiment CSV
    plot_heatmaps(df)           — heatmaps from backprop experiment CSV
"""

import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Loss history
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_history(loss_history: list, save_path: str) -> None:
    """Single loss curve (any algorithm)."""
    if not loss_history:
        return
    _ensure_dir(save_path)
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Prediction surface
# ──────────────────────────────────────────────────────────────────────────────

def plot_prediction(
    model,
    weights,          # np.ndarray for GADNN, None for trained DNN
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    save_path: str,
    scaler=None,
    x_min: float = -2.0,
    x_max: float = 2.0,
) -> None:
    """
    Plots model predictions vs. training data.
    Works for both DNN (weights=None, uses model directly) and GADNN
    (weights=np.ndarray, passed to model.forward).
    Uses original (unscaled) domain for axis when scaler is provided.
    """
    _ensure_dir(save_path)
    input_dim = X_train.shape[1]
    X_np = X_train.detach().cpu().numpy()
    Y_np = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train

    # Generate grid in original domain
    margin = (x_max - x_min) * 0.1
    x_lo, x_hi = x_min - margin, x_max + margin

    with torch.no_grad():
        if input_dim == 1:
            grid_orig = np.linspace(x_lo, x_hi, 1000).reshape(-1, 1)
            grid_input = torch.tensor(
                scaler.transform(grid_orig) if scaler else grid_orig, dtype=torch.float32
            )
            y_pred = _predict(model, weights, grid_input).numpy()

            plt.figure(figsize=(10, 6))
            x_plot = grid_orig
            x_scatter = scaler.inverse_transform(X_np) if scaler else X_np
            plt.scatter(x_scatter, Y_np, alpha=0.5, color="blue", s=10, label="Training Data")
            plt.plot(x_plot, y_pred, color="red", linewidth=2, label="Prediction")
            plt.xlabel("X")
            plt.ylabel("Y")

        else:  # 2D
            resolution = 100
            x1 = np.linspace(x_lo, x_hi, resolution)
            x2 = np.linspace(x_lo, x_hi, resolution)
            X1, X2 = np.meshgrid(x1, x2)
            grid_orig = np.column_stack([X1.ravel(), X2.ravel()])
            grid_input = torch.tensor(
                scaler.transform(grid_orig) if scaler else grid_orig, dtype=torch.float32
            )
            y_pred = _predict(model, weights, grid_input).numpy().reshape(resolution, resolution)
            x_scatter = scaler.inverse_transform(X_np) if scaler else X_np

            plt.figure(figsize=(10, 8))
            plt.contourf(X1, X2, y_pred, cmap="viridis", levels=50)
            plt.colorbar(label="Prediction")
            plt.scatter(
                x_scatter[:, 0], x_scatter[:, 1],
                c=Y_np.ravel(), cmap="viridis",
                edgecolors="white", linewidth=0.5, s=20, label="Training Data",
            )
            plt.xlabel("X1")
            plt.ylabel("X2")

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Image reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def plot_image_reconstruction(model, weights, image_dataset, save_path: str, scaler=None) -> None:
    """
    Side-by-side of the ground-truth luminance image and the model's reconstruction.
    Useful for diagnosing whether poor output comes from the network, the luminance
    target, or the plotting itself.
    """
    coords, H, W = image_dataset.full_grid()
    if scaler is not None:
        coords_input = torch.tensor(scaler.transform(coords.numpy()), dtype=torch.float32)
    else:
        coords_input = coords
    with torch.no_grad():
        y_pred = _predict(model, weights, coords_input).numpy().reshape(H, W)

    ground_truth = image_dataset.full_luminance()  # (H, W), same coefficients as y_train

    _ensure_dir(save_path)
    panel_w = max(4, W / 100)
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, max(3, H / 100)))
    axes[0].imshow(ground_truth, cmap="gray", origin="upper", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth (luminance)")
    axes[0].axis("off")
    axes[1].imshow(y_pred, cmap="gray", origin="upper", vmin=0, vmax=1)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Discontinuity regions
# ──────────────────────────────────────────────────────────────────────────────

def plot_regions(
    model,
    weights,          # np.ndarray for GADNN; np.ndarray of flat params for DNN
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    save_path: str,
    x_min: float = -2.0,
    x_max: float = 2.0,
    model_type: str = "gadnn",  # "gadnn" | "dnn"
    top_eps: int = 20,
    scaler=None,
) -> None:
    """
    Unified region plotter. Dispatches to GADNN or DNN region analysis.
    For GADNN: uses model.forward with return_regions=True.
    For DNN: uses TorchDiscontinuousNeuralNetworkAnalyzer.
    """
    _ensure_dir(save_path)
    if model_type == "gadnn":
        _plot_regions_gadnn(model, weights, X_train, y_train, save_path, x_min, x_max, scaler)
    else:
        _plot_regions_dnn(model, weights, X_train, y_train, save_path, x_min, x_max, top_eps, scaler)


def _plot_regions_gadnn(model, weights, X_train, y_train, save_path, x_min, x_max, scaler): #//TODO: Add epsilon top-k filtering to focus on most significant regions or integrate into DNN analyzer for consistent region importance ranking across model types or create unified region analyzer that abstracts over model internals and provides a common interface for both GADNN and DNN, allowing for consistent region extraction and plotting logic regardless of the underlying model architecture.
    input_dim = X_train.shape[1]
    margin = (x_max - x_min) * 0.1
    x_lo, x_hi = x_min - margin, x_max + margin
    resolution = 500

    Y_np = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    X_np = X_train.detach().cpu().numpy()

    if input_dim == 1:
        grid_orig = np.linspace(x_lo, x_hi, resolution).reshape(-1, 1)
        grid_input = torch.tensor(
            scaler.transform(grid_orig) if scaler else grid_orig, dtype=torch.float32
        )
        y_pred, regions = model.forward(weights, grid_input, return_regions=True)
        region_ids = _hash_regions(regions)
        x_plot = grid_orig
        x_scatter = scaler.inverse_transform(X_np) if scaler else X_np

        plt.figure(figsize=(10, 6))
        y_vals = y_pred.detach().numpy()
        y_lo, y_hi = y_vals.min(), y_vals.max()
        rng = (y_hi - y_lo) or 1.0
        plt.imshow(
            region_ids.reshape(1, -1),
            extent=[x_lo, x_hi, y_lo - 0.1 * rng, y_hi + 0.1 * rng],
            aspect="auto", cmap="tab20", alpha=0.3,
        )
        plt.plot(x_plot, y_vals, color="black", linewidth=2, label="Prediction")
        plt.scatter(x_scatter, Y_np, color="red", s=10, label="Data")
        plt.legend()
        plt.title("Discontinuity Regions")

    else:  # 2D
        resolution = 200
        x1 = np.linspace(x_lo, x_hi, resolution)
        x2 = np.linspace(x_lo, x_hi, resolution)
        X1, X2 = np.meshgrid(x1, x2)
        grid_orig = np.column_stack([X1.ravel(), X2.ravel()])
        grid_input = torch.tensor(
            scaler.transform(grid_orig) if scaler else grid_orig, dtype=torch.float32
        )
        y_pred, regions = model.forward(weights, grid_input, return_regions=True)
        region_ids = _hash_regions(regions).reshape(resolution, resolution)
        x_scatter = scaler.inverse_transform(X_np) if scaler else X_np

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(
            region_ids, origin="lower",
            extent=[x_lo, x_hi, x_lo, x_hi], cmap="tab20",
        )
        axes[0].scatter(x_scatter[:, 0], x_scatter[:, 1], c="white", s=8, alpha=0.5)
        axes[0].set_title("Discontinuity Regions")
        axes[0].set_xlabel("X1"); axes[0].set_ylabel("X2")

        y_surf = y_pred.detach().numpy().reshape(resolution, resolution)
        c = axes[1].contourf(X1, X2, y_surf, cmap="viridis", levels=50)
        fig.colorbar(c, ax=axes[1], label="Prediction")
        axes[1].scatter(
            x_scatter[:, 0], x_scatter[:, 1],
            c=Y_np.ravel(), cmap="viridis", edgecolors="white", s=10,
        )
        axes[1].set_title("Prediction Surface")
        axes[1].set_xlabel("X1"); axes[1].set_ylabel("X2")

        plt.tight_layout()

    plt.savefig(save_path)
    plt.close()


def _plot_regions_dnn(model, weights, X_train, y_train, save_path, x_min, x_max, top_eps, scaler):
    from src.analysis.torch_analyzer import analyze_and_plot_regions

    # Load weights into a copy of the model if pre-training weights supplied
    if weights is not None:
        model = copy.deepcopy(model)
        torch.nn.utils.vector_to_parameters(
            torch.tensor(weights, dtype=torch.float32),
            model.parameters(),
        )

    analyze_and_plot_regions(
        model,
        X_data=X_train,
        Y_samples=y_train,
        save_path=save_path,
        show_plot=False,
        top_eps=top_eps,
        x_min=x_min,
        x_max=x_max,
        scaler=scaler,
    )


# ──────────────────────────────────────────────────────────────────────────────
# BaB-specific: per-iteration snapshots
# ──────────────────────────────────────────────────────────────────────────────

def plot_bab_iterations(
    model,
    history_log: list,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    get_iter_path,       # callable(iteration: int) -> str
    generations_per_iter: int,
    scaler=None,
    x_min: float = -2.0,
    x_max: float = 2.0,
) -> None:
    """Plot the top-3 branch solutions from each BaB iteration."""
    for i, iter_data in enumerate(history_log):
        top_solutions = sorted(
            iter_data.get("branches", []),
            key=lambda b: b["fitness"],
            reverse=True,
        )[:3]
        if not top_solutions:
            continue
        snapshot_model = model
        if "delta" in iter_data:
            snapshot_model = copy.copy(model)
            snapshot_model.delta = iter_data["delta"]

        save_path = get_iter_path(i)
        _ensure_dir(save_path)

        input_dim = X_train.shape[1]
        X_np = X_train.detach().cpu().numpy()
        Y_np = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train

        margin = (x_max - x_min) * 0.1
        x_lo, x_hi = x_min - margin, x_max + margin

        fig, axes = plt.subplots(len(top_solutions), 1, figsize=(10, 5 * len(top_solutions)), constrained_layout=True)
        if len(top_solutions) == 1:
            axes = [axes]

        with torch.no_grad():
            if input_dim == 1:
                grid_orig = np.linspace(x_lo, x_hi, 1000).reshape(-1, 1)
                grid_t = torch.tensor(scaler.transform(grid_orig) if scaler else grid_orig, dtype=torch.float32)
                x_scatter = scaler.inverse_transform(X_np) if scaler else X_np

                for j, sol_data in enumerate(top_solutions):
                    y_pred = snapshot_model.forward(sol_data["solution"], grid_t).cpu().numpy()
                    axes[j].scatter(x_scatter, Y_np, alpha=0.5, color="blue", s=10)
                    axes[j].plot(grid_orig, y_pred, color="red", linewidth=2)
                    axes[j].set_title(f"Rank {j+1} — Fitness: {sol_data['fitness']:.5f}")
            else:
                resolution = 100
                x1 = np.linspace(x_lo, x_hi, resolution)
                x2 = np.linspace(x_lo, x_hi, resolution)
                X1, X2 = np.meshgrid(x1, x2)
                grid_orig = np.column_stack([X1.ravel(), X2.ravel()])
                grid_t = torch.tensor(scaler.transform(grid_orig) if scaler else grid_orig, dtype=torch.float32)
                x_scatter = scaler.inverse_transform(X_np) if scaler else X_np

                for j, sol_data in enumerate(top_solutions):
                    y_pred = snapshot_model.forward(sol_data["solution"], grid_t).cpu().numpy().reshape(resolution, resolution)
                    c = axes[j].contourf(X1, X2, y_pred, cmap="viridis", levels=50)
                    fig.colorbar(c, ax=axes[j], label="Prediction")
                    axes[j].set_title(f"Rank {j+1} — Fitness: {sol_data['fitness']:.5f}")

        current_gen = (i + 1) * generations_per_iter
        fig.suptitle(f"BaB Iteration {i} (Gen {current_gen}) — Top {len(top_solutions)} Branches")
        plt.savefig(save_path)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# BaB-specific: loss history with branch lineage colour coding
# ──────────────────────────────────────────────────────────────────────────────

def plot_bab_loss_history(
    history_log: list,
    generations_per_iter: int,
    config_name: str,
    save_path: str,
) -> None:
    """
    Plots loss history for BaB top-3 branches per iteration.
    Branches that descend from a previous top-3 parent inherit its colour;
    forks get lighter/darker variants; fresh lineages get new palette colours.
    """
    if not history_log:
        return
    _ensure_dir(save_path)

    plt.figure(figsize=(12, 8))
    palette_rgb = [mcolors.to_rgb(c) for c in sns.color_palette("tab10")]
    next_color_idx = 0
    prev_winner_colors = {}

    for i, iter_data in enumerate(history_log):
        branches = iter_data.get("branches", [])
        sorted_branches = sorted(branches, key=lambda x: x["fitness"], reverse=True)
        top3 = sorted_branches[:3]

        branches_by_parent = {}
        for b in top3:
            pid = b.get("parent_id")
            branches_by_parent.setdefault(pid, []).append(b)

        current_winner_colors = {}

        for pid, siblings in branches_by_parent.items():
            if pid in prev_winner_colors:
                base_color = prev_winner_colors[pid]
                for idx, branch in enumerate(siblings):
                    if idx == 0:
                        branch_color = base_color
                    elif idx == 1:
                        h, s, v = mcolors.rgb_to_hsv(base_color)
                        branch_color = mcolors.hsv_to_rgb((h, max(0.2, s * 0.6), min(1.0, v * 1.3)))
                    else:
                        h, s, v = mcolors.rgb_to_hsv(base_color)
                        branch_color = mcolors.hsv_to_rgb((h, min(1.0, s * 1.2), max(0.3, v * 0.7)))
                    current_winner_colors[branch["id"]] = branch_color
            else:
                for branch in siblings:
                    current_winner_colors[branch["id"]] = palette_rgb[next_color_idx % len(palette_rgb)]
                    next_color_idx += 1

        for branch in top3:
            uid = branch["id"]
            color = current_winner_colors[uid]
            inherited = branch.get("parent_id") in prev_winner_colors
            local_hist = np.array(branch.get("local_history", []))
            if len(local_hist) == 0:
                continue
            start_gen = i * generations_per_iter
            gens = np.arange(start_gen, start_gen + len(local_hist))
            plt.plot(gens, local_hist, color=color, alpha=0.9, linewidth=2.5)
            marker = "o" if inherited else "x"
            plt.scatter([start_gen], [local_hist[0]], color=color, marker=marker, s=40)

        prev_winner_colors = current_winner_colors.copy()

    plt.yscale("log")
    plt.xlabel("Global Generation")
    plt.ylabel("Loss (log scale)")
    plt.title(f"BaB Loss History — Top 3 Branches — {config_name}")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Grouped / combined plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_combined_loss_histories(histories: dict, save_path: str) -> None:
    """Overlay one loss curve per variation on a single log-scale axes."""
    if not histories:
        return
    _ensure_dir(save_path)
    plt.figure(figsize=(10, 6))
    for label, hist in histories.items():
        if hist:
            plt.plot(hist, label=label)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss History — All Variations")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def tile_images(
    paths: list,
    save_path: str,
    nrows: int,
    ncols: int,
    row_labels: list = None,
    col_labels: list = None,
) -> None:
    """
    Load PNGs from disk and arrange them in an nrows × ncols grid.
    None entries or missing files produce a blank cell with "N/A" text.
    """
    _ensure_dir(save_path)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, path in enumerate(paths):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if path is not None and os.path.exists(path):
            ax.imshow(plt.imread(path))
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx in range(len(paths), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    if col_labels:
        for c, label in enumerate(col_labels[:ncols]):
            axes[0][c].set_title(label, fontsize=12, fontweight="bold")

    if row_labels:
        for r, label in enumerate(row_labels[:nrows]):
            axes[r][0].text(
                -0.08, 0.5, label,
                transform=axes[r][0].transAxes,
                ha="right", va="center", fontsize=11, fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy post-hoc analysis helpers (read saved CSV)
# ──────────────────────────────────────────────────────────────────────────────

def plot_grouped_bar_charts(df, output_dir: str = "result_plots") -> None:
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    value_vars = [col for col in df.columns if col.startswith("final_loss_")]
    df_melted = pd.melt(
        df, id_vars=["function", "architecture", "learning_rate", "lambda_reg"],
        value_vars=value_vars, var_name="model", value_name="loss",
    )
    df_melted["model"] = df_melted["model"].str.replace("final_loss_", "")
    for func in df["function"].unique():
        for arch_tuple in df["architecture"].unique():
            g = sns.catplot(
                data=df_melted[(df_melted["function"] == func) & (df_melted["architecture"] == arch_tuple)],
                x="learning_rate", y="loss", hue="model",
                col="lambda_reg", kind="bar", height=6, aspect=1.2, legend_out=True,
            )
            arch_str = "_".join(map(str, arch_tuple))
            g.fig.suptitle(f'Model Comparison on "{func}" — Arch {arch_str}', y=1.03)
            g.set_axis_labels("Learning Rate", "Final Loss (log)")
            g.set(yscale="log")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(output_dir, f"bar_chart_{func}_{arch_str}.png"))
            plt.close(g.fig)


def plot_heatmaps(df, output_dir: str = "result_plots") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for model_loss in [c for c in df.columns if c.startswith("final_loss_")]:
        for func in df["function"].unique():
            for arch in df["architecture"].unique():
                subset = df[(df["function"] == func) & (df["architecture"] == arch)]
                pivot = subset.pivot_table(values=model_loss, index="learning_rate", columns="lambda_reg")
                arch_str = "_".join(map(str, arch))
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
                plt.title(f"Heatmap {model_loss} — {func} — Arch {arch_str}")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"heatmap_{func}_{arch_str}_{model_loss}.png"))
                plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _predict(model, weights, grid_input: torch.Tensor) -> torch.Tensor:
    """Forward pass dispatching on model type.

    GADNN is stateless: weights are passed directly to forward.
    DNN (torch.nn.Module) is stateful: weights, if supplied, are loaded into
    a copy before calling forward normally.
    """
    if isinstance(model, torch.nn.Module):
        if weights is not None:
            model = copy.deepcopy(model)
            torch.nn.utils.vector_to_parameters(
                torch.tensor(weights, dtype=torch.float32),
                model.parameters(),
            )
        model.eval()
        return model(grid_input)
    return model.forward(weights, grid_input)


def _hash_regions(regions: np.ndarray) -> np.ndarray:
    """Map each unique region vector to a unique integer id."""
    if regions.shape[1] == 0:
        return np.zeros(regions.shape[0], dtype=int)
    void = np.ascontiguousarray(regions).view(
        np.dtype((np.void, regions.dtype.itemsize * regions.shape[1]))
    )
    _, idx = np.unique(void, return_inverse=True)
    return idx


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compat aliases (used by BaBTrainer and retired experiment runners)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_and_plot_gadnn_regions(model, solution, X_data, Y_samples=None, save_path=None, x_min=None, x_max=None):
    """Backward-compat alias → plot_regions(..., model_type='gadnn')."""
    _plot_regions_gadnn(
        model, solution, X_data, Y_samples,
        save_path or "regions_unnamed.png",
        x_min if x_min is not None else -2.0,
        x_max if x_max is not None else 2.0,
        scaler=None,
    )


def plot_bab_iteration_best(model, top_3_solutions, X_train, y_train, iteration, gen_per_iter, config_name, save_dir="PyGAD_BAB_Plots"):
    """Backward-compat alias → plot_bab_iterations."""
    history_log = [{"iteration": iteration, "branches": top_3_solutions}]

    def get_path(i):
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"{config_name}_iter_{iteration}.png")

    plot_bab_iterations(
        model=model,
        history_log=history_log,
        X_train=X_train,
        y_train=y_train,
        get_iter_path=get_path,
        generations_per_iter=gen_per_iter,
    )


def plot_ga_result(model, solution, X_train, y_train, generation, config_name, save_dir="PyGAD_plots"):
    """Backward-compat alias → plot_prediction with legacy save path."""
    _ensure_dir(save_dir)
    save_path = os.path.join(save_dir, f"ga_result_{config_name}_gen_{generation}.png")
    plot_prediction(model, solution, X_train, y_train, save_path)


def plot_combined_loss_history(histories: dict, config_base_name: str, save_dir: str = "loss_history") -> None:
    """Backward-compat alias. Plots multiple named loss curves on one figure."""
    if not histories:
        return
    _ensure_dir(os.path.join(save_dir, "_"))
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.plot(hist, label=name)
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss History — {config_base_name}")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"compare_{config_base_name}.png"))
    plt.close()
