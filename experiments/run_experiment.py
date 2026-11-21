"""
Unified experiment runner.

Edit src/config/base_config.py to control:
    ACTIVE_GROUPS      — which function/architecture groups to run
    ACTIVE_ALGORITHMS  — which training algorithms to use
    ACTIVE_VARIATIONS  — which activation variants (cont/disc/pwlu)
    LOSS_FUNCTIONS     — mse | mae
    BASE_ACTIVATIONS   — True=ReLU, False=ELU
    OUTPUT_DIR / EXPERIMENT_NAME — where results are saved
"""

import itertools
from copy import copy
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from datetime import datetime

from src.config import base_config as config
from src.config.base_config import get_variations, get_active_groups
from src.data.generators import prepare_experiment_data
from src.experiment.dispatcher import dispatch
from src.experiment.logger import ExperimentLogger
from src.visualization.plotter import (
    plot_loss_history,
    plot_prediction,
    plot_image_reconstruction,
    plot_regions,
    plot_bab_iterations,
    plot_bab_loss_history,
    plot_combined_loss_histories,
    tile_images,
)


def run(enable_plotting: bool = True) -> None:
    # ── Experiment name ──────────────────────────────────────────────────────
    exp_name = config.EXPERIMENT_NAME or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = ExperimentLogger(config.OUTPUT_DIR, exp_name)
    print(f"Experiment: {exp_name}")
    print(f"Output:     {logger.root}")

    # ── Seed ─────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    # ── Outer loops ──────────────────────────────────────────────────────────
    active_groups = get_active_groups()
    combos = list(itertools.product(
        active_groups,
        config.LOSS_FUNCTIONS,
        config.BASE_ACTIVATIONS,
    ))

    def _group_items(group: dict) -> list:
        """Returns (data_key, display_name) pairs for each item in a group."""
        if "image_paths" in group:
            return [
                (p, os.path.splitext(os.path.basename(p))[0])
                for p in group["image_paths"]
            ]
        return [(f, f) for f in group["functions"]]

    total = sum(
        len(_group_items(g)) * len(g["hidden_architectures"]) * len(config.ACTIVE_ALGORITHMS)
        * len(get_variations(g["discontinuous_indices"]))
        for g in active_groups
    ) * len(config.LOSS_FUNCTIONS) * len(config.BASE_ACTIVATIONS)
    run_idx = 0

    for group, loss_fn, base_act in combos:
        group_name = group["name"]

        for data_key, func_name in _group_items(group):
            print(f"\n[{group_name}] Function: {func_name}")
            X_train, y_train, scaler, image_dataset = prepare_experiment_data(data_key, group)

            for hidden_arch in group["hidden_architectures"]:
                full_arch = [group["input_dim"]] + hidden_arch
                variations = get_variations(group["discontinuous_indices"])

                for algo in config.ACTIVE_ALGORITHMS:
                    loss_histories: dict = {}  # variation → loss_history list
                    plot_paths: dict = {}      # variation → {descriptor: path}

                    for var_cfg in variations:
                        run_idx += 1
                        tag = f"{algo}/{var_cfg['prefix']}/{func_name}/{hidden_arch}"
                        print(f"  [{run_idx}/{total}] {tag}")

                        result = dispatch(
                            algorithm=algo,
                            variation_cfg=var_cfg,
                            full_arch=full_arch,
                            X_train=X_train,
                            y_train=y_train,
                            func_name=func_name,
                            loss_function=loss_fn,
                            base_activation=base_act,
                        )

                        print(f"    Final loss: {result.final_loss:.6f}  "
                              f"({result.runtime_seconds:.1f}s)")

                        logger.log_result(result, group_name=group_name)

                        if enable_plotting:
                            paths = _save_plots(result, logger, X_train, y_train, scaler, group, image_dataset)
                        else:
                            paths = {}
                        loss_histories[var_cfg["prefix"]] = result.loss_history
                        plot_paths[var_cfg["prefix"]] = paths

                    if enable_plotting and len(variations) > 1:
                        _save_grouped_plots(loss_histories, plot_paths, logger, result, image_dataset)

    print(f"\nDone. Results in: {logger.root}")


def _save_plots(result, logger, X_train, y_train, scaler, group, image_dataset) -> dict:
    x_min, x_max = group["x_min"], group["x_max"]
    model_type = "gadnn" if result.algorithm != "backprop" else "dnn"
    is_disc = result.variation != "cont"
    paths = {}
    pre_model = result.trained_model
    if result.pre_delta is not None:
        pre_model = copy(pre_model)
        pre_model.delta = result.pre_delta

    # Loss history — stays in main folder alongside the combined chart
    if result.loss_history:
        p = logger.get_plot_path(result, "loss_histories", "loss_history")
        plot_loss_history(result.loss_history, p)
        paths["loss_history"] = p

    # BaB: specialised loss history → _individual subfolder
    if result.algorithm == "bab" and result.metadata.get("history_log"):
        p = logger.get_plot_path(result, "loss_histories", "bab_loss_history", individual=True)
        plot_bab_loss_history(
            history_log=result.metadata["history_log"],
            generations_per_iter=result.metadata.get("generations_per_iter", 1),
            config_name=f"{result.variation}_{result.function}",
            save_path=p,
        )
        paths["bab_loss_history"] = p

    # Pre-training prediction → _individual subfolder
    if result.pre_weights is not None:
        p = logger.get_plot_path(result, "predictions", "predictions_pre", individual=True)
        plot_prediction(
            pre_model, result.pre_weights,
            X_train, y_train, p,
            scaler=scaler, x_min=x_min, x_max=x_max,
        )
        paths["predictions_pre"] = p

    # Post-training prediction → _individual subfolder
    weights_post = None if result.algorithm == "backprop" else result.best_weights
    p = logger.get_plot_path(result, "predictions", "predictions_post", individual=True)
    plot_prediction(
        result.trained_model, weights_post,
        X_train, y_train, p,
        scaler=scaler, x_min=x_min, x_max=x_max,
    )
    paths["predictions_post"] = p

    # Pre-training regions → _individual subfolder
    if is_disc and result.pre_weights is not None:
        p = logger.get_plot_path(result, "regions", "regions_pre", individual=True)
        plot_regions(
            pre_model, result.pre_weights,
            X_train, y_train, p,
            x_min=x_min, x_max=x_max,
            model_type=model_type,
            scaler=scaler,
        )
        paths["regions_pre"] = p

    # Post-training regions → _individual subfolder
    if is_disc:
        weights_for_regions = None if result.algorithm == "backprop" else result.best_weights
        p = logger.get_plot_path(result, "regions", "regions_post", individual=True)
        plot_regions(
            result.trained_model, weights_for_regions,
            X_train, y_train, p,
            x_min=x_min, x_max=x_max,
            model_type=model_type,
            scaler=scaler,
        )
        paths["regions_post"] = p

    # Image reconstruction → _individual subfolder
    if image_dataset is not None:
        weights_recon = None if result.algorithm == "backprop" else result.best_weights
        p = logger.get_plot_path(result, "reconstructions", "image_reconstruction", individual=True)
        plot_image_reconstruction(
            result.trained_model, weights_recon, image_dataset, p,
            scaler=scaler,
        )
        paths["image_reconstruction"] = p

    # BaB: per-iteration snapshots → _individual subfolder (handled in get_bab_iter_path)
    if result.algorithm == "bab" and result.metadata.get("history_log"):
        plot_bab_iterations(
            model=result.trained_model,
            history_log=result.metadata["history_log"],
            X_train=X_train,
            y_train=y_train,
            get_iter_path=lambda i: logger.get_bab_iter_path(result, i),
            generations_per_iter=result.metadata.get("generations_per_iter", 1),
            scaler=scaler,
            x_min=x_min,
            x_max=x_max,
        )

    return paths


def _save_grouped_plots(loss_histories, plot_paths, logger, result, image_dataset):
    variations = list(loss_histories.keys())
    disc_variations = [v for v in variations if v != "cont"]

    # Combined loss history overlay
    plot_combined_loss_histories(
        loss_histories,
        logger.get_grouped_plot_path(result, "loss_histories", "combined_loss_history"),
    )

    # Predictions pre+post — N_var rows × 2 cols
    pred_paths = []
    for var in variations:
        pred_paths.append(plot_paths[var].get("predictions_pre"))
        pred_paths.append(plot_paths[var].get("predictions_post"))
    tile_images(
        paths=pred_paths,
        save_path=logger.get_grouped_plot_path(result, "predictions", "combined_predictions_prepost"),
        nrows=len(variations),
        ncols=2,
        row_labels=variations,
        col_labels=["Pre-training", "Post-training"],
    )

    # Regions pre+post — disc/pwlu rows × 2 cols
    if disc_variations:
        region_paths = []
        for var in disc_variations:
            region_paths.append(plot_paths[var].get("regions_pre"))
            region_paths.append(plot_paths[var].get("regions_post"))
        tile_images(
            paths=region_paths,
            save_path=logger.get_grouped_plot_path(result, "regions", "combined_regions_prepost"),
            nrows=len(disc_variations),
            ncols=2,
            row_labels=disc_variations,
            col_labels=["Pre-training", "Post-training"],
        )

    # Reconstructions — N_var rows × 1 col (image groups only)
    if image_dataset is not None:
        recon_paths = [plot_paths[var].get("image_reconstruction") for var in variations]
        tile_images(
            paths=recon_paths,
            save_path=logger.get_grouped_plot_path(result, "reconstructions", "combined_reconstructions"),
            nrows=len(variations),
            ncols=1,
            row_labels=variations,
        )


if __name__ == "__main__":
    run()
