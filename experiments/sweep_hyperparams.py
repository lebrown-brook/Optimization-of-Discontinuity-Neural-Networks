"""
Hyperparameter sweep utility.

Set SWEEP_ALGORITHM and PARAM_GRID below, then run:
    python experiments/sweep_hyperparams.py

Results are saved as a pivot CSV under experiments/results/sweeps/.
To sweep ES instead, set SWEEP_ALGORITHM = "es" and provide ES-compatible keys in PARAM_GRID.
"""

import itertools
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from datetime import datetime

from src.config import base_config as config
from src.config.base_config import get_variations, get_active_groups
from src.data.generators import prepare_experiment_data
from src.experiment.dispatcher import dispatch


# ──────────────────────────────────────────────────────────────────────────────
# Sweep configuration — edit this section
# ──────────────────────────────────────────────────────────────────────────────

SWEEP_ALGORITHM = "ga"   # "ga" | "es" | "bab" | "backprop"

PARAM_GRID = {
    # GA sweep example (keys must match GA_SETTINGS / ES_SETTINGS keys)
    "num_parents_mating":   [10, 15],
    "keep_elitism":         [1, 2, 3, 4],
    "crossover_type":       ["scattered", "single_point"],
    "mutation_probability": [[0.10, 0.02], [0.05, 0.01]],
}

# Which group, function, and hidden arch to sweep over
# (keep small — sweep runs |PARAM_GRID| × |funcs| × |archs| × |variations| experiments)
SWEEP_GROUP    = "paper_2d"
SWEEP_FUNCTION = "g_l"
SWEEP_ARCH     = [16, 8, 4]    # hidden layers only


# ──────────────────────────────────────────────────────────────────────────────

def run_sweep() -> None:
    # Build cartesian product of all param values
    keys = list(PARAM_GRID.keys())
    grid = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    print(f"Total configurations to sweep: {len(grid)}")

    # Find the group config
    group = next((g for g in config.EXPERIMENT_GROUPS if g["name"] == SWEEP_GROUP), None)
    if group is None:
        raise ValueError(f"Group '{SWEEP_GROUP}' not found in EXPERIMENT_GROUPS.")

    full_arch = [group["input_dim"]] + SWEEP_ARCH
    X_train, y_train, scaler = prepare_experiment_data(SWEEP_FUNCTION, group)
    variations = get_variations(group["discontinuous_indices"])
    loss_fn = config.LOSS_FUNCTIONS[0]
    base_act = config.BASE_ACTIVATIONS[0]

    rows = []

    for cfg_idx, values in enumerate(grid):
        override = dict(zip(keys, values))
        cfg_str = "_".join(f"{k}={v}" for k, v in override.items())
        print(f"[{cfg_idx+1}/{len(grid)}] {cfg_str}")

        for var_cfg in variations:
            result = dispatch(
                algorithm=SWEEP_ALGORITHM,
                variation_cfg=var_cfg,
                full_arch=full_arch,
                X_train=X_train,
                y_train=y_train,
                func_name=SWEEP_FUNCTION,
                loss_function=loss_fn,
                base_activation=base_act,
                settings_override=override,
            )
            row = {**override, "variation": var_cfg["prefix"], "final_loss": result.final_loss}
            rows.append(row)
            print(f"  {var_cfg['prefix']}: loss={result.final_loss:.6f}")

    if not rows:
        print("No results collected.")
        return

    df = pd.DataFrame(rows)

    # Pivot: index = hyperparams, columns = variation, values = final_loss
    # mutation_probability values may be lists — stringify so they're hashable
    if "mutation_probability" in df.columns:
        df["mutation_probability"] = df["mutation_probability"].apply(str)
    pivot = df.pivot_table(
        index=keys,
        columns="variation",
        values="final_loss",
        aggfunc="first",
    )
    pivot["avg_loss"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg_loss")

    print("\n=== Sweep Summary (sorted by avg loss) ===")
    print(pivot.to_string())

    out_dir = os.path.join(config.OUTPUT_DIR, "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{SWEEP_ALGORITHM}_sweep_{ts}.csv")
    pivot.to_csv(out_path)
    print(f"\nSweep summary saved to: {out_path}")


if __name__ == "__main__":
    run_sweep()
