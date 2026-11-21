from pathlib import Path

import torch

SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Reference lists (use these when populating EXPERIMENT_GROUPS below)
# ──────────────────────────────────────────────────────────────────────────────
PAPER_FUNCTIONS = ["g_l", "g_s", "g_eta", "g_gamma"]
BASIC_TEST_FUNCTIONS = [
    "multiple_steps_function",
    "linear_with_varying_slopes",
    "piecewise_linear_function",
    "sine_with_discontinuity",
]


# ──────────────────────────────────────────────────────────────────────────────
# Experiment groups
# Each group pairs a set of functions with matching architectures and domain.
# hidden_architectures: specify HIDDEN layers only — input/output are inferred.
# discontinuous_indices: connection indices in the FULL architecture
#   (full = [input_dim] + hidden + [1], zero-indexed).
#   E.g., for [2, 16, 8, 4, 1] the connections are 0..3; index 2 = 8→4 layer.
# ──────────────────────────────────────────────────────────────────────────────
EXPERIMENT_GROUPS = [
    {
        "name": "paper_2d",
        "functions": ["g_l"],           # subset of PAPER_FUNCTIONS to run
        "input_dim": 2,
        "x_min": -2.0,
        "x_max": 2.0,
        "num_samples": 200,
        "hidden_architectures": [
            [16, 8, 4],
            # [32, 16, 8],
        ],
        "discontinuous_indices": [1, 2],  # connection 1 = 16→8, connection 2 = 8→4
    },
    {
        "name": "basic_1d",
        "functions": ["step_function"],   # subset of BASIC_TEST_FUNCTIONS to run
        "input_dim": 1,
        "x_min": -10.0,
        "x_max": 10.0,
        "num_samples": 200,
        "hidden_architectures": [
            [8, 4],
            # [16, 8],
        ],
        "discontinuous_indices": [0],     # connection 0 = first layer
    },
    {
        "name": "edge_image",
        "image_paths": [                  # set to actual image paths before running
            #r"C:\Users\brook\Downloads\PythonProject\PythonProject\EdgeDetectionData\images\train\8049.jpg",
            r"C:\Users\brook\Downloads\PythonProject\PythonProject\EdgeDetectionData\images\train\310007.jpg"
        ],
        "input_dim": 2,
        "x_min": -1.0,
        "x_max": 1.0,
        "num_samples": 1000,
        "hidden_architectures": [
            [128,128,64,7,8],
            # [256,128, 64,31],
            # [512, 256, 256, 127],
        ],
        "discontinuous_indices": [2],  # connection 0 = 2d input to first layer

    }
]

# ──────────────────────────────────────────────────────────────────────────────
# Experiment controls
# ──────────────────────────────────────────────────────────────────────────────
ACTIVE_GROUPS      = ["paper_2d"] #, "basic_1d", "edge_image"]
ACTIVE_ALGORITHMS  = ["backprop", "ga", "es", "bab"]
ACTIVE_VARIATIONS  = ["cont", "disc", "pwlu"]
LOSS_FUNCTIONS     = ["mse"]
BASE_ACTIVATIONS   = [True]                     # True = ReLU, False = ELU

DELTA_DECAY_FACTOR = 0.99   # per-generation δ decay for PWLU

# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = str(Path(__file__).resolve().parents[2] / "experiments" / "results")
EXPERIMENT_NAME  = 'logchecker'

# ──────────────────────────────────────────────────────────────────────────────
# Algorithm settings
# ──────────────────────────────────────────────────────────────────────────────

USE_INVERSION = True    # fitness = 1/(loss+ε) if True, else fitness = -loss
FLIP_ALPHAS   = [0.01, 0.1]

BACKPROP_SETTINGS = {
    "num_epochs":    100,
    "batch_size":    64,
    "learning_rates":  [0.01],
    "lambda_regs":     [0.0],
    "top_eps":         None,
    "seeds":           [SEED],
}

GA_SETTINGS = {
    # --- General Settings ---
    "num_generations":        100,
    "sol_per_pop":            50,   # population size

    # --- Mating Pool ---
    "num_parents_mating":     10,
    "keep_elitism":           2,
    "keep_parents":           0,
    "parent_selection_type":  "sss",

    # --- Initialization ---
    "init_range_low":         -1,
    "init_range_high":        1,

    # --- Crossover ---
    "crossover_type":         "single_point",

    # --- Mutation ---
    "mutation_type":          "adaptive",
    "mutation_probability":   [0.1, 0.02],  # [P_high, P_low]

    "random_seed":            SEED,
}

ES_SETTINGS = {
    # Core evolution loop
    "num_generations":    15,
    "population_size_P":  50,       # number of parents
    "offspring_lambda":   10,        # offspring per parent, aiming to be log
    "use_grad_step":      True,     # zeroth-order gradient step

    # Zeroth-order guidance
    "sigma_low":          0.07,     # minimum mutation std
    "sigma_high":         0.07,     # maximum mutation std
    "alpha":              0.01,     # gradient step size

    # Region memory
    "mu_per_region":      1,        # solutions kept per region
    "max_regions":        1000,     # max regions in archive

    # Noise scheduler
    "sigma_decay_rate":   0.05,     # adaptive noise decay per region visit

    # Initialization
    "init_range_low":     -1.0,
    "init_range_high":    1.0,

    # Reproducibility
    "random_seed":        SEED,
}

BAB_SETTINGS = {
    "num_branches":                15,
    "generations_per_bab_iteration": 2,
    "total_generations":           6,       #TODO: make automatically num_branches *  gen_per_bab_iterations
    "perturbation_scale":          0.01,   # noise magnitude for seed perturbation
    "diversity_threshold":         0.1,    # Hamming distance threshold for branch clustering
    "p_dropout":                   0.2,    # dropout probability for weight zeroing perturbation
    "seed_length":                 10,     # solutions per seed cluster
    "seeds_used_for_perturbation": 4,      # top-K seeds used for new population generation
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_variations(discontinuous_indices: list) -> list:
    """Return variation configs filtered to ACTIVE_VARIATIONS."""
    all_vars = [
        {"prefix": "cont", "activation": "disc",  "indices": []},
        {"prefix": "disc", "activation": "disc",  "indices": discontinuous_indices},
        {"prefix": "pwlu", "activation": "pwlu",  "indices": discontinuous_indices},
    ]
    return [v for v in all_vars if v["prefix"] in ACTIVE_VARIATIONS]


def get_active_groups() -> list:
    return [g for g in EXPERIMENT_GROUPS if g["name"] in ACTIVE_GROUPS]
