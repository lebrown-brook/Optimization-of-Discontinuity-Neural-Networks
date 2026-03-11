import torch

SEED = 42

PAPER_FUNCTIONS = ["g_l", "g_s", "g_eta", "g_gamma"]
BASIC_TEST_FUNCTIONS = ["multiple_steps_function", "linear_with_varying_slopes", "piecewise_linear_function", "sine_with_discontinuity"]
TEST_FUNCTIONS = BASIC_TEST_FUNCTIONS # + PAPER_FUNCTIONS

# Architectures: [input_size, hidden_layer_1, hidden_layer_2, ...]
ARCHITECTURES = [
    [2, 8, 8],
    [2, 16, 8, 4],
    [1, 8, 8],
    [1, 16, 8, 4]
]

LOSS_FUNCTIONS = ["mse"]#, "mae"]
BASE_ACTIVATIONS = [True] # True for ReLU, False for ELU.
USE_INVERSION = True # 1/loss otherwise -loss
DELTA_DECAY_FACTOR = 0.99 # Decay factor for delta parameter per generation
FLIP_ALPHAS = [ 0.01, 0.1]

# --- GA Settings ---
GA_SETTINGS = {
    # --- General Settings ---
    "num_generations": 400,
    "sol_per_pop": 50,

    # --- Mating Pool ---
    "num_parents_mating": 10,
    "keep_elitism": 2,
    "keep_parents": 0,
    "parent_selection_type": "sss",

    # --- Initialization ---
    "init_range_low": -1,
    "init_range_high": 1,

    # --- Crossover ---
    "crossover_type": "single_point",

    # --- Mutation ---
    "mutation_type": "adaptive",
    "mutation_probability": [0.1, 0.02],  #[P_high, P_low]


    "random_seed": SEED
}

# --- BaB Settings ---
BAB_SETTINGS = {
    "num_branches": 5,       # amount of times more total computations than GA. If we have population size of 50, way more branches are not useful
    "generations_per_bab_iteration": 50,
    "total_generations": 500,
    "perturbation_scale": 0.01,
    "diversity_threshold": 0.1,
    "p_dropout": 0.2,
    "seed_length": 10,  #Since population size is big, we can use bigger seed length. Adjust population to use all seeds
    "seeds_used_for_perturbation": 4
}

def get_base_variations(base_discontinuous_indices):
    return [
        #{"prefix": "cont", "activation": "disc", "indices": []},
        {"prefix": "disc", "activation": "disc", "indices": base_discontinuous_indices},
        #{"prefix": "pwlu", "activation": "pwlu", "indices": base_discontinuous_indices},
        #{"prefix": "disc_flip_aware", "activation": "disc", "indices": base_discontinuous_indices, "flip_aware": True},
        #{"prefix": "pwlu_flip_aware", "activation": "pwlu", "indices": base_discontinuous_indices, "flip_aware": True}
    ]
