# Technical Overview — dnn_research

This document is a deep reference for the entire codebase. Read [README.md](README.md) first for orientation and setup.

---

## Table of Contents

1. [Research Background](#1-research-background)
2. [Models](#2-models)
3. [Activation Functions](#3-activation-functions)
4. [Training Algorithms](#4-training-algorithms)
5. [Benchmark Functions](#5-benchmark-functions)
6. [Data Generation](#6-data-generation)
7. [Configuration System](#7-configuration-system)
8. [Experiment Pipeline](#8-experiment-pipeline)
9. [Visualization & Analysis](#9-visualization--analysis)
10. [Testing](#10-testing)
11. [Design Notes & Caveats](#11-design-notes--caveats)

---

## 1. Research Background

Standard neural networks with smooth activations (ReLU, ELU, sigmoid) are universal approximators of *continuous* functions. Many real-world problems involve discontinuous targets — sharp boundaries in physical simulations, image segmentation, piecewise-defined signals.

Della Santa & Pieraccini (JCAM, 2023) propose adding a **discontinuous component** to individual neurons:

```
h(x) = σ(Wx + b) + ε ⊙ H(Wx + b)
```

where `σ` is a standard activation (ReLU or ELU), `H` is the Heaviside step function, and `ε` is a learnable vector of jump magnitudes. This allows the network to represent functions that have finite jumps at the boundaries of activation regions (the set where `Wx + b > 0`).

This repository implements this framework in PyTorch and evaluates three optimization strategies: evolutionary (GA, ES), hybrid (BaB), and gradient-based (backpropagation with STE).

---

## 2. Models

### 2.1 `DNN` / `PWLUDNN` — `src/models/dnn.py`

Standard `torch.nn.Module` networks. Layers are stacked as:

```
Linear → [Activation] → [Optional discontinuous jump]
```

Constructor arguments:
- `input_size`, `hidden_layers: list[int]`, `output_size`
- `discontinuous_indices: list[int]` — which hidden layers get a discontinuous activation
- `Disc_Activation: str` — `"disc"` (Heaviside) or `"pwlu"` (piecewise linear)
- `Relu_true: bool` — `True` for ReLU base, `False` for ELU base

`PWLUDNN` is a thin variant that always uses the piecewise-linear activation.

### 2.2 `GADNN` — `src/models/gadnn.py`

A **non-`nn.Module`** class designed for evolutionary optimization. Instead of carrying `torch.nn.Parameter` tensors, it accepts a **flat numpy/tensor vector `w`** (a "chromosome") at forward time and reconstructs all weights, biases, and ε values from it.

**Flat parameter layout for one layer** (if that layer is discontinuous):
```
[W_0 … W_{out×in−1} | b_0 … b_{out−1} | ε_0 … ε_{out−1}]
```
The output layer never gets ε parameters.

`_precalculate_architecture()` runs once at init and computes pre-sliced index ranges for every layer, eliminating per-call dictionary lookups.

**Key methods:**

| Method | Description |
|---|---|
| `forward(w, x, return_regions=False)` | Performs the forward pass; optionally returns the region matrix |
| `get_epsilon_values(w)` | Extracts ε values for each discontinuous neuron |
| `capture_activation_regions(w, x)` | Returns `(region_matrix, eps_metadata)` |
| `get_epsilon_indices_in_w()` | Global indices of ε entries in `w` |

`num_genes` (int) is the length of the required chromosome, auto-calculated at init.

### 2.3 `BaseModel` — `src/models/base.py`

Abstract class providing the interface: `forward()`, `get_num_parameters()`, `from_vector()`.

---

## 3. Activation Functions

### 3.1 Heaviside — `src/activations/discontinuous.py`

`heaviside_function` is a `torch.autograd.Function`:
- **Forward**: `(x > 0).float()` — exact step
- **Backward**: zero gradient everywhere (subgradient)

`heaviside_function_STE` replaces the zero backward with a **Straight-Through Estimator** using `hardtanh` as a surrogate, allowing gradient-based training through the discontinuity.

`DiscontinuousLayer` (torch.nn.Module variant, used by `DNN`) computes:
```
out = σ(Wx + b) + ε ⊙ H(Wx + b)
```
`DiscontinuousLayer2` uses two separate linear transforms — one for the continuous part and one for the Heaviside argument — providing more expressive discontinuities.

> **Note:** A bug exists in `DiscontinuousLayer.forward()` — the computed jump `y` is calculated but not added to the output (line 81 returns `self.act(x)` instead of `self.act(x) + y`). This is flagged in the source with a comment. The `GADNN` forward path (`src/models/gadnn.py:164`) does correctly apply the jump.

### 3.2 Piecewise-Linear Activation (PWLU) — `src/activations/piecewise_linear.py`

`PiecewiseLinearActivationFunction` is a custom autograd function with three regions based on pre-activation value `z`:

| Region | Condition | Output |
|---|---|---|
| 0 | `z ≤ 0` | ReLU: `0`; ELU: `exp(z) − 1` |
| 1 | `0 < z ≤ δ` | Blends smoothly; slope `= ε/δ + 1` |
| 2 | `z > δ` | Linear: `z + ε` |

As `δ → 0` this converges to a discontinuous step of size `ε`. In BaB training, `δ` is decayed each iteration by `DELTA_DECAY_FACTOR` (default `0.99` per generation).

`PiecewiseLinearActivation` wraps it as an `nn.Module` with learnable `epsilon` and `delta` parameters.

---

## 4. Training Algorithms

### 4.1 Backpropagation — `src/algorithms/backprop/trainer.py`

Standard supervised training using PyTorch's Adam optimizer.

- **LR scheduling**: `ReduceLROnPlateau` (monitors val loss)
- **Early stopping**: patience of 250 epochs
- **L2 regularization**: `lambda_reg` applied to all parameters
- **Delta decay**: every 10 epochs, `delta *= DELTA_DECAY_FACTOR` for PWLU layers
- **Gradient flow through H**: Uses `heaviside_function_STE` when `STE_true=True`

### 4.2 Genetic Algorithm — `src/algorithms/ga/ga_flip_aware.py` + `src/models/gadnn.py`

Uses **PyGAD** (`pygad.GA`) with the `GADNN` chromosome representation.

Default GA configuration (`GA_SETTINGS` in config):

| Parameter | Value | Notes |
|---|---|---|
| `sol_per_pop` | 50 | Population size |
| `num_parents_mating` | 10 | Parents per generation |
| `keep_elitism` | 2 | Elites copied directly |
| `parent_selection_type` | `"sss"` | Steady-state selection |
| `crossover_type` | `"single_point"` | |
| `mutation_type` | `"adaptive"` | |
| `mutation_probability` | `[0.1, 0.02]` | High/low adaptive rates |
| `init_range_low/high` | `[-1, 1]` | Chromosome initialization |

**Fitness function** (defined per experiment in the runner):
```python
fitness = 1.0 / (loss + 1e-8)   # USE_INVERSION=True (default)
# or
fitness = -loss                   # USE_INVERSION=False
```

**Flip-aware variant** (`FlipAwareGA`): After each generation, computes the Hamming distance between child and parent activation region matrices and subtracts a diversity penalty `alpha × avg_flips` from fitness. This discourages solutions that arbitrarily flip the network's decision boundaries, stabilizing search in discontinuous landscapes.

### 4.3 Evolution Strategy — `src/algorithms/es/trainer.py` + `es/region_memory.py`

A `(μ, λ)` evolution strategy with region-aware memory.

**Core loop** (per generation):
1. Sample `λ=4` offspring per parent by adding Gaussian noise with `σ`
2. Evaluate offspring; compute regions via `GADNN.forward(..., return_regions=True)`
3. Update `RegionMemoryES`: keep the best `μ` solutions per unique region
4. Select next generation's parents from the memory pool (weighted by fitness)

**`RegionMemoryES`** (`es/region_memory.py`):
- Stores up to `max_regions=1000` distinct activation regions
- Each region's entry holds the best `mu_per_region=1` solution(s)
- When memory is full, lowest-fitness regions are pruned
- **Adaptive noise**: `σ = σ_high / (1 + decay_rate × visits)` per region — reduces exploration once a region is well-characterized

This allows the ES to maintain a diverse multi-modal population, one peak per activation region.

### 4.4 Branch-and-Bound (BaB) — `src/algorithms/bab/trainer.py`

The primary experiment algorithm. Orchestrates multiple parallel GA branches.

**Algorithm outline:**
```
1. Initialize num_branches (=5) GA instances with random populations
For each BaB iteration (total_generations / generations_per_iter):
    2. Run each branch for generations_per_iter (=50) generations
    3. Collect all solutions across all branches; compute their region matrices
    4. Select seeds for next iteration:
       a. Sort solutions by fitness (descending)
       b. Cluster by region similarity (Hamming distance ≤ diversity_threshold=0.1)
       c. One seed cluster → one branch; fill up to num_branches clusters
    5. Create new populations from seeds using 3 perturbation strategies:
       - He/Kaiming noise (scaled to fan-in) on weight entries
       - Dropout: zero out weights in one random hidden layer (p=0.2)
       - Antithetic: ±Gaussian noise pairs
    6. Reinitialize branches with new populations
    7. Decay δ (for PWLU variants)
Return global best solution across all iterations and branches
```

**Branch seeding** uses `seeds_used_for_perturbation=4` top solutions, distributing `pop_size - len(seed_list)` offspring among them with equal quotas.

**Loss history tracking**: Each branch accumulates a `loss_histories[branch_id]` list. When a branch descends from a top-3 parent, its history is stitched (parent + child history concatenated) for smooth visualization.

---

## 5. Benchmark Functions

All four paper functions take a 2D input `x ∈ ℝ²` and are evaluated on `[-2, 2]²`.

### `g_l` — Linear boundary discontinuity
```
g_l(x) = 2·sin(1.25π·‖x‖) + 4    if x₂ ≤ 2x₁
          2·sin(0.75π·‖x‖)          otherwise
```

### `g_s` — Rectangular region discontinuity
```
g_s(x) = −2x₁² + 6                 if x ∈ [−1,1] × [0, +∞)
          4·exp((1 − x₁²)/2)        otherwise
```

### `g_eta` — Exponential boundary, three regions
```
g_eta(x) = sin(0.4π(x₁+x₂))        if x₂ ≥ exp(x₁)
            sin(0.7π(x₁+x₂)) − 4   if x₂ < exp(x₁) − 1
            sin(π(x₁+x₂)) + 4      otherwise
```

### `g_gamma` — Unit-disk boundary discontinuity
```
g_gamma(x) = sin(π(x₁+x₂)) + 4     if ‖x‖² ≤ 1
              sin(0.4π(x₁+x₂))      otherwise
```

**1D basic functions** (`BASIC_TEST_FUNCTIONS`) are also available: `step_function`, `piecewise_linear_function`, `sine_with_discontinuity`, `linear_with_varying_slopes`, `multiple_steps_function`. These use `input_dim=1` and `x_range=10`.

---

## 6. Data Generation

**`src/data/generators.py` — `generate_data()`**

```python
X_raw, y = generate_data(
    num_samples=200,
    function="g_l",
    input_dim=2,
    x_min=-2.0,
    x_max=2.0
)
```

Returns raw `torch.Tensor` arrays. The runner applies `sklearn.preprocessing.StandardScaler` to `X` before training; `y` is used as-is.

**`src/data/image_dataset.py`** — utility for loading image data from `EdgeDetectionData/` for visual experiments (not used in the primary BaB runner).

---

## 7. Configuration System

All experiment knobs live in **`src/config/base_config.py`**. Import via `import src.config as config`.

### Top-level constants

| Constant | Default | Description |
|---|---|---|
| `SEED` | `42` | Global RNG seed |
| `TEST_FUNCTIONS` | `["g_l"]` | Functions to evaluate |
| `ARCHITECTURES` | `[[2, 16, 8, 4]]` | Layer size lists |
| `LOSS_FUNCTIONS` | `["mse"]` | `"mse"` or `"mae"` |
| `BASE_ACTIVATIONS` | `[True]` | `True`=ReLU, `False`=ELU |
| `DELTA_DECAY_FACTOR` | `0.99` | Per-generation δ decay for PWLU |
| `USE_INVERSION` | `True` | Fitness = `1/(loss+ε)` vs. `−loss` |
| `FLIP_ALPHAS` | `[0.01, 0.1]` | Diversity penalty weights for flip-aware GA |

### `GA_SETTINGS`
PyGAD constructor arguments forwarded directly. `num_generations` is overridden by the BaB controller at runtime.

### `ES_SETTINGS`
Controls region-aware ES. Key fields: `sigma_low/high` (noise range), `sigma_decay_rate`, `mu_per_region`, `max_regions`, `offspring_lambda`.

### `BAB_SETTINGS`
Controls the BaB trainer. Key fields: `num_branches`, `generations_per_bab_iteration`, `total_generations`, `perturbation_scale`, `diversity_threshold`, `p_dropout`, `seed_length`, `seeds_used_for_perturbation`.

### `BACKPROP_SETTINGS`
Standard training controls: `num_epochs`, `batch_size`, `learning_rates`, `lambda_regs`.

### `get_base_variations(base_discontinuous_indices)`
Returns three experiment variants for each configuration:

| Prefix | `activation` | `indices` | Effect |
|---|---|---|---|
| `cont` | `"disc"` | `[]` | Continuous baseline (no ε jump) |
| `disc` | `"disc"` | provided | Heaviside discontinuous |
| `pwlu` | `"pwlu"` | provided | Piecewise-linear activation |

The BaB runner filters out any variant with `"flip_aware": True` (those belong to the standalone GA runner) and prepends `"BaB_"` to each prefix.

---

## 8. Experiment Pipeline

`experiment_runner_BaB.py` is the canonical entry point.

```
run_bab_experiments()
│
├── Iterate over (function × architecture × loss × base_activation)
│   │
│   ├── generate_data() → X_raw, y
│   ├── StandardScaler → X_train
│   ├── get_base_variations() → [cont, disc, pwlu]
│   │
│   └── For each variation:
│       ├── Create GADNN(model_args)
│       ├── Define fitness_func_factory(shared_model)  [closure over X_train, y]
│       ├── BaBTrainer(model_class, model_args, X_train, y, ga_settings, bab_settings, ...)
│       ├── trainer.train() → best_sol, best_fit, ga_inst, history_log
│       ├── Append to results[] and write CSV
│       ├── plot_ga_result(...)     → PyGAD_BAB_Plots/
│       └── plot_bab_loss_history(...)
│
└── Final CSV: BaB_experiment_log_<timestamp>.csv
```

**Output files and directories:**

| Path | Contents |
|---|---|
| `BaB_experiment_log_<ts>.csv` | One row per (function, arch, loss, variation): fitness, final loss, timestamp |
| `PyGAD_BAB_Plots/` | PNG: best solution surface + region overlays per iteration |
| `loss_history/` | Training loss curves |
| `PyGAD_plots/` | Standalone GA run outputs |

---

## 9. Visualization & Analysis

**`src/visualization/plotter.py`**

| Function | Description |
|---|---|
| `plot_grouped_bar_charts()` | Side-by-side bars comparing models across functions |
| `plot_heatmaps()` | 2D heatmap of performance over hyperparameter grid |
| `plot_loss_history()` | Training loss curves |
| `plot_bab_iteration_best()` | Best solution surface from each BaB iteration |
| `analyze_and_plot_gadnn_regions()` | Overlays activation region boundaries on the input space |

`analyze_and_plot_gadnn_regions()` calls `GADNN.capture_activation_regions()` on a dense grid of inputs and colors each grid point by its region assignment, revealing how the network partitions input space.

**`src/analysis/torch_analyzer.py`** — computes per-neuron epsilon statistics and region size distributions for a trained GADNN.

---

## 10. Testing

Tests live in `tests/integration/` and run with pytest.

| File | What it checks |
|---|---|
| `test_gadnn.py` | GADNN forward pass shape and dtype |
| `test_gadnn_regions.py` | `capture_activation_regions()` returns correct matrix shape |
| `verify_gadnn_ga.py` | Full GA loop runs without error using GADNN fitness function |
| `check_csv.py` | Validates schema and non-emptiness of experiment log CSVs |

Run all:
```bash
pytest tests/ -v
```

---

## 11. Design Notes & Caveats

**`DiscontinuousLayer` bug** — `src/activations/discontinuous.py:81` returns `self.act(x)` and silently drops the computed jump `y`. Code using `DNN` / `PWLUDNN` through backprop is therefore training a standard continuous network. Only `GADNN` (used in evolutionary runs) correctly applies the jump (`base + epsilon * H(x)`).

**CPU-only conda environment** — `environment.yml` pins `cpuonly`. CUDA is checked at runtime (`torch.cuda.is_available()`) and seeded if found, but the default install does not include GPU binaries. Swap `cpuonly` for the appropriate CUDA package if GPU training is desired.

**PyGAD dependency not in `environment.yml`** — `pygad` is imported throughout the codebase but is not listed in `environment.yml` or `pyproject.toml`. Install manually: `pip install pygad`.

**`scikit-learn` not in `environment.yml`** — `StandardScaler` from `sklearn` is used in the runner. Install manually: `pip install scikit-learn` or add to `environment.yml`.

**`num_generations=1` in config** — `GA_SETTINGS["num_generations"]` and `BAB_SETTINGS["generations_per_bab_iteration"]` are both set to `1` in the committed config (debug values). For real experiments, restore `generations_per_bab_iteration=50` and ensure `total_generations` is set appropriately.

**Stateless GADNN** — `GADNN` is not an `nn.Module`. It holds no parameter state. All weights live in the chromosome `w` passed to `forward()`. This makes it thread-safe for parallel branch evaluation but incompatible with standard PyTorch training loops.

**Region key stability** — Region matrices are compared entry-by-entry as numpy arrays. For large networks or inputs the region space grows exponentially; `max_regions=1000` in `ES_SETTINGS` provides a hard cap with LRU-style pruning.
