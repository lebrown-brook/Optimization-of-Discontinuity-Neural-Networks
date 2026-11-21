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

Standard networks built from continuous activations such as ReLU, ELU, and sigmoid produce continuous outputs. Many real-world problems involve discontinuous targets — sharp boundaries in physical simulations, image segmentation, piecewise-defined signals.

Della Santa & Pieraccini (JCAM, 2023) propose adding a **discontinuous component** to individual neurons:

```
h(x) = σ(Wx + b) + ε ⊙ H(Wx + b)
```

where `σ` is a standard activation (ReLU or ELU), `H` is the Heaviside step function, and `ε` is a learnable vector of jump magnitudes. This allows the network to represent functions that have finite jumps at the boundaries of activation regions (the set where `Wx + b > 0`).

This repository implements this framework in PyTorch and compares backpropagation, a genetic algorithm (GA), an evolution strategy (ES), and a multi-branch GA controller (BaB).

---

## 2. Models

### 2.1 `DNN` — `src/models/dnn.py`

`DNN` is a `torch.nn.Module` with an activation after each hidden linear layer and a linear output layer.

```
Hidden linear layer → ReLU / ELU, DiscontinuousActivation, or PiecewiseLinearActivation
```

Constructor arguments:
- `input_size`, `hidden_layers: list[int]`, `output_size`
- `discontinuous_indices: list[int]` — which hidden layers get a discontinuous activation
- `Disc_Activation: str` — `"disc"` (Heaviside) or `"pwlu"` (piecewise linear)
- `Relu_true: bool` — `True` for ReLU base, `False` for ELU base

The same `DNN` class handles all three variants. With an empty `discontinuous_indices` list, every hidden layer uses the base activation.

### 2.2 `GADNN` — `src/models/gadnn.py`

A **non-`nn.Module`** class designed for evolutionary optimization. Instead of carrying `torch.nn.Parameter` tensors, it accepts a **flat numpy/tensor vector `w`** (a "chromosome") at forward time and reconstructs all weights, biases, and ε values from it.

**Flat parameter layout for one layer** (if that layer is discontinuous):
```
[W_0 … W_{out×in−1} | b_0 … b_{out−1} | ε_0 … ε_{out−1}]
```
The output layer never gets ε parameters.

`_precalculate_architecture()` runs once at init and computes pre-sliced index ranges for every layer, so the forward pass can reuse them.

**Key methods:**

| Method | Description |
|---|---|
| `forward(w, x, return_regions=False)` | Performs the forward pass; optionally returns the region matrix |
| `get_epsilon_values(w)` | Extracts ε values for each discontinuous neuron |
| `capture_activation_regions(w, x)` | Returns `(region_matrix, eps_metadata)` |
| `get_epsilon_indices_in_w()` | Global indices of ε entries in `w` |

`num_genes` is the chromosome length. For both Heaviside and PWLU, each region flag indicates whether a selected neuron's pre-activation is strictly positive.

### 2.3 `BaseModel` — `src/models/base.py`

Abstract class providing the interface: `forward()`, `get_num_parameters()`, `from_vector()`.

---

## 3. Activation Functions

### 3.1 Heaviside — `src/activations/discontinuity_layers.py`

`heaviside_function` is a `torch.autograd.Function`:
- **Forward**: `(x > 0).float()` — exact step
- **Backward**: zero gradient everywhere (subgradient)

`heaviside_function_STE` keeps the same forward step and applies `hardtanh` to the upstream gradient in the backward pass.

`DiscontinuousActivation` in `DNN` applies the base activation and adds the jump. `GADNN` evaluates the same expression using the parameters from its chromosome.

```text
out = σ(z) + ε ⊙ H(z), where z = Wx + b
```

### 3.2 Piecewise-Linear Activation (PWLU) — `src/activations/discontinuity_layers.py`

`PiecewiseLinearActivationFunction` is a custom autograd function with three regions based on pre-activation value `z`:

| Region | Condition | Output |
|---|---|---|
| 0 | `z ≤ 0` | ReLU: `0`; ELU: `exp(z) − 1` |
| 1 | `0 < z ≤ δ` | Linear transition with slope `1 + ε/δ` |
| 2 | `z > δ` | Linear: `z + ε` |

As `δ → 0`, the transition narrows and the activation approaches the Heaviside variant. GA and ES decay `δ` between generations. BaB applies `DELTA_DECAY_FACTOR ** generations_per_iter` between iterations. The evolutionary trainers return the final value used for training. Backprop applies the factor every ten epochs.

`PiecewiseLinearActivation` wraps it as an `nn.Module` with learnable `epsilon` and a scalar `delta` updated by the trainer.

---

## 4. Training Algorithms

### 4.1 Backpropagation — `src/algorithms/backprop/trainer.py`

Standard supervised training using PyTorch's Adam optimizer.

- **LR scheduling**: `ReduceLROnPlateau` (monitors mean training loss per epoch)
- **Early stopping**: patience of 250 epochs, based on training loss
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

**Fitness function** (defined per experiment in the dispatcher):
```python
fitness = 1.0 / (loss + 1e-8)   # USE_INVERSION=True (default)
# or
fitness = -loss                   # USE_INVERSION=False
```

The regular runner uses tournament selection (`K_tournament=40`) and mutation probabilities `[0.05, 0.02]` for the continuous variant. It uses steady-state selection and `[0.1, 0.02]` for Heaviside and PWLU. Passing `settings_override` bypasses those per-variant overrides.

`ga_flip_aware.py` contains optional callbacks and operators for penalizing changes in activation regions between parents and offspring. For annealed PWLU, the unified runner uses a PyGAD adapter that refreshes the entering population before parent selection. It retains PyGAD's generation-local fitness reuse and bypasses fitness lookup in saved histories, which span multiple delta values. Annealed runs support a fixed generation budget or a fitness-target stopping criterion.

### 4.3 Evolution Strategy — `src/algorithms/es/trainer.py` + `es/region_memory.py`

The evolution strategy samples parents from a fitness-weighted archive and evaluates Gaussian perturbations of each parent. The checked-in settings sample 50 parents and 10 offspring per parent. With `use_grad_step=True`, it also evaluates a candidate guided by a fitness gradient estimated from the perturbations.

`RegionMemoryES` keys each region by the bytes of its activation matrix on the training inputs. The checked-in settings keep one candidate per region and up to 1,000 regions, pruning by each region's best fitness. The continuous variant uses one shared region with room for the configured parent population. The mutation scale decays with visits to a region according to `sigma_high / (1 + sigma_decay_rate * visits)`.

For annealed PWLU, ES refreshes the bounded archive at each delta transition before sampling parents. With multiple PWLU layers, it also recomputes region keys because later preactivations can depend on earlier ramps.

### 4.4 Branch-and-Bound (BaB) — `src/algorithms/bab/trainer.py`

BaB runs several GA branches sequentially within each iteration, then uses fitness and activation-region diversity to seed the next round. The search is heuristic and provides no certified bound or optimality guarantee.

**Algorithm outline:**
```
1. Initialize num_branches (=3 in the checked-in config) GA instances with random populations
For each BaB iteration (total_generations / generations_per_iter):
    2. Run each branch for generations_per_iter generations
    3. Collect all solutions across all branches; compute their region matrices
    4. If another iteration remains, select seeds:
       a. Sort solutions by fitness (descending)
       b. For Heaviside and PWLU, group candidates within a normalized Hamming
          distance of diversity_threshold=0.1 from each cluster head
       c. Fill any remaining branch slots by fitness. The continuous variant
          selects seeds by fitness alone.
    5. Advance δ for PWLU and create new populations from seeds using 3 perturbation strategies:
       - He/Kaiming noise (scaled to fan-in) on weight entries
       - Dropout: randomly zero weights in one selected hidden layer (p=0.2)
       - Antithetic: ±Gaussian noise pairs
    6. Reinitialize branches with new populations
For annealed PWLU, rank one finalist per branch per round at the final round's δ
Return the best solution under the final objective
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

**1D basic functions** in `src/data/generators.py` include: `step_function`, `piecewise_linear_function`, `sine_with_discontinuity`, `linear_with_varying_slopes`, `multiple_steps_function`. The checked-in `basic_1d` group runs `step_function` with `input_dim=1` on `[-10, 10]`.

---

## 6. Data Generation

**`src/data/generators.py`**

```python
X_train, y_train, scaler, image_dataset = prepare_experiment_data(data_key, group_cfg)
```

`data_key` is a benchmark function name or an image path. For benchmarks, `generate_data()` samples inputs uniformly and evaluates the named function. For images, `load_image_data()` samples pixel coordinates in `[-1, 1]²` and converts RGB values to grayscale luminance. Both paths standardize inputs with `StandardScaler` and return the unscaled targets and fitted scaler.

`generate_data()` can also be called directly when custom sampling is needed:
```python
X_raw, y = generate_data(num_samples=200, function="g_l", input_dim=2, x_min=-2.0, x_max=2.0)
```

The fourth return value is a `SingleImageDataset` for image groups and `None` for benchmarks. The runner uses the full coordinate grid for grayscale reconstruction.

---

## 7. Configuration System

All experiment knobs live in **`src/config/base_config.py`**.

### Experiment groups

`EXPERIMENT_GROUPS` pairs each dataset with compatible architectures and domain settings. The two benchmark groups are shown below.

```python
EXPERIMENT_GROUPS = [
    {
        "name": "paper_2d",
        "functions": ["g_l"],           # subset of PAPER_FUNCTIONS to run
        "input_dim": 2,
        "x_min": -2.0,  "x_max": 2.0,
        "num_samples": 200,
        "hidden_architectures": [[16, 8, 4]],  # hidden layers only
        "discontinuous_indices": [1, 2],        # 16→8 and 8→4
    },
    {
        "name": "basic_1d",
        "functions": ["step_function"],
        "input_dim": 1,
        "x_min": -10.0,  "x_max": 10.0,
        "num_samples": 200,
        "hidden_architectures": [[8, 4]],
        "discontinuous_indices": [0],
    },
]
```

The runner auto-builds `full_arch = [input_dim] + hidden_arch` before handing off to model constructors (which always use `output_size=1`). `discontinuous_indices` uses zero-based indices for connections into hidden layers. In `[2, 16, 8, 4, 1]`, index 2 selects `8→4`. The output connection stays linear.

The `edge_image` group uses `image_paths` in place of `functions`. The checked-in configuration uses 1,000 pixel samples, hidden sizes `[128, 128, 64, 7, 8]`, and discontinuous index `[2]`. The runner uses the image filename stem in the CSV and plot paths.

Reference constants `PAPER_FUNCTIONS` and `BASIC_TEST_FUNCTIONS` are kept at the top of the file as lists to copy from when populating group `"functions"` keys.

### Experiment controls

| Constant | Checked-in value | Description |
|---|---|---|
| `ACTIVE_GROUPS` | `["paper_2d", "basic_1d", "edge_image"]` | Groups from `EXPERIMENT_GROUPS` to run |
| `ACTIVE_ALGORITHMS` | `["backprop", "ga", "es", "bab"]` | `"backprop"` \| `"ga"` \| `"es"` \| `"bab"` |
| `ACTIVE_VARIATIONS` | `["cont", "disc", "pwlu"]` | Activation variants |
| `LOSS_FUNCTIONS` | `["mse"]` | `"mse"` or `"mae"` |
| `BASE_ACTIVATIONS` | `[True]` | `True`=ReLU, `False`=ELU |
| `DELTA_DECAY_FACTOR` | `0.99` | PWLU decay factor; cadence depends on the trainer |
| `USE_INVERSION` | `True` | Fitness = `1/(loss+ε)` vs. `−loss` |
| `OUTPUT_DIR` | `experiments/results` | Root output directory, resolved from the project location |
| `EXPERIMENT_NAME` | `"smoke_test"` | `None` → auto `"exp_{timestamp}"` |

### Algorithm settings

`GA_SETTINGS`, `ES_SETTINGS`, `BAB_SETTINGS`, and `BACKPROP_SETTINGS` are configuration dictionaries read by the dispatcher and trainers. The dispatcher may selectively override keys at runtime (e.g. BaB fixes `num_parents_mating=10`; per-variation selection strategy is applied for GA). A `settings_override` dict can also be passed into `dispatch()` to override any key for hyperparameter sweeps.

### Helper functions

`get_variations(discontinuous_indices)` returns the filtered variation configs:

| `prefix` | `activation` | `indices` | Effect |
|---|---|---|---|
| `cont` | `"disc"` | `[]` | Continuous baseline (no ε jump) |
| `disc` | `"disc"` | provided | Heaviside discontinuous |
| `pwlu` | `"pwlu"` | provided | Piecewise-linear activation |

Only variations whose `prefix` is in `ACTIVE_VARIATIONS` are returned.

`get_active_groups()` returns the subset of `EXPERIMENT_GROUPS` matching `ACTIVE_GROUPS`.

---

## 8. Experiment Pipeline

### Entry point — `experiments/run_experiment.py`

```
run(enable_plotting=True)
│
├── ExperimentLogger(OUTPUT_DIR, experiment_name)
│
└── For each (group, loss_fn, base_act):
    └── For each function or image path:
        ├── prepare_experiment_data(data_key, group) → X_train, y_train, scaler, image_dataset
        └── For each hidden_arch in group["hidden_architectures"]:
            ├── full_arch = [input_dim] + hidden_arch
            └── For each algo in ACTIVE_ALGORITHMS:
                └── For each var_cfg in get_variations(discontinuous_indices):
                    ├── dispatch(algo, var_cfg, full_arch, X_train, y_train, ...) → TrainingResult
                    ├── logger.log_result(result, group_name)   → appends to experiment_log.csv
                    └── Save per-variation plots when enabled
                After all variations, save grouped plots when enabled
                and more than one variation ran.
```

### Algorithm dispatcher — `src/experiment/dispatcher.py`

`dispatch(algorithm, variation_cfg, full_arch, X_train, y_train, func_name, loss_function, base_activation, settings_override)` routes to one of four runner functions. Each runner:
1. Constructs the appropriate model (`DNN` for backprop, `GADNN` for evolutionary)
2. Calls the algorithm's trainer
3. Returns a `TrainingResult`

Pre-training weights are captured for every algorithm so that before/after plots can be generated:
- **GA**: `ga_instance.population[0].copy()` before `ga_instance.run()`
- **ES**: a separate random uniform sample from the configured initialization range
- **BaB**: a separate random uniform sample from the GA initialization range
- **Backprop**: `parameters_to_vector(model.parameters())` before training

### `TrainingResult` — `src/experiment/results.py`

A `@dataclass` returned by every dispatcher path:

| Field | Type | Description |
|---|---|---|
| `algorithm` | `str` | `"backprop"` \| `"ga"` \| `"es"` \| `"bab"` |
| `variation` | `str` | `"cont"` \| `"disc"` \| `"pwlu"` |
| `function` | `str` | Benchmark name or image filename stem |
| `input_dim` | `int` | |
| `hidden_architecture` | `list[int]` | Hidden layer sizes only |
| `loss_function` | `str` | `"mse"` or `"mae"` |
| `base_activation` | `bool` | `True`=ReLU |
| `final_loss` | `float` | |
| `loss_history` | `list[float]` | One entry per generation/epoch |
| `best_weights` | `np.ndarray \| None` | Flat chromosome or flattened DNN parameters |
| `trained_model` | model instance | |
| `pre_weights` | `np.ndarray \| None` | Pre-training flat weights |
| `pre_delta` | `float \| None` | Initial delta for evolutionary PWLU plots |
| `metadata` | `dict` | Algo-specific extras (see below) |
| `runtime_seconds` | `float` | |

`metadata` keys by algorithm: `generations` / `pop_size` (GA, ES); `regions_discovered`, `stored_candidates`, `accepted_total`, `evaluated_total` (ES); `total_generations`, `num_branches`, `generations_per_iter`, `history_log` (BaB); `epochs_run`, `learning_rate`, `lambda_reg`, `seed` (backprop). The evolutionary trainers also report `refresh_evaluations` and `refresh_seconds` for delta-related rescoring.

### Logger — `src/experiment/logger.py`

`ExperimentLogger` manages the output directory tree and the unified CSV log.

```python
logger = ExperimentLogger(output_dir, experiment_name)
logger.log_result(result, group_name)
logger.get_plot_path(result, "regions", "regions_pre", individual=True)
# → experiments/results/exp_X/regions/bab/g_l/_individual/disc_[16,8,4]_mse_regions_pre.png
logger.get_bab_iter_path(result, 2)
# → experiments/results/exp_X/predictions/bab/g_l/_individual/disc_[16,8,4]_mse_iter_02.png
```

**CSV schema** — unified across all algorithms:
```
timestamp | algorithm | group | function | hidden_arch | variation |
activation | loss_function | final_loss | total_iters | runtime_s |
pop_size | regions_discovered | num_branches
```
Algorithm-specific columns are empty when the trainer does not report them. The first result from a new logger replaces any existing CSV for that experiment name. Later results from the same logger append rows.

### Output directory structure

```text
experiments/results/{experiment_name}/
  experiment_log.csv
  loss_histories/{algo}/{item}/   # per-variation and combined loss curves
  predictions/{algo}/{item}/      # before/after predictions and BaB snapshots
  regions/{algo}/{item}/          # before/after maps for Heaviside and PWLU
  reconstructions/{algo}/{item}/  # grayscale reconstructions for image groups
```

Per-variation plots use `{variation}_{arch}_{loss}_{descriptor}.png`. Most sit under `_individual/`. The main loss curves and grouped plots sit at the top of their plot folders. Grouped plots use `combined_{arch}_{loss}_{descriptor}.png`. Filenames omit the base activation, so running both ReLU and ELU under the same experiment name can overwrite plots.

### Hyperparameter sweep — `experiments/sweep_hyperparams.py`

The script defines a Cartesian grid in `PARAM_GRID` and selects its inputs using `SWEEP_ALGORITHM`, `SWEEP_GROUP`, `SWEEP_FUNCTION`, and `SWEEP_ARCH`. It is intended to pass each combination as `settings_override` to `dispatch()` and save a pivot CSV under `{OUTPUT_DIR}/sweeps/`.

The current script expects three values from `prepare_experiment_data()`. The function returns four, so the sweep stops before dispatching any runs.

---

## 9. Visualization & Analysis

The unified runner's plotting functions live in `src/visualization/plotter.py`. The runner passes output paths from `ExperimentLogger`, and plotting helpers create parent directories as needed.

### Public API

| Function | When called | Output |
|---|---|---|
| `plot_loss_history(loss_history, save_path)` | Every algorithm with a non-empty history | Single log-scale loss curve |
| `plot_prediction(model, weights, X_train, y_train, save_path, scaler, x_min, x_max)` | Pre- and post-training for all algorithms | 1D: scatter + line. 2D: contour surface |
| `plot_image_reconstruction(model, weights, image_dataset, save_path, scaler)` | Image groups only | Grayscale reconstruction compared with the source |
| `plot_regions(model, weights, X_train, y_train, save_path, x_min, x_max, model_type, top_eps, scaler)` | Pre- and post-training for `disc`/`pwlu` variants | Region color map (dispatches GADNN vs DNN) |
| `plot_bab_iterations(model, history_log, X_train, y_train, get_iter_path, generations_per_iter, ...)` | BaB runs only | One figure per iteration showing top-3 branch predictions |
| `plot_bab_loss_history(history_log, generations_per_iter, config_name, save_path)` | BaB runs only | Multi-branch loss history with lineage colour coding |
| `plot_grouped_bar_charts(df, output_dir)` | Post-hoc analysis from CSV | Grouped bar charts comparing algorithms |
| `plot_heatmaps(df, output_dir)` | Post-hoc analysis from CSV | Heatmaps over hyperparameter grids |

### `plot_prediction` — DNN vs GADNN dispatch

For a PyTorch module, the plotter calls `model(grid_input)`. If pre-training weights are supplied, it loads them into a copy of the model first. For GADNN, it calls `model.forward(weights, grid_input)`.

The evaluation grid is always generated in the original (unscaled) domain and passed through `scaler.transform()` for model input. Scatter points are inverse-transformed for display.

### `plot_regions` — region dispatch

`model_type="gadnn"` → `_plot_regions_gadnn()`: calls `model.forward(weights, grid, return_regions=True)`. The returned `(N, D)` region matrix is converted to integer IDs by `_hash_regions()` (numpy void-view trick, O(N log N)) and colormapped with `tab20`. 1D: background color band behind the prediction curve. 2D: side-by-side panels — region map on left, prediction surface on right.

`model_type="dnn"` → `_plot_regions_dnn()`: deep-copies the model and loads flat weights via `vector_to_parameters()` when pre-training weights are supplied, then delegates to `analyze_and_plot_regions()` in `src/analysis/torch_analyzer.py`.

### `plot_bab_loss_history` — branch lineage colours

Each branch's `local_history` segment is drawn as a continuous line. Lineage is tracked across iterations: a branch inheriting from a previous top-3 winner uses the same colour; sibling branches from the same parent get HSV-shifted variants (lighter/darker); fresh lineages draw a new `tab10` colour. Circle markers indicate inherited starts; × markers indicate fresh starts.

### Backward-compat aliases

Four aliases at the bottom of plotter.py exist for `BaBTrainer`'s internal `enable_plotting=True` code path, which imports by the old function names. They are never reached in normal use (the dispatcher sets `enable_plotting=False` on `BaBTrainer`):

| Alias | Wraps |
|---|---|
| `analyze_and_plot_gadnn_regions` | `_plot_regions_gadnn` |
| `plot_bab_iteration_best` | `plot_bab_iterations` |
| `plot_ga_result` | `plot_prediction` |
| `plot_combined_loss_history` | inline multi-curve variant |

### `src/analysis/torch_analyzer.py`

`TorchDiscontinuousNeuralNetworkAnalyzer` and `analyze_and_plot_regions()` — compute per-neuron epsilon statistics and region size distributions for a trained DNN. Called by `_plot_regions_dnn()`.

---

## 10. Testing

Run the focused pytest check for GADNN region capture:

```bash
python -m pytest tests/integration/test_gadnn_regions.py
```

This test checks matrix shape, binary region values, and epsilon metadata. `test_image_dataloader.py` imports from the obsolete top-level `image_dataset` module. `test_gadnn.py` runs a GA demo and opens a plot at import time, so collecting the full suite can launch training.

---

## 11. Design Notes & Caveats

**Environment** — `environment.yml` specifies Python 3.10 and CPU PyTorch. The runner also requires PyGAD and scikit-learn. The sweep and post-processing scripts import pandas. These three packages are absent from `environment.yml`.

**Smoke-test budgets** — The checked-in settings budget 3 backpropagation epochs, 3 generations each for GA and ES, and 6 generations per BaB branch in blocks of 2 across 3 branches.

**Loss reporting** — Backpropagation tracks average training loss including any configured L2 penalty. Evolutionary trainers score candidates on the training samples. The runner does not create a validation split.

**Before/after plots** — Evolutionary PWLU plots use the stored initial delta for pre-training weights and the stored round delta for BaB snapshots. Backpropagation plots restore pre-training weights and use the trained model's layer deltas.

**BaB `enable_plotting` bypass** — `BaBTrainer` has its own internal plotting path (called when `enable_plotting=True`) that uses old function names. The dispatcher always passes `enable_plotting=False`; all plotting is handled by `run_experiment._save_plots()` instead, which uses the structured output paths from `ExperimentLogger`.
