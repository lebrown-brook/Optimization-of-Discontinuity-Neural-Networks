# dnn_research — Discontinuous Neural Networks

A PyTorch research framework for training neural networks that can approximate **discontinuous functions**, implementing and extending the theoretical work of Della Santa & Pieraccini (JCAM, 2023). The project provides multiple optimization paradigms — backpropagation, genetic algorithms, evolution strategies, and a branch-and-bound-inspired genetic search — alongside a suite of benchmark functions for systematic evaluation.

---

## Key Innovations

- **Learnable discontinuous activations** — a Heaviside-based jump term `ε ⊙ H(x)` is added to standard ReLU/ELU neurons at designated layers, letting the network learn the location and magnitude of discontinuities
- **Piecewise-linear activation (PWLU)** — a continuous approximation with a linear transition that approaches a discontinuous jump as `δ` shrinks
- **Region-aware optimization** — algorithms track which neurons sit above or below their discontinuity threshold (the "activation region") and use this structure to guide search and preserve diversity
- **Branch-and-Bound (BaB) GA controller** — runs multiple GA branches and selects seeds for the next round using fitness and activation-region diversity

---

## Repository Structure

```
PythonProject/
├── src/
│   ├── activations/          # Heaviside, STE, piecewise-linear activations
│   ├── algorithms/
│   │   ├── backprop/         # Adam + LR scheduling + early stopping
│   │   ├── ga/               # Optional flip-aware PyGAD operators
│   │   ├── es/               # Region-aware evolution strategy
│   │   └── bab/              # Branch-and-bound GA controller
│   ├── config/
│   │   └── base_config.py    # Central config: EXPERIMENT_GROUPS, algorithm settings
│   ├── data/
│   │   ├── generators.py     # Data sampling + prepare_experiment_data()
│   │   ├── image_dataset.py  # Pixel coordinates and RGB samples
│   │   └── test_functions.py # Benchmark functions g_l, g_s, g_eta, g_gamma
│   ├── experiment/
│   │   ├── dispatcher.py     # Algorithm dispatch → TrainingResult
│   │   ├── logger.py         # ExperimentLogger: paths + unified CSV
│   │   └── results.py        # TrainingResult dataclass
│   ├── models/
│   │   ├── dnn.py            # DNN with continuous, Heaviside, or PWLU activations
│   │   ├── gadnn.py          # GADNN — flat-vector model for evolutionary search
│   │   └── base.py           # Abstract base class
│   ├── analysis/             # TorchDiscontinuousNeuralNetworkAnalyzer
│   └── visualization/
│       └── plotter.py        # Unified plot functions (all path-parameterized)
├── experiments/
│   ├── run_experiment.py     # Unified entry point
│   └── sweep_hyperparams.py  # Hyperparameter grid search
├── tests/
│   └── integration/          # pytest integration tests
├── pyproject.toml
└── environment.yml
```

For a full technical description of every component, see [OVERVIEW.md](OVERVIEW.md).

---

## Installation

```bash
conda env create -f environment.yml
conda activate deltann
python -m pip install pygad scikit-learn
python -m pip install -e .
```

**Python 3.10** and CPU PyTorch are specified in the conda environment. The runner also requires PyGAD and scikit-learn, installed separately above. The sweep and post-processing scripts also import pandas.

---

## Quick Start

The checked-in settings in `src/config/base_config.py` are for smoke tests. Before running, update the machine-specific path in the `edge_image` group's `image_paths` or remove `"edge_image"` from `ACTIVE_GROUPS` for a benchmark-only run. Set a new `EXPERIMENT_NAME` or use `None` to create a timestamped name. Reusing a name overwrites the CSV on the first logged result and can overwrite plots.

```bash
python experiments/run_experiment.py
```

| Setting | Checked-in value | Description |
|---|---|---|
| `ACTIVE_GROUPS` | `["paper_2d", "basic_1d", "edge_image"]` | Which function/architecture groups to run |
| `ACTIVE_ALGORITHMS` | `["backprop", "ga", "es", "bab"]` | `"backprop"` \| `"ga"` \| `"es"` \| `"bab"` |
| `ACTIVE_VARIATIONS` | `["cont", "disc", "pwlu"]` | Which activation variants to run |
| `LOSS_FUNCTIONS` | `["mse"]` | `"mse"` or `"mae"` |
| `BASE_ACTIVATIONS` | `[True]` | `True`=ReLU, `False`=ELU |
| `EXPERIMENT_NAME` | `"smoke_test"` | `None` → auto `"exp_{timestamp}"` |
| `OUTPUT_DIR` | `experiments/results` | Root output directory, resolved from the project location |

Each entry in `EXPERIMENT_GROUPS` pairs benchmark functions or image paths with input dimensions, domain bounds, a sample count, hidden architectures, and discontinuous layer indices.

Outputs are written to:
```
experiments/results/{experiment_name}/
    experiment_log.csv          — one row per run, unified schema across all algorithms
    loss_histories/{algo}/{fn}/ — loss curves
    predictions/{algo}/{fn}/    — prediction surface plots (pre- and post-training)
    regions/{algo}/{fn}/        — discontinuity region maps (disc/pwlu variants only)
    reconstructions/{algo}/{fn}/ — grayscale reconstructions (image groups only)
```

---

## Hyperparameter Sweep

`experiments/sweep_hyperparams.py` defines a parameter grid and is intended to write a pivot CSV under `experiments/results/sweeps/`. The current script stops during setup because it expects three values from `prepare_experiment_data()`, which returns four.

---

## Running Tests

Run the focused pytest check for GADNN region capture:

```bash
python -m pytest tests/integration/test_gadnn_regions.py
```

The image loader test imports from an outdated module path. Collecting `test_gadnn.py` can launch training and open a plot because the GA demo runs at import time.

---

## Reference

Della Santa, F., & Pieraccini, S. (2023). *Discontinuous Neural Networks and Discontinuity Learning.* Journal of Computational and Applied Mathematics, 419, 114678.
