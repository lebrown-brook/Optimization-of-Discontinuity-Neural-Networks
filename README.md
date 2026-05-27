# dnn_research — Discontinuous Neural Networks

A PyTorch research framework for training neural networks that can approximate **discontinuous functions**, implementing and extending the theoretical work of Della Santa & Pieraccini (JCAM, 2023). The project provides multiple optimization paradigms — backpropagation, genetic algorithms, evolution strategies, and branch-and-bound search — alongside a suite of benchmark functions for systematic evaluation.

---

## Key Innovations

- **Learnable discontinuous activations** — a Heaviside-based jump term `ε ⊙ H(x)` is added to standard ReLU/ELU neurons at designated layers, letting the network learn the location and magnitude of discontinuities
- **Piecewise-linear activation (PWLU)** — a smooth, differentiable approximation that converges to a discontinuous step as the `δ` parameter decays
- **Region-aware optimization** — algorithms track which neurons sit above or below their discontinuity threshold (the "activation region") and use this structure to guide search and preserve diversity
- **Branch-and-Bound (BaB) parallel GA** — multiple GA branches evolve independently, then periodically exchange solutions based on regional diversity, preventing premature convergence

---

## Repository Structure

```
PythonProject/
├── src/
│   ├── activations/          # Heaviside, STE, piecewise-linear activations
│   ├── algorithms/
│   │   ├── backprop/         # Adam + LR scheduling + early stopping
│   │   ├── ga/               # Flip-aware genetic algorithm (PyGAD)
│   │   ├── es/               # Region-aware evolution strategy
│   │   └── bab/              # Branch-and-bound GA controller
│   ├── config/               # Central hyperparameter config (base_config.py)
│   ├── data/
│   │   ├── generators.py     # Train/test data sampling
│   │   └── test_functions.py # Benchmark functions g_l, g_s, g_eta, g_gamma
│   ├── experiment/           # Runner and CSV/JSON tracker
│   ├── models/
│   │   ├── dnn.py            # Standard DNN / PWLUDNN (torch.nn.Module)
│   │   ├── gadnn.py          # GADNN — flat-vector model for evolutionary search
│   │   └── base.py           # Abstract base class
│   ├── utils/
│   └── visualization/        # Plots: loss curves, heatmaps, region analysis
├── tests/
│   └── integration/          # pytest integration tests
├── experiment_runner_BaB.py  # Main experiment entry point
├── pyproject.toml
└── environment.yml
```

For a full technical description of every component, see [OVERVIEW.md](OVERVIEW.md).

---

## Installation

```bash
conda env create -f environment.yml
conda activate deltann
pip install -e .          # install the src/ package in editable mode
```

**Python 3.10** is pinned in the conda environment. Key dependencies: `pytorch` (CPU), `numpy`, `matplotlib`, `scipy`, `torchvision`, `pygad`, `scikit-learn`, `pytest`.

---

## Quick Start

```bash
# Run the Branch-and-Bound experiment (entry point)
python experiment_runner_BaB.py
```

The runner reads all settings from `src/config/base_config.py`. Edit that file to change:

| Setting | Default | Description |
|---|---|---|
| `TEST_FUNCTIONS` | `["g_l"]` | Which benchmark functions to evaluate |
| `ARCHITECTURES` | `[[2, 16, 8, 4]]` | Layer sizes (input → hidden → output) |
| `LOSS_FUNCTIONS` | `["mse"]` | `"mse"` or `"mae"` |
| `BAB_SETTINGS["total_generations"]` | `500` | Total GA generation budget |
| `BAB_SETTINGS["num_branches"]` | `5` | Parallel branch count |

Outputs are written to:
- `BaB_experiment_log_<timestamp>.csv` — tabular results
- `PyGAD_BAB_Plots/` — solution and region visualizations per iteration

---

## Running Tests

```bash
pytest tests/
```

Integration tests cover GADNN forward pass, region capture, GA integration, and CSV log validation.

---

## Reference

Della Santa, F., & Pieraccini, S. (2023). *Discontinuous Neural Networks and Discontinuity Learning.* Journal of Computational and Applied Mathematics, 419, 114678.
