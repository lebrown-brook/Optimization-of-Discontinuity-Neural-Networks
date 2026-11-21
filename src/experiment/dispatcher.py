"""
Algorithm dispatcher: thin wrappers that call existing trainers and package
results as TrainingResult. No training logic lives here.
"""

import copy
import random
import time
from typing import Optional

import numpy as np
import pygad
import torch

from src.config import base_config as config
from src.models.dnn import DNN
from src.models.gadnn import GADNN
from src.algorithms.backprop.trainer import backprop
from src.algorithms.es.trainer import initialize_region_memory, train_region_aware_es
from src.algorithms.es.region_memory import RegionMemoryES
from src.algorithms.bab.trainer import BaBTrainer
from src.algorithms.fitness import score_predictions
from src.algorithms.ga import ga_flip_aware
from src.algorithms.ga.pwlu_annealing import PWLUAnnealingGA
from src.experiment.results import TrainingResult


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_gadnn(full_arch, variation_cfg, base_activation):
    return GADNN(
        input_size=full_arch[0],
        hidden_layers=full_arch[1:],
        output_size=1,
        discontinuous_indices=variation_cfg["indices"],
        Disc_Activation=variation_cfg["activation"],
        Relu_true=base_activation,
    )


def _score_chromosome(model, weights, X_train, y_train, loss_function, use_inversion):
    with torch.no_grad():
        predictions = model.forward(weights, X_train)
        return score_predictions(predictions, y_train, loss_function, use_inversion)


def _make_fitness_func(model, X_train, y_train, loss_function, use_inversion):
    def fitness_func(ga_instance, solution, solution_idx):
        fitness, _ = _score_chromosome(model, solution, X_train, y_train, loss_function, use_inversion)
        return fitness
    return fitness_func


def _fitness_to_loss(fitness, use_inversion):
    if use_inversion:
        if fitness == -1e9:
            return 1e9
        return 1.0 / fitness - 1e-8 if fitness else float("inf")
    return -fitness


# ──────────────────────────────────────────────────────────────────────────────
# Backpropagation
# ──────────────────────────────────────────────────────────────────────────────

def run_backprop(
    variation_cfg: dict,
    full_arch: list,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    func_name: str,
    loss_function: str,
    base_activation: bool,
    settings_override: Optional[dict] = None,
) -> TrainingResult:
    bp = config.BACKPROP_SETTINGS.copy()
    if settings_override:
        bp.update(settings_override)

    results = []
    for seed in bp["seeds"]:
        for lr in bp["learning_rates"]:
            for l_reg in bp["lambda_regs"]:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = DNN(
                    input_size=full_arch[0],
                    hidden_layers=full_arch[1:],
                    output_size=1,
                    discontinuous_indices=variation_cfg["indices"],
                    Disc_Activation=variation_cfg["activation"],
                    Relu_true=base_activation,
                    STE_true=(variation_cfg["prefix"] == "disc"),
                    name=f"DNN-{variation_cfg['prefix']}",
                )

                # Capture flat pre-training weights
                pre_weights = torch.nn.utils.parameters_to_vector(
                    model.parameters()
                ).detach().cpu().numpy().copy()

                t0 = time.time()
                _, final_losses, loss_histories = backprop(
                    [model],
                    X_train,
                    y_train,
                    num_epochs=bp["num_epochs"],
                    learning_rate=lr,
                    delta_factor=config.DELTA_DECAY_FACTOR,
                    lambda_reg=l_reg,
                    batch_size=bp["batch_size"],
                    loss_function=loss_function,
                )
                runtime = time.time() - t0

                best_weights = torch.nn.utils.parameters_to_vector(
                    model.parameters()
                ).detach().cpu().numpy().copy()

                results.append(TrainingResult(
                    algorithm="backprop",
                    variation=variation_cfg["prefix"],
                    function=func_name,
                    input_dim=full_arch[0],
                    hidden_architecture=full_arch[1:],
                    loss_function=loss_function,
                    base_activation=base_activation,
                    final_loss=final_losses[0],
                    loss_history=loss_histories[0],
                    best_weights=best_weights,
                    trained_model=model,
                    pre_weights=pre_weights,
                    metadata={
                        "epochs_run":    len(loss_histories[0]),
                        "learning_rate": lr,
                        "lambda_reg":    l_reg,
                        "seed":          seed,
                    },
                    runtime_seconds=runtime,
                ))

    # Return best result across seeds/lr/l_reg
    return min(results, key=lambda r: r.final_loss)


# ──────────────────────────────────────────────────────────────────────────────
# Genetic Algorithm
# ──────────────────────────────────────────────────────────────────────────────

def run_ga(
    variation_cfg: dict,
    full_arch: list,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    func_name: str,
    loss_function: str,
    base_activation: bool,
    settings_override: Optional[dict] = None,
) -> TrainingResult:
    use_inv = config.USE_INVERSION
    delta_decay = config.DELTA_DECAY_FACTOR
    seed = config.SEED

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    model = _build_gadnn(full_arch, variation_cfg, base_activation)
    num_genes = model.num_genes
    pre_delta = model.delta if model.pwlu_layer_count else None
    annealing = pre_delta is not None and delta_decay != 1.0

    loss_history = []

    def on_generation(ga_instance):
        _, best_fit, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
        loss = _fitness_to_loss(best_fit, use_inv)
        loss_history.append(loss)
        gen = ga_instance.generations_completed
        if gen % 50 == 0:
            print(f"GA ({variation_cfg['prefix']}/{func_name}) gen={gen}, loss={loss:.4f}")

    ga_cfg = config.GA_SETTINGS.copy()
    if settings_override:
        ga_cfg.update(settings_override)

    # Per-variation selection strategy (only when not doing a sweep override)
    if settings_override is None:
        if not variation_cfg["indices"]:  # continuous — tournament
            ga_cfg.update({
                "parent_selection_type": "tournament",
                "K_tournament":          40,
                "mutation_probability":  [0.05, 0.02],
            })
        else:                              # discontinuous — SSS
            ga_cfg.update({
                "parent_selection_type": "sss",
                "mutation_probability":  [0.1, 0.02],
            })

    ga_cfg.update({
        "num_genes":     num_genes,
        "fitness_func":  _make_fitness_func(model, X_train, y_train, loss_function, use_inv),
        "on_generation": on_generation,
    })

    if annealing:
        ga_instance = PWLUAnnealingGA(model, delta_decay, **ga_cfg)
    else:
        ga_instance = pygad.GA(**ga_cfg)
    pre_weights = ga_instance.population[0].copy()

    t0 = time.perf_counter()
    ga_instance.run()
    solution, best_fit, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    if pre_delta is not None:
        _, final_loss = _score_chromosome(model, solution, X_train, y_train, loss_function, use_inv)
    else:
        final_loss = _fitness_to_loss(best_fit, use_inv)
    runtime = time.perf_counter() - t0

    return TrainingResult(
        algorithm="ga",
        variation=variation_cfg["prefix"],
        function=func_name,
        input_dim=full_arch[0],
        hidden_architecture=full_arch[1:],
        loss_function=loss_function,
        base_activation=base_activation,
        final_loss=final_loss,
        loss_history=loss_history,
        best_weights=solution,
        trained_model=model,
        pre_weights=pre_weights,
        pre_delta=pre_delta,
        metadata={
            "generations": ga_cfg["num_generations"],
            "pop_size":    ga_cfg["sol_per_pop"],
            "refresh_evaluations": ga_instance.refresh_evaluations if annealing else 0,
            "refresh_seconds": ga_instance.refresh_seconds if annealing else 0.0,
        },
        runtime_seconds=runtime,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Evolution Strategy
# ──────────────────────────────────────────────────────────────────────────────

def run_es(
    variation_cfg: dict,
    full_arch: list,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    func_name: str,
    loss_function: str,
    base_activation: bool,
    settings_override: Optional[dict] = None,
) -> TrainingResult:
    use_inv = config.USE_INVERSION
    delta_decay = config.DELTA_DECAY_FACTOR

    es_cfg = config.ES_SETTINGS.copy()
    if settings_override:
        es_cfg.update(settings_override)

    seed = int(es_cfg.get("random_seed", config.SEED))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    model = _build_gadnn(full_arch, variation_cfg, base_activation)
    pre_delta = model.delta if model.pwlu_layer_count else None

    population_size = int(es_cfg["population_size_P"])
    use_regions = bool(variation_cfg["indices"])
    mu_per_region = population_size if not use_regions else int(es_cfg["mu_per_region"])
    max_regions = int(es_cfg.get("max_regions", 1000))

    region_memory = RegionMemoryES(mu_per_region=mu_per_region, max_regions=max_regions)
    initialize_region_memory(
        model=model,
        x_train=X_train,
        y_train=y_train,
        loss_name=loss_function,
        use_inversion=use_inv,
        region_memory=region_memory,
        rng=rng,
        population_size=population_size,
        init_low=float(es_cfg["init_range_low"]),
        init_high=float(es_cfg["init_range_high"]),
        use_regions=use_regions,
    )

    pre_weights = rng.uniform(
        float(es_cfg["init_range_low"]),
        float(es_cfg["init_range_high"]),
        size=(model.num_genes,),
    ).astype(np.float32)

    t0 = time.perf_counter()
    es_result = train_region_aware_es(
        model=model,
        x_train=X_train,
        y_train=y_train,
        loss_name=loss_function,
        use_inversion=use_inv,
        delta_decay_factor=delta_decay,
        region_memory=region_memory,
        rng=rng,
        es_settings=es_cfg,
        use_regions=use_regions,
    )
    runtime = time.perf_counter() - t0

    return TrainingResult(
        algorithm="es",
        variation=variation_cfg["prefix"],
        function=func_name,
        input_dim=full_arch[0],
        hidden_architecture=full_arch[1:],
        loss_function=loss_function,
        base_activation=base_activation,
        final_loss=es_result.best_loss if es_result.best_loss is not None else float("inf"),
        loss_history=es_result.loss_history,
        best_weights=es_result.best_weights,
        trained_model=model,
        pre_weights=pre_weights,
        pre_delta=pre_delta,
        metadata={
            "generations":        int(es_cfg["num_generations"]),
            "pop_size":           population_size,
            "regions_discovered": es_result.regions_discovered,
            "stored_candidates":  es_result.stored_candidates,
            "accepted_total":     es_result.accepted_total,
            "evaluated_total":    es_result.evaluated_total,
            "refresh_evaluations": es_result.refresh_evaluations,
            "refresh_seconds":    es_result.refresh_seconds,
        },
        runtime_seconds=runtime,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Branch-and-Bound GA
# ──────────────────────────────────────────────────────────────────────────────

def run_bab(
    variation_cfg: dict,
    full_arch: list,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    func_name: str,
    loss_function: str,
    base_activation: bool,
    settings_override: Optional[dict] = None,
) -> TrainingResult:
    use_inv = config.USE_INVERSION
    delta_decay = config.DELTA_DECAY_FACTOR

    bab_cfg = config.BAB_SETTINGS.copy()
    if settings_override:
        bab_cfg.update(settings_override)

    model_args = dict(
        input_size=full_arch[0],
        hidden_layers=full_arch[1:],
        output_size=1,
        discontinuous_indices=variation_cfg["indices"],
        Disc_Activation=variation_cfg["activation"],
        Relu_true=base_activation,
    )
    shared_model = GADNN(**model_args)
    pre_delta = shared_model.delta if shared_model.pwlu_layer_count else None

    fitness_func = _make_fitness_func(shared_model, X_train, y_train, loss_function, use_inv)
    fitness_to_loss = lambda f: _fitness_to_loss(f, use_inv)

    ga_cfg = config.GA_SETTINGS.copy()
    ga_cfg.update({
        "num_parents_mating":    10,
        "keep_elitism":          2,
        "mutation_probability":  [0.1, 0.02],
        "parent_selection_type": "sss",
        "num_genes":             shared_model.num_genes,
        "fitness_func":          fitness_func,
    })

    trainer = BaBTrainer(
        model_class=GADNN,
        model_args=model_args,
        X_train=X_train,
        y_train=y_train,
        ga_settings=ga_cfg,
        bab_settings=bab_cfg,
        fitness_to_loss=fitness_to_loss,
        shared_model=shared_model,
        delta_decay=delta_decay,
        enable_plotting=False,  # plots handled by run_experiment
    )

    # Capture a random initial solution as pre-training weights
    init_pop = np.random.uniform(
        low=config.GA_SETTINGS["init_range_low"],
        high=config.GA_SETTINGS["init_range_high"],
        size=(shared_model.num_genes,),
    )

    t0 = time.perf_counter()
    best_sol, best_fit, _, history_log = trainer.train()
    if pre_delta is not None:
        _, final_loss = _score_chromosome(shared_model, best_sol, X_train, y_train, loss_function, use_inv)
    else:
        final_loss = fitness_to_loss(best_fit)
    runtime = time.perf_counter() - t0

    # Build loss history from the top-1 branch loss histories
    loss_history = []
    if trainer.branch_ids:
        final_branches = sorted(
            [(trainer.loss_histories.get(b_id, []), b_id) for b_id in trainer.branch_ids],
            key=lambda x: x[0][-1] if x[0] else float("inf"),
        )
        if final_branches and final_branches[0][0]:
            loss_history = final_branches[0][0]

    return TrainingResult(
        algorithm="bab",
        variation=variation_cfg["prefix"],
        function=func_name,
        input_dim=full_arch[0],
        hidden_architecture=full_arch[1:],
        loss_function=loss_function,
        base_activation=base_activation,
        final_loss=final_loss,
        loss_history=loss_history,
        best_weights=best_sol,
        trained_model=shared_model,
        pre_weights=init_pop,
        pre_delta=pre_delta,
        metadata={
            "total_generations":   bab_cfg["total_generations"],
            "num_branches":        bab_cfg["num_branches"],
            "generations_per_iter": bab_cfg["generations_per_bab_iteration"],
            "history_log":         history_log,
            "refresh_evaluations": trainer.refresh_evaluations,
            "refresh_seconds":     trainer.refresh_seconds,
        },
        runtime_seconds=runtime,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Unified dispatch entry point
# ──────────────────────────────────────────────────────────────────────────────

_DISPATCHERS = {
    "backprop": run_backprop,
    "ga":       run_ga,
    "es":       run_es,
    "bab":      run_bab,
}


def dispatch(
    algorithm: str,
    variation_cfg: dict,
    full_arch: list,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    func_name: str,
    loss_function: str,
    base_activation: bool,
    settings_override: Optional[dict] = None,
) -> TrainingResult:
    fn = _DISPATCHERS.get(algorithm)
    if fn is None:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {list(_DISPATCHERS)}")
    return fn(
        variation_cfg=variation_cfg,
        full_arch=full_arch,
        X_train=X_train,
        y_train=y_train,
        func_name=func_name,
        loss_function=loss_function,
        base_activation=base_activation,
        settings_override=settings_override,
    )
