from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from .region_memory import ESSolutionRecord, RegionMemoryES


@dataclass
class ESTrainingResult:
    loss_history: list[float] = field(default_factory=list)
    accepted_total: int = 0
    evaluated_total: int = 0
    best_fitness: float | None = None
    best_loss: float | None = None
    best_weights: np.ndarray | None = None
    regions_discovered: int = 0
    stored_candidates: int = 0


def _evaluate_solution(model, weights, x_train, y_train, loss_name: str, use_inversion: bool, use_regions: bool = True):
    with torch.no_grad():
        if use_regions:
            predictions, region_matrix = model.forward(weights, x_train, return_regions=True)
        else:
            predictions = model.forward(weights, x_train, return_regions=False)
            region_matrix = np.array([], dtype=np.uint8)

    if torch.isnan(predictions).any():
        return -1e9, 1e9, region_matrix

    if loss_name == "mse":
        loss = torch.nn.functional.mse_loss(predictions, y_train).item()
    elif loss_name == "mae":
        loss = torch.nn.functional.l1_loss(predictions, y_train).item()
    else:
        loss = 1e9

    fitness = 1.0 / (loss + 1e-8) if use_inversion else -loss
    return float(fitness), float(loss), region_matrix


def initialize_region_memory(
    model,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    loss_name: str,
    use_inversion: bool,
    region_memory: RegionMemoryES,
    rng: np.random.Generator,
    population_size: int,
    init_low: float,
    init_high: float,
    use_regions: bool = True,
):
    num_genes = model.num_genes
    init_population = rng.uniform(init_low, init_high, size=(population_size, num_genes)).astype(np.float32)

    for weights in init_population:
        fit, loss, regions = _evaluate_solution(model, weights, x_train, y_train, loss_name, use_inversion, use_regions=use_regions)
        region_memory.add_candidate(weights, fit, loss, regions)


def train_region_aware_es(
    model,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    loss_name: str,
    use_inversion: bool,
    delta_decay_factor: float,
    region_memory: RegionMemoryES,
    rng: np.random.Generator,
    es_settings: dict,
    on_generation: Optional[Callable[[int, ESSolutionRecord, int, int, RegionMemoryES], None]] = None,
    use_regions: bool = True,
) -> ESTrainingResult:
    population_size = int(es_settings["population_size_P"])
    offspring_lambda = int(es_settings["offspring_lambda"])
    sigma_low = float(es_settings.get("sigma_low", 0.02))
    sigma_high = float(es_settings.get("sigma_high", 0.2))
    sigma_decay_rate = float(es_settings.get("sigma_decay_rate", 0.05))
    alpha = float(es_settings["alpha"])
    use_grad_step = bool(es_settings.get("use_grad_step", True))
    num_generations = int(es_settings["num_generations"])

    num_genes = model.num_genes
    result = ESTrainingResult()

    for gen_idx in range(num_generations):
        parents = region_memory.sample_parents(population_size, rng)
        if not parents:
            break
            
        parents.sort(key=lambda p: float(p.fitness), reverse=True)

        accepted_this_gen = 0
        evaluated_this_gen = 0

        for rank, parent in enumerate(parents):
            parent_w = parent.weights
            parent_fit = float(parent.fitness)
            
            visits = region_memory.get_visits(parent.region_key)
            parent_sigma = sigma_high / (1.0 + sigma_decay_rate * visits)

            noises = []
            sampled_fitness = []

            for _ in range(offspring_lambda):
                noise = rng.normal(0.0, parent_sigma, size=num_genes).astype(np.float32)
                candidate_w = (parent_w + noise).astype(np.float32)

                fit, loss, regions = _evaluate_solution(
                    model, candidate_w, x_train, y_train, loss_name, use_inversion, use_regions=use_regions
                )
                evaluated_this_gen += 1

                noises.append(noise)
                sampled_fitness.append(fit)

                if region_memory.add_candidate(candidate_w, fit, loss, regions):
                    accepted_this_gen += 1

            if use_grad_step and noises:
                grad_est = np.zeros(num_genes, dtype=np.float32)
                centered = np.array(sampled_fitness, dtype=np.float32) - parent_fit

                for coeff, eps in zip(centered, noises):
                    grad_est += coeff * eps

                grad_est /= (len(noises) * max(parent_sigma, 1e-12))
                guided_w = (parent_w + alpha * grad_est).astype(np.float32)

                fit, loss, regions = _evaluate_solution(
                    model, guided_w, x_train, y_train, loss_name, use_inversion, use_regions=use_regions
                )
                evaluated_this_gen += 1

                if region_memory.add_candidate(guided_w, fit, loss, regions):
                    accepted_this_gen += 1

        result.accepted_total += accepted_this_gen
        result.evaluated_total += evaluated_this_gen

        best = region_memory.get_best()
        if best is None:
            break

        result.loss_history.append(best.loss)

        if hasattr(model, "delta"):
            model.delta *= delta_decay_factor

        if on_generation is not None:
            on_generation(gen_idx, best, accepted_this_gen, evaluated_this_gen, region_memory)

    best = region_memory.get_best()
    if best is not None:
        result.best_fitness = float(best.fitness)
        result.best_loss = float(best.loss)
        result.best_weights = np.array(best.weights, copy=True)

    result.regions_discovered = region_memory.total_regions_discovered
    result.stored_candidates = len(region_memory)

    return result
