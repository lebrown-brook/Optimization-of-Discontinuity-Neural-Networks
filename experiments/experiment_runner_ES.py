import itertools
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import sys 
import os 

from src.config import base_config as config
from src.data.generators import generate_data
from src.algorithms.es.trainer import initialize_region_memory, train_region_aware_es
from src.models.gadnn import GADNN
from src.algorithms.es.region_memory import RegionMemoryES
from src.visualization.plotter import (
    analyze_and_plot_gadnn_regions,
    plot_combined_loss_history,
    plot_ga_result,
)


def run_es_experiments(es_settings_override=None, save_dir="loss_history_ES", enable_plotting=True, filename_prefix=""):
    default_seed = config.SEED

    paper_functions = config.PAPER_FUNCTIONS
    test_functions = config.TEST_FUNCTIONS
    architectures = config.ARCHITECTURES
    loss_functions = config.LOSS_FUNCTIONS
    base_activations = config.BASE_ACTIVATIONS
    use_inversion = config.USE_INVERSION
    delta_decay_factor = config.DELTA_DECAY_FACTOR

    es_settings = config.ES_SETTINGS.copy()
    if es_settings_override:
        es_settings.update(es_settings_override)

    run_seed = int(es_settings.get("random_seed", default_seed))

    configs = list(itertools.product(test_functions, architectures, loss_functions, base_activations))

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"ES_Experiments/ES_experiment_log_{timestamp}.csv"

    print(f"Starting ES experiments. Log file: {log_filename}")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    for i, (func_name, arch, loss_name, relu_true) in enumerate(configs):
        input_dim = 2 if func_name in paper_functions else 1
        if arch[0] != input_dim:
            continue

        print(f"Running experiment {i + 1}/{len(configs)}: Func={func_name}, Arch={arch}, Loss={loss_name}")

        x_range = 2.0 if func_name in paper_functions else 10.0
        X_train_raw, y_train = generate_data(
            num_samples=200,
            function=func_name,
            input_dim=input_dim,
            x_min=-x_range,
            x_max=x_range,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw.numpy())
        X_train = torch.tensor(X_train_scaled, dtype=torch.float32)

        if func_name in paper_functions:
            base_discontinuous_indices = [2, 3]
        else:
            base_discontinuous_indices = [0]

        variations = config.get_base_variations(base_discontinuous_indices)
        combined_histories = {}

        for var in variations:
            prefix = var["prefix"]
            activation = var["activation"]
            indices = var["indices"]

            config_name = f"{prefix}_{func_name}_arch_{'_'.join(map(str, arch))}_{loss_name}"
            if filename_prefix:
                config_name = f"{filename_prefix}{config_name}"
            print(f"  Running variation: {prefix}_{config_name}")

            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            random.seed(run_seed)
            rng = np.random.default_rng(run_seed)

            model = GADNN(
                arch[0],
                arch[1:],
                1,
                discontinuous_indices=indices,
                Disc_Activation=activation,
                Relu_true=relu_true,
            )
            num_genes = model.num_genes

            population_size = int(es_settings["population_size_P"])
            
            use_regions = bool(indices)
            if not use_regions:
                mu_per_region = population_size
            else:
                mu_per_region = int(es_settings["mu_per_region"])
                
            num_generations = int(es_settings["num_generations"])
            init_low = float(es_settings["init_range_low"])
            init_high = float(es_settings["init_range_high"])
            max_regions = int(es_settings.get("max_regions", 1000))

            region_memory = RegionMemoryES(mu_per_region=mu_per_region, max_regions=max_regions)

            initialize_region_memory(
                model=model,
                x_train=X_train,
                y_train=y_train,
                loss_name=loss_name,
                use_inversion=use_inversion,
                region_memory=region_memory,
                rng=rng,
                population_size=population_size,
                init_low=init_low,
                init_high=init_high,
                use_regions=use_regions,
            )

            x_min_scaled = X_train.min().item()
            x_max_scaled = X_train.max().item()
            margin = (x_max_scaled - x_min_scaled) * 0.1
            x_min_plot = x_min_scaled - margin
            x_max_plot = x_max_scaled + margin

            if indices and enable_plotting:
                best_init = region_memory.get_best()
                if best_init is not None:
                    analyze_and_plot_gadnn_regions(
                        model,
                        best_init.weights,
                        X_data=X_train,
                        Y_samples=y_train,
                        save_path=f"PyGAD_Pre_and_Post_Training_Discontinuity_Regions/regions_{config_name}_pre_training.png",
                        x_min=x_min_plot,
                        x_max=x_max_plot,
                    )

            def on_generation(gen_idx, best, accepted_this_gen, evaluated_this_gen, current_memory):
                if (gen_idx + 1) % 10 == 0 or gen_idx == 0:
                    print(
                        f"  Gen {gen_idx + 1}: Best Loss = {best.loss:.6f}, Regions = {current_memory.num_regions}, "
                        f"Stored = {len(current_memory)}, Accepted = {accepted_this_gen}/{evaluated_this_gen}, Total Seen = {current_memory.total_regions_discovered}"
                    )
                    if enable_plotting:
                        plot_ga_result(model, best.weights, X_train, y_train, gen_idx + 1, config_name)

            training_result = train_region_aware_es(
                model=model,
                x_train=X_train,
                y_train=y_train,
                loss_name=loss_name,
                use_inversion=use_inversion,
                delta_decay_factor=delta_decay_factor,
                region_memory=region_memory,
                rng=rng,
                es_settings=es_settings,
                on_generation=on_generation,
                use_regions=use_regions,
            )

            combined_histories[prefix] = training_result.loss_history

            final_loss = training_result.best_loss
            best_fitness = training_result.best_fitness
            best_weights = training_result.best_weights

            if indices and enable_plotting and best_weights is not None:
                analyze_and_plot_gadnn_regions(
                    model,
                    best_weights,
                    X_data=X_train,
                    Y_samples=y_train,
                    save_path=f"PyGAD_Pre_and_Post_Training_Discontinuity_Regions/regions_{config_name}_post_training.png",
                    x_min=x_min_plot,
                    x_max=x_max_plot,
                )

            best_fit_text = f"{best_fitness:.6f}" if best_fitness is not None else "N/A"
            final_loss_text = f"{final_loss:.6f}" if final_loss is not None else "N/A"
            print(f"  Result: Best Fitness: {best_fit_text}, Final Loss: {final_loss_text}")

            results.append(
                {
                    "function": func_name,
                    "architecture": str(arch),
                    "loss_function": loss_name,
                    "activation": activation,
                    "variation": prefix,
                    "best_fitness": best_fitness,
                    "final_loss": final_loss,
                    "generations": num_generations,
                    "pop_size": population_size,
                    "regions_discovered": training_result.regions_discovered,
                    "stored_candidates": training_result.stored_candidates,
                    "accepted_total": training_result.accepted_total,
                    "evaluated_total": training_result.evaluated_total,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            pd.DataFrame(results).to_csv(log_filename, index=False)

        config_base_name = f"{func_name}_arch_{'_'.join(map(str, arch))}_{loss_name}"
        if filename_prefix:
            config_base_name = f"{filename_prefix}{config_base_name}"

        plot_combined_loss_history(combined_histories, config_base_name, save_dir=save_dir)

    return results


if __name__ == "__main__":
    run_es_experiments(filename_prefix="2d_tests")
