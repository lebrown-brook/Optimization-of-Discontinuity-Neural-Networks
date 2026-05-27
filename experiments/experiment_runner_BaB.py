import itertools
import numpy as np
import torch
import pandas as pd

import sys
import os

from src.config import base_config as config
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from src.data.generators import generate_data
from src.models.gadnn import GADNN
from src.algorithms.bab.trainer import BaBTrainer
from src.visualization.plotter import plot_ga_result

def run_bab_experiments(enable_plotting=True):
    # Configuration
    seed = config.SEED

    # Functions to test
    paper_functions = config.PAPER_FUNCTIONS
    test_functions = config.TEST_FUNCTIONS

    # Architectures
    architectures = config.ARCHITECTURES

    loss_functions = config.LOSS_FUNCTIONS
    base_activations = config.BASE_ACTIVATIONS
    use_inversion = config.USE_INVERSION
    delta_decay_factor = config.DELTA_DECAY_FACTOR

    # BaB Settings
    bab_settings = config.BAB_SETTINGS
    ga_settings = config.GA_SETTINGS
    ga_settings.update({
        "num_parents_mating": 10,
        "keep_elitism": 2,
        "mutation_probability": [0.1, 0.02],
        "parent_selection_type": "sss"
    })
    # Generate combinations
    configs = list(itertools.product(test_functions, architectures, loss_functions, base_activations))

    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"BaB_experiment_log_{timestamp}.csv"

    print(f"Starting BaB experiments. Log file: {log_filename}")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for i, (func_name, arch, loss_name, relu_true) in enumerate(configs):
        # Determine input dimension
        input_dim = 2 if func_name in paper_functions else 1

        if arch[0] != input_dim:
            continue

        print(f"Running experiment {i+1}/{len(configs)}: Func={func_name}, Arch={arch}, Loss={loss_name}")

        # Generate Data
        x_range = 2.0 if func_name in paper_functions else 10.0
        X_train_raw, y_train = generate_data(num_samples=200, function=func_name, input_dim=input_dim, x_min=-x_range, x_max=x_range)

        # Scale Input Data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw.numpy())
        X_train = torch.tensor(X_train_scaled, dtype=torch.float32)

        # Determine discontinuous indices
        if func_name in paper_functions:
            base_discontinuous_indices = [2, 3]
        else:
            base_discontinuous_indices = [0]

        # Define variations (BaB is usually just one variant, but we can test different activations)
        # We use the config variations but BaB replaces the "flip_aware" logic with its own branching.
        # So we filter for "flip_aware" being False or ignore it?
        # Actually BaB IS the "flip aware" strategy. So we just run it for the activation types.

        # Taking base variations from config
        all_variations = config.get_base_variations(base_discontinuous_indices)
        # Filter: unique prefixes (removing explicit flip_aware duplicates if they exist in config or treating them)
        # Assuming we just want to test "pwlu" and "disc" (if enabled) with BaB.
        # Let's take variations that are NOT explicitly flip_aware in the list,
        # because BaB logic replaces the simple flip_aware GA.
        variations = [v for v in all_variations if not v.get("flip_aware")]

        combined_histories = {}

        for var in variations:
            prefix = f"BaB_{var['prefix']}" # Add BaB prefix
            activation = var["activation"]
            indices = var["indices"]
            div = bab_settings.get("diversity_threshold")
            config_name = f"diversity_{div}_{prefix}_{func_name}_arch_{'_'.join(map(str, arch))}_{loss_name}"
            print(f"  Running variation: {config_name}")

            # Model Arguments
            model_args = {
                "input_size": arch[0],
                "hidden_layers": arch[1:],
                "output_size": 1,
                "discontinuous_indices": indices,
                "Disc_Activation": activation,
                "Relu_true": relu_true
            }

            # Create a template model for gene count
            model_dummy = GADNN(**model_args)
            num_genes = model_dummy.num_genes

            # Prepare Fitness Function (Standard MSE)
            # BaBTrainer instantiates new GAs, so we need to pass a fitness function that works with the model.
            # But GADNN instance is needed inside fitness func?
            # BaBTrainer creates `model_template`.
            # PyGAD fitness signature: func(ga_instance, solution, solution_idx)
            # The fitness function needs access to the model structure (GADNN forward) and Data.
            # We can define it here using closure over X_train, y_train and a (shared) model instance?
            # BaBTrainer has `self.model_template`. But `fitness_func` is passed in `ga_settings`.
            # We need a way to pass the model to fitness_func.
            # Usually we define fitness_func globally or here where 'model' is available.
            # But BaBTrainer manages its own `model_template`.

            # Solution: We define a wrapper class or function that uses a freshly instantiated model or a shared one.
            # Since GADNN forward is stateless regarding weights (we pass weights), we can use one shared instance.

            def fitness_func_factory(model_instance):
                def fitness_func(ga_instance, solution, solution_idx):
                    with torch.no_grad():
                        predictions = model_instance.forward(solution, X_train)
                        # Nan check
                        if torch.isnan(predictions).any():
                            return -1e9

                    if loss_name == "mse":
                        loss = torch.nn.functional.mse_loss(predictions, y_train).item()
                    elif loss_name == "mae":
                        loss = torch.nn.functional.l1_loss(predictions, y_train).item()
                    else:
                        loss = 1e9

                    fitness = 1.0 / (loss + 1e-8) if use_inversion else -loss
                    return fitness
                return fitness_func

            # Fitness to Loss conversion for logging
            def fitness_to_loss(f):
                 if use_inversion:
                      return 1.0 / (f + 1e-8)
                 else:
                      return -f

            # Create the BaB Trainer
            # We create a shared model instance for the fitness function
            shared_model = GADNN(**model_args)

            current_ga_settings = ga_settings.copy()
            current_ga_settings["num_genes"] = num_genes
            current_ga_settings["fitness_func"] = fitness_func_factory(shared_model)

        # Remove on_generation/on_fitness from settings if they rely on specific GA instance logic
            # BaB might handle generation logging differently.
            # We can add a simple on_generation to log progress of branches?
            # Let's keep it simple.
            if "on_generation" in current_ga_settings:
                del current_ga_settings["on_generation"]
            if "on_fitness" in current_ga_settings:
                 del current_ga_settings["on_fitness"]

            trainer = BaBTrainer(
                model_class=GADNN,
                model_args=model_args,
                X_train=X_train,
                y_train=y_train,
                ga_settings=current_ga_settings,
                bab_settings=bab_settings,
                fitness_to_loss=fitness_to_loss,
                config_name=config_name,
                shared_model=shared_model,
                delta_decay=delta_decay_factor,
                enable_plotting=enable_plotting
            )

            # Train
            best_sol, best_fit, final_ga_inst, history_log = trainer.train()

            # Final Loss
            final_loss = 1.0 / best_fit - 1e-8 if use_inversion else -best_fit
            print(f"  Result: Best Fitness: {best_fit:.6f}, Final Loss: {final_loss:.6f}")

            # Logging
            results.append({
                "function": func_name,
                "architecture": str(arch),
                "loss_function": loss_name,
                "activation": activation,
                "variation": prefix,
                "best_fitness": best_fit,
                "final_loss": final_loss,
                "timestamp": datetime.now().isoformat()
            })

            pd.DataFrame(results).to_csv(log_filename, index=False)

            # Plot Result (using the best found solution)
            plot_ga_result(shared_model, best_sol, X_train, y_train, bab_settings["total_generations"], config_name, save_dir="PyGAD_BAB_Plots")

            # Plot BaB Loss History (Advanced)
            from src.visualization.plotter import plot_bab_loss_history
            plot_bab_loss_history(
                history_logs=history_log,
                generations_per_iter=bab_settings["generations_per_bab_iteration"],
                config_name=config_name,
                use_inversion=use_inversion
            )

            # Store history (extract top 3 from trainer) (Backwards Compatibility / Debugging)
            # Find active branches in the last iteration
            if trainer.branch_ids:
                # Get the last iteration's branch IDs
                # Since branch_ids are reset every iteration in BaBTrainer, self.branch_ids holds the current (last) ones.
                final_branches = []
                for b_id in trainer.branch_ids:
                     hist = trainer.loss_histories.get(b_id, [])
                     if hist:
                         final_val = hist[-1]
                         final_branches.append((final_val, b_id, hist))

                # Sort by final loss (ascending)
                final_branches.sort(key=lambda x: x[0])

                # Take top 3
                top3 = final_branches[:3]

                for i, (val, b_id, hist) in enumerate(top3):
                     combined_histories[f"{prefix}_rank{i+1}"] = hist
            else:
                 # Fallback if no branches (shouldn't happen)
                 pass

if __name__ == "__main__":
    run_bab_experiments()
