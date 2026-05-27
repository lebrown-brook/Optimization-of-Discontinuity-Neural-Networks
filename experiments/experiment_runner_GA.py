import itertools
import random
import numpy as np
import torch
import pandas as pd
import pygad
import sys 
import os

from src.config import base_config as config
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from src.data.generators import generate_data
from src.models.gadnn import GADNN
from src.visualization.plotter import plot_ga_result, analyze_and_plot_gadnn_regions, plot_combined_loss_history
from src.algorithms.ga import ga_flip_aware

def run_ga_experiments(ga_settings_override=None, save_dir="loss_history_GA", enable_plotting=True, filename_prefix=""):
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
    # Alpha values to test for flip-aware variations
    flip_alphas = config.FLIP_ALPHAS
    # --- GA Settings ---
    ga_settings = config.GA_SETTINGS.copy()
    if ga_settings_override:
        ga_settings.update(ga_settings_override)

    # Generate combinations
    configs = list(itertools.product(test_functions, architectures, loss_functions, base_activations))

    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"GA_experiment_log_{timestamp}.csv"

    print(f"Starting GA experiments. Log file: {log_filename}")

    # Set random seed for reproducibility


    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for i, (func_name, arch, loss_name, relu_true) in enumerate(configs):
        # Determine input dimension based on function type
        if func_name in paper_functions:
            input_dim = 2
        else:
            input_dim = 1

        # Check if architecture matches input dimension
        if arch[0] != input_dim:
            # print(f"Skipping architecture {arch} for function {func_name} (input dim mismatch)")
            continue

        print(f"Running experiment {i+1}/{len(configs)}: Func={func_name}, Arch={arch}, Loss={loss_name}")

        # Generate Data
        # Using generate_data from src.data.generators
        # Note: generate_data returns torch tensors
        # We use a wider range [-2, 2] for paper functions as seen in Testfunctions_paper.py heatmap generation
        x_range = 2.0 if func_name in paper_functions else 10.0
        X_train_raw, y_train = generate_data(num_samples=200, function=func_name, input_dim=input_dim, x_min=-x_range, x_max=x_range)

        # Scale Input Data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw.numpy())
        X_train = torch.tensor(X_train_scaled, dtype=torch.float32)

        # Determine discontinuous indices based on function type (similar to experiment_runner.py)
        if func_name in paper_functions:
            base_discontinuous_indices = [2, 3] # Example indices, adjust as needed or make dynamic
        else:
            base_discontinuous_indices = [0]

        # Define variations
        base_variations = config.get_base_variations(base_discontinuous_indices)



        variations = []
        for var in base_variations:
            # If flip_aware is enabled, we create a specific variation for each alpha value.
            # The original "template" variation (without explicit alpha) is NOT added, preventing duplication of the default alpha.
            if var.get("flip_aware"):
                for alpha in flip_alphas:
                    new_var = var.copy()
                    new_var["alpha"] = alpha
                    variations.append(new_var)
            else:
                variations.append(var)

        combined_histories = {}

        for var in variations:
            prefix = var["prefix"]
            activation = var["activation"]
            indices = var["indices"]
            is_flip_aware = var.get("flip_aware", False)
            current_alpha = var.get("alpha", 0.0)

            if is_flip_aware:
                ga_flip_aware.reset_flip_aware_state()
                # Update prefix to include alpha, distinguishing the results
                prefix = f"{prefix}_alpha_{current_alpha}"

            config_name = f"{prefix}_{func_name}_arch_{'_'.join(map(str, arch))}_{loss_name}"
            if filename_prefix:
                config_name = f"{filename_prefix}{config_name}"
            print(f"  Running variation: {prefix}_{config_name}")

            # GADNN setup
            input_size = arch[0]
            hidden_layers = arch[1:]
            output_size = 1

            # Create a template model to get weight dimensions and structure
            model = GADNN(
                input_size,
                hidden_layers,
                output_size,
                discontinuous_indices=indices,
                Disc_Activation=activation,
                Relu_true=relu_true
            )
            num_genes = model.num_genes
            if is_flip_aware:
                num_genes += ga_flip_aware.ID_GENES_COUNT

            loss_history = []

            # Define Fitness Function
            def fitness_func(ga_instance, solution, solution_idx):
                # Forward pass with the solution (weights)
                # GADNN.forward takes (weights, input)

                # Note: X_train is already a tensor from generate_data

                # Slice weights if flip_aware (remove ID genes for model, but keep solution for ID extraction)
                # GADNN forward handles extra weights gracefully usually, but let's be strict if needed.
                # Actually ga_flip_aware.ID_GENES_COUNT are at the end.
                weights_for_model = solution[:-ga_flip_aware.ID_GENES_COUNT] if is_flip_aware else solution

                with torch.no_grad():
                    if is_flip_aware:
                        predictions, regions = model.forward(weights_for_model, X_train, return_regions=True)

                        if torch.isnan(predictions).any():
                             print(f"[CRITICAL] NaNs in predictions at Gen {ga_instance.generations_completed}, Sol {solution_idx}")
                             return -1e9 # Strong penalty

                        # Store regions logic moved to on_fitness to capture whole population (including elites)
                        # ga_flip_aware.store_child_regions(solution_idx, regions)
                    else:
                        predictions = model.forward(weights_for_model, X_train)
                        if torch.isnan(predictions).any():
                             print(f"[CRITICAL] NaNs in predictions at Gen {ga_instance.generations_completed}, Sol {solution_idx}")
                             return -1e9

                # Calculate Loss
                if loss_name == "mse":
                    loss = torch.nn.functional.mse_loss(predictions, y_train).item()
                elif loss_name == "mae":
                    loss = torch.nn.functional.l1_loss(predictions, y_train).item()
                else:
                    print("Not implemented loss function:")
                    loss = 1e9

                # Fitness is inverse of loss.
                fitness = 1.0 / (loss + 1e-8)  if use_inversion else -loss

                if is_flip_aware:
                    # Calculate metric using the captured regions and the full solution (containing IDs)
                    # Use it as a reward for novelty (flips), but only if the model is decent enough

                    if loss < 4.0:
                        flip_metric = ga_flip_aware.calculate_flip_metric(regions, solution, ga_settings["sol_per_pop"], alpha=current_alpha)
                        fitness += flip_metric

                return fitness


            def on_generation(ga_instance):


                best_sol, best_fit, _ = ga_instance.best_solution()

                if not is_flip_aware:
                    loss = 1.0 / best_fit - 1e-8 if use_inversion else -best_fit  # needs to be adjusted for flip_metric  if is_flip_aware
                    loss_history.append(loss)
                else:
                    #calculate regression loss without flip metric for logging
                    predictions = model.forward(best_sol[:-ga_flip_aware.ID_GENES_COUNT], X_train)
                    if loss_name == "mse":
                        loss = torch.nn.functional.mse_loss(predictions, y_train).item()
                    elif loss_name == "mae":
                        loss = torch.nn.functional.l1_loss(predictions, y_train).item()
                    else:
                        loss = 1e9
                    loss_history.append(loss)


                # Decay delta parameter
                if hasattr(model, 'delta'):
                    model.delta *= delta_decay_factor

                if ga_instance.generations_completed % 10 == 0 or ga_instance.generations_completed == 1 :
                    print(f"  Gen {ga_instance.generations_completed}: Best Loss = {loss:.6f}")
                    #print whole population solution values for debugging

                    # Pass raw X_train for plotting to see original scale if needed, or scaled.
                    # plot_ga_result expects X_train. If we pass scaled X_train, plots will be in scaled domain.
                    # To plot in original domain, we might need to inverse transform inside plot_ga_result or pass scaler.
                    # For now, let's plot in scaled domain or pass raw and let model predict on scaled grid?
                    # The model is trained on scaled data. So it expects scaled input.
                    # plot_ga_result generates a grid based on input range.
                    # If we pass X_train (scaled), it generates scaled grid, predicts, and plots.
                    # This is fine for visualization of fit.
                    # When flip aware, best_sol has extra genes. GADNN usually ignores them if not stripped,
                    # but plot_ga_result implementation passes solution to model.forward.
                    # Since GADNN parses weights by slicing, extra genes at end are ignored automatically.
                    if enable_plotting:
                        plot_ga_result(model, best_sol, X_train, y_train, ga_instance.generations_completed, config_name)

            # Create GA Instance
            # Prepare settings (start with base config)
            current_ga_settings = ga_settings.copy()

            # --- OVERRIDE SETTINGS BASED ON MODEL TYPE ---
            # Only apply defaults if not running an optimization (no overrides provided)
            # Continuous: Tournament Selection, slightly lower mutation
            # Disc/PWLU: SSS Selection, higher mutation
            if ga_settings_override is None:
                if not indices: # Continuous Model (empty indices list)
                    current_ga_settings.update({
                        "num_parents_mating": 10,
                        "keep_elitism": 2,
                        "mutation_probability": [0.05, 0.02],
                        "parent_selection_type": "tournament",
                        "K_tournament": 40
                    })
                else: # Discontinuous / PWLU Model
                    current_ga_settings.update({
                        "num_parents_mating": 10,
                        "keep_elitism": 2,
                        "mutation_probability": [0.1, 0.02],
                        "parent_selection_type": "sss"
                    })
            # ---------------------------------------------

            # Add dynamic parameters
            current_ga_settings.update({
                "num_genes": num_genes,
                "fitness_func": fitness_func,
                "on_generation": on_generation,
            })

            # Set random seeds
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            # Apply Flip-Aware specific logic overrides
            if is_flip_aware:
                current_ga_settings["on_fitness"] = ga_flip_aware.get_on_fitness_wrapper(model, X_train)
                current_ga_settings["crossover_type"] = ga_flip_aware.get_custom_crossover_func()
                current_ga_settings["mutation_type"] = ga_flip_aware.get_custom_mutation_func()

            ga_instance = pygad.GA(**current_ga_settings)

            # Calculate scaled range for plotting
            x_min_scaled = X_train.min().item()
            x_max_scaled = X_train.max().item()
            margin = (x_max_scaled - x_min_scaled) * 0.1
            x_min_plot = x_min_scaled - margin
            x_max_plot = x_max_scaled + margin

            # Plot Pre-Training Regions (if discontinuous)
            if indices and enable_plotting:
                # Use first individual as initial random weights
                initial_solution = ga_instance.population[0]
                analyze_and_plot_gadnn_regions(
                    model,
                    initial_solution,
                    X_data=X_train,
                    Y_samples=y_train,
                    save_path=f"PyGAD_Pre_and_Post_Training_Discontinuity_Regions/regions_{config_name}_pre_training.png",
                    x_min=x_min_plot, x_max=x_max_plot
                )

            # Run GA
            ga_instance.run()



            # Store history for combined plot
            combined_histories[prefix] = loss_history

            # Get best solution
            solution, solution_fitness, solution_idx = ga_instance.best_solution()

            # Plot Post-Training Regions (if discontinuous)
            if indices and enable_plotting:
                analyze_and_plot_gadnn_regions(
                    model,
                    solution,
                    X_data=X_train,
                    Y_samples=y_train,
                    save_path=f"PyGAD_Pre_and_Post_Training_Discontinuity_Regions/regions_{config_name}_post_training.png",
                    x_min=x_min_plot, x_max=x_max_plot
                )

            # Calculate final loss
            final_loss = loss_history[-1] if loss_history else None


            print(f"  Result: Best Fitness: {solution_fitness:.6f}, Final Loss: {final_loss:.6f}")

            # Log results
            results.append({
                "function": func_name,
                "architecture": str(arch),
                "loss_function": loss_name,
                "activation": activation,
                "variation": prefix,
                "best_fitness": solution_fitness,
                "final_loss": final_loss,
                "generations": ga_settings["num_generations"],
                "pop_size": ga_settings["sol_per_pop"],
                "timestamp": datetime.now().isoformat()
            })

            # Save intermediate results
            df = pd.DataFrame(results)
            df.to_csv(log_filename, index=False)

        # Plot combined loss history for this configuration
        config_base_name = f"{func_name}_arch_{'_'.join(map(str, arch))}_{loss_name}"
        if filename_prefix:
            config_base_name = f"{filename_prefix}{config_base_name}"

        plot_combined_loss_history(combined_histories, config_base_name, save_dir=save_dir)

    return results

if __name__ == "__main__":
    filename_prefix = "adaptive_mutation_"
    run_ga_experiments(filename_prefix = filename_prefix)
