import os
import sys

import csv
import itertools
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.models.dnn import DNN
from src.config import base_config as config

from src.data.generators import (
    create_model_with_shared_init,
    generate_data,
)
from src.algorithms.backprop.trainer import backprop
from src.analysis.torch_analyzer import analyze_and_plot_regions

from sklearn.preprocessing import StandardScaler



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiments():
    # Define the configurations to test
    paper_functions = config.PAPER_FUNCTIONS
    basic_test_functions = config.BASIC_TEST_FUNCTIONS
    test_functions = config.TEST_FUNCTIONS

    NUM_EPOCHS = config.BACKPROP_SETTINGS["num_epochs"]
    BATCH_SIZE = config.BACKPROP_SETTINGS["batch_size"]

    learning_rates = config.BACKPROP_SETTINGS["learning_rates"]
    lambda_regs = config.BACKPROP_SETTINGS["lambda_regs"]
    architectures = config.ARCHITECTURES
    discontinuous_indices = []  # implement later for different discontinuity settings

    Relu_true = False #config.Base_ACTIVATIONS[0]
    top_eps = config.BACKPROP_SETTINGS["top_eps"]
    seeds = config.BACKPROP_SETTINGS["seeds"]

    # Generate all combinations of configurations
    configs = list(itertools.product(
        test_functions, architectures,  learning_rates, lambda_regs
    ))

    # Log file setup
    log_file = f"ReLU_experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    header_written = False

    # Dictionary to store results for plotting
    # Key: (func, tuple(arch), lr, l_reg)
    # Value: list of dicts with keys 'seed', 'dnn_names', 'loss_histories', 'final_losses'
    results_data = {}

    # Run experiments
    for i, (func, arch, lr, l_reg) in enumerate(configs):
        input_size = arch[0]
        hidden_layers = arch[1:]

        # Determine if function is 2D (paper functions)
        is_paper_func = func in paper_functions

        # Skip invalid combinations
        if is_paper_func:
            if input_size != 2:              #Learning rate
                continue
        if not is_paper_func:
            if lr < 0.01 or input_size != 1:   #Learning rate smaller than 0.01 does not make sense for basic test functions
                continue

        config_key = (func, tuple(arch), lr, l_reg)
        if config_key not in results_data:
            results_data[config_key] = []

        if is_paper_func:
            discontinuous_indices = [2,3]
        else:
            discontinuous_indices = [0]

        # Generate data with fixed seed (same for all 5 seed runs of this config)
        if is_paper_func:
            x_min, x_max = -2, 2
        else:
            x_min, x_max = -10, 10

        np.random.seed(42) # Fixed seed for data generation
        x_samples, y_samples = generate_data(1000, function=func, input_dim=input_size, x_min=x_min, x_max=x_max)

        # Scale input data
        scaler = StandardScaler()
        x_samples_scaled = torch.tensor(scaler.fit_transform(x_samples), dtype=torch.float32)

        for seed in seeds:
            config_name = f"f_{func}_arch_{'_'.join(map(str, arch))}_lr_{lr}_lreg_{l_reg}_seed_{seed}"
            print(f"Running experiment {i+1}/{len(configs)} (Seed {seed}): {config_name}________time: {datetime.now().strftime('%H:%M:%S')}")

            # Create models for comparison
            nn = create_model_with_shared_init(
                DNN,
                input_size=input_size,
                hidden_layers=hidden_layers,
                output_size=1,
                discontinuous_indices=[],  # No discontinuity
                Relu_true=Relu_true,
                name="NN",
                shared_seed=seed,
            )

            epsilon_init = torch.zeros(hidden_layers[discontinuous_indices[0]])   # EPSILON INIT CONSTANT DIMENSION


            dnn_relu_no_ste = create_model_with_shared_init(
                DNN,
                input_size=input_size,
                hidden_layers=hidden_layers,
                output_size=1,
                discontinuous_indices=discontinuous_indices,
                Relu_true=Relu_true,
                STE_true=False,
                name="DNN-ReLU-noSTE",
                Disc_Activation='disc',
                epsilon_init=epsilon_init,
                shared_seed=seed,
            )
            dnn_relu_ste = create_model_with_shared_init(
                DNN,
                input_size=input_size,
                hidden_layers=hidden_layers,
                output_size=1,
                discontinuous_indices=discontinuous_indices,
                Relu_true=Relu_true,
                STE_true=True,
                name="DNN-ReLU-STE",
                Disc_Activation='disc',
                epsilon_init=epsilon_init,
                shared_seed=seed,
            )
            dnn_pwlu = create_model_with_shared_init(
                DNN,
                input_size=input_size,
                hidden_layers=hidden_layers,
                output_size=1,
                discontinuous_indices=discontinuous_indices,
                Relu_true=Relu_true,
                STE_true=False, # STE is not used by PWLU
                delta=1,
                name="DNN-PWLU",
                Disc_Activation="pwlu",
                epsilon_init=epsilon_init,
                shared_seed=seed,
            )

            # Find comparable architecture
            target_params = count_parameters(dnn_pwlu)
            best_arch = list(hidden_layers)
            min_diff = float('inf')

            # Search for architecture with similar parameter count by adjusting first layer
            original_l0 = hidden_layers[0]
            for l0_cand in range(original_l0, original_l0 + 10):
                temp_arch = [l0_cand] + hidden_layers[1:]
                # Create temporary model to count parameters (no seed needed for counting)
                temp_model = DNN(input_size, temp_arch, 1, discontinuous_indices=[], Relu_true=Relu_true)
                p_count = count_parameters(temp_model)
                diff = abs(p_count - target_params)
                if diff < min_diff:
                    min_diff = diff
                    best_arch = temp_arch

            nn_comparable = create_model_with_shared_init(
                DNN,
                input_size=input_size,
                hidden_layers=best_arch,
                output_size=1,
                discontinuous_indices=[],
                Relu_true=Relu_true,
                name=f"NN-Comparable",
                shared_seed=seed,
            )

            p_nn = count_parameters(nn)
            p_nn_comp = count_parameters(nn_comparable)
            p_pwlu = count_parameters(dnn_pwlu)
            print(f"  Params: NN={p_nn}, PWLU={p_pwlu}, Comparable={p_nn_comp} (Arch: {best_arch})")



            dnn_list = [nn_comparable, dnn_relu_ste, dnn_relu_no_ste, dnn_pwlu]#[nn, nn_comparable, dnn_relu_no_ste, dnn_pwlu]

            if not header_written:
                log_header = [
                    "config_name",
                    "function",
                    "architecture",
                    # "learning_rate",
                    # "lambda_reg",
                    "seed"
                ] + [f"loss_{m.name}" for m in dnn_list] #+ [
                    # "params_nn",
                    # "params_nn_comparable",
                    # "params_pwlu"
                #]
                with open(log_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(log_header)
                header_written = True


            # Create result directory
            import os
            if not os.path.exists('Pre_and_Post_Training_Discontinuity_Regions'):
                os.makedirs('Pre_and_Post_Training_Discontinuity_Regions')

            # Analyze and plot DNN regions before training
            if dnn_relu_ste in dnn_list: analyze_and_plot_regions(dnn_relu_ste, X_data= x_samples_scaled, Y_samples=y_samples, save_path=f"Pre_and_Post_Training_Discontinuity_Regions/regions_{config_name}_dnn_ste_pre_training.png", show_plot=True, x_min=x_min, x_max=x_max, top_eps=top_eps, scaler=scaler)
            if dnn_relu_no_ste in dnn_list: analyze_and_plot_regions(dnn_relu_no_ste, X_data= x_samples_scaled, Y_samples=y_samples, save_path=f"Pre_and_Post_Training_Discontinuity_Regions/regions_{config_name}_dnn_pre_training.png", show_plot=False, x_min= x_min, x_max=x_max, top_eps=top_eps, scaler=scaler)
            if dnn_pwlu in dnn_list: analyze_and_plot_regions(dnn_pwlu, X_data= x_samples_scaled, Y_samples=y_samples, save_path=f"Pre_and_Post_Training_Discontinuity_Regions/regions_{config_name}_pwlu_pre_training.png", show_plot=False, x_min=x_min, x_max=x_max, top_eps=top_eps, scaler=scaler)

            # Train models
            _, final_losses, loss_histories = backprop(
                dnn_list,
                x_samples_scaled,
                y_samples,
                num_epochs=NUM_EPOCHS,
                learning_rate=lr,
                delta_factor=0.999,
                lambda_reg=l_reg,
                config_name=config_name,
                scaler=scaler,
                batch_size=BATCH_SIZE
            )

            # Store results for post-experiment plotting
            results_data[config_key].append({
                'seed': seed,
                'dnn_names': [m.name for m in dnn_list],
                'loss_histories': loss_histories,
                'final_losses': final_losses
            })

            if dnn_relu_ste in dnn_list: analyze_and_plot_regions(dnn_relu_ste, X_data= x_samples_scaled, Y_samples=y_samples, save_path=f"Pre_and_Post_Training_Discontinuity_Regions/regions_{config_name}_dnn_ste_post_training.png", show_plot=False, x_min=x_min, x_max=x_max, top_eps=top_eps, scaler=scaler)
            if dnn_relu_no_ste in dnn_list: analyze_and_plot_regions(dnn_relu_no_ste, X_data= x_samples_scaled, Y_samples=y_samples, save_path=f"Pre_and_Post_Training_Discontinuity_Regions/regions_{config_name}_dnn_post_training.png", show_plot=False, x_min=x_min, x_max=x_max, top_eps=top_eps, scaler=scaler)
            if dnn_pwlu in dnn_list: analyze_and_plot_regions(dnn_pwlu, X_data= x_samples_scaled, Y_samples=y_samples, save_path=f"Pre_and_Post_Training_Discontinuity_Regions/regions_{config_name}_pwlu_post_training.png", show_plot=False, x_min=x_min, x_max=x_max, top_eps=top_eps, scaler=scaler)

            # Log results
            log_row = [
                config_name,
                func,
                arch,
                # lr,
                # l_reg,
                seed
            ] + final_losses #+ [
                # p_nn,
                # p_nn_comp,
                # p_pwlu
            #]
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(log_row)

    # Plot aggregated results
    plot_aggregated_results(results_data)

def plot_aggregated_results(results_data):
    # Ensure directory exists
    output_dir = "aggregated_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for config_key, runs in results_data.items():
        func, arch, lr, l_reg = config_key
        config_str = f"{func}_arch{'_'.join(map(str, arch))}_lr{lr}_lreg{l_reg}"

        if not runs:
            continue

        # Get list of model names from the first run (assuming same models for all seeds)
        dnn_names = runs[0]['dnn_names']

        # Plot 1: Loss histories combined of all seeds for the same model
        for model_idx, model_name in enumerate(dnn_names):
            plt.figure(figsize=(10, 6))
            for run in runs:
                seed = run['seed']
                loss_history = run['loss_histories'][model_idx]
                plt.plot(loss_history, label=f'Seed {seed}')

            plt.title(f'Loss History per Seed - {model_name}\n{config_str}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{output_dir}/loss_history_seeds_{model_name}_{config_str}.png")
            plt.close()

        # Plot 2: Loss histories of all models (best seed for each)
        plt.figure(figsize=(10, 6))
        for model_idx, model_name in enumerate(dnn_names):
            # Find run with best final loss for this model
            best_run = min(runs, key=lambda r: r['final_losses'][model_idx])
            best_seed = best_run['seed']
            best_loss_history = best_run['loss_histories'][model_idx]
            best_final_loss = best_run['final_losses'][model_idx]

            plt.plot(best_loss_history, label=f'{model_name} (Best Seed {best_seed}, Final Loss {best_final_loss:.4e})')

        plt.title(f'Loss History Best Seeds Comparison\n{config_str}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/loss_history_comparison_{config_str}.png")
        plt.close()

if __name__ == "__main__":
    run_experiments()
