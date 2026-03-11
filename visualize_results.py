import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import torch
import numpy as np
from datetime import datetime
import matplotlib.colors as mcolors

def find_latest_log_file():
    """Finds the most recent experiment log file in the current directory."""
    list_of_files = glob.glob('experiment_log_*.csv')
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def create_result_plots_dir():
    """Creates the 'result_plots' directory if it doesn't exist."""
    if not os.path.exists('result_plots'):
        os.makedirs('result_plots')

def plot_grouped_bar_charts(df):
    """Generates and saves grouped bar charts comparing model performance."""

    # Identify value variables dynamically
    value_vars = [col for col in df.columns if col.startswith('final_loss_')]

    # Reshape the dataframe from wide to long format for easier plotting
    df_melted = pd.melt(df,
                        id_vars=['function', 'architecture', 'learning_rate', 'lambda_reg'],
                        value_vars=value_vars,
                        var_name='model',
                        value_name='loss')

    # Clean up model names for the legend
    df_melted['model'] = df_melted['model'].str.replace('final_loss_', '')

    # Create a plot for each function and architecture combination
    for func in df['function'].unique():
        for arch_tuple in df['architecture'].unique():
            # Use catplot for easy faceting
            g = sns.catplot(
                data=df_melted[(df_melted['function'] == func) & (df_melted['architecture'] == arch_tuple)],
                x='learning_rate',
                y='loss',
                hue='model',
                col='lambda_reg', # Create a separate column for each lambda value
                kind='bar',
                height=6,
                aspect=1.2,
                legend_out=True
            )

            # Set plot titles and labels
            arch_str = '_'.join(map(str, arch_tuple))
            g.fig.suptitle(f'Model Comparison on "{func}" with Architecture {arch_str}', y=1.03)
            g.set_axis_labels("Learning Rate", "Final Loss (log scale)")
            g.set_titles("Lambda = {col_name}")
            g.set(yscale="log") # Use a log scale for better visibility of differences

            # Save the figure
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f'result_plots/bar_chart_{func}_{arch_str}.png')
            plt.close(g.fig)

def plot_heatmaps(df):
    """Generates and saves heatmaps of model performance."""
    model_losses = [col for col in df.columns if col.startswith('final_loss_')]

    for model_loss in model_losses:
        for func in df['function'].unique():
            for arch in df['architecture'].unique():
                subset = df[(df['function'] == func) & (df['architecture'] == arch)]
                pivot_table = subset.pivot_table(
                    values=model_loss,
                    index='learning_rate',
                    columns='lambda_reg'
                )

                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis")
                arch_str = '_'.join(map(str, arch))
                plt.title(f'Heatmap of {model_loss} on {func} with Architecture {arch_str}')
                plt.xlabel('Lambda Regularization')
                plt.ylabel('Learning Rate')
                plt.tight_layout()
                plt.savefig(f'result_plots/heatmap_{func}_{arch_str}_{model_loss}.png')
                plt.close()

def main():
    log_file = find_latest_log_file()
    if log_file is None:
        print("No experiment log file found. Please run 'experiment_runner.py' first.")
        return

    print(f"Visualizing results from: {log_file}")
    df = pd.read_csv(log_file)

    # Convert string representation of lists to tuples, which are hashable
    df['architecture'] = df['architecture'].apply(lambda x: tuple(eval(x)))

    create_result_plots_dir()
    plot_grouped_bar_charts(df)
    plot_heatmaps(df)
    print("Visualizations saved in 'result_plots' directory.")

def plot_ga_result(model, solution, X_train, y_train, generation, config_name, save_dir='PyGAD_plots'):
    """
    Plots the GA result: Model predictions vs True data.
    """
    # 1. Prepare Data
    X_np = X_train.detach().cpu().numpy()
    Y_np = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train

    # 2. Predict
    with torch.no_grad():
        x_min, x_max = X_np.min(), X_np.max()
        x_range = x_max - x_min
        if x_range == 0: x_range = 1
        x_min_plot = x_min - 0.1 * x_range
        x_max_plot = x_max + 0.1 * x_range

        if not os.path.exists(save_dir): os.makedirs(save_dir)

        if X_np.shape[1] == 1:
            resolution = 1000
            X_grid = np.linspace(x_min_plot, x_max_plot, resolution).reshape(-1, 1)
            X_grid_torch = torch.tensor(X_grid, dtype=torch.float32)
            y_pred = model.forward(solution, X_grid_torch).cpu().numpy()

            # 3. Plot
            plt.figure(figsize=(10, 6))
            plt.scatter(X_np, Y_np, label='Training Data', alpha=0.5, color='blue', s=10)
            plt.plot(X_grid, y_pred, label='Model Prediction', color='red', linewidth=2)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'GA Result - {config_name} - Gen {generation}')
            plt.legend()

            save_path = f'{save_dir}/ga_result_{config_name}_gen_{generation}.png'
            plt.savefig(save_path)
            plt.close()
        elif X_np.shape[1] == 2:
            resolution = 100
            x1 = np.linspace(x_min_plot, x_max_plot, resolution)
            x2 = np.linspace(x_min_plot, x_max_plot, resolution)
            X1, X2 = np.meshgrid(x1, x2)
            X_grid = np.column_stack([X1.ravel(), X2.ravel()])
            X_grid_torch = torch.tensor(X_grid, dtype=torch.float32)

            y_pred = model.forward(solution, X_grid_torch).cpu().numpy().reshape(resolution, resolution)

            plt.figure(figsize=(10, 8))
            plt.contourf(X1, X2, y_pred, cmap='viridis', levels=50)
            plt.colorbar(label='Prediction')
            plt.title(f'GA Result 2D - {config_name} - Gen {generation}')

            save_path = f'{save_dir}/ga_result_{config_name}_gen_{generation}.png'
            plt.savefig(save_path)
            plt.close()

def analyze_and_plot_gadnn_regions(model, solution, X_data, Y_samples=None, save_path=None, x_min=None, x_max=None):
    """
    Plots the discontinuity regions of the model.

    TOP EPSILON NOT IMPLEMENTED
    """
    if x_min is None: x_min = -2
    if x_max is None: x_max = 2

    resolution = 500
    if X_data.shape[1] == 1:
        x_grid = np.linspace(x_min, x_max, resolution).reshape(-1, 1)
        x_tensor = torch.tensor(x_grid, dtype=torch.float32)

        # Use return_regions=True to get regions directly
        y_pred, regions = model.forward(solution, x_tensor, return_regions=True)
        # y_pred is tensor, regions is numpy

        # Simple hash for regions
        if regions.shape[1] > 0:
            powers = 2 ** np.arange(regions.shape[1])
            # Cap powers to avoid overflow if many neurons.
            # If > 60 neurons, this overflows int64.
            # Just use pca or just sum for simple viz?
            # Or just unique index map
            # Use safe unique mapping
            # Convert to void type to hash rows
            regions_void = np.ascontiguousarray(regions).view(np.dtype((np.void, regions.dtype.itemsize * regions.shape[1])))
            _, idx = np.unique(regions_void, return_inverse=True)
            region_ids = idx
        else:
            region_ids = np.zeros(resolution)

        plt.figure(figsize=(10, 6))

        # Plot regions
        y_min, y_max = y_pred.min().item(), y_pred.max().item()
        y_range = y_max - y_min if y_max != y_min else 1.0

        plt.imshow(region_ids.reshape(1, -1), extent=[x_min, x_max, y_min - 0.1*y_range, y_max + 0.1*y_range], aspect='auto', cmap='tab20', alpha=0.3)

        plt.plot(x_grid, y_pred.detach().numpy(), color='black', linewidth=2, label='Prediction')

        if Y_samples is not None:
             X_np = X_data.detach().cpu().numpy()
             Y_np = Y_samples.detach().cpu().numpy() if isinstance(Y_samples, torch.Tensor) else Y_samples
             plt.scatter(X_np, Y_np, color='red', s=10, label='Data')

        plt.legend()
        plt.title('Regions and Prediction')

        if save_path:
            dirname = os.path.dirname(save_path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            plt.savefig(save_path)
            plt.close()

    else:
        # 2D Case - simplified region plot?
        pass

def plot_bab_iteration_best(model, top_3_solutions, X_train, y_train, iteration, gen_per_iter, config_name, save_dir='PyGAD_BAB_Plots'):
    """
    Plots the best 3 solutions of the current BaB iteration in subplots.
    top_3_solutions: List of dicts containing 'solution', 'fitness', 'id'
    """
    if not top_3_solutions:
        return

    num_plots = len(top_3_solutions)
    if num_plots == 0:
        return

    # Create figure with subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), constrained_layout=True)
    if num_plots == 1:
        axes = [axes]

    X_np = X_train.detach().cpu().numpy()
    Y_np = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train

    # Calculate global generation for title
    current_gen = (iteration + 1) * gen_per_iter

    # Prepare plotting grid
    with torch.no_grad():
        x_min, x_max = X_np.min(), X_np.max()
        x_range = x_max - x_min
        if x_range == 0: x_range = 1
        x_min_plot = x_min - 0.1 * x_range
        x_max_plot = x_max + 0.1 * x_range

        if X_np.shape[1] == 1:
            resolution = 1000
            X_grid = np.linspace(x_min_plot, x_max_plot, resolution).reshape(-1, 1)
            X_grid_torch = torch.tensor(X_grid, dtype=torch.float32)

            for i, sol_data in enumerate(top_3_solutions):
                ax = axes[i]
                solution = sol_data['solution']
                fit = sol_data['fitness']
                bid = sol_data['id']

                y_pred = model.forward(solution, X_grid_torch).cpu().numpy()

                ax.scatter(X_np, Y_np, label='Training Data', alpha=0.5, color='blue', s=10)
                ax.plot(X_grid, y_pred, label='Model Prediction', color='red', linewidth=2)
                ax.set_title(f"Rank {i+1} (ID: {bid}) - Fitness: {fit:.5f}")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                if i == 0:
                    ax.legend()

        elif X_np.shape[1] == 2:
            resolution = 100
            x1 = np.linspace(x_min_plot, x_max_plot, resolution)
            x2 = np.linspace(x_min_plot, x_max_plot, resolution)
            X1, X2 = np.meshgrid(x1, x2)
            X_grid = np.column_stack([X1.ravel(), X2.ravel()])
            X_grid_torch = torch.tensor(X_grid, dtype=torch.float32)

            for i, sol_data in enumerate(top_3_solutions):
                ax = axes[i]
                solution = sol_data['solution']
                fit = sol_data['fitness']
                bid = sol_data['id']

                y_pred = model.forward(solution, X_grid_torch).cpu().numpy().reshape(resolution, resolution)

                c = ax.contourf(X1, X2, y_pred, cmap='viridis', levels=50)
                fig.colorbar(c, ax=ax, label='Prediction')
                ax.set_title(f"Rank {i+1} (ID: {bid}) - Fitness: {fit:.5f}")

    fig.suptitle(f"BaB Iteration {iteration} (Global Gen {current_gen}) - Top {num_plots} Branches")

    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # Removed as per request
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = f'{save_dir}/{config_name}_iter_{iteration}.png'
    plt.savefig(save_path)
    print(f"Saved BaB Iteration plot to {save_path}")
    plt.close(fig)

def plot_combined_loss_history(histories, config_base_name, save_dir='loss_history'):
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.plot(hist, label=name)
    plt.yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss History Comparison - {config_base_name}')

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/compare_{config_base_name}.png')
    plt.close()

def plot_bab_loss_history(history_logs, generations_per_iter, config_name, use_inversion=False):
    """
    Plots the loss history for Branch and Bound training.

    Logic update:
    1. Identify Top 3 best branches per iteration.
    2. Continuity:
       - If a branch comes from a parent that was Top 3 in previous round, it inherits color base.
       - If multiple Top 3 branches come from the SAME parent (Fork), they get variations of the parent color (gradient).
       - If a branch starts fresh (no Top 3 parent), it gets a new distinct color.
    """
    if not history_logs:
        return

    # Prepare plot
    plt.figure(figsize=(12, 8))

    # Palette of colors
    palette = sns.color_palette("tab10")
    # Convert palette to RGB tuples for easy manipulation
    palette_rgb = [mcolors.to_rgb(c) for c in palette]
    next_color_idx = 0

    # Track assigned colors for branches: branch_id -> color (RGB tuple or hex)
    prev_winner_colors = {} # parent_id -> color

    for i, iter_data in enumerate(history_logs):
        branches = iter_data["branches"]

        # Sort branches by fitness descending (best first)
        sorted_branches = sorted(branches, key=lambda x: x['fitness'], reverse=True)

        # Pick Top 3
        top3 = sorted_branches[:3]

        # Group by parent to identify forks
        branches_by_parent = {}
        for b in top3:
            pid = b.get('parent_id')
            if pid not in branches_by_parent:
                branches_by_parent[pid] = []
            branches_by_parent[pid].append(b)

        current_winner_colors = {}

        # Assign colors
        for pid, siblings in branches_by_parent.items():
            # Check if parent was a winner
            if pid in prev_winner_colors:
                base_color = prev_winner_colors[pid]

                # If inherited, generate variations for siblings
                # Strategy:
                # 0th sibling (best): base_color
                # 1st sibling: lighter
                # 2nd sibling: darker (or even lighter)

                for idx, branch in enumerate(siblings):
                    if idx == 0:
                        branch_color = base_color
                    else:
                        # Modify color for forks
                        # Convert to HSV, shift Value or Saturation
                        h, s, v = mcolors.rgb_to_hsv(base_color)

                        # Variation strategy:
                        # idx 1: brighter (increase Value, decrease Saturation)
                        # idx 2: darker (decrease Value)

                        if idx == 1:
                            # Lighter
                            s_new = max(0.2, s * 0.6)
                            v_new = min(1.0, v * 1.3)
                            branch_color = mcolors.hsv_to_rgb((h, s_new, v_new))
                        else:
                            # Darker / Different
                            s_new = min(1.0, s * 1.2)
                            v_new = max(0.3, v * 0.7)
                            branch_color = mcolors.hsv_to_rgb((h, s_new, v_new))

                    current_winner_colors[branch['id']] = branch_color

                    # Store color property on branch object for plotting loop below or map via ID
                    # We'll map via ID.
            else:
                # New distinct colors for each sibling
                # They don't share a "Top 3" parent, so they are distinct lineages or restarts
                for branch in siblings:
                    # Pick next from palette
                    base_color = palette_rgb[next_color_idx % len(palette_rgb)]
                    next_color_idx += 1
                    current_winner_colors[branch['id']] = base_color

        # Plot loop
        for rank, branch in enumerate(top3):
            uid = branch['id']
            color = current_winner_colors[uid]
            pid = branch.get('parent_id')
            inherited = (pid in prev_winner_colors)

            # Prepare data
            local_hist = np.array(branch['local_history'])
            if len(local_hist) == 0:
                continue

            # X-Axis
            start_gen = i * generations_per_iter
            generations = np.arange(start_gen, start_gen + len(local_hist))

            # Plot
            plt.plot(generations, local_hist, color=color, alpha=0.9, linewidth=2.5)

            # Marker
            if inherited:
                 plt.scatter([start_gen], [local_hist[0]], color=color, s=20)
            else:
                 plt.scatter([start_gen], [local_hist[0]], color=color, marker='x', s=40)

        # Update winners for next round
        prev_winner_colors = current_winner_colors.copy()

    plt.yscale('log')
    plt.xlabel('Global Generation')
    plt.ylabel('Loss (Log Scale)')
    plt.title(f'BaB Loss History (Top 3 active branches per Iteration) - {config_name}')
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("PyGAD_BAB_Plots", exist_ok=True)
    save_path = f"Bab_loss_history/loss_history_{config_name}_{timestamp}.png"
    plt.savefig(save_path)
    print(f"Saved BaB loss history plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
