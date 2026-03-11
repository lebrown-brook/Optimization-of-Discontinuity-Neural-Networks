import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


# Ensure we can import from the root directory
sys.path.append(os.getcwd())
import basic_tests
from DNN import DNN
from basic_tests import (
    create_model_with_shared_init,
    generate_data,
    train_dnn,
    print_dnn_parameters
)
import Testfunctions_paper as tfp

from sklearn.preprocessing import StandardScaler
from deltaNN.torch_nnanalyzers import TorchDiscontinuousNeuralNetworkAnalyzer

def analyze_and_plot_regions(model, X_data=None, Y_samples=None, save_path=None, show_plot=False, top_eps=20, resolution=100, x_min=-2, x_max=2, scaler=None):
    """
    Analyzes the continuity regions of a DNN model and plots the results.

    Args:
        model: The DNN model to analyze.
        X_data: Data for fitting epsilon rank. If None, random data is generated.
        Y_samples: Target samples for plotting.
        save_path: Path to save the figure. If None, figure is not saved.
        show_plot: Whether to show the plot using plt.show().
        top_eps: Number of top epsilons to consider for regions.
        resolution: Resolution of the grid for visualization.
        x_min: Minimum x value for the grid.
        x_max: Maximum x value for the grid.
        scaler: StandardScaler object to transform grid data for plotting if provided.
    """
  #POINTS OF CONTEMPTION: So far, X_data is used for eps_ranking. Should it be the grid instead always or is the eps_rank on the training data important?
    # Create Analyzer
    print("Initializing Analyzer...")
    analyzer = TorchDiscontinuousNeuralNetworkAnalyzer(model)

    # Determine input size
    if hasattr(model, 'input_size'):
        input_size = model.input_size
    else:
        # Fallback: try to infer from first layer
        try:
            first_layer = next(model.children())
            if isinstance(first_layer, torch.nn.Sequential):
                first_layer = first_layer[0]
            input_size = first_layer.in_features
        except:
            print("Could not infer input size, defaulting to 2.")
            input_size = 2

    # Analyze regions
    print(f"Generating grid for input size {input_size}...")

    if input_size == 1:
        # 1D Grid
        X_grid = np.linspace(x_min, x_max, resolution).reshape(-1, 1)
    else:
        # 2D Grid (default)
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(x_min, x_max, resolution))
        X_grid = np.c_[xx.ravel(), yy.ravel()]

    if scaler is not None:
        X_grid_input = scaler.transform(X_grid)
        X_grid_torch = torch.tensor(X_grid_input, dtype=torch.float32)
    else:
        X_grid_torch = torch.tensor(X_grid, dtype=torch.float32)

    # Generate some dummy data for ranking if not provided
    if X_data is None:
        print("Fitting epsilon rank with grid data...")
        X_data_rank = X_grid_torch
    else:
        print("Fitting epsilon rank with provided data...")
        #Assume X_data already scaled if scaler is provided
        X_data_rank = X_data

    analyzer.fit_eps_rank(X_data_rank)

    print(f"Epsilon Ranking (Top {top_eps}):")
    if analyzer.epsweights_sorted:
        if top_eps is None: top_eps = len(analyzer.epsweights_sorted)
        for e in analyzer.epsweights_sorted[-top_eps:]:
            # Find the rank value using the global index
            rank_val = analyzer.eps_rank[e['eps_globalind']]
            print(f"Layer {e['layer_idx']}, Eps {e['eps_idx']}, Val: {e['value']:.4f}, Rank: {rank_val:.4f}")

    # Get regions for top epsilons
    print("Analyzing continuity regions...")
    region_vectors, X_membership, region_vectors_counts, X_region_matrix, which_boundaries = analyzer.contregions(X_grid_torch, top_eps=top_eps)


    print(f"Found {len(region_vectors)} unique regions.")

    # Plot
    if input_size == 1:
        fig = plt.figure(figsize=(8, 6))
        # 1D Plot
        # Get predictions
        model.eval()
        with torch.no_grad():
            y_pred = model(X_grid_torch).cpu().numpy()

        # Plot function
        plt.plot(X_grid, y_pred, label='Prediction', color='black')

        if X_data is not None and Y_samples is not None:
             X_np = X_data.detach().cpu().numpy() if isinstance(X_data, torch.Tensor) else X_data
             Y_np = Y_samples.detach().cpu().numpy() if isinstance(Y_samples, torch.Tensor) else Y_samples
             plt.scatter(X_np, Y_np, label='Target Samples', color='red', s=10, alpha=0.6)

        # Calculate extent for background
        y_min_val, y_max_val = y_pred.min(), y_pred.max()
        if Y_samples is not None:
             Y_np = Y_samples.detach().cpu().numpy() if isinstance(Y_samples, torch.Tensor) else Y_samples
             y_min_val = min(y_min_val, Y_np.min())
             y_max_val = max(y_max_val, Y_np.max())

        y_range = y_max_val - y_min_val
        if y_range == 0: y_range = 1.0
        y_min_plot = y_min_val - 0.1 * y_range
        y_max_plot = y_max_val + 0.1 * y_range

        # Overlay regions as background
        plt.imshow(
            X_membership.reshape(1, -1),
            extent=[x_min, x_max, y_min_plot, y_max_plot],
            aspect='auto',
            cmap='tab20',
            alpha=0.3,
            origin='lower'
        )

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'Continuity Regions (Top {top_eps} Epsilons)')
    else:
        # 2D Plot
        fig, axes = plt.subplots(2, 1, figsize=(8, 12))

        # Subplot 1: Continuity Regions
        axes[0].imshow(X_membership.reshape(resolution, resolution), extent=[x_min, x_max, x_min, x_max], origin='lower', cmap='tab20')
        axes[0].set_xlabel('x1')
        axes[0].set_ylabel('x2')
        axes[0].set_title(f'Continuity Regions (Top {top_eps} Epsilons)')

        # Subplot 2: Prediction Heatmap
        model.eval()
        with torch.no_grad():
            y_pred = model(X_grid_torch).cpu().numpy()

        im = axes[1].imshow(y_pred.reshape(resolution, resolution), extent=[x_min, x_max, x_min, x_max], origin='lower', cmap='viridis')
        axes[1].set_xlabel('x1')
        axes[1].set_ylabel('x2')
        axes[1].set_title('Prediction Heatmap')
        fig.colorbar(im, ax=axes[1], label='Prediction')

        if X_data is not None and Y_samples is not None:
             X_np = X_data.detach().cpu().numpy() if isinstance(X_data, torch.Tensor) else X_data
             Y_np = Y_samples.detach().cpu().numpy() if isinstance(Y_samples, torch.Tensor) else Y_samples
             # Scatter x1, x2. Color by Y_np
             axes[1].scatter(X_np[:, 0], X_np[:, 1], c=Y_np.ravel(), cmap='viridis', edgecolors='white', s=20, label='Samples')

        plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def main():
    # 1. Create a simple model with 2D input and discontinuous layers
    input_size = 1
    hidden_layers = [4, 4 , 2 ]
    output_size = 1
    discontinuous_indices = [0] # Make both hidden layers discontinuous

    print("Creating DNN model...")
    model = DNN(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        discontinuous_indices=discontinuous_indices,
        Disc_Activation="pwlu", # Uses DiscontinuousActivation
        STE_true=False,
        Relu_true=False
    )
    func = "linear_with_varying_slopes"
    top_eps = 5
    x_min, x_max =-10 , 10

    x_samples, y_samples = generate_data(1000, function=func, input_dim=input_size, x_min=x_min, x_max=x_max)
    scaler = StandardScaler()
    x_samples_scaled = torch.tensor(scaler.fit_transform(x_samples), dtype=torch.float32)

    analyze_and_plot_regions(model, X_data=x_samples, Y_samples=y_samples, save_path=f"test_torch_analyzer_plot_pre_training.png",
                             show_plot=False, x_min=x_min, x_max=x_max, top_eps=top_eps, scaler=scaler)

    # Train models
    _, final_losses, _ = train_dnn(
        [model],
        x_samples_scaled,
        y_samples,
        num_epochs=5000,
        learning_rate=0.01,
        delta_factor=0.99,
        lambda_reg=0.0,
        config_name="",
        scaler=scaler,
        batch_size= None
    )
    analyze_and_plot_regions(model, X_data=x_samples, Y_samples=y_samples, save_path=f"test_torch_analyzer_plot_post_training.png",
                             show_plot=False, x_min=x_min, x_max=x_max, top_eps=top_eps, scaler=scaler)



if __name__ == "__main__":
    main()

