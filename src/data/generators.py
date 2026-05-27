from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import test_functions as tfp
from sklearn.preprocessing import StandardScaler

#create a simple step function to approximate with the DNN class
def step_function(x: np.ndarray) -> np.ndarray:
    return np.where(x > -1, 3.0, 0.0)

def piecewise_linear_function(x: np.ndarray) -> np.ndarray:
   #f(x) = -2x + 1 for x < -5
    # f(x)= 3 for -5 <= x <= 5
    # f(x) = 0.5x - 2 for x > 5
    y = np.zeros_like(x)
    y[x < -5] = -2.0 * x[x < -5] + 1.0
    y[x >= -5] = 3.0
    y[x > 5] = 0.5 * x[x > 5] - 2.0
    return y

def multiple_steps_function(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    y[x > -5] = 2.0
    y[x > 0] = -1.0
    y[x > 5] = 3.0
    return y

def sine_with_discontinuity(x: np.ndarray) -> np.ndarray:
    y = np.sin(x)
    y[x > 2] += 2
    return y

def linear_with_varying_slopes(x: np.ndarray) -> np.ndarray:
    return np.where(x < -5, -0.5 * x,
                    np.where(x < 0, 2 * x,
                             np.where(x < 5, -x + 5, 0.5 * x)))

#generate some sample data
def generate_data(num_samples: int = 100, function: str = "step_function", input_dim: int = 1, x_min: float = -10.0, x_max: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:


    if hasattr(tfp, function):
        input_size = 2
        x = np.random.uniform(low=x_min, high=x_max, size=(num_samples, input_dim))
        y = getattr(tfp, function)(x)
    elif function in globals():
        input_size = 1
        x = np.random.uniform(low=x_min, high=x_max, size=(num_samples, input_dim))
        y = globals()[function](x)
    else:
        raise ValueError(f"Unknown function: {function}")
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def plot_loss_history(loss_histories: List[List[float]], dnn_list: Sequence[torch.nn.Module], config_name: str = "") -> None:
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(loss_histories):
        if len(losses) > 0:
            plt.plot(losses, label=f'{dnn_list[i].name} Loss')

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Loss History ({config_name})')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)

    import os
    if not os.path.exists('loss_history'):
        os.makedirs('loss_history')

    save_path = f'loss_history/{config_name}.png' if config_name != "" else 'loss_history/loss_history_test.png'
    plt.savefig(save_path)
    plt.close()




#print parameters of the DNN
def print_dnn_parameters(dnn: torch.nn.Module, showvalue = False) -> None:
    print(f"\n{'='*20} Model: {dnn.name} {'='*20}")
    for name, module in dnn.named_modules():
        # Skip container modules, focus on layers with parameters
        params = list(module.named_parameters(recurse=False))
        if params:
            print(f"\n[Layer: {name}] Type: {type(module).__name__}")
            for p_name, p in params:
                print(f"  - {p_name} (shape: {list(p.shape)}):")
                if showvalue:
                    print(f"    {p.data}")

def plotNN_list(dnn_list: Sequence[torch.nn.Module], x_samples: torch.Tensor, y_samples: torch.Tensor, n: Optional[int] = None, config_name: str = "", scaler=None) -> None:
    with torch.no_grad():
        input_dim = x_samples.shape[1]
        plt.figure(figsize=(15, 5 * len(dnn_list)))

        # Prepare original x_samples for plotting and range calculation
        x_samples_np = x_samples.cpu().numpy()
        if scaler is not None:
            x_samples_orig = scaler.inverse_transform(x_samples_np)
        else:
            x_samples_orig = x_samples_np

        # Calculate ranges with margin
        mins = x_samples_orig.min(axis=0)
        maxs = x_samples_orig.max(axis=0)
        ranges = maxs - mins
        # Handle case where range is 0 (single point)
        ranges[ranges == 0] = 1.0

        mins -= 0.1 * ranges
        maxs += 0.1 * ranges

        # Generate grid for plotting
        if input_dim == 1:
            x_grid_orig = np.linspace(mins[0], maxs[0], 1000).reshape(-1, 1)
            grid_inputs_orig = x_grid_orig
        elif input_dim == 2:
            x_grid = np.linspace(mins[0], maxs[0], 100)
            y_grid = np.linspace(mins[1], maxs[1], 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            grid_inputs_orig = np.column_stack((X.ravel(), Y.ravel()))

        # Scale grid inputs for model prediction
        if scaler is not None:
            grid_inputs_scaled = scaler.transform(grid_inputs_orig)
            grid_inputs = torch.tensor(grid_inputs_scaled, dtype=torch.float32)
        else:
            grid_inputs = torch.tensor(grid_inputs_orig, dtype=torch.float32)

        # Ensure device compatibility
        if len(dnn_list) > 0:
            device = next(dnn_list[0].parameters()).device
            grid_inputs = grid_inputs.to(device)

        for i, dnn in enumerate(dnn_list):
            if input_dim == 1:
                y_pred = dnn(grid_inputs)
                plt.subplot(len(dnn_list), 1, i + 1)
                plt.plot(grid_inputs_orig, y_pred.cpu().numpy(), color='blue', label=f'{dnn.name} Prediction')
                plt.scatter(x_samples_orig, y_samples.cpu().numpy(), color='red', label='Training Data')
                plt.xlabel('Input')
                plt.ylabel('Output')
            elif input_dim == 2:
                Z = dnn(grid_inputs).reshape(X.shape).cpu().detach().numpy()
                plt.subplot(len(dnn_list), 1, i + 1)
                cp = plt.contourf(X, Y, Z, cmap='viridis', levels=50)
                plt.colorbar(cp)
                # Scatter plot training data
                plt.scatter(x_samples_orig[:, 0], x_samples_orig[:, 1], c=y_samples.cpu().numpy().flatten(), cmap='viridis', edgecolors='white', linewidth=0.5, label='Training Data')
                plt.xlabel('Input X1')
                plt.ylabel('Input X2')

            title_suffix = f"epoch={n}" if n is not None else "final"
            plt.title(f'({config_name}) Approximation {title_suffix}')
            plt.legend()

        plt.tight_layout()

        # Create a plots directory if it doesn't exist or skip saving if no config_name for the testing
        if config_name == "":
            plt.close()
            return

        import os
        if not os.path.exists('plots_advanced'):
            os.makedirs('plots_advanced')

        save_path = f'plots_advanced/dnns_plot_{dnn.name}_{config_name}_{n}.png'
        plt.savefig(save_path)
        plt.close()

def create_model_with_shared_init(model_cls: Callable, *args, shared_seed: int = 42, **kwargs) -> torch.nn.Module:
    """Set global RNGs to a shared seed then create the model so initial weights match."""
    torch.manual_seed(shared_seed)
    np.random.seed(shared_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(shared_seed)
    return model_cls(*args, **kwargs)



#main function that runs the tests
if __name__ == "__main__":
    pass
