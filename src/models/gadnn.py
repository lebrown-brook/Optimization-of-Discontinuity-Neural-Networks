import torch
import torch.nn.functional as F
import numpy as np
import pygad as pg
from src.activations import discontinuous as dl
from src.activations import piecewise_linear as pwlu

class GADNN:
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        discontinuous_indices=None,
        Disc_Activation="disc",
        delta=1.0,
        Relu_true=True
    ):
        """
        A helper class to define a Neural Network architecture (potentially with discontinuous layers)
        and provide a forward function that accepts a flat parameter vector 'w', suitable for PyGAD.
        Optimized to reduce overhead during forward passes.
        """
        self.input_size = input_size
        self.hidden_layers = list(hidden_layers) if hidden_layers else []
        self.output_size = output_size
        self.discontinuous_indices = set(discontinuous_indices) if discontinuous_indices else set()
        self.Disc_Activation = Disc_Activation
        self.delta = float(delta)
        self.Relu_true = bool(Relu_true)

        # Layer sizes: [Input, Hidden1, Hidden2, ..., Output]
        self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]

        # Pre-calculate slices and shapes to avoid dictionary overhead in forward loop
        self.layer_params = []
        self.eps_structure = []
        self.num_genes = 0

        self._precalculate_architecture()

    @property
    def num_total_epsilon_neurons(self):
        """Returns the total number of neurons that have epsilon parameters."""
        return len(self.eps_structure)

    def _precalculate_architecture(self):
        """
        Calculates the indices and shapes for all parameters once during initialization.
        Stores them in a list-based structure (self.layer_params) for fast iteration.
        """
        current_idx = 0
        global_eps_idx = 0

        # Iterate over connections between layers (n layers -> n-1 connections)
        num_connections = len(self.layer_sizes) - 1

        for i in range(num_connections):
            in_f = self.layer_sizes[i]
            out_f = self.layer_sizes[i+1]

            # 1. Weights
            w_shape = (out_f, in_f)
            w_size = out_f * in_f
            w_slice = slice(current_idx, current_idx + w_size)
            current_idx += w_size

            # 2. Biases
            b_shape = (out_f,)
            b_size = out_f
            b_slice = slice(current_idx, current_idx + b_size)
            current_idx += b_size

            # 3. Epsilons (only for discontinuous layers, not output layer)
            is_output_layer = (i == num_connections - 1)
            eps_slice = None
            eps_shape = None
            has_epsilon = False

            if not is_output_layer and i in self.discontinuous_indices:
                has_epsilon = True
                eps_shape = (out_f,)
                eps_size = out_f
                eps_slice = slice(current_idx, current_idx + eps_size)

                # Store epsilon metadata specifically for analysis functions
                for neuron_idx in range(out_f):
                    self.eps_structure.append({
                        'disclayer_ind': i,
                        'eps_ind': neuron_idx,
                        'eps_globalind': global_eps_idx,
                        # Store absolute index in w for fast retrieval later
                        'w_index': current_idx + neuron_idx
                    })
                    global_eps_idx += 1

                current_idx += eps_size

            # Store all info needed for this layer's forward pass
            self.layer_params.append({
                'w_slice': w_slice,
                'w_shape': w_shape,
                'b_slice': b_slice,
                'b_shape': b_shape,
                'has_epsilon': has_epsilon,
                'eps_slice': eps_slice,
                'eps_shape': eps_shape,
                'is_output': is_output_layer
            })

        self.num_genes = current_idx


    def forward(self, w, x, return_regions=False):
        """
        Optimized forward pass using pre-calculated slices.
        """
        # Input validation (only if raw list/numpy passed)
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w, dtype=torch.float32)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        out = x
        collected_regions_list = []

        # Iterate directly over pre-calculated layer info
        for layer_info in self.layer_params:

            # 1. Linear Transformation
            # Extract weight and bias using pre-calculated slices
            weight = w[layer_info['w_slice']].reshape(layer_info['w_shape'])
            bias = w[layer_info['b_slice']].reshape(layer_info['b_shape'])

            out = F.linear(out, weight, bias)

            # If it's the output layer, we are done with this iteration
            if layer_info['is_output']:
                break

            # 2. Activation
            if layer_info['has_epsilon']:
                # Discontinuous Activation
                epsilon = w[layer_info['eps_slice']].reshape(layer_info['eps_shape'])

                if self.Disc_Activation == "disc":
                    # Logic: base(x) + epsilon * heaviside(x)
                    # Use custom autograd function for Heaviside (gradient=0)
                    h_val = dl.heaviside_function.apply(out)

                    if return_regions:
                        # Convert to numpy for storage/analysis
                        h_val_np = h_val.detach().cpu().numpy()
                        collected_regions_list.append(h_val_np)

                    jump = epsilon * h_val

                    # Calculate Base Activation part
                    if self.Relu_true:
                        base = F.relu(out)
                    else:
                        base = F.elu(out)

                    out = base + jump

                elif self.Disc_Activation == "pwlu":
                    if return_regions:
                        # Regions for PWLU
                        regions = torch.zeros_like(out)
                        mask_pos = (out > 0)
                        regions[mask_pos] = 1.0
                        regions_np = regions.detach().cpu().numpy()
                        collected_regions_list.append(regions_np)

                    out = pwlu.PiecewiseLinearActivationFunction.apply(out, epsilon, self.delta, self.Relu_true)
            else:
                # Standard Activation
                if self.Relu_true:
                    out = F.relu(out)
                else:
                    out = F.elu(out)

        if return_regions:
            if collected_regions_list:
                full_region_matrix = np.hstack(collected_regions_list)
            else:
                full_region_matrix = np.empty((x.shape[0], 0))
            return out, full_region_matrix

        return out

    def get_epsilon_values(self, w):
        """
        Extracts epsilon values from the flat parameter vector w using pre-calculated structure.
        Much faster than reconstructing the whole param dict.
        """
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w, dtype=torch.float32)

        eps_values_list = []

        # self.eps_structure already contains absolute indices
        # No need to slice and reshape tensors anymore
        for eps_meta in self.eps_structure:
            val = w[eps_meta['w_index']].item()

            eps_entry = eps_meta.copy()
            eps_entry['value'] = val
            eps_values_list.append(eps_entry)

        return eps_values_list

    def get_epsilon_indices_in_w(self):
        """
        Returns a list of global indices in the flat parameter vector 'w'.
        """
        return [meta['w_index'] for meta in self.eps_structure]

    def capture_activation_regions(self, w, x):
        """
        Runs a forward pass and captures the activation regions (which side of the discontinuity).

        Args:
            w: Flat parameter vector (chromosome).
            x: Input data grid.

        Returns:
            X_region_matrix: Matrix of shape (num_samples, num_epsilons) with 0/1 values.
            eps_values_list: List of dictionaries with epsilon metadata and values.
        """
        _, region_matrix = self.forward(w, x, return_regions=True)
        eps_values_list = self.get_epsilon_values(w)
        return region_matrix, eps_values_list

def create_gadnn_model_func(input_size, hidden_layers, output_size, **kwargs):
    """
    Factory function to create a GADNN instance and return its forward method and num_genes.

    Returns:
        model_func: Callable (w, x) -> y
        num_genes: int
    """
    gadnn = GADNN(input_size, hidden_layers, output_size, **kwargs)
    return gadnn.forward, gadnn.num_genes


def create_ga_instance(
        self,
        fitness_func,
        num_generations: int = 100,
        sol_per_pop: int = 20,
        num_parents_mating: int = 5,
        init_range_low: float = -1.0,
        init_range_high: float = 1.0,
        parent_selection_type: str = "sss",
        keep_parents: int = 2,
        crossover_type: str = "single_point",
        mutation_type: str = "random",
        mutation_percent_genes: float = 10.0,
        **kwargs
):
    """
    Creates a PyGAD GA instance configured for this GADNN model.

    Args:
        fitness_func (callable): The function to evaluate the fitness of a solution.
            It must accept (ga_instance, solution, solution_idx) and return a fitness value.
        num_generations (int): Number of generations to run the evolutionary algorithm.
        sol_per_pop (int): Number of solutions (chromosomes) in the population.
        num_parents_mating (int): Number of parents to select from the population for mating.
        init_range_low (float): Lower bound for the random initialization of genes.
        init_range_high (float): Upper bound for the random initialization of genes.
        parent_selection_type (str): Type of parent selection.
            Options: 'sss' (Steady State), 'rws' (Roulette Wheel), 'sus' (Stochastic Universal),
            'rank', 'random', 'tournament'.
        keep_parents (int): Number of parents to keep in the next generation (elitism).
            -1 means keep all parents.
        crossover_type (str): Type of crossover operation.
            Options: 'single_point', 'two_points', 'uniform', 'scattered'.
        mutation_type (str): Type of mutation operation.
            Options: 'random', 'swap', 'inversion', 'scramble', 'adaptive'.
        mutation_percent_genes (float): Percentage of genes to mutate in a solution.
        **kwargs: Additional arguments to pass to the pygad.GA constructor.

    Returns:
        pygad.GA: An instance of the Genetic Algorithm configured with the specified parameters.
    """
    return pg.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=self.num_genes,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        **kwargs
    )

