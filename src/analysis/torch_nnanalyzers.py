import torch
import numpy as np
import copy
import sys
import os

# Ensure we can import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.activations import piecewise_linear as pwlu

class TorchDiscontinuousNeuralNetworkAnalyzer:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

        self.disclayers = []
        self.epsweights = []

        # Identify discontinuous layers
        # We iterate over named modules to find DiscontinuousActivation or PiecewiseLinearActivation
        for name, module in self.model.named_modules():
            if isinstance(module, pwlu.DiscontinuousActivation) or isinstance(module, pwlu.PiecewiseLinearActivation):
                self.disclayers.append({
                    'name': name,
                    'module': module,
                    'epsilon': module.epsilon
                })

        if not self.disclayers:
            print("Warning: No DiscontinuousActivation layers found.")

        # Flatten epsilons for ranking
        self.n_eps = 0
        for layer_idx, layer_info in enumerate(self.disclayers):
            eps_tensor = layer_info['epsilon']
            for i in range(eps_tensor.numel()):
                self.epsweights.append({
                    'value': eps_tensor[i].item(),
                    'layer_idx': layer_idx,
                    'eps_idx': i,
                    'eps_globalind': self.n_eps
                })
                self.n_eps += 1

        self.eps_rank = None
        self.epsweights_sorted = None

        # For capturing activations
        self.activations = {}
        self.hooks = []

    def _register_hooks(self):
        self.activations = {}
        self.hooks = []

        def get_activation(name):
            def hook(model, input, output):
                # input is a tuple (x,)
                self.activations[name] = input[0].detach()
            return hook

        for layer_info in self.disclayers:
            name = layer_info['name']
            hook = layer_info['module'].register_forward_hook(get_activation(name))
            self.hooks.append(hook)

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def fit_eps_rank(self, X_data, norm_ord=np.inf, weighted=True):
        # X_data should be a torch tensor
        if not isinstance(X_data, torch.Tensor):
            X_data = torch.tensor(X_data, dtype=torch.float32).to(self.device)

        # Handle 1D input (assume it's [N_samples] -> [N_samples, 1])
        if X_data.dim() == 1:
            X_data = X_data.unsqueeze(1)

        self.model.eval()

        with torch.no_grad():
            original_preds = self.model(X_data).detach().cpu().numpy()

        eps_rank = []
        sum_value = sum([abs(e['value']) for e in self.epsweights])
        if sum_value == 0:
            sum_value = 1.0 # Avoid division by zero

        for eps_info in self.epsweights:
            layer_idx = eps_info['layer_idx']
            eps_idx = eps_info['eps_idx']
            value = eps_info['value']

            # Temporarily set epsilon to 0
            with torch.no_grad():
                original_eps_val = self.disclayers[layer_idx]['module'].epsilon[eps_idx].item()
                self.disclayers[layer_idx]['module'].epsilon[eps_idx] = 0.0

                new_preds = self.model(X_data).detach().cpu().numpy()

                norm_ind = np.linalg.norm(original_preds - new_preds, ord=norm_ord)

                if weighted:
                    rank_val = (norm_ind * abs(value)) / sum_value
                else:
                    rank_val = norm_ind

                eps_rank.append(rank_val)

                # Restore epsilon
                self.disclayers[layer_idx]['module'].epsilon[eps_idx] = original_eps_val

        self.eps_rank = eps_rank
        # Sort epsweights by rank (ascending, so last elements are most important)
        sorted_indices = np.argsort(self.eps_rank)
        self.epsweights_sorted = [self.epsweights[i] for i in sorted_indices]

        return self.eps_rank

    def contregions(self, X, which_boundaries=None, top_eps=None):
        # X: numpy array or tensor
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X

        # Handle 1D input (assume it's [N_samples] -> [N_samples, 1])
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(1)

        # Determine which epsilons to consider
        if which_boundaries is None:
             # Default: all
             which_boundaries = [e['eps_globalind'] for e in self.epsweights]

        if top_eps is not None:
            if self.epsweights_sorted is None:
                raise ValueError("Run fit_eps_rank first!")

            # epsweights_sorted is sorted ascending, so we take the last top_eps
            top_indices = [e['eps_globalind'] for e in self.epsweights_sorted[-top_eps:]]
            # Intersect with which_boundaries
            which_boundaries = [idx for idx in which_boundaries if idx in top_indices]

        # Run model to get activations
        self._register_hooks()
        self.model.eval()
        with torch.no_grad():
            self.model(X_tensor)

        X_region_matrix = np.zeros((X_tensor.shape[0], len(which_boundaries)))

        for col_idx, global_eps_idx in enumerate(which_boundaries):
             eps_info = self.epsweights[global_eps_idx]
             l_idx = eps_info['layer_idx']
             e_idx = eps_info['eps_idx']

             layer_name = self.disclayers[l_idx]['name']
             act = self.activations[layer_name] # (N, neurons)

             # State is 1 if input > 0 else 0
             val = (act[:, e_idx] > 0).float().cpu().numpy()
             X_region_matrix[:, col_idx] = val

        self._remove_hooks()

        # Find unique regions
        region_vectors, X_membership, region_vectors_counts = np.unique(
            X_region_matrix, axis=0, return_inverse=True, return_counts=True
        )

        return region_vectors, X_membership, region_vectors_counts, X_region_matrix, which_boundaries

