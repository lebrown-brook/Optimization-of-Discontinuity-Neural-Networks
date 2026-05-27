from typing import List, Optional, Sequence
import numpy as np
import torch
from collections import OrderedDict
from src.activations import piecewise_linear as pwlu


class DNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers: Sequence[int],
        output_size: int,
        discontinuous_indices: Optional[Sequence[int]] = None,
        epsilon_init: Optional[Sequence[float]] = None,
        STE_true: bool = False,
        Disc_Activation: str = "disc",
        name: str = "DNN",
        delta: float = 0.5,
        Relu_true: bool = True,
    ) -> None:
        """Feed-forward network with optional discontinuous activations at given layer indices.

        Args:
            Disc_Activation (str): "disc" for discontinuous activation (Heaviside jump), "pwlu" for Piecewise Linear Unit.
            Relu_true (bool): If True, uses ReLU as base activation. If False, uses ELU.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = list(hidden_layers)
        self.output_size = output_size
        self.discontinuous_indices = list(discontinuous_indices) if discontinuous_indices else []
        self.epsilon_init = epsilon_init
        self.STE_true = STE_true
        self.Relu_true = Relu_true
        self.Disc_Activation = Disc_Activation
        self.delta = float(delta)
        self.name = str(name)

        self.layers = self.build_network()

    def build_network(self) -> torch.nn.Sequential:
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        layers = OrderedDict()
        for i in range(len(layer_sizes) - 2):
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
            if i in self.discontinuous_indices:
                linear_layer = torch.nn.Linear(in_f, out_f)
                torch.nn.init.zeros_(linear_layer.bias) # Initialize bias to 0 to match TF
                layers[f"discontinuous_dense_{i}"] = linear_layer
                if self.Disc_Activation == "disc":
                    layers[f"discontinuous_activation_{i}"] = pwlu.DiscontinuousActivation(
                        output_size=out_f, epsilon_init=self.epsilon_init, STE_true=self.STE_true, Relu_true=self.Relu_true
                    )
                elif self.Disc_Activation == "pwlu":
                    layers[f"discontinuous_activation_{i}"] = pwlu.PiecewiseLinearActivation(
                        output_size=out_f, epsilon_init=self.epsilon_init, delta=self.delta, RelU_true=self.Relu_true
                    )
            else:
                linear_layer = torch.nn.Linear(in_f, out_f)
                torch.nn.init.zeros_(linear_layer.bias) # Initialize bias to 0 to match TF
                layers[f"dense_{i}"] = linear_layer
                layers[f"{'relu' if self.Relu_true else 'elu'}_{i}"] = torch.nn.ReLU() if self.Relu_true else torch.nn.ELU()

        output_layer = torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
        torch.nn.init.zeros_(output_layer.bias) # Initialize bias to 0 to match TF
        layers["output_layer"] = output_layer
        return torch.nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PWLUDNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers: Sequence[int],
        output_size: int,
        discontinuous_indices: Optional[Sequence[int]] = None,
        epsilon_init: Optional[Sequence[float]] = None,
        STE_true: bool = False,
        Relu_true: bool = True,
        delta: float = 0.5,
        name: str = "PWLUDNN",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = list(hidden_layers)
        self.output_size = output_size
        self.discontinuous_indices = list(discontinuous_indices) if discontinuous_indices else []
        self.epsilon_init = epsilon_init
        self.STE_true = STE_true
        self.Relu_true = Relu_true
        self.delta = float(delta)
        self.name = str(name)

        self.layers = self.build_network()

    def build_network(self) -> torch.nn.Sequential:
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        layers = OrderedDict()
        for i in range(len(layer_sizes) - 2):
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
            if i in self.discontinuous_indices:
                layers[f"discontinuous_{i}"] = torch.nn.Linear(in_f, out_f)
                layers[f"PWLU_discontinuous_{i}"] = pwlu.PiecewiseLinearActivation(
                    out_f, epsilon_init=self.epsilon_init, delta=self.delta
                )
            else:
                layers[f"dense_{i}"] = torch.nn.Linear(in_f, out_f)
                layers[f"{'relu' if self.Relu_true else 'elu'}_{i}"] = torch.nn.ReLU() if self.Relu_true else torch.nn.ELU()

        layers["output_layer"] = torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
        return torch.nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)



