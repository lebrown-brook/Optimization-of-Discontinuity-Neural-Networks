from typing import Any, Optional, Tuple
import torch
# import PiecewiseLinearActivation as pwlu  <-- Removed to avoid circular import


class heaviside_function(torch.autograd.Function):
    """Differentiable Heaviside step function. Gradient is zero everywhere."""
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        return (input > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        # Subgradient is zero everywhere
        return torch.zeros_like(grad_output)


class heaviside_function_STE(torch.autograd.Function):
    """Heaviside step function with Straight-Through Estimator (STE)."""
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        # Use a straight-through estimator (STE)
        input, = ctx.saved_tensors
        # Using hardtanh as a surrogate gradient for the identity function in [-1, 1]
        return torch.nn.functional.hardtanh(grad_output)


class DiscontinuousLayer(torch.nn.Module):
    """
    NOTE: This layer appears to be incomplete or deprecated.
    The forward pass calculates a discontinuous jump 'y' but does not use it in the final return value.
    It was likely intended to be 'return self.act(x) + y'.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        epsilon_init: Optional[torch.Tensor] = None,
        STE_true: bool = False,
        Disc_Activation: str = "relu",
        delta: float = 0.5, # Added for pwlu
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.STE_true = STE_true
        self.Disc_Activation = Disc_Activation
        self.delta = delta
        self.epsilon_init = epsilon_init

        torch.nn.init.xavier_uniform_(self.linear.weight)

        if self.Disc_Activation == "relu":
            self.act = torch.nn.ReLU()
        elif self.Disc_Activation == "elu":
            self.act = torch.nn.ELU()
        elif self.Disc_Activation == "pwlu":
            import PiecewiseLinearActivation as pwlu
            # Correctly pass parameters to PiecewiseLinearActivation
            self.act = pwlu.PiecewiseLinearActivation(output_size, epsilon_init=self.epsilon_init, delta=self.delta)
        else:
            raise ValueError(f"Unknown activation: {self.Disc_Activation}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if epsilon_init is not None:
            self.epsilon = torch.nn.Parameter(epsilon_init.clone().detach().to(dtype=torch.float32, device=device))
        else:
            self.epsilon = torch.nn.Parameter(torch.randn(output_size, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        heaviside_fn = heaviside_function_STE if self.STE_true else heaviside_function
        y = torch.mul(self.epsilon, heaviside_fn.apply(x))

        # The original implementation was likely missing the addition of y
        # return self.act(x) + y
        return self.act(x)



# discontinuous layer with to different linear layers
class DiscontinuousLayer2(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(DiscontinuousLayer2, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)
        self.linear2 = torch.nn.Linear(input_size, output_size)
        self.relu = torch.nn.ReLU()
        self.epsilon = torch.nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        return self.relu(x1) + torch.mul(self.epsilon, heaviside_function.apply(x2))
