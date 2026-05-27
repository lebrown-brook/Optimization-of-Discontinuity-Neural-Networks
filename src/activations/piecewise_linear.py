from typing import Optional
import torch
from . import discontinuous as dl

class PiecewiseLinearActivationFunction(torch.autograd.Function):
    """Custom autograd function for piecewise linear activation."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, epsilon: torch.Tensor, delta: float, relu_true: bool):
        ctx.save_for_backward(input, epsilon)
        ctx.delta = float(delta)
        ctx.relu_true = bool(relu_true)

        if ctx.relu_true:
            case1 = torch.zeros_like(input)
        else:
            case1 = torch.exp(input) - torch.ones_like(input)

        case2 = (epsilon / delta + 1) * input
        case3 = input + epsilon

        output = torch.where(
            input <= 0, case1,
            torch.where(input <= delta, case2, case3)
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, epsilon = ctx.saved_tensors
        delta = ctx.delta
        relu_true = ctx.relu_true

        if relu_true:
            grad_input = torch.where(
                input <= 0, torch.zeros_like(input),
                torch.where(input <= delta, (epsilon / delta + 1), torch.ones_like(input))
            )
        else:
            grad_input = torch.where(
                input <= 0, torch.exp(input),
                torch.where(input <= delta, (epsilon / delta + 1), torch.ones_like(input))
            )

        grad_epsilon = torch.where(
            input <= 0, torch.zeros_like(input),
            torch.where(input <= delta, input / delta, torch.ones_like(input))
        )

        # adjust grad_epsilon shape if needed (sum over batch)
        if grad_epsilon.shape != epsilon.shape:
            grad_epsilon = grad_epsilon.sum(dim=0)

        return grad_output * grad_input, grad_output * grad_epsilon, None, None


class PiecewiseLinearActivation(torch.nn.Module):
    def __init__(self, output_size: int, epsilon_init: Optional[torch.Tensor] = None, delta: float = 0.1, RelU_true: bool = True):
        super().__init__()
        self.delta = float(delta)
        self.Relu_true = bool(RelU_true)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if epsilon_init is not None:
            self.epsilon = torch.nn.Parameter(epsilon_init.clone().detach().to(dtype=torch.float32, device=device))
        else:
            self.epsilon = torch.nn.Parameter(torch.randn(output_size, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return PiecewiseLinearActivationFunction.apply(x, self.epsilon, self.delta,self.Relu_true)


class DiscontinuousActivation(torch.nn.Module):
    def __init__(self, output_size: int, epsilon_init: Optional[torch.Tensor] = None, STE_true: bool = False, Relu_true: bool = True):
        super().__init__()
        self.STE_true = bool(STE_true)
        self.Relu_true = bool(Relu_true)
        self.epsilon_init = epsilon_init

        self.relu = torch.nn.ReLU() if self.Relu_true else torch.nn.ELU()
        self.eps_act = dl.heaviside_function_STE if self.STE_true else dl.heaviside_function

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if epsilon_init is not None:
            self.epsilon = torch.nn.Parameter(epsilon_init.clone().detach().to(dtype=torch.float32, device=device))
        else:
            self.epsilon = torch.nn.Parameter(torch.randn(output_size, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.mul(self.epsilon, self.eps_act.apply(x))
        x = self.relu(x)
        return x + y