from abc import ABC, abstractmethod
import torch
import numpy as np

class BaseModel(torch.nn.Module, ABC):
    """Abstract base class for all neural network models."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def get_num_parameters(self) -> int:
        """Returns the total number of tunable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def from_vector(self, weights: np.ndarray) -> None:
        """Load weights from a 1D vector (useful for GA/ES)."""
        pass # To be implemented by models that support vector-based initialization (GADNN, etc)
