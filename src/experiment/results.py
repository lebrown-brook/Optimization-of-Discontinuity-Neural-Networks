from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TrainingResult:
    algorithm: str          # "backprop" | "ga" | "es" | "bab"
    variation: str          # "cont" | "disc" | "pwlu"
    function: str
    input_dim: int
    hidden_architecture: List[int]
    loss_function: str
    base_activation: bool   # True = ReLU, False = ELU

    final_loss: float
    loss_history: List[float]

    # Post-training: flat weight vector for GADNN, or model state for DNN
    best_weights: Optional[np.ndarray] = None
    trained_model: Optional[Any] = None  # model instance — needed for forward/region calls

    # Pre-training weights (before evolution / gradient steps), used for pre-training plots
    pre_weights: Optional[np.ndarray] = None

    # Algorithm-specific extras logged to CSV
    # Keys by algorithm:
    #   ga:       generations, pop_size, flip_aware
    #   es:       generations, pop_size, regions_discovered, stored_candidates, accepted_total, evaluated_total
    #   bab:      total_generations, num_branches, generations_per_iter, history_log
    #   backprop: epochs_run, learning_rate, lambda_reg, seed
    metadata: Dict[str, Any] = field(default_factory=dict)

    runtime_seconds: float = 0.0
    pre_delta: Optional[float] = None
