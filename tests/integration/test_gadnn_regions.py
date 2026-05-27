import torch
import numpy as np
from src.models.gadnn import GADNN

def test_gadnn_regions():
    # Setup GADNN
    input_size = 2
    hidden_layers = [4, 4]
    output_size = 1
    discontinuous_indices = [0, 1] # Check multiple layers

    gadnn = GADNN(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        discontinuous_indices=discontinuous_indices,
        Disc_Activation="disc"
    )

    # Create random weights
    num_genes = gadnn.num_genes
    w = np.random.uniform(-1, 1, size=(num_genes,))

    # Create random input grid
    num_samples = 10
    x = np.random.uniform(-1, 1, size=(num_samples, input_size))
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # Capture regions
    X_region_matrix, eps_values = gadnn.capture_activation_regions(w, x_tensor)

    print("X_region_matrix shape:", X_region_matrix.shape)
    print("Number of epsilons:", len(eps_values))

    # Expected number of epsilons: 4 (layer 0) + 4 (layer 1) = 8
    expected_eps = 8
    assert len(eps_values) == expected_eps, f"Expected {expected_eps} epsilons, got {len(eps_values)}"

    # Expected matrix shape: (10, 8)
    assert X_region_matrix.shape == (num_samples, expected_eps), f"Expected shape {(num_samples, expected_eps)}, got {X_region_matrix.shape}"

    # Check if values are 0 or 1
    unique_vals = np.unique(X_region_matrix)
    print("Unique values in matrix:", unique_vals)
    assert np.all(np.isin(unique_vals, [0, 1])), "Matrix should only contain 0 and 1"

    print("Epsilon values sample:", eps_values[0])
    assert 'value' in eps_values[0], "Epsilon dictionary should contain 'value'"
    assert 'eps_globalind' in eps_values[0], "Epsilon dictionary should contain 'eps_globalind'"

    print("Test passed!")

if __name__ == "__main__":
    test_gadnn_regions()
