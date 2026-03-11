import numpy as np
import torch
import pygad as pg
import matplotlib.pyplot as plt
import time
from GADNN import create_gadnn_model_func

# 1. Define Architecture
input_size = 1
hidden_layers = [8, 8]
output_size = 1
discontinuous_indices = [0] # Make the first hidden layer discontinuous

# Create the model function and get the number of genes
model_func, num_genes = create_gadnn_model_func(
    input_size=input_size,
    hidden_layers=hidden_layers,
    output_size=output_size,
    discontinuous_indices=discontinuous_indices,
    Disc_Activation="relu", # or "pwlu"
    Relu_true=True
)

print(f"Total parameters (genes): {num_genes}")

# 2. Generate Data
X = np.linspace(-np.pi, np.pi, 50).reshape(-1, 1)
y_true = np.sin(X) + 0.1 * np.random.randn(*X.shape)

# 3. Define Fitness Function
# Pre-convert data to tensor to avoid overhead in fitness function
X_tensor = torch.tensor(X, dtype=torch.float32)

def fitness_func(ga_instance, solution, solution_idx):
    # solution is 'w'
    # model_func expects (w, x)

    # Run model
    # model_func returns a tensor, convert to numpy
    y_pred = model_func(solution, X_tensor).detach().numpy()

    mse = np.mean((y_pred - y_true)**2)
    fitness = -mse
    return fitness

def create_ga_instance(fitness_func, num_genes):
    """
    Creates and returns a PyGAD GA instance with configured parameters.
    """
    ga_instance = pg.GA(
        # --- General Settings ---
        # Number of generations to run the evolutionary algorithm.
        num_generations=1000,
        # Number of solutions (chromosomes) in the population.
        sol_per_pop=10,
        # Total number of parameters (genes) in each solution.
        num_genes=num_genes,
        # The function to evaluate the fitness of a solution.
        fitness_func=fitness_func,

        # --- Mating Pool ---
        # Number of parents to select from the population for mating.
        num_parents_mating=4,
        # Number of parents to keep in the next generation (elitism). -1 means keep all.
        keep_parents=1,
        # Type of parent selection. "sss" = Steady State Selection.
        parent_selection_type="sss",

        # --- Initialization ---
        # Lower bound for random initialization of genes.
        init_range_low=-1,
        # Upper bound for random initialization of genes.
        init_range_high=1,

        # --- Crossover ---
        # Type of crossover operation. "single_point" splits parents at one point.
        crossover_type="single_point",

        # --- Mutation ---
        # Type of mutation operation. "random" adds random noise.
        mutation_type="random",
        # Percentage of genes to mutate in each solution.
        mutation_percent_genes=10
    )
    return ga_instance

# 4. Setup PyGAD
ga_instance = create_ga_instance(fitness_func, num_genes)

# 5. Run
print("Running GA...")
start_time = time.time()
ga_instance.run()
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# 6. Results
solution, solution_fitness, _ = ga_instance.best_solution()
print(f"Best Fitness: {solution_fitness}")

y_pred_best = model_func(solution, torch.tensor(X, dtype=torch.float32)).detach().numpy()

plt.figure()
plt.scatter(X, y_true, label="True")
plt.plot(X, y_pred_best, label="Pred", color="red")
plt.legend()
plt.title("GADNN Test")
plt.show()
