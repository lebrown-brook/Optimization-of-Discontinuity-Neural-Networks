import numpy as np
import torch
import pygad as pg
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Define a simple non-linear function for the regression problem
# f(w; x) = w0 * sin(w1 * x + w2) + w3
def model_func(w, x):
    """
    Simple non-linear model.
    w: parameters [w0, w1, w2, w3]
    x: input data
    """
    # Ensure inputs are tensors if using torch operations, or numpy if using numpy
    # Here we use numpy for simplicity in this standalone example,
    # but GADNN uses torch.
    return w[0] * np.sin(w[1] * x + w[2]) + w[3]

def fitness_func_factory(X, y_true):
    """
    Creates a fitness function with closure over data.
    """
    def fitness_func(ga_instance, solution, solution_idx):
        # solution is the vector of parameters 'w'
        y_pred = model_func(solution, X)

        # Mean Squared Error
        mse = np.mean((y_pred - y_true)**2)

        # Fitness should be maximized.
        # We use 1 / (mse + epsilon) to convert minimization to maximization.
        fitness = 1.0 / (mse + 1e-8)
        return fitness
    return fitness_func

def run_ga_experiment(X, y_true):
    """
    Sets up and runs the Genetic Algorithm.
    """

    num_genes = 4 # w0, w1, w2, w3
    fitness_function = fitness_func_factory(X, y_true)

    # --- PyGAD Instance Configuration ---
    ga_instance = pg.GA(

        # Number of generations to run the evolutionary process.
        # More generations allow for more refinement but take longer.
        num_generations=100,

        # Number of solutions (chromosomes) to select as parents for mating.
        # Should be less than sol_per_pop.
        num_parents_mating=5,

        # The fitness function to evaluate each solution.
        # Must accept (ga_instance, solution, solution_idx).
        fitness_func=fitness_function,

        # Number of solutions (individuals) in the population.
        # Larger populations maintain more diversity but increase computational cost per generation.
        sol_per_pop=20,

        # Number of parameters (genes) in each solution.
        # Corresponds to the size of the weight vector 'w'.
        num_genes=num_genes,

        # Lower bound for the random initialization of genes.
        init_range_low=-5,

        # Upper bound for the random initialization of genes.
        init_range_high=5,

        # Type of parent selection.
        # 'sss' = Steady State Selection
        # 'rws' = Roulette Wheel Selection
        # 'sus' = Stochastic Universal Sampling
        # 'rank' = Rank Selection
        # 'random' = Random Selection
        # 'tournament' = Tournament Selection
        parent_selection_type="sss",

        # Number of parents to keep in the next generation without change (Elitism).
        # -1 means keep all parents.
        keep_parents=2,

        # Type of crossover operation.
        # 'single_point', 'two_points', 'uniform', 'scattered'.
        crossover_type="single_point",

        # Type of mutation operation.
        # 'random', 'swap', 'inversion', 'scramble', 'adaptive'.
        mutation_type="random",

        # Percentage of genes to mutate in a solution.
        # Controls the exploration capability.
        mutation_percent_genes=25
    )

    print("Starting GA run...")
    start_time = datetime.now()

    ga_instance.run()

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"GA run finished in {duration}")

    return ga_instance

if __name__ == "__main__":
    # 1. Generate Synthetic Data
    # Target function: y = 2.5 * sin(1.5 * x + 0.5) + 1.0
    # True parameters: [2.5, 1.5, 0.5, 1.0]
    np.random.seed(42)
    X = np.linspace(-np.pi, np.pi, 100)
    true_w = [2.5, 1.5, 0.5, 1.0]
    y_true = model_func(true_w, X) + 0.1 * np.random.randn(*X.shape)

    # 2. Run Experiment
    ga_instance = run_ga_experiment(X, y_true)

    # 3. Analyze Results
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best Solution Parameters: {solution}")
    print(f"Best Fitness: {solution_fitness}")

    # 4. Visualize
    y_pred_best = model_func(solution, X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y_true, label="Noisy Data", alpha=0.6)
    plt.plot(X, model_func(true_w, X), 'g--', label="True Function", linewidth=2)
    plt.plot(X, y_pred_best, 'r-', label="GA Prediction", linewidth=2)
    plt.legend()
    plt.title(f"GA Regression Result\nBest Fitness: {solution_fitness:.4f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    # Save the plot instead of showing it, as we are in a headless environment usually
    # But user might run this locally.
    # plt.show()
    plt.savefig("Evolutionary Algorithms/ga_regression_result.png")
    print("Plot saved to Evolutionary Algorithms/ga_regression_result.png")

