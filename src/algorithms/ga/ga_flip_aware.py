import numpy as np
import torch

# Constants
ID_GENES_COUNT = 3  # [Parent1_ID, Parent2_ID, Self_ID]
# Indices relative to end
IDX_PARENT1 = -3
IDX_PARENT2 = -2
IDX_SELF = -1

# Global storage for the region matrices of the previous generation
last_gen_region_matrices = None
# Storage for the current generation, populated during fitness evaluation
current_gen_region_matrices = None

def reset_flip_aware_state(sol_per_pop=None):
    """Resets the global state for a new experiment run."""
    global last_gen_region_matrices, current_gen_region_matrices
    last_gen_region_matrices = None
    if sol_per_pop is not None:
         # Initialize with None or empty dictionary to handle arbitrary indices safely
         current_gen_region_matrices = {}
    else:
         current_gen_region_matrices = {}

def store_child_regions(solution_idx, region_matrix):
    """Stores the computed regions for a child in the current generation buffer."""
    global current_gen_region_matrices
    if current_gen_region_matrices is None:
        current_gen_region_matrices = {}
    current_gen_region_matrices[solution_idx] = region_matrix

def calculate_flip_metric(child_regions, solution, sol_per_pop, alpha=0.1):
    """
    Calculates the flip metric (Hamming distance) given the child's computed regions and its genes (for parent IDs).
    Returns a positive value (alpha * avg_flips).
    """
    global last_gen_region_matrices

    #first generation doesnt have parents
    if last_gen_region_matrices is None:
        return 0.0

    p1_id = int(solution[IDX_PARENT1])
    p2_id = int(solution[IDX_PARENT2])

    parents_regions = []

    # Check bounds and existence
    # We use a dict for last_gen to be safe? Or list?
    # If list, we must ensure indices are valid.
    # PyGAD uses indices 0..sol_per_pop-1.

    if isinstance(last_gen_region_matrices, dict):
        if p1_id in last_gen_region_matrices:
            parents_regions.append(last_gen_region_matrices[p1_id])
        elif len(last_gen_region_matrices) > 0 and p1_id >= 0:
             # Only warn if we expect history (len > 0) and ID looks valid (>=0)
             print(f"[WARNING] Parent 1 ID {p1_id} not found in history (Gen Active).")

        if p2_id in last_gen_region_matrices:
            parents_regions.append(last_gen_region_matrices[p2_id])
        elif len(last_gen_region_matrices) > 0 and p2_id >= 0:
             print(f"[WARNING] Parent 2 ID {p2_id} not found in history (Gen Active).")
    elif isinstance(last_gen_region_matrices, (list, np.ndarray)):
        if 0 <= p1_id < len(last_gen_region_matrices): parents_regions.append(last_gen_region_matrices[p1_id])
        if 0 <= p2_id < len(last_gen_region_matrices): parents_regions.append(last_gen_region_matrices[p2_id])

    if not parents_regions:
        return 0.0

    total_flips = 0.0
    for p_reg in parents_regions:
        diff = np.abs(child_regions - p_reg)
        total_flips += np.sum(diff)

    #average over (X_data points * epsilon_params) and number of parents
    avg_flips = total_flips / (len(parents_regions) * np.prod(child_regions.shape))
    # Coefficient
    return alpha * avg_flips

# Wrapper to be hooked into on_fitness (executes after fitness, BEFORE crossover)
def get_on_fitness_wrapper(model, X_train):
    def on_fitness(ga_instance, population_fitness):
        """
        Executed after fitness calculation of the whole population.
        We use this to snapshot the region matrices of the current population (Gen N),
        which will become the parents for Gen N+1.
        """
        global last_gen_region_matrices

        current_pop = ga_instance.population
        pop_size = current_pop.shape[0]

        # New storage for this generation
        new_last_gen = {}

        # Iterate over ALL individuals (including elites that may not have triggered fitness_func this round)
        for i in range(pop_size):
            solution = current_pop[i]
            # 1. Update Self_ID (gene) to match current index
            # This ensures that when this individual is selected as a parent,
            # the child inherits this index as the Parent ID.
            solution[IDX_SELF] = i # In-place modification of genes

            # 2. Calculate Region Matrix
            # We must re-calculate (or initially calculate for elites) to be safe and consistent.
            # Slice off ID genes for the model
            weights_for_model = solution[:-ID_GENES_COUNT]

            # Note: We do this for everyone.
            # Optimization: If we had a reliable way to map fitness_func's cached regions to this 'i',
            # we could skip computation. But since solution_idx in fitness_func might not match 'i'
            # (especially with elites/sorting), we re-compute. Safest approach.

            with torch.no_grad():
                _, regions = model.forward(weights_for_model, X_train, return_regions=True)

            # Store in the snapshot
            new_last_gen[i] = regions
            #print all values of solution for debugging



        # Update global history
        last_gen_region_matrices = new_last_gen

        # Debug/Status
        # print(f"  [FlipAware] Gen {ga_instance.generations_completed}: Stored regions for {len(last_gen_region_matrices)} individuals.")

    return on_fitness


def get_custom_crossover_func(ID_GENES_COUNT=3):
    def crossover_func(parents, offspring_size, ga_instance):
        """
        Custom crossover that only crosses weights and propagates Parent IDs.
        """
        offspring = []
        idx = 0
        num_parents = parents.shape[0]
        while len(offspring) < offspring_size[0]:
            # Simple cyclic parent selection from the passed parents
            parent1_idx = idx % num_parents
            parent2_idx = (parent1_idx + 1 + (idx // num_parents)) % num_parents
                                            #rounded down division to avoid same parent
            parent1 = parents[parent1_idx]
            parent2 = parents[parent2_idx]

            # Create child container
            child = np.empty_like(parent1)

            # --- WEIGHTS CROSSOVER ---
            # Single Point Crossover on the Weight part
            gene_count_weights = parents.shape[1] - ID_GENES_COUNT
            if gene_count_weights > 1:
                crossover_point = np.random.randint(1, gene_count_weights)
            else:
                crossover_point = 0

            child[:crossover_point] = parent1[:crossover_point]
            child[crossover_point:gene_count_weights] = parent2[crossover_point:gene_count_weights]

            # --- ID ASSIGNMENT ---
            # Read Self_ID from parents (indices match what was stored in last generation)
            p1_self_id = int(parent1[IDX_SELF])
            p2_self_id = int(parent2[IDX_SELF])

            child[IDX_PARENT1] = p1_self_id
            child[IDX_PARENT2] = p2_self_id
            child[IDX_SELF] = -1 # Placeholder for child's own ID

            offspring.append(child)
            idx += 1

        return np.array(offspring)
    return crossover_func

def get_custom_mutation_func(mutation_type="random"):
    def mutation_func(offspring, ga_instance):
        """
        Custom mutation that avoids mutating the ID genes.
        """
        mutation_percent = ga_instance.mutation_percent_genes

        for i in range(offspring.shape[0]):
            gene_count_weights = offspring.shape[1] - ID_GENES_COUNT

            # Calculate number of genes to mutate
            num_mutations = int(gene_count_weights * (mutation_percent / 100))
            if num_mutations < 1:
                num_mutations = 1

            # Randomly select genes to mutate (excluding IDs)
            gene_indices = np.random.choice(range(gene_count_weights), num_mutations, replace=False)

            if mutation_type == "random":
                 # Adding random noise
                 random_range_low = ga_instance.init_range_low
                 random_range_high = ga_instance.init_range_high

                 random_values = np.random.uniform(random_range_low, random_range_high, size=num_mutations)
                 offspring[i, gene_indices] = offspring[i, gene_indices] + random_values
            else:
                # Default fallback if other types used (simplified)
                 offspring[i, gene_indices] += np.random.normal(0, 0.1, size=num_mutations)

        return offspring
    return mutation_func

