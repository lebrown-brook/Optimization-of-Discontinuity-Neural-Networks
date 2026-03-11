import numpy as np
import torch
import pygad
from visualize_results import plot_bab_iteration_best, analyze_and_plot_gadnn_regions

class BaBTrainer:
    def __init__(self,
                 model_class,
                 model_args,
                 X_train,
                 y_train,
                 ga_settings,
                 bab_settings=None, # Just to catch if someone passes it wrong, actually it is bab_settings
                 fitness_to_loss=None,
                 config_name="experiment",
                 shared_model=None,
                 delta_decay=1.0,
                 enable_plotting=True):
        """
        Branch and Bound Training Controller for GADNN.

        Replaces standard GA run with a multi-branch approach that periodically
        prunes and branches based on solution quality and regional diversity.

        Args:
            model_class: The class of the model (e.g., GADNN)
            model_args: Dictionary of arguments for initializing the model.
            X_train: Training data input (torch.Tensor).
            y_train: Training data targets (torch.Tensor).
            ga_settings: Dictionary of PyGAD settings (must include 'fitness_func').
            bab_settings: Dictionary containing BaB specific settings:
                - num_branches (int): Number of parallel branches (GA instances).
                - generations_per_bab_iteration (int): Generations to run per BaB step.
                - total_generations (int): Total generations (proxy for total budget).
                - perturbation_scale (float): Scale of noise for perturbation.
                - diversity_threshold (float): Hamming distance threshold for diversity.
            fitness_to_loss (callable): Function to convert fitness to loss for logging.
                                        Defaults to -fitness.
            shared_model: The model instance used in the fitness function (if any).
                          Used to update parameters like delta during training.
            delta_decay (float): Factor to decay delta by after each iteration.
            enable_plotting (bool): Whether to generate plots during training.
        """
        self.model_class = model_class
        self.model_args = model_args
        self.X_train = X_train
        self.y_train = y_train
        self.ga_settings = ga_settings
        self.bab_settings = bab_settings
        self.fitness_to_loss = fitness_to_loss if fitness_to_loss else lambda f: -f
        self.config_name = config_name
        self.delta_decay = delta_decay
        self.enable_plotting = enable_plotting

        # Unpack BaB settings with defaults
        self.num_branches = bab_settings.get("num_branches", 5)
        self.generations_per_iter = bab_settings.get("generations_per_bab_iteration", 50)
        self.total_generations = bab_settings.get("total_generations", 1000)
        self.perturbation_scale = bab_settings.get("perturbation_scale", 0.01)
        self.diversity_threshold = bab_settings.get("diversity_threshold", 0.05)
        self.p_dropout = bab_settings.get("p_dropout", 0.2)
        self.seed_length = bab_settings.get("seed_length", 1)
        # Derived settings
        self.num_bab_iterations = max(1, self.total_generations // self.generations_per_iter)

        # State
        self.branches = [] # List of pygad.GA instances
        self.branch_ids = [] # List of IDs corresponding to self.branches
        self.loss_histories = {} # ID -> List of loss values
        self.top_3_ids_prev = set() # IDs of top 3 branches from previous iteration
        self.parent_map = {} # ID -> Parent ID
        self.history_log = [] # List of iteration stats

        # Helper model instance for region calculation (weights are passed to forward)
        # We assume the model class can vary, but usually it is GADNN
        if shared_model is not None:
             self.model_template = shared_model
        else:
             self.model_template = self.model_class(**self.model_args)

    def train(self):
        """
        Executes the Branch and Bound training process.
        Returns:
            best_solution: The best solution found.
            best_fitness: The fitness of the best solution.
            last_ga_instance: One of the GA instances (for compatibility/plotting).
        """
        print(f"Starting BaB Training: {self.num_branches} branches, "
              f"{self.num_bab_iterations} iterations of {self.generations_per_iter} gens.")

        # 1. Initialize branches (Random Start)
        self._initialize_branches_random()

        best_global_solution = None
        best_global_fitness = -float('inf')

        for iter_idx in range(self.num_bab_iterations):
            print(f"\n--- BaB Iteration {iter_idx + 1}/{self.num_bab_iterations} ---")

            # 2. Run each branch
            for b_idx, ga_inst in enumerate(self.branches):
                # Optionally print progress
                # print(f"  Branch {b_idx + 1}: Running...") # Reduce spam
                ga_inst.run()

            # 3. Collect all solutions from all branches
            all_solutions = self._collect_solutions_from_branches()

            # --- Logging for Visualization ---
            current_iter_log = {
                "iteration": iter_idx,
                "branches": []
            }

            # Identify Top 3 Branches for this iteration (needed for logic and logging)
            # We aggregate best fitness per branch
            branch_stats = []
            for b_id, ga_inst in zip(self.branch_ids, self.branches):
                best_sol, best_fit, _ = ga_inst.best_solution()

                # Get local history
                full_hist = self.loss_histories.get(b_id, [])
                parent_id = self.parent_map.get(b_id)

                if parent_id and parent_id in self.loss_histories:
                    parent_len = len(self.loss_histories[parent_id])
                    local_hist = full_hist[parent_len:]
                else:
                    local_hist = full_hist

                stats = {
                    'id': b_id,
                    'parent_id': parent_id,
                    'fitness': best_fit,
                    'solution': best_sol,
                    'local_history': local_hist,
                    'full_history': full_hist # helpful for debug or final export
                }
                branch_stats.append(stats)

                current_iter_log["branches"].append(stats)

            self.history_log.append(current_iter_log)

            # Find iteration best and update global best (Process all solutions including non-best of pop)
            if not all_solutions:
                print("  [Warning] No solutions collected.")
                continue

            iter_best = max(all_solutions, key=lambda x: x['fitness'])
            if iter_best['fitness'] > best_global_fitness:
                best_global_fitness = iter_best['fitness']
                best_global_solution = iter_best['solution']
                print(f"  [New Global Best] Fitness: {best_global_fitness:.6f}")
            else:
                 print(f"  [Iteration Best] Fitness: {iter_best['fitness']:.6f} (Global: {best_global_fitness:.6f})")

            # Identify Top 3 Branches for this iteration
            # Sort by fitness descending
            sorted_branches = sorted(branch_stats, key=lambda x: x['fitness'], reverse=True)
            top_3 = sorted_branches[:3]
            self.top_3_ids_prev = set(b['id'] for b in top_3)
            #print all ids and fitnesses in sorted branches for debug in a row like 1.: ID: xxx, Fitness: yyy, 2.: ID: xxx, Fitness: yyy, ...
            print( " Branch Ranking: " + ", ".join([f"{i+1}. ID: {b['id']}, Fitness: {b['fitness']:.6f}" for i, b in enumerate(sorted_branches)]))


            if self.enable_plotting:
                # Plot top 3 solutions
                plot_bab_iteration_best(
                    self.model_template,
                    top_3,
                    self.X_train,
                    self.y_train,
                    iter_idx,
                    self.generations_per_iter,
                    config_name=self.config_name
                )

                # Analyze and plot regions for top 3 solutions
                x_min_scaled = self.X_train.min().item()
                x_max_scaled = self.X_train.max().item()
                margin = (x_max_scaled - x_min_scaled) * 0.1
                x_min_plot = x_min_scaled - margin
                x_max_plot = x_max_scaled + margin

                for rank, sol_data in enumerate(top_3):
                     analyze_and_plot_gadnn_regions(
                        self.model_template,
                        sol_data['solution'],
                        X_data=self.X_train,
                        Y_samples=self.y_train,
                        save_path=f"PyGAD_BAB_Plots/regions_{self.config_name}_iter_{iter_idx}_rank_{rank+1}.png",
                        x_min=x_min_plot, x_max=x_max_plot
                     )

            # Decay delta if applicable
            if self.delta_decay != 1.0:
                self._decay_delta()

            # If not the last iteration, perform branching (Filter -> Perturb -> Re-init)
            if iter_idx < self.num_bab_iterations - 1:
                # 4. Filter and Select Seeds
                seeds = self._select_seeds(all_solutions)
                msg_div = f" (Diversity Threshold: {self.diversity_threshold})" if "cont" not in self.config_name else ""
                print(f"  Selected {len(seeds)} seeds for next iteration{msg_div}.")

                # 5. Create new branches from seeds (Perturbation)
                self._update_branches_with_seeds(seeds, iter_idx + 1)
            #reduce delta by delta decay factor ^(num_bab_iterations) if applicable


        # If no solution found (unlikely), return something safe
        if best_global_solution is None and len(self.branches) > 0:
            best_global_solution, best_global_fitness, _ = self.branches[0].best_solution()

        # Update the first branch to hold the best solution for compatibility if needed
        # (e.g. if caller expects ga_instance.best_solution() to return the global best)
        if self.branches:
            # We can't easily force PyGAD to adopt a best solution without hacking population,
            # but we return the solution explicitly.
            pass

        return best_global_solution, best_global_fitness, self.branches[0], self.history_log

    def _initialize_branches_random(self):
        """Initializes num_branches GA instances with random populations."""
        self.branches = []
        self.branch_ids = []
        self.parent_map = {}
        for i in range(self.num_branches):
            b_id = f"iter_0_branch_{i}"
            self.branch_ids.append(b_id)
            self.loss_histories[b_id] = [] # separate empty history
            self.parent_map[b_id] = None

            # Create standard GA instance (random init is default in PyGAD if initial_population=None)
            ga_inst = self._create_ga_instance(initial_population=None, branch_id=b_id)
            self.branches.append(ga_inst)

    def _create_ga_instance(self, initial_population=None, branch_id=None):
        """Creates a PyGAD GA instance with strict generation limits."""
        settings = self.ga_settings.copy()

        # Override num_generations for the step
        settings["num_generations"] = self.generations_per_iter

        if initial_population is not None:
             settings["initial_population"] = initial_population
             # Ensure sol_per_pop matches the provided population size
             settings["sol_per_pop"] = len(initial_population)

        # Add on_generation callback
        # We need to preserve existing on_generation or wrap it?
        # The user's experiment_runner might have one.
        # But we want to log OUR history.

        user_on_gen = settings.get("on_generation")

        def wrapped_on_generation(ga_inst):
            # Log Loss
            if branch_id:
                _, best_fit, _ = ga_inst.best_solution()
                loss = self.fitness_to_loss(best_fit)
                if branch_id in self.loss_histories:
                    self.loss_histories[branch_id].append(loss)

            # Call user callback if exists
            if user_on_gen:
                user_on_gen(ga_inst)

        settings["on_generation"] = wrapped_on_generation

        return pygad.GA(**settings)

    def _collect_solutions_from_branches(self):
        """
        Collects all solutions from all branches and computes their regions.
        Returns a list of dicts: {'solution': np.array, 'fitness': float, 'regions': np.array, 'branch_id': str}
        """
        collected = []

        for ga_inst, b_id in zip(self.branches, self.branch_ids):
            population = ga_inst.population
            fitnesses = ga_inst.last_generation_fitness

            # Use 'sol_per_pop' to iterate or just len(population)
            count = len(population)

            for i in range(count):
                sol = population[i]
                fit = fitnesses[i]

                # Compute Regions
                # NOTE: We assume self.model_template is stateless (forward only needs weights)

                with torch.no_grad():
                    # Handle potential ID genes if Model is Flip-Aware but model class expects pure weights.
                    # We check if solution is longer than expected genes.
                    w = sol
                    if len(sol) > self.model_template.num_genes:
                        # Assume extra genes are at the end (Flip-Aware IDs)
                        w = sol[:self.model_template.num_genes]

                    # Forward pass to get regions
                    # Verify model supports return_regions
                    try:
                        _, regions = self.model_template.forward(w, self.X_train, return_regions=True)
                    except TypeError:
                         # Fallback if model doesn't support regions (should not happen with GADNN)
                         if "cont" not in self.config_name:
                             print("  [Warning] Model does not support region extraction. Regions set to zeros.")
                         regions = np.zeros((self.X_train.shape[0], 0))

                collected.append({
                    "solution": sol, # Keep full solution (including IDs if any) for perturbation
                    "fitness": fit,
                    "regions": regions,
                    "branch_id": b_id
                })

        return collected

    def _select_seeds(self, all_solutions):
        """
        Selects new starting points based on fitness and regional diversity.
        Returns a list of dicts: {'solution_list': [sol1, ...], 'branch_id': ...}
        """
        # Sort by fitness descending (higher is better)
        sorted_sols = sorted(all_solutions, key=lambda x: x['fitness'], reverse=True)

        # structure: list of {'head': solution_dict, 'members': [solution_dict, ...]}
        branches = []

        is_continuous = "cont" in self.config_name

        # Pass 1: Cluster unassigned solutions into branches
        for cand in sorted_sols:
            # Check if candidate belongs to an existing branch (close to head)
            assigned = False

            if not is_continuous:
                for branch in branches:
                    head = branch['head']
                    dist = self._calculate_hamming_distance(cand['regions'], head['regions'])

                    if dist <= self.diversity_threshold:
                        # Close enough: Add to members if space allows
                        if len(branch['members']) < self.seed_length:
                            branch['members'].append(cand)
                        assigned = True
                        break # Found its branch

            if not assigned:
                # Diverse from all existing branch heads.
                # Start new branch if we have slots
                if len(branches) < self.num_branches:
                    new_branch = {
                        'head': cand,
                        'members': [cand] # Head is first member
                    }
                    branches.append(new_branch)
                elif is_continuous:
                    break
                else:
                    # No slots for new branches and didn't fit in existing.
                    # It is discarded/ignored in this greedy approach.
                    pass

        # Pass 2: Fallback - Fill empty branch slots if we didn't find enough diverse heads
        if len(branches) < self.num_branches:
             if not is_continuous:
                 print(f"    [Info] Only found {len(branches)} diverse branches. Filling up with best remaining.")

             # Collect all currently used solutions (ref check or id check)
             used_solutions = set()
             for b in branches:
                 for m in b['members']:
                     # Use numpy array bytes or simple object identity if valid
                     # Using solution as key might be slow/tricky, let's use index in sorted list logic?
                     # Or just naive iteration.
                     pass

             # Simpler approach: Iterate sorted_sols again and pick anything not used
             # To do this efficiently, we can mark 'used' during Pass 1.
             # Re-implementing Pass 1 with 'used' flags would be cleaner but let's just do a linear scan
             # checking membership.

             for cand in sorted_sols:
                 if len(branches) >= self.num_branches:
                     break

                 # Check if cand is already in a branch
                 is_used = False
                 for b in branches:
                     # Check reference equality
                     if any(cand is m for m in b['members']):
                         is_used = True
                         break

                 if not is_used:
                     # Add as new branch head
                     new_branch = {
                         'head': cand,
                         'members': [cand]
                     }
                     branches.append(new_branch)

        #NON DIVERSE BRANCHES ARE NOT CLUSTERED, EACH HEAD IS ITS OWN BRANCH
        # Prepare final output format
        final_seeds = []
        for b in branches:
            # We construct an object that resembles the old 'seed' dict but contains list
            # We use the head's branch_id for continuity tracking
            head = b['head']
            members = [m['solution'] for m in b['members']]

            final_seeds.append({
                'solution_list': members, # New field
                'solution': head['solution'], # Keep primary for compatibility if needed elsewhere
                'regions': head['regions'],   # Keep primary
                'fitness': head['fitness'],   # Keep primary
                'branch_id': head.get('branch_id')
            })

        if len(final_seeds) != self.num_branches:
             print(f"    [Warning] Seed selection resulted in {len(final_seeds)} branches, expected {self.num_branches}.")

        return final_seeds

    def _calculate_hamming_distance(self, r1, r2):
        """Hamming distance between two region matrices (proportion of differing entries)."""
        # Convert to numpy if tensor
        if isinstance(r1, torch.Tensor): r1 = r1.cpu().numpy()
        if isinstance(r2, torch.Tensor): r2 = r2.cpu().numpy()

        if r1.size == 0 or r2.size == 0:
            return 0.0

        # Element-wise comparison
        matches = (r1 == r2)
        return 1.0 - np.mean(matches.astype(float) if hasattr(matches, 'astype') else float(matches))

    def _update_branches_with_seeds(self, seeds, iter_idx):
        """Creates new populations from seeds and re-initializes branches."""
        self.branches = []
        self.branch_ids = []
        self.parent_map = {}
        sol_per_pop = self.ga_settings.get("sol_per_pop", 20)

        for i, seed_data in enumerate(seeds):
            seed_list = seed_data['solution_list']
            parent_branch_id = seed_data.get('branch_id')

            new_id = f"iter_{iter_idx}_branch_{i}"
            self.branch_ids.append(new_id)
            self.parent_map[new_id] = parent_branch_id

            # Stitch Selection Logic
            if parent_branch_id in self.top_3_ids_prev:
                # Stitch: Copy parent history
                # Ensure we copy to avoid reference issues
                self.loss_histories[new_id] = list(self.loss_histories.get(parent_branch_id, []))
            else:
                # Treat distinct: Start fresh
                self.loss_histories[new_id] = []

            # Create population centered around this seed
            # Use seed_list solutions and fill remaining with perturbations of the FIRST seed (head)
            new_population = self._create_population_perturbation_multi(seed_list, sol_per_pop)

            ga_inst = self._create_ga_instance(initial_population=new_population, branch_id=new_id)
            self.branches.append(ga_inst)

    def _create_population_perturbation_multi(self, seed_list, pop_size):
        """
        Generates a population.
        Includes all solutions in seed_list.
        Fills the rest (pop_size - len(seed_list)) with perturbations of top seeds.
        """
        # Start with the seed list
        new_pop = [s.copy() for s in seed_list]

        num_needed = pop_size - len(new_pop)
        if num_needed <= 0:
            if num_needed < 0:
                print(f"  [Warning] Seed list larger than population size ({len(seed_list)} > {pop_size}). Truncating.")
            return np.array(new_pop[:pop_size])

        # Get number of seeds to use for perturbation
        k_seeds = self.bab_settings.get("seeds_used_for_perturbation", 1)
        k = min(k_seeds, len(seed_list))
        perturb_seeds = seed_list[:k]

        # Distribute the num_needed slots among the k seeds
        base_count = num_needed // k
        remainder = num_needed % k

        layer_params = getattr(self.model_template, "layer_params", [])

        for i in range(k):
            # How many offspring for this seed?
            count = base_count + (1 if i < remainder else 0)

            if count <= 0:
                continue

            seed_sol = perturb_seeds[i]

            # Split budget
            n_he = count // 3
            n_dropout = count // 3
            n_antithetic = count - n_he - n_dropout

            new_pop.extend(self._perturb_he_kaiming(seed_sol, n_he, layer_params))
            new_pop.extend(self._perturb_dropout(seed_sol, n_dropout, layer_params))
            new_pop.extend(self._perturb_antithetic(seed_sol, n_antithetic))

        return np.array(new_pop)

    def _create_population_perturbation(self, seed_sol, pop_size):
        # Legacy wrapper or unused
        return self._create_population_perturbation_multi([seed_sol], pop_size)

    def _decay_delta(self):
        """Decays the delta parameter of the models."""
        if hasattr(self.model_template, 'delta'):
            decay_amount = self.delta_decay ** self.generations_per_iter
            self.model_template.delta *= decay_amount
            if 'pwlu' in self.config_name:
                print(f"  [Delta Decay] Updated model_template.delta to {self.model_template.delta:.6f}")

    def _perturb_he_kaiming(self, seed_sol, count, layer_params):
        """Perturbs weights with He initialization variance."""
        mutants = []
        for _ in range(count):
            mutant = seed_sol.copy()
            if layer_params:
                for lp in layer_params:
                    # Only perturb weights (not biases or epsilon)
                    # He init variance: std = sqrt(2 / fan_in)
                    fan_in = lp['w_shape'][1]
                    std_he = np.sqrt(2.0 / fan_in)

                    w_slice = lp['w_slice']
                    w_len = w_slice.stop - w_slice.start

                    noise = np.random.normal(loc=0.0, scale=std_he, size=w_len)
                    mutant[w_slice] += noise
            else:
                # Fallback if layer_params not available
                noise = np.random.normal(loc=0.0, scale=self.perturbation_scale, size=seed_sol.shape)
                mutant += noise
            mutants.append(mutant)
        return mutants

    def _perturb_dropout(self, seed_sol, count, layer_params):
        """Randomly drops weights in one hidden layer."""
        # Identify "middle" layers
        middle_layer_indices = []
        if len(layer_params) > 2:
            middle_layer_indices = list(range(1, len(layer_params) - 1))

        # Dropout parameter p (probability to drop)
        p = self.p_dropout
        mutants = []

        for _ in range(count):
            mutant = seed_sol.copy()

            if middle_layer_indices:
                # Pick one random middle layer
                target_idx = int(np.random.choice(middle_layer_indices))
                target_layer = layer_params[target_idx]

                # Create mask for this layer's weights
                w_slice = target_layer['w_slice']
                w_len = w_slice.stop - w_slice.start

                # logic: 1 (keep) with prob 1-p, 0 (drop) with prob p
                keep_mask = np.random.binomial(1, 1.0 - p, size=w_len)

                # Apply mask (drop weights to 0)
                mutant[w_slice] *= keep_mask

            mutants.append(mutant)
        return mutants

    def _perturb_antithetic(self, seed_sol, count):
        """Adds +/- Gaussian noise."""
        mutants = []
        remaining = count
        while remaining > 0:
            # Base noise
            noise = np.random.normal(loc=0.0, scale=self.perturbation_scale, size=seed_sol.shape)

            # Positive mutation
            mutant_pos = seed_sol + noise
            mutants.append(mutant_pos)
            remaining -= 1

            # Negative mutation (if space allows)
            if remaining > 0:
                mutant_neg = seed_sol - noise
                mutants.append(mutant_neg)
                remaining -= 1
        return mutants
