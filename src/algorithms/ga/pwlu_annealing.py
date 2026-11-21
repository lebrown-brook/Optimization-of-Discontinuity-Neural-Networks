from time import perf_counter

import pygad


class PWLUAnnealingGA(pygad.GA):
    """Refresh scores before selection when the PWLU objective changes."""

    def __init__(self, model, delta_factor, **settings):
        super().__init__(**settings)
        self.model = model
        self.delta_factor = delta_factor
        self.refresh_evaluations = 0
        self.refresh_seconds = 0.0
        if self.stop_criteria and any(criterion[0] == "saturate" for criterion in self.stop_criteria):
            raise ValueError("PWLU annealing requires a fixed generation budget or a reach stop criterion.")

    def cal_pop_fitness(self):
        # PyGAD's saved histories span multiple deltas. Its parent and elite
        # caches are still usable once the current population is refreshed.
        save_solutions, save_best_solutions = self.save_solutions, self.save_best_solutions
        self.save_solutions = self.save_best_solutions = False
        try:
            return super().cal_pop_fitness()
        finally:
            self.save_solutions, self.save_best_solutions = save_solutions, save_best_solutions

    def run_loop_head(self, best_solution_fitness):
        if self.generations_completed:
            next_delta = self.model.delta * self.delta_factor
            if next_delta != self.model.delta:
                self.model.delta = next_delta
                start = perf_counter()
                parents = self.last_generation_parents
                elites = self.last_generation_elitism
                self.last_generation_parents = self.last_generation_elitism = None
                try:
                    self.last_generation_fitness = self.cal_pop_fitness()
                finally:
                    self.last_generation_parents = parents
                    self.last_generation_elitism = elites
                self.refresh_evaluations += len(self.population)
                self.refresh_seconds += perf_counter() - start

        super().run_loop_head(best_solution_fitness)
