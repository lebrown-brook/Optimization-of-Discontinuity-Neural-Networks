from dataclasses import dataclass
import heapq
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ESSolutionRecord:
    solution_id: int
    weights: np.ndarray
    fitness: float
    loss: float
    region_key: bytes


class RegionMemoryES:
    """
    Stores top-mu solutions per region and provides weighted parent sampling.

    Region lookup uses region_matrix.tobytes() as hashable dictionary key.
    """

    def __init__(self, mu_per_region: int, max_regions: int = 1000):
        if mu_per_region <= 0:
            raise ValueError("mu_per_region must be positive.")

        self.mu_per_region = int(mu_per_region)
        self.max_regions = int(max_regions)
        self.total_regions_discovered = 0
        self._min_elite_fitness = -float('inf')
        self._seen_regions: Dict[bytes, int] = {}
        
        self._next_solution_id = 0

        # region_key -> min-heap of (fitness, solution_id), worst-at-root
        self._region_heaps: Dict[bytes, List[tuple]] = {}

        # solution_id -> ESSolutionRecord
        self._solutions: Dict[int, ESSolutionRecord] = {}

        self._sampling_cache_dirty = True
        self._sampling_ids: List[int] = []
        self._sampling_weights: np.ndarray = np.array([], dtype=np.float64)

    def __len__(self) -> int:
        return len(self._solutions)

    @property
    def num_regions(self) -> int:
        return len(self._region_heaps)

    def _new_solution_id(self) -> int:
        sid = self._next_solution_id
        self._next_solution_id += 1
        return sid

    def _normalize_sampling_weights(self, fitness_values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(fitness_values)
        if not finite.any():
            return np.ones_like(fitness_values, dtype=np.float64) / max(1, fitness_values.size)

        safe = fitness_values.copy()
        min_finite = np.min(safe[finite])

        # Shift to non-negative and add epsilon to avoid zero-sum vector.
        safe = safe - min_finite
        safe[~finite] = 0.0
        safe = safe + 1e-12

        total = safe.sum()
        if total <= 0.0:
            return np.ones_like(safe, dtype=np.float64) / max(1, safe.size)

        return safe / total

    def _rebuild_sampling_cache(self) -> None:
        self._sampling_ids = list(self._solutions.keys())
        if not self._sampling_ids:
            self._sampling_weights = np.array([], dtype=np.float64)
            self._sampling_cache_dirty = False
            return

        fitness_values = np.array(
            [self._solutions[sid].fitness for sid in self._sampling_ids],
            dtype=np.float64,
        )
        self._sampling_weights = self._normalize_sampling_weights(fitness_values)
        self._sampling_cache_dirty = False

    def _prune_regions(self) -> None:
        if len(self._region_heaps) <= self.max_regions:
            return

        region_max_fit = {
            rid: max(f for f, sid in heap) for rid, heap in self._region_heaps.items()
        }
        
        sorted_regions = sorted(region_max_fit.keys(), key=lambda r: region_max_fit[r], reverse=True)
        bad_regions = sorted_regions[self.max_regions:]
        
        for bad_r in bad_regions:
            bad_heap = self._region_heaps.pop(bad_r)
            for _, sid in bad_heap:
                self._solutions.pop(sid, None)

        self._min_elite_fitness = float(region_max_fit[sorted_regions[self.max_regions - 1]])
        self._sampling_cache_dirty = True

    def get_visits(self, region_key: bytes) -> int:
        return self._seen_regions.get(region_key, 0)

    def add_candidate(self, weights: np.ndarray, fitness: float, loss: float, region_matrix: np.ndarray) -> bool:
        """
        Adds a candidate if it belongs to a new region, if region list has space,
        or if it outperforms the current worst candidate in that region.

        Returns True only when the candidate is accepted into memory.
        """
        region_key = np.ascontiguousarray(region_matrix).tobytes()
        
        if region_key not in self._seen_regions:
            self._seen_regions[region_key] = 0
            self.total_regions_discovered += 1
            
        if region_key not in self._region_heaps and len(self._region_heaps) >= self.max_regions and fitness < self._min_elite_fitness:
            return False

        heap = self._region_heaps.setdefault(region_key, [])

        if len(heap) < self.mu_per_region:
            sid = self._new_solution_id()
            record = ESSolutionRecord(
                solution_id=sid,
                weights=np.array(weights, dtype=np.float32, copy=True),
                fitness=float(fitness),
                loss=float(loss),
                region_key=region_key,
            )
            self._solutions[sid] = record
            heapq.heappush(heap, (record.fitness, sid))
            self._sampling_cache_dirty = True
            return True

        worst_fitness, worst_sid = heap[0]
        if float(fitness) <= float(worst_fitness):
            return False

        sid = self._new_solution_id()
        record = ESSolutionRecord(
            solution_id=sid,
            weights=np.array(weights, dtype=np.float32, copy=True),
            fitness=float(fitness),
            loss=float(loss),
            region_key=region_key,
        )
        self._solutions[sid] = record

        _, removed_sid = heapq.heapreplace(heap, (record.fitness, sid))
        self._solutions.pop(removed_sid, None)

        self._sampling_cache_dirty = True
        return True

    def sample_parents(self, num_parents: int, rng: np.random.Generator) -> List[ESSolutionRecord]:
        self._prune_regions()

        if num_parents <= 0 or not self._solutions:
            return []

        if self._sampling_cache_dirty:
            self._rebuild_sampling_cache()

        idx = rng.choice(
            len(self._sampling_ids),
            size=num_parents,
            replace=True,
            p=self._sampling_weights,
        )
        
        selected = [self._solutions[self._sampling_ids[i]] for i in idx]
        for parent in selected:
            if parent.region_key in self._seen_regions:
                self._seen_regions[parent.region_key] += 1
                
        return selected

    def get_best(self) -> Optional[ESSolutionRecord]:
        if not self._solutions:
            return None

        best_sid = max(self._solutions.keys(), key=lambda sid: self._solutions[sid].fitness)
        return self._solutions[best_sid]
