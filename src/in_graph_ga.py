"""Population search restricted to the observed replace-only edit graph.

Unlike de novo GraphGA, every offspring is an indexed neighbor reached by a
known BRICS replacement. This matches EditGraph's action space so that a
success gap cannot be attributed to library access alone.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from rdkit import Chem

from src.beam_planner import EditGraphStore, Properties, oracle_properties


class InGraphGA:
    """Genetic algorithm over observed replace-only edges only."""

    def __init__(
        self,
        store: EditGraphStore,
        scales: np.ndarray,
        population_size: int = 20,
        offspring_size: int = 20,
        oracle_budget: int = 29,
        max_outgoing: int = 500,
        replace_only: bool = True,
        seed: int = 42,
        use_cached_labels: bool = False,
    ):
        self.store = store
        self.scales = scales.astype(np.float64)
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.oracle_budget = oracle_budget
        self.max_outgoing = max_outgoing
        self.replace_only = replace_only
        self.seed = seed
        # If True, read DuckDB labels (observed-graph protocol, still counted).
        self.use_cached_labels = use_cached_labels

    def _props(self, smiles: str) -> Properties | None:
        if self.use_cached_labels:
            return self.store.molecule(smiles)
        return oracle_properties(smiles)

    def _neighbors(self, smiles: str) -> list[str]:
        return [
            edge["smiles_b"]
            for edge in self.store.outgoing(
                smiles, self.replace_only, self.max_outgoing
            )
        ]

    def optimize(
        self, start_smiles: str, target: Properties, episode_seed: int = 0
    ) -> tuple[str, Properties, int]:
        seed = self.seed + episode_seed
        random.seed(seed)
        np.random.seed(seed)

        cache: dict[str, tuple[float, Properties]] = {}

        def score(smiles: str) -> float:
            if smiles in cache:
                return cache[smiles][0]
            if len(cache) >= self.oracle_budget:
                return 0.0
            props = self._props(smiles)
            if props is None:
                return 0.0
            distance = float(
                np.linalg.norm((props.array() - target.array()) / self.scales)
            )
            value = float(np.exp(-distance))
            cache[smiles] = (value, props)
            return value

        start_score = score(start_smiles)
        if start_score <= 0.0 and start_smiles not in cache:
            props = self._props(start_smiles)
            if props is None:
                raise ValueError(f"Invalid InGraphGA start: {start_smiles}")
            return start_smiles, props, 0

        population = [start_smiles]
        attempts = 0
        while len(population) < self.population_size and attempts < 1000:
            attempts += 1
            parent = random.choice(population)
            neigh = self._neighbors(parent)
            if not neigh:
                continue
            child = random.choice(neigh)
            if child not in population:
                population.append(child)

        generations = 0
        stagnant = 0
        while len(cache) < self.oracle_budget and generations < 100:
            generations += 1
            previous = len(cache)
            scored = [(score(s), s) for s in population]
            scored.sort(key=lambda item: item[0], reverse=True)
            population = [s for _, s in scored[: self.population_size]]
            scores = np.asarray(
                [max(v, 1e-10) for v, _ in scored[: self.population_size]]
            )
            probabilities = scores / scores.sum()

            children: list[str] = []
            for _ in range(self.offspring_size):
                # Crossover: pick two parents; mutate the fitter by one graph edge.
                i, j = np.random.choice(
                    len(population), size=2, replace=True, p=probabilities
                )
                parent = population[int(i)] if scores[int(i)] >= scores[int(j)] else population[int(j)]
                neigh = self._neighbors(parent)
                if not neigh:
                    continue
                children.append(random.choice(neigh))
            if not children:
                # Forced mutate from best
                neigh = self._neighbors(population[0])
                if not neigh:
                    break
                children = [random.choice(neigh) for _ in range(min(20, len(neigh)))]
            population.extend(children)
            stagnant = stagnant + 1 if len(cache) == previous else 0
            if stagnant >= 10:
                break

        if not cache:
            props = self._props(start_smiles)
            assert props is not None
            return start_smiles, props, 1
        best_smiles, (_, best_props) = max(cache.items(), key=lambda item: item[1][0])
        return best_smiles, best_props, len(cache)
