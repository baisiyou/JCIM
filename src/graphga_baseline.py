"""Fixed-start adapter for the official mol-opt GraphGA operators."""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path

import numpy as np
from rdkit import Chem

from src.beam_planner import Properties, oracle_properties


def _load_graphga_operators(repository: Path):
    """Load official crossover/mutate files without importing mol-opt's heavy deps."""
    package_name = "_external_graph_ga"
    package = types.ModuleType(package_name)
    package.__path__ = [str(repository / "molopt" / "graph_ga")]
    sys.modules[package_name] = package

    def load(name: str):
        path = repository / "molopt" / "graph_ga" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load GraphGA module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    crossover_module = load("crossover")
    mutate_module = load("mutate")
    return crossover_module.crossover, mutate_module.mutate


class GraphGABaseline:
    """GraphGA using official operators and a target-distance objective."""

    def __init__(
        self,
        repository: Path,
        scales: np.ndarray,
        population_size: int = 20,
        offspring_size: int = 20,
        mutation_rate: float = 0.067,
        oracle_budget: int = 200,
        seed: int = 42,
    ):
        self.crossover, self.mutate = _load_graphga_operators(repository)
        self.scales = scales.astype(np.float64)
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.oracle_budget = oracle_budget
        self.seed = seed

    def optimize(
        self, start_smiles: str, target: Properties, episode_seed: int = 0
    ) -> tuple[str, Properties, int]:
        seed = self.seed + episode_seed
        random.seed(seed)
        np.random.seed(seed)

        start = Chem.MolFromSmiles(start_smiles)
        if start is None:
            raise ValueError(f"Invalid GraphGA start: {start_smiles}")

        cache: dict[str, tuple[float, Properties]] = {}

        def score(mol) -> float:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            if smiles in cache:
                return cache[smiles][0]
            if len(cache) >= self.oracle_budget:
                return 0.0
            props = oracle_properties(smiles)
            if props is None:
                return 0.0
            distance = np.linalg.norm(
                (props.array() - target.array()) / self.scales
            )
            value = float(np.exp(-distance))
            cache[smiles] = (value, props)
            return value

        population = [Chem.Mol(start)]
        attempts = 0
        while len(population) < self.population_size and attempts < 500:
            attempts += 1
            child = self.mutate(Chem.Mol(start), 1.0)
            if child is not None:
                population.append(child)
        population.extend(Chem.Mol(start) for _ in range(self.population_size - len(population)))

        generations = 0
        stagnant = 0
        while len(cache) < self.oracle_budget and generations < 100:
            generations += 1
            previous_calls = len(cache)
            scored = [(score(mol), mol) for mol in population if mol is not None]
            scored.sort(key=lambda item: item[0], reverse=True)
            population = [mol for _, mol in scored[: self.population_size]]
            scores = np.asarray([max(value, 1e-10) for value, _ in scored[: self.population_size]])
            probabilities = scores / scores.sum()

            children = []
            for _ in range(self.offspring_size):
                parent_indices = np.random.choice(
                    len(population), size=2, replace=True, p=probabilities
                )
                child = self.crossover(
                    Chem.Mol(population[int(parent_indices[0])]),
                    Chem.Mol(population[int(parent_indices[1])]),
                )
                if child is not None:
                    child = self.mutate(child, self.mutation_rate)
                if child is not None:
                    children.append(child)
            if not children:
                children = [
                    child
                    for child in (self.mutate(Chem.Mol(population[0]), 1.0) for _ in range(20))
                    if child is not None
                ]
            if not children:
                break
            population.extend(children)
            stagnant = stagnant + 1 if len(cache) == previous_calls else 0
            if stagnant >= 10:
                break

        if not cache:
            props = oracle_properties(start_smiles)
            assert props is not None
            return start_smiles, props, 1
        best_smiles, (_, best_props) = max(cache.items(), key=lambda item: item[1][0])
        return best_smiles, best_props, len(cache)
