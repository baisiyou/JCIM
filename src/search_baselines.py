"""Exact and classical graph-search baselines for EditGraph evaluation."""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from src.beam_planner import (
    BeamPlanner,
    EditGraphStore,
    EditStep,
    EffectScorer,
    Properties,
    SearchState,
    oracle_properties,
)
from src.featurize import EDIT_TYPE_TO_ID, smiles_to_fp


@dataclass
class PlannerResult:
    smiles: str
    properties: Properties
    path: list[EditStep] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def _stats_template() -> dict[str, int]:
    return {
        "expanded_states": 0,
        "scored_edges": 0,
        "oracle_calls": 0,
    }


class BudgetMatchedBeamPlanner(BeamPlanner):
    """Beam search that stops after a fixed number of live oracle queries."""

    def __init__(self, *args, oracle_budget: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_budget = oracle_budget

    def _budget_exhausted(self) -> bool:
        return (
            self.oracle_budget is not None
            and self.last_stats["oracle_calls"] >= self.oracle_budget
        )

    def plan(self, target: Properties, retrieval_k: int = 10, start_smiles: str | None = None):
        self.last_stats = _stats_template()
        if start_smiles is None:
            raise ValueError("Budget-matched planners require a fixed start_smiles")
        props = oracle_properties(start_smiles)
        self.last_stats["oracle_calls"] += 1
        if props is None:
            raise ValueError(f"Invalid start: {start_smiles}")
        beam = [
            SearchState(
                smiles=start_smiles,
                properties=props,
                score=self.distance(props, target),
                visited=frozenset({start_smiles}),
            )
        ]
        best = beam[0]
        if self.succeeds(best.properties, target):
            return best

        for depth in range(1, self.max_steps + 1):
            if self._budget_exhausted():
                break
            expanded: list[SearchState] = []
            for state in beam:
                if self._budget_exhausted():
                    break
                self.last_stats["expanded_states"] += 1
                edges = [
                    e
                    for e in self.store.outgoing(
                        state.smiles, self.replace_only, self.max_outgoing
                    )
                    if e["smiles_b"] not in state.visited
                ]
                if not edges:
                    continue
                self.last_stats["scored_edges"] += len(edges)

                if self.ranker == "random":
                    n_select = min(self.actions_per_state, len(edges))
                    selected = self.rng.choice(len(edges), size=n_select, replace=False)
                    predicted = np.stack([e["stored_delta"] for e in edges])
                elif self.ranker == "live_oracle":
                    scores = []
                    for edge in edges:
                        if self._budget_exhausted():
                            break
                        next_props = oracle_properties(edge["smiles_b"])
                        self.last_stats["oracle_calls"] += 1
                        if next_props is None:
                            scores.append(np.inf)
                            continue
                        scores.append(self.distance(next_props, target))
                    selected = np.argsort(scores)[: self.actions_per_state]
                    predicted = np.zeros((len(edges), 3))
                else:
                    # effect / legacy oracle-stored
                    if self.ranker == "effect":
                        assert self.scorer is not None
                        predicted = self.scorer.predict(
                            [state.smiles] * len(edges),
                            [e["frag_old"] for e in edges],
                            [e["frag_new"] for e in edges],
                            [e["edit_type"] for e in edges],
                        )
                    else:
                        predicted = np.stack([e["stored_delta"] for e in edges])
                        self.last_stats["oracle_calls"] += len(edges)
                    pred_props = state.properties.array()[None, :] + predicted
                    pred_dist = np.linalg.norm(
                        (pred_props - target.array()[None, :]) / self.scales[None, :],
                        axis=1,
                    )
                    selected = np.argsort(pred_dist)[: self.actions_per_state]

                for idx in selected:
                    if self._budget_exhausted():
                        break
                    edge = edges[int(idx)]
                    pred_delta = predicted[int(idx)].astype(np.float64)
                    if self.oracle_correction or self.ranker == "live_oracle":
                        next_props = oracle_properties(edge["smiles_b"])
                        self.last_stats["oracle_calls"] += 1
                        if next_props is None:
                            continue
                        oracle_delta = next_props.array() - state.properties.array()
                    else:
                        pred_arr = state.properties.array() + pred_delta
                        next_props = Properties(*pred_arr.tolist())
                        oracle_delta = pred_delta.copy()
                    score = self.distance(next_props, target) + self.step_penalty * depth
                    step = EditStep(
                        source=state.smiles,
                        target=edge["smiles_b"],
                        edit_type=edge["edit_type"],
                        frag_old=edge["frag_old"],
                        frag_new=edge["frag_new"],
                        predicted_delta=pred_delta,
                        oracle_delta=oracle_delta,
                        oracle_properties=next_props,
                    )
                    expanded.append(
                        SearchState(
                            smiles=edge["smiles_b"],
                            properties=next_props,
                            score=score,
                            path=state.path + [step],
                            visited=state.visited | {edge["smiles_b"]},
                        )
                    )
            if not expanded:
                break
            unique: dict[str, SearchState] = {}
            for state in sorted(expanded, key=lambda s: s.score):
                unique.setdefault(state.smiles, state)
            beam = list(unique.values())[: self.beam_width]
            if beam[0].score < best.score:
                best = beam[0]
            success = [s for s in beam if self.succeeds(s.properties, target)]
            if success:
                return min(success, key=lambda s: (len(s.path), s.score))
        return best


class _GraphSearchBase:
    """Shared helpers for exact graph search baselines."""

    def __init__(
        self,
        store: EditGraphStore,
        scales: np.ndarray,
        tolerances: np.ndarray,
        max_outgoing: int | None = None,
        replace_only: bool = True,
    ):
        self.store = store
        self.scales = scales.astype(np.float64)
        self.tolerances = tolerances.astype(np.float64)
        # Exact cached search uses the full adjacency list (None = uncapped).
        self.max_outgoing = max_outgoing
        self.replace_only = replace_only
        self.last_stats = _stats_template()

    def distance(self, props: Properties, target: Properties) -> float:
        return float(np.linalg.norm((props.array() - target.array()) / self.scales))

    def succeeds(self, props: Properties, target: Properties) -> bool:
        return bool(np.all(np.abs(props.array() - target.array()) <= self.tolerances))

    def _outgoing_edges(self, state: SearchState) -> list[dict]:
        return [
            e
            for e in self.store.outgoing(
                state.smiles, self.replace_only, self.max_outgoing
            )
            if e["smiles_b"] not in state.visited
        ]

    def _child_state(
        self,
        state: SearchState,
        edge: dict,
        next_props: Properties,
        depth: int,
        score: float,
    ) -> SearchState:
        step = EditStep(
            source=state.smiles,
            target=edge["smiles_b"],
            edit_type=edge["edit_type"],
            frag_old=edge["frag_old"],
            frag_new=edge["frag_new"],
            predicted_delta=edge["stored_delta"],
            oracle_delta=next_props.array() - state.properties.array(),
            oracle_properties=next_props,
        )
        return SearchState(
            smiles=edge["smiles_b"],
            properties=next_props,
            score=score,
            path=state.path + [step],
            visited=state.visited | {edge["smiles_b"]},
        )


class ExactGraphBFSPlanner(_GraphSearchBase):
    """Full depth-limited BFS using indexed molecule properties (transductive ceiling).

    When episode targets are built from replace-only random walks on the same graph,
    this baseline should reach ~100% success on endpoint_walk suites because the
    generating walk endpoint lies in the searched neighborhood.
    """

    def __init__(
        self,
        store: EditGraphStore,
        scales: np.ndarray,
        tolerances: np.ndarray,
        max_depth: int = 3,
        max_outgoing: int | None = None,
        replace_only: bool = True,
    ):
        super().__init__(store, scales, tolerances, max_outgoing, replace_only)
        self.max_depth = max_depth

    def _cached_props(self, smiles: str) -> Properties | None:
        return self.store.molecule(smiles)

    def plan(self, target: Properties, start_smiles: str) -> SearchState:
        self.last_stats = _stats_template()
        start_props = self._cached_props(start_smiles)
        if start_props is None:
            raise ValueError(start_smiles)
        root = SearchState(
            smiles=start_smiles,
            properties=start_props,
            score=self.distance(start_props, target),
            visited=frozenset({start_smiles}),
        )
        best = root
        if self.succeeds(start_props, target):
            return root

        queue: list[tuple[int, SearchState]] = [(0, root)]
        while queue:
            depth, state = queue.pop(0)
            if depth >= self.max_depth:
                continue
            self.last_stats["expanded_states"] += 1
            edges = self._outgoing_edges(state)
            self.last_stats["scored_edges"] += len(edges)
            for edge in edges:
                next_props = self._cached_props(edge["smiles_b"])
                if next_props is None:
                    continue
                child = self._child_state(
                    state,
                    edge,
                    next_props,
                    depth + 1,
                    self.distance(next_props, target) + 0.02 * (depth + 1),
                )
                if child.score < best.score:
                    best = child
                if self.succeeds(next_props, target):
                    return child
                queue.append((depth + 1, child))
        return best


class CachedLabelBestFirstPlanner(_GraphSearchBase):
    """Best-first / A* search with cached indexed labels instead of live oracle calls."""

    def __init__(
        self,
        store: EditGraphStore,
        scales: np.ndarray,
        tolerances: np.ndarray,
        max_steps: int = 3,
        max_outgoing: int | None = None,
        step_penalty: float = 0.02,
        replace_only: bool = True,
    ):
        super().__init__(store, scales, tolerances, max_outgoing, replace_only)
        self.max_steps = max_steps
        self.step_penalty = step_penalty

    def plan(self, target: Properties, start_smiles: str) -> SearchState:
        self.last_stats = _stats_template()
        start_props = self.store.molecule(start_smiles)
        if start_props is None:
            raise ValueError(start_smiles)
        start = SearchState(
            smiles=start_smiles,
            properties=start_props,
            score=self.distance(start_props, target),
            visited=frozenset({start_smiles}),
        )
        best = start
        if self.succeeds(start_props, target):
            return start

        open_heap: list[tuple[float, int, int, SearchState]] = [
            (start.score, 0, 0, start)
        ]
        counter = 0
        while open_heap:
            _, depth, _, state = heapq.heappop(open_heap)
            if depth >= self.max_steps:
                continue
            self.last_stats["expanded_states"] += 1
            edges = self._outgoing_edges(state)
            self.last_stats["scored_edges"] += len(edges)
            for edge in edges:
                next_props = self.store.molecule(edge["smiles_b"])
                if next_props is None:
                    continue
                h = self.distance(next_props, target)
                f = self.step_penalty * (depth + 1) + h
                child = self._child_state(state, edge, next_props, depth + 1, f)
                if h < best.score:
                    best = child
                if self.succeeds(next_props, target):
                    return child
                counter += 1
                heapq.heappush(open_heap, (f, depth + 1, counter, child))
        return best


class BudgetExactNeighborPlanner(_GraphSearchBase):
    """Greedy depth expansion that live-oracles every outgoing neighbor within budget."""

    def __init__(
        self,
        store: EditGraphStore,
        scales: np.ndarray,
        tolerances: np.ndarray,
        max_steps: int = 3,
        max_outgoing: int = 500,
        replace_only: bool = True,
        oracle_budget: int = 200,
    ):
        super().__init__(store, scales, tolerances, max_outgoing, replace_only)
        self.max_steps = max_steps
        self.oracle_budget = oracle_budget

    def _budget_exhausted(self) -> bool:
        return self.last_stats["oracle_calls"] >= self.oracle_budget

    def plan(self, target: Properties, start_smiles: str) -> SearchState:
        self.last_stats = _stats_template()
        start_props = oracle_properties(start_smiles)
        self.last_stats["oracle_calls"] += 1
        if start_props is None:
            raise ValueError(start_smiles)
        state = SearchState(
            smiles=start_smiles,
            properties=start_props,
            score=self.distance(start_props, target),
            visited=frozenset({start_smiles}),
        )
        best = state
        if self.succeeds(start_props, target):
            return state

        for depth in range(1, self.max_steps + 1):
            if self._budget_exhausted():
                break
            self.last_stats["expanded_states"] += 1
            edges = self._outgoing_edges(state)
            self.last_stats["scored_edges"] += len(edges)
            scored: list[tuple[float, SearchState]] = []
            for edge in edges:
                if self._budget_exhausted():
                    break
                next_props = oracle_properties(edge["smiles_b"])
                self.last_stats["oracle_calls"] += 1
                if next_props is None:
                    continue
                child = self._child_state(
                    state,
                    edge,
                    next_props,
                    depth,
                    self.distance(next_props, target) + 0.02 * depth,
                )
                scored.append((child.score, child))
                if self.succeeds(next_props, target):
                    return child
            if not scored:
                break
            state = min(scored, key=lambda item: item[0])[1]
            if state.score < best.score:
                best = state
        return best


class ExhaustiveDepth3Planner:
    """Depth-limited exhaustive expansion with live oracle labels only."""

    def __init__(
        self,
        store: EditGraphStore,
        scales: np.ndarray,
        tolerances: np.ndarray,
        max_depth: int = 3,
        max_outgoing: int = 500,
        replace_only: bool = True,
        max_leaf_evals: int | None = None,
    ):
        self.store = store
        self.scales = scales.astype(np.float64)
        self.tolerances = tolerances.astype(np.float64)
        self.max_depth = max_depth
        self.max_outgoing = max_outgoing
        self.replace_only = replace_only
        self.max_leaf_evals = max_leaf_evals
        self.last_stats = _stats_template()
        self._graph = _GraphSearchBase(
            store, scales, tolerances, max_outgoing, replace_only
        )

    def distance(self, props: Properties, target: Properties) -> float:
        return float(np.linalg.norm((props.array() - target.array()) / self.scales))

    def succeeds(self, props: Properties, target: Properties) -> bool:
        return bool(np.all(np.abs(props.array() - target.array()) <= self.tolerances))

    def plan(self, target: Properties, start_smiles: str) -> SearchState:
        self.last_stats = _stats_template()
        start_props = oracle_properties(start_smiles)
        self.last_stats["oracle_calls"] += 1
        if start_props is None:
            raise ValueError(start_smiles)
        root = SearchState(
            smiles=start_smiles,
            properties=start_props,
            score=self.distance(start_props, target),
            visited=frozenset({start_smiles}),
        )
        best = root
        if self.succeeds(start_props, target):
            return root

        queue: list[tuple[int, SearchState]] = [(0, root)]
        leaves: list[SearchState] = []
        while queue:
            if self.max_leaf_evals is not None and len(leaves) >= self.max_leaf_evals:
                break
            depth, state = queue.pop(0)
            if depth >= self.max_depth:
                leaves.append(state)
                continue
            self.last_stats["expanded_states"] += 1
            edges = self._graph._outgoing_edges(state)
            self.last_stats["scored_edges"] += len(edges)
            for edge in edges:
                next_props = oracle_properties(edge["smiles_b"])
                self.last_stats["oracle_calls"] += 1
                if next_props is None:
                    continue
                child = self._graph._child_state(
                    state,
                    edge,
                    next_props,
                    depth + 1,
                    self.distance(next_props, target) + 0.02 * (depth + 1),
                )
                if child.score < best.score:
                    best = child
                if self.succeeds(next_props, target):
                    leaves.append(child)
                else:
                    queue.append((depth + 1, child))

        success = [s for s in leaves if self.succeeds(s.properties, target)]
        if success:
            return min(success, key=lambda s: (len(s.path), s.score))
        return best


class AStarPlanner:
    """Best-first search with admissible heuristic h = normalized property distance."""

    def __init__(
        self,
        store: EditGraphStore,
        scales: np.ndarray,
        tolerances: np.ndarray,
        max_steps: int = 3,
        max_outgoing: int = 200,
        step_penalty: float = 0.02,
        replace_only: bool = True,
        branch_limit: int | None = None,
    ):
        self.store = store
        self.scales = scales.astype(np.float64)
        self.tolerances = tolerances.astype(np.float64)
        self.max_steps = max_steps
        self.max_outgoing = max_outgoing
        self.step_penalty = step_penalty
        self.replace_only = replace_only
        self.branch_limit = branch_limit
        self.last_stats = _stats_template()
        self._graph = _GraphSearchBase(
            store, scales, tolerances, max_outgoing, replace_only
        )

    def distance(self, props: Properties, target: Properties) -> float:
        return float(np.linalg.norm((props.array() - target.array()) / self.scales))

    def succeeds(self, props: Properties, target: Properties) -> bool:
        return bool(np.all(np.abs(props.array() - target.array()) <= self.tolerances))

    def plan(self, target: Properties, start_smiles: str) -> SearchState:
        self.last_stats = _stats_template()
        start_props = oracle_properties(start_smiles)
        self.last_stats["oracle_calls"] += 1
        if start_props is None:
            raise ValueError(start_smiles)
        start = SearchState(
            smiles=start_smiles,
            properties=start_props,
            score=self.distance(start_props, target),
            visited=frozenset({start_smiles}),
        )
        best = start
        if self.succeeds(start_props, target):
            return start

        open_heap: list[tuple[float, int, int, SearchState]] = [
            (start.score, 0, 0, start)
        ]
        counter = 0
        while open_heap:
            _, depth, _, state = heapq.heappop(open_heap)
            if depth >= self.max_steps:
                continue
            self.last_stats["expanded_states"] += 1
            edges = self._graph._outgoing_edges(state)
            if self.branch_limit is not None:
                edges = edges[: self.branch_limit]
            self.last_stats["scored_edges"] += len(edges)
            for edge in edges:
                next_props = oracle_properties(edge["smiles_b"])
                self.last_stats["oracle_calls"] += 1
                if next_props is None:
                    continue
                g = self.step_penalty * (depth + 1)
                h = self.distance(next_props, target)
                f = g + h
                child = self._graph._child_state(state, edge, next_props, depth + 1, f)
                if h < best.score:
                    best = child
                if self.succeeds(next_props, target):
                    return child
                counter += 1
                heapq.heappush(open_heap, (f, depth + 1, counter, child))
        return best


class MatchedPairDeltaScorer:
    """Average observed training deltas for matched fragment replacements."""

    def __init__(
        self,
        train_pairs: Path,
        fp_bits: int = 2048,
        fp_radius: int = 2,
        max_rows: int = 200_000,
    ):
        df = pd.read_parquet(
            train_pairs,
            columns=[
                "smiles_a",
                "edit_type",
                "frag_old",
                "frag_new",
                "delta_qed",
                "delta_logp",
                "delta_mw",
            ],
        ).head(max_rows)
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.global_mean = df[["delta_qed", "delta_logp", "delta_mw"]].mean().to_numpy()
        grouped = (
            df.groupby(["edit_type", "frag_old", "frag_new"], dropna=False)[
                ["delta_qed", "delta_logp", "delta_mw"]
            ]
            .mean()
            .reset_index()
        )
        self.lookup: dict[tuple[str, str, str], np.ndarray] = {}
        for row in grouped.itertuples(index=False):
            key = (
                str(row.edit_type),
                str(row.frag_old or ""),
                str(row.frag_new or ""),
            )
            self.lookup[key] = np.asarray(
                [row.delta_qed, row.delta_logp, row.delta_mw], dtype=np.float32
            )
        self._fp_cache: dict[str, np.ndarray] = {"": np.zeros(fp_bits, dtype=np.float32)}

    def _fp(self, smiles: str) -> np.ndarray:
        if smiles not in self._fp_cache:
            self._fp_cache[smiles] = smiles_to_fp(
                smiles, n_bits=self.fp_bits, radius=self.fp_radius
            )
        return self._fp_cache[smiles]

    def predict(
        self,
        sources: list[str],
        frag_old: list[str],
        frag_new: list[str],
        edit_types: list[str],
    ) -> np.ndarray:
        out = np.zeros((len(sources), 3), dtype=np.float32)
        for i, (src, old, new, kind) in enumerate(
            zip(sources, frag_old, frag_new, edit_types)
        ):
            key = (kind, old or "", new or "")
            out[i] = self.lookup.get(key, self.global_mean)
        return out


class SklearnEffectScorer:
    """Ridge or gradient-boosted trees on the same features as the MLP."""

    def __init__(self, model, fp_bits: int, fp_radius: int, y_mean: np.ndarray, y_std: np.ndarray):
        self.model = model
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)
        self._fp_cache: dict[str, np.ndarray] = {"": np.zeros(fp_bits, dtype=np.float32)}

    def _fp(self, smiles: str) -> np.ndarray:
        if smiles not in self._fp_cache:
            self._fp_cache[smiles] = smiles_to_fp(
                smiles, n_bits=self.fp_bits, radius=self.fp_radius
            )
        return self._fp_cache[smiles]

    def _features(
        self,
        sources: list[str],
        frag_old: list[str],
        frag_new: list[str],
        edit_types: list[str],
    ) -> np.ndarray:
        rows = []
        for src, old, new, kind in zip(sources, frag_old, frag_new, edit_types):
            edit = np.zeros(3, dtype=np.float32)
            edit[EDIT_TYPE_TO_ID.get(kind, 0)] = 1.0
            rows.append(
                np.concatenate([self._fp(src), self._fp(old), self._fp(new), edit])
            )
        return np.stack(rows)

    def predict(
        self,
        sources: list[str],
        frag_old: list[str],
        frag_new: list[str],
        edit_types: list[str],
    ) -> np.ndarray:
        if not sources:
            return np.empty((0, 3), dtype=np.float32)
        x = self._features(sources, frag_old, frag_new, edit_types)
        pred_std = np.asarray(self.model.predict(x), dtype=np.float32)
        return pred_std * self.y_std + self.y_mean


def train_sklearn_effect_scorer(
    train_pairs: Path,
    model_kind: str,
    fp_bits: int = 2048,
    fp_radius: int = 2,
    max_rows: int = 200_000,
    seed: int = 42,
) -> SklearnEffectScorer:
    df = pd.read_parquet(
        train_pairs,
        columns=[
            "smiles_a",
            "edit_type",
            "frag_old",
            "frag_new",
            "delta_qed",
            "delta_logp",
            "delta_mw",
            "qed_a",
            "logp_a",
            "mw_a",
        ],
    ).head(max_rows)
    y = df[["delta_qed", "delta_logp", "delta_mw"]].to_numpy(dtype=np.float32)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0
    y_stdized = (y - y_mean) / y_std

    scorer = SklearnEffectScorer.__new__(SklearnEffectScorer)
    scorer.fp_bits = fp_bits
    scorer.fp_radius = fp_radius
    scorer.y_mean = y_mean
    scorer.y_std = y_std
    scorer._fp_cache = {"": np.zeros(fp_bits, dtype=np.float32)}

    rows = []
    for row in df.itertuples(index=False):
        edit = np.zeros(3, dtype=np.float32)
        edit[EDIT_TYPE_TO_ID.get(row.edit_type, 0)] = 1.0
        rows.append(
            np.concatenate(
                [
                    scorer._fp(row.smiles_a),
                    scorer._fp(str(row.frag_old or "")),
                    scorer._fp(str(row.frag_new or "")),
                    edit,
                ]
            )
        )
    x = np.stack(rows)
    if model_kind == "linear":
        base = Ridge(alpha=1.0)
    elif model_kind == "rf":
        base = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=40,
                max_depth=8,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=1,
            )
        )
    else:
        base = MultiOutputRegressor(
            GradientBoostingRegressor(
                random_state=seed, max_depth=3, n_estimators=50, subsample=0.5
            )
        )
    base.fit(x, y_stdized)
    scorer.model = base
    return scorer


class AdditiveMWScorer:
    """Exact additive ΔMW from fragment weights; QED/LogP from MMP lookup mean."""

    def __init__(self, train_pairs: Path, max_rows: int = 200_000):
        self.mmp = MatchedPairDeltaScorer(train_pairs, max_rows=max_rows)
        self._mw_cache: dict[str, float] = {"": 0.0}

    def _frag_mw(self, smiles: str) -> float:
        if smiles not in self._mw_cache:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            self._mw_cache[smiles] = float(Descriptors.MolWt(mol)) if mol is not None else 0.0
        return self._mw_cache[smiles]

    def predict(
        self,
        sources: list[str],
        frag_old: list[str],
        frag_new: list[str],
        edit_types: list[str],
    ) -> np.ndarray:
        base = self.mmp.predict(sources, frag_old, frag_new, edit_types)
        out = base.copy()
        for i, (old, new) in enumerate(zip(frag_old, frag_new)):
            out[i, 2] = self._frag_mw(new or "") - self._frag_mw(old or "")
        return out


class StaticGraphHeuristicScorer:
    """No learned effects: rank by exact fragment ΔMW only (QED/LogP assumed 0)."""

    def __init__(self) -> None:
        self._mw_cache: dict[str, float] = {"": 0.0}

    def _frag_mw(self, smiles: str) -> float:
        if smiles not in self._mw_cache:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            self._mw_cache[smiles] = float(Descriptors.MolWt(mol)) if mol is not None else 0.0
        return self._mw_cache[smiles]

    def predict(
        self,
        sources: list[str],
        frag_old: list[str],
        frag_new: list[str],
        edit_types: list[str],
    ) -> np.ndarray:
        out = np.zeros((len(sources), 3), dtype=np.float32)
        for i, (old, new) in enumerate(zip(frag_old, frag_new)):
            out[i, 2] = self._frag_mw(new or "") - self._frag_mw(old or "")
        return out


class UncertaintyAwareEnsembleScorer:
    """Mean effect over an ensemble; exposes per-edge std for optimistic ranking."""

    def __init__(self, members: list, uncertainty_coef: float = 0.25):
        if not members:
            raise ValueError("ensemble requires members")
        self.members = members
        self.uncertainty_coef = float(uncertainty_coef)

    def predict(
        self,
        sources: list[str],
        frag_old: list[str],
        frag_new: list[str],
        edit_types: list[str],
    ) -> np.ndarray:
        preds = np.stack(
            [m.predict(sources, frag_old, frag_new, edit_types) for m in self.members],
            axis=0,
        )
        return preds.mean(axis=0).astype(np.float32)

    def predict_uncertainty(
        self,
        sources: list[str],
        frag_old: list[str],
        frag_new: list[str],
        edit_types: list[str],
    ) -> np.ndarray:
        preds = np.stack(
            [m.predict(sources, frag_old, frag_new, edit_types) for m in self.members],
            axis=0,
        )
        # Scalar uncertainty per edge: mean axis-wise std of predicted deltas.
        return preds.std(axis=0).mean(axis=1).astype(np.float32)


class KNNPropertyTransformScorer:
    """kNN over training source molecules, then average matched-pair delta."""

    def __init__(
        self,
        train_pairs: Path,
        k: int = 8,
        fp_bits: int = 2048,
        fp_radius: int = 2,
        max_sources: int = 50_000,
    ):
        self.k = k
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        df = pd.read_parquet(
            train_pairs,
            columns=[
                "smiles_a",
                "edit_type",
                "frag_old",
                "frag_new",
                "delta_qed",
                "delta_logp",
                "delta_mw",
                "qed_a",
                "logp_a",
                "mw_a",
            ],
        ).head(max_sources * 4)
        sources = df.drop_duplicates("smiles_a").head(max_sources)
        self.source_smiles = sources["smiles_a"].tolist()
        self.source_props = sources[["qed_a", "logp_a", "mw_a"]].to_numpy(dtype=np.float64)
        self.source_fps = np.stack(
            [
                smiles_to_fp(s, n_bits=fp_bits, radius=fp_radius)
                for s in self.source_smiles
            ]
        )
        self.mmp = MatchedPairDeltaScorer(train_pairs, fp_bits=fp_bits, fp_radius=fp_radius)
        self._fp_cache: dict[str, np.ndarray] = {"": np.zeros(fp_bits, dtype=np.float32)}

    def _fp(self, smiles: str) -> np.ndarray:
        if smiles not in self._fp_cache:
            self._fp_cache[smiles] = smiles_to_fp(
                smiles, n_bits=self.fp_bits, radius=self.fp_radius
            )
        return self._fp_cache[smiles]

    def predict(
        self,
        sources: list[str],
        frag_old: list[str],
        frag_new: list[str],
        edit_types: list[str],
    ) -> np.ndarray:
        base = self.mmp.predict(sources, frag_old, frag_new, edit_types)
        out = base.copy()
        for i, src in enumerate(sources):
            fp = self._fp(src)
            sims = self.source_fps @ fp / (
                np.linalg.norm(self.source_fps, axis=1) * np.linalg.norm(fp) + 1e-8
            )
            nn = np.argsort(-sims)[: self.k]
            # shift prediction toward property-space neighbors' typical movement
            neighbor_shift = self.source_props[nn].mean(axis=0) - self.source_props[nn[0]]
            out[i] += 0.05 * neighbor_shift.astype(np.float32)
        return out


class DirectDestPropertyScorer:
    """Predict absolute destination properties from Morgan fingerprint of m'.

    Optional conditioning concatenates the current-state property vector so the
    model can learn g(m' | p(m)). Used with BeamPlanner(ranker='direct').
    """

    def __init__(
        self,
        model,
        fp_bits: int,
        fp_radius: int,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        conditioned: bool = False,
    ):
        self.model = model
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)
        self.conditioned = conditioned
        self._fp_cache: dict[str, np.ndarray] = {"": np.zeros(fp_bits, dtype=np.float32)}

    def _fp(self, smiles: str) -> np.ndarray:
        if smiles not in self._fp_cache:
            self._fp_cache[smiles] = smiles_to_fp(
                smiles, n_bits=self.fp_bits, radius=self.fp_radius
            )
        return self._fp_cache[smiles]

    def predict_properties(
        self,
        dest_smiles: list[str],
        source_props: np.ndarray | None = None,
        **_context,
    ) -> np.ndarray:
        if not dest_smiles:
            return np.empty((0, 3), dtype=np.float32)
        rows = []
        for i, smi in enumerate(dest_smiles):
            fp = self._fp(smi)
            if self.conditioned:
                if source_props is None:
                    raise ValueError("conditioned direct scorer needs source_props")
                rows.append(np.concatenate([fp, np.asarray(source_props, dtype=np.float32)]))
            else:
                rows.append(fp)
        x = np.stack(rows)
        pred = np.asarray(self.model.predict(x), dtype=np.float32)
        return pred * self.y_std + self.y_mean

    def predict(
        self,
        sources: list[str],
        frag_old: list[str],
        frag_new: list[str],
        edit_types: list[str],
    ) -> np.ndarray:
        # Not used by ranker='direct'; provided for duck-typing completeness.
        raise RuntimeError("Use predict_properties with ranker='direct'")


def train_direct_dest_scorer(
    train_pairs: Path,
    model_kind: str = "linear",
    fp_bits: int = 2048,
    fp_radius: int = 2,
    max_rows: int = 50_000,
    seed: int = 42,
    conditioned: bool = False,
) -> DirectDestPropertyScorer:
    """Train ridge/GBT to map destination Morgan FP (+ optional p(m)) → p(m')."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor

    df = pd.read_parquet(
        train_pairs,
        columns=[
            "smiles_a",
            "smiles_b",
            "qed_a",
            "logp_a",
            "mw_a",
            "qed_b",
            "logp_b",
            "mw_b",
        ],
    )
    # Unique destinations keep training closer to a node property model.
    df = df.drop_duplicates("smiles_b").head(max_rows)
    y = df[["qed_b", "logp_b", "mw_b"]].to_numpy(dtype=np.float32)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0
    y_z = (y - y_mean) / y_std

    scorer = DirectDestPropertyScorer.__new__(DirectDestPropertyScorer)
    scorer.fp_bits = fp_bits
    scorer.fp_radius = fp_radius
    scorer.y_mean = y_mean.astype(np.float32)
    scorer.y_std = y_std.astype(np.float32)
    scorer.conditioned = conditioned
    scorer._fp_cache = {"": np.zeros(fp_bits, dtype=np.float32)}

    rows = []
    for row in df.itertuples(index=False):
        fp = smiles_to_fp(str(row.smiles_b), n_bits=fp_bits, radius=fp_radius)
        if conditioned:
            src_p = np.asarray([row.qed_a, row.logp_a, row.mw_a], dtype=np.float32)
            rows.append(np.concatenate([fp, src_p]))
        else:
            rows.append(fp)
    x = np.stack(rows)

    if model_kind == "linear":
        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(x, y_z)
    elif model_kind == "tree":
        model = MultiOutputRegressor(
            GradientBoostingRegressor(
                random_state=seed, max_depth=3, n_estimators=50, learning_rate=0.1
            )
        )
        model.fit(x, y_z)
    else:
        raise ValueError(model_kind)
    scorer.model = model
    return scorer
