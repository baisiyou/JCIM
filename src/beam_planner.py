"""Effect-guided beam search over the observed molecular edit graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED

from src.effect_model import EffectPredictor
from src.featurize import EDIT_TYPE_TO_ID, smiles_to_fp

PROPERTY_NAMES = ("qed", "logp", "mw")


@dataclass(frozen=True)
class Properties:
    qed: float
    logp: float
    mw: float

    def array(self) -> np.ndarray:
        return np.asarray([self.qed, self.logp, self.mw], dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(PROPERTY_NAMES, self.array())}


@dataclass
class EditStep:
    source: str
    target: str
    edit_type: str
    frag_old: str
    frag_new: str
    predicted_delta: np.ndarray
    oracle_delta: np.ndarray
    oracle_properties: Properties

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "edit_type": self.edit_type,
            "frag_old": self.frag_old,
            "frag_new": self.frag_new,
            "predicted_delta": dict(zip(PROPERTY_NAMES, self.predicted_delta.tolist())),
            "oracle_delta": dict(zip(PROPERTY_NAMES, self.oracle_delta.tolist())),
            "oracle_properties": self.oracle_properties.as_dict(),
        }


@dataclass
class SearchState:
    smiles: str
    properties: Properties
    score: float
    path: list[EditStep] = field(default_factory=list)
    visited: frozenset[str] = field(default_factory=frozenset)


def oracle_properties(smiles: str) -> Properties | None:
    """Recompute exact RDKit properties; return None for invalid molecules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Properties(
            qed=float(QED.qed(mol)),
            logp=float(Crippen.MolLogP(mol)),
            mw=float(Descriptors.MolWt(mol)),
        )
    except Exception:
        return None


class EffectScorer:
    """Checkpoint-backed batched edit-effect inference."""

    def __init__(self, checkpoint: Path, device: str = "auto"):
        if device == "auto":
            resolved = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            resolved = device
        self.device = torch.device(resolved)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg = ckpt["cfg"]
        self.fp_bits = int(cfg["fp_bits"])
        self.fp_radius = int(cfg["fp_radius"])
        self.y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
        self.y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
        self.model = EffectPredictor(
            fp_bits=self.fp_bits,
            hidden=int(cfg["hidden"]),
            dropout=float(cfg["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self._fp_cache: dict[str, np.ndarray] = {
            "": np.zeros(self.fp_bits, dtype=np.float32)
        }

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
        if not sources:
            return np.empty((0, 3), dtype=np.float32)
        mol = torch.from_numpy(np.stack([self._fp(s) for s in sources]))
        old = torch.from_numpy(np.stack([self._fp(s) for s in frag_old]))
        new = torch.from_numpy(np.stack([self._fp(s) for s in frag_new]))
        edit = np.zeros((len(sources), 3), dtype=np.float32)
        for i, kind in enumerate(edit_types):
            edit[i, EDIT_TYPE_TO_ID[kind]] = 1.0
        with torch.no_grad():
            pred_std = self.model(
                mol.to(self.device),
                old.to(self.device),
                new.to(self.device),
                torch.from_numpy(edit).to(self.device),
            )
        return pred_std.cpu().numpy() * self.y_std + self.y_mean


class EditGraphStore:
    """Read-only indexed access to molecules and outgoing edit edges."""

    def __init__(self, database: Path):
        self.con = duckdb.connect(str(database), read_only=True)

    def close(self) -> None:
        self.con.close()

    def retrieve(
        self,
        target: Properties,
        scales: np.ndarray,
        limit: int,
    ) -> list[tuple[str, Properties, float]]:
        rows = self.con.execute(
            """
            SELECT m.smiles, m.qed, m.logp, m.mw,
                   sqrt(
                     pow((m.qed - ?) / ?, 2) +
                     pow((m.logp - ?) / ?, 2) +
                     pow((m.mw - ?) / ?, 2)
                   ) AS distance
            FROM molecules m
            INNER JOIN sources s ON s.smiles = m.smiles
            ORDER BY distance
            LIMIT ?
            """,
            [
                target.qed,
                float(scales[0]),
                target.logp,
                float(scales[1]),
                target.mw,
                float(scales[2]),
                limit,
            ],
        ).fetchall()
        return [
            (row[0], Properties(float(row[1]), float(row[2]), float(row[3])), float(row[4]))
            for row in rows
        ]

    def molecule(self, smiles: str) -> Properties | None:
        row = self.con.execute(
            "SELECT qed, logp, mw FROM molecules WHERE smiles = ?", [smiles]
        ).fetchone()
        if row is None:
            return None
        return Properties(*map(float, row))

    def random_source(self, rng: np.random.Generator) -> tuple[str, Properties]:
        """Sample a random searchable start molecule from the indexed source set."""
        total = self.con.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        if not total:
            raise RuntimeError("No sources available for random start")
        offset = int(rng.integers(total))
        smiles = self.con.execute(
            "SELECT smiles FROM sources LIMIT 1 OFFSET ?", [offset]
        ).fetchone()[0]
        props = self.molecule(smiles)
        if props is None:
            raise RuntimeError(f"Missing molecule row for source: {smiles}")
        return smiles, props

    def outgoing(
        self,
        smiles: str,
        replace_only: bool,
        limit: int | None,
    ) -> list[dict]:
        """Return outgoing edges ordered by destination SMILES (deterministic).

        ``limit=None`` returns the full adjacency list (used by exact BFS/A*).
        Finite limits are for budgeted beam/GA rankers only.
        """
        edit_clause = "AND edit_type = 'replace'" if replace_only else ""
        if limit is None or limit <= 0:
            rows = self.con.execute(
                f"""
                SELECT smiles_b, edit_type, frag_old, frag_new,
                       delta_qed, delta_logp, delta_mw
                FROM edges
                WHERE smiles_a = ? {edit_clause}
                ORDER BY smiles_b
                """,
                [smiles],
            ).fetchall()
        else:
            rows = self.con.execute(
                f"""
                SELECT smiles_b, edit_type, frag_old, frag_new,
                       delta_qed, delta_logp, delta_mw
                FROM edges
                WHERE smiles_a = ? {edit_clause}
                ORDER BY smiles_b
                LIMIT ?
                """,
                [smiles, limit],
            ).fetchall()
        return [
            {
                "smiles_b": row[0],
                "edit_type": row[1],
                "frag_old": row[2] or "",
                "frag_new": row[3] or "",
                "stored_delta": np.asarray(row[4:7], dtype=np.float64),
            }
            for row in rows
        ]


class BeamPlanner:
    def __init__(
        self,
        store: EditGraphStore,
        scorer: EffectScorer | None,
        scales: np.ndarray,
        tolerances: np.ndarray,
        beam_width: int = 10,
        actions_per_state: int = 20,
        max_outgoing: int = 2000,
        max_steps: int = 3,
        step_penalty: float = 0.02,
        replace_only: bool = True,
        ranker: str = "effect",
        oracle_correction: bool = True,
        seed: int = 42,
        uncertainty_coef: float = 0.0,
        correction_period: int = 1,
    ):
        if ranker not in {"effect", "oracle", "random", "live_oracle", "direct"}:
            raise ValueError(f"Unknown ranker: {ranker}")
        if ranker == "effect" and scorer is None:
            raise ValueError("Effect ranker requires an EffectScorer")
        if ranker == "direct" and scorer is None:
            raise ValueError("Direct ranker requires a destination property scorer")
        if not oracle_correction and ranker not in {"effect", "direct"}:
            raise ValueError("Predictor-only search requires the effect/direct ranker")
        if correction_period < 1:
            raise ValueError("correction_period must be >= 1")
        self.store = store
        self.scorer = scorer
        self.scales = scales.astype(np.float64)
        self.tolerances = tolerances.astype(np.float64)
        self.beam_width = beam_width
        self.actions_per_state = actions_per_state
        self.max_outgoing = max_outgoing
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.replace_only = replace_only
        self.ranker = ranker
        self.oracle_correction = oracle_correction
        self.correction_period = int(correction_period)
        self.uncertainty_coef = float(uncertainty_coef)
        self.rng = np.random.default_rng(seed)
        self.last_stats: dict[str, int] = {}
        self.oracle_budget: int | None = None
        # Deduplicate label queries by canonical SMILES within one plan() call.
        self._query_cache: dict[str, Properties] = {}
        self.count_start_query: bool = True
        self.dedupe_queries: bool = True

    def _budget_exhausted(self) -> bool:
        return (
            self.oracle_budget is not None
            and self.last_stats["oracle_calls"] >= self.oracle_budget
        )

    def _query_label(self, smiles: str) -> Properties | None:
        """Charge one label query per unique SMILES (unless dedupe disabled)."""
        if self.dedupe_queries and smiles in self._query_cache:
            return self._query_cache[smiles]
        if self._budget_exhausted():
            return self._query_cache.get(smiles)
        props = oracle_properties(smiles)
        self.last_stats["oracle_calls"] += 1
        if props is not None and self.dedupe_queries:
            self._query_cache[smiles] = props
        return props

    def distance(self, props: Properties, target: Properties) -> float:
        return float(np.linalg.norm((props.array() - target.array()) / self.scales))

    def succeeds(self, props: Properties, target: Properties) -> bool:
        return bool(np.all(np.abs(props.array() - target.array()) <= self.tolerances))

    def plan(
        self,
        target: Properties,
        retrieval_k: int = 10,
        start_smiles: str | None = None,
    ) -> SearchState:
        self.last_stats = {
            "expanded_states": 0,
            "scored_edges": 0,
            "oracle_calls": 0,
        }
        self._query_cache = {}
        if start_smiles:
            if self.count_start_query:
                props = self._query_label(start_smiles)
            else:
                props = oracle_properties(start_smiles)
                if props is not None:
                    self._query_cache[start_smiles] = props
            if props is None:
                raise ValueError(f"Invalid --start-smiles: {start_smiles}")
            starts = [(start_smiles, props, self.distance(props, target))]
        else:
            starts = self.store.retrieve(target, self.scales, retrieval_k)
            if not starts:
                raise RuntimeError("No retrievable graph nodes found")

        beam = [
            SearchState(
                smiles=smiles,
                properties=props,
                score=distance,
                visited=frozenset({smiles}),
            )
            for smiles, props, distance in starts
        ]
        best = min(beam, key=lambda state: state.score)
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
                    edge
                    for edge in self.store.outgoing(
                        state.smiles, self.replace_only, self.max_outgoing
                    )
                    if edge["smiles_b"] not in state.visited
                ]
                if not edges:
                    continue

                self.last_stats["scored_edges"] += len(edges)
                if self.ranker == "effect":
                    assert self.scorer is not None
                    # Optional destination-only Δ models (matched Direct-feature ablation).
                    if getattr(self.scorer, "uses_destination", False) and hasattr(
                        self.scorer, "predict_dest"
                    ):
                        predicted = self.scorer.predict_dest(
                            [edge["smiles_b"] for edge in edges]
                        )
                    else:
                        predicted = self.scorer.predict(
                            [state.smiles] * len(edges),
                            [edge["frag_old"] for edge in edges],
                            [edge["frag_new"] for edge in edges],
                            [edge["edit_type"] for edge in edges],
                        )
                    predicted_props = state.properties.array()[None, :] + predicted
                    predicted_distance = np.linalg.norm(
                        (predicted_props - target.array()[None, :])
                        / self.scales[None, :],
                        axis=1,
                    )
                    if self.uncertainty_coef > 0 and hasattr(
                        self.scorer, "predict_uncertainty"
                    ):
                        unc = self.scorer.predict_uncertainty(
                            [state.smiles] * len(edges),
                            [edge["frag_old"] for edge in edges],
                            [edge["frag_new"] for edge in edges],
                            [edge["edit_type"] for edge in edges],
                        )
                        predicted_distance = (
                            predicted_distance - self.uncertainty_coef * unc
                        )
                    selected = np.argsort(predicted_distance)[: self.actions_per_state]
                elif self.ranker == "direct":
                    assert self.scorer is not None
                    dests = [edge["smiles_b"] for edge in edges]
                    if hasattr(self.scorer, "predict_properties"):
                        predicted_props = self.scorer.predict_properties(
                            dests,
                            source_props=state.properties.array(),
                            sources=[state.smiles] * len(edges),
                            frag_old=[edge["frag_old"] for edge in edges],
                            frag_new=[edge["frag_new"] for edge in edges],
                            edit_types=[edge["edit_type"] for edge in edges],
                        )
                    else:
                        # Fallback: treat predict() as absolute properties.
                        predicted_props = self.scorer.predict(
                            dests,
                            [""] * len(dests),
                            [""] * len(dests),
                            ["replace"] * len(dests),
                        )
                    predicted = predicted_props - state.properties.array()[None, :]
                    predicted_distance = np.linalg.norm(
                        (predicted_props - target.array()[None, :])
                        / self.scales[None, :],
                        axis=1,
                    )
                    selected = np.argsort(predicted_distance)[: self.actions_per_state]
                elif self.ranker == "oracle":
                    predicted = np.stack([edge["stored_delta"] for edge in edges])
                    self.last_stats["oracle_calls"] += len(edges)
                    predicted_props = state.properties.array()[None, :] + predicted
                    predicted_distance = np.linalg.norm(
                        (predicted_props - target.array()[None, :])
                        / self.scales[None, :],
                        axis=1,
                    )
                    selected = np.argsort(predicted_distance)[: self.actions_per_state]
                else:
                    predicted = np.stack([edge["stored_delta"] for edge in edges])
                    n_select = min(self.actions_per_state, len(edges))
                    selected = self.rng.choice(
                        len(edges), size=n_select, replace=False
                    )

                for idx in selected:
                    if self._budget_exhausted():
                        break
                    edge = edges[int(idx)]
                    pred_delta = predicted[int(idx)].astype(np.float64)
                    # Periodic correction: oracle at depths where (depth+1) % period == 0.
                    use_oracle = self.oracle_correction and (
                        (depth + 1) % self.correction_period == 0
                    )
                    if use_oracle:
                        next_props = self._query_label(edge["smiles_b"])
                        if next_props is None:
                            continue
                        oracle_delta = next_props.array() - state.properties.array()
                    else:
                        # Predictor-only state update (open-loop or off-period step).
                        if Chem.MolFromSmiles(edge["smiles_b"]) is None:
                            continue
                        pred_arr = state.properties.array() + pred_delta
                        next_props = Properties(
                            float(pred_arr[0]), float(pred_arr[1]), float(pred_arr[2])
                        )
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
                    if self.succeeds(next_props, target):
                        return expanded[-1]

            if not expanded:
                break
            unique: dict[str, SearchState] = {}
            for state in sorted(expanded, key=lambda item: item.score):
                unique.setdefault(state.smiles, state)
            beam = list(unique.values())[: self.beam_width]
            if beam[0].score < best.score:
                best = beam[0]
            successful = [
                state for state in beam if self.succeeds(state.properties, target)
            ]
            if successful:
                return min(successful, key=lambda state: (len(state.path), state.score))

        return best
