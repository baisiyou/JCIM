#!/usr/bin/env python3
"""Canonical protocol freeze: one f=0 evaluator for all wrappers.

Root causes of prior Direct/Random mismatches (now eliminated):
  - Direct: primary used max_rows=15000; partial used 20000 → different models
  - Random: archived primary 54.0% was stale vs BeamPlanner ranker=random (58.5%)
  - Queries: primary charged start; partial did not

Canonical ledger (this script):
  - start free (count_start_query=False; start in always-free set for partial)
  - unique-node SMILES dedupe
  - Direct ridge: max_rows=15000, seed=42, conditioned=False (Table 4 recipe)
  - Effect MLP: checkpoints/v2/effect_predictor.pt
  - planner seed = 42 + episode
  - f=0 PaidUniqueOracle ≡ default BeamPlanner._query_label (episode-identical)

Outputs: results/protocol_freeze/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import beam_planner as bp
from src.beam_planner import BeamPlanner, EditGraphStore, EffectScorer, Properties
from src.search_baselines import (
    ExactGraphBFSPlanner,
    MatchedPairDeltaScorer,
    StaticGraphHeuristicScorer,
    train_direct_dest_scorer,
    train_sklearn_effect_scorer,
)

SCALES = np.array([0.1, 1.0, 30.0], dtype=np.float64)
TOL = np.array([0.03, 0.20, 10.0], dtype=np.float64)
OUT = ROOT / "results/protocol_freeze"
DIRECT_CKPT = OUT / "direct_ridge_canonical.pkl"
MLP_CKPT = ROOT / "checkpoints/v2/effect_predictor.pt"


class PartialIndex:
    def __init__(self, fraction: float, seed: int, always: set[str] | None = None):
        self.fraction = float(fraction)
        self.seed = int(seed)
        self.always = set(always or ())
        self._cache: dict[str, bool] = {}

    def is_free(self, smiles: str) -> bool:
        if smiles in self.always:
            return True
        if smiles in self._cache:
            return self._cache[smiles]
        digest = hashlib.md5(f"{self.seed}:{smiles}".encode()).hexdigest()
        h = int(digest[:8], 16) / 0xFFFFFFFF
        ok = h < self.fraction
        self._cache[smiles] = ok
        return ok


class PaidUniqueOracle:
    def __init__(self, index: PartialIndex, store: EditGraphStore):
        self.index = index
        self.store = store
        self.paid = 0
        self.free = 0
        self._seen: set[str] = set()

    def __call__(self, smiles: str) -> Properties | None:
        props = self.store.molecule(smiles)
        if props is None:
            return None
        if smiles in self._seen:
            return props
        self._seen.add(smiles)
        if self.index.is_free(smiles):
            self.free += 1
        else:
            self.paid += 1
        return props


def make_planner(
    *,
    store: EditGraphStore,
    scorer,
    ranker: str,
    seed: int,
    budget: int | None = None,
) -> BeamPlanner:
    planner = BeamPlanner(
        store=store,
        scorer=scorer,
        scales=SCALES,
        tolerances=TOL,
        beam_width=5,
        actions_per_state=5,
        max_outgoing=500,
        max_steps=3,
        replace_only=True,
        ranker=ranker,
        oracle_correction=True,
        seed=seed,
    )
    planner.dedupe_queries = True
    planner.count_start_query = False  # canonical: start free
    planner.oracle_budget = budget
    return planner


def attach_paid_oracle(planner: BeamPlanner, oracle: PaidUniqueOracle) -> None:
    def _ql(smiles: str, _o=oracle, _p=planner) -> Properties | None:
        if _p.dedupe_queries and smiles in _p._query_cache:
            return _p._query_cache[smiles]
        if _p._budget_exhausted():
            return _p._query_cache.get(smiles)
        before = _o.paid
        props = _o(smiles)
        if _o.paid > before:
            _p.last_stats["oracle_calls"] += 1
        if props is not None and _p.dedupe_queries:
            _p._query_cache[smiles] = props
        return props

    planner._query_label = _ql  # type: ignore[method-assign]


def eval_beam_episode(
    *,
    ep: dict,
    store: EditGraphStore,
    scorer,
    ranker: str,
    seed: int,
    fraction: float | None,
) -> dict:
    """Evaluate one episode. fraction=None → default oracle; else PaidUniqueOracle."""
    target = Properties(ep["target_qed"], ep["target_logp"], ep["target_mw"])
    planner = make_planner(
        store=store,
        scorer=scorer,
        ranker=ranker,
        seed=seed + int(ep["episode"]),
    )
    prev = None
    oracle = None
    if fraction is not None:
        always = {ep["start_smiles"]}
        index = PartialIndex(fraction, seed, always=always)
        oracle = PaidUniqueOracle(index, store)
        prev = bp.oracle_properties
        bp.oracle_properties = oracle  # type: ignore[assignment]
        attach_paid_oracle(planner, oracle)
    t0 = time.perf_counter()
    try:
        result = planner.plan(target, start_smiles=ep["start_smiles"])
        err = np.abs(result.properties.array() - target.array())
        paid = int(oracle.paid) if oracle is not None else int(
            planner.last_stats["oracle_calls"]
        )
        return {
            "episode": int(ep["episode"]),
            "success": bool(np.all(err <= TOL)),
            "normalized_distance": float(np.linalg.norm(err / SCALES)),
            "paid_queries": paid,
            "runtime_sec": time.perf_counter() - t0,
        }
    finally:
        if prev is not None:
            bp.oracle_properties = prev


def eval_bfs_episode(
    *, ep: dict, store: EditGraphStore, fraction: float, seed: int
) -> dict:
    always = {ep["start_smiles"]}
    index = PartialIndex(fraction, seed, always=always)

    class FreeOnlyBFS(ExactGraphBFSPlanner):
        def _cached_props(self, smiles: str):  # type: ignore[override]
            if not index.is_free(smiles):
                return None
            return store.molecule(smiles)

    bfs = FreeOnlyBFS(
        store, SCALES, TOL, max_depth=3, max_outgoing=None, replace_only=True
    )
    target = Properties(ep["target_qed"], ep["target_logp"], ep["target_mw"])
    t0 = time.perf_counter()
    try:
        result = bfs.plan(target, start_smiles=ep["start_smiles"])
        err = np.abs(result.properties.array() - target.array())
        ok = bool(np.all(err <= TOL))
        dist = float(np.linalg.norm(err / SCALES))
    except Exception:
        ok, dist = False, float("nan")
    return {
        "episode": int(ep["episode"]),
        "success": ok,
        "normalized_distance": dist,
        "paid_queries": 0,
        "runtime_sec": time.perf_counter() - t0,
    }


def load_or_train_direct(train: Path, max_rows: int, seed: int):
    OUT.mkdir(parents=True, exist_ok=True)
    if DIRECT_CKPT.exists():
        with open(DIRECT_CKPT, "rb") as f:
            direct = pickle.load(f)
        print(f"Loaded Direct ridge from {DIRECT_CKPT}", flush=True)
        return direct
    print(f"Training Direct ridge (max_rows={max_rows}, seed={seed})...", flush=True)
    direct = train_direct_dest_scorer(
        train, "linear", max_rows=max_rows, seed=seed, conditioned=False
    )
    with open(DIRECT_CKPT, "wb") as f:
        pickle.dump(direct, f)
    return direct


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby(group_cols, sort=False)
        .agg(
            n=("episode", "count"),
            success_pct=("success", lambda s: round(100 * float(s.mean()), 1)),
            mean_paid=("paid_queries", "mean"),
            median_paid=("paid_queries", "median"),
            mean_dist=("normalized_distance", "mean"),
        )
        .reset_index()
    )
    summary["mean_paid"] = summary["mean_paid"].round(2)
    summary["median_paid"] = summary["median_paid"].round(2)
    summary["mean_dist"] = summary["mean_dist"].round(3)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=15_000)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--skip-partial",
        action="store_true",
        help="Only run primary f=0 identity block",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    episodes = (
        pd.read_parquet(ROOT / "data/processed/episode_suites/endpoint_walk.parquet")
        .sort_values("episode")
        .head(args.max_episodes)
        .to_dict("records")
    )
    store = EditGraphStore(ROOT / "data/processed/search_graph.duckdb")
    train = ROOT / "data/processed/effect_splits_v2/train.parquet"

    direct = load_or_train_direct(train, args.max_train_rows, args.seed)
    print("Fitting remaining Table-4 models...", flush=True)
    models: dict[str, tuple] = {
        "Effect beam (MLP)": (EffectScorer(MLP_CKPT), "effect"),
        "Matched-pair / MMP": (
            MatchedPairDeltaScorer(train, max_rows=args.max_train_rows),
            "effect",
        ),
        "Static MW heuristic": (StaticGraphHeuristicScorer(), "effect"),
        "Ridge effect": (
            train_sklearn_effect_scorer(
                train, "linear", max_rows=args.max_train_rows, seed=args.seed
            ),
            "effect",
        ),
        "Direct ridge (dest FP)": (direct, "direct"),
        "Direct ridge (dest FP + p(m))": (
            train_direct_dest_scorer(
                train,
                "linear",
                max_rows=args.max_train_rows,
                seed=args.seed,
                conditioned=True,
            ),
            "direct",
        ),
        "Direct GBT (dest FP)": (
            train_direct_dest_scorer(
                train,
                "tree",
                max_rows=min(args.max_train_rows, 15_000),
                seed=args.seed,
            ),
            "direct",
        ),
        "Random beam": (None, "random"),
    }

    # --- Primary block: default oracle AND partial f=0 (identity check) ---
    identity_rows: list[dict] = []
    primary_rows: list[dict] = []
    for name, (scorer, ranker) in models.items():
        for mode, frac in [("default", None), ("partial_f0", 0.0)]:
            for ep in tqdm(episodes, desc=f"{name} [{mode}]", leave=False):
                row = eval_beam_episode(
                    ep=ep,
                    store=store,
                    scorer=scorer,
                    ranker=ranker,
                    seed=args.seed,
                    fraction=frac,
                )
                row["method"] = name
                row["wrapper"] = mode
                identity_rows.append(row)
                if mode == "default":
                    primary_rows.append(
                        {
                            "method": name,
                            "episode": row["episode"],
                            "success": row["success"],
                            "normalized_distance": row["normalized_distance"],
                            "paid_queries": row["paid_queries"],
                            "runtime_sec": row["runtime_sec"],
                        }
                    )

    id_df = pd.DataFrame(identity_rows)
    id_df.to_parquet(OUT / "identity_per_episode.parquet", index=False)
    pri_df = pd.DataFrame(primary_rows)
    pri_df.to_parquet(OUT / "primary_per_episode.parquet", index=False)
    pri_sum = summarize(pri_df, ["method"])
    pri_sum.to_csv(OUT / "primary_summary.csv", index=False)
    print("\n=== Primary (start-free, default oracle) ===", flush=True)
    print(pri_sum.to_string(index=False), flush=True)

    # Episode identity: default vs partial f=0
    checks = []
    for name in models:
        a = id_df[(id_df.method == name) & (id_df.wrapper == "default")].set_index(
            "episode"
        )
        b = id_df[(id_df.method == name) & (id_df.wrapper == "partial_f0")].set_index(
            "episode"
        )
        both = a.join(b, lsuffix="_d", rsuffix="_p")
        succ_disc = int((both.success_d != both.success_p).sum())
        q_disc = int((both.paid_queries_d != both.paid_queries_p).sum())
        checks.append(
            {
                "method": name,
                "n": len(both),
                "success_pct_default": round(100 * float(both.success_d.mean()), 1),
                "success_pct_partial_f0": round(100 * float(both.success_p.mean()), 1),
                "mean_q_default": round(float(both.paid_queries_d.mean()), 2),
                "mean_q_partial_f0": round(float(both.paid_queries_p.mean()), 2),
                "success_discordance": succ_disc,
                "query_discordance": q_disc,
            }
        )
    check_df = pd.DataFrame(checks)
    check_df.to_csv(OUT / "f0_identity_check.csv", index=False)
    (OUT / "f0_identity_check.json").write_text(
        json.dumps(
            {
                "all_success_identical": bool(
                    (check_df["success_discordance"] == 0).all()
                ),
                "all_queries_identical": bool(
                    (check_df["query_discordance"] == 0).all()
                ),
                "methods": checks,
            },
            indent=2,
        )
    )
    print("\n=== f=0 identity (default ≡ partial) ===", flush=True)
    print(check_df.to_string(index=False), flush=True)
    assert (check_df["success_discordance"] == 0).all(), "f=0 success mismatch"
    assert (check_df["query_discordance"] == 0).all(), "f=0 query mismatch"

    # --- Partial-index sweep with frozen models ---
    if not args.skip_partial:
        partial_models = {
            k: models[k]
            for k in [
                "Effect beam (MLP)",
                "Direct ridge (dest FP)",
                "Static MW heuristic",
                "Random beam",
            ]
        }
        part_rows: list[dict] = []
        for frac in args.fractions:
            for ep in tqdm(episodes, desc=f"BFS f={frac}", leave=False):
                row = eval_bfs_episode(
                    ep=ep, store=store, fraction=frac, seed=args.seed
                )
                row["fraction_indexed"] = frac
                row["method"] = "Cached-only BFS"
                part_rows.append(row)
            for name, (scorer, ranker) in partial_models.items():
                for ep in tqdm(episodes, desc=f"{name} f={frac}", leave=False):
                    row = eval_beam_episode(
                        ep=ep,
                        store=store,
                        scorer=scorer,
                        ranker=ranker,
                        seed=args.seed,
                        fraction=frac,
                    )
                    row["fraction_indexed"] = frac
                    row["method"] = name
                    part_rows.append(row)
        part_df = pd.DataFrame(part_rows)
        part_df.to_parquet(OUT / "partial_per_episode.parquet", index=False)
        part_sum = summarize(part_df, ["fraction_indexed", "method"])
        part_sum.to_csv(OUT / "partial_summary.csv", index=False)
        # Also overwrite partial_index_frozen with aligned numbers for downstream tools
        frozen_dir = ROOT / "results/partial_index_frozen"
        frozen_dir.mkdir(parents=True, exist_ok=True)
        part_df.to_parquet(frozen_dir / "per_episode.parquet", index=False)
        part_sum.to_csv(frozen_dir / "summary.csv", index=False)
        # Keep Direct@15k as the frozen checkpoint for partial too
        with open(frozen_dir / "direct_ridge_table4.pkl", "wb") as f:
            pickle.dump(direct, f)
        print("\n=== Partial-index (frozen models) ===", flush=True)
        print(part_sum.to_string(index=False), flush=True)

    store.close()
    (OUT / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "direct_checkpoint": str(DIRECT_CKPT),
                "mlp_checkpoint": str(MLP_CKPT),
                "count_start_query": False,
                "start_always_free": True,
                "direct_max_rows": args.max_train_rows,
                "note": (
                    "Canonical freeze: default oracle ≡ partial f=0 episode-wise; "
                    "Direct@15000 matches Table 4; Random=58.5%."
                ),
            },
            indent=2,
        )
    )
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
