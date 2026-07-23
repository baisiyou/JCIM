#!/usr/bin/env python3
"""Fill exact-baseline query counts under the canonical unique-node ledger.

Also dumps MMFF transfer / Scheme~B McNemar / binomial checks so the paper
can contract ``confirmed reversal'' language to what the statistics support.

Canonical ledger (matches scripts/40_protocol_freeze.py):
  - start free
  - one charge per unique canonical SMILES
  - RDKit labels via EditGraphStore.molecule

Outputs: results/exact_baselines_unique_node/, results/mmff_stats/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from math import comb
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import beam_planner as bp
from src import search_baselines as sb
from src.beam_planner import BeamPlanner, EditGraphStore, Properties
from src.search_baselines import (
    AStarPlanner,
    BudgetExactNeighborPlanner,
    BudgetMatchedBeamPlanner,
    ExhaustiveDepth3Planner,
)

SCALES = np.array([0.1, 1.0, 30.0], dtype=np.float64)
TOL = np.array([0.03, 0.20, 10.0], dtype=np.float64)
OUT = ROOT / "results/exact_baselines_unique_node"
MMFF_OUT = ROOT / "results/mmff_stats"

SUITE_FILES = {
    "endpoint_walk": "endpoint_walk.parquet",
    "independent_box": "independent_box.parquet",
    "leave_property_region_out": "leave_property_region_out.parquet",
    "unsatisfiable_box": "unsatisfiable_box.parquet",
}


class UniqueNodeLedger:
    """Charge one paid query per unique SMILES; start always free."""

    def __init__(self, store: EditGraphStore, start_smiles: str):
        self.store = store
        self.start = start_smiles
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
        if smiles == self.start:
            self.free += 1
        else:
            self.paid += 1
        return props


def exact_binomial_two_sided(n10: int, n01: int) -> float:
    n = n10 + n01
    if n == 0:
        return 1.0
    lo, hi = min(n10, n01), max(n10, n01)
    p = 0.0
    for k in range(n + 1):
        if k <= lo or k >= hi:
            p += comb(n, k) * (0.5**n)
    return float(min(1.0, p))


def mcnemar_pair(a: pd.Series, b: pd.Series) -> dict:
    both = pd.concat([a.astype(bool), b.astype(bool)], axis=1, keys=["a", "b"]).dropna()
    n10 = int((both.a & ~both.b).sum())
    n01 = int((~both.a & both.b).sum())
    return {
        "n": int(len(both)),
        "rate_a": round(100 * float(both.a.mean()), 1),
        "rate_b": round(100 * float(both.b.mean()), 1),
        "diff_pp": round(100 * float(both.a.mean() - both.b.mean()), 1),
        "n10": n10,
        "n01": n01,
        "mcnemar_exact_p": round(exact_binomial_two_sided(n10, n01), 4),
    }


def run_method(
    *,
    name: str,
    ep: dict,
    store: EditGraphStore,
    seed: int,
) -> dict:
    target = Properties(ep["target_qed"], ep["target_logp"], ep["target_mw"])
    start = ep["start_smiles"]
    ledger = UniqueNodeLedger(store, start)
    prev_bp = bp.oracle_properties
    prev_sb = sb.oracle_properties
    # search_baselines binds oracle_properties at import time; patch both modules.
    bp.oracle_properties = ledger  # type: ignore[assignment]
    sb.oracle_properties = ledger  # type: ignore[assignment]
    t0 = time.perf_counter()
    try:
        if name == "Exhaustive depth-3":
            planner = ExhaustiveDepth3Planner(
                store, SCALES, TOL, max_depth=3, max_outgoing=500
            )
            result = planner.plan(target, start)
        elif name == "A* best-first":
            planner = AStarPlanner(
                store,
                SCALES,
                TOL,
                max_steps=3,
                max_outgoing=500,
                branch_limit=None,
            )
            result = planner.plan(target, start)
        elif name == "Exact-property beam":
            planner = BudgetMatchedBeamPlanner(
                store=store,
                scorer=None,
                scales=SCALES,
                tolerances=TOL,
                beam_width=5,
                actions_per_state=5,
                max_outgoing=500,
                max_steps=3,
                replace_only=True,
                ranker="live_oracle",
                oracle_correction=True,
                seed=seed + int(ep["episode"]),
                oracle_budget=None,
            )
            result = planner.plan(target, start_smiles=start)
        elif name == "Oracle greedy":
            planner = BudgetMatchedBeamPlanner(
                store=store,
                scorer=None,
                scales=SCALES,
                tolerances=TOL,
                beam_width=1,
                actions_per_state=1,
                max_outgoing=500,
                max_steps=3,
                replace_only=True,
                ranker="live_oracle",
                oracle_correction=True,
                seed=seed + int(ep["episode"]),
                oracle_budget=None,
            )
            result = planner.plan(target, start_smiles=start)
        elif name == "Budget exact-neighbor":
            planner = BudgetExactNeighborPlanner(
                store,
                SCALES,
                TOL,
                max_steps=3,
                max_outgoing=500,
                replace_only=True,
                oracle_budget=200,
            )
            result = planner.plan(target, start)
        else:
            raise ValueError(name)

        err = np.abs(result.properties.array() - target.array())
        return {
            "method": name,
            "episode": int(ep["episode"]),
            "success": bool(np.all(err <= TOL)),
            "normalized_distance": float(np.linalg.norm(err / SCALES)),
            "paid_queries": int(ledger.paid),
            "legacy_planner_calls": int(
                getattr(planner, "last_stats", {}).get("oracle_calls", 0)
            ),
            "runtime_sec": time.perf_counter() - t0,
        }
    finally:
        bp.oracle_properties = prev_bp
        sb.oracle_properties = prev_sb


def dump_mmff_stats() -> dict:
    MMFF_OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(ROOT / "results/nonfree_mmff/per_episode.parquet")
    # normalize method names
    pivot = {
        m: tr[tr.method == m].set_index("episode")["success"]
        for m in tr.method.unique()
    }
    pairs = []
    random = pivot["Random beam"]
    for other in [
        "Effect MLP (QED/LogP/MW-trained)",
        "Direct ridge (dest FP)",
        "Ridge effect",
        "Static MW heuristic",
    ]:
        if other in pivot:
            pairs.append(
                {"contrast": f"Random vs {other}", **mcnemar_pair(random, pivot[other])}
            )

    sb = pd.read_parquet(ROOT / "results/mmff_scheme_b/per_episode.parquet")
    sb40 = sb[sb.budget == 40]
    sb_pairs = []
    r40 = sb40[sb40.method == "Random beam"].set_index("episode")["success"]
    for other in [
        "SuiteA Effect MLP (transfer)",
        "MMFF effect ridge",
        "Uncertainty UCB",
        "MMFF direct ridge",
        "Static MW heuristic (transfer)",
    ]:
        o = sb40[sb40.method == other].set_index("episode")["success"]
        if len(o):
            sb_pairs.append(
                {"contrast": f"Random vs {other} @Q=40", **mcnemar_pair(r40, o)}
            )

    out = {
        "transfer_n50": {
            "rates": {
                m: round(100 * float(tr[tr.method == m].success.mean()), 1)
                for m in tr.method.unique()
            },
            "mcnemar_vs_random": pairs,
            "note": (
                "Random vs Effect MLP is NOT significant (McNemar p=0.58 on n=50); "
                "the Suite~A ordering reversal is therefore not confirmed. "
                "Random vs Direct ridge is significant (p=0.021) but Direct is already "
                "the weaker Suite~A comparator under MMFF transfer."
            ),
        },
        "scheme_b_q40_n30": {
            "rates": {
                m: round(100 * float(sb40[sb40.method == m].success.mean()), 1)
                for m in sb40.method.unique()
            },
            "mcnemar_vs_random": sb_pairs,
            "success_step_pp": round(100 / 30, 2),
            "note": (
                "At n=30 (step 3.33pp), no Random vs surrogate contrast is significant; "
                "report inconclusive / no detectable MMFF-surrogate advantage."
            ),
        },
    }
    (MMFF_OUT / "mmff_statistical_checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["endpoint_walk"],
        choices=list(SUITE_FILES) + ["all"],
    )
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--mmff-only", action="store_true")
    args = parser.parse_args()

    mmff = dump_mmff_stats()
    if args.mmff_only:
        return 0

    if args.skip_exact:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    suites = list(SUITE_FILES) if "all" in args.suites else args.suites
    methods = [
        "Exhaustive depth-3",
        "A* best-first",
        "Exact-property beam",
        "Oracle greedy",
        "Budget exact-neighbor",
    ]
    store = EditGraphStore(ROOT / "data/processed/search_graph.duckdb")
    rows: list[dict] = []
    for suite in suites:
        path = ROOT / "data/processed/episode_suites" / SUITE_FILES[suite]
        eps = (
            pd.read_parquet(path)
            .sort_values("episode")
            .head(args.max_episodes)
            .to_dict("records")
        )
        for name in methods:
            for ep in tqdm(eps, desc=f"{suite}/{name}", leave=False):
                row = run_method(name=name, ep=ep, store=store, seed=args.seed)
                row["suite"] = suite
                rows.append(row)

    store.close()
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "per_episode.parquet", index=False)
    summary = (
        df.groupby(["suite", "method"], sort=False)
        .agg(
            n=("episode", "count"),
            success_pct=("success", lambda s: round(100 * float(s.mean()), 1)),
            mean_paid=("paid_queries", "mean"),
            median_paid=("paid_queries", "median"),
            mean_legacy_calls=("legacy_planner_calls", "mean"),
            mean_dist=("normalized_distance", "mean"),
        )
        .reset_index()
    )
    summary["mean_paid"] = summary["mean_paid"].round(1)
    summary["median_paid"] = summary["median_paid"].round(1)
    summary["mean_legacy_calls"] = summary["mean_legacy_calls"].round(1)
    summary["mean_dist"] = summary["mean_dist"].round(3)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "count_start_query": False,
                "dedupe": "unique_node",
                "mmff_stats": str(MMFF_OUT / "mmff_statistical_checks.json"),
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False), flush=True)

    # Endpoint oracle column for Table exact
    ep = summary[summary.suite == "endpoint_walk"][
        ["method", "success_pct", "mean_paid"]
    ]
    (OUT / "endpoint_oracle_column.json").write_text(
        json.dumps(ep.to_dict(orient="records"), indent=2)
    )
    print("\nEndpoint Oracle column:", flush=True)
    print(ep.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
