#!/usr/bin/env python3
"""Confirmatory v2: disjoint n=1000 + hard-budget structured missingness + multi-seed MLP.

Protocol upgrades vs script 31:
  1. Test/val starts exclude the frozen n=200 headline suite (and each other, and train ends).
  2. Structured missingness under a shared hard *paid* unique-node budget Q;
     availability bit is never passed to the scorer (hide_availability).
  3. Evaluate pre-frozen retrain seeds {42..46} on the new confirmatory test.

Outputs: results/confirm_n1000_v2/
Suites: data/processed/episode_suites_confirm_v2/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import binomtest
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_episodes import make_endpoint_walk_episodes
from src.beam_planner import BeamPlanner, EditGraphStore, EffectScorer, Properties
from src.search_baselines import (
    MatchedPairDeltaScorer,
    StaticGraphHeuristicScorer,
    train_direct_dest_scorer,
    train_sklearn_effect_scorer,
)
import src.beam_planner as bp

SCALES = np.array([0.1, 1.0, 30.0], dtype=np.float64)
TOL = np.array([0.03, 0.20, 10.0], dtype=np.float64)

OUT = ROOT / "results/confirm_n1000_v2"
SUITE_DIR = ROOT / "data/processed/episode_suites_confirm_v2"
DB = ROOT / "data/processed/search_graph.duckdb"
TRAIN = ROOT / "data/processed/effect_splits_v2/train.parquet"
TEST_PAIRS = ROOT / "data/processed/effect_splits_v2/test.parquet"
FROZEN200 = ROOT / "data/processed/episode_suites/endpoint_walk.parquet"
SEED_CKPT = ROOT / "results/multiseed_fixed200/checkpoints"


def mcnemar(a: pd.Series, b: pd.Series) -> dict:
    common = a.index.intersection(b.index)
    a = a.loc[common].astype(bool)
    b = b.loc[common].astype(bool)
    ao = int((a & ~b).sum())
    bo = int((~a & b).sum())
    n = ao + bo
    p = float(binomtest(ao, n, 0.5).pvalue) if n else 1.0
    return {
        "n": int(len(common)),
        "a_rate": float(a.mean()),
        "b_rate": float(b.mean()),
        "gap_pp": float((a.mean() - b.mean()) * 100),
        "a_only": ao,
        "b_only": bo,
        "p_value": p,
    }


def bootstrap_gap(
    a: pd.Series, b: pd.Series, n_boot: int = 5000, seed: int = 42
) -> tuple[float, float]:
    common = a.index.intersection(b.index)
    a = a.loc[common].astype(bool).to_numpy()
    b = b.loc[common].astype(bool).to_numpy()
    rng = np.random.default_rng(seed)
    boots = []
    n = len(a)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append((a[idx].mean() - b[idx].mean()) * 100)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def murcko(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
    except Exception:
        return ""


def build_suites(n_test: int, n_val: int, seed: int, force: bool = False) -> dict:
    SUITE_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(TRAIN, columns=["smiles_a", "smiles_b"])
    excluded = set(train["smiles_a"]).union(train["smiles_b"])
    frozen_starts = set(pd.read_parquet(FROZEN200)["start_smiles"].astype(str))
    excluded |= frozen_starts

    test_path = SUITE_DIR / f"endpoint_walk_n{n_test}_disjoint.parquet"
    val_path = SUITE_DIR / f"endpoint_walk_val_n{n_val}_disjoint.parquet"
    store = EditGraphStore(DB)
    jitter = np.array([0.02, 0.15, 7.5], dtype=np.float64)

    if force or not test_path.exists():
        print(
            f"Building disjoint test n={n_test} (exclude {len(frozen_starts)} frozen starts)...",
            flush=True,
        )
        # Different construction seed than frozen-200 (which used 42) to avoid walk collisions.
        eps = make_endpoint_walk_episodes(
            store=store,
            test_pairs=TEST_PAIRS,
            n_episodes=n_test,
            walk_depth=3,
            max_outgoing=500,
            min_distance=1.0,
            scales=SCALES,
            seed=seed + 100,  # 142 if seed=42
            excluded_starts=excluded,
            suite_name="endpoint_walk_confirm_v2",
            target_jitter=jitter,
        )
        pd.DataFrame(eps).to_parquet(test_path, index=False)
    else:
        print(f"Reuse {test_path}", flush=True)

    test_starts = set(pd.read_parquet(test_path)["start_smiles"].astype(str))
    if force or not val_path.exists():
        print(f"Building disjoint val n={n_val}...", flush=True)
        eps = make_endpoint_walk_episodes(
            store=store,
            test_pairs=TEST_PAIRS,
            n_episodes=n_val,
            walk_depth=3,
            max_outgoing=500,
            min_distance=1.0,
            scales=SCALES,
            seed=seed + 101,
            excluded_starts=excluded | test_starts,
            suite_name="endpoint_walk_val_v2",
            target_jitter=jitter,
        )
        pd.DataFrame(eps).to_parquet(val_path, index=False)
    else:
        print(f"Reuse {val_path}", flush=True)

    store.close()
    val_starts = set(pd.read_parquet(val_path)["start_smiles"].astype(str))
    test_starts = set(pd.read_parquet(test_path)["start_smiles"].astype(str))
    isolation = {
        "n_frozen_excluded": len(frozen_starts),
        "test_n": len(test_starts),
        "val_n": len(val_starts),
        "test_cap_frozen": len(test_starts & frozen_starts),
        "val_cap_frozen": len(val_starts & frozen_starts),
        "test_cap_val": len(test_starts & val_starts),
        "test_path": str(test_path),
        "val_path": str(val_path),
        "construction_seed_test": seed + 100,
        "construction_seed_val": seed + 101,
    }
    assert isolation["test_cap_frozen"] == 0, isolation
    assert isolation["val_cap_frozen"] == 0, isolation
    assert isolation["test_cap_val"] == 0, isolation
    (SUITE_DIR / "manifest.json").write_text(json.dumps(isolation, indent=2))
    print(json.dumps(isolation, indent=2), flush=True)
    return isolation


def eval_beam(
    *,
    name: str,
    episodes: list[dict],
    store: EditGraphStore,
    scorer,
    ranker: str,
    seed: int,
    budget: int | None = None,
) -> pd.DataFrame:
    rows = []
    for ep in tqdm(episodes, desc=name, leave=False):
        target = Properties(ep["target_qed"], ep["target_logp"], ep["target_mw"])
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
            seed=seed + int(ep["episode"]),
        )
        planner.dedupe_queries = True
        planner.count_start_query = True
        if budget is not None:
            planner.oracle_budget = int(budget)
        t0 = time.perf_counter()
        result = planner.plan(target, start_smiles=ep["start_smiles"])
        err = np.abs(result.properties.array() - target.array())
        rows.append(
            {
                "method": name,
                "episode": int(ep["episode"]),
                "success": bool(np.all(err <= TOL)),
                "normalized_distance": float(np.linalg.norm(err / SCALES)),
                "oracle_calls": int(planner.last_stats.get("oracle_calls", 0)),
                "runtime_sec": time.perf_counter() - t0,
                "budget": -1 if budget is None else int(budget),
            }
        )
    return pd.DataFrame(rows)


# ---- Missingness indices ----------------------------------------------------


class MCARIndex:
    def __init__(self, fraction_free: float, seed: int, always: set[str] | None = None):
        self.fraction_free = float(fraction_free)
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
        ok = h < self.fraction_free
        self._cache[smiles] = ok
        return ok


class ScaffoldBlockIndex:
    def __init__(self, blocked: set[str], always: set[str] | None = None):
        self.blocked = blocked
        self.always = set(always or ())
        self._cache: dict[str, bool] = {}

    def is_free(self, smiles: str) -> bool:
        if smiles in self.always:
            return True
        if smiles in self._cache:
            return self._cache[smiles]
        scaf = murcko(smiles)
        ok = True if scaf == "" else scaf not in self.blocked
        self._cache[smiles] = ok
        return ok


class PropertyRegionIndex:
    """Paid inside central property box; free outside.

    Membership is precomputed into a SMILES set so is_free never returns
    property values to the planner—only a free/paid bit. Under hide_availability
    the scorer never sees that bit; residual leak via paid-budget accounting is
    reported explicitly in the manuscript.
    """

    def __init__(self, paid_smiles: set[str], always: set[str] | None = None):
        self.paid = paid_smiles
        self.always = set(always or ())

    def is_free(self, smiles: str) -> bool:
        if smiles in self.always:
            return True
        return smiles not in self.paid


class AllPaid:
    def is_free(self, smiles: str) -> bool:
        return False


def eval_hard_budget_hidden(
    *,
    name: str,
    episodes: list[dict],
    store: EditGraphStore,
    scorer,
    ranker: str,
    index,
    seed: int,
    paid_budget: int,
    charge_all: bool = False,
) -> pd.DataFrame:
    """Hard paid (or charge-all) budget; scorer never receives availability bit."""
    rows = []
    for ep in tqdm(episodes, desc=name, leave=False):
        always_start = {ep["start_smiles"]}
        # Re-wrap index with start always free for construction continuity
        if hasattr(index, "always"):
            index.always = always_start  # type: ignore[attr-defined]

        paid = 0
        free = 0
        seen: set[str] = set()

        def oracle(smiles: str):
            nonlocal paid, free
            props = store.molecule(smiles)
            if props is None:
                return None
            if smiles in seen:
                return props
            seen.add(smiles)
            is_free = bool(index.is_free(smiles))
            if charge_all or not is_free:
                paid += 1
            else:
                free += 1
            return props

        prev = bp.oracle_properties
        bp.oracle_properties = oracle  # type: ignore[assignment]
        try:
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
                seed=seed + int(ep["episode"]),
            )
            planner.dedupe_queries = True
            planner.count_start_query = True
            planner.oracle_budget = int(paid_budget)

            def _ql(smiles: str, _p=planner):
                if _p.dedupe_queries and smiles in _p._query_cache:
                    return _p._query_cache[smiles]
                if _p._budget_exhausted():
                    return _p._query_cache.get(smiles)
                before = paid
                props = oracle(smiles)
                # Budget tracks charged revelations only (paid counter).
                if paid > before:
                    _p.last_stats["oracle_calls"] += 1
                # Availability bit intentionally NOT attached to returned Properties.
                if props is not None and _p.dedupe_queries:
                    _p._query_cache[smiles] = props
                return props

            planner._query_label = _ql  # type: ignore[method-assign]
            target = Properties(ep["target_qed"], ep["target_logp"], ep["target_mw"])
            t0 = time.perf_counter()
            result = planner.plan(target, start_smiles=ep["start_smiles"])
            err = np.abs(result.properties.array() - target.array())
            rows.append(
                {
                    "method": name,
                    "episode": int(ep["episode"]),
                    "success": bool(np.all(err <= TOL)),
                    "normalized_distance": float(np.linalg.norm(err / SCALES)),
                    "paid_queries": int(paid),
                    "free_hits": int(free),
                    "budget_Q": int(paid_budget),
                    "charge_all": bool(charge_all),
                    "runtime_sec": time.perf_counter() - t0,
                }
            )
        finally:
            bp.oracle_properties = prev
    return pd.DataFrame(rows)


def fit_models(max_train_rows: int, seed: int, mlp_ckpt: Path):
    mlp = EffectScorer(mlp_ckpt)
    direct = train_direct_dest_scorer(
        TRAIN, "linear", max_rows=max_train_rows, seed=seed, conditioned=False
    )
    ridge = train_sklearn_effect_scorer(
        TRAIN, "linear", max_rows=max_train_rows, seed=seed
    )
    static = StaticGraphHeuristicScorer()
    mmp = MatchedPairDeltaScorer(TRAIN, max_rows=max_train_rows)
    return {
        "Effect beam (MLP)": (mlp, "effect"),
        "Direct ridge (dest FP)": (direct, "direct"),
        "Ridge effect": (ridge, "effect"),
        "Static MW heuristic": (static, "effect"),
        "Matched-pair / MMP": (mmp, "effect"),
        "Random beam": (None, "random"),
    }


def select_comparator(val_df: pd.DataFrame) -> str:
    rates = (
        val_df[val_df.method != "Effect beam (MLP)"]
        .groupby("method")["success"]
        .mean()
        .sort_values(ascending=False)
    )
    return str(rates.index[0])


def run_confirm(models: dict, store: EditGraphStore, seed: int) -> dict:
    val_eps = (
        pd.read_parquet(SUITE_DIR / "endpoint_walk_val_n200_disjoint.parquet")
        .sort_values("episode")
        .to_dict("records")
    )
    test_eps = (
        pd.read_parquet(SUITE_DIR / "endpoint_walk_n1000_disjoint.parquet")
        .sort_values("episode")
        .to_dict("records")
    )
    print(f"=== VAL n={len(val_eps)} (comparator freeze) ===", flush=True)
    val_frames = []
    for name, (scorer, ranker) in models.items():
        val_frames.append(
            eval_beam(
                name=name,
                episodes=val_eps,
                store=store,
                scorer=scorer,
                ranker=ranker,
                seed=seed,
            )
        )
    val_df = pd.concat(val_frames, ignore_index=True)
    val_df.to_parquet(OUT / "val_per_episode.parquet", index=False)
    comparator = select_comparator(val_df)
    val_rates = (
        val_df.groupby("method")["success"].mean().mul(100).round(2).to_dict()
    )
    (OUT / "frozen_comparator.json").write_text(
        json.dumps({"comparator": comparator, "val_rates": val_rates}, indent=2)
    )
    print("Frozen comparator:", comparator, val_rates, flush=True)

    print(f"=== TEST n={len(test_eps)} (disjoint confirmatory) ===", flush=True)
    # Primary pair + a few references
    focus = [
        "Effect beam (MLP)",
        comparator,
        "Static MW heuristic",
        "Random beam",
    ]
    # dedupe while preserving order
    seen = set()
    focus = [m for m in focus if not (m in seen or seen.add(m))]
    test_frames = []
    for name in focus:
        scorer, ranker = models[name]
        test_frames.append(
            eval_beam(
                name=name,
                episodes=test_eps,
                store=store,
                scorer=scorer,
                ranker=ranker,
                seed=seed,
            )
        )
    test_df = pd.concat(test_frames, ignore_index=True)
    test_df.to_parquet(OUT / "test_per_episode.parquet", index=False)

    piv = {
        m: g.set_index("episode")["success"]
        for m, g in test_df.groupby("method")
    }
    mlp = piv["Effect beam (MLP)"]
    comp = piv[comparator]
    stats = mcnemar(mlp, comp)
    lo, hi = bootstrap_gap(mlp, comp)
    summary = {
        "comparator": comparator,
        "mlp_pct": round(100 * float(mlp.mean()), 1),
        "comparator_pct": round(100 * float(comp.mean()), 1),
        "gap_pp": round(stats["gap_pp"], 1),
        "mcnemar": stats,
        "bootstrap_ci95_pp": [round(lo, 1), round(hi, 1)],
        "mean_queries": {
            m: round(float(g.oracle_calls.mean()), 2)
            for m, g in test_df.groupby("method")
        },
        "isolation": json.loads((SUITE_DIR / "manifest.json").read_text()),
    }
    (OUT / "confirm_summary.json").write_text(json.dumps(summary, indent=2))
    rates = (
        test_df.groupby("method")
        .agg(
            success_pct=("success", lambda s: round(100 * float(s.mean()), 1)),
            mean_queries=("oracle_calls", "mean"),
            mean_dist=("normalized_distance", "mean"),
        )
        .sort_values("success_pct", ascending=False)
    )
    rates.to_csv(OUT / "test_summary.csv")
    print(rates, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_multiseed(store: EditGraphStore, seed: int, seeds: list[int]) -> pd.DataFrame:
    test_eps = (
        pd.read_parquet(SUITE_DIR / "endpoint_walk_n1000_disjoint.parquet")
        .sort_values("episode")
        .to_dict("records")
    )
    direct = train_direct_dest_scorer(
        TRAIN, "linear", max_rows=20_000, seed=seed, conditioned=False
    )
    frames = []
    for s in seeds:
        ckpt = SEED_CKPT / f"seed_{s}" / "effect_predictor.pt"
        if not ckpt.exists():
            print(f"SKIP missing {ckpt}", flush=True)
            continue
        mlp = EffectScorer(ckpt)
        print(f"=== Multi-seed MLP seed={s} on confirmatory n={len(test_eps)} ===", flush=True)
        df_m = eval_beam(
            name=f"Effect MLP seed={s}",
            episodes=test_eps,
            store=store,
            scorer=mlp,
            ranker="effect",
            seed=seed,
        )
        df_m["train_seed"] = s
        df_m["method_family"] = "Effect MLP"
        frames.append(df_m)
        df_d = eval_beam(
            name="Direct ridge (dest FP)",
            episodes=test_eps,
            store=store,
            scorer=direct,
            ranker="direct",
            seed=seed,
        )
        df_d["train_seed"] = s
        df_d["method_family"] = "Direct ridge"
        frames.append(df_d)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT / "multiseed_per_episode.parquet", index=False)

    # Per-seed MLP vs Direct
    comps = []
    for s, g in df.groupby("train_seed"):
        a = g[g.method_family == "Effect MLP"].set_index("episode")["success"]
        b = g[g.method_family == "Direct ridge"].set_index("episode")["success"]
        stats = mcnemar(a, b)
        comps.append(
            {
                "train_seed": int(s),
                "mlp_pct": round(100 * float(a.mean()), 1),
                "direct_pct": round(100 * float(b.mean()), 1),
                "gap_pp": round(stats["gap_pp"], 1),
                "p_value": stats["p_value"],
                "a_only": stats["a_only"],
                "b_only": stats["b_only"],
            }
        )
    comp_df = pd.DataFrame(comps)
    comp_df.to_csv(OUT / "multiseed_mlp_vs_direct.csv", index=False)
    summary = {
        "n_seeds": int(len(comp_df)),
        "mlp_mean_pct": round(float(comp_df.mlp_pct.mean()), 2),
        "mlp_std_pct": round(float(comp_df.mlp_pct.std(ddof=1)), 2)
        if len(comp_df) > 1
        else 0.0,
        "direct_mean_pct": round(float(comp_df.direct_pct.mean()), 2),
        "gap_mean_pp": round(float(comp_df.gap_pp.mean()), 2),
        "gap_std_pp": round(float(comp_df.gap_pp.std(ddof=1)), 2)
        if len(comp_df) > 1
        else 0.0,
        "per_seed": comps,
    }
    (OUT / "multiseed_summary.json").write_text(json.dumps(summary, indent=2))
    print(comp_df.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return df


def run_structured_missing(
    store: EditGraphStore,
    models: dict,
    seed: int,
    paid_budget: int,
    n_episodes: int,
) -> pd.DataFrame:
    episodes = (
        pd.read_parquet(FROZEN200)
        .sort_values("episode")
        .head(n_episodes)
        .to_dict("records")
    )
    # Scaffold block from episode starts
    scaf_counts: dict[str, int] = {}
    for ep in episodes:
        sc = murcko(ep["start_smiles"])
        if sc:
            scaf_counts[sc] = scaf_counts.get(sc, 0) + 1
    ordered = sorted(scaf_counts, key=scaf_counts.get, reverse=True)
    n_block = max(1, int(len(ordered) * 0.25))
    blocked = set(ordered[:n_block])

    # Property-region paid set: precompute SMILES in central 50% box (sample)
    mols = store.con.execute(
        "SELECT smiles, qed, logp, mw FROM molecules USING SAMPLE 80000"
    ).fetchdf()
    q_lo, q_hi = mols["qed"].quantile([0.25, 0.75])
    l_lo, l_hi = mols["logp"].quantile([0.25, 0.75])
    m_lo, m_hi = mols["mw"].quantile([0.25, 0.75])
    inside = mols[
        (mols.qed >= q_lo)
        & (mols.qed <= q_hi)
        & (mols.logp >= l_lo)
        & (mols.logp <= l_hi)
        & (mols.mw >= m_lo)
        & (mols.mw <= m_hi)
    ]
    paid_smiles = set(inside["smiles"].astype(str))

    # Also mark any episode-neighborhood molecules by querying store for starts' props
    # (membership only via precomputed set + on-the-fly add for evaluated molecules)
    def enrich_paid(smiles: str) -> None:
        if smiles in paid_smiles:
            return
        props = store.molecule(smiles)
        if props is None:
            return
        if (
            q_lo <= props.qed <= q_hi
            and l_lo <= props.logp <= l_hi
            and m_lo <= props.mw <= m_hi
        ):
            paid_smiles.add(smiles)

    for ep in episodes:
        enrich_paid(ep["start_smiles"])

    class PropertyRegionLazy(PropertyRegionIndex):
        def is_free(self, smiles: str) -> bool:
            enrich_paid(smiles)
            return super().is_free(smiles)

    settings = {
        "all_paid": AllPaid(),
        "mcar_f0.5": MCARIndex(0.5, seed=seed),
        "scaffold_block": ScaffoldBlockIndex(blocked),
        "property_region": PropertyRegionLazy(paid_smiles),
    }
    meta = {
        "paid_budget_Q": paid_budget,
        "hide_availability": True,
        "n_blocked_scaffolds": len(blocked),
        "property_box": {
            "qed": [float(q_lo), float(q_hi)],
            "logp": [float(l_lo), float(l_hi)],
            "mw": [float(m_lo), float(m_hi)],
        },
        "n_paid_smiles_precomputed": len(paid_smiles),
        "protocol": (
            "Hard paid unique-node budget Q; free reads do not increment the counter; "
            "scorer never receives the free/paid bit. Property-region membership uses a "
            "precomputed/on-query SMILES set (not returned as features)."
        ),
    }
    (OUT / "structured_missing_meta.json").write_text(json.dumps(meta, indent=2))

    focus = {
        "Effect beam (MLP)": models["Effect beam (MLP)"],
        "Direct ridge (dest FP)": models["Direct ridge (dest FP)"],
        "Random beam": models["Random beam"],
    }
    frames = []
    for setting_name, index in settings.items():
        for charge_all in (False, True):
            tag = "charge_all" if charge_all else "paid_only"
            for method, (scorer, ranker) in focus.items():
                df = eval_hard_budget_hidden(
                    name=f"{method}|{setting_name}|{tag}",
                    episodes=episodes,
                    store=store,
                    scorer=scorer,
                    ranker=ranker,
                    index=index,
                    seed=seed,
                    paid_budget=paid_budget,
                    charge_all=charge_all,
                )
                df["setting"] = setting_name
                df["base_method"] = method
                df["accounting"] = tag
                frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT / "structured_missing_per_episode.parquet", index=False)
    summary = (
        out.groupby(["accounting", "setting", "base_method"], sort=False)
        .agg(
            success_pct=("success", lambda s: round(100 * float(s.mean()), 1)),
            mean_paid=("paid_queries", "mean"),
            mean_free=("free_hits", "mean"),
            mean_dist=("normalized_distance", "mean"),
        )
        .reset_index()
    )
    summary["mean_paid"] = summary["mean_paid"].round(2)
    summary["mean_free"] = summary["mean_free"].round(2)
    summary.to_csv(OUT / "structured_missing_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=20_000)
    parser.add_argument("--paid-budget", type=int, default=15)
    parser.add_argument("--missing-episodes", type=int, default=200)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--mlp-ckpt",
        type=Path,
        default=ROOT / "checkpoints/v2/effect_predictor.pt",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["build", "confirm", "multiseed", "missing"],
        choices=["build", "confirm", "multiseed", "missing", "all"],
    )
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    args = parser.parse_args()
    if "all" in args.stages:
        args.stages = ["build", "confirm", "multiseed", "missing"]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    if "build" in args.stages:
        build_suites(1000, 200, args.seed, force=args.force_rebuild)

    store = EditGraphStore(DB)
    models = fit_models(args.max_train_rows, args.seed, args.mlp_ckpt)

    if "confirm" in args.stages:
        run_confirm(models, store, args.seed)

    if "multiseed" in args.stages:
        run_multiseed(store, args.seed, args.train_seeds)

    if "missing" in args.stages:
        run_structured_missing(
            store, models, args.seed, args.paid_budget, args.missing_episodes
        )

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
