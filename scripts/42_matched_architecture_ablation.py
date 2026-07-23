#!/usr/bin/env python3
"""Matched architecture cross-ablation: Effect↔Direct × Ridge↔MLP.

Isolates whether Suite~A's MLP–ridge residual is Δ-parameterization vs
architecture / feature design.

Canonical ledger matches scripts/40_protocol_freeze.py:
  start-free unique-node; seed=42+episode; A=5,B=5,T=3; max_rows=15000.

Core 2×2 (matched training rows=15000):
  Effect context→Δ : ridge / mlp
  Direct dest→p    : ridge / mlp

Matched-info extensions:
  Direct context→p : ridge / mlp   (Effect features, absolute target)
  Effect dest→Δ    : ridge / mlp   (Direct features, delta target)

Also reports the production Effect MLP checkpoint (1e6-pair train) as reference.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.beam_planner import BeamPlanner, EditGraphStore, EffectScorer, Properties
from src.matched_ablations import (
    load_torch_scorer,
    save_torch_scorer,
    train_matched_model,
)
from src.search_baselines import train_direct_dest_scorer, train_sklearn_effect_scorer

SCALES = np.array([0.1, 1.0, 30.0], dtype=np.float64)
TOL = np.array([0.03, 0.20, 10.0], dtype=np.float64)
OUT = ROOT / "results/matched_architecture_ablation"
PROD_MLP = ROOT / "checkpoints/v2/effect_predictor.pt"
DIRECT_PKL = ROOT / "results/protocol_freeze/direct_ridge_canonical.pkl"


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    p = float(binomtest(n10, n01 + n10, 0.5).pvalue) if (n01 + n10) else 1.0
    return {
        "n10": n10,
        "n01": n01,
        "delta_pp": float(100.0 * (a.mean() - b.mean())),
        "p_value": p,
    }


def make_planner(store, scorer, ranker: str, seed: int) -> BeamPlanner:
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
    planner.count_start_query = False
    return planner


def eval_episode(ep: dict, store, scorer, ranker: str, seed: int) -> dict:
    target = Properties(ep["target_qed"], ep["target_logp"], ep["target_mw"])
    planner = make_planner(store, scorer, ranker, seed + int(ep["episode"]))
    t0 = time.perf_counter()
    result = planner.plan(target, start_smiles=ep["start_smiles"])
    runtime = time.perf_counter() - t0
    err = np.abs(result.properties.array() - target.array())
    success = bool(np.all(err <= TOL))
    dist = float(np.linalg.norm(err / SCALES))
    q = int(planner.last_stats["oracle_calls"])
    return {
        "episode": int(ep["episode"]),
        "success": success,
        "normalized_distance": dist,
        "paid_queries": q,
        "runtime_sec": runtime,
    }


def bootstrap_ci(succ: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(succ)
    rates = [
        float(succ[rng.integers(0, n, n)].mean() * 100.0) for _ in range(n_boot)
    ]
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--max-train-rows", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    train = ROOT / "data/processed/effect_splits_v2/train.parquet"
    episodes = (
        pd.read_parquet(ROOT / "data/processed/episode_suites/endpoint_walk.parquet")
        .sort_values("episode")
        .head(args.max_episodes)
        .to_dict("records")
    )
    store = EditGraphStore(ROOT / "data/processed/search_graph.duckdb")

    # Spec: (display_name, arch, feature, target)
    specs = [
        ("Effect ridge (context→Δ)", "ridge", "context", "delta"),
        ("Effect MLP (context→Δ, 15k)", "mlp", "context", "delta"),
        ("Direct ridge (dest→p)", "ridge", "dest", "absolute"),
        ("Direct MLP (dest→p)", "mlp", "dest", "absolute"),
        ("Direct ridge (context→p)", "ridge", "context", "absolute"),
        ("Direct MLP (context→p)", "mlp", "context", "absolute"),
        ("Effect ridge (dest→Δ)", "ridge", "dest", "delta"),
        ("Effect MLP (dest→Δ)", "mlp", "dest", "delta"),
    ]

    models: dict[str, tuple] = {}
    train_meta: dict[str, dict] = {}

    # Production Effect MLP reference (full train).
    models["Effect MLP (prod, 1e6)"] = (EffectScorer(PROD_MLP, device=args.device), "effect")
    train_meta["Effect MLP (prod, 1e6)"] = {
        "arch": "mlp",
        "feature": "context",
        "target": "delta",
        "note": "checkpoints/v2/effect_predictor.pt",
    }

    # Prefer frozen Direct ridge pickle when present.
    if DIRECT_PKL.exists():
        with open(DIRECT_PKL, "rb") as f:
            models["Direct ridge (dest→p)"] = (pickle.load(f), "direct")
        train_meta["Direct ridge (dest→p)"] = {
            "arch": "ridge",
            "feature": "dest",
            "target": "absolute",
            "source": str(DIRECT_PKL),
        }

    for name, arch, feature, target in specs:
        if name in models:
            continue
        ckpt = OUT / f"{name.replace(' ', '_').replace('→', 'to').replace(',', '')}.pt"
        meta_path = ckpt.with_suffix(".json")
        pkl_path = ckpt.with_suffix(".pkl")

        if args.skip_train and (ckpt.exists() or pkl_path.exists()):
            if ckpt.exists():
                scorer = load_torch_scorer(ckpt, device=args.device)
                ranker = json.loads(meta_path.read_text())["ranker"]
            else:
                with open(pkl_path, "rb") as f:
                    bundle = pickle.load(f)
                scorer, ranker = bundle["scorer"], bundle["ranker"]
            models[name] = (scorer, ranker)
            train_meta[name] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            continue

        print(f"Training {name}...", flush=True)
        scorer, ranker, meta = train_matched_model(
            train,
            arch=arch,
            feature=feature,
            target=target,
            max_rows=args.max_train_rows,
            seed=args.seed,
            epochs=args.epochs,
            device=args.device,
        )
        meta["name"] = name
        train_meta[name] = meta
        meta_path.write_text(json.dumps(meta, indent=2))
        if arch == "mlp":
            save_torch_scorer(ckpt, scorer, meta)
        else:
            with open(pkl_path, "wb") as f:
                pickle.dump({"scorer": scorer, "ranker": ranker, "meta": meta}, f)
        models[name] = (scorer, ranker)
        print(
            f"  done: mode={meta['mode']} n={meta.get('n_rows')} "
            f"params={meta.get('n_params')} val={meta.get('best_val_loss')}",
            flush=True,
        )

    # Canonical Effect ridge via existing helper (sanity / exact recipe match).
    if "Effect ridge (context→Δ)" not in models:
        models["Effect ridge (context→Δ)"] = (
            train_sklearn_effect_scorer(
                train, "linear", max_rows=args.max_train_rows, seed=args.seed
            ),
            "effect",
        )

    rows = []
    for name, (scorer, ranker) in models.items():
        for ep in tqdm(episodes, desc=name):
            r = eval_episode(ep, store, scorer, ranker, args.seed)
            r["method"] = name
            rows.append(r)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "episode_detail.csv", index=False)

    summary_rows = []
    piv = detail.pivot(index="episode", columns="method", values="success")
    q_piv = detail.pivot(index="episode", columns="method", values="paid_queries")
    for name in models:
        succ = piv[name].to_numpy(dtype=bool)
        qs = q_piv[name].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(succ, seed=args.seed)
        summary_rows.append(
            {
                "method": name,
                "success_pct": float(succ.mean() * 100),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "mean_q": float(qs.mean()),
                "median_q": float(np.median(qs)),
                **{f"meta_{k}": v for k, v in train_meta.get(name, {}).items() if k in {
                    "arch", "feature", "target", "mode", "n_rows", "n_params", "in_dim"
                }},
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("success_pct", ascending=False)
    summary.to_csv(OUT / "summary.csv", index=False)

    # Key pairwise contrasts vs production Effect MLP and vs Direct ridge.
    ref_mlp = "Effect MLP (prod, 1e6)"
    ref_direct = "Direct ridge (dest→p)"
    contrasts = []
    for name in models:
        if name == ref_mlp:
            continue
        contrasts.append({"vs": ref_mlp, "method": name, **mcnemar(piv[name], piv[ref_mlp])})
    if ref_direct in piv.columns:
        for name in models:
            if name == ref_direct:
                continue
            contrasts.append(
                {"vs": ref_direct, "method": name, **mcnemar(piv[name], piv[ref_direct])}
            )
    # Architecture isolation: same features, MLP vs ridge
    pairs = [
        ("Effect MLP (context→Δ, 15k)", "Effect ridge (context→Δ)"),
        ("Direct MLP (dest→p)", "Direct ridge (dest→p)"),
        ("Direct MLP (context→p)", "Direct ridge (context→p)"),
        ("Effect MLP (dest→Δ)", "Effect ridge (dest→Δ)"),
        # Parameterization isolation: same arch, effect vs direct features/target
        ("Effect MLP (context→Δ, 15k)", "Direct MLP (dest→p)"),
        ("Effect ridge (context→Δ)", "Direct ridge (dest→p)"),
        ("Effect MLP (context→Δ, 15k)", "Direct MLP (context→p)"),
        ("Direct MLP (dest→p)", "Effect MLP (dest→Δ)"),
    ]
    for a, b in pairs:
        if a in piv.columns and b in piv.columns:
            contrasts.append({"vs": b, "method": a, **mcnemar(piv[a], piv[b])})

    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(OUT / "contrasts.csv", index=False)

    (OUT / "train_meta.json").write_text(json.dumps(train_meta, indent=2, default=str))
    (OUT / "config.json").write_text(
        json.dumps(
            {
                "max_episodes": args.max_episodes,
                "max_train_rows": args.max_train_rows,
                "seed": args.seed,
                "epochs": args.epochs,
                "methods": list(models.keys()),
            },
            indent=2,
        )
    )

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print("\n=== Key contrasts ===")
    key = contrast_df[
        contrast_df["method"].isin(
            [
                "Direct MLP (dest→p)",
                "Effect MLP (context→Δ, 15k)",
                "Effect ridge (context→Δ)",
                "Direct MLP (context→p)",
                "Effect MLP (dest→Δ)",
            ]
        )
    ]
    print(key.to_string(index=False))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
