#!/usr/bin/env python3
"""Recompute Table-2 path/density strata using frozen Effect/Random successes.

Merges results/benchmark_diagnostics/per_episode_diagnostics.parquet with
results/protocol_freeze/primary_per_episode.parquet so stratified Random rates
weight to the canonical 58.5% (not the legacy 54.0% archive).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/benchmark_diagnostics"


def main() -> int:
    diag = pd.read_parquet(OUT / "per_episode_diagnostics.parquet")
    pri = pd.read_parquet(ROOT / "results/protocol_freeze/primary_per_episode.parquet")
    effect = pri[pri.method == "Effect beam (MLP)"][["episode", "success"]].rename(
        columns={"success": "effect"}
    )
    rand = pri[pri.method == "Random beam"][["episode", "success"]].rename(
        columns={"success": "random"}
    )
    df = diag.merge(effect, on="episode").merge(rand, on="episode")
    if len(df) != 200:
        raise SystemExit(f"expected 200 episodes, got {len(df)}")
    if abs(df.effect.mean() - 0.795) > 1e-9 or abs(df.random.mean() - 0.585) > 1e-9:
        raise SystemExit(
            f"overall rates drifted: effect={df.effect.mean():.4f} random={df.random.mean():.4f}"
        )

    path_rows = []
    for L, g in df.groupby("shortest_feasible_path"):
        path_rows.append(
            {
                "shortest_path": int(L),
                "n": int(len(g)),
                "effect_pct": round(100 * g.effect.mean(), 1),
                "random_pct": round(100 * g.random.mean(), 1),
                "mean_local_solutions": round(g.solutions_in_depth_le_max.mean(), 1),
            }
        )

    df = df.copy()
    df["density_tertile"] = pd.qcut(
        df["indexed_solution_density"], 3, labels=["Low", "Mid", "High"]
    )
    dens_rows = []
    for t, g in df.groupby("density_tertile", observed=True):
        dens_rows.append(
            {
                "density_tertile": str(t),
                "n": int(len(g)),
                "effect_pct": round(100 * g.effect.mean(), 1),
                "random_pct": round(100 * g.random.mean(), 1),
                "mean_local_solutions": round(g.solutions_in_depth_le_max.mean(), 1),
                "mean_indexed_density": round(g.indexed_solution_density.mean(), 1),
                "mean_shortest_path": round(g.shortest_feasible_path.mean(), 3),
                "effect_beam_success": float(g.effect.mean()),
                "random_beam_success": float(g.random.mean()),
                "cached_bfs_success": 1.0,
            }
        )

    pd.DataFrame(path_rows).to_csv(OUT / "path_length_with_success_frozen.csv", index=False)
    dens_df = pd.DataFrame(dens_rows)
    dens_df.to_csv(OUT / "density_tertiles_with_success_frozen.csv", index=False)
    dens_df.to_csv(OUT / "density_tertiles_with_success.csv", index=False)
    meta = {
        "source_success": "results/protocol_freeze/primary_per_episode.parquet",
        "source_structure": "results/benchmark_diagnostics/per_episode_diagnostics.parquet",
        "effect_overall": 79.5,
        "random_overall": 58.5,
        "path_length": path_rows,
        "density_tertiles": dens_rows,
    }
    (OUT / "frozen_stratified_success.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
