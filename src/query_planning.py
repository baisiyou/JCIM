"""Hidden-label query planning protocol and amortized cost accounting."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb


@dataclass
class PrecomputeCost:
    """One-time offline cost to build the searchable edit graph."""

    annotate_sec: float | None = None
    neighbor_pairs_sec: float | None = None
    index_build_sec: float | None = None
    effect_train_sec: float | None = None
    n_molecules: int = 0
    n_edges: int = 0
    n_sources: int = 0
    database_bytes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeCost:
  """Per-episode online query cost."""

  oracle_calls: int = 0
  scored_edges: int = 0
  expanded_states: int = 0
  runtime_sec: float = 0.0
  precompute_amortized_sec: float = 0.0

  @property
  def total_sec(self) -> float:
      return self.runtime_sec + self.precompute_amortized_sec


def read_precompute_cost(database: Path, logs_dir: Path | None = None) -> PrecomputeCost:
    """Summarize offline graph statistics; parse build logs when present."""
    cost = PrecomputeCost()
    if database.exists():
        cost.database_bytes = database.stat().st_size
        con = duckdb.connect(str(database), read_only=True)
        cost.n_molecules = int(con.execute("SELECT COUNT(*) FROM molecules").fetchone()[0])
        cost.n_edges = int(con.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        cost.n_sources = int(con.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
        con.close()

    if logs_dir and logs_dir.exists():
        for name, field in (
            ("step1_annotate.log", "annotate_sec"),
            ("step2_neighbors.log", "neighbor_pairs_sec"),
        ):
            path = logs_dir / name
            if path.exists() and "seconds" in path.read_text(errors="ignore").lower():
                # best-effort: look for trailing timing lines in our pipeline logs
                for line in reversed(path.read_text(errors="ignore").splitlines()[-20:]):
                    if "sec" in line.lower():
                        try:
                            token = [t for t in line.replace("=", " ").split() if t.replace(".", "", 1).isdigit()][-1]
                            setattr(cost, field, float(token))
                        except (IndexError, ValueError):
                            pass
                        break
    return cost


def amortized_precompute_sec(cost: PrecomputeCost, n_episodes: int) -> float:
    """Spread one-time offline work across evaluated episodes."""
    if n_episodes <= 0:
        return 0.0
    parts = [
        cost.annotate_sec or 0.0,
        cost.neighbor_pairs_sec or 0.0,
        cost.index_build_sec or 0.0,
        cost.effect_train_sec or 0.0,
    ]
    return sum(parts) / n_episodes


def write_cost_report(path: Path, precompute: PrecomputeCost, episode_rows: list[dict]) -> None:
    n = max(len(episode_rows), 1)
    amort = amortized_precompute_sec(precompute, n)
    summary = {
        "protocol": "hidden_node_label_query_planning",
        "precompute": precompute.to_dict(),
        "amortized_precompute_sec_per_episode": amort,
        "online": {
            "mean_oracle_calls": sum(r.get("oracle_calls", 0) for r in episode_rows) / n,
            "mean_runtime_sec": sum(r.get("runtime_sec", 0.0) for r in episode_rows) / n,
            "mean_total_sec": sum(
                r.get("runtime_sec", 0.0) + amort for r in episode_rows
            )
            / n,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


LABEL_MODE_HIDDEN = "hidden"
LABEL_MODE_TRANSDUCTIVE = "transductive"
