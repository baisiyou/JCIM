"""Matched architecture / feature cross-ablations for Effect vs Direct rankers.

2x2 core:
  Effect MLP / Effect ridge  (source+frag → Δ)
  Direct MLP / Direct ridge  (dest FP → p)

Matched-info extensions:
  Direct MLP/ridge with source+frag context → absolute p(m')
  Effect MLP/ridge with destination FP only → Δ
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.featurize import EDIT_TYPE_TO_ID, smiles_to_fp


class PropertyMLP(nn.Module):
    """Same depth/width recipe as EffectPredictor; variable input dim."""

    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
        )
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _fp_cache(smiles: list[str], fp_bits: int, fp_radius: int) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {"": np.zeros(fp_bits, dtype=np.float32)}
    for s in tqdm(sorted({x for x in smiles if x}), desc="Fingerprints", leave=False):
        cache[s] = smiles_to_fp(s, n_bits=fp_bits, radius=fp_radius)
    return cache


def _edit_oh(kind: str) -> np.ndarray:
    v = np.zeros(3, dtype=np.float32)
    v[EDIT_TYPE_TO_ID.get(kind, 0)] = 1.0
    return v


class _ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y_z: np.ndarray, y_raw: np.ndarray):
        self.x = x.astype(np.float32)
        self.y_z = y_z.astype(np.float32)
        self.y_raw = y_raw.astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y_z[idx]),
            torch.from_numpy(self.y_raw[idx]),
        )


def _train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    val_frac: float = 0.1,
) -> tuple[PropertyMLP, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    y_mean = y[tr_idx].mean(axis=0).astype(np.float32)
    y_std = y[tr_idx].std(axis=0).astype(np.float32)
    y_std[y_std < 1e-6] = 1.0
    y_z = (y - y_mean) / y_std

    train_loader = DataLoader(
        _ArrayDataset(x[tr_idx], y_z[tr_idx], y[tr_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        _ArrayDataset(x[val_idx], y_z[val_idx], y[val_idx]),
        batch_size=batch_size,
        shuffle=False,
    )

    torch.manual_seed(seed)
    model = PropertyMLP(in_dim=x.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.SmoothL1Loss()

    best_state = None
    best_val = float("inf")
    patience_left = 3
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)

        model.eval()
        val_abs = np.zeros(3, dtype=np.float64)
        val_n = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, yraw in val_loader:
                pred = model(xb.to(device))
                val_loss += crit(pred, yb.to(device)).item() * xb.size(0)
                pred_raw = pred.cpu().numpy() * y_std + y_mean
                val_abs += np.abs(pred_raw - yraw.numpy()).sum(axis=0)
                val_n += xb.size(0)
        val_loss /= max(val_n, 1)
        mae = (val_abs / max(val_n, 1)).tolist()
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss / max(tr_n, 1),
                "val_loss": val_loss,
                "val_mae": mae,
            }
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 3
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "best_val_loss": float(best_val),
        "epochs_run": len(history),
        "history": history,
        "n_params": int(n_params),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "in_dim": int(x.shape[1]),
    }
    return model.cpu(), y_mean, y_std, meta


class TorchPropertyScorer:
    """Generic torch scorer with Direct / Effect / dest-Δ / context-Direct modes."""

    def __init__(
        self,
        model: PropertyMLP,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        *,
        mode: str,
        fp_bits: int = 2048,
        fp_radius: int = 2,
        device: str = "auto",
    ):
        self.model = model
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)
        self.mode = mode
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.device = _device(device)
        self.model.to(self.device)
        self.model.eval()
        self._fp_cache: dict[str, np.ndarray] = {
            "": np.zeros(fp_bits, dtype=np.float32)
        }
        self.uses_destination = mode == "effect_dest"

    def _fp(self, smiles: str) -> np.ndarray:
        if smiles not in self._fp_cache:
            self._fp_cache[smiles] = smiles_to_fp(
                smiles, n_bits=self.fp_bits, radius=self.fp_radius
            )
        return self._fp_cache[smiles]

    def _predict_x(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            pred = self.model(torch.from_numpy(x.astype(np.float32)).to(self.device))
        return pred.cpu().numpy() * self.y_std + self.y_mean

    def predict_properties(
        self,
        dest_smiles: list[str],
        source_props: np.ndarray | None = None,
        sources: list[str] | None = None,
        frag_old: list[str] | None = None,
        frag_new: list[str] | None = None,
        edit_types: list[str] | None = None,
        **_kwargs,
    ) -> np.ndarray:
        if not dest_smiles:
            return np.empty((0, 3), dtype=np.float32)
        if self.mode == "direct_dest":
            x = np.stack([self._fp(s) for s in dest_smiles])
        elif self.mode == "direct_context":
            assert sources is not None and frag_old is not None
            assert frag_new is not None and edit_types is not None
            rows = []
            for src, old, new, kind in zip(sources, frag_old, frag_new, edit_types):
                rows.append(
                    np.concatenate(
                        [
                            self._fp(src),
                            self._fp(str(old or "")),
                            self._fp(str(new or "")),
                            _edit_oh(kind),
                        ]
                    )
                )
            x = np.stack(rows)
        else:
            raise RuntimeError(f"predict_properties not valid for mode={self.mode}")
        return self._predict_x(x).astype(np.float32)

    def predict(
        self,
        sources: list[str],
        frag_old: list[str],
        frag_new: list[str],
        edit_types: list[str],
    ) -> np.ndarray:
        if not sources:
            return np.empty((0, 3), dtype=np.float32)
        if self.mode != "effect_context":
            raise RuntimeError(f"predict() expects effect_context; got {self.mode}")
        rows = []
        for src, old, new, kind in zip(sources, frag_old, frag_new, edit_types):
            rows.append(
                np.concatenate(
                    [
                        self._fp(src),
                        self._fp(str(old or "")),
                        self._fp(str(new or "")),
                        _edit_oh(kind),
                    ]
                )
            )
        return self._predict_x(np.stack(rows)).astype(np.float32)

    def predict_dest(self, dest_smiles: list[str]) -> np.ndarray:
        if not dest_smiles:
            return np.empty((0, 3), dtype=np.float32)
        if self.mode != "effect_dest":
            raise RuntimeError(f"predict_dest expects effect_dest; got {self.mode}")
        x = np.stack([self._fp(s) for s in dest_smiles])
        return self._predict_x(x).astype(np.float32)


class RidgeFeatureScorer:
    """Ridge on matched feature recipes (context or dest)."""

    def __init__(
        self,
        model: Ridge,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        *,
        mode: str,
        fp_bits: int = 2048,
        fp_radius: int = 2,
    ):
        self.model = model
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)
        self.mode = mode
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self._fp_cache: dict[str, np.ndarray] = {
            "": np.zeros(fp_bits, dtype=np.float32)
        }
        self.uses_destination = mode == "effect_dest"

    def _fp(self, smiles: str) -> np.ndarray:
        if smiles not in self._fp_cache:
            self._fp_cache[smiles] = smiles_to_fp(
                smiles, n_bits=self.fp_bits, radius=self.fp_radius
            )
        return self._fp_cache[smiles]

    def _predict_x(self, x: np.ndarray) -> np.ndarray:
        pred = np.asarray(self.model.predict(x), dtype=np.float32)
        return pred * self.y_std + self.y_mean

    def predict_properties(
        self,
        dest_smiles: list[str],
        source_props: np.ndarray | None = None,
        sources: list[str] | None = None,
        frag_old: list[str] | None = None,
        frag_new: list[str] | None = None,
        edit_types: list[str] | None = None,
        **_kwargs,
    ) -> np.ndarray:
        if not dest_smiles:
            return np.empty((0, 3), dtype=np.float32)
        if self.mode == "direct_dest":
            x = np.stack([self._fp(s) for s in dest_smiles])
        elif self.mode == "direct_context":
            assert sources and frag_old and frag_new and edit_types
            rows = [
                np.concatenate(
                    [
                        self._fp(src),
                        self._fp(str(old or "")),
                        self._fp(str(new or "")),
                        _edit_oh(kind),
                    ]
                )
                for src, old, new, kind in zip(sources, frag_old, frag_new, edit_types)
            ]
            x = np.stack(rows)
        else:
            raise RuntimeError(f"predict_properties not valid for mode={self.mode}")
        return self._predict_x(x).astype(np.float32)

    def predict(
        self,
        sources: list[str],
        frag_old: list[str],
        frag_new: list[str],
        edit_types: list[str],
    ) -> np.ndarray:
        if not sources:
            return np.empty((0, 3), dtype=np.float32)
        if self.mode != "effect_context":
            raise RuntimeError(f"predict() expects effect_context; got {self.mode}")
        rows = [
            np.concatenate(
                [
                    self._fp(src),
                    self._fp(str(old or "")),
                    self._fp(str(new or "")),
                    _edit_oh(kind),
                ]
            )
            for src, old, new, kind in zip(sources, frag_old, frag_new, edit_types)
        ]
        return self._predict_x(np.stack(rows)).astype(np.float32)

    def predict_dest(self, dest_smiles: list[str]) -> np.ndarray:
        if not dest_smiles:
            return np.empty((0, 3), dtype=np.float32)
        if self.mode != "effect_dest":
            raise RuntimeError(f"predict_dest expects effect_dest; got {self.mode}")
        return self._predict_x(np.stack([self._fp(s) for s in dest_smiles])).astype(
            np.float32
        )


def _build_xy(
    train_pairs: Path,
    *,
    feature: str,
    target: str,
    max_rows: int,
    fp_bits: int,
    fp_radius: int,
    unique_dest: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cols = [
        "smiles_a",
        "smiles_b",
        "edit_type",
        "frag_old",
        "frag_new",
        "delta_qed",
        "delta_logp",
        "delta_mw",
        "qed_b",
        "logp_b",
        "mw_b",
    ]
    df = pd.read_parquet(train_pairs, columns=cols)
    if unique_dest:
        df = df.drop_duplicates("smiles_b")
    df = df.head(max_rows)

    smiles_needed = (
        df["smiles_a"].tolist()
        + df["smiles_b"].tolist()
        + df["frag_old"].fillna("").astype(str).tolist()
        + df["frag_new"].fillna("").astype(str).tolist()
    )
    cache = _fp_cache(smiles_needed, fp_bits, fp_radius)

    rows = []
    for row in df.itertuples(index=False):
        if feature == "dest":
            rows.append(cache[str(row.smiles_b)])
        elif feature == "context":
            rows.append(
                np.concatenate(
                    [
                        cache[str(row.smiles_a)],
                        cache[str(row.frag_old or "")],
                        cache[str(row.frag_new or "")],
                        _edit_oh(row.edit_type),
                    ]
                )
            )
        else:
            raise ValueError(feature)
    x = np.stack(rows)
    if target == "delta":
        y = df[["delta_qed", "delta_logp", "delta_mw"]].to_numpy(dtype=np.float32)
    elif target == "absolute":
        y = df[["qed_b", "logp_b", "mw_b"]].to_numpy(dtype=np.float32)
    else:
        raise ValueError(target)
    info = {
        "feature": feature,
        "target": target,
        "n_rows": int(len(df)),
        "unique_dest": unique_dest,
        "max_rows": max_rows,
    }
    return x, y, info


def train_matched_model(
    train_pairs: Path,
    *,
    arch: str,
    feature: str,
    target: str,
    max_rows: int = 15_000,
    seed: int = 42,
    fp_bits: int = 2048,
    fp_radius: int = 2,
    hidden: int = 512,
    dropout: float = 0.1,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "auto",
) -> tuple[object, str, dict]:
    """Return (scorer, ranker_str, meta)."""
    unique_dest = feature == "dest" and target == "absolute"
    x, y, info = _build_xy(
        train_pairs,
        feature=feature,
        target=target,
        max_rows=max_rows,
        fp_bits=fp_bits,
        fp_radius=fp_radius,
        unique_dest=unique_dest,
    )

    if feature == "dest" and target == "absolute":
        mode = "direct_dest"
        ranker = "direct"
    elif feature == "context" and target == "absolute":
        mode = "direct_context"
        ranker = "direct"
    elif feature == "context" and target == "delta":
        mode = "effect_context"
        ranker = "effect"
    elif feature == "dest" and target == "delta":
        mode = "effect_dest"
        ranker = "effect"
    else:
        raise ValueError((feature, target))

    meta = {"arch": arch, "mode": mode, "ranker": ranker, **info}

    if arch == "ridge":
        y_mean = y.mean(axis=0).astype(np.float32)
        y_std = y.std(axis=0).astype(np.float32)
        y_std[y_std < 1e-6] = 1.0
        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(x, (y - y_mean) / y_std)
        scorer = RidgeFeatureScorer(
            model, y_mean, y_std, mode=mode, fp_bits=fp_bits, fp_radius=fp_radius
        )
        meta["n_params"] = int(np.prod(model.coef_.shape) + model.coef_.shape[0])
        return scorer, ranker, meta

    if arch == "mlp":
        model, y_mean, y_std, train_meta = _train_mlp(
            x,
            y,
            hidden=hidden,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            device=_device(device),
        )
        scorer = TorchPropertyScorer(
            model,
            y_mean,
            y_std,
            mode=mode,
            fp_bits=fp_bits,
            fp_radius=fp_radius,
            device=device,
        )
        meta.update(train_meta)
        return scorer, ranker, meta

    raise ValueError(arch)


def save_torch_scorer(path: Path, scorer: TorchPropertyScorer, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": scorer.model.state_dict(),
            "y_mean": scorer.y_mean,
            "y_std": scorer.y_std,
            "mode": scorer.mode,
            "fp_bits": scorer.fp_bits,
            "fp_radius": scorer.fp_radius,
            "in_dim": scorer.model.in_dim,
            "meta": meta,
        },
        path,
    )


def load_torch_scorer(path: Path, device: str = "auto") -> TorchPropertyScorer:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PropertyMLP(in_dim=int(ckpt["in_dim"]))
    model.load_state_dict(ckpt["model_state"])
    return TorchPropertyScorer(
        model,
        np.asarray(ckpt["y_mean"]),
        np.asarray(ckpt["y_std"]),
        mode=ckpt["mode"],
        fp_bits=int(ckpt["fp_bits"]),
        fp_radius=int(ckpt["fp_radius"]),
        device=device,
    )
