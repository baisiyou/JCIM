"""Supervised fragment-edit policy for an M⁴olGen-style baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class EditPolicy(nn.Module):
    """
    Scores an edit by predicted distance improvement toward a numeric target.

    Input: [fp(mol) || target_norm(3) || fp(frag_old) || fp(frag_new) || edit_oh]
    Output: scalar ŷ ≈ ||p_cur - t||_s - ||p_next - t||_s
    """

    def __init__(
        self,
        fp_bits: int = 2048,
        hidden: int = 512,
        dropout: float = 0.1,
        n_edit_types: int = 3,
        target_dim: int = 3,
    ):
        super().__init__()
        in_dim = fp_bits * 3 + target_dim + n_edit_types
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.fp_bits = fp_bits
        self.n_edit_types = n_edit_types
        self.target_dim = target_dim

    def forward(
        self,
        fp_mol: torch.Tensor,
        target_norm: torch.Tensor,
        fp_old: torch.Tensor,
        fp_new: torch.Tensor,
        edit_oh: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([fp_mol, target_norm, fp_old, fp_new, edit_oh], dim=-1)
        return self.net(x).squeeze(-1)
