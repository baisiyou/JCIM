#!/usr/bin/env python3
"""Generate JCIM TOC graphic + method/graph-search schematic."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "paper" / "figures",
    ROOT / "paper" / "templates" / "jcim" / "figures",
    ROOT / "paper" / "templates" / "aaai2027" / "figures",
    ROOT / "paper" / "templates" / "neurips2026" / "figures",
    ROOT / "paper" / "templates" / "icml2026" / "figures",
]

# Clean-hit episode 65 endpoints for TOC chemistry
START = "CNC(=O)c1ccc(CN2CCOC(C(N)=O)C2)cc1"
END = "CNC(=O)c1ccc(CN(C)c2ccc(F)c(F)c2)cc1"  # intermediate-ish; use final if available
FINAL = "Cc1nc(Nc2ccc(N(C)Cc3ccc(C(=O)NC)cc3)cc2)sc1"  # may be wrong — load from traj


def _mol_img(smiles: str, size=(280, 220)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=size)


def _load_case_smiles():
    import json

    traj = ROOT / "results" / "case_studies" / "case_clean_ep65" / "trajectory.json"
    if traj.exists():
        data = json.loads(traj.read_text())
        steps = data["steps"]
        return steps[0]["smiles"], steps[-1]["smiles"]
    return START, FINAL


def save_all(name: str, fig: plt.Figure) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=300, bbox_inches="tight", facecolor="white")
        if name.endswith(".png") and d.name == "figures" and "jcim" in str(d):
            # ACS often wants TIFF for TOC as well
            tif = d / name.replace(".png", ".tif")
            fig.savefig(tif, dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", name)


def make_method_schematic() -> None:
    """Main-text Figure: edit-graph search schematic."""
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc="#f7faf8", ec="#2f6f4e", title_c="#1b4332"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.6,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=11, fontweight="bold", color=title_c)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.65 - 0.32 * i, line, ha="center", va="top", fontsize=8.5, color="#243028")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.8,
                color="#4a5560",
            )
        )

    # Row of graph nodes
    ax.text(0.35, 3.85, "Observed BRICS edit graph (MOSES)", fontsize=10, fontweight="bold", color="#1b4332")
    node_xy = [(0.7, 2.9), (1.7, 3.35), (1.7, 2.45), (2.7, 2.9), (3.5, 3.4), (3.5, 2.4)]
    for i, (x, y) in enumerate(node_xy):
        circ = plt.Circle((x, y), 0.22, facecolor="#d8efe3" if i else "#ffe8cc", edgecolor="#2f6f4e", lw=1.4, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, f"$m_{i}$" if i < 4 else ("$\\cdots$" if i == 4 else "$m^\\star$"), ha="center", va="center", fontsize=8, zorder=4)
    # edges
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)]:
        x0, y0 = node_xy[a]
        x1, y1 = node_xy[b]
        ax.plot([x0, x1], [y0, y1], color="#8aa396", lw=1.2, zorder=1)
    ax.text(2.1, 1.95, "replace-only edges; $p(m)$ indexed", ha="center", fontsize=7.5, color="#556066")

    box(
        4.3,
        2.05,
        2.35,
        1.85,
        "Effect model",
        [r"$(m,f_{\mathrm{old}},f_{\mathrm{new}})$", r"$\rightarrow \widehat{\Delta}p$", "trained once"],
        fc="#eef3ff",
        ec="#3a5a9a",
        title_c="#243868",
    )
    box(
        6.95,
        2.05,
        2.35,
        1.85,
        "Beam search",
        ["rank $A{=}5$ edits", "depth $\\leq 3$", "hidden-label queries"],
        fc="#fff6e8",
        ec="#b06a10",
        title_c="#7a4500",
    )
    box(
        9.55,
        2.05,
        2.2,
        1.85,
        "Oracle verify",
        ["RDKit $p(m')$", "accept if", r"$|p-p^\star|\leq\varepsilon$"],
        fc="#fdeeee",
        ec="#a33b3b",
        title_c="#7a1f1f",
    )

    arrow(3.75, 2.9, 4.25, 2.9)
    arrow(6.65, 2.9, 6.9, 2.9)
    arrow(9.3, 2.9, 9.5, 2.9)

    # Bottom strip: two protocols
    box(
        0.35,
        0.25,
        5.4,
        1.45,
        "Protocol A — observed graph",
        ["read cached labels (0 queries)", "exact BFS / $A^\\ast$ ceiling: 100% endpoint-walk"],
        fc="#f4f7f5",
        ec="#5a7264",
        title_c="#2c3e34",
    )
    box(
        6.0,
        0.25,
        5.75,
        1.45,
        "Protocol B — hidden labels (ML budget)",
        [
            "withhold cached $p(m')$; 1 query / unique node",
            "Effect 79.5%@13.8q vs Direct 77.5%@14.9q",
        ],
        fc="#f7f5f2",
        ec="#6e6254",
        title_c="#3d3428",
    )

    ax.text(
        6.0,
        4.05,
        "EditGraph: effect-guided navigation on an indexed fragment-edit graph",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#1a2a22",
    )
    save_all("method_schematic.png", fig)
    plt.close(fig)


def make_toc_graphic() -> None:
    """ACS TOC graphic sized for the official 3.25in x 1.75in tocentry box."""
    # High-DPI source; content fills ACS default abstract/TOC graphic aspect (~1.857).
    fig = plt.figure(figsize=(6.5, 3.5))
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.70])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.0)
    ax.axis("off")

    def panel(x, y, w, h, title, lines, ec, fc, title_c):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=2.4,
                edgecolor=ec,
                facecolor=fc,
            )
        )
        ax.text(
            x + w / 2,
            y + h - 0.36,
            title,
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
            color=title_c,
        )
        for i, line in enumerate(lines):
            ax.text(
                x + w / 2,
                y + h - 0.95 - 0.45 * i,
                line,
                ha="center",
                va="top",
                fontsize=11,
                color="#243028",
            )

    panel(
        0.15,
        0.45,
        4.65,
        2.30,
        "Full index",
        ["BFS 100% @ 0 queries", "Learning unnecessary"],
        ec="#2f6f4e",
        fc="#eef7f1",
        title_c="#1b4332",
    )
    panel(
        5.20,
        0.45,
        4.65,
        2.30,
        "Hidden labels",
        ["MLP 79.5% vs Direct 77.5%", "Small residual"],
        ec="#3a5a9a",
        fc="#eef2fb",
        title_c="#243868",
    )
    ax.text(
        5.0,
        0.14,
        "Assumptions determine apparent advantage",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#2c3e50",
    )
    fig.suptitle(
        "Label-access audit on molecular edit graphs",
        fontsize=13,
        color="#1a2a22",
        y=0.96,
        fontweight="bold",
    )

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            d / "toc_graphic.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.02,
        )
        if "jcim" in str(d):
            fig.savefig(
                d / "toc_graphic.tif",
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
                pad_inches=0.02,
            )
    (ROOT / "paper" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(
        ROOT / "paper" / "figures" / "toc_graphic.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.02,
    )
    print("wrote toc_graphic.png/.tif (ACS 3.25x1.75 source)")
    plt.close(fig)


def main() -> int:
    make_method_schematic()
    make_toc_graphic()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
