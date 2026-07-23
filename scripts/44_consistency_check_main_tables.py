#!/usr/bin/env python3
"""Automatic consistency checks for JCIM main-table frozen numbers.

Verifies:
  1. Protocol-freeze headline rates (Effect 79.5, Direct 77.5, Random 58.5, MMP 66.5)
  2. Table-2 path/density strata reweight to those overalls
  3. Exact unique-node query means (571.5 / 61.5 / …)
  4. Manuscript/JCIM tex do not contain forbidden legacy Random/MMP rates
  5. No \\ref{tab:matched_arch} in main (SI-only label)
  6. Main Exact-property caption does not cite archived 82.5%

Exit nonzero on any failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/consistency_checks"
EPS = 1e-6

# Forbidden as *headline* Random/MMP rates in main manuscript (legacy archives).
FORBIDDEN_MAIN_PATTERNS = [
    (r"Random beam\s*&\s*54\.0", "legacy Random 54.0 in table row"),
    (r"Random beam reaches 54\.0", "legacy Random 54.0 in prose"),
    (r"Matched-pair / MMP\s*&\s*62\.5", "legacy MMP 62.5"),
    (r"Matched-pair / MMP\s*&\s*64\.0", "legacy MMP 64.0 success (should be 66.5)"),
    (r"Random 89\.7", "legacy path Random 89.7"),
    (r"Random 52\.1", "legacy path Random 52.1"),
    (r"Random 13\.0\\%", "legacy path Random 13.0"),
    (r"Random 49\.3", "legacy dens Random 49.3"),
    (r"Random 59\.7", "legacy dens Random 59.7"),
    (r"archived live-RDKit run was \$82\.5", "82.5 must stay out of main Table-1 caption"),
    (r"\\ref\{tab:matched_arch\}", "SI-only label cannot be \\ref'd from main"),
]


def approx(a: float, b: float, tol: float = 0.051) -> bool:
    """Allow 0.05pp rounding (e.g. 91.379 -> 91.4)."""
    return abs(float(a) - float(b)) <= tol


def check_freeze_headlines() -> list[str]:
    errs: list[str] = []
    pri = pd.read_parquet(ROOT / "results/protocol_freeze/primary_per_episode.parquet")
    expect = {
        "Effect beam (MLP)": 79.5,
        "Direct ridge (dest FP)": 77.5,
        "Random beam": 58.5,
        "Matched-pair / MMP": 66.5,
        "Static MW heuristic": 74.0,
        "Ridge effect": 76.0,
    }
    for name, rate in expect.items():
        g = pri[pri.method == name]
        if g.empty:
            errs.append(f"missing method in freeze: {name}")
            continue
        got = 100.0 * g.success.mean()
        if not approx(got, rate, 0.051):
            errs.append(f"freeze {name}: got {got:.2f} expected {rate}")
    return errs


def check_strata_reweight() -> list[str]:
    errs: list[str] = []
    path = ROOT / "results/benchmark_diagnostics/frozen_stratified_success.json"
    if not path.exists():
        return [f"missing {path}; run scripts/43_frozen_stratified_diagnostics.py"]
    meta = json.loads(path.read_text())
    if not approx(meta["random_overall"], 58.5) or not approx(meta["effect_overall"], 79.5):
        errs.append(
            f"stratified meta overalls {meta['effect_overall']}/{meta['random_overall']} "
            "≠ 79.5/58.5"
        )

    def weighted(rows: list[dict], key: str) -> float:
        n = sum(r["n"] for r in rows)
        return sum(r["n"] * r[key] for r in rows) / n

    for label, rows in [
        ("path_length", meta["path_length"]),
        ("density_tertiles", meta["density_tertiles"]),
    ]:
        wr = weighted(rows, "random_pct")
        we = weighted(rows, "effect_pct")
        if not approx(wr, 58.5, 0.06):
            errs.append(f"{label} weighted Random {wr:.3f} ≠ 58.5")
        if not approx(we, 79.5, 0.06):
            errs.append(f"{label} weighted Effect {we:.3f} ≠ 79.5")

    # Cross-check against live merge (not only JSON).
    diag = pd.read_parquet(
        ROOT / "results/benchmark_diagnostics/per_episode_diagnostics.parquet"
    )
    pri = pd.read_parquet(ROOT / "results/protocol_freeze/primary_per_episode.parquet")
    eff = pri[pri.method == "Effect beam (MLP)"][["episode", "success"]].rename(
        columns={"success": "effect"}
    )
    rnd = pri[pri.method == "Random beam"][["episode", "success"]].rename(
        columns={"success": "random"}
    )
    df = diag.merge(eff, on="episode").merge(rnd, on="episode")
    if len(df) != 200:
        errs.append(f"diag∩freeze episodes = {len(df)} (want 200)")
    if not approx(100 * df.random.mean(), 58.5) or not approx(100 * df.effect.mean(), 79.5):
        errs.append("live merge overalls drifted from freeze")
    return errs


def check_exact_queries() -> list[str]:
    errs: list[str] = []
    summary = ROOT / "results/exact_baselines_unique_node/summary.csv"
    pe = ROOT / "results/exact_baselines_unique_node/per_episode.parquet"
    if summary.exists():
        df = pd.read_csv(summary)
        means = df.set_index("method")["mean_paid"]
        succ = df.set_index("method")["success_pct"]
    elif pe.exists():
        df = pd.read_parquet(pe)
        means = df.groupby("method")["paid_queries"].mean()
        succ = df.groupby("method")["success"].mean() * 100
    else:
        return [f"missing exact baselines under {summary.parent}"]

    expect_q = {
        "Exhaustive depth-3": 571.5,
        "A* best-first": 61.5,
        "Exact-property beam": 53.4,
    }
    expect_s = {"Exact-property beam": 82.0, "Exhaustive depth-3": 100.0}
    for name, q in expect_q.items():
        if name not in means.index:
            errs.append(f"exact summary missing {name}")
            continue
        if not approx(means[name], q, 0.15):
            errs.append(f"{name} mean q {means[name]} ≠ {q}")
    for name, s in expect_s.items():
        if name not in succ.index:
            continue
        if not approx(succ[name], s, 0.15):
            errs.append(f"{name} success {succ[name]} ≠ {s}")
    return errs


def check_tex_forbidden() -> list[str]:
    errs: list[str] = []
    files = [
        ROOT / "paper/manuscript.tex",
        ROOT / "paper/templates/jcim/main.tex",
    ]
    for path in files:
        text = path.read_text()
        for pat, msg in FORBIDDEN_MAIN_PATTERNS:
            if re.search(pat, text):
                errs.append(f"{path.relative_to(ROOT)}: {msg} ({pat})")
    # Main must contain frozen Table-2 Random strata
    ms = (ROOT / "paper/manuscript.tex").read_text()
    for needle in ["Random 91.4", "Random 57.3", "Random 19.6", "Random 46.3", "Random 70.1"]:
        if needle not in ms:
            errs.append(f"manuscript missing frozen stratum text: {needle}")
    if "Point-est.\\ Pareto" not in ms and "Point-est. Pareto" not in ms:
        if "Point-est" not in ms:
            errs.append("manuscript missing Point-est Pareto wording for Table 8")
    if "SI matched-architecture table" not in ms and "SI matched-arch" not in ms:
        errs.append("manuscript claims table should cite SI matched-architecture table")
    return errs


def check_f0_identity() -> list[str]:
    path = ROOT / "results/protocol_freeze/f0_identity_check.json"
    if not path.exists():
        return [f"missing {path}"]
    meta = json.loads(path.read_text())
    errs = []
    if not meta.get("all_success_identical") or not meta.get("all_queries_identical"):
        errs.append("f0 identity check failed")
    return errs


def check_query_ledger() -> list[str]:
    """Start-free unique-node means for Effect/Direct match Table 1."""
    errs: list[str] = []
    pri = ROOT / "results/protocol_freeze/primary_per_episode.parquet"
    if not pri.exists():
        return [f"missing {pri}"]
    df = pd.read_parquet(pri)
    expect = {
        "Effect beam (MLP)": (79.5, 13.8),
        "Direct ridge (dest FP)": (77.5, 14.9),
        "Random beam": (58.5, 23.8),
    }
    qcol = "paid_queries" if "paid_queries" in df.columns else "queries"
    if qcol not in df.columns:
        # try common aliases
        for c in ("unique_node_queries", "n_queries", "queries_start_free"):
            if c in df.columns:
                qcol = c
                break
        else:
            return [f"no query column in {pri.name}: {list(df.columns)}"]
    for name, (succ, q) in expect.items():
        g = df[df.method == name]
        if g.empty:
            errs.append(f"ledger missing {name}")
            continue
        got_s = 100.0 * g.success.mean()
        got_q = float(g[qcol].mean())
        if not approx(got_s, succ):
            errs.append(f"ledger {name} success {got_s:.2f} ≠ {succ}")
        if not approx(got_q, q, 0.15):
            errs.append(f"ledger {name} queries {got_q:.2f} ≠ {q}")
    return errs


def check_fig1_caption_numbers() -> list[str]:
    errs: list[str] = []
    ms = (ROOT / "paper/manuscript.tex").read_text()
    # LaTeX: Effect MLP $79.5\%$@$13.8$q vs.\ direct ridge $77.5\%$@$14.9$q
    needles = [
        r"79\.5\\%@\$13\.8\$q",
        r"77\.5\\%@\$14\.9\$q",
    ]
    # Caption must use table numbers, not "~15 queries"
    if re.search(r"~15\s*quer", ms, re.I) or re.search(r"approx\.?\s*15\s*quer", ms, re.I):
        errs.append("manuscript still has ~15-query schematic wording")
    for pat in needles:
        if not re.search(pat, ms):
            errs.append(f"manuscript missing Fig1/table-aligned pattern: {pat}")
    # Source figure script must match
    fig_py = ROOT / "scripts/23_make_toc_and_method_figures.py"
    if fig_py.exists():
        src = fig_py.read_text()
        if "13.8q" not in src or "14.9q" not in src:
            errs.append("figure script missing 13.8q/14.9q labels")
        if "~15" in src or "15 queries" in src:
            errs.append("figure script still has ~15 queries wording")
    return errs


def check_pdf_unresolved_refs() -> list[str]:
    errs: list[str] = []
    for rel in [
        "paper/templates/jcim/main.pdf",
        "paper/templates/jcim/JCIM_manuscript.pdf",
        "paper/templates/jcim/si.pdf",
        "paper/templates/jcim/JCIM_SI.pdf",
    ]:
        path = ROOT / rel
        if not path.exists():
            continue
        # PDF text extraction via pdftotext if available; else raw bytes for '??'
        try:
            import subprocess

            r = subprocess.run(
                ["pdftotext", "-layout", str(path), "-"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            text = r.stdout if r.returncode == 0 else ""
        except (FileNotFoundError, OSError):
            text = ""
        raw = path.read_bytes()
        # Unresolved LaTeX refs often appear as '??' in text layer
        if text:
            # Ignore URLs / emails; look for isolated ??
            bad = re.findall(r"(?<![.?0-9])\?\?(?![.?0-9])", text)
            # Filter common false positives in ACS templates
            if bad and re.search(r"Table\s*\?\?|Figure\s*\?\?|Eq(?:uation)?\.\s*\?\?|Section\s*\?\?", text):
                errs.append(f"{rel}: unresolved cross-ref (Table/Figure/Eq ??)")
            elif len(bad) >= 3 and "??" in text:
                # soft: many ?? might be ligature artifacts; only flag cite patterns
                if re.search(r"\[\?\?\]|\(\?\?\)", text):
                    errs.append(f"{rel}: unresolved citation markers")
        # Always flag literal '??' near Table in raw if compressed streams allow
        if b"Table ??" in raw or b"Figure ??" in raw or b"Table??" in raw:
            errs.append(f"{rel}: raw PDF contains Table/Figure ??")
    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=OUT / "report.json")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sections = {
        "freeze_headlines": check_freeze_headlines(),
        "strata_reweight": check_strata_reweight(),
        "exact_queries": check_exact_queries(),
        "tex_forbidden": check_tex_forbidden(),
        "f0_identity": check_f0_identity(),
        "query_ledger": check_query_ledger(),
        "fig1_caption_numbers": check_fig1_caption_numbers(),
        "pdf_unresolved_refs": check_pdf_unresolved_refs(),
    }
    n_err = sum(len(v) for v in sections.values())
    report = {"ok": n_err == 0, "n_errors": n_err, "sections": sections}
    args.json_out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    if n_err:
        print(f"\nFAILED: {n_err} consistency error(s)", file=sys.stderr)
        return 1
    print("\nPASSED: main-table consistency checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
