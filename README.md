# EditGraph — JCIM review reproducibility package

Code, environment pins, frozen manifests (SHA-256), checkpoint, and scripts
supporting the JCIM manuscript:

**Auditing Multi-Property Navigation and Label Access on Molecular Edit Graphs**

Authors: Yuran Zhang (McGill University); Rubing Zhang (Vanier College).

This repository is the durable review archive for peer review. It matches the
content previously packaged as `editgraph_anon_review_20260722.tar.gz`.

## Quick start

```bash
conda env create -f environment.yml
conda activate causalmol
python scripts/44_consistency_check_main_tables.py
```

Primary regeneration scripts:

```bash
python scripts/40_protocol_freeze.py --skip-partial
python scripts/41_exact_baselines_unique_node.py
python scripts/43_frozen_stratified_diagnostics.py
```

See `docs/REVIEW_REPRODUCIBILITY.md` and `README_REVIEW.md` for the full
checklist.

## Large artifacts (not in git)

Place these under the SHA-256 hashes recorded in
`results/canonical_frozen200/manifest.json`:

- `data/processed/search_graph.duckdb`
- `data/processed/episode_suites/endpoint_walk.parquet`
- `data/processed/effect_splits_v2/train.parquet`

Raw MOSES molecules follow the original MOSES distribution terms.

## Contents

| Path | Role |
|------|------|
| `src/` | Search, effect model, baselines |
| `scripts/` | Protocol freeze, exact baselines, diagnostics |
| `checkpoints/v2/` | Effect MLP weights + metrics |
| `results/` | Frozen manifests and table summaries |
| `configs/` | Model config |
| `paper/` | Manuscript TeX snapshot |
