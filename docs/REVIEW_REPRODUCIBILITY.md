# Anonymous review reproducibility snapshot

Prepare before JCIM submission (anonymous GitHub and/or Zenodo/OSF).

## Must include
- `environment.yml`, `requirements.txt`
- `scripts/` (at least `40_protocol_freeze.py`, `41_exact_baselines_unique_node.py`,
  `42_matched_architecture_ablation.py`, `43_frozen_stratified_diagnostics.py`,
  `37_confirm_v2_disjoint.py`)
- `src/`
- `results/canonical_frozen200/manifest.json` (suite/checkpoint/DuckDB SHA-256)
- `results/protocol_freeze/f0_identity_check.json`
- `results/protocol_freeze/primary_summary.csv`
- `results/exact_baselines_unique_node/summary.csv` (queries 571.5 / 61.5 / …)
- `results/benchmark_diagnostics/frozen_stratified_success.json`
- `checkpoints/v2/effect_predictor.pt` (+ metrics JSON) **or** download URL with matching hash
- Pointer/README for obtaining `data/processed/search_graph.duckdb` and episode parquet
  under the hashes in the manifest (large files may be Zenodo-only)

## Verify locally
```bash
conda env create -f environment.yml
python scripts/40_protocol_freeze.py --skip-partial   # f=0 identity
python scripts/41_exact_baselines_unique_node.py      # exact queries
python scripts/43_frozen_stratified_diagnostics.py    # Table-2 strata
```

## Cover letter
Insert the anonymous archive URL and a content hash (Zenodo DOI or git commit SHA)
at submission time; replace with the public repo at camera-ready.
