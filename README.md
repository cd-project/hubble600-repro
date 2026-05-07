# Hubble-600 Full Repro Bundle

Self-contained bundle for reproducing the Hubble-600 multi-path audit run.

## Included
- `code/checker_ourdef_memorization_noscaffold_dualbest.py` (core evaluator)
- `data/sampled_targets_hubble600.csv` (600 targets; 100 per dup bin)
- `data/DATASET_INFO.json` (schema + hash)
- `scripts/run_hubble600.sh` (main run entry)
- `scripts/summarize_hubble600.py` (result summarization)
- `requirements.txt`

## Setup
```bash
cd repro_hubble600_full
python -m pip install -r requirements.txt
```

## Run
```bash
# optionally set a specific Python env and model
# export PYTHON_BIN=/home/seal12/miniconda3/envs/hubbleeval/bin/python
# export MODEL_NAME=allegrolab/hubble-1b-100b_toks-perturbed-hf

bash scripts/run_hubble600.sh
```

## Summarize
```bash
python scripts/summarize_hubble600.py --run_dir outputs/repro_hubble600_run_<timestamp>
```

## Expected artifacts
Inside each run dir:
- `results_live.csv`, `target_compute_live.csv`
- `restart_dual_live.csv`
- `checkpoint_metrics.csv`
- `loss_trajectory_32.csv`
- `path_diag.csv`
- `run_manifest.json`
- `run.log`
