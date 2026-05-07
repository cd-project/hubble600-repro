#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON_BIN:-python}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/repro_hubble600_run_${TS}"
mkdir -p "$OUT_DIR"

TARGETS="data/sampled_targets_hubble600.csv"
MODEL="${MODEL_NAME:-allegrolab/hubble-1b-100b_toks-perturbed-hf}"

nohup "$PY" code/checker_ourdef_memorization_noscaffold_dualbest.py \
  --sampled_targets_csv "$TARGETS" \
  --model_name_or_path "$MODEL" \
  --output_path "$OUT_DIR/results.csv" \
  --target_compute_output_path "$OUT_DIR/target_compute.csv" \
  --manifest_output_path "$OUT_DIR/run_manifest.json" \
  --live_output_path "$OUT_DIR/results_live.csv" \
  --live_target_compute_output_path "$OUT_DIR/target_compute_live.csv" \
  --live_restart_dual_output_path "$OUT_DIR/restart_dual_live.csv" \
  --checkpoint_metrics_output_path "$OUT_DIR/checkpoint_metrics.csv" \
  --loss_trajectory_output_path "$OUT_DIR/loss_trajectory_32.csv" \
  --path_diag_output_path "$OUT_DIR/path_diag.csv" \
  --num_restarts 5 \
  --steps_per_restart 256 \
  --adv_prefix_len 10 \
  --discrete_optimizer gcg \
  --scaffold_fraction 0.5 \
  --optimize_mode continuation \
  --candidate_topk 96 \
  --candidate_topk_final 96 \
  --batch_size 96 \
  --mini_batch_size 96 \
  --gcg_no_improve_patience 96 \
  --gcg_no_improve_patience_final 96 \
  --checkpoint_steps 128,256 \
  --trajectory_log_every 32 \
  --periodic_decode_every 0 \
  --dtype bfloat16 \
  --seed 0 \
  --continue_on_restart_error \
  --log_every 10 \
  > "$OUT_DIR/run.log" 2>&1 &

PID=$!
echo "started_pid=$PID"
echo "out_dir=$OUT_DIR"
