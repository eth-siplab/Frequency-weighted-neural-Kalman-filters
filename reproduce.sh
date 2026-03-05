#!/usr/bin/env bash
set -uo pipefail

# Reproduce all paper table experiments.
# Usage: bash reproduce.sh [--device cuda:0] [--max-parallel 3]

PY="${PY:-python}"
SAVE_DIR="./checkpoints/paper"
EPOCHS=30
BATCH=128
LR="1e-3"
WD="0"
SCHED="none"
DEVICE="cuda:0"
MAX_PARALLEL=3

# Parse optional arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device) DEVICE="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

MODELS=(deep_kf kalman_net classical_kf bayes_knet autoreg_kf recurrent_kalman_network recursive_knet)
FFTS=(0 0.01 0.1)
SEEDS=(0 1 2)

run_experiments() {
  local save_dir="$1"; shift
  local datasets=("$@")

  local total=$(( ${#MODELS[@]} * ${#datasets[@]} * ${#FFTS[@]} * ${#SEEDS[@]} ))
  echo "Runs: $total | Save dir: $save_dir"
  mkdir -p "$save_dir"

  local i=0
  for m in "${MODELS[@]}"; do
    for d in "${datasets[@]}"; do
      for fft in "${FFTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          local out="${save_dir}/${m}_${d}_fft${fft}_seed${seed}"

          if [ -f "${out}/${m}_${d}_test_metrics.json" ]; then
            echo "[SKIP] $out"
            continue
          fi

          ((i+=1))
          echo "[${i}/${total}] ${m} ${d} fft=${fft} seed=${seed}"

          $PY trainer.py \
            --model "$m" --dataset "$d" \
            --epochs "$EPOCHS" --batch-size "$BATCH" \
            --lr "$LR" --weight-decay "$WD" --scheduler "$SCHED" \
            --fft-weight "$fft" --seed "$seed" \
            --early-stop --early-restore-best --early-patience 10 \
            --val-ratio 0.1 --test-ratio 0.1 \
            --viz-samples 2 \
            --save-dir "$out" --device "$DEVICE" --quiet &

          while :; do
            active=$(jobs -rp | wc -l)
            (( active < MAX_PARALLEL )) && break
            wait -n || true
          done
        done
      done
    done
  done
  wait
}

# ── Synthetic datasets ────────────────────────────────────────────────────────
echo "=== Synthetic experiments ==="
run_experiments "$SAVE_DIR" lorenz pendulum

# ── EuRoC MAV tracking (real robotics data) ───────────────────────────────────
echo ""
echo "=== EuRoC MAV tracking ==="
run_experiments "$SAVE_DIR" euroc_tracking

echo ""
echo "All experiments completed. Results in $SAVE_DIR"
