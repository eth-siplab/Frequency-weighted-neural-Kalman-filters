#!/usr/bin/env bash
set -uo pipefail

# Reproduce all paper table experiments with fixed code (fix/fft-loss-and-iir branch)
# Config matches reproduce/ folder: lr=1e-3, wd=0, sched=none, epochs=30, batch=128

PY="/home/adnan/miniconda3/envs/torch/bin/python"
SAVE_DIR="./checkpoints/reproduce_fixed_v2"
EPOCHS=30
BATCH=128
LR="1e-3"
WD="0"
SCHED="none"
DEVICE="cuda:0"

# ── GPU scheduling ────────────────────────────────────────────────────────────
# Before each launch, check actual free GPU memory via nvidia-smi.
# Need at least _MIN_FREE_MB free to launch another job.
# After the first few jobs, _MIN_FREE_MB = mean of last 5 jobs' memory usage.
GPU_ID="${DEVICE##*:}"
[[ "$GPU_ID" == "cuda" ]] && GPU_ID=0

_MIN_FREE_MB_DEFAULT=4000   # 4 GB fallback before we have measurements
_job_mem=()

# Query actual free GPU memory right now (MB)
_gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free \
    --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null | tr -d ' '
}

# Query how much memory a specific PID is using (MB)
_pid_gpu_mb() {
  nvidia-smi --query-compute-apps=pid,used_gpu_memory \
    --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null \
    | awk -F', ' -v p="$1" '$1==p {print $2; exit}'
}

# Record a job's peak memory over 30s into the rolling window (last 5)
_record_job_mem() {
  local pid=$1 mem max=0
  for _ in 1 2 3; do
    sleep 10
    mem=$(_pid_gpu_mb "$pid")
    if [[ "$mem" =~ ^[0-9]+$ ]] && (( mem > max )); then max=$mem; fi
  done
  if (( max > 0 )); then
    _job_mem+=("$max")
    (( ${#_job_mem[@]} > 5 )) && _job_mem=("${_job_mem[@]:1}")
  fi
}

# Min free memory required = max of recorded job memory × 1.5, or default
_min_free_mb() {
  local n=${#_job_mem[@]}
  if (( n == 0 )); then echo "$_MIN_FREE_MB_DEFAULT"; return; fi
  local max=0
  for m in "${_job_mem[@]}"; do (( m > max )) && max=$m; done
  echo $(( max * 3 / 2 ))
}

# Wait until GPU has enough free memory for one more job
wait_for_gpu_slot() {
  local needed free
  needed=$(_min_free_mb)
  while :; do
    free=$(_gpu_free_mb)
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= needed )); then
      break
    fi
    sleep 5
  done
}

MODELS=(deep_kf kalman_net classical_kf bayes_knet autoreg_kf recurrent_kalman_network recursive_knet)
DATASETS=(lorenz pendulum)
FFTS=(0 0.01 0.1)
SEEDS=(0 1 2)

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#FFTS[@]} * ${#SEEDS[@]} ))
echo "Total runs: $TOTAL | Save dir: $SAVE_DIR"
echo "GPU scheduling: launch when free VRAM >= avg job mem (default ${_MIN_FREE_MB_DEFAULT} MB) on GPU ${GPU_ID}"

mkdir -p "$SAVE_DIR"

i=0
for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do
    for fft in "${FFTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        out="${SAVE_DIR}/${m}_${d}_fft${fft}_lr${LR}_wd${WD}_${SCHED}_seed${seed}"

        # Skip if already completed
        if [ -f "${out}/${m}_${d}_test_metrics.json" ]; then
          echo "[SKIP] $out (test metrics exist)"
          continue
        fi

        ((i+=1))
        echo "[${i}/${TOTAL}] ${m} ${d} fft=${fft} seed=${seed}"

        $PY trainer.py \
          --model "$m" --dataset "$d" \
          --epochs "$EPOCHS" --batch-size "$BATCH" \
          --lr "$LR" --weight-decay "$WD" --scheduler "$SCHED" \
          --fft-weight "$fft" --seed "$seed" \
          --early-stop --early-restore-best --early-patience 10 \
          --val-ratio 0.1 --test-ratio 0.1 \
          --viz-samples 2 \
          --save-dir "$out" --device "$DEVICE" --quiet &

        _record_job_mem $!
        wait_for_gpu_slot
      done
    done
  done
done

wait
echo "Synthetic experiments completed. Results in $SAVE_DIR"

# # ── EuRoC MAV IMU (real robotics data) ────────────────────────────────────────
# # Uses real IMU observations (gyro+accel) with 10-D state (pos+vel+quat).
# # MH_01 excluded: its IMU data was not downloaded; 10 sequences remain.
# EUROC_DIR="./checkpoints/euroc_tracking"
# EUROC_MODELS=(deep_kf kalman_net classical_kf bayes_knet autoreg_kf recurrent_kalman_network recursive_knet)
# EUROC_FFTS=(0 0.01 0.1)
# EUROC_SEEDS=(0 1 2)
# EUROC_DARGS='{"sequences":["MH_02_easy","MH_03_medium","MH_04_difficult","MH_05_difficult","V1_01_easy","V1_02_medium","V1_03_difficult","V2_01_easy","V2_02_medium","V2_03_difficult"],"download":false}'
#
# EUROC_TOTAL=$(( ${#EUROC_MODELS[@]} * ${#EUROC_FFTS[@]} * ${#EUROC_SEEDS[@]} ))
# echo ""
# echo "EuRoC runs: $EUROC_TOTAL | Save dir: $EUROC_DIR"
# mkdir -p "$EUROC_DIR"
#
# j=0
# for m in "${EUROC_MODELS[@]}"; do
#   for fft in "${EUROC_FFTS[@]}"; do
#     for seed in "${EUROC_SEEDS[@]}"; do
#       out="${EUROC_DIR}/${m}_euroc_tracking_fft${fft}_lr${LR}_wd${WD}_${SCHED}_seed${seed}"
#
#       if [ -f "${out}/${m}_euroc_tracking_test_metrics.json" ]; then
#         echo "[SKIP] $out (test metrics exist)"
#         continue
#       fi
#
#       ((j+=1))
#       echo "[${j}/${EUROC_TOTAL}] ${m} euroc_tracking fft=${fft} seed=${seed}"
#
#       $PY trainer.py \
#         --model "$m" --dataset euroc_tracking \
#         "--dataset-args=${EUROC_DARGS}" \
#         --epochs "$EPOCHS" --batch-size "$BATCH" \
#         --lr "$LR" --weight-decay "$WD" --scheduler "$SCHED" \
#         --nonlinear-obs \
#         --fft-weight "$fft" --seed "$seed" \
#         --early-stop --early-restore-best --early-patience 10 \
#         --val-ratio 0.1 --test-ratio 0.1 \
#         --viz-samples 2 \
#         --save-dir "$out" --device "$DEVICE" --quiet &
#
#       sleep 3
#       _record_job_mem $!
#       wait_for_gpu_slot
#     done
#   done
# done
#
# wait
# echo "All experiments completed (synthetic + EuRoC). Results in $SAVE_DIR and $EUROC_DIR"
