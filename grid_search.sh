#!/usr/bin/env bash
set -euo pipefail

# --- Defaults ---
GPUS="0"; NPROC=1; PY="python3"; EPOCHS=10; BATCH=128; NUM_WORKERS=0; SAVE_DIR="./checkpoints/minisweep"
PRINT_CMDS=0; QUIET=0; SKIP=0
# default skip patterns for "best" checkpoints
SKIP_GLOBS=(best*.pt best*.pth *best*.ckpt)

# --- Grid (shrinkable via flags) ---
MODELS=(kalman_net classical_kf deep_kf bayes_knet autoreg_kf recurrent_kalman_network recursive_knet)
DATASETS=(lorenz pendulum)
FFTS=(0 0.01 0.1 0.5 1.0 2.0 5.0)
LRS=(1e-3 1e-4)
WDS=(0 1e-4)
SCHEDS=(none cosine)
SEEDS=(0 1 2)

# --- Args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2;;
    --nproc-per-gpu) NPROC="$2"; shift 2;;
    --py) PY="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --save-dir) SAVE_DIR="$2"; shift 2;;
    --print-cmds) PRINT_CMDS=1; shift 1;;
    --quiet) QUIET=1; shift 1;;
    --skip-if-present) SKIP=1; shift 1;;
    --skip-globs) IFS=',' read -r -a SKIP_GLOBS <<< "$2"; shift 2;;
    --model) MODELS=("$2"); shift 2;;
    --dataset) DATASETS=("$2"); shift 2;;
    --fft-weight) FFTS=("$2"); shift 2;;
    --lr) LRS=("$2"); shift 2;;
    --wd) WDS=("$2"); shift 2;;
    --scheduler) SCHEDS=("$2"); shift 2;;
    --seed) SEEDS=("$2"); shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

mkdir -p "$SAVE_DIR"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"; (( ${#GPU_IDS[@]} )) || { echo "No GPUs given" >&2; exit 1; }

# Exactly NPROC slots per GPU
DEVS=(); for g in "${GPU_IDS[@]}"; do for ((k=0;k<NPROC;k++)); do DEVS+=("$g"); done; done
SLOTS=${#DEVS[@]}; (( SLOTS )) || { echo "No slots" >&2; exit 1; }

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#FFTS[@]} * ${#LRS[@]} * ${#WDS[@]} * ${#SCHEDS[@]} * ${#SEEDS[@]} ))
(( TOTAL )) || { echo "No combinations to run. Check your narrowing flags." >&2; exit 1; }
(( QUIET )) || echo "Total runs: $TOTAL | GPUs: ${GPU_IDS[*]} | procs/GPU: $NPROC | out: $SAVE_DIR"

# helper: return 0 if a "best" checkpoint exists for this combo
has_best() {
  local dir="$1"
  shopt -s nullglob

  # Yeni kural: alt klasörde *_best/model.pt
  # Örn: $dir/recurrent_kalman_network_lorenz_best/model.pt
  for f in "$dir"/*_best/model.pt; do
    [[ -e "$f" ]] && return 0
  done

  # Eski kural: üst dizinde çeşitli best kalıpları (opsiyonel)
  for pat in "${SKIP_GLOBS[@]}"; do
    for f in "$dir"/$pat; do
      [[ -e "$f" ]] && return 0
    done
  done

  return 1
}

i=0
for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do
    for fft in "${FFTS[@]}"; do
      for lr in "${LRS[@]}"; do
        for wd in "${WDS[@]}"; do
          for sch in "${SCHEDS[@]}"; do
            for seed in "${SEEDS[@]}"; do
              dev="${DEVS[$(( i % SLOTS ))]}"
              out="${SAVE_DIR}/${m}_${d}_fft${fft}_lr${lr}_wd${wd}_${sch}_seed${seed}"

              if (( SKIP )) && has_best "$out"; then
                (( PRINT_CMDS )) && echo "[SKIP] $out (best checkpoint present)"
                continue
              fi

              cmd=( "$PY" trainer.py --model "$m" --dataset "$d"
                    --epochs "$EPOCHS" --batch-size "$BATCH" --num-workers "$NUM_WORKERS"
                    --lr "$lr" --weight-decay "$wd" --scheduler "$sch" --fft-weight "$fft"
                    --early-stop --early-restore-best --viz-samples 2 --val-ratio 0.1 --test-ratio 0.1
                    --seed "$seed" --save-dir "$out" --device "cuda:0" )

              (( QUIET )) && cmd+=( --quiet )
              (( PRINT_CMDS )) && echo "[GPU ${dev}] ${cmd[*]}"
              CUDA_VISIBLE_DEVICES="$dev" "${cmd[@]}" &
              ((i+=1))
              # throttle: cap parallel jobs at SLOTS
              while :; do
                active=$(jobs -p | wc -l || true)
                (( active < SLOTS )) && break
                wait -n || true
              done
            done
          done
        done
      done
    done
  done
done

wait
