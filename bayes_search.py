#!/usr/bin/env python3
"""
wandb_sweep.py
A zero-invasive W&B sweep launcher for your KalmanNet zoo.

- Creates a separate sweep for each (model, dataset) pair.
- Spawns `trainer.py` as a subprocess, so we don't touch your trainer at all.
- Parses epoch logs like: [Epoch 001] train_loss=... train_nrmse=... | val_loss=... val_nrmse=...
- Loads the saved JSON metrics (val/test) that trainer.py writes and pushes them to W&B summary.
- Saves each run into ./checkpoints/sweeps/<project>/<model>/<dataset>/<run_id> for cleanliness.

Usage examples:
  # Single combo, Bayesian search
  python wandb_sweep.py --project kalman-sweeps --entity myteam \
    --dataset lorenz --model knet_v2 --method bayes --count 30 --device cuda

  # Run all model-dataset combos, random search, 20 trials each
  python wandb_sweep.py --project kalman-sweeps --entity myteam --all --count 20

Requires:
  pip install wandb
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import wandb

# ----------------------------
# Constants and Registrations
# ----------------------------

# Valid datasets and models, kept in sync with trainer.py
DATASETS = ["lorenz", "pendulum"]
MODELS = ["knet", "bayes_knet", "autoreg_kf", "recursive_knet", "rkn"]

# Default fixed dataset args (we don't sweep dataset difficulty by default)
DEFAULT_DATASET_ARGS = {
    "lorenz": {
        "num_trajectories": 3200,
        "sequence_length": 200,
        "process_noise_std": 0.1,
        "measurement_noise_std": 0.3,
        "partial_obs": True,
        "device": "cpu",
    },
    "pendulum": {
        "num_trajectories": 3200,
        "sequence_length": 200,
        "process_noise_std": 0.05,
        "measurement_noise_std": 0.1,
        "control_amplitude": 0.3,
        "device": "cpu",
    },
}

# Common training search space
COMMON_SWEEP_PARAMS = {
    # Optimizer hparams
    "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 3e-3},
    "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
    # Training knobs
    "batch_size": {"values": [32, 64, 128]},
    "grad_clip": {"values": [0.5, 1.0, 2.0]},
    "epochs": {"values": [15, 20, 30]},
    # Data loader
    "num_workers": {"values": [0]},
    # RNG
    "seed": {"values": [42, 1337, 2025]},
}

# Model-specific search spaces (merged with COMMON_SWEEP_PARAMS)
MODEL_SWEEP_PARAMS = {
    "knet_v1": {
        "m.hidden_dim": {"values": [256, 512, 1024, 2048]},
        "m.num_layers": {"values": [1, 2, 3]},
    },
    "knet_v2": {
        "m.in_mult": {"values": [3, 5, 7]},
        "m.out_mult": {"values": [20, 40, 60]},
    },
    "bayes_knet": {
        "m.hidden_dim": {"values": [256, 512, 1024]},
        "m.num_layers": {"values": [1, 2]},
        "m.dropout_p": {"distribution": "uniform", "min": 0.05, "max": 0.5},
        "m.num_mc_samples": {"values": [5, 10, 20]},
    },
    "autoreg_kf": {
        "m.ar_order": {"values": [1, 2, 3, 4]},
        "m.hidden_dim": {"values": [64, 128, 256]},
        # keep num_objects=1 by default unless you really want to simulate multi-object
    },
    "recursive_knet": {
        "m.config.nb_layer_FC1": {"values": [1, 2, 3]},
        "m.config.FC1_mult": {"values": [1, 2, 3]},
        "m.config.nb_layer_GRU": {"values": [1, 2]},
        "m.config.hidden_size_mult": {"values": [1, 2, 3]},
        "m.config.nb_layer_FC2": {"values": [1, 2, 3]},
        "m.config.FC2_mult": {"values": [1, 2, 3]},
        "m.config.gain_bound": {"values": [0.5, 1.0]},
        "m.config.chol_diag_eps": {"values": [1e-8, 1e-6]},
        "m.config.R_jitter": {"values": [1e-8, 1e-6]},
    },
    "rkn": {
        "m.latent_state_dim": {"values": [12, 15, 20, 24]},
        "m.latent_obs_dim": {"values": [8, 10, 12]},
        "m.num_basis": {"values": [8, 12, 16, 20]},
        "m.bandwidth": {"values": [2, 3, 4]},
        # enc/dec width presets
        "m.encoder_hidden_units": {"values": ["[128,64]", "[256,128]"]},
        "m.decoder_hidden_units": {"values": ["[64,128]", "[128,256]"]},
    },
}

# ----------------------------
# Helpers
# ----------------------------

EPOCH_RE = re.compile(
    r"\[Epoch\s+(\d+)\]\s+train_loss=([\-0-9\.eE]+)\s+train_nrmse=([\-0-9\.eE]+)\s+\|\s+val_loss=([\-0-9\.eE]+)\s+val_nrmse=([\-0-9\.eE]+)"
)


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_dumps_compact(d: Dict[str, Any]) -> str:
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


def _unflatten(prefix: str, flat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a nested dict from keys like "m.config.nb_layer_FC1" (or "d.*").
    Keeps only keys that start with the given prefix.
    """
    result: Dict[str, Any] = {}
    plen = len(prefix)
    for k, v in flat.items():
        if not k.startswith(prefix):
            continue
        path = k[plen:].split(".")  # drop prefix and split
        cursor = result
        for part in path[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[path[-1]] = v
    return result


def _coerce_types_for_json(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    wandb.config values come in as strings sometimes; fix obvious JSON-encoded lists.
    """
    fixed = {}
    for k, v in d.items():
        if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
            try:
                fixed[k] = json.loads(v)
                continue
            except Exception:
                pass
        fixed[k] = v
    return fixed


def build_sweep_config(
    model: str, dataset: str, method: str, project: str
) -> Dict[str, Any]:
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choices: {MODELS}")
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choices: {DATASETS}")
    if method not in {"bayes", "random", "grid"}:
        raise ValueError("method must be one of: bayes | random | grid")

    # Merge param spaces
    parameters = {}
    parameters.update(COMMON_SWEEP_PARAMS)
    parameters.update(MODEL_SWEEP_PARAMS.get(model, {}))

    sweep = {
        "name": f"{project}-{model}-{dataset}",
        "method": method,
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": parameters,
        "early_terminate": {"type": "hyperband", "s": 2, "eta": 3, "max_iter": 30},
    }
    return sweep


def build_trainer_command(
    model: str, dataset: str, base_save_dir: Path, config: Dict[str, Any], device: str
) -> Tuple[str, Path]:
    """
    Build the CLI string for trainer.py from wandb.config.
    """
    cfg = _coerce_types_for_json(dict(config))

    # Split out model_args (m.*) and dataset_args (d.*) from config
    model_args = _unflatten("m.", cfg)
    dataset_args = DEFAULT_DATASET_ARGS[dataset].copy()  # fixed by default
    dataset_args.update(_unflatten("d.", cfg))

    # Core training args
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    epochs = int(cfg.get("epochs", 20))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    seed = int(cfg.get("seed", 42))

    run_id = wandb.run.id if wandb.run else f"run_{int(time.time())}"
    save_dir = base_save_dir / model / dataset / run_id
    _mkdir(save_dir)

    cmd = (
        f"{shlex.quote(sys.executable)} trainer.py "
        f"--dataset {shlex.quote(dataset)} "
        f"--model {shlex.quote(model)} "
        f"--dataset-args {shlex.quote(_json_dumps_compact(dataset_args))} "
        f"--model-args {shlex.quote(_json_dumps_compact(model_args))} "
        f"--batch-size {batch_size} "
        f"--num-workers {num_workers} "
        f"--lr {lr} "
        f"--weight-decay {weight_decay} "
        f"--epochs {epochs} "
        f"--grad-clip {grad_clip} "
        f"--device {shlex.quote(device)} "
        f"--seed {seed} "
        f"--val-ratio 0.1 "
        f"--test-ratio 0.0 "
        f"--save-dir {shlex.quote(str(save_dir))} "
        f"--viz-samples 1 --viz-dims 3 --viz-set val"
    )
    return cmd, save_dir


def stream_and_log(proc: subprocess.Popen):
    """
    Stream trainer.py stdout, mirror to console, parse epoch metrics, and log to wandb.
    """
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").rstrip()
        print(line)
        m = EPOCH_RE.search(line)
        if m and wandb.run is not None:
            epoch = int(m.group(1))
            tr_loss = float(m.group(2))
            tr_nrmse = float(m.group(3))
            val_loss = float(m.group(4))
            val_nrmse = float(m.group(5))
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": tr_loss,
                    "train/nrmse": tr_nrmse,
                    "val/loss": val_loss,
                    "val/nrmse": val_nrmse,
                },
                step=epoch,
            )


def finalize_and_log_artifacts(save_dir: Path, model: str, dataset: str):
    """
    After trainer exits, pull JSON metrics into W&B summary and log plots as artifacts if present.
    """
    # metrics path as per trainer.py naming
    metrics_val = save_dir / f"{model}_{dataset}_val_metrics.json"
    if metrics_val.exists():
        with open(metrics_val, "r") as f:
            data = json.load(f)
        overall = data.get("overall", {})
        # push to summary
        for k, v in overall.items():
            wandb.run.summary[f"val/{k}"] = v

    # log comparison plots if any
    pngs = sorted(save_dir.glob("val_sample*_dim*.png"))
    if pngs:
        art = wandb.Artifact(
            name=f"plots-{model}-{dataset}-{wandb.run.id}", type="plots"
        )
        for p in pngs:
            art.add_file(str(p), name=p.name)
        wandb.log_artifact(art)


def run_training():
    """
    W&B agent entrypoint function. Builds the command and runs trainer.py.
    """
    cfg = dict(wandb.config)
    model = (
        wandb.run.tags[0].split("=")[-1]
        if wandb.run and wandb.run.tags
        else cfg.get("_model")
    )
    dataset = (
        wandb.run.tags[1].split("=")[-1]
        if wandb.run and wandb.run.tags
        else cfg.get("_dataset")
    )
    device = cfg.get("_device", "cuda" if torch_cuda_available() else "cpu")
    base_save_dir = Path(cfg.get("_base_save_dir", "./checkpoints/sweeps")).resolve()
    _mkdir(base_save_dir)

    cmd, save_dir = build_trainer_command(model, dataset, base_save_dir, cfg, device)

    # record for provenance
    wandb.config.update(
        {"_resolved_cmd": cmd, "_save_dir": str(save_dir)}, allow_val_change=True
    )

    env = os.environ.copy()
    # ensure python finds the local modules when spawned
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + (
        os.pathsep + str(Path(__file__).parent.resolve())
    )

    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent.resolve()),
        env=env,
        bufsize=1,
    )

    # read epoch logs live
    try:
        stream_and_log(proc)
    finally:
        proc.wait()

    # Pull JSON metrics and plots
    finalize_and_log_artifacts(save_dir, model, dataset)


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ----------------------------
# CLI
# ----------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=str, default="kalman-filter")
    ap.add_argument("--entity", type=str, default=None, help="adnanhd")
    ap.add_argument("--dataset", type=str, choices=DATASETS, default=None)
    ap.add_argument("--model", type=str, choices=MODELS, default=None)
    ap.add_argument(
        "--all", action="store_true", help="Launch sweeps for all (model,dataset) pairs"
    )
    ap.add_argument(
        "--method", type=str, choices=["bayes", "random", "grid"], default="bayes"
    )
    ap.add_argument("--count", type=int, default=20, help="Trials per sweep")
    ap.add_argument(
        "--device", type=str, default="cuda" if torch_cuda_available() else "cpu"
    )
    ap.add_argument("--base-save-dir", type=str, default="./checkpoints/sweeps")
    return ap.parse_args()


def launch_one(
    project: str,
    entity: str,
    model: str,
    dataset: str,
    method: str,
    count: int,
    device: str,
    base_save_dir: str,
):
    sweep_cfg = build_sweep_config(model, dataset, method, project)

    # Add hidden defaults so run_training knows what to do
    sweep_cfg["parameters"].update(
        {
            "_model": {"value": model},
            "_dataset": {"value": dataset},
            "_device": {"value": device},
            "_base_save_dir": {"value": base_save_dir},
        }
    )

    sweep_id = wandb.sweep(sweep=sweep_cfg, project=project, entity=entity)
    print(f"Created sweep {sweep_id} for ({model}, {dataset})")

    # Agent will run `count` trials sequentially
    def _agent():
        # decorate runs with stable tags for clarity
        tags = [f"model={model}", f"dataset={dataset}"]
        with wandb.init(project=project, entity=entity, tags=tags):
            run_training()

    # Wandb agent uses a function signature; we wrap to pass tags via init inside run_training
    wandb.agent(sweep_id, function=_agent, count=count, project=project, entity=entity)


def main():
    args = parse_args()

    if not args.all and (args.dataset is None or args.model is None):
        print(
            "Pick a single combo with --dataset and --model, or use --all.",
            file=sys.stderr,
        )
        sys.exit(2)

    combos = []
    if args.all:
        for m in MODELS:
            for d in DATASETS:
                combos.append((m, d))
    else:
        combos.append((args.model, args.dataset))

    for m, d in combos:
        launch_one(
            project=args.project,
            entity=args.entity,
            model=m,
            dataset=d,
            method=args.method,
            count=args.count,
            device=args.device,
            base_save_dir=args.base_save_dir,
        )


if __name__ == "__main__":
    main()
