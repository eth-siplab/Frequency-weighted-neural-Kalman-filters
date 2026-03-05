#!/usr/bin/env python3
"""
Enhanced trainer.py with WandBHandler logging support alongside stream and file logging.
Features proper logging.Handler integration for seamless wandb logging.
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import yaml
from plotly.subplots import make_subplots
from typing_extensions import Callable

import wandb

# Local imports - assuming these exist based on your project structure
from loader import (
    EuRocMAVDataset,
    EuRoCTrackingDataset,
    LorenzDataset,
    NCLTDataset,
    OxfordRobotCarDataset,
    PendulumDataset,
    UWBIMUDataset,
)
from models import (
    AutoRegKF,
    BayesianKalmanNet,
    ClassicalKalmanFilter,
    DeepKalmanFilter,
    KalmanNet,
    RecurrentKalmanNetwork,
    RecursiveKalmanNet,
)

# from spectral_loss import SpectralLoss

KalmanFilterType = Union[
    AutoRegKF,
    BayesianKalmanNet,
    KalmanNet,
    RecurrentKalmanNetwork,
    RecursiveKalmanNet,
    ClassicalKalmanFilter,
    DeepKalmanFilter,
]
CriterionType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class DatasetType(Protocol):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def __len__(self) -> int: ...


# ----------------------------
# WandB Logging Handler
# ----------------------------


class WandBHandler(logging.Handler):
    """Custom logging handler that sends logs to Weights & Biases."""

    def __init__(self, enabled: bool = False, **wandb_kwargs: Any):
        super().__init__()
        self.enabled = enabled
        self.run = None
        self.epoch_regex = re.compile(
            r"\[Epoch\s+(\d+)\]\s+train_loss=([\-0-9\.eE]+)\s+train_nrmse=([\-0-9\.eE]+)\s+\|\s+val_loss=([\-0-9\.eE]+)\s+val_nrmse=([\-0-9\.eE]+)"
        )

        if self.enabled:
            self.run = wandb.init(**wandb_kwargs)

        # Set formatter to just capture the message
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord):
        """Process a logging record and send metrics to wandb if it matches epoch pattern."""
        if not self.enabled or self.run is None:
            return

        try:
            msg = self.format(record)

            # Check if this is an epoch metrics line
            match = self.epoch_regex.search(msg)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                train_nrmse = float(match.group(3))
                val_loss = float(match.group(4))
                val_nrmse = float(match.group(5))

                # Log metrics to wandb
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/nrmse": train_nrmse,
                        "val/loss": val_loss,
                        "val/nrmse": val_nrmse,
                    },
                    step=epoch,
                )

        except Exception:
            # Don't let wandb errors break the training
            self.handleError(record)

    def log_config(self, config: Dict[str, Any]):
        """Log configuration to wandb."""
        if self.enabled and self.run is not None:
            wandb.config.update(config)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Manually log metrics to wandb."""
        if self.enabled and self.run is not None:
            wandb.log(metrics, step=step)

    def log_artifact(self, file_path: str, name: str, artifact_type: str = "file"):
        """Log file as artifact to wandb."""
        if self.enabled and self.run is not None:
            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)

    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.run is not None:
            wandb.finish()
            self.run = None


# ----------------------------
# Data & Model builders
# ----------------------------


def build_dataset(
    name: str, cfg: "TrainConfig"
) -> torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]:
    kwargs = dict(cfg.dataset_args or {})
    if name.lower() in ["lorenz", "lorenzdataset"]:
        return LorenzDataset(**kwargs)
    elif name.lower() in ["pendulum", "pendulumdataset"]:
        return PendulumDataset(**kwargs)
    elif name.lower() in ["robotcar", "robotcardataset"]:
        return OxfordRobotCarDataset(**kwargs)
    elif name.lower() in ["uwbimu", "uipdb", "uwbimudataset"]:
        return UWBIMUDataset(**kwargs)
    elif name.lower() in ["nclt", "ncltdataset"]:
        return NCLTDataset(**kwargs)
    elif name.lower() in ["euroc", "eurocmavdataset"]:
        return EuRocMAVDataset(**kwargs)
    elif name.lower() in ["euroc_tracking", "euroctrackingdataset"]:
        return EuRoCTrackingDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def build_model(
    name: str,
    state_dim: int,
    obs_dim: int,
    control_dim: int,
    cfg: "TrainConfig",
    *,
    nonlinear_obs: bool = False,
) -> nn.Module:
    args = cfg.model_args or {}
    name_l = name.lower()
    if name_l in ["knet", "kalman", "kalmanet", "kalman_net"]:
        return KalmanNet(
            state_dim=state_dim, obs_dim=obs_dim, control_dim=control_dim, **args
        )
    if name_l in ["bayes_knet", "bayesian", "bayesian_kalman", "bayesian_kalman_net"]:
        return BayesianKalmanNet(
            state_dim=state_dim, obs_dim=obs_dim, control_dim=control_dim, **args
        )
    if name_l in ["autoreg_kf", "autoreg", "ar_kf"]:
        return AutoRegKF(
            state_dim=state_dim, obs_dim=obs_dim, **args
        )  # ignore control_dim
    if name_l in ["recursive_knet", "recursive_kalman", "recursive_kalman_net"]:
        return RecursiveKalmanNet(
            state_dim=state_dim, obs_dim=obs_dim, **args
        )  # ignore control_dim
    if name_l in ["rkn", "recurrent_kalman_network", "recurrent_kalman_networks"]:
        return RecurrentKalmanNetwork(
            state_dim=state_dim, obs_dim=obs_dim, **args
        )  # ignore control_dim
    if name_l in ["classical_kf", "ckf", "kf"]:
        return ClassicalKalmanFilter(
            state_dim=state_dim, obs_dim=obs_dim, control_dim=control_dim, **args
        )
    if name_l in ["deep_kf", "dkf"]:
        args.setdefault("use_nonlinear_obs", nonlinear_obs)
        return DeepKalmanFilter(
            state_dim=state_dim, obs_dim=obs_dim, control_dim=control_dim, **args
        )
    raise ValueError(f"Unknown model: {name}")


def split_dataset_three(
    ds: DatasetType, val_ratio: float, test_ratio: float, seed: int = 42
):
    n = len(ds)
    if not (
        0.0 < val_ratio < 1.0
        and 0.0 < test_ratio < 1.0
        and val_ratio + test_ratio < 1.0
    ):
        raise ValueError("val_ratio and test_ratio must be in (0,1) and sum to < 1")
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(
            "Not enough samples for a 3-way split; reduce val_ratio/test_ratio."
        )

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        ds, [n_train, n_val, n_test], generator=gen
    )
    return train_ds, val_ds, test_ds


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: "TrainConfig"):
    if not cfg.scheduler or cfg.scheduler.lower() == "none":
        return None
    args = cfg.scheduler_args or {}
    if cfg.scheduler.lower() == "cosine":
        return lrs.CosineAnnealingLR(optimizer, T_max=cfg.epochs, **args)
    elif cfg.scheduler.lower() == "step":
        return lrs.StepLR(optimizer, step_size=args.get("step_size", 10), **args)
    elif cfg.scheduler.lower() == "plateau":
        return lrs.ReduceLROnPlateau(optimizer, mode="min", **args)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def get_dims_from_dataset(ds: torch.utils.data.Dataset) -> Tuple[int, int, int]:
    for sample in ds:
        obs = sample["observations"]
        states = sample["states"]
        state_dim = states.shape[-1]
        obs_dim = obs.shape[-1]
        control_dim = sample["controls"].shape[-1] if "controls" in sample else 0
        return state_dim, obs_dim, control_dim
    raise RuntimeError("Empty dataset")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ----------------------------
# Logging Setup
# ----------------------------


def setup_logger(
    save_dir: str, wandb_handler: Optional[WandBHandler] = None, quiet: bool = False
) -> logging.Logger:
    """Setup logger with console, file, and optional wandb handlers."""
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear old handlers in case of multiple invocations
    while logger.handlers:
        logger.handlers.pop()

    # Console handler with message-only format (so epoch regex still matches)
    if not quiet:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    # File handler with timestamped format
    fh = logging.FileHandler(os.path.join(save_dir, "training.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    # WandB handler if provided
    if wandb_handler:
        wandb_handler.setLevel(logging.INFO)
        logger.addHandler(wandb_handler)
        # Store reference for manual metric logging

    return logger


# ----------------------------
# Config
# ----------------------------


@dataclass
class TrainConfig:
    dataset: str = "lorenz"
    model: str = "knet"
    dataset_args: Optional[Dict[str, Any]] = None
    model_args: Optional[Dict[str, Any]] = None
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_dir: str = "./checkpoints"
    save_every: int = 0

    # Loss
    loss: str = "mse"
    fft_weight: float = 0.0  # spectral L1 on |FFT| of observations

    # Observation model
    nonlinear_obs: bool = False  # use learned h(x) instead of fixed H

    # Dataset split ratios
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Scheduler
    scheduler: Optional[str] = "none"  # none | cosine | step | plateau
    scheduler_args: Optional[Dict[str, Any]] = None

    # Early stopping
    early_stop: bool = True
    early_patience: int = 10
    early_min_delta: float = 0.0
    early_restore_best: bool = True

    # Visualization
    viz_samples: int = 0
    viz_dims: int = 3
    viz_set: str = "val"  # val | test

    # WandB configuration
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    # Logging
    quiet: bool = False


# ----------------------------
# Training utilities
# ----------------------------


def fft_loss(
    pred_obs: torch.Tensor,  # [B, T, obs_dim]
    target_obs: torch.Tensor,  # [B, T, obs_dim]
    time_dim: int = 1,
) -> torch.Tensor:
    """
    Spectral reconstruction loss (Eq. 13 in paper):
        L_Φ = || |F(ŷ)| - |F(ỹ)| ||_1

    Phase-invariant L1 loss on magnitude spectra of predicted vs noiseless
    observations. Inputs should already be in observation space (H @ states).
    """
    F_pred = torch.fft.rfft(pred_obs, dim=time_dim)
    F_target = torch.fft.rfft(target_obs, dim=time_dim)

    mag_pred = F_pred.abs()
    mag_target = F_target.abs()

    return F.l1_loss(mag_pred, mag_target)


def num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    var_target = torch.var(target)
    return torch.sqrt(mse / (var_target + 1e-8))


def train_one_epoch(
    model: KalmanFilterType,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CriterionType,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_nrmse = 0.0
    total_mse = 0.0
    total_grad_norm = 0.0
    count = 0

    for batch in dataloader:
        obs = batch["observations"].to(device)
        target_states = batch["states"].to(device)
        controls = batch["controls"].to(device) if "controls" in batch else None

        optimizer.zero_grad()
        preds = model(obs, controls=controls, initial_state=target_states[:, 0, :])
        loss = criterion(preds["states"], target_states)
        loss.backward()

        # Compute gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip > 0 else float("inf"))
        total_grad_norm += float(grad_norm)

        optimizer.step()

        total_loss += float(loss.item())
        total_nrmse += nrmse(preds["states"], target_states)
        total_mse += mse_loss(preds["states"], target_states)
        count += 1

    if count == 0:
        return {"loss": float("nan"), "nrmse": float("nan"), "mse": float("nan"), "grad_norm": float("nan")}
    return {
        "loss": float(total_loss) / count,
        "nrmse": float(total_nrmse) / count,
        "mse": float(total_mse) / count,
        "grad_norm": float(total_grad_norm) / count,
    }


@torch.no_grad()
def evaluate(
    model: KalmanFilterType,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Returns a dict with aggregate metrics over the whole loader:
      - MSE: mean squared error
      - RMSE: root mean squared error
      - MAE: mean absolute error
      - NRMSE: RMSE normalized by the global std of targets
      - R2: coefficient of determination (because why not)

    Shapes can be [B, T, D] or whatever; we flatten everything.
    NaNs/Infs are ignored.
    """
    model.eval()

    sse = 0.0  # sum of squared errors
    sae = 0.0  # sum of absolute errors
    n = 0  # number of scalar elements compared

    sum_y = 0.0  # for global mean/std of targets
    sum_y2 = 0.0

    for batch in dataloader:
        obs = batch["observations"].to(device)
        y = batch["states"].to(device)
        controls = batch["controls"].to(device) if "controls" in batch else None

        result = model(obs, controls=controls, initial_state=y[:, 0, :])
        y_hat = result["states"]

        err = y_hat - y

        # mask out non-finite values if they appear
        mask = torch.isfinite(err) & torch.isfinite(y)
        if not torch.all(mask):
            err = err[mask]
            y = y[mask]

        sse += float((err**2).sum().item())
        sae += float(err.abs().sum().item())
        n += int(err.numel())

        sum_y += float(y.sum().item())
        sum_y2 += float((y**2).sum().item())

    if n == 0:
        # nothing to evaluate; enjoy your NaNs
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "nrmse": float("nan"),
            "r2": float("nan"),
        }

    mse = sse / n
    rmse = math.sqrt(mse)
    mae = sae / n

    mean_y = sum_y / n
    var_y = max(sum_y2 / n - mean_y**2, 0.0)
    std_y = math.sqrt(var_y) if var_y > 0 else 0.0

    nrmse = rmse / (std_y + eps)
    r2 = 1.0 - (sse / (n * var_y + eps)) if std_y > 0 else float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,  # normalized by global std of targets
        "r2": r2,
    }


def maybe_visualize(
    cfg: "TrainConfig",
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    save_dir: str,
    split: str,
):
    if cfg.viz_samples <= 0:
        return []

    os.makedirs(os.path.join(save_dir, "viz"), exist_ok=True)
    viz_files = []
    remaining_samples = cfg.viz_samples

    model = model.to(cfg.device).eval()

    for index, batch in enumerate(dataloader):
        obs = batch["observations"].to(cfg.device)
        target_states = batch["states"].to(cfg.device)
        controls = batch["controls"].to(cfg.device) if "controls" in batch else None

        with torch.no_grad():
            preds = model(
                obs, controls=controls, initial_state=target_states[:, 0, :]
            )["states"].to("cpu")

        target_states_cpu = target_states.to("cpu")
        B, T, D = target_states_cpu.shape
        num = min(remaining_samples, B)
        dims = min(cfg.viz_dims, D)
        remaining_samples -= num

        # save data for further analysis
        out_pt = os.path.join(save_dir, "viz", f"{split}_batch{index}.pt")
        torch.save(
            {"preds": preds, "target_states": target_states_cpu, "obs": obs.to("cpu")},
            out_pt,
        )

        t = np.arange(T)

        for i in range(num):
            fig = make_subplots(
                rows=dims, cols=1, shared_xaxes=True, vertical_spacing=0.02
            )

            for d in range(dims):
                y_true = target_states_cpu[i, :, d].numpy()
                y_pred = preds[i, :, d].numpy()

                fig.add_trace(
                    go.Scatter(x=t, y=y_true, mode="lines", name=f"true {d}"),
                    row=d + 1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(x=t, y=y_pred, mode="lines", name=f"filt {d}"),
                    row=d + 1,
                    col=1,
                )
                fig.update_yaxes(title_text=f"dim {d}", row=d + 1, col=1)

            fig.update_xaxes(title_text="t", row=dims, col=1)
            fig.update_layout(
                title=f"{cfg.dataset} | {cfg.model} | split={split} | batch={index}, sample={i} | T={T}, dims_shown={dims}/{D}",
                height=220 * dims + 80,
                width=1000,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                margin=dict(l=50, r=20, t=60, b=40),
            )

            base = f"{split}_sample{i}_dim{dims}"

            # # Prefer PNG to preserve your previous interface if kaleido is installed.
            # png_path = os.path.join(save_dir, "viz", base + ".png")
            # fig.write_image(png_path, scale=2)
            # viz_files.append(png_path)

            # Fallback to interactive HTML without extra deps.
            html_path = os.path.join(save_dir, "viz", base + ".html")
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
            viz_files.append(html_path)

        if remaining_samples <= 0:
            break

    return viz_files


def save_metrics(
    save_dir: str, model: str, dataset: str, split: str, overall: Dict[str, float]
):
    ensure_dir(save_dir)
    payload = {"overall": overall}
    path = os.path.join(save_dir, f"{model}_{dataset}_{split}_metrics.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    path = os.path.join(save_dir, f"{model}_{dataset}_{split}_metrics.yaml")
    with open(path, "w") as f:
        yaml.dump(payload, f, indent=2)


def save_configs(save_dir: str, args_ns: argparse.Namespace, cfg: TrainConfig):
    ensure_dir(save_dir)
    # Raw argparse vars
    with open(os.path.join(save_dir, "run_args.json"), "w") as f:
        json.dump(vars(args_ns), f, indent=2)
    # Normalized TrainConfig
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)


# ----------------------------
# Main training loop
# ----------------------------


def main():
    args = parse_args()

    # Build config
    cfg = TrainConfig(
        dataset=args.dataset,
        model=args.model,
        dataset_args=args.dataset_args,
        model_args=args.model_args,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fft_weight=args.fft_weight,
        nonlinear_obs=args.nonlinear_obs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        save_every=args.save_every,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        scheduler=args.scheduler,
        scheduler_args=args.scheduler_args,
        early_stop=args.early_stop,
        early_patience=args.early_patience,
        early_min_delta=args.early_min_delta,
        early_restore_best=args.early_restore_best,
        viz_samples=args.viz_samples,
        viz_dims=args.viz_dims,
        viz_set=args.viz_set,
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        quiet=args.quiet,
    )

    # Setup WandB handler
    wandb_handler = None
    if cfg.wandb_enabled:
        wandb_kwargs = {}
        if cfg.wandb_project:
            wandb_kwargs["project"] = cfg.wandb_project
        if cfg.wandb_entity:
            wandb_kwargs["entity"] = cfg.wandb_entity
        if cfg.wandb_name:
            wandb_kwargs["name"] = cfg.wandb_name
        if cfg.wandb_tags:
            wandb_kwargs["tags"] = cfg.wandb_tags

        wandb_handler = WandBHandler(enabled=True, **wandb_kwargs)
        wandb_handler.log_config(asdict(cfg))

    # Setup logger
    ensure_dir(cfg.save_dir)
    logger = setup_logger(cfg.save_dir, wandb_handler, cfg.quiet)

    # Save configs
    save_configs(cfg.save_dir, args, cfg)

    # Deterministic seeding BEFORE model creation
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Dataset
    full_ds = build_dataset(cfg.dataset, cfg)
    train_ds, val_ds, test_ds = split_dataset_three(
        full_ds, cfg.val_ratio, cfg.test_ratio, cfg.seed
    )

    # DataLoaders with seeded generator for deterministic shuffle
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        generator=g,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # Model dimensions
    state_dim, obs_dim, control_dim = get_dims_from_dataset(full_ds)

    # Model
    model = build_model(
        cfg.model, state_dim, obs_dim, control_dim, cfg,
        nonlinear_obs=cfg.nonlinear_obs,
    )
    model.to(cfg.device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Scheduler
    scheduler = build_scheduler(optimizer, cfg)

    # Info logs
    logger.info(f"Config: {asdict(cfg)}")
    logger.info(
        f"State dim: {state_dim} | Obs dim: {obs_dim} | Control dim: {control_dim} | Params: {num_params(model)} | "
        f"Train size: {len(train_ds)} | Val size: {len(val_ds)} | Test size: {len(test_ds)}"
    )

    # Early stopping state
    best_val = math.inf
    best_state = None
    epochs_bad = 0

    # Observation projection for spectral loss (project states -> obs space)
    h_net = getattr(model, "h_net", None)  # learned nonlinear obs model
    H = getattr(model, "H", None)  # [obs_dim, state_dim] linear buffer

    def criterion(preds, target_states):
        mse = mse_loss(preds, target_states)
        if cfg.fft_weight > 0 and (h_net is not None or H is not None):
            if h_net is not None:
                # Nonlinear: project through learned observation network
                B, T, S = preds.shape
                pred_obs = h_net(preds.reshape(B * T, S)).reshape(B, T, -1)
                target_obs = h_net(
                    target_states.reshape(B * T, S)
                ).reshape(B, T, -1)
            else:
                # Linear: H @ states
                pred_obs = torch.einsum("os,bts->bto", H, preds)
                target_obs = torch.einsum("os,bts->bto", H, target_states)
            fft = fft_loss(pred_obs, target_obs)
            return mse + cfg.fft_weight * fft
        return mse

    # Training loop
    t_start_total = time.time()
    nan_epochs = 0
    for epoch in range(1, cfg.epochs + 1):
        t_epoch = time.time()
        train_overall = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            torch.device(cfg.device),
            cfg.grad_clip,
        )
        train_loss = train_overall["loss"]
        train_n = train_overall["nrmse"]
        grad_norm = train_overall["grad_norm"]

        val_result = evaluate(model, val_loader, torch.device(cfg.device))
        val_loss = val_result["mse"]
        val_n = val_result["nrmse"]
        epoch_sec = time.time() - t_epoch

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, lrs.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Manual log additional metrics to wandb (learning rate)
        if wandb_handler:
            wandb_handler.log_metrics(
                {"lr": optimizer.param_groups[0]["lr"], "grad_norm": grad_norm}, step=epoch
            )

        # Epoch line (automatically captured by WandBHandler regex)
        logger.info(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} train_nrmse={train_n:.6f} | "
            f"val_loss={val_loss:.6f} val_nrmse={val_n:.6f} | "
            f"grad_norm={grad_norm:.4f} lr={optimizer.param_groups[0]['lr']:.3e} ({epoch_sec:.1f}s)"
        )

        # NaN early abort: stop if loss is NaN for 3 consecutive epochs
        if math.isnan(train_loss) or math.isnan(val_loss):
            nan_epochs += 1
            if nan_epochs >= 3:
                logger.info(
                    f"Aborting: NaN loss for {nan_epochs} consecutive epochs. "
                    f"Model has diverged."
                )
                break
        else:
            nan_epochs = 0

        # Save every N
        if cfg.save_every and (epoch % cfg.save_every == 0):
            out_dir = os.path.join(
                cfg.save_dir, f"{cfg.model}_{cfg.dataset}_epoch{epoch:03d}"
            )
            ensure_dir(out_dir)
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

        # Early stopping check
        improved = (best_val - val_loss) > max(cfg.early_min_delta, 0.0)
        if improved:
            best_val = val_loss
            epochs_bad = 0
            if cfg.early_restore_best:
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            # Always save a "best" checkpoint for convenience
            best_dir = os.path.join(cfg.save_dir, f"{cfg.model}_{cfg.dataset}_best")
            ensure_dir(best_dir)
            torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
        else:
            epochs_bad += 1

        if cfg.early_stop and epochs_bad >= cfg.early_patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch} "
                f"(no improvement for {epochs_bad} epochs)."
            )
            break

    # Optionally restore the best weights
    if cfg.early_restore_best and best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - t_start_total
    logger.info(f"Training completed in {total_time:.1f}s ({epoch} epochs)")

    # Final metrics
    overall_val = evaluate(model, val_loader, torch.device(cfg.device))

    save_metrics(cfg.save_dir, cfg.model, cfg.dataset, "val", overall_val)
    logger.info(
        f"Validation | MSE={overall_val['mse']:.6f} RMSE={overall_val['rmse']:.6f} "
        f"MAE={overall_val['mae']:.6f} NRMSE={overall_val['nrmse']:.6f} R2={overall_val['r2']:.6f}"
    )

    overall_test = evaluate(model, test_loader, torch.device(cfg.device))
    save_metrics(cfg.save_dir, cfg.model, cfg.dataset, "test", overall_test)
    logger.info(
        f"Test       | MSE={overall_test['mse']:.6f} RMSE={overall_test['rmse']:.6f} "
        f"MAE={overall_test['mae']:.6f} NRMSE={overall_test['nrmse']:.6f} R2={overall_test['r2']:.6f}"
    )

    # Log final metrics to wandb
    if wandb_handler:
        metrics = {"final/val" + k.lower(): v for k, v in overall_val.items()}
        metrics.update({"final/test" + k.lower(): v for k, v in overall_test.items()})
        wandb_handler.log_metrics(metrics)

    # Visualizations
    viz_files = []
    if cfg.viz_set == "val":
        viz_files = maybe_visualize(cfg, val_loader, model, cfg.save_dir, "val")
    else:
        viz_files = maybe_visualize(cfg, test_loader, model, cfg.save_dir, "test")

    # Log visualization artifacts to wandb
    if wandb_handler and viz_files:
        for viz_file in viz_files:
            viz_path = Path(viz_file)
            wandb_handler.log_artifact(
                str(viz_file),
                name=f"visualization_{viz_path.stem}",
                artifact_type="visualization",
            )

    # Finish wandb run
    if wandb_handler:
        wandb_handler.finish()


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kalman network models")

    # Core arguments
    parser.add_argument("--dataset", default="lorenz", help="Dataset name")
    parser.add_argument("--model", default="knet", help="Model name")
    parser.add_argument("--dataset-args", type=json.loads, help="Dataset args as JSON")
    parser.add_argument("--model-args", type=json.loads, help="Model args as JSON")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)

    # Loss
    parser.add_argument("--fft-weight", type=float, default=0.0)
    parser.add_argument(
        "--nonlinear-obs", action="store_true",
        help="Use learned observation network h(x) instead of fixed linear H",
    )

    # I/O
    parser.add_argument("--save-dir", default="./checkpoints/latest")
    parser.add_argument("--save-every", type=int, default=0)

    # Dataset splits
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    # Scheduler
    parser.add_argument(
        "--scheduler", default="none", choices=["none", "cosine", "step", "plateau"]
    )
    parser.add_argument(
        "--scheduler-args", type=json.loads, help="Scheduler args as JSON"
    )

    # Early stopping
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-patience", type=int, default=10)
    parser.add_argument("--early-min-delta", type=float, default=0.0)
    parser.add_argument("--early-restore-best", action="store_true")

    # Visualization
    parser.add_argument("--viz-samples", type=int, default=0)
    parser.add_argument("--viz-dims", type=int, default=3)
    parser.add_argument("--viz-set", default="val", choices=["val", "test"])

    # WandB arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        dest="wandb_enabled",
        help="Enable WandB logging",
    )
    parser.add_argument(
        "--wandb-project", default="kalman-filter", help="WandB project name"
    )
    parser.add_argument("--wandb-entity", help="WandB entity/team name")
    parser.add_argument("--wandb-name", help="WandB run name")
    parser.add_argument("--wandb-tags", nargs="+", help="WandB tags")

    # Logging
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    return parser.parse_args()


if __name__ == "__main__":
    main()
