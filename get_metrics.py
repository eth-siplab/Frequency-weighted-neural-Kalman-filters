#!/usr/bin/env python3
"""
get_metrics.py

Retrieves all metrics from grid_search.sh outputs and converts them into CSV format.
Handles 3-seed experiments with mean/std aggregation.
Supports both JSON and YAML metrics files (requires PyYAML: pip install pyyaml).

Usage:
    # Basic metrics extraction
    python get_metrics.py --save-dir ./checkpoints/minisweep --output metrics_summary.csv
    python get_metrics.py --save-dir ./checkpoints/minisweep --split val --output val_metrics.csv
    
    # Find best hyperparameters for each (model, dataset, fft_weight) group
    python get_metrics.py --save-dir ./checkpoints/minisweep --best-configs
    python get_metrics.py --save-dir ./checkpoints/minisweep --best-configs --best-metric r2_mean --maximize
    python get_metrics.py --save-dir ./checkpoints/minisweep --best-configs --best-output best_hyperparams.csv
    
The script will warn about:
- Checkpoint folders with unparseable directory names
- Checkpoint folders missing metrics files entirely
- Individual metrics files that cannot be loaded

Best configuration features:
- Groups results by (model, dataset, fft_weight) combinations
- Finds optimal hyperparameters (lr, weight_decay, scheduler) for each group
- Auto-detects best metric to optimize (prefers NRMSE, then R2, then MSE/MAE)
- Outputs both console summary and optional CSV file
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


def parse_run_directory_name(dirname: str) -> Optional[Dict[str, str]]:
    """
    Parse grid search run directory name to extract hyperparameters.
    Expected format: {model}_{dataset}_fft{fft}_lr{lr}_wd{wd}_{scheduler}_seed{seed}

    Args:
        dirname: Directory name from grid search

    Returns:
        Dict with parsed parameters or None if parsing fails
    """
    # Known model and dataset names from grid_search.sh
    KNOWN_MODELS = [
        "kalman_net",
        "classical_kf",
        "deep_kf",
        "bayesian_kf",
        "autoreg_kf",
        "recurrent_kf",
        "recursive_kf",
        "recursive_knet",
        "recurrent_kalman_network",  # Common variations
    ]
    KNOWN_DATASETS = ["lorenz", "pendulum"]

    # First try: Work backwards from suffix pattern
    suffix_pattern = r"_fft([0-9.]+)_lr([0-9e.-]+)_wd([0-9e.-]+)_(.+?)_seed(\d+)$"
    suffix_match = re.search(suffix_pattern, dirname)

    if not suffix_match:
        return None

    fft, lr, wd, scheduler, seed = suffix_match.groups()

    # Extract the model_dataset prefix
    prefix = dirname[: suffix_match.start()]

    # Try to split prefix using known model/dataset names
    model, dataset = None, None

    # Try each known model to see if it's a prefix
    for known_model in KNOWN_MODELS:
        if prefix.startswith(known_model + "_"):
            candidate_dataset = prefix[len(known_model) + 1 :]
            if candidate_dataset in KNOWN_DATASETS:
                model = known_model
                dataset = candidate_dataset
                break

    if model is None or dataset is None:
        # Fallback: Try simple underscore split and validate
        parts = prefix.split("_")
        if len(parts) >= 2:
            # Try different split points
            for i in range(1, len(parts)):
                candidate_model = "_".join(parts[:i])
                candidate_dataset = "_".join(parts[i:])

                if candidate_dataset in KNOWN_DATASETS:
                    model = candidate_model
                    dataset = candidate_dataset
                    break

    if model is None or dataset is None:
        return None

    return {
        "model": model,
        "dataset": dataset,
        "fft_weight": float(fft),
        "lr": float(lr),
        "weight_decay": float(wd),
        "scheduler": scheduler,
        "seed": int(seed),
    }


def load_metrics_from_file(metrics_path: Path) -> Optional[Dict[str, float]]:
    """
    Load metrics from JSON or YAML file.

    Args:
        metrics_path: Path to metrics file (JSON or YAML)

    Returns:
        Dictionary of metrics or None if loading fails
    """
    try:
        with open(metrics_path, "r") as f:
            if metrics_path.suffix.lower() == ".json":
                data = json.load(f)
            elif metrics_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                return None
        return data.get("overall", {})
    except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError, KeyError):
        return None


def collect_all_metrics(save_dir: str, split: str = "val") -> List[Dict]:
    """
    Collect all metrics from grid search output directory.

    Args:
        save_dir: Root directory containing grid search outputs
        split: Which split to collect ('val' or 'test')

    Returns:
        List of dictionaries containing hyperparameters and metrics
    """
    save_path = Path(save_dir)
    if not save_path.exists():
        raise FileNotFoundError(f"Save directory does not exist: {save_dir}")

    results = []
    no_metrics_count = 0

    # Iterate through all subdirectories
    for run_dir in save_path.iterdir():
        if not run_dir.is_dir():
            continue

        # Parse hyperparameters from directory name
        params = parse_run_directory_name(run_dir.name)
        if params is None:
            print(f"Warning: Could not parse directory name: {run_dir.name}")
            continue

        # Look for metrics files (JSON and YAML)
        model = params["model"]
        dataset = params["dataset"]
        json_file = run_dir / f"{model}_{dataset}_{split}_metrics.json"
        yaml_file = run_dir / f"{model}_{dataset}_{split}_metrics.yaml"

        # Check which files exist
        json_exists = json_file.exists()
        yaml_exists = yaml_file.exists()

        if not json_exists and not yaml_exists:
            print(
                f"Warning: No {split} metrics files found in checkpoint folder: {run_dir.name}"
            )
            no_metrics_count += 1
            continue

        # Prefer JSON over YAML if both exist
        metrics_file = json_file if json_exists else yaml_file
        file_format = "JSON" if json_exists else "YAML"

        # Load metrics
        metrics = load_metrics_from_file(metrics_file)
        if metrics is None:
            print(f"Warning: Could not load {file_format} metrics from: {metrics_file}")
            continue

        # Combine parameters and metrics
        result = {**params, **metrics}
        results.append(result)

    if no_metrics_count > 0:
        print(
            f"Warning: {no_metrics_count} checkpoint folders had no {split} metrics files"
        )

    return results


def aggregate_by_hyperparams(results: List[Dict]) -> List[Dict]:
    """
    Aggregate metrics across seeds for each unique hyperparameter combination.

    Args:
        results: List of individual run results

    Returns:
        List of aggregated results with mean/std across seeds
    """
    # Group by hyperparameters (excluding seed)
    grouped = defaultdict(list)

    for result in results:
        # Create key from hyperparameters (exclude seed and metrics)
        key_params = {
            k: v
            for k, v in result.items()
            if k not in ["seed"]
            and not isinstance(v, (int, float))
            or k in ["model", "dataset", "scheduler"]
        }
        key_params.update(
            {
                k: v
                for k, v in result.items()
                if k in ["fft_weight", "lr", "weight_decay"]
            }
        )

        # Convert to hashable key
        key = tuple(sorted(key_params.items()))
        grouped[key].append(result)

    aggregated = []

    for key, group in grouped.items():
        if not group:
            continue

        # Get hyperparameters from first result
        base_result = group[0].copy()

        # Remove seed from base result
        base_result.pop("seed", None)

        # Get all metric names (exclude hyperparameters)
        metric_names = set()
        for result in group:
            metric_names.update(
                k
                for k, v in result.items()
                if isinstance(v, (int, float))
                and k not in ["seed", "fft_weight", "lr", "weight_decay"]
            )

        # Aggregate metrics across seeds
        for metric_name in metric_names:
            values = [
                result.get(metric_name)
                for result in group
                if result.get(metric_name) is not None
            ]

            if not values:
                continue

            base_result[f"{metric_name}_mean"] = np.mean(values)
            base_result[f"{metric_name}_std"] = (
                np.std(values, ddof=1) if len(values) > 1 else 0.0
            )
            base_result[f"{metric_name}_count"] = len(values)

        # Add seed information
        seeds = [result["seed"] for result in group]
        base_result["seeds"] = ",".join(map(str, sorted(seeds)))
        base_result["num_seeds"] = len(seeds)

        aggregated.append(base_result)

    return aggregated


def find_best_hyperparams_per_group(
    results: List[Dict], metric: str = "nrmse_mean", minimize: bool = True
) -> List[Dict]:
    """
    Find the best hyperparameter combination for each (model, dataset, fft_weight) group.

    Args:
        results: List of aggregated results with mean/std metrics
        metric: Metric to optimize for (e.g., 'nrmse_mean', 'r2_mean')
        minimize: Whether to minimize (True) or maximize (False) the metric

    Returns:
        List of best configurations per group
    """
    # Group by (model, dataset, fft_weight)
    grouped = defaultdict(list)

    for result in results:
        key = (result.get("model"), result.get("dataset"), result.get("fft_weight"))
        grouped[key].append(result)

    best_configs = []

    for (model, dataset, fft_weight), group in grouped.items():
        if not group:
            continue

        # Filter results that have the target metric
        valid_results = [r for r in group if metric in r and r[metric] is not None]

        if not valid_results:
            print(
                f"Warning: No valid {metric} values for {model}/{dataset}/fft{fft_weight}"
            )
            continue

        # Find best result
        if minimize:
            best_result = min(valid_results, key=lambda x: x[metric])
        else:
            best_result = max(valid_results, key=lambda x: x[metric])

        # Add group identifier
        best_result_copy = best_result.copy()
        best_result_copy["group_key"] = f"{model}_{dataset}_fft{fft_weight}"
        best_result_copy["best_metric"] = metric
        best_result_copy["best_value"] = best_result[metric]

        best_configs.append(best_result_copy)

    return best_configs


def print_best_hyperparams_summary(best_configs: List[Dict], metric: str):
    """
    Print a summary of best hyperparameters for each group.

    Args:
        best_configs: List of best configurations per group
        metric: The metric that was optimized
    """
    if not best_configs:
        print("No best configurations found.")
        return

    print(f"\n=== BEST HYPERPARAMETERS PER GROUP (optimized for {metric}) ===")
    print(
        f"{'Group':<45} {'Best '+metric:<15} {'LR':<12} {'WD':<12} {'Scheduler':<12} {'Seeds':<8}"
    )
    print("-" * 108)

    # Sort by group name for consistent output
    sorted_configs = sorted(best_configs, key=lambda x: x.get("group_key", ""))

    for config in sorted_configs:
        group = config.get("group_key", "unknown")
        best_val = config.get("best_value", 0.0)
        lr = config.get("lr", 0.0)
        wd = config.get("weight_decay", 0.0)
        scheduler = config.get("scheduler", "none")
        seeds = config.get("num_seeds", 0)

        print(
            f"{group:<45} {best_val:<15.6f} {lr:<12.2e} {wd:<12.2e} {scheduler:<12} {seeds:<8}"
        )

    print(f"\nFound best hyperparameters for {len(best_configs)} groups")
    print("Use --best-output <filename> to save detailed results to CSV")


def save_best_configs_to_csv(best_configs: List[Dict], output_path: str):
    """
    Save best configurations to a separate CSV file.

    Args:
        best_configs: List of best configurations
        output_path: Path to output CSV file
    """
    if not best_configs:
        print("No best configurations to save.")
        return

    # Get all column names
    all_columns = set()
    for config in best_configs:
        all_columns.update(config.keys())

    # Sort columns: group info first, then hyperparameters, then metrics
    priority_cols = [
        "group_key",
        "model",
        "dataset",
        "fft_weight",
        "lr",
        "weight_decay",
        "scheduler",
        "best_metric",
        "best_value",
        "num_seeds",
        "seeds",
    ]
    metric_cols = sorted([col for col in all_columns if col not in priority_cols])

    columns = [col for col in priority_cols if col in all_columns] + metric_cols

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        # Sort by group key
        sorted_configs = sorted(best_configs, key=lambda x: x.get("group_key", ""))
        writer.writerows(sorted_configs)

    print(f"Saved {len(best_configs)} best configurations to {output_path}")


def detect_best_metric(results: List[Dict]) -> Tuple[str, bool]:
    """
    Auto-detect the best metric to optimize based on available metrics.

    Args:
        results: List of results to analyze

    Returns:
        Tuple of (metric_name, minimize_flag)
    """
    if not results:
        return "nrmse_mean", True

    sample_result = results[0]
    available_metrics = [k for k in sample_result.keys() if k.endswith("_mean")]

    # Priority order: prefer NRMSE (lower is better), then R2 (higher is better), then MSE, then MAE
    metric_preferences = [
        ("nrmse_mean", True),
        ("r2_mean", False),
        ("mse_mean", True),
        ("mae_mean", True),
        ("rmse_mean", True),
    ]

    for metric, minimize in metric_preferences:
        if metric in available_metrics:
            return metric, minimize

    # Fallback to first available metric (assume minimize)
    if available_metrics:
        return available_metrics[0], True

    return "nrmse_mean", True


def save_to_csv(results: List[Dict], output_path: str):
    """
    Save results to CSV file.

    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
    """
    if not results:
        print("No results to save.")
        return

    # Get all column names
    all_columns = set()
    for result in results:
        all_columns.update(result.keys())

    # Sort columns: hyperparameters first, then metrics
    hyperparam_cols = [
        "model",
        "dataset",
        "fft_weight",
        "lr",
        "weight_decay",
        "scheduler",
        "seeds",
        "num_seeds",
    ]
    metric_cols = sorted([col for col in all_columns if col not in hyperparam_cols])

    columns = [col for col in hyperparam_cols if col in all_columns] + metric_cols

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        # Sort results by model, dataset, then other hyperparameters
        sorted_results = sorted(
            results,
            key=lambda x: (
                x.get("model", ""),
                x.get("dataset", ""),
                x.get("fft_weight", 0),
                x.get("lr", 0),
                x.get("weight_decay", 0),
                x.get("scheduler", ""),
            ),
        )

        writer.writerows(sorted_results)

    print(f"Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract metrics from grid search results"
    )
    parser.add_argument(
        "--save-dir",
        default="./checkpoints/minisweep",
        help="Root directory containing grid search outputs",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Which split to extract metrics for",
    )
    parser.add_argument(
        "--output", default="metrics_summary.csv", help="Output CSV file path"
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Do not aggregate across seeds, output individual runs",
    )
    parser.add_argument(
        "--best-configs",
        action="store_true",
        help="Find and display best hyperparameters for each (model, dataset, fft_weight) group",
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        default="auto",
        help="Metric to optimize for best configs (default: auto-detect)",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Maximize the metric instead of minimizing (use with --best-metric)",
    )
    parser.add_argument(
        "--best-output",
        type=str,
        default=None,
        help="Save best configurations to separate CSV file",
    )

    args = parser.parse_args()

    print(f"Collecting {args.split} metrics from: {args.save_dir}")

    # Count total checkpoint directories for comparison
    save_path = Path(args.save_dir)
    total_dirs = (
        len([d for d in save_path.iterdir() if d.is_dir()]) if save_path.exists() else 0
    )

    # Collect all metrics
    results = collect_all_metrics(args.save_dir, args.split)

    if not results:
        print("No results found!")
        return

    print(
        f"Found {len(results)} individual runs out of {total_dirs} checkpoint directories"
    )

    # Aggregate across seeds unless disabled
    if not args.no_aggregate:
        results = aggregate_by_hyperparams(results)
        print(f"Aggregated to {len(results)} unique hyperparameter combinations")

    # Save to CSV
    save_to_csv(results, args.output)

    # Find best hyperparameters if requested
    if args.best_configs and not args.no_aggregate:
        # Determine metric and optimization direction
        if args.best_metric == "auto":
            metric, minimize = detect_best_metric(results)
            print(
                f"Auto-detected optimization metric: {metric} ({'minimize' if minimize else 'maximize'})"
            )
        else:
            metric = args.best_metric
            minimize = not args.maximize
            print(
                f"Using specified metric: {metric} ({'minimize' if minimize else 'maximize'})"
            )

        # Find best configurations
        best_configs = find_best_hyperparams_per_group(results, metric, minimize)

        if best_configs:
            print_best_hyperparams_summary(best_configs, metric)

            # Save to separate file if requested
            if args.best_output:
                save_best_configs_to_csv(best_configs, args.best_output)
            else:
                # Generate default filename
                base_name = Path(args.output).stem
                best_output = f"{base_name}_best_configs.csv"
                save_best_configs_to_csv(best_configs, best_output)
        else:
            print("No best configurations found (possibly due to missing metrics)")

    elif args.best_configs and args.no_aggregate:
        print(
            "Warning: --best-configs requires aggregated data. Remove --no-aggregate to use this feature."
        )

    # Print summary statistics
    if results:
        print(f"\nSummary:")
        models = set(r.get("model") for r in results)
        datasets = set(r.get("dataset") for r in results)
        print(f"Models: {sorted(models)}")
        print(f"Datasets: {sorted(datasets)}")

        # Show sample metrics
        sample_result = results[0]
        metric_names = [
            k
            for k in sample_result.keys()
            if k.endswith("_mean")
            or (
                not args.no_aggregate
                and isinstance(sample_result[k], (int, float))
                and k not in ["fft_weight", "lr", "weight_decay", "num_seeds"]
            )
        ]
        if metric_names:
            print(f"Available metrics: {sorted(metric_names)}")

    # Final warning summary
    missing_count = total_dirs - len(results)
    if missing_count > 0:
        print(
            f"\nWARNING: {missing_count} checkpoint directories were skipped due to missing or invalid metrics files"
        )
        print("Check the warnings above for details on which directories had issues.")


if __name__ == "__main__":
    main()
