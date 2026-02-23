"""
Model registry and selection — selects the best model from training results
and provides a lightweight local registry that complements MLflow.
"""

import json
import mlflow
from pathlib import Path

from src.config.settings import PROJECT_ROOT, get_mlflow_config, get_paths_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def select_best_model(results: list[dict], metric: str = "f1_score") -> dict:
    """Select the best model from training results.

    Args:
        results: List of dicts with model_type, metrics, model.
        metric: Metric to optimize (default: f1_score).

    Returns:
        The best result dict.

    """
    if not results:
        raise ValueError("No results to compare")

    best = max(results, key=lambda r: r["metrics"][metric])
    logger.info(
        f"Best model: {best['model_type']} "
        f"with {metric}={best['metrics'][metric]:.4f}"
    )
    return best


def get_model_leaderboard(metric: str = "f1_score") -> list[dict]:
    """Query MLflow for all runs and create a leaderboard.

    Returns:
        Sorted list of model results from MLflow.
    """
    mlflow_config = get_mlflow_config()
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

    experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])
    if experiment is None:
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
    )

    if runs.empty:
        return []

    leaderboard: list[dict] = []
    for _, run in runs.iterrows():
        entry = {
            "run_id": run["run_id"],
            "model_type": run.get("tags.model_type", "unknown"),
            "accuracy": run.get("metrics.accuracy", 0),
            "precision": run.get("metrics.precision", 0),
            "recall": run.get("metrics.recall", 0),
            "f1_score": run.get("metrics.f1_score", 0),
            "roc_auc": run.get("metrics.roc_auc", 0),
        }
        leaderboard.append(entry)

    return leaderboard


def save_metrics(metrics: dict, filepath: str | Path | None = None) -> Path:
    """Save metrics to JSON file (for DVC tracking)."""
    if filepath is None:
        paths_config = get_paths_config()
        filepath = PROJECT_ROOT / paths_config["models_dir"] / "metrics.json"

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {filepath}")
    return filepath
