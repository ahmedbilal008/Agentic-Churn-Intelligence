"""
Model training with MLflow tracking.
Trains Logistic Regression, Random Forest, and XGBoost; logs params, metrics,
artifacts, and plots to MLflow for each run.
"""

import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — safe for servers
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
)

from src.config.settings import (
    PROJECT_ROOT,
    get_model_config,
    get_mlflow_config,
    get_paths_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model Factory — config-driven model creation
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}


def create_model(
    model_type: str, hyperparameters: dict | None = None
) -> LogisticRegression | RandomForestClassifier | XGBClassifier:
    """Create a model instance with given hyperparameters.

    Args:
        model_type: One of logistic_regression, random_forest, xgboost.
        hyperparameters: Override default hyperparameters from params.yaml.

    Returns:
        Configured model instance (unfitted).
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    # Start with config defaults, allow overrides
    config = get_model_config(model_type)
    if isinstance(config, dict):
        config = dict(config)  # copy
    if hyperparameters:
        config = {**config, **hyperparameters}

    model_class = MODEL_REGISTRY[model_type]
    model = model_class(**config)

    logger.info(f"Created {model_type} model with params: {config}")
    return model


def train_model(
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
) -> dict:
    """Train a model and return it with training info."""
    logger.info(f"Training {model_type} on {X_train.shape[0]} samples...")
    model.fit(X_train, y_train)
    logger.info(f"{model_type} training complete")

    return {"model": model, "model_type": model_type}


def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict:
    """Evaluate a trained model and return comprehensive metrics.

    Returns:
        Dictionary with all metrics and evaluation artifacts.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    confusion = {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}

    # Feature importance (model-dependent)
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])

    feature_importance: dict[str, float] = {}
    if importance is not None and feature_names is not None:
        feature_importance = dict(zip(feature_names, importance.tolist()))
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "predictions": y_pred,
        "probabilities": y_proba,
        "feature_importance": feature_importance,
    }


def log_to_mlflow(
    model: object,
    model_type: str,
    metrics: dict,
    confusion: dict,
    feature_importance: dict,
    y_test: np.ndarray,
    y_proba: np.ndarray,
    hyperparameters: dict,
    plots_dir: Path,
) -> str:
    """Log training run to MLflow.

    Returns:
        MLflow run ID.
    """
    mlflow_config = get_mlflow_config()
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=f"{model_type}_churn") as run:
        # Log hyperparameters
        for param_name, param_value in hyperparameters.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("model_type", model_type)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log confusion matrix values
        for key, val in confusion.items():
            mlflow.log_metric(f"confusion_{key}", val)

        # Log the sklearn-compatible model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Generate and log plots
        plots_dir.mkdir(parents=True, exist_ok=True)

        # --- Confusion Matrix Plot ---
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_array = np.array(
            [[confusion["TN"], confusion["FP"]], [confusion["FN"], confusion["TP"]]]
        )
        disp = ConfusionMatrixDisplay(cm_array, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {model_type}")
        cm_path = plots_dir / f"confusion_matrix_{model_type}.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {model_type}")
        ax.legend()
        roc_path = plots_dir / f"roc_curve_{model_type}.png"
        fig.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(roc_path))

        # --- Feature Importance Plot (top 15) ---
        if feature_importance:
            top_features = dict(list(feature_importance.items())[:15])
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(
                list(top_features.keys())[::-1],
                list(top_features.values())[::-1],
            )
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance — {model_type}")
            fi_path = plots_dir / f"feature_importance_{model_type}.png"
            fig.savefig(fi_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(fi_path))

        # Tags
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag(
            "framework", "sklearn" if model_type != "xgboost" else "xgboost"
        )

        run_id = run.info.run_id
        logger.info(f"MLflow run logged: {run_id} ({model_type})")

        return run_id


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------


def save_model(
    model: object, model_type: str, path: str | Path | None = None
) -> Path:
    """Save trained model to disk."""
    if path is None:
        paths_config = get_paths_config()
        path = PROJECT_ROOT / paths_config["models_dir"] / f"{model_type}.pkl"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path


def load_model(
    model_type: str | None = None, path: str | Path | None = None
) -> object:
    """Load a trained model from disk.

    Args:
        model_type: If provided, loads models/{model_type}.pkl.
        path: Direct path override.
    """
    if path is None:
        paths_config = get_paths_config()
        if model_type:
            path = PROJECT_ROOT / paths_config["models_dir"] / f"{model_type}.pkl"
        else:
            path = PROJECT_ROOT / paths_config["models_dir"] / "best_model.pkl"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model
