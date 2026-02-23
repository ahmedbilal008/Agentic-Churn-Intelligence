"""
MCP server — exposes 10 tools for AI agent access to the churn ML system.
Uses SSE transport (FastMCP). Business logic stays in core/ and services/.
"""

import json
import os
import sys
import threading
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP

# Ensure project root is in path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.config.settings import (
    PROJECT_ROOT,
    get_paths_config,
    get_data_config,
    get_feature_config,
    get_server_config,
)
from src.core.data.preprocessor import ChurnPreprocessor
from src.core.features.engineer import engineer_features
from src.core.models.trainer import load_model
from src.core.explainability.shap_explainer import ShapExplainer, load_background_data
from src.schemas.models import CustomerFeatures, CustomerRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Initialize MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="Churn Intelligence Platform",
    instructions=(
        "An ML-powered churn prediction system. You can predict customer churn, "
        "explain predictions with SHAP, compare models, retrain models, and "
        "inspect the dataset. Use predict_churn for individual predictions, "
        "explain_prediction for SHAP-based explanations, compare_models for "
        "the model leaderboard, and retrain_model to trigger retraining."
    ),
)


# ---------------------------------------------------------------------------
# Cached Model Manager
# ---------------------------------------------------------------------------


class _ModelCache:
    """Lazy-loading cache for models and preprocessors.

    Avoids reloading from disk on every tool call.
    Cache is cleared after retraining so the next call gets the new model.
    """

    _best_model = None
    _preprocessor = None
    _metadata = None
    _explainer = None

    @classmethod
    def get_model(cls):
        if cls._best_model is None:
            cls._best_model = load_model()
        return cls._best_model

    @classmethod
    def get_preprocessor(cls):
        if cls._preprocessor is None:
            cls._preprocessor = ChurnPreprocessor.load()
        return cls._preprocessor

    @classmethod
    def get_metadata(cls):
        if cls._metadata is None:
            paths = get_paths_config()
            meta_path = PROJECT_ROOT / paths["models_dir"] / "metadata.pkl"
            if meta_path.exists():
                cls._metadata = joblib.load(meta_path)
            else:
                cls._metadata = {
                    "feature_names": [],
                    "best_model_type": "unknown",
                }
        return cls._metadata

    @classmethod
    def get_explainer(cls):
        if cls._explainer is None:
            model = cls.get_model()
            metadata = cls.get_metadata()
            try:
                background = load_background_data()
            except FileNotFoundError:
                background = None
            cls._explainer = ShapExplainer(
                model, metadata["feature_names"], background
            )
        return cls._explainer

    @classmethod
    def reload(cls):
        """Clear cache to force reload (after retraining)."""
        cls._best_model = None
        cls._preprocessor = None
        cls._metadata = None
        cls._explainer = None


def _prepare_features(customer_data: dict) -> np.ndarray:
    """Transform raw customer data into model-ready features.

    This replicates the exact same preprocessing pipeline used during
    training, ensuring consistency between training and inference.
    """
    # Create single-row DataFrame
    df = pd.DataFrame([customer_data])

    # Apply same feature engineering
    df = engineer_features(df)

    # Apply saved preprocessor (same encoding + scaling as training)
    preprocessor = _ModelCache.get_preprocessor()
    X = preprocessor.transform(df)

    return X


# ---------------------------------------------------------------------------
# Dataset-derived defaults (mode for categoricals, median for numericals)
# Used when users provide only a few key features.
# ---------------------------------------------------------------------------

_DATASET_DEFAULTS: dict[str, str | int | float] = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 29,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 1397.47,
}


def _apply_defaults(kwargs: dict) -> dict:
    """Merge user-supplied features with dataset defaults."""
    merged = dict(_DATASET_DEFAULTS)
    # Override only the keys the user actually provided
    for k, v in kwargs.items():
        if v is not None:
            merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# Tool 1: Predict Churn
# ---------------------------------------------------------------------------


@mcp.tool()
def predict_churn(
    tenure: int | None = None,
    MonthlyCharges: float | None = None,
    Contract: str | None = None,
    InternetService: str | None = None,
    TotalCharges: float | None = None,
    gender: str | None = None,
    SeniorCitizen: int | None = None,
    Partner: str | None = None,
    Dependents: str | None = None,
    PhoneService: str | None = None,
    MultipleLines: str | None = None,
    OnlineSecurity: str | None = None,
    OnlineBackup: str | None = None,
    DeviceProtection: str | None = None,
    TechSupport: str | None = None,
    StreamingTV: str | None = None,
    StreamingMovies: str | None = None,
    PaperlessBilling: str | None = None,
    PaymentMethod: str | None = None,
) -> str:
    """Predict customer churn probability and risk level.

    All parameters are optional. Unspecified features use dataset defaults
    (most common value for categoricals, median for numericals).
    Key features: tenure, MonthlyCharges, Contract, InternetService.

    Returns churn probability, risk category (LOW/MEDIUM/HIGH),
    the top features driving the prediction, and the model used.
    """
    try:
        customer_data = _apply_defaults({
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        })

        # Validate input
        CustomerFeatures(**customer_data)

        # Prepare features and predict
        X = _prepare_features(customer_data)
        model = _ModelCache.get_model()

        proba = model.predict_proba(X)[0]
        churn_prob = float(proba[1])

        # Risk categorization
        if churn_prob >= 0.7:
            risk = "HIGH"
        elif churn_prob >= 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # Quick top drivers via SHAP
        top_drivers: list[dict] = []
        try:
            explainer = _ModelCache.get_explainer()
            explanation = explainer.explain_single(X)
            top_drivers = explanation["feature_contributions"][:5]
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

        metadata = _ModelCache.get_metadata()

        result = {
            "churn_probability": round(churn_prob, 4),
            "risk_level": risk,
            "top_drivers": top_drivers,
            "model_used": metadata.get("best_model_type", "unknown"),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 2: Explain Prediction
# ---------------------------------------------------------------------------


@mcp.tool()
def explain_prediction(
    tenure: int | None = None,
    MonthlyCharges: float | None = None,
    Contract: str | None = None,
    InternetService: str | None = None,
    TotalCharges: float | None = None,
    gender: str | None = None,
    SeniorCitizen: int | None = None,
    Partner: str | None = None,
    Dependents: str | None = None,
    PhoneService: str | None = None,
    MultipleLines: str | None = None,
    OnlineSecurity: str | None = None,
    OnlineBackup: str | None = None,
    DeviceProtection: str | None = None,
    TechSupport: str | None = None,
    StreamingTV: str | None = None,
    StreamingMovies: str | None = None,
    PaperlessBilling: str | None = None,
    PaymentMethod: str | None = None,
) -> str:
    """Get detailed SHAP explanation for a churn prediction.

    All parameters are optional. Unspecified features use dataset defaults.
    Returns each feature's SHAP contribution to the prediction,
    the base prediction value, and the top positive/negative drivers.
    Uses SHAP (SHapley Additive exPlanations) from game theory.
    """
    try:
        customer_data = _apply_defaults({
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        })

        X = _prepare_features(customer_data)
        explainer = _ModelCache.get_explainer()
        explanation = explainer.explain_single(X)

        return json.dumps(explanation, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 3: Model Metrics
# ---------------------------------------------------------------------------


@mcp.tool()
def get_model_metrics(model_name: str = "best") -> str:
    """Get performance metrics for a specific model.

    Args:
        model_name: Model to query — 'best', 'logistic_regression',
                    'random_forest', or 'xgboost'.

    Returns metrics including accuracy, precision, recall, F1,
    ROC-AUC, and confusion matrix summary.
    """
    try:
        paths = get_paths_config()
        metrics_path = (
            PROJECT_ROOT / paths["models_dir"] / "training_metrics.json"
        )

        if not metrics_path.exists():
            return json.dumps(
                {"error": "No training metrics found. Train models first."}
            )

        with open(metrics_path) as f:
            all_metrics = json.load(f)

        if model_name == "best":
            model_name = all_metrics["best_model"]

        if model_name not in all_metrics.get("models", {}):
            available = list(all_metrics.get("models", {}).keys())
            return json.dumps(
                {
                    "error": f"Model '{model_name}' not found. Available: {available}"
                }
            )

        metrics = all_metrics["models"][model_name]

        # Also load evaluation for confusion matrix
        eval_path = PROJECT_ROOT / paths["models_dir"] / "evaluation.json"
        confusion = {}
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
            if eval_data.get("best_model_type") == model_name:
                confusion = eval_data.get("confusion_matrix", {})

        result = {
            "model_name": model_name,
            "accuracy": metrics.get("accuracy", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1_score": metrics.get("f1_score", 0),
            "roc_auc": metrics.get("roc_auc", 0),
            "confusion_matrix": confusion,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 4: Compare Models
# ---------------------------------------------------------------------------


@mcp.tool()
def compare_models() -> str:
    """Compare all trained models and return a leaderboard.

    Shows performance metrics for each model type, ranked by F1 score.
    This demonstrates the value of MLflow experiment tracking.
    """
    try:
        paths = get_paths_config()
        metrics_path = (
            PROJECT_ROOT / paths["models_dir"] / "training_metrics.json"
        )

        if not metrics_path.exists():
            return json.dumps(
                {"error": "No training metrics found. Train models first."}
            )

        with open(metrics_path) as f:
            all_metrics = json.load(f)

        # Build leaderboard
        leaderboard = []
        for model_name, metrics in all_metrics.get("models", {}).items():
            leaderboard.append({"model": model_name, **metrics})

        # Sort by F1 score
        leaderboard.sort(key=lambda x: x.get("f1_score", 0), reverse=True)

        result = {
            "leaderboard": leaderboard,
            "best_model": all_metrics.get("best_model", "unknown"),
            "ranking_metric": "f1_score",
            "total_models": len(leaderboard),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 5: Dataset Summary
# ---------------------------------------------------------------------------


@mcp.tool()
def get_dataset_summary() -> str:
    """Get a comprehensive summary of the training dataset.

    Returns row count, feature count, missing values, class distribution,
    churn rate, and basic statistics. Useful for data understanding.
    """
    try:
        data_config = get_data_config()
        filepath = PROJECT_ROOT / data_config["raw_path"]

        if not filepath.exists():
            return json.dumps({"error": f"Dataset not found at {filepath}"})

        df = pd.read_csv(filepath)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Class distribution
        churn_counts = df["Churn"].value_counts().to_dict()
        total = len(df)
        churn_rate = churn_counts.get("Yes", 0) / total if total > 0 else 0

        # Missing values
        tc_empty = df["TotalCharges"].isna().sum()
        missing_dict: dict[str, int] = {}
        if tc_empty > 0:
            missing_dict["TotalCharges"] = int(tc_empty)

        # Feature types
        numerical = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical = df.select_dtypes(include=["object"]).columns.tolist()

        result = {
            "total_rows": total,
            "total_features": df.shape[1],
            "numerical_features": numerical,
            "categorical_features": categorical,
            "missing_values": missing_dict,
            "class_distribution": {
                str(k): int(v) for k, v in churn_counts.items()
            },
            "churn_rate": round(churn_rate, 4),
            "tenure_stats": {
                "mean": round(float(df["tenure"].mean()), 1),
                "median": round(float(df["tenure"].median()), 1),
                "max": int(df["tenure"].max()),
            },
            "monthly_charges_stats": {
                "mean": round(float(df["MonthlyCharges"].mean()), 2),
                "median": round(float(df["MonthlyCharges"].median()), 2),
                "max": round(float(df["MonthlyCharges"].max()), 2),
            },
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 6: Global Feature Importance
# ---------------------------------------------------------------------------


@mcp.tool()
def get_feature_importance(top_n: int = 10) -> str:
    """Get global SHAP feature importance rankings.

    Shows which features most influence churn predictions across all
    customers, computed as mean absolute SHAP values.

    Args:
        top_n: Number of top features to return (default 10).
    """
    try:
        explainer = _ModelCache.get_explainer()
        background = load_background_data()
        importance = explainer.global_importance(background)

        # Take top_n
        top = dict(list(importance.items())[:top_n])

        result = {
            "top_features": [
                {"feature": k, "importance": round(v, 4)}
                for k, v in top.items()
            ],
            "total_features": len(importance),
            "method": "mean_abs_shap",
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Retrain Lock — prevents concurrent retraining
# ---------------------------------------------------------------------------

_retrain_lock = threading.Lock()
_last_train_time: str | None = None


# ---------------------------------------------------------------------------
# Tool 6: Retrain Model (with safety lock)
# ---------------------------------------------------------------------------


@mcp.tool()
def retrain_model(
    model_type: str = "random_forest",
    hyperparameters: str = "{}",
) -> str:
    """Retrain a model with the full ML pipeline.

    Triggers: data loading → feature engineering → preprocessing →
    training → MLflow logging → evaluation → model saving.

    Includes concurrency protection — only one retrain can run at a time.
    If another retrain is in progress, returns immediately with status.

    Args:
        model_type: Which model to train — 'logistic_regression',
                    'random_forest', 'xgboost', or 'all' for all three.
        hyperparameters: JSON string of hyperparameter overrides, e.g.
                         '{"n_estimators": 300, "max_depth": 12}'

    Returns training results including metrics and MLflow run ID.
    WARNING: This operation takes time as it runs the full pipeline.
    """
    global _last_train_time

    # Attempt to acquire lock (non-blocking)
    if not _retrain_lock.acquire(blocking=False):
        return json.dumps(
            {
                "status": "rejected",
                "error": "A retraining job is already in progress. Please wait.",
            }
        )

    try:
        hp = (
            json.loads(hyperparameters)
            if hyperparameters and hyperparameters != "{}"
            else None
        )

        from src.services.pipelines.training_pipeline import (
            run_full_pipeline,
            run_single_model_training,
        )

        if model_type == "all":
            result = run_full_pipeline(hyperparameters=hp)
        else:
            result = run_single_model_training(model_type, hp)

        # Update last training timestamp
        _last_train_time = datetime.now(timezone.utc).isoformat()

        # Clear cached models so next prediction uses new model
        _ModelCache.reload()

        return json.dumps(
            {
                "status": "success",
                "model_type": result["training"]["best_model"],
                "metrics": result["training"]["metrics"],
                "mlflow_run_id": result["training"]["mlflow_run_id"],
                "trained_at": _last_train_time,
                "message": (
                    f"Model retrained successfully. "
                    f"New best: {result['training']['best_model']}"
                ),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

    finally:
        _retrain_lock.release()


# ---------------------------------------------------------------------------
# Tool 7: Add Customer Record
# ---------------------------------------------------------------------------


@mcp.tool()
def add_customer_record(
    customerID: str,
    gender: str,
    SeniorCitizen: int,
    Partner: str,
    Dependents: str,
    tenure: int,
    PhoneService: str,
    MultipleLines: str,
    InternetService: str,
    OnlineSecurity: str,
    OnlineBackup: str,
    DeviceProtection: str,
    TechSupport: str,
    StreamingTV: str,
    StreamingMovies: str,
    Contract: str,
    PaperlessBilling: str,
    PaymentMethod: str,
    MonthlyCharges: float,
    TotalCharges: float,
    Churn: str = "No",
) -> str:
    """Add a new customer record to the dataset.

    Appends a validated row to the raw data CSV. After adding data,
    call retrain_model to update the model with the new data.

    This demonstrates data lifecycle management — new data flows
    into the system and can trigger pipeline re-execution.
    """
    try:
        new_record = {
            "customerID": customerID,
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
            "Churn": Churn,
        }

        # Validate with Pydantic
        CustomerRecord(**new_record)

        # Append to raw data
        data_config = get_data_config()
        filepath = PROJECT_ROOT / data_config["raw_path"]

        df_new = pd.DataFrame([new_record])

        if filepath.exists():
            df_existing = pd.read_csv(filepath)
            # Check for duplicate customerID
            if customerID in df_existing["customerID"].values:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"Customer {customerID} already exists",
                    }
                )
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(filepath, index=False)

        return json.dumps(
            {
                "status": "success",
                "message": (
                    f"Customer {customerID} added. "
                    f"Dataset now has {len(df_combined)} records."
                ),
                "suggestion": "Call retrain_model to update the model with the new data.",
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ---------------------------------------------------------------------------
# Tool 8: Get Active Model Info
# ---------------------------------------------------------------------------


@mcp.tool()
def get_active_model_info() -> str:
    """Get detailed information about the currently active model.

    Returns:
        Model name, training timestamp, dataset size, metrics,
        feature count, and version identifiers.
    """
    try:
        paths = get_paths_config()
        models_dir = PROJECT_ROOT / paths["models_dir"]
        data_config = get_data_config()

        metadata = _ModelCache.get_metadata()
        model_name = metadata.get("best_model_type", "unknown")

        # Load training metrics
        metrics_path = models_dir / "training_metrics.json"
        metrics: dict = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                all_metrics = json.load(f)
            metrics = all_metrics.get("models", {}).get(model_name, {})

        # Model file info (last modified = training time proxy)
        model_path = models_dir / "best_model.pkl"
        trained_at = None
        if model_path.exists():
            mtime = model_path.stat().st_mtime
            trained_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        # Dataset size
        raw_path = PROJECT_ROOT / data_config["raw_path"]
        dataset_rows = 0
        if raw_path.exists():
            dataset_rows = sum(1 for _ in open(raw_path)) - 1  # exclude header

        # Feature count
        feature_names = metadata.get("feature_names", [])

        result = {
            "model_name": model_name,
            "trained_at": trained_at or _last_train_time or "unknown",
            "dataset_size": dataset_rows,
            "feature_count": len(feature_names),
            "models_trained": metadata.get("model_types_trained", []),
            "metrics": metrics,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 9: System Status
# ---------------------------------------------------------------------------


@mcp.tool()
def system_status() -> str:
    """Get system health and readiness status.

    Returns server health, model loaded status, last training time,
    and pipeline availability. Essential for monitoring and debugging.
    """
    try:
        paths = get_paths_config()
        models_dir = PROJECT_ROOT / paths["models_dir"]
        data_config = get_data_config()

        # Check model availability
        model_path = models_dir / "best_model.pkl"
        model_loaded = model_path.exists()

        # Check preprocessor
        preprocessor_path = models_dir / "preprocessor.pkl"
        preprocessor_ready = preprocessor_path.exists()

        # Check SHAP background data
        shap_path = models_dir / "shap_background.pkl"
        shap_ready = shap_path.exists()

        # Check data
        raw_path = PROJECT_ROOT / data_config["raw_path"]
        data_available = raw_path.exists()

        # Last training time from file modification time
        last_training = None
        if model_path.exists():
            mtime = model_path.stat().st_mtime
            last_training = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        # Check if retrain is currently running
        retrain_in_progress = _retrain_lock.locked()

        # Overall health
        all_ok = model_loaded and preprocessor_ready and data_available
        health = "healthy" if all_ok else "degraded"

        result = {
            "health": health,
            "model_loaded": model_loaded,
            "preprocessor_ready": preprocessor_ready,
            "shap_ready": shap_ready,
            "data_available": data_available,
            "last_training_time": last_training or _last_train_time or "never",
            "retrain_in_progress": retrain_in_progress,
            "pipeline_available": not retrain_in_progress,
            "server_uptime": "running",
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"health": "error", "error": str(e)})


# ---------------------------------------------------------------------------
# Server Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    server_config = get_server_config()

    parser = argparse.ArgumentParser(
        description="Churn Intelligence MCP Server"
    )
    parser.add_argument(
        "--transport",
        choices=["sse", "stdio", "streamable-http"],
        default="sse",
        help="MCP transport protocol (default: sse for Gemini compatibility)",
    )
    parser.add_argument(
        "--host", default=server_config["host"], help="Server host"
    )
    parser.add_argument(
        "--port", type=int, default=server_config["port"], help="Server port"
    )

    args = parser.parse_args()

    logger.info(
        f"Starting Churn Intelligence MCP server "
        f"({args.transport} on {args.host}:{args.port})"
    )

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(
            transport="streamable-http", host=args.host, port=args.port
        )
