"""
Training pipeline — orchestrates the 4-stage DVC pipeline:
ingest → features → train → evaluate.

Can be run as individual DVC stages or called programmatically for retraining.
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config.settings import (
    PROJECT_ROOT,
    get_data_config,
    get_feature_config,
    get_model_config,
    get_paths_config,
)
from src.core.data.loader import load_raw_data, clean_data, save_dataframe
from src.core.data.validator import validate_dataframe
from src.core.data.preprocessor import ChurnPreprocessor
from src.core.features.engineer import engineer_features
from src.core.models.trainer import (
    create_model,
    train_model,
    evaluate_model,
    log_to_mlflow,
    save_model,
)
from src.core.models.registry import select_best_model, save_metrics
from src.core.explainability.shap_explainer import save_background_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Stages
# ---------------------------------------------------------------------------


def stage_ingest() -> pd.DataFrame:
    """Stage 1: Load, clean, and validate raw data.

    Reads the raw CSV, fixes types, handles missing values, and
    validates a sample using Pydantic schemas.
    """
    logger.info("=" * 60)
    logger.info("STAGE: Data Ingestion & Validation")
    logger.info("=" * 60)

    data_config = get_data_config()

    # Load raw data
    df = load_raw_data()

    # Clean data (type conversions, dedup)
    df = clean_data(df)

    # Validate with Pydantic (sample for speed, full in production)
    df, validation_report = validate_dataframe(df, sample_size=200)
    logger.info(f"Validation report: {json.dumps(validation_report, indent=2, default=str)}")

    # Save cleaned data
    processed_dir = PROJECT_ROOT / data_config["processed_dir"]
    save_dataframe(df, processed_dir / "cleaned.csv")

    return df


def stage_feature_engineering(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stage 2: Create derived features.

    Engineers domain-specific features BEFORE the train/test split.
    These are all row-level transforms, so no data leakage.
    """
    logger.info("=" * 60)
    logger.info("STAGE: Feature Engineering")
    logger.info("=" * 60)

    data_config = get_data_config()
    processed_dir = PROJECT_ROOT / data_config["processed_dir"]

    if df is None:
        df = pd.read_csv(processed_dir / "cleaned.csv")

    # Engineer new features
    df = engineer_features(df)

    # Save featured data
    save_dataframe(df, processed_dir / "featured.csv")

    return df


def stage_train(
    df: pd.DataFrame | None = None,
    model_types: list[str] | None = None,
    hyperparameters: dict | None = None,
) -> dict:
    """Stage 3: Train models with MLflow tracking.

    Splits data, fits preprocessor on training set ONLY,
    trains all model types, logs to MLflow, selects the best.

    Args:
        df: Featured DataFrame. If None, loads from disk.
        model_types: List of models to train. Defaults to all three.
        hyperparameters: Override hyperparameters per model type.
            Example: {"random_forest": {"n_estimators": 300}}

    Returns:
        Dictionary with best model info, all results, and paths.
    """
    logger.info("=" * 60)
    logger.info("STAGE: Model Training")
    logger.info("=" * 60)

    data_config = get_data_config()
    feature_config = get_feature_config()
    paths_config = get_paths_config()
    processed_dir = PROJECT_ROOT / data_config["processed_dir"]
    models_dir = PROJECT_ROOT / paths_config["models_dir"]
    plots_dir = PROJECT_ROOT / paths_config["plots_dir"]

    if df is None:
        df = pd.read_csv(processed_dir / "featured.csv")

    # Separate target and features
    target_col = feature_config["target_column"]
    id_col = feature_config["id_column"]

    # Encode target: Yes=1, No=0
    y = df[target_col].map({"Yes": 1, "No": 0})

    # Drop target and ID columns
    cols_to_drop = [target_col]
    if id_col in df.columns:
        cols_to_drop.append(id_col)
    X_raw = df.drop(columns=cols_to_drop)

    # Train/test split BEFORE preprocessing (prevents data leakage!)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        stratify=y,  # Maintain class balance in both splits
    )

    logger.info(f"Train set: {len(X_train_raw)} | Test set: {len(X_test_raw)}")
    logger.info(
        f"Churn rate — Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}"
    )

    # Fit preprocessor on training data ONLY (critical for no leakage)
    preprocessor = ChurnPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = preprocessor.get_feature_names()

    # Save preprocessor
    preprocessor.save()

    # Save test data for evaluation stage
    processed_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(X_test, processed_dir / "X_test.pkl")
    joblib.dump(y_test.values, processed_dir / "y_test.pkl")

    # Save background data for SHAP (100 random training samples)
    rng = np.random.RandomState(42)
    bg_indices = rng.choice(
        len(X_train), size=min(100, len(X_train)), replace=False
    )
    save_background_data(X_train[bg_indices])

    # Train all specified models
    if model_types is None:
        model_types = ["logistic_regression", "random_forest", "xgboost"]

    all_results: list[dict] = []

    for model_type in model_types:
        logger.info(f"\n--- Training {model_type} ---")

        # Get hyperparameters (config defaults + overrides)
        model_hp = dict(get_model_config(model_type))
        if hyperparameters and model_type in hyperparameters:
            model_hp = {**model_hp, **hyperparameters[model_type]}

        # Create and train model
        model = create_model(model_type, model_hp)
        train_result = train_model(model, X_train, y_train, model_type)

        # Evaluate
        eval_result = evaluate_model(
            train_result["model"], X_test, y_test, feature_names
        )

        # Log to MLflow
        run_id = log_to_mlflow(
            model=train_result["model"],
            model_type=model_type,
            metrics=eval_result["metrics"],
            confusion=eval_result["confusion_matrix"],
            feature_importance=eval_result["feature_importance"],
            y_test=y_test.values,
            y_proba=eval_result["probabilities"],
            hyperparameters=model_hp,
            plots_dir=plots_dir,
        )

        # Save individual model
        save_model(train_result["model"], model_type)

        all_results.append(
            {
                "model_type": model_type,
                "model": train_result["model"],
                "metrics": eval_result["metrics"],
                "confusion_matrix": eval_result["confusion_matrix"],
                "feature_importance": eval_result["feature_importance"],
                "mlflow_run_id": run_id,
            }
        )

        logger.info(
            f"{model_type} — "
            f"F1: {eval_result['metrics']['f1_score']:.4f} | "
            f"AUC: {eval_result['metrics']['roc_auc']:.4f}"
        )

    # Select best model by F1 score
    best = select_best_model(all_results, metric="f1_score")
    save_model(best["model"], "best_model", models_dir / "best_model.pkl")

    # Save training metrics for DVC tracking
    training_metrics = {
        "best_model": best["model_type"],
        "models": {r["model_type"]: r["metrics"] for r in all_results},
    }
    save_metrics(training_metrics, models_dir / "training_metrics.json")

    # Save model metadata (feature names, etc.)
    metadata = {
        "best_model_type": best["model_type"],
        "feature_names": feature_names,
        "model_types_trained": model_types,
    }
    joblib.dump(metadata, models_dir / "metadata.pkl")

    logger.info(
        f"\nBest model: {best['model_type']} "
        f"(F1: {best['metrics']['f1_score']:.4f})"
    )

    return {
        "best_model": best,
        "all_results": all_results,
        "feature_names": feature_names,
    }


def stage_evaluate() -> dict:
    """Stage 4: Detailed evaluation and SHAP analysis.

    Loads the saved test data and best model, computes detailed
    metrics, generates SHAP global importance plot.
    """
    logger.info("=" * 60)
    logger.info("STAGE: Model Evaluation")
    logger.info("=" * 60)

    data_config = get_data_config()
    paths_config = get_paths_config()
    processed_dir = PROJECT_ROOT / data_config["processed_dir"]
    models_dir = PROJECT_ROOT / paths_config["models_dir"]

    # Load test data and model
    X_test = joblib.load(processed_dir / "X_test.pkl")
    y_test = joblib.load(processed_dir / "y_test.pkl")
    metadata = joblib.load(models_dir / "metadata.pkl")

    from src.core.models.trainer import load_model

    best_model = load_model(path=models_dir / "best_model.pkl")

    feature_names = metadata["feature_names"]

    # Full evaluation
    eval_result = evaluate_model(best_model, X_test, y_test, feature_names)

    # SHAP global importance
    shap_importance: dict = {}
    try:
        from src.core.explainability.shap_explainer import (
            ShapExplainer,
            load_background_data,
        )

        background = load_background_data()
        explainer = ShapExplainer(best_model, feature_names, background)
        shap_importance = explainer.global_importance(
            X_test[:200]
        )  # Sample for speed
        explainer.save_global_plot(X_test[:200])
    except Exception as e:
        logger.warning(f"SHAP evaluation failed: {e}")

    # Save evaluation results
    evaluation = {
        "best_model_type": metadata["best_model_type"],
        "metrics": eval_result["metrics"],
        "confusion_matrix": eval_result["confusion_matrix"],
        "top_features": dict(
            list(eval_result["feature_importance"].items())[:10]
        ),
        "shap_top_features": dict(list(shap_importance.items())[:10]),
    }

    eval_path = models_dir / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    logger.info(f"Evaluation saved to {eval_path}")
    logger.info(f"Best model metrics: {eval_result['metrics']}")

    return evaluation


# ---------------------------------------------------------------------------
# Full Pipeline Runners (called by MCP tools)
# ---------------------------------------------------------------------------


def run_full_pipeline(
    model_types: list[str] | None = None,
    hyperparameters: dict | None = None,
) -> dict:
    """Run the complete pipeline end-to-end.

    Used by MCP retrain_model tool.
    """
    logger.info("Running full training pipeline...")

    df = stage_ingest()
    df = stage_feature_engineering(df)
    train_results = stage_train(df, model_types, hyperparameters)
    eval_results = stage_evaluate()

    return {
        "training": {
            "best_model": train_results["best_model"]["model_type"],
            "metrics": train_results["best_model"]["metrics"],
            "mlflow_run_id": train_results["best_model"]["mlflow_run_id"],
        },
        "evaluation": eval_results,
    }


def run_single_model_training(
    model_type: str,
    hyperparameters: dict | None = None,
) -> dict:
    """Train a single model type (for targeted retraining via MCP)."""
    hp = {model_type: hyperparameters} if hyperparameters else None
    return run_full_pipeline(model_types=[model_type], hyperparameters=hp)


# ---------------------------------------------------------------------------
# CLI Entry Point (for DVC)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Churn Intelligence Training Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["ingest", "features", "train", "evaluate", "full"],
        required=True,
        help="Pipeline stage to run",
    )

    args = parser.parse_args()

    if args.stage == "ingest":
        stage_ingest()
    elif args.stage == "features":
        stage_feature_engineering()
    elif args.stage == "train":
        stage_train()
    elif args.stage == "evaluate":
        stage_evaluate()
    elif args.stage == "full":
        results = run_full_pipeline()
        print(json.dumps(results["training"]["metrics"], indent=2))
