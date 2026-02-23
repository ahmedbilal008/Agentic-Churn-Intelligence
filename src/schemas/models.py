"""
Pydantic schemas — shared data contracts used for input validation at
ingestion boundaries and MCP tool inputs/outputs.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ---------------------------------------------------------------------------
# Data Validation Schemas
# ---------------------------------------------------------------------------


class CustomerRecord(BaseModel):
    """Validates a raw customer record from the dataset.

    Used during data ingestion to catch data quality issues early.
    """

    customerID: str
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    Churn: Optional[str] = None  # Optional: not present during inference

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in ("Male", "Female"):
            raise ValueError(f"Invalid gender: {v}")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        valid = ("Month-to-month", "One year", "Two year")
        if v not in valid:
            raise ValueError(
                f"Invalid contract type: {v}. Must be one of {valid}"
            )
        return v


# ---------------------------------------------------------------------------
# MCP Tool Input/Output Schemas
# ---------------------------------------------------------------------------


class CustomerFeatures(BaseModel):
    """Input schema for prediction — raw customer features.

    This is what an AI agent passes to predict_churn.
    Note: No customerID or Churn label needed.
    """

    gender: str = Field(description="Male or Female")
    SeniorCitizen: int = Field(
        ge=0, le=1, description="1 if senior citizen, 0 otherwise"
    )
    Partner: str = Field(description="Yes or No")
    Dependents: str = Field(description="Yes or No")
    tenure: int = Field(ge=0, description="Months as customer")
    PhoneService: str = Field(description="Yes or No")
    MultipleLines: str = Field(description="Yes, No, or No phone service")
    InternetService: str = Field(description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(description="Yes, No, or No internet service")
    OnlineBackup: str = Field(description="Yes, No, or No internet service")
    DeviceProtection: str = Field(
        description="Yes, No, or No internet service"
    )
    TechSupport: str = Field(description="Yes, No, or No internet service")
    StreamingTV: str = Field(description="Yes, No, or No internet service")
    StreamingMovies: str = Field(description="Yes, No, or No internet service")
    Contract: str = Field(
        description="Month-to-month, One year, or Two year"
    )
    PaperlessBilling: str = Field(description="Yes or No")
    PaymentMethod: str = Field(
        description="Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)"
    )
    MonthlyCharges: float = Field(ge=0, description="Monthly charge amount")
    TotalCharges: float = Field(ge=0, description="Total charges to date")


class PredictionResponse(BaseModel):
    """Output schema for churn prediction."""

    churn_probability: float = Field(description="Probability of churn (0-1)")
    risk_level: str = Field(description="LOW, MEDIUM, or HIGH")
    top_drivers: list[dict] = Field(
        default_factory=list,
        description="Top features driving the prediction",
    )
    model_used: str = Field(default="", description="Name of the model used")


class ExplanationResponse(BaseModel):
    """Output schema for SHAP-based prediction explanation."""

    feature_contributions: list[dict] = Field(
        description="Each feature's SHAP contribution to the prediction"
    )
    base_value: float = Field(description="Model's base prediction (average)")
    prediction_value: float = Field(description="Final prediction value")
    top_positive_drivers: list[str] = Field(
        description="Features pushing toward churn"
    )
    top_negative_drivers: list[str] = Field(
        description="Features pushing away from churn"
    )


class ModelMetricsResponse(BaseModel):
    """Output schema for model metrics."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: dict[str, int] = Field(
        description="TP, TN, FP, FN counts"
    )


class ModelComparisonResponse(BaseModel):
    """Output schema for model comparison leaderboard."""

    leaderboard: list[dict]
    best_model: str
    best_metric: str = Field(
        default="f1_score", description="Metric used for ranking"
    )


class DatasetSummaryResponse(BaseModel):
    """Output schema for dataset summary."""

    total_rows: int
    total_features: int
    numerical_features: list[str]
    categorical_features: list[str]
    missing_values: dict[str, int]
    class_distribution: dict[str, int]
    churn_rate: float


class RetrainRequest(BaseModel):
    """Input schema for model retraining."""

    model_type: str = Field(
        default="random_forest",
        description="Model type: logistic_regression, random_forest, or xgboost",
    )
    hyperparameters: Optional[dict] = Field(
        default=None,
        description="Optional hyperparameter overrides",
    )


class RetrainResponse(BaseModel):
    """Output schema for retraining results."""

    status: str
    model_type: str
    metrics: dict[str, float]
    mlflow_run_id: str
    message: str
