"""
Feature engineering — creates domain-specific derived columns from raw customer data.
All transforms are row-level (no global stats), safe to apply before the train/test split.
"""

import pandas as pd
import numpy as np

from src.utils.logger import get_logger


logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw customer data.

    These features are created BEFORE train/test split and BEFORE
    encoding/scaling, so they don't cause data leakage.

    Works on both full datasets and single-row DataFrames (for inference).
    """
    df_feat = df.copy()

    # --- Tenure Groups ---
    # Categorize customer loyalty into meaningful segments.
    # Business insight: churn patterns differ sharply by loyalty tier.
    df_feat["tenure_group"] = pd.cut(
        df_feat["tenure"],
        bins=[-1, 12, 24, 48, 60, np.inf],
        labels=["0-12", "12-24", "24-48", "48-60", "60+"],
        right=True,
    ).astype(str)

    # --- Average Monthly Spend ---
    # TotalCharges / tenure — reveals if a customer is overpaying
    # relative to their loyalty. Handle division by zero for new customers.
    df_feat["avg_charges_per_month"] = np.where(
        df_feat["tenure"] > 0,
        df_feat["TotalCharges"] / df_feat["tenure"],
        df_feat["MonthlyCharges"],
    )

    # --- Charges Ratio ---
    # MonthlyCharges / TotalCharges — high ratio = new customer paying a lot.
    df_feat["charges_ratio"] = np.where(
        df_feat["TotalCharges"] > 0,
        df_feat["MonthlyCharges"] / df_feat["TotalCharges"],
        1.0,
    )

    # --- Number of Services ---
    # Customers using more services have higher switching costs → churn less.
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df_feat["num_services"] = df_feat[service_columns].apply(
        lambda row: sum(
            1
            for val in row
            if val not in ("No", "No phone service", "No internet service")
        ),
        axis=1,
    )

    # --- Has Internet Service ---
    # Binary: simplifies the 3-category InternetService into presence/absence.
    df_feat["has_internet"] = np.where(
        df_feat["InternetService"] != "No", "Yes", "No"
    )

    # --- Senior with High Charges ---
    # Interaction feature: seniors paying >$70/month churn at higher rates.
    df_feat["senior_high_charges"] = (
        (df_feat["SeniorCitizen"] == 1) & (df_feat["MonthlyCharges"] > 70)
    ).astype(int)

    logger.info(
        f"Engineered features: {df.shape[1]} → {df_feat.shape[1]} columns"
    )
    return df_feat
