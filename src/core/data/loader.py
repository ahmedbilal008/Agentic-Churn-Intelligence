"""
Data loading and cleaning — handles CSV ingestion, type fixes,
and deduplication. Isolated so data sources can be swapped without
touching training logic.
"""

import pandas as pd
from pathlib import Path

from src.config.settings import PROJECT_ROOT, get_data_config
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_raw_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load raw churn data from CSV.

    Args:
        filepath: Path to CSV file. Defaults to config path.

    Returns:
        Raw DataFrame as-is from the CSV.
    """
    if filepath is None:
        config = get_data_config()
        filepath = PROJECT_ROOT / config["raw_path"]

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(
        f"Loaded {len(df)} records with {df.shape[1]} columns from {filepath.name}"
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform initial data cleaning.

    Handles:
    - TotalCharges type conversion (stored as string in raw CSV)
    - Missing value imputation for TotalCharges
    - Duplicate removal

    Returns:
        Cleaned DataFrame.
    """
    df_clean = df.copy()

    # TotalCharges is object type in raw CSV (has empty strings)
    # This is a common real-world data issue
    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"], errors="coerce"
    )

    # New customers (tenure=0) have empty TotalCharges — fill with 0
    null_count = df_clean["TotalCharges"].isnull().sum()
    if null_count > 0:
        logger.info(f"Filling {null_count} missing TotalCharges values with 0.0")
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Remove duplicate customer records
    dupes = df_clean.duplicated(subset=["customerID"], keep="first").sum()
    if dupes > 0:
        logger.warning(f"Removing {dupes} duplicate customer records")
        df_clean = df_clean.drop_duplicates(subset=["customerID"], keep="first")

    logger.info(
        f"Cleaned data: {len(df_clean)} records, {df_clean.shape[1]} columns"
    )
    return df_clean


def save_dataframe(df: pd.DataFrame, filepath: str | Path) -> Path:
    """Save DataFrame to CSV, creating directories as needed."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} records to {filepath}")
    return filepath
