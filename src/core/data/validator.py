"""
Data validation — validates ingested rows against Pydantic schemas
to catch type errors and out-of-range values before they reach the model.
"""

import pandas as pd

from src.schemas.models import CustomerRecord
from src.utils.logger import get_logger


logger = get_logger(__name__)


def validate_dataframe(
    df: pd.DataFrame, sample_size: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """Validate dataframe rows against Pydantic schema.

    Args:
        df: DataFrame to validate.
        sample_size: If set, only validate first N rows (for speed).

    Returns:
        Tuple of (valid_df, validation_report).
    """
    rows_to_validate = df if sample_size is None else df.head(sample_size)

    valid_indices: list[int] = []
    errors: list[dict] = []

    for idx, row in rows_to_validate.iterrows():
        try:
            CustomerRecord(**row.to_dict())
            valid_indices.append(idx)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    total = len(rows_to_validate)
    report = {
        "total_rows": total,
        "valid_rows": len(valid_indices),
        "invalid_rows": len(errors),
        "validation_rate": len(valid_indices) / total if total > 0 else 0,
        "sample_errors": errors[:5],  # first 5 errors for debugging
    }

    logger.info(
        f"Validation: {report['valid_rows']}/{report['total_rows']} rows valid "
        f"({report['validation_rate']:.1%})"
    )

    if sample_size is not None:
        # If we only validated a sample, return the full dataframe
        return df, report

    # Return only valid rows
    valid_df = df.loc[valid_indices].reset_index(drop=True)
    return valid_df, report
