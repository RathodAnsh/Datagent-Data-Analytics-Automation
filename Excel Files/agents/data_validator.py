#4. data_validator.py

import pandas as pd

def validate_dataset(df: pd.DataFrame):
    issues = []

    if df.isnull().sum().sum() > 0:
        issues.append("Dataset contains missing values.")

        issues.append("Dataset contains duplicate rows.")

    empty_cols = df.columns[df.isnull().all()]
    if len(empty_cols) > 0:
        issues.append(f"Empty columns found: {list(empty_cols)}")

    if issues:
        return False, issues

    return True, "Dataset is clean"
