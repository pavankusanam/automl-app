from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd


MAX_CATEGORY_EXAMPLES = 15
FULL_DATASET_ROW_LIMIT = 1000


def infer_task_hint(series: pd.Series) -> str:
    non_null = series.dropna()
    nunique = int(non_null.nunique())

    if pd.api.types.is_bool_dtype(series):
        return "classification"

    if pd.api.types.is_numeric_dtype(series):
        if nunique <= 20:
            return "classification"
        return "regression"

    return "classification"


def classify_feature_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def dataset_summary(df: pd.DataFrame, target_column: str) -> str:
    feature_cols = [col for col in df.columns if col != target_column]
    summary: Dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "target_column": target_column,
        "task_hint": infer_task_hint(df[target_column]),
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "columns": [],
        "target_preview": df[target_column].head(10).astype(str).tolist(),
        "sample_rows": df.head(8).astype(str).to_dict(orient="records"),
    }

    for col in df.columns:
        col_info: Dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "feature_type": classify_feature_dtype(df[col]),
            "nunique": int(df[col].nunique(dropna=True)),
            "missing": int(df[col].isna().sum()),
        }

        if col != target_column:
            if col_info["feature_type"] in {"categorical", "boolean"}:
                values = df[col].dropna().astype(str).unique().tolist()[:MAX_CATEGORY_EXAMPLES]
                col_info["category_examples"] = values
            elif col_info["feature_type"] == "numeric":
                desc = df[col].describe().to_dict()
                col_info["stats"] = {k: (None if pd.isna(v) else float(v)) for k, v in desc.items()}

        summary["columns"].append(col_info)

    if len(df) <= FULL_DATASET_ROW_LIMIT:
        summary["full_dataset_csv"] = df.to_csv(index=False)
    else:
        summary["full_dataset_csv"] = None
        summary["sampling_note"] = (
            "Dataset is larger than the safe inline limit. Use the schema and sample rows to generate code."
        )

    return json.dumps(summary, indent=2)


def build_system_prompt() -> str:
    return """
You are an expert Python machine learning engineer.

Your job is to generate ONLY valid Python code, with no markdown, no backticks, and no explanations.

You must write a complete training script for a tabular AutoML system using:
- pandas
- numpy
- scikit-learn
- joblib
- pathlib
- json

The code will be executed with these pre-defined variables:
- DATASET_PATH: absolute path to the uploaded CSV
- TARGET_COLUMN: the target column selected by the user
- WORKSPACE_DIR: workspace directory path
- RANDOM_SEED: integer random seed

Strict requirements:
1. Load the CSV from DATASET_PATH.
2. Use TARGET_COLUMN as the prediction target.
3. Support BOTH classification and regression by inspecting the target column.
4. Perform preprocessing automatically:
   - handle missing values
   - encode categorical features
   - keep numeric features
   - safely handle boolean columns
   - ignore target leakage
   - drop obvious ID-like columns when appropriate
5. Choose ONE suitable scikit-learn model only.
6. Split data into train/test.
7. Fit a preprocessing object and model separately.
8. Save these files inside WORKSPACE_DIR/artifacts:
   - model.pkl
   - preprocessor.pkl
   - schema.json
   - metadata.json
9. schema.json must describe RAW INPUT FEATURES for manual single-record prediction UI.
10. metadata.json must include:
   - task
   - target_column
   - model_name
   - metrics
   - train_rows
   - test_rows
11. TRAINING_OUTPUT dictionary must be created at the end with:
   - success
   - task
   - metrics
   - model_name
12. Use pathlib paths, not raw file open() unless absolutely necessary.
13. Do not import or use:
   - os
   - sys
   - subprocess
   - socket
   - requests
   - shutil
   - pickle
14. Do not delete files.
15. Do not print huge outputs.
16. The code must be deterministic where possible.

For model choice:
- For classification, prefer robust baseline models such as RandomForestClassifier or LogisticRegression depending on the data.
- For regression, prefer robust baseline models such as RandomForestRegressor, Ridge, or LinearRegression depending on the data.

For schema.json:
- Include a list called raw_features.
- Each item should contain:
  - name
  - input_type: one of numeric, categorical, boolean, text
  - required
  - choices (for categorical/boolean if practical, else [])
  - default

For metrics:
- Classification: accuracy, f1_macro, precision_macro, recall_macro
- Regression: rmse, mae, r2

Return ONLY executable Python code.
""".strip()


def build_user_prompt(df: pd.DataFrame, target_column: str) -> str:
    summary = dataset_summary(df, target_column)
    return f"""
Dataset summary:
{summary}

Selected target column:
{target_column}

Generate the Python training script now.
""".strip()


def build_retry_prompt(previous_code: str, error_text: str, df: pd.DataFrame, target_column: str) -> str:
    summary = dataset_summary(df, target_column)
    return f"""
The previous Python code failed.

Dataset summary:
{summary}

Target column:
{target_column}

Previous code:
{previous_code}

Error:
{error_text}

Rewrite the entire script from scratch.
Return ONLY valid Python code.
""".strip()