from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from artifact_utils import (
    DATASET_PATH,
    GENERATED_CODE_PATH,
    LAST_ERROR_PATH,
    artifacts_ready,
    build_artifact_zip,
    ensure_workspace,
    load_metadata,
    load_schema,
    predict_single_record,
    reset_run_state,
    save_uploaded_file,
)
from execution_engine import run_automl_flow


load_dotenv()
ensure_workspace()

st.set_page_config(page_title="LLM-Powered AutoML", layout="wide")
st.title("LLM-Powered AutoML System")
st.caption("Upload CSV → choose target → Gemini generates ML code → train → predict one record")


MAX_FILE_SIZE_MB = 10
MAX_ROWS = 50_000


def validate_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"Dataset has {len(df)} rows. Limit is {MAX_ROWS}.")
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns.")
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(f"Duplicate column names found: {duplicate_cols}")


def coerce_manual_input(raw_value: Any, feature: Dict[str, Any]) -> Any:
    input_type = feature.get("input_type", "text")
    if raw_value is None:
        return None

    if input_type == "numeric":
        return float(raw_value)

    if input_type == "boolean":
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).strip().lower() in {"true", "1", "yes"}

    return raw_value


def render_prediction_form(schema: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    raw_features = schema.get("raw_features", [])

    for feature in raw_features:
        name = feature["name"]
        input_type = feature.get("input_type", "text")
        choices = feature.get("choices", [])
        default = feature.get("default")

        if input_type == "numeric":
            values[name] = st.number_input(
                label=name,
                value=float(default) if default not in [None, ""] else 0.0,
                step=1.0,
            )
        elif input_type == "boolean":
            bool_default = bool(default) if default is not None else False
            values[name] = st.selectbox(name, [True, False], index=0 if bool_default else 1)
        elif input_type == "categorical" and choices:
            default_index = choices.index(default) if default in choices else 0
            values[name] = st.selectbox(name, choices, index=default_index)
        else:
            values[name] = st.text_input(name, value="" if default is None else str(default))

    return values


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File is {file_size_mb:.2f} MB. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
        validate_dataset(df)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Basic Statistics")
    st.dataframe(df.describe(include="all").fillna(""), use_container_width=True)

    target_column = st.selectbox("Select the target column", options=df.columns.tolist())

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Train AutoML Model", type="primary"):
            try:
                reset_run_state()
                dataset_path = save_uploaded_file(uploaded_file.getvalue())

                with st.spinner("Generating code, training model, and creating artifacts..."):
                    training_output, generated_code = run_automl_flow(df, dataset_path, target_column)

                st.session_state["trained"] = True
                st.session_state["training_output"] = training_output
                st.session_state["generated_code"] = generated_code
                st.success("Training completed successfully.")

            except Exception as exc:
                st.session_state["trained"] = False
                st.error(str(exc))
                if LAST_ERROR_PATH.exists():
                    with st.expander("Show last execution error"):
                        st.code(LAST_ERROR_PATH.read_text(encoding="utf-8"), language="text")

    with col2:
        if GENERATED_CODE_PATH.exists():
            with st.expander("Generated training code"):
                st.code(GENERATED_CODE_PATH.read_text(encoding="utf-8"), language="python")

    if artifacts_ready():
        metadata = load_metadata()
        schema = load_schema()

        st.subheader("Training Result")
        st.json(metadata)

        st.download_button(
            label="Download trained artifacts",
            data=build_artifact_zip(),
            file_name="automl_artifacts.zip",
            mime="application/zip",
        )

        st.subheader("Single Record Prediction")
        with st.form("prediction_form"):
            raw_values = render_prediction_form(schema)
            submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                final_record = {}
                for feature in schema.get("raw_features", []):
                    name = feature["name"]
                    final_record[name] = coerce_manual_input(raw_values.get(name), feature)

                prediction, probabilities = predict_single_record(final_record)

                st.success(f"Prediction: {prediction}")
                if probabilities:
                    st.write("Class probabilities:")
                    st.json(probabilities)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")