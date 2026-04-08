from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR / "workspace"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
ARTIFACTS_DIR = WORKSPACE_DIR / "artifacts"
LOGS_DIR = WORKSPACE_DIR / "logs"
GENERATED_DIR = WORKSPACE_DIR / "generated"

DATASET_PATH = UPLOADS_DIR / "dataset.csv"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
SCHEMA_PATH = ARTIFACTS_DIR / "schema.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
GENERATED_CODE_PATH = GENERATED_DIR / "generated_train.py"
LAST_ERROR_PATH = LOGS_DIR / "last_error.txt"


def ensure_workspace() -> None:
    for path in [WORKSPACE_DIR, UPLOADS_DIR, ARTIFACTS_DIR, LOGS_DIR, GENERATED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def reset_run_state() -> None:
    ensure_workspace()
    for path in [ARTIFACTS_DIR, LOGS_DIR, GENERATED_DIR]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(file_bytes: bytes) -> Path:
    ensure_workspace()
    DATASET_PATH.write_bytes(file_bytes)
    return DATASET_PATH


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def artifacts_ready() -> bool:
    return MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() and SCHEMA_PATH.exists() and METADATA_PATH.exists()


def load_schema() -> Dict[str, Any]:
    return read_json(SCHEMA_PATH)


def load_metadata() -> Dict[str, Any]:
    return read_json(METADATA_PATH)


def predict_single_record(record: Dict[str, Any]) -> Tuple[Any, Optional[Dict[str, float]]]:
    if not artifacts_ready():
        raise FileNotFoundError("Artifacts are not ready. Train the model first.")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    one_row = pd.DataFrame([record])
    transformed = preprocessor.transform(one_row)
    prediction = model.predict(transformed)[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(transformed)[0]
            classes = list(getattr(model, "classes_", []))
            if classes:
                probabilities = {str(cls): float(score) for cls, score in zip(classes, proba)}
        except Exception:
            probabilities = None

    return prediction, probabilities


def build_artifact_zip() -> bytes:
    ensure_workspace()
    memory_file = io.BytesIO()

    with zipfile.ZipFile(memory_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [MODEL_PATH, PREPROCESSOR_PATH, SCHEMA_PATH, METADATA_PATH, GENERATED_CODE_PATH]:
            if path.exists():
                zf.write(path, arcname=path.name)

    memory_file.seek(0)
    return memory_file.read()