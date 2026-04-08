from __future__ import annotations

import ast
import builtins
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from artifact_utils import (
    ARTIFACTS_DIR,
    GENERATED_CODE_PATH,
    LAST_ERROR_PATH,
    WORKSPACE_DIR,
    artifacts_ready,
    write_text,
)
from llm_engine import GeminiCodeGenerator
from prompt_builder import build_retry_prompt, build_system_prompt, build_user_prompt


RANDOM_SEED = 42
MAX_RETRIES = 2

ALLOWED_ROOT_IMPORTS = {
    "pandas",
    "numpy",
    "sklearn",
    "joblib",
    "json",
    "math",
    "pathlib",
    "statistics",
    "collections",
    "re",
    "warnings",
}

BANNED_CALL_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}

BANNED_ATTRS = {
    "system",
    "popen",
    "remove",
    "unlink",
    "rmtree",
    "rename",
    "replace",
    "chmod",
    "chown",
}

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def safe_import(name: str, globals_=None, locals_=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_ROOT_IMPORTS:
        raise ImportError(f"Import '{name}' is not allowed.")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


SAFE_BUILTINS["__import__"] = safe_import


def validate_generated_code(code: str) -> None:
    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_ROOT_IMPORTS:
                    raise ValueError(f"Disallowed import: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise ValueError("Relative imports are not allowed.")
            root = node.module.split(".")[0]
            if root not in ALLOWED_ROOT_IMPORTS:
                raise ValueError(f"Disallowed import-from module: {node.module}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES:
                raise ValueError(f"Disallowed function call: {node.func.id}")

        if isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                raise ValueError(f"Disallowed attribute usage: {node.attr}")


def execute_code(code: str, dataset_path: Path, target_column: str) -> Dict[str, Any]:
    validate_generated_code(code)
    write_text(GENERATED_CODE_PATH, code)

    exec_globals: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "DATASET_PATH": str(dataset_path),
        "TARGET_COLUMN": target_column,
        "WORKSPACE_DIR": str(WORKSPACE_DIR),
        "RANDOM_SEED": RANDOM_SEED,
    }
    exec_locals: Dict[str, Any] = {}

    compiled = compile(code, str(GENERATED_CODE_PATH), "exec")
    exec(compiled, exec_globals, exec_locals)

    training_output = exec_locals.get("TRAINING_OUTPUT", exec_globals.get("TRAINING_OUTPUT"))
    if not isinstance(training_output, dict):
        raise ValueError("Generated code did not create a valid TRAINING_OUTPUT dictionary.")

    if not artifacts_ready():
        raise ValueError(
            f"Training finished but required artifacts were not created in: {ARTIFACTS_DIR}"
        )

    return training_output


def run_automl_flow(df: pd.DataFrame, dataset_path: Path, target_column: str) -> Tuple[Dict[str, Any], str]:
    generator = GeminiCodeGenerator()
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(df, target_column)

    last_error = ""
    previous_code = ""

    for attempt in range(MAX_RETRIES + 1):
        if attempt == 0:
            code = generator.generate_code(system_prompt, user_prompt)
        else:
            retry_prompt = build_retry_prompt(previous_code, last_error, df, target_column)
            code = generator.generate_code(system_prompt, retry_prompt)

        previous_code = code

        try:
            output = execute_code(code, dataset_path, target_column)
            return output, code
        except Exception as exc:
            last_error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            write_text(LAST_ERROR_PATH, last_error)

    raise RuntimeError(
        "AutoML training failed after all retries.\n\nLast error:\n" + last_error
    )