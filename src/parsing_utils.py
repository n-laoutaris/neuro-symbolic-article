"""Utilities for reading and parsing text and JSON files used in the dissertation pipeline."""

import json
from pathlib import Path


def setup_run_log(file_name: str, artifact_dir: str = "Testing_Artifacts_Test") -> str:
    """Create a Markdown log file for the current run using the same naming as the artifacts."""
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    log_path = artifact_path / f"{file_name}_log.md"
    if log_path.exists():
        log_path.unlink()

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Pipeline Run Log\n\n")

    return str(log_path)


def append_run_log(message: str, file_name: str, artifact_dir: str = "Testing_Artifacts_Test") -> None:
    """Append a single terminal message to the current run's Markdown log file."""
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    log_path = artifact_path / f"{file_name}_log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

# Read txt file into a string (used for prompts)
def read_txt(path: str) -> str:
    """Reads a text file and returns its contents as a string.

    Args:
        path: The file path to the text file.

    Returns:
        The contents of the file as a string.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Read a JSON file and return it as string or raw JSON
def read_json(path: str, raw: bool = False):
    """Reads a JSON file and returns its contents.

    Args:
        path: The file path to the JSON file.
        raw: If True, returns the raw dictionary; otherwise, returns a formatted JSON string.

    Returns:
        The JSON data as a dictionary if raw is True, or as a formatted string otherwise.
    """
    with open(path, "r") as f:
        json_file = json.load(f)
    if raw:
        return json_file
    # Else, convert JSON to string
    return json.dumps(json_file, indent=2)