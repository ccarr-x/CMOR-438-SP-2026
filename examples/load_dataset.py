"""
Dataset loaders for example notebooks and scripts.

Paths are resolved relative to this file so clones work on any machine.
"""

from pathlib import Path

import pandas as pd

_EXAMPLES_ROOT = Path(__file__).resolve().parent
_DATASETS_DIR = _EXAMPLES_ROOT / "datasets"


def dataset_path(filename: str) -> Path:
    """Return path to a file under ``examples/datasets``."""
    p = _DATASETS_DIR / filename
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {p}")
    return p


def load_spam_dataset():
    """Load the spam email dataset."""
    return pd.read_csv(dataset_path("spam_email_dataset.csv"))


def load_dataset(filename: str):
    """Load any CSV from ``examples/datasets``."""
    return pd.read_csv(dataset_path(filename))
