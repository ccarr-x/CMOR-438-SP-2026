"""
Simple dataset loader that works regardless of notebook working directory.
"""

import pandas as pd
from pathlib import Path

def load_spam_dataset():
    """Load the spam email dataset using absolute path."""
    dataset_path = Path("/Users/ccarr21/Desktop/cmor438_carr/CMOR-438-SP-2026/examples/datasets/spam_email_dataset.csv")
    return pd.read_csv(dataset_path)

def load_dataset(filename):
    """Load any dataset from the examples/datasets directory."""
    dataset_path = Path(f"/Users/ccarr21/Desktop/cmor438_carr/CMOR-438-SP-2026/examples/datasets/{filename}")
    return pd.read_csv(dataset_path)
