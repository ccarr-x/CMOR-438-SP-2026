# Processing Utilities

The `processing` package provides shared preprocessing and postprocessing utilities used across models.

## Files

- `preprocessing.py`
  - Data splitting helpers (e.g. train/test split)
  - Feature scaling / normalization helpers
- `postprocessing.py`
  - Common evaluation metrics
  - Reporting helpers for model outputs

## Usage

```python
from rice_ml.processing.preprocessing import train_test_split, standardize
from rice_ml.processing.postprocessing import PostProcessor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = standardize(X_train, X_test)

processor = PostProcessor()
metrics = processor.classification_metrics(y_test, y_pred)
```

## Tests

Processing tests are in `tests/test_processing/`.
