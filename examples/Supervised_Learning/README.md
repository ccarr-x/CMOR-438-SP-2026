# Supervised learning examples

Notebooks that walk through classifiers and regressors under `rice_ml.Supervised_Learning`: trees, linear and logistic models, k-NN, perceptrons, MLP-style networks, and ensemble methods.

## Notebooks

| Notebook | Focus |
|----------|--------|
| `Decision_Trees.ipynb` | Decision trees on tabular data (includes spam-style exploration where noted in cells). |
| `Decision_Tree_Classifier.ipynb` | Decision tree classification workflow. |
| `Decision_Tree_Regressor.ipynb` | Decision tree regression workflow. |
| `Logistic_Regression.ipynb` | Logistic regression for classification. |
| `Linear_Regression.ipynb` | Ordinary linear regression. |
| `K_Nearest_Neighbors.ipynb` | k-NN classification. |
| `The_Perceptron.ipynb` | Single-layer perceptron (linear boundary). |
| `The_Multilayer_Perceptron.ipynb` | Multilayer perceptron-style training. |
| `Ensemble_Methods/Random_Forest.ipynb` | Random forest (single notebook variant). |
| `Ensemble_Methods/Random_Forests.ipynb` | Random forests — alternate / extended walkthrough. |

## Prerequisites

- **Python** 3.9 or newer (matches `pyproject.toml`).
- Install the library in editable mode from the **repository root**:

  ```bash
  pip install -e .
  pip install -e ".[dev]"   # optional: pytest and tooling
  ```

- Jupyter: `pip install notebook` or use VS Code / Cursor with the **same interpreter** you used for `pip install -e .` so `import rice_ml` resolves to your checkout.

## How to run

From the project root:

```bash
jupyter notebook examples/Supervised_Learning/
```

Or open a single `.ipynb` in Cursor/VS Code and select that kernel.

## Data and paths

- Shared CSVs live in **`examples/datasets/`** (e.g. `spam_email_dataset.csv`, `Breast_Cancer_Research.csv`).
- Notebooks that load spam email data search upward from the current working directory for `examples/datasets/spam_email_dataset.csv`. Run from the repo root or from `examples/` so discovery succeeds.
- **`examples/load_dataset.py`** resolves paths relative to the script — safe for clones on other machines.

## Tips

- After editing **`src/rice_ml/`**, restart the notebook kernel before re-importing.
- If imports seem stale, confirm the kernel’s Python matches where you ran `pip install -e .` (e.g. conda vs system Python).

## Related

- [Examples hub](../README.md)
- [API reference](../../docs/api.md)
- [Supervised package README](../../src/rice_ml/Supervised_Learning/README.md)
