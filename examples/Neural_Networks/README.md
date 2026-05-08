# Neural networks examples

Demonstrates **`rice_ml.Neural_Networks`**: custom dense networks for classification and regression, plus preprocessing helpers used in coursework-style workflows.

## Notebooks

| Notebook | Focus |
|----------|--------|
| **`Neural_Network.ipynb`** | **SigmoidNeuralNetwork** (binary classification on tabular features) and **DenseNetwork** (regression); uses `rice_ml.processing` for splits and metrics; includes loss curves and optional ROC-style plots where applicable. |

## Prerequisites

- Python **3.9+**, editable install from repo root:

  ```bash
  pip install -e .
  ```

- Same Jupyter **kernel** as the environment where `rice_ml` is installed (avoid mixing system Python with a conda env that has an older copy of the package).

## How to run

```bash
jupyter notebook examples/Neural_Networks/
```

## Data and paths

- Example data: **`examples/datasets/spam_email_dataset.csv`** (and sklearn-generated regression data built in-notebook).
- The notebook finds `spam_email_dataset.csv` by walking parents of `Path.cwd()` — keep the working directory at **repo root** or **`examples/`** when opening Jupyter from different folders.
- Prefer **`examples/load_dataset.py`** (`load_spam_dataset()`, `load_dataset(name)`) for portable paths.

## Tips

- Restart the kernel after changing **`postprocessing`** or network classes under **`src/rice_ml/`**.
- ROC metrics apply to **classification** outputs; regression sections should use regression metrics only.

## Related

- [Examples hub](../README.md)
- [Neural_Networks package README](../../src/rice_ml/Neural_Networks/README.md)
- [Processing helpers](../../src/rice_ml/processing/README.md)
