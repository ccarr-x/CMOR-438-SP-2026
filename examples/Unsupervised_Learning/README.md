# Unsupervised Learning Examples

This folder contains notebooks demonstrating unsupervised algorithms in `rice_ml`.

## Notebooks

- `K_Means_Clustering.ipynb`
  - Demonstrates K-Means clustering and visualization
- `DBSCAN.ipynb`
  - Demonstrates density-based clustering and noise detection
- `PCA.ipynb`
  - Demonstrates dimensionality reduction and reconstruction

## Running notebooks

From project root:

```bash
jupyter notebook examples/Unsupervised_Learning/
```

If needed, install project dependencies first:

```bash
pip install -e .
```

## Dataset helpers

Use `examples/load_dataset.py` and files under `examples/datasets/` for local dataset loading.
