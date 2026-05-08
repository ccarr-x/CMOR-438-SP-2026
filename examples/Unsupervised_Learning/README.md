# Unsupervised learning examples

Notebooks for clustering and dimensionality reduction using **`rice_ml.Unsupervised_Learning`** (`KMeans`, DBSCAN, PCA, etc.).

## Notebooks

| Notebook | Focus |
|----------|--------|
| **`K_Means_Clustering.ipynb`** | K-means: fit, predict, cluster centers, typical visualizations. |
| **`DBSCAN.ipynb`** | Density-based clustering; noise points and parameter sensitivity. |
| **`PCA.ipynb`** | Principal components, variance explained, low-dimensional views or reconstruction ideas. |

## Prerequisites

- Python **3.9+**:

  ```bash
  pip install -e .
  ```

- Jupyter with the same interpreter as above.

## How to run

From the repository root:

```bash
jupyter notebook examples/Unsupervised_Learning/
```

## Data and paths

- Many notebooks use **synthetic or built-in arrays**; no shared CSV is required unless a cell loads one explicitly.
- For CSV examples elsewhere in the course, files live under **`examples/datasets/`**.
- **`examples/load_dataset.py`** can load any file from `examples/datasets/` using portable paths.

## Tips

- Restart the kernel after updating **`src/rice_ml/Unsupervised_Learning/`**.
- For **KMeans**, initialization depends on `task` (`"random"` vs `"kmeans++"`); see the class docstring and tests if behavior looks surprising.

## Related

- [Examples hub](../README.md)
- [Unsupervised package README](../../src/rice_ml/Unsupervised_Learning/README.md)
- [API reference](../../docs/api.md)
