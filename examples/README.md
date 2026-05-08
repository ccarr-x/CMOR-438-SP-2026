# Examples directory

**Jupyter notebooks** that demonstrate how to install, import, and run **`rice_ml`** for CMOR 438–style coursework: supervised models, unsupervised methods, and neural-network classes.

---

## Quick start

From the **repository root** (so imports and dataset discovery behave consistently):

```bash
pip install -e .
jupyter notebook examples/
```

Optional dev tooling (pytest, etc.):

```bash
pip install -e ".[dev]"
```

---

## Folder guide

Each subfolder has its own **README** with notebook tables and folder-specific notes:

| Folder | README | Topics |
|--------|--------|--------|
| **`Supervised_Learning/`** | [README](Supervised_Learning/README.md) | Trees, linear/logistic regression, k-NN, perceptron, MLP, ensembles |
| **`Supervised_Learning/Ensemble_Methods/`** | [README](Supervised_Learning/Ensemble_Methods/README.md) | Random forest notebooks |
| **`Unsupervised_Learning/`** | [README](Unsupervised_Learning/README.md) | K-means, DBSCAN, PCA |
| **`Neural_Networks/`** | [README](Neural_Networks/README.md) | `SigmoidNeuralNetwork`, `DenseNetwork`, processing metrics |

---

## Notebook index (short)

### Supervised learning

- `Supervised_Learning/Decision_Trees.ipynb`
- `Supervised_Learning/Decision_Tree_Classifier.ipynb`
- `Supervised_Learning/Decision_Tree_Regressor.ipynb`
- `Supervised_Learning/Logistic_Regression.ipynb`
- `Supervised_Learning/Linear_Regression.ipynb`
- `Supervised_Learning/K_Nearest_Neighbors.ipynb`
- `Supervised_Learning/The_Perceptron.ipynb`
- `Supervised_Learning/The_Multilayer_Perceptron.ipynb`
- `Supervised_Learning/Ensemble_Methods/Random_Forest.ipynb`
- `Supervised_Learning/Ensemble_Methods/Random_Forests.ipynb`

### Unsupervised learning

- `Unsupervised_Learning/K_Means_Clustering.ipynb`
- `Unsupervised_Learning/DBSCAN.ipynb`
- `Unsupervised_Learning/PCA.ipynb`

### Neural networks

- `Neural_Networks/Neural_Network.ipynb`

---

## Data files

| File | Typical use |
|------|-------------|
| `datasets/spam_email_dataset.csv` | Email / spam classification examples |
| `datasets/Breast_Cancer_Research.csv` | Tabular classification demos |

**Portable loading:** use **`load_dataset.py`** (`load_spam_dataset()`, `load_dataset("filename.csv")`) so paths work on any machine. Notebooks that search for CSVs relative to the working directory run most reliably when the notebook **cwd** is the repo root or `examples/`.

---

## Helpers

- **`load_dataset.py`** — resolves `examples/datasets/` relative to this file (good for grading machines and other clones).

---

## Conventions & troubleshooting

- **Kernel restart** after editing anything under **`src/rice_ml/`** before re-running imports.
- Use one **Python environment** for both `pip install -e .` and Jupyter (mismatched kernels often cause “wrong” or stale `rice_ml`).
- Slow first import of plotting code: matplotlib may build a font cache; setting `MPLCONFIGDIR` to a writable directory can help in CI or restricted home folders.

---

## Related documentation

- [Project README](../README.md)
- [API reference](../docs/api.md)
- [Contributing guide](../CONTRIBUTING.md)
- [Tests README](../tests/README.md)
