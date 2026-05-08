# CMOR-438-SP-2026

Machine learning implementations for CMOR 438 at Rice University, with educational model code, notebooks, and tests.

## Quick Start

```bash
git clone https://github.com/ccarr-x/CMOR-438-SP-2026.git
cd CMOR-438-SP-2026
pip install -e .
```

Install dev tools (pytest) and run tests:

```bash
pip install -e ".[dev]"
pytest
```

Open examples:

```bash
jupyter notebook examples/
```

## Library Areas

- `src/rice_ml/Supervised_Learning/` - supervised models
- `src/rice_ml/Unsupervised_Learning/` - clustering and PCA
- `src/rice_ml/Neural_Networks/` - neural-network classes
- `src/rice_ml/processing/` - preprocessing/postprocessing utilities

## Example Notebooks

See `examples/README.md` for the full notebook index.

Common entry points:

- `examples/Supervised_Learning/Ensemble_Methods/Random_Forests.ipynb`
- `examples/Supervised_Learning/Decision_Trees.ipynb`
- `examples/Unsupervised_Learning/K_Means_Clustering.ipynb`
- `examples/Unsupervised_Learning/DBSCAN.ipynb`
- `examples/Unsupervised_Learning/PCA.ipynb`

## Project Structure

```text
CMOR-438-SP-2026/
├── src/rice_ml/
├── examples/
├── tests/
├── docs/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Testing

```bash
pytest tests/supervised
pytest tests/unsupervised
pytest tests/neural_networks
pytest tests/test_processing
```

## Documentation

- `docs/api.md`
- `docs/README.md`
- `examples/README.md`
- `CONTRIBUTING.md`

## Notes

- This project is educational and optimized for readability and learning.
- Some notebooks are exploratory and may include iterative cells.
