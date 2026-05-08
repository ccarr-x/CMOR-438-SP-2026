# Test Suite Guide

This directory contains unit tests for the `rice_ml` package.

## Structure

- `supervised/` - Tests for supervised learning models
- `unsupervised/` - Tests for clustering and PCA
- `neural_networks/` - Tests for neural network implementations
- `test_processing/` - Tests for preprocessing and postprocessing helpers

## Prerequisites

Install the package and pytest:

```bash
pip install -e ".[dev]"
```

Setting `PYTHONPATH=src` is optional if you installed editable (`pip install -e .`), since imports resolve via site-packages.

Some neural preprocessing tests require TensorFlow (`pip install tensorflow`); those tests skip automatically if it is not installed.

## Run all tests

```bash
PYTHONPATH=src pytest
```

Many supervised tests still target older sklearn-like APIs (`fit`, `{0,1}` labels, attribute names). Updating those tests to match the current `rice_ml` APIs remains **work in progress**; processing and neural-network tests listed below are aligned.

## Known-good subsets

```bash
PYTHONPATH=src pytest tests/test_processing tests/neural_networks
PYTHONPATH=src pytest tests/unsupervised/test_k_means_clustering.py tests/unsupervised/test_pca.py tests/unsupervised/test_dbscan.py
```

## Run by area

```bash
pytest tests/supervised
pytest tests/unsupervised
pytest tests/neural_networks
pytest tests/test_processing
```

## Coverage (optional)

```bash
pytest --cov=rice_ml
```
