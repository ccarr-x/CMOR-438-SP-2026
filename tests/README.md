# Test Suite Guide

This directory contains unit tests for the `rice_ml` package.

## Structure

- `supervised/` - Tests for supervised learning models
- `unsupervised/` - Tests for clustering and PCA
- `neural_networks/` - Tests for neural network implementations
- `test_processing/` - Tests for preprocessing and postprocessing helpers

## Run all tests

```bash
pytest
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
