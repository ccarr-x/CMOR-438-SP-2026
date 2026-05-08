# Unsupervised Learning Module

This module contains custom implementations of unsupervised learning algorithms in `rice_ml`.

## Contents

- `k_means_clustering.py` - K-Means clustering (`random` and `kmeans++` init modes)
- `dbscan.py` - Density-based clustering (DBSCAN)
- `pca.py` - Principal Component Analysis for dimensionality reduction

## Usage

```python
from rice_ml.Unsupervised_Learning.k_means_clustering import KMeans
from rice_ml.Unsupervised_Learning.dbscan import DBSCAN
from rice_ml.Unsupervised_Learning.pca import PCA

# K-Means
kmeans = KMeans(n_clusters=3, task="kmeans++", random_state=42)
labels = kmeans.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.7, min_samples=5)
labels = dbscan.fit_predict(X)

# PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

## Related notebooks

See `examples/Unsupervised_Learning/`:

- `K_Means_Clustering.ipynb`
- `DBSCAN.ipynb`
- `PCA.ipynb`

## Tests

Unsupervised tests are located in `tests/unsupervised/`.
