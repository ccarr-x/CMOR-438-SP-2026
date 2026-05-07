# API Documentation

This document provides detailed API documentation for all modules and classes in the rice_ml library.

## Table of Contents

- [Supervised Learning](#supervised-learning)
  - [Random Forest](#random-forest)
  - [Decision Trees](#decision-trees)
  - [Logistic Regression](#logistic-regression)
  - [K-Nearest Neighbors](#k-nearest-neighbors)
  - [Linear Regression](#linear-regression)
  - [Perceptron](#perceptron)
  - [Multi-layer Perceptron](#multi-layer-perceptron)
  - [Single Neuron Model](#single-neuron-model)
- [Processing](#processing)
  - [Preprocessing](#preprocessing)
  - [Postprocessing](#postprocessing)
- [Neural Networks](#neural-networks)

## Supervised Learning

### Random Forest

#### `RandomForestClassifier`

A random forest classifier using ensemble learning with decision trees.

```python
from rice_ml.Supervised_Learning.ensemble_methods.random_forest_class import RandomForestClassifier
```

**Parameters:**
- `n_estimators` (int, default=100): Number of trees in the forest
- `max_depth` (int, default=None): Maximum depth of the trees
- `min_samples_split` (int, default=2): Minimum samples required to split a node
- `min_samples_leaf` (int, default=1): Minimum samples required at leaf node
- `criterion` (str, default='gini'): Splitting criterion ('gini' or 'entropy')
- `random_state` (int, default=None): Random seed for reproducibility

**Methods:**
- `fit(X, y)`: Train the classifier
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get class probabilities

**Example:**
```python
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
```

#### `RandomForestRegressor`

A random forest regressor for continuous target variables.

```python
from rice_ml.Supervised_Learning.ensemble_methods.random_forest_class import RandomForestRegressor
```

**Parameters:** Same as `RandomForestClassifier`

**Methods:** Same as `RandomForestClassifier`

### Decision Trees

#### `DecisionTreeClassifier`

A decision tree classifier for binary and multiclass classification.

```python
from rice_ml.Supervised_Learning.decision_tree_class import DecisionTreeClassifier
```

**Parameters:**
- `max_depth` (int, default=None): Maximum depth of the tree
- `min_samples_split` (int, default=2): Minimum samples required to split
- `min_samples_leaf` (int, default=1): Minimum samples at leaf nodes
- `criterion` (str, default='gini'): Splitting criterion
- `random_state` (int, default=None): Random seed

**Methods:**
- `fit(X, y)`: Train the decision tree
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get class probabilities

### Logistic Regression

#### `LogisticRegression`

Binary classification using logistic regression.

```python
from rice_ml.Supervised_Learning.logistic_regression_class import LogisticRegression
```

**Parameters:**
- `learning_rate` (float, default=0.01): Learning rate for gradient descent
- `n_iterations` (int, default=1000): Number of iterations
- `regularization` (float, default=0.0): L2 regularization strength
- `random_state` (int, default=None): Random seed

**Methods:**
- `fit(X, y)`: Train the logistic regression model
- `predict(X)`: Make binary predictions
- `predict_proba(X)`: Get probability predictions

### K-Nearest Neighbors

#### `KNeighborsClassifier`

Classification based on k-nearest neighbors algorithm.

```python
from rice_ml.Supervised_Learning.knn_class import KNeighborsClassifier
```

**Parameters:**
- `n_neighbors` (int, default=5): Number of neighbors to consider
- `weights` (str, default='uniform'): Weight function ('uniform' or 'distance')
- `metric` (str, default='euclidean'): Distance metric
- `algorithm` (str, default='auto'): Algorithm for nearest neighbors

**Methods:**
- `fit(X, y)`: Store training data
- `predict(X)`: Make predictions
- `kneighbors(X, n_neighbors=None)`: Find k-neighbors

### Linear Regression

#### `LinearRegression`

Linear regression for continuous target prediction.

```python
from rice_ml.Supervised_Learning.linear_regression_class import LinearRegression
```

**Parameters:**
- `fit_intercept` (bool, default=True): Whether to calculate intercept
- `normalize` (bool, default=False): Whether to normalize features
- `learning_rate` (float, default=0.01): Learning rate for gradient descent
- `n_iterations` (int, default=1000): Number of iterations

**Methods:**
- `fit(X, y)`: Train the linear regression model
- `predict(X)`: Make predictions
- `score(X, y)`: Calculate R² score

### Perceptron

#### `Perceptron`

Binary linear classifier using perceptron algorithm.

```python
from rice_ml.Supervised_Learning.perceptron_class import Perceptron
```

**Parameters:**
- `learning_rate` (float, default=0.01): Learning rate
- `n_iterations` (int, default=1000): Number of iterations
- `random_state` (int, default=None): Random seed

**Methods:**
- `fit(X, y)`: Train the perceptron
- `predict(X)`: Make binary predictions
- `predict_proba(X)`: Get confidence scores

### Multi-layer Perceptron

#### `MultiLayerPerceptron`

Neural network with hidden layers for classification/regression.

```python
from rice_ml.Supervised_Learning.multilayer_perceptron_class import MultiLayerPerceptron
```

**Parameters:**
- `hidden_layers` (list, default=[100]): List of hidden layer sizes
- `activation` (str, default='relu'): Activation function
- `learning_rate` (float, default=0.01): Learning rate
- `n_iterations` (int, default=1000): Number of iterations
- `batch_size` (int, default=32): Batch size for training
- `random_state` (int, default=None): Random seed

**Methods:**
- `fit(X, y)`: Train the neural network
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get probabilities (classification)

### Single Neuron Model

#### `SingleNeuronModel`

Basic neural network unit with one neuron.

```python
from rice_ml.Supervised_Learning.single_neuron_model_class import SingleNeuronModel
```

**Parameters:**
- `activation` (str, default='sigmoid'): Activation function
- `learning_rate` (float, default=0.01): Learning rate
- `n_iterations` (int, default=1000): Number of iterations

**Methods:**
- `fit(X, y)`: Train the single neuron
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get probability outputs

## Processing

### Preprocessing

#### `train_test_split`

Split data into training and testing sets.

```python
from rice_ml.processing.preprocessing import train_test_split
```

**Parameters:**
- `X` (array-like): Features
- `y` (array-like): Target variable
- `test_size` (float, default=0.2): Proportion of test set
- `random_state` (int, default=None): Random seed
- `stratify` (array-like, default=None): Stratification target

**Returns:**
- `X_train, X_test, y_train, y_test`: Split datasets

**Example:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### `standardize`

Standardize features by removing mean and scaling to unit variance.

```python
from rice_ml.processing.preprocessing import standardize
```

**Parameters:**
- `X` (array-like): Input features
- `fit` (bool, default=True): Whether to fit scaler

**Returns:**
- `X_scaled`: Standardized features

### Postprocessing

#### `PostProcessor`

Class for evaluating model predictions and computing metrics.

```python
from rice_ml.processing.postprocessing import PostProcessor
```

**Methods:**

##### `classification_metrics(y_true, y_pred, average='weighted')`

Compute classification metrics.

**Parameters:**
- `y_true` (array-like): True labels
- `y_pred` (array-like): Predicted labels
- `average` (str): Averaging method for multi-class

**Returns:**
- `dict`: Dictionary containing:
  - `classification_error`: Error rate
  - `precision`: Precision score
  - `confusion_matrix`: Confusion matrix

##### `accuracy(y_true, y_pred)`

Calculate accuracy score.

**Parameters:**
- `y_true` (array-like): True labels
- `y_pred` (array-like): Predicted labels

**Returns:**
- `float`: Accuracy score

##### `regression_metrics(y_true, y_pred)`

Compute regression metrics.

**Parameters:**
- `y_true` (array-like): True values
- `y_pred` (array-like): Predicted values

**Returns:**
- `dict`: Dictionary containing:
  - `mae`: Mean absolute error
  - `mse`: Mean squared error
  - `rmse`: Root mean squared error
  - `r2`: R² score
  - Additional statistical measures

##### `precision(y_true, y_pred, average='weighted')`

Calculate precision score.

##### `recall(y_true, y_pred, average='weighted')`

Calculate recall score.

##### `f1_score(y_true, y_pred, average='weighted')`

Calculate F1 score.

##### `r2_score(y_true, y_pred)`

Calculate R² score for regression.

## Neural Networks

The neural network modules provide various architectures and training algorithms for deep learning tasks.

### Available Networks

- **Feedforward Networks**: Basic multi-layer perceptrons
- **Activation Functions**: Sigmoid, ReLU, tanh, etc.
- **Optimization Algorithms**: Gradient descent, Adam, etc.
- **Regularization**: Dropout, L2 regularization

### Usage Pattern

```python
from rice_ml.Neural_Networks import FeedforwardNetwork
from rice_ml.processing.preprocessing import standardize

# Prepare data
X_scaled = standardize(X_train)

# Create network
network = FeedforwardNetwork(
    layers=[X.shape[1], 64, 32, 1],
    activation='relu',
    learning_rate=0.01
)

# Train
network.fit(X_scaled, y_train, epochs=100)

# Predict
predictions = network.predict(X_test_scaled)
```

## Error Handling

All classes include proper error handling and validation:

- Input validation for array shapes and types
- Meaningful error messages
- Graceful handling of edge cases
- Consistent return types

## Performance Considerations

- Algorithms are optimized for educational clarity over raw performance
- Memory efficient implementations where possible
- Parallel processing support where applicable
- Compatibility with numpy arrays for speed

## Integration with Scikit-learn

All major classes follow scikit-learn's API conventions:

- `fit(X, y)` for training
- `predict(X)` for predictions
- `score(X, y)` for evaluation
- Consistent parameter naming

This makes it easy to swap between rice_ml and scikit-learn implementations for comparison.
