# Supervised Learning Module

Comprehensive implementations of supervised learning algorithms for classification and regression tasks.

## 📁 Module Structure

```
Supervised_Learning/
├── ensemble_methods/
│   ├── __init__.py
│   └── random_forest_class.py
├── decision_tree_class.py
├── logistic_regression_class.py
├── knn_class.py
├── linear_regression_class.py
├── perceptron_class.py
├── multilayer_perceptron_class.py
├── single_neuron_model_class.py
└── __init__.py
```

## 🌲 Ensemble Methods

### Random Forest

**File**: `ensemble_methods/random_forest_class.py`

**Classes**:
- `RandomForestClassifier`: Classification using decision tree ensembles
- `RandomForestRegressor`: Regression using decision tree ensembles

**Key Features**:
- Bootstrap sampling for training each tree
- Feature randomness for diversity
- Voting/averaging for predictions
- Out-of-bag error estimation

**Usage**:
```python
from rice_ml.Supervised_Learning.ensemble_methods.random_forest_class import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
```

## 🌳 Decision Trees

### Decision Tree Classifier

**File**: `decision_tree_class.py`

**Class**: `DecisionTreeClassifier`

**Key Features**:
- Recursive binary splitting
- Gini impurity or entropy criteria
- Pruning for overfitting prevention
- Tree visualization support

**Usage**:
```python
from rice_ml.Supervised_Learning.decision_tree_class import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=10, criterion='gini', random_state=42)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
```

## 📈 Classification Algorithms

### Logistic Regression

**File**: `logistic_regression_class.py`

**Class**: `LogisticRegression`

**Key Features**:
- Gradient descent optimization
- L2 regularization support
- Multi-class extension (one-vs-rest)
- Probability predictions

**Usage**:
```python
from rice_ml.Supervised_Learning.logistic_regression_class import LogisticRegression

lr = LogisticRegression(learning_rate=0.01, n_iterations=1000, regularization=0.1)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
probabilities = lr.predict_proba(X_test)
```

### K-Nearest Neighbors

**File**: `knn_class.py`

**Class**: `KNeighborsClassifier`

**Key Features**:
- Multiple distance metrics (Euclidean, Manhattan, Minkowski)
- Weighted voting options
- Efficient k-nearest neighbor search
- Cross-validation for k selection

**Usage**:
```python
from rice_ml.Supervised_Learning.knn_class import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

### Perceptron

**File**: `perceptron_class.py`

**Class**: `Perceptron`

**Key Features**:
- Online learning algorithm
- Linear separability detection
- Convergence guarantee for linearly separable data
- Foundation for neural networks

**Usage**:
```python
from rice_ml.Supervised_Learning.perceptron_class import Perceptron

perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
perceptron.fit(X_train, y_train)
predictions = perceptron.predict(X_test)
```

## 📊 Regression Algorithms

### Linear Regression

**File**: `linear_regression_class.py`

**Class**: `LinearRegression`

**Key Features**:
- Ordinary least squares solution
- Gradient descent optimization
- Regularization options (Ridge, Lasso)
- Feature scaling support

**Usage**:
```python
from rice_ml.Supervised_Learning.linear_regression_class import LinearRegression

lr = LinearRegression(fit_intercept=True, learning_rate=0.01, n_iterations=1000)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
score = lr.score(X_test, y_test)  # R² score
```

### Random Forest Regressor

**File**: `ensemble_methods/random_forest_class.py`

**Class**: `RandomForestRegressor`

**Key Features**:
- Ensemble of decision trees for regression
- Bootstrap aggregating
- Feature importance estimation
- Robust to outliers

**Usage**:
```python
from rice_ml.Supervised_Learning.ensemble_methods.random_forest_class import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_reg.fit(X_train, y_train)
predictions = rf_reg.predict(X_test)
```

## 🧠 Neural Networks

### Multi-layer Perceptron

**File**: `multilayer_perceptron_class.py`

**Class**: `MultiLayerPerceptron`

**Key Features**:
- Deep neural network architecture
- Backpropagation training
- Multiple activation functions
- Batch gradient descent

**Usage**:
```python
from rice_ml.Supervised_Learning.multilayer_perceptron_class import MultiLayerPerceptron

mlp = MultiLayerPerceptron(
    hidden_layers=[64, 32],
    activation='relu',
    learning_rate=0.01,
    n_iterations=1000
)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
```

### Single Neuron Model

**File**: `single_neuron_model_class.py`

**Class**: `SingleNeuronModel`

**Key Features**:
- Basic neural network unit
- Various activation functions
- Gradient-based learning
- Foundation for complex networks

**Usage**:
```python
from rice_ml.Supervised_Learning.single_neuron_model_class import SingleNeuronModel

neuron = SingleNeuronModel(activation='sigmoid', learning_rate=0.01)
neuron.fit(X_train, y_train)
predictions = neuron.predict(X_test)
```

## 🎯 Common Interface

All supervised learning algorithms follow a consistent interface:

### Core Methods

- `fit(X, y)`: Train the model on training data
- `predict(X)`: Make predictions on new data
- `score(X, y)`: Evaluate model performance (where applicable)

### Common Parameters

- `random_state`: Random seed for reproducibility
- `learning_rate`: Learning rate for gradient-based methods
- `n_iterations`: Number of training iterations
- `max_depth`: Maximum depth for tree-based methods

## 📊 Performance Comparison

All implementations include performance comparisons with scikit-learn:

- **Accuracy**: Prediction accuracy comparison
- **Speed**: Training and prediction time
- **Memory**: Memory usage analysis
- **Scalability**: Performance on different dataset sizes

## 🔧 Best Practices

### Data Preparation

1. **Feature scaling**: Standardize/normalize features
2. **Train-test split**: Use proper data splitting
3. **Cross-validation**: For hyperparameter tuning
4. **Missing values**: Handle appropriately

### Model Selection

1. **Start simple**: Begin with linear models
2. **Consider data size**: Choose algorithms based on dataset size
3. **Interpretability**: Trade-off between performance and interpretability
4. **Computational resources**: Consider training time and memory

### Hyperparameter Tuning

1. **Grid search**: Systematic parameter exploration
2. **Random search**: Random parameter sampling
3. **Cross-validation**: Robust performance estimation
4. **Learning curves**: Analyze model behavior

## 🧪 Testing

Each algorithm includes comprehensive tests:

- **Unit tests**: Individual component testing
- **Integration tests**: End-to-end workflow
- **Performance tests**: Benchmark against scikit-learn
- **Edge cases**: Handle unusual inputs

Run tests with:
```bash
pytest tests/test_supervised_learning/
```

## 📚 Examples

See the [examples directory](../../examples/Supervised_Learning/) for detailed usage examples:

- [Random Forest](../../examples/Supervised_Learning/Ensemble_Methods/Random_Forest.ipynb)
- [Decision Trees](../../examples/Supervised_Learning/Decision_Trees/)
- [K-Nearest Neighbors](../../examples/Supervised_Learning/K_Nearest_Neighbors.ipynb)
- [Logistic Regression](../../examples/Supervised_Learning/Logistic_Regression.ipynb)

## 🔗 Related Modules

- **Processing**: Data preprocessing and postprocessing utilities
- **Neural Networks**: Advanced neural network architectures
- **Unsupervised Learning**: Clustering and dimensionality reduction

## 📖 API Reference

For detailed API documentation, see [docs/api.md](../../../docs/api.md).
