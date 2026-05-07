# rice_ml Package

The core machine learning library package containing all algorithms and utilities.

## 📁 Package Structure

```
rice_ml/
├── Supervised_Learning/
│   ├── ensemble_methods/
│   │   ├── __init__.py
│   │   └── random_forest_class.py
│   ├── decision_tree_class.py
│   ├── logistic_regression_class.py
│   ├── knn_class.py
│   ├── linear_regression_class.py
│   ├── perceptron_class.py
│   ├── multilayer_perceptron_class.py
│   └── single_neuron_model_class.py
├── Neural_Networks/
│   └── [neural network implementations]
├── processing/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── postprocessing.py
└── __init__.py
```

## 🚀 Quick Import

```python
# Import specific algorithms
from rice_ml.Supervised_Learning.ensemble_methods.random_forest_class import RandomForestClassifier
from rice_ml.processing.preprocessing import train_test_split
from rice_ml.processing.postprocessing import PostProcessor

# Or import entire package
import rice_ml
```

## 📚 Module Documentation

### Supervised Learning

#### Ensemble Methods
- **RandomForestClassifier**: Ensemble learning with decision trees
- **RandomForestRegressor**: Regression version of random forest

#### Classification Algorithms
- **DecisionTreeClassifier**: Tree-based classification
- **LogisticRegression**: Binary classification with gradient descent
- **KNeighborsClassifier**: Instance-based learning
- **Perceptron**: Linear binary classifier

#### Regression Algorithms
- **LinearRegression**: Linear modeling with gradient descent
- **RandomForestRegressor**: Ensemble regression

#### Neural Networks
- **MultiLayerPerceptron**: Deep neural network
- **SingleNeuronModel**: Basic neural network unit

### Processing

#### Preprocessing
- **train_test_split**: Data splitting utilities
- **standardize**: Feature standardization
- **normalize**: Data normalization

#### Postprocessing
- **PostProcessor**: Model evaluation and metrics
- **classification_metrics**: Classification performance measures
- **regression_metrics**: Regression performance measures

## 🔧 Usage Examples

### Basic Classification

```python
from rice_ml.Supervised_Learning.ensemble_methods.random_forest_class import RandomForestClassifier
from rice_ml.processing.preprocessing import train_test_split
from rice_ml.processing.postprocessing import PostProcessor

# Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
processor = PostProcessor()
predictions = rf.predict(X_test)
metrics = processor.classification_metrics(y_test, predictions)
```

### Data Processing Pipeline

```python
from rice_ml.processing.preprocessing import standardize
from rice_ml.processing.postprocessing import PostProcessor

# Standardize features
X_scaled = standardize(X_train)

# Train model
model.fit(X_scaled, y_train)

# Evaluate results
processor = PostProcessor()
accuracy = processor.accuracy(y_test, predictions)
```

## 🎯 Design Principles

1. **Educational Focus**: Clear, understandable implementations
2. **Consistent API**: Follows scikit-learn conventions
3. **Comprehensive**: Covers major ML algorithms
4. **Well-Documented**: Detailed docstrings and examples
5. **Testable**: Unit tests for all components

## 🔗 Dependencies

- numpy: Numerical computations
- pandas: Data manipulation
- matplotlib: Visualization
- scikit-learn: Comparison and utilities
- seaborn: Statistical visualization

## 📖 API Reference

For detailed API documentation, see [docs/api.md](../../docs/api.md).

## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

## 🤝 Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.
