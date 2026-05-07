# CMOR-438-SP-2026: Rice Machine Learning Library

A comprehensive machine learning library developed for CMOR 438 at Rice University, featuring custom implementations of popular algorithms and utilities for data science workflows.

## 🚀 Features

### Supervised Learning Algorithms
- **Random Forest** - Ensemble learning with decision trees
- **Decision Trees** - Tree-based classification and regression
- **Logistic Regression** - Linear classification algorithm
- **K-Nearest Neighbors** - Instance-based learning
- **Linear Regression** - Linear modeling for regression
- **Perceptron** - Binary linear classifier
- **Multi-layer Perceptron** - Neural network implementation
- **Single Neuron Model** - Basic neural network unit

### Neural Networks
- Custom neural network implementations
- Support for various architectures
- Training and evaluation utilities

### Data Processing
- **Preprocessing** - Data cleaning, transformation, and preparation
- **Postprocessing** - Model evaluation, metrics, and analysis
- Train-test splitting utilities
- Data standardization and normalization

### Visualization & Analysis
- Integrated plotting capabilities with matplotlib and seaborn
- Model performance visualization
- Feature analysis tools

## 📦 Installation

### Quick Install
```bash
pip install git+https://github.com/ccarr-x/CMOR-438-SP-2026.git
```

### Development Install
```bash
git clone https://github.com/ccarr-x/CMOR-438-SP-2026.git
cd CMOR-438-SP-2026
pip install -e .
```

### From PyPI (when published)
```bash
pip install rice-ml
```

## 🛠️ Requirements

- Python >= 3.8
- numpy >= 2.0
- matplotlib >= 3.5
- pandas >= 1.1
- scikit-learn >= 1.0
- seaborn >= 0.12
- mlxtend >= 0.23
- tabulate >= 0.9

## 📖 Quick Start

### Basic Usage

```python
from rice_ml.Supervised_Learning.ensemble_methods.random_forest_class import RandomForestClassifier
from rice_ml.processing.preprocessing import train_test_split
from rice_ml.processing.postprocessing import PostProcessor
import pandas as pd

# Load your data
df = pd.read_csv("your_dataset.csv")
X = df[features].values
y = df[target].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)

# Evaluate results
processor = PostProcessor()
metrics = processor.classification_metrics(y_test, predictions)
print(metrics)
```

### Available Modules

```python
# Classification algorithms
from rice_ml.Supervised_Learning.ensemble_methods.random_forest_class import RandomForestClassifier
from rice_ml.Supervised_Learning.decision_tree_class import DecisionTreeClassifier
from rice_ml.Supervised_Learning.logistic_regression_class import LogisticRegression
from rice_ml.Supervised_Learning.knn_class import KNeighborsClassifier
from rice_ml.Supervised_Learning.perceptron_class import Perceptron

# Regression algorithms
from rice_ml.Supervised_Learning.ensemble_methods.random_forest_class import RandomForestRegressor
from rice_ml.Supervised_Learning.linear_regression_class import LinearRegression

# Neural networks
from rice_ml.Supervised_Learning.multilayer_perceptron_class import MultiLayerPerceptron
from rice_ml.Supervised_Learning.single_neuron_model_class import SingleNeuronModel

# Processing utilities
from rice_ml.processing.preprocessing import train_test_split, standardize
from rice_ml.processing.postprocessing import PostProcessor
```

## 📚 Examples

The `examples/` directory contains comprehensive Jupyter notebooks demonstrating each algorithm:

- **Random Forest** - `/examples/Supervised_Learning/Ensemble_Methods/Random_Forest.ipynb`
- **Decision Trees** - `/examples/Supervised_Learning/Decision_Trees/`
- **K-Nearest Neighbors** - `/examples/Supervised_Learning/K_Nearest_Neighbors.ipynb`
- **Neural Networks** - `/examples/Neural_Networks/`
- **And many more...**

Each example includes:
- Algorithm implementation details
- Performance evaluation
- Visualization of results
- Comparison with scikit-learn baselines

## 🏗️ Project Structure

```
CMOR-438-SP-2026/
├── src/
│   └── rice_ml/
│       ├── Supervised_Learning/
│       │   ├── ensemble_methods/
│       │   ├── decision_tree_class.py
│       │   ├── logistic_regression_class.py
│       │   ├── knn_class.py
│       │   ├── linear_regression_class.py
│       │   ├── perceptron_class.py
│       │   ├── multilayer_perceptron_class.py
│       │   └── single_neuron_model_class.py
│       ├── Neural_Networks/
│       └── processing/
│           ├── preprocessing.py
│           └── postprocessing.py
├── examples/
│   ├── Supervervised_Learning/
│   ├── Unsupervised_Learning/
│   ├── Neural_Networks/
│   ├── datasets/
│   └── utils/
├── tests/
├── docs/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=rice_ml
```

## 📊 Performance

Our implementations are designed for educational purposes and include:
- Detailed algorithm implementations
- Performance comparisons with scikit-learn
- Comprehensive evaluation metrics
- Visualization tools for understanding algorithm behavior

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install in development mode: `pip install -e .`
4. Make your changes
5. Add tests for new functionality
6. Run tests: `pytest`
7. Submit a pull request

## 📖 Documentation

- [Installation Guide](README_INSTALLATION.md)
- [API Documentation](docs/api.md)
- [Examples Guide](examples/README.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 📄 License

This project is part of CMOR 438 at Rice University.

## 🙏 Acknowledgments

- Rice University CMOR 438 course
- Contributors and students who helped develop these implementations
- The scikit-learn team for inspiration and benchmark implementations

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the examples directory for usage patterns
- Review the API documentation

---

**Built with ❤️ for Rice University CMOR 438**
