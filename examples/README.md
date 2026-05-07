# Examples Directory

This directory contains comprehensive Jupyter notebooks demonstrating the use of the rice_ml library algorithms and utilities. Each example includes detailed explanations, code implementations, and performance evaluations.

## 📚 Table of Contents

- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Neural Networks](#neural-networks)
- [Datasets](#datasets)
- [Utilities](#utilities)

## Supervised Learning

### Ensemble Methods

#### 🌲 Random Forest
**File**: `Supervised_Learning/Ensemble_Methods/Random_Forest.ipynb`

**Overview**: Implementation and evaluation of Random Forest classifiers and regressors using ensemble learning.

**Contents**:
- Random Forest algorithm explanation
- Classification and regression tasks
- Performance comparison with scikit-learn
- Feature importance analysis
- Hyperparameter tuning examples

**Key Features**:
- Spam email dataset classification
- Ensemble learning visualization
- Out-of-bag error estimation
- Feature selection techniques

**Algorithms Demonstrated**:
- `RandomForestClassifier`
- `RandomForestRegressor`
- Ensemble voting mechanisms
- Bootstrap sampling

---

### Decision Trees

#### 🌳 Decision Tree Classifier
**File**: `Supervised_Learning/Decision_Trees/Decision_Tree_Classifier.ipynb`

**Overview**: Complete implementation of decision tree algorithm with visualization and analysis.

**Contents**:
- Decision tree theory and implementation
- Tree visualization techniques
- Pruning strategies
- Overfitting analysis
- Comparison with ensemble methods

**Key Features**:
- Interactive tree visualization
- Splitting criteria comparison (Gini vs Entropy)
- Depth optimization
- Cross-validation analysis

---

### Classification Algorithms

#### 🎯 Logistic Regression
**File**: `Supervised_Learning/Logistic_Regression.ipynb`

**Overview**: Binary classification using logistic regression with gradient descent optimization.

**Contents**:
- Logistic function and sigmoid activation
- Gradient descent implementation
- Regularization techniques
- Convergence analysis
- Multi-class extension

**Key Features**:
- Learning rate optimization
- Regularization strength tuning
- Decision boundary visualization
- Probability calibration

#### 👥 K-Nearest Neighbors
**File**: `Supervised_Learning/K_Nearest_Neighbors.ipynb`

**Overview**: Instance-based learning algorithm with distance metrics and optimization.

**Contents**:
- Distance metric comparison (Euclidean, Manhattan, Minkowski)
- K-value optimization
- Weighted vs uniform voting
- Computational complexity analysis
- Curse of dimensionality discussion

**Key Features**:
- Distance metric visualization
- K-value selection using cross-validation
- Computational efficiency analysis
- High-dimensional data handling

#### 📈 Linear Regression
**File**: `Supervised_Learning/Linear_Regression.ipynb`

**Overview**: Linear modeling for regression tasks with various optimization approaches.

**Contents**:
- Ordinary least squares solution
- Gradient descent implementation
- Regularization (Ridge, Lasso)
- Feature scaling importance
- Residual analysis

**Key Features**:
- Multiple optimization methods comparison
- Regularization path visualization
- Feature importance analysis
- Diagnostic plots

#### 🧠 Perceptron
**File**: `Supervised_Learning/The Perceptron/Perceptron.ipynb`

**Overview**: Binary linear classifier and foundation of neural networks.

**Contents**:
- Perceptron learning algorithm
- Convergence theorem
- Linear separability
- Multi-class extension
- Historical context

**Key Features**:
- Learning rule visualization
- Convergence analysis
- Decision boundary plotting
- Comparison with logistic regression

---

### Neural Networks

#### 🌐 Multi-layer Perceptron
**File**: `Supervised_Learning/Multi_Layer_Perceptron.ipynb`

**Overview**: Deep neural network implementation with backpropagation.

**Contents**:
- Feedforward architecture
- Backpropagation algorithm
- Activation functions comparison
- Weight initialization strategies
- Gradient descent variants

**Key Features**:
- Layer-by-layer visualization
- Activation function comparison
- Learning rate scheduling
- Overfitting prevention

#### 🔗 Single Neuron Model
**File**: `Supervised_Learning/Single_Neuron_Model.ipynb`

**Overview**: Basic neural network unit as foundation for deep learning.

**Contents**:
- Single neuron computation
- Activation functions
- Learning algorithms
- Biological inspiration
- Mathematical foundations

**Key Features**:
- Activation function visualization
- Learning rule comparison
- Convergence analysis
- Biological analogy

---

## Unsupervised Learning

### Clustering Algorithms

#### 🎨 K-Means Clustering
**File**: `Unsupervised_Learning/K_Means.ipynb`

**Overview**: Implementation of K-means clustering algorithm with convergence analysis.

**Contents**:
- K-means algorithm explanation
- Centroid initialization strategies
- Convergence criteria
- Elbow method for K selection
- Silhouette analysis

**Key Features**:
- Clustering visualization
- Convergence animation
- K-value optimization
- Real-world applications

#### 📊 PCA (Principal Component Analysis)
**File**: `Unsupervised_Learning/PCA.ipynb`

**Overview**: Dimensionality reduction technique for data compression and visualization.

**Contents**:
- Eigenvalue decomposition
- Variance explained analysis
- Component selection
- Data reconstruction
- Visualization techniques

**Key Features**:
- Eigenvalue spectrum visualization
- Component importance analysis
- 2D/3D data visualization
- Noise reduction examples

---

## Neural Networks

### Deep Learning Architectures

#### 🧠 Neural Network Fundamentals
**File**: `Neural_Networks/Neural_Network_Basics.ipynb`

**Overview**: Introduction to neural networks and deep learning concepts.

**Contents**:
- Neural network history
- Biological inspiration
- Mathematical foundations
- Training algorithms
- Applications overview

**Key Features**:
- Historical timeline
- Biological neuron analogy
- Mathematical formulations
- Real-world examples

#### 🔄 Backpropagation
**File**: `Neural_Networks/Backpropagation.ipynb`

**Overview**: Detailed implementation of backpropagation algorithm.

**Contents**:
- Chain rule application
- Gradient computation
- Weight updates
- Computational graphs
- Optimization techniques

**Key Features**:
- Gradient flow visualization
- Computational graph representation
- Optimization comparison
- Numerical stability analysis

---

## Datasets

### Available Datasets

#### 📧 Spam Email Dataset
**File**: `datasets/spam_email_dataset.csv`

**Description**: Email classification dataset for spam detection.

**Features**:
- Email metadata (sender, subject, timestamp)
- Content features (word counts, links, attachments)
- Sender reputation scores
- Binary spam/ham labels

**Usage Examples**:
- Random Forest classification
- Feature importance analysis
- Text preprocessing
- Model comparison

#### 🎗️ Breast Cancer Dataset
**File**: `datasets/Breast_cancer_Research (1).csv`

**Description**: Medical dataset for breast cancer classification.

**Features**:
- Clinical measurements
- Patient demographics
- Diagnostic features
- Binary classification labels

**Usage Examples**:
- Medical diagnosis
- Feature selection
- Model interpretability
- Cross-validation

#### 📱 Smartphone Addiction Dataset
**File**: `datasets/Smartphone_Addiction_Dataset.csv`

**Description**: Behavioral dataset for addiction level prediction.

**Features**:
- Usage patterns
- Psychological metrics
- Demographic information
- Multi-class labels

**Usage Examples**:
- Multi-class classification
- Behavioral analysis
- Feature engineering
- Regression tasks

---

## Utilities

### Helper Functions

#### 📁 Dataset Loading
**File**: `utils/dataset_utils.py`

**Description**: Robust dataset loading utilities for cross-platform compatibility.

**Features**:
- Automatic path resolution
- Multiple fallback strategies
- Error handling and reporting
- Cross-platform compatibility

**Usage**:
```python
from utils.dataset_utils import load_dataset
df = load_dataset("spam_email_dataset.csv")
```

#### 📦 Package Installation
**File**: `utils/repo_imports.py`

**Description**: Automatic package installation and import management.

**Features**:
- Git-based installation
- Local development mode
- Dependency management
- Import error handling

**Usage**:
```python
from utils.repo_imports import setup_rice_ml_imports
success, modules, message = setup_rice_ml_imports()
```

---

## 🚀 Getting Started

### Prerequisites

1. Install the rice_ml package:
```bash
pip install git+https://github.com/ccarr-x/CMOR-438-SP-2026.git
```

2. Navigate to the desired example directory:
```bash
cd examples/Supervised_Learning/Ensemble_Methods/
```

3. Launch Jupyter notebook:
```bash
jupyter notebook Random_Forest.ipynb
```

### Running Examples

Each notebook is self-contained and includes:
- Required imports and setup
- Data loading and preprocessing
- Algorithm implementation
- Results visualization
- Performance evaluation

### Customization

- **Datasets**: Replace with your own data using the same format
- **Parameters**: Experiment with different hyperparameters
- **Algorithms**: Compare different approaches on the same data
- **Visualizations**: Customize plots and analysis

---

## 📊 Performance Benchmarks

All examples include performance comparisons with scikit-learn implementations:

- **Accuracy**: Classification accuracy comparison
- **Speed**: Training and prediction time analysis
- **Memory**: Memory usage profiling
- **Scalability**: Performance on different dataset sizes

Results are summarized in each notebook with detailed analysis.

---

## 🤝 Contributing

### Adding New Examples

1. Create a new notebook in the appropriate directory
2. Follow the established template structure
3. Include comprehensive documentation
4. Add performance comparisons
5. Update this README file

### Example Template

Each example notebook should include:
- **Introduction**: Algorithm overview and objectives
- **Implementation**: Code with detailed comments
- **Experiments**: Parameter exploration and analysis
- **Results**: Performance evaluation and visualization
- **Conclusion**: Summary and insights

---

## 📚 Additional Resources

- [API Documentation](../docs/api.md)
- [Installation Guide](../README_INSTALLATION.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Main Project README](../README.md)

---

**Built for educational purposes at Rice University CMOR 438**
