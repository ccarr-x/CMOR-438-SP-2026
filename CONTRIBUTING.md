# Contributing to CMOR-438-SP-2026

Thank you for your interest in contributing to the Rice Machine Learning Library! This document provides guidelines and instructions for contributors.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues**: Check if the issue has already been reported
2. **Use issue templates**: Use the appropriate template for bug reports or feature requests
3. **Provide detailed information**: Include steps to reproduce, expected vs actual behavior
4. **Include environment details**: Python version, OS, package versions

### Submitting Pull Requests

1. **Fork the repository**: Create a personal fork on GitHub
2. **Create a feature branch**: Use descriptive branch names
3. **Make your changes**: Follow coding standards and guidelines
4. **Add tests**: Ensure your changes are well-tested
5. **Update documentation**: Keep docs in sync with code changes
6. **Submit PR**: Create a pull request with clear description

## 🛠️ Development Setup

### Prerequisites

- Python >= 3.8
- Git
- GitHub account

### Setting Up Development Environment

1. **Clone your fork**:
```bash
git clone https://github.com/YOUR_USERNAME/CMOR-438-SP-2026.git
cd CMOR-438-SP-2026
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**:
```bash
pip install -e ".[dev]"
```

4. **Install pre-commit hooks** (optional but recommended):
```bash
pre-commit install
```

### Development Workflow

1. **Create a new branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the guidelines below

3. **Run tests**:
```bash
pytest
```

4. **Check code style**:
```bash
black src/
flake8 src/
```

5. **Commit your changes**:
```bash
git add .
git commit -m "feat: add new algorithm implementation"
```

6. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

7. **Create pull request** on GitHub

## 📝 Coding Standards

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **isort**: Import sorting
- **mypy**: Type checking (optional)

### Python Guidelines

1. **Follow PEP 8**: Standard Python style guide
2. **Use type hints**: Add type annotations for all functions
3. **Write docstrings**: Use Google-style docstrings
4. **Keep functions small**: Single responsibility principle
5. **Use meaningful names**: Descriptive variable and function names

### Example Code Structure

```python
from typing import Optional, Union
import numpy as np


class ExampleClass:
    """Brief description of the class.
    
    Longer description with details about the class purpose and usage.
    
    Attributes:
        param1: Description of parameter 1
        param2: Description of parameter 2
    """
    
    def __init__(self, param1: int, param2: Optional[str] = None):
        """Initialize the ExampleClass.
        
        Args:
            param1: Description of parameter 1
            param2: Description of parameter 2 (optional)
        """
        self.param1 = param1
        self.param2 = param2
    
    def example_method(self, data: np.ndarray) -> float:
        """Process the input data and return a result.
        
        Args:
            data: Input array to process
            
        Returns:
            Processed result
            
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        
        # Implementation here
        return float(np.mean(data))
```

## 🧪 Testing Guidelines

### Test Structure

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark critical functions
- **Example tests**: Ensure notebooks run correctly

### Writing Tests

1. **Use pytest**: Our testing framework
2. **Follow naming convention**: `test_*.py` files, `test_*` functions
3. **Test edge cases**: Include boundary conditions and error cases
4. **Use fixtures**: For common test setup
5. **Mock external dependencies**: When necessary

### Example Test

```python
import pytest
import numpy as np
from rice_ml.Supervised_Learning.example_module import ExampleClass


class TestExampleClass:
    """Test suite for ExampleClass."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.example = ExampleClass(param1=42)
    
    def test_initialization(self):
        """Test class initialization."""
        assert self.example.param1 == 42
        assert self.example.param2 is None
    
    def test_example_method(self):
        """Test the example method."""
        data = np.array([1, 2, 3, 4, 5])
        result = self.example.example_method(data)
        assert result == 3.0
    
    def test_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError):
            self.example.example_method("invalid")
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_example.py

# Run with coverage
pytest --cov=rice_ml --cov-report=html

# Run performance tests
pytest tests/performance/
```

## 📚 Documentation Guidelines

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int = 0) -> bool:
    """Brief description of the function.
    
    Detailed description explaining what the function does, any
    important considerations, and usage examples.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: 0)
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: If param1 is invalid
        TypeError: If types are incorrect
    
    Example:
        >>> result = function_name("test", 5)
        >>> print(result)
        True
    """
```

### Documentation Updates

1. **API docs**: Update `docs/api.md` for new functions/classes
2. **Examples**: Add notebook examples for new features
3. **README**: Update main README for major changes
4. **Changelog**: Maintain CHANGELOG.md for version history

## 🎯 Types of Contributions

### Bug Fixes

- **Small fixes**: Documentation typos, minor code issues
- **Major fixes**: Algorithmic errors, performance issues
- **Security fixes**: Vulnerability patches

### New Features

- **New algorithms**: Implement additional ML algorithms
- **Utilities**: Add helper functions and tools
- **Visualizations**: Enhanced plotting capabilities
- **Performance**: Optimization improvements

### Documentation

- **API docs**: Improve function/class documentation
- **Tutorials**: Create step-by-step guides
- **Examples**: Add new example notebooks
- **Translations**: Documentation in other languages

### Infrastructure

- **CI/CD**: Improve testing and deployment
- **Tools**: Development tooling and scripts
- **Templates**: Issue and PR templates

## 🔄 Pull Request Process

### Before Submitting

1. **Test thoroughly**: Ensure all tests pass
2. **Update docs**: Keep documentation current
3. **Check style**: Run code formatting tools
4. **Review changes**: Self-review your PR

### PR Description Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added tests for new functionality
- [ ] All existing tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process

1. **Automated checks**: CI/CD pipeline validation
2. **Peer review**: At least one maintainer approval
3. **Discussion**: Address feedback and questions
4. **Merge**: Once all requirements are met

## 🏷️ Release Process

### Version Management

We use semantic versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update version**: Modify `pyproject.toml`
2. **Update changelog**: Add release notes
3. **Tag release**: Create git tag
4. **Publish**: Upload to PyPI (when applicable)

## 🤖 Development Tools

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
```

### IDE Configuration

Recommended VS Code settings (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true
}
```

## 🎓 Educational Focus

This project is primarily educational. When contributing:

1. **Clarity over performance**: Prioritize understandable code
2. **Documentation**: Explain algorithms and implementations
3. **Examples**: Provide clear usage examples
4. **Comparisons**: Include scikit-learn benchmarks
5. **Learning resources**: Reference educational materials

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For general questions and ideas
- **Email**: For maintainers (see README.md)

### Resources

- [API Documentation](docs/api.md)
- [Examples Guide](examples/README.md)
- [Installation Guide](README_INSTALLATION.md)

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to the Rice Machine Learning Library! Your contributions help make machine learning education more accessible and comprehensive.
