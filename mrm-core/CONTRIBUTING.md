# Contributing to MRM Core

Thank you for your interest in contributing to MRM Core!

## Development Setup

1. Clone the repository:
```bash
git clone https/github.com/your-org/mrm-core
cd mrm-core
```

2. Install dependencies:
```bash
pip install poetry
poetry install
```

3. Run tests:
```bash
poetry run pytest
```

## Code Style

- We use Black for code formatting
- We use isort for import sorting
- We use mypy for type checking

Run formatting:
```bash
poetry run black mrm/
poetry run isort mrm/
```

## Adding New Tests

To add a new validation test:

1. Create test class inheriting from `MRMTest`:

```python
from mrm.tests.base import MRMTest, TestResult
from mrm.tests.library import register_test

@register_test
class MyCustomTest(MRMTest):
    name = "category.MyCustomTest"
    description = "Description of what this tests"
    tags = ["tag1", "tag2"]
    
    def run(self, model, dataset, **config):
        # Your test logic here
        return TestResult(
            passed=True,
            score=0.95,
            details={'key': 'value'}
        )
```

2. Add to appropriate module in `mrm/tests/builtin/`

3. Write tests for your test in `tests/`

4. Update documentation

## Adding Backend Adapters

To add a new backend:

1. Create adapter class inheriting from `BackendAdapter`:

```python
from mrm.backends.base import BackendAdapter

class MyBackend(BackendAdapter):
    def __init__(self, **config):
        super().__init__(**config)
        # Your initialization
    
    def register_model(self, model_config, model_artifact):
        # Implementation
        pass
    
    # Implement other required methods
```

2. Add to `mrm/backends/`

3. Register in `pyproject.toml`:

```toml
[tool.poetry.plugins."mrm.backends"]
mybackend = "mrm.backends.mybackend:MyBackend"
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions
- Update examples if adding new features

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Add tests
4. Update documentation
5. Run formatting and tests
6. Submit pull request

## Questions?

Open an issue or start a discussion on GitHub.
