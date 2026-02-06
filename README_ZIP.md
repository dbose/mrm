# MRM Core - Complete Package

## What's Included

This zip file contains the complete MRM Core framework with all features implemented.

## Contents

###  Core Framework (~3,500 lines of Python)

**Main Package** (`mrm/`)
- `cli/` - Command-line interface (Typer-based)
- `core/` - Project management, DAG, catalog, references
- `backends/` - Storage backends (local, MLflow)
- `tests/` - Test framework and built-in tests
- `engine/` - Test execution engine
- `utils/` - Utilities

###  Documentation (9 files)

1. **README.md** - Main overview and features
2. **GETTING_STARTED.md** - Step-by-step usage guide
3. **QUICKSTART.md** - Quick reference
4. **ARCHITECTURE.md** - Technical architecture (12KB)
5. **MODEL_REFERENCES.md** - ref() and catalog guide
6. **REFERENCE_TYPES.md** - All model source types
7. **DAG_FEATURES.md** - Dependency graph features
8. **COMPLETE_FEATURES.md** - Full feature summary
9. **PROJECT_SUMMARY.md** - Project overview

###  Examples (2 working examples)

- `examples/example_usage.py` - Basic usage demo
- `examples/dag_example.py` - DAG and dependencies demo
- `credit_risk_example/` - Generated example project

###  Configuration

- `pyproject.toml` - Poetry configuration
- `setup.py` - Setuptools configuration
- `Makefile` - Development tasks
- `LICENSE` - Apache 2.0 license
- `.gitignore` - Git ignore patterns

## Features

###  Complete dbt-Style Functionality

- **DAG Management** - Model dependencies with `depends_on`
- **ref()** - Reference models by name
- **Graph Operators** - `+model+`, `model+`, `+model`
- **Topological Sort** - Automatic dependency ordering
- **Parallel Execution** - Run independent models concurrently

###  Model Sources

- **Local Files** - pickle, joblib
- **Python Classes** - Import and instantiate
- **MLflow** - Model registry integration
- **HuggingFace** - Direct Hub integration
- **S3/Cloud** - Cloud storage support
- **Model References** - ref() to other models
- **Catalogs** - Internal model registries

###  Testing Framework

- **10 Built-in Tests**
  - Dataset: MissingValues, ClassImbalance, OutlierDetection, FeatureDistribution
  - Model: Accuracy, ROCAUC, Gini, Precision, Recall, F1Score
- **Custom Tests** - Easy plugin system
- **Test Suites** - Reusable test collections
- **Parallel Execution** - Multi-threaded test runner

###  CLI Commands

```bash
mrm init project-name       # Initialize project
mrm test                    # Run tests
mrm test --select +model    # Graph operators
mrm list models             # List resources
mrm debug --show-dag        # Show dependency graph
```

## Quick Start

### 1. Extract the Zip

```bash
unzip mrm-core.zip
cd mrm-core
```

### 2. Install

```bash
# Option 1: Development mode
pip install -e .

# Option 2: With Poetry
pip install poetry
poetry install
```

### 3. Run Example

```bash
# Basic example
python examples/example_usage.py

# DAG example
python examples/dag_example.py
```

### 4. Create Your Project

```bash
mrm init my-models
cd my-models
mrm test
```

## File Structure

```
mrm-core/
├── README.md                    # Main documentation
├── GETTING_STARTED.md          # Usage guide
├── ARCHITECTURE.md             # Technical details
├── MODEL_REFERENCES.md         # ref() guide
├── REFERENCE_TYPES.md          # Model sources
├── DAG_FEATURES.md             # DAG functionality
├── COMPLETE_FEATURES.md        # Feature summary
├── PROJECT_SUMMARY.md          # Project overview
├── QUICKSTART.md               # Quick reference
├── DELIVERY_SUMMARY.md         # Delivery notes
│
├── pyproject.toml              # Poetry config
├── setup.py                    # Setuptools config
├── Makefile                    # Dev tasks
├── LICENSE                     # Apache 2.0
├── CONTRIBUTING.md             # How to contribute
│
├── mrm/                        # Main package
│   ├── __init__.py
│   ├── cli/                    # CLI interface
│   │   ├── __init__.py
│   │   └── main.py
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── project.py          # Project management
│   │   ├── dag.py              # Dependency graph
│   │   ├── catalog.py          # Model catalog
│   │   ├── references.py       # Model references
│   │   └── init.py             # Project init
│   ├── backends/               # Storage backends
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── local.py
│   │   └── mlflow.py
│   ├── tests/                  # Test framework
│   │   ├── __init__.py
│   │   ├── base.py             # Test base classes
│   │   ├── library.py          # Test registry
│   │   └── builtin/            # Built-in tests
│   │       ├── __init__.py
│   │       ├── tabular.py      # Dataset tests
│   │       └── model.py        # Model tests
│   ├── engine/                 # Execution engine
│   │   ├── __init__.py
│   │   └── runner.py
│   ├── docs/                   # Documentation
│   │   └── __init__.py
│   └── utils/                  # Utilities
│       ├── __init__.py
│       └── yaml_utils.py
│
├── examples/                   # Working examples
│   ├── example_usage.py        # Basic demo
│   └── dag_example.py          # DAG demo
│
└── templates/                  # Project templates
```

## What You Can Do

### 1. Define Model Hierarchies

```yaml
model:
  name: expected_loss
  depends_on:
    - pd_model
    - lgd_model
  location:
    type: python_class
    path: src/models/el.py
```

### 2. Use External Models

```yaml
model:
  name: sentiment
  location: "hf/ProsusAI/finbert:sentiment-analysis"
```

### 3. Test with Dependencies

```bash
mrm test --select +expected_loss  # Test with dependencies
mrm test --select pd_model+        # Test downstream
mrm test --threads 4               # Parallel execution
```

### 4. Build Custom Tests

```python
from mrm.tests.base import MRMTest, TestResult
from mrm.tests.library import register_test

@register_test
class MyTest(MRMTest):
    name = "custom.MyTest"
    
    def run(self, model, dataset, **config):
        # Your test logic
        return TestResult(passed=True, score=0.95)
```

## Requirements

- Python 3.9+
- Dependencies (auto-installed):
  - typer[all]
  - pydantic
  - pyyaml
  - rich
  - pandas
  - numpy
  - scikit-learn
  - scipy

Optional:
- mlflow (for MLflow backend)
- transformers (for HuggingFace)
- great-expectations (for GE integration)

## Next Steps

1. **Read Documentation**
   - Start with `GETTING_STARTED.md`
   - Then `MODEL_REFERENCES.md`
   - Review `ARCHITECTURE.md` for details

2. **Run Examples**
   - `python examples/example_usage.py`
   - `python examples/dag_example.py`

3. **Create Your Project**
   - `mrm init my-project`
   - Define your models
   - Run tests

4. **Extend**
   - Add custom tests
   - Create templates
   - Build backends

## Support

- Documentation: All included in this package
- Examples: `examples/` directory
- Issues: See CONTRIBUTING.md

## License

Apache License 2.0 - See LICENSE file

## Version

MRM Core v0.1.0 - Complete with DAG, ref(), and HuggingFace support

Built: February 2026

---

**Everything you need to modernize model risk management workflows!**
