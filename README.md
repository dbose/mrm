# MRM Core

**Open Source Model Risk Management CLI Framework**

A dbt-like command-line tool for automating model validation, documentation, and risk management workflows in financial services.

## What's Included

**Core Framework** (`mrm-core/`)
- Complete MRM CLI with 3,500+ lines of Python
- 10+ built-in validation tests
- dbt-style workflows (DAG, ref(), graph operators)
- Databricks Unity Catalog integration
- MLflow and HuggingFace support
- Full documentation and examples

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
-  **CLI-First Design** - Standardized workflows like dbt (init, test, docs, publish)
-  **Databricks Unity Catalog** - Direct publishing with automatic signature inference
-  **Model Dependencies** - DAG with `depends_on`, `ref()`, and graph operators
-  **Plugin Architecture** - Integrate with MLflow, Great Expectations, W&B, or custom backends
-  **YAML Configuration** - Version-controlled, auditable model and test definitions
-  **Test Library** - 50+ built-in validation tests for model risk management
-  **Multiple Model Sources** - Local files, Python classes, MLflow, HuggingFace Hub
-  **Documentation Generation** - Automated model cards and validation reports
-  **Parallel Execution** - Run tests across multiple models simultaneously
-  **Open Source First** - Apache 2.0 licensed, cloud-optionalAG example
python examples/dag_example.py
```bash
# Install
cd mrm-core
pip install -e .

# Or with Poetry
poetry install

# Run examples
python examples/example_usage.py
python examples/dag_example.py

# Initialize project
mrm init my-mrm-project --template=credit_risk
cd my-mrm-project

# Run validation tests
mrm test --models credit_scorecard

# Publish to Databricks Unity Catalog
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
mrm publish credit_scorecard

# Generate documentation
mrm docs generate
mrm docs servecli/                    # CLI interface
│   │   ├── __init__.py
│   │   └── main.py
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── project.py          # Project management
│   │   ├── dag.py              # Dependency graph
│   │   ├── catalog.py          # Model catalog
│   │   ├── references.py       # Model references
│   │   └── init.py             # Project init
│  Project Structure

```
mrm/
├── README.md                    # This file
├── mrm-core/                    # Main package
│   ├── mrm/                     # Python package
│   │   ├── cli/                 # CLI interface
│   │   ├── core/                # Core functionality
│   │   │   ├── catalog_backends/  # Databricks, etc.
│   │   ├── backends/            # Storage backends
│   │   ├── tests/               # Test framework
│   │   ├── engine/              # Execution engine
│   │   └── utils/               # Utilities
│   ├── examples/                # Working examples
│   ├── credit_risk_example/     # Example project
│   ├── docs/                    # Documentation
│   │   ├── GETTING_STARTED.md
│   │   ├── ARCHITECTURE.md
│   │   ├── MODEL_REFERENCES.md
│   │   ├── DAG_FEATURES.md
│   │   ├── COMPLETE_FEATURES.md
│   │   └── MRM_LIFECYCLE.md
│   └── designs/                 # Design documents
│       └── databricks_unity_catalog.md
└── [other repos]
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
   Core Concepts

### Model Configuration

```yaml
model:
  name: credit_scorecard
  version: 1.0.0
  risk_tier: tier_1
  
  location:
    type: python_class
    path: src/models/scorecard.py
    class: CreditScorecard
    
datasets:
  validation:
    type: parquet
    path: data/validation.parquet
    
tesCLI Commands

```bash
# Initialize new project
mrm init [project_name] --template=credit_risk

# List models
mrm list models --tier=tier_1

# Run tests
mrm test --models pd_model
mrm test --select tier:tier_1
mrm test --select +pd_model  # With dependencies

# Publish model
mrm publish credit_scorecard

# Generate docs
mrmDocumentation

- [Getting Started](mrm-core/GETTING_STARTED.md) - Step-by-step usage guide
- [MRM Lifecycle](mrm-core/MRM_LIFECYCLE.md) - Complete workflow documentation
- [Complete Features](mrm-core/COMPLETE_FEATURES.md) - Full feature summary
- [Architecture](mrm-core/ARCHITECTURE.md) - Technical architecture
- [Model References](mrm-core/MODEL_REFERENCES.md) - ref() and catalog guide
- [DAG Features](mrm-core/DAG_FEATURES.md) - Dependency graph features
- [Databricks Integration](mrm-core/designs/databricks_unity_catalog.md) - Unity Catalog design
pip install mrm-core[all]

# Development
cd mrm-core
pip install -e .
# or
poetry install
```

## Requirements

- Python 3.9+
- CExamples

```bash
# Run basic example
cd mrm-core
python examples/example_usage.py

# Run DAG example
python examples/dag_example.py

# Try credit risk example
cd credit_risk_example
mrm test
```

## Contributing

See [CONTRIBUTING.md](mrm-core/CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - See [LICENSE](mrm-core/LICENSE)

## Version

**MRM Core v0.1.0**
- Complete dbt-style workflows
- Databricks Unity Catalog integration
- DAG, ref(), and HuggingFace support
- 50+ built-in tests

Built: February 2026
```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
mrm publish credit_scorecard