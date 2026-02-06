# MRM Core

**Open Source Model Risk Management CLI Framework**

A dbt-like command-line tool for automating model validation, documentation, and risk management workflows in financial services.

## Features

-  **CLI-First Design** - Standardized workflows like dbt (init, test, docs, compile)
-  **Plugin Architecture** - Integrate with MLflow, Great Expectations, W&B, or custom backends
-  **YAML Configuration** - Version-controlled, auditable model and test definitions
-  **Test Library** - 50+ built-in validation tests for model risk management
-  **Documentation Generation** - Automated model cards and validation reports
-  **Parallel Execution** - Run tests across multiple models simultaneously
-  **Open Source First** - Apache 2.0 licensed, cloud-optional

## Quick Start

```bash
# Install
pip install mrm-core

# Initialize project
mrm init my-mrm-project --template=credit_risk
cd my-mrm-project

# Run validation tests
mrm test --models credit_scorecard

# Generate documentation
mrm docs generate

# Serve docs locally
mrm docs serve
```

## Installation

```bash
# Core only (local filesystem backend)
pip install mrm-core

# With MLflow support
pip install mrm-core[mlflow]

# With Great Expectations
pip install mrm-core[ge]

# Everything
pip install mrm-core[all]

# Development
git clone https/github.com/your-org/mrm-core
cd mrm-core
poetry install
```

## Project Structure

```
my-mrm-project/
├── mrm_project.yml          # Project configuration
├── profiles.yml             # Environment configs
├── models/                  # Model definitions
│   ├── credit_risk/
│   │   └── pd_model.yml
│   └── market_risk/
│       └── var_model.yml
├── tests/                   # Test definitions
│   ├── validation/
│   └── custom/
│       └── my_tests.py
├── docs/                    # Documentation templates
└── .mrm/                    # Local state (gitignored)
```

## Core Concepts

### Models

Define models in YAML with metadata, location, datasets, and tests:

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
    
tests:
  - test_suite: credit_risk
  - test: model.Gini
    config:
      min_score: 0.65
```

### Tests

Built-in test library covers:
- Dataset quality (missing values, outliers, distributions)
- Model performance (accuracy, calibration, discrimination)
- Model stability (PSI, CSI, drift detection)
- Regulatory compliance (stress testing, sensitivity analysis)

Custom tests are Python classes:

```python
from mrm.tests.base import MRMTest

class MyCustomTest(MRMTest):
    name = "custom.my_test"
    
    def run(self, model, dataset, **config):
        # Your validation logic
        return {
            'passed': True,
            'score': 0.95,
            'details': {...}
        }
```

### Backends

Plugin architecture supports multiple backends:
- **Local**: Filesystem storage (default)
- **MLflow**: Model registry and experiment tracking
- **Great Expectations**: Data quality validation
- **W&B**: Experiment tracking and visualization
- **Custom**: Build your own adapter

## CLI Commands

```bash
# Initialize new project
mrm init [project_name] --template=credit_risk

# List models
mrm list models --tier=tier_1

# Compile project
mrm compile --models pd_model

# Run tests
mrm test --models pd_model
mrm test --select tier:tier_1
mrm test --suite credit_risk

# Generate docs
mrm docs generate --models pd_model
mrm docs serve --port 8080

# Register model
mrm register --model pd_model --backend mlflow

# Show test results
mrm show test-results --model pd_model
```

## Selection Syntax (dbt-style)

```bash
# By model name
mrm test --models pd_model

# By risk tier
mrm test --select tier:tier_1

# By owner
mrm test --select owner:credit_team

# Graph operators
mrm test --select pd_model+     # Model and downstream
mrm test --select +pd_model     # Model and upstream
mrm test --select @pd_model     # Model only

# Exclude
mrm test --exclude pd_model
```

## Configuration

### mrm_project.yml

```yaml
name: quantitative_models
version: 1.0.0

governance:
  risk_tiers:
    tier_1:
      validation_frequency: quarterly
      required_tests: [backtesting, sensitivity]

test_suites:
  credit_risk:
    - roc_auc
    - calibration
    - population_stability

backends:
  default: local
  
  local:
    type: filesystem
    path: ~/.mrm/data
```

### profiles.yml

```yaml
mrm:
  outputs:
    dev:
      backend: local
    prod:
      backend: mlflow
      tracking_uri: http/mlflow.company.com
  target: dev
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        mrm CLI                               │
│  (Typer-based, similar to dbt CLI)                          │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼─────┐
│ Config │      │ Plugins  │
│ Layer  │      │ System   │
└───┬────┘      └────┬─────┘
    │                │
┌───▼─────▼────┐     │
│  Execution   │◄────┘
│   Engine     │
└───┬──────────┘
    │
┌───▼──────────┐
│   Storage    │
│   Layer      │
└──────────────┘
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - see [LICENSE](LICENSE)

## Roadmap

### v0.1 (Current)
-  CLI with core commands
-  Local backend
-  Basic test library
-  YAML configuration

### v0.2 (Next)
- MLflow integration
- Great Expectations integration
- 50+ built-in tests
- Enhanced documentation generation

### v0.3
- State management
- Advanced selectors
- Test macros
- Lineage tracking

### v1.0
- Production-ready
- Comprehensive test library
- Multi-backend support
- Enterprise features

## Support

- Documentation: https/mrm-core.readthedocs.io
- Issues: https/github.com/your-org/mrm-core/issues
- Discussions: https/github.com/your-org/mrm-core/discussions

## Acknowledgments

Inspired by dbt's elegant CLI design and the urgent need for modern model risk management tooling in financial services.
