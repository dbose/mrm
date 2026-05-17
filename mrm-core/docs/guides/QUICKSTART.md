# MRM Core - Quick Start Guide

## Installation

### From Source (Development)

```bash
git clone https/github.com/your-org/mrm-core
cd mrm-core
pip install -e .
```

### With Poetry

```bash
cd mrm-core
poetry install
poetry shell
```

## Quick Start

### 1. Create a New Project

```bash
mrm init my-mrm-project --template=credit_risk
cd my-mrm-project
```

This creates a project structure:
```
my-mrm-project/
├── mrm_project.yml          # Project configuration
├── profiles.yml             # Environment configs
├── models/                  # Model definitions
│   └── credit_risk/
│       └── scorecard.yml    # Example model
├── tests/custom/            # Custom tests
├── data/                    # Datasets
└── .mrm/                    # Local state
```

### 2. Define a Model

Create `models/my_model.yml`:

```yaml
model:
  name: my_credit_model
  version: 1.0.0
  risk_tier: tier_1
  
  location:
    type: file
    path: models/my_model.pkl
  
datasets:
  validation:
    type: csv
    path: data/validation.csv

tests:
  - test_suite: credit_risk
  - test: model.Gini
    config:
      min_score: 0.40
```

### 3. Run Tests

```bash
# Test all models
mrm test

# Test specific model
mrm test --models my_credit_model

# Test by risk tier
mrm test --select tier:tier_1

# Parallel execution
mrm test --threads 4
```

### 4. View Results

```bash
# List models
mrm list models

# List available tests
mrm list tests

# Show test suites
mrm list suites

# Debug configuration
mrm debug --show-config
```

## Example: Complete Workflow

```python
# 1. Train a model
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

# Load data
train = pd.read_csv('data/training.csv')
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Save
with open('models/my_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

```bash
# 2. Define model in YAML (see above)

# 3. Run validation
mrm test --models my_credit_model

# 4. View test history
mrm show test-results --model my_credit_model
```

## Configuration

### Project Config (`mrm_project.yml`)

```yaml
name: my_project
version: 1.0.0

governance:
  risk_tiers:
    tier_1:
      validation_frequency: quarterly
      required_tests: [backtesting, sensitivity]

test_suites:
  credit_risk:
    - model.Gini
    - model.ROCAUC

backends:
  local:
    type: filesystem
    path: ~/.mrm/data
```

### Profiles Config (`profiles.yml`)

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

## Custom Tests

Create `tests/custom/my_tests.py`:

```python
from mrm.tests.base import MRMTest, TestResult
from mrm.tests.library import register_test

@register_test
class MyStressTest(MRMTest):
    name = "custom.StressTest"
    description = "Custom stress testing"
    
    def run(self, model, dataset, **config):
        # Your logic here
        score = 0.95
        passed = score > config.get('threshold', 0.90)
        
        return TestResult(
            passed=passed,
            score=score,
            details={'scenario': 'adverse'}
        )
```

Use in model YAML:
```yaml
tests:
  - test: custom.StressTest
    config:
      threshold: 0.85
```

## MLflow Integration

1. Install MLflow support:
```bash
pip install mrm-core[mlflow]
```

2. Update `profiles.yml`:
```yaml
mrm:
  outputs:
    prod:
      backend: mlflow
      tracking_uri: http/localhost:5000
      experiment_name: model-validation
```

3. Run tests:
```bash
mrm test --profile prod
```

## Next Steps

- Read the full documentation
- Explore example projects
- Join the community
- Contribute to the project

## Common Issues

### Model Not Loading
- Check file paths are relative to project root
- Verify model format (pkl, joblib, etc.)

### Tests Failing
- Check dataset format matches configuration
- Verify threshold values are appropriate
- Review test requirements

### Backend Errors
- Ensure backend is configured correctly
- Check network connectivity for remote backends
- Verify credentials if using MLflow/cloud backends

## Getting Help

- TBA
