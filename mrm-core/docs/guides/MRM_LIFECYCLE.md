# MRM Lifecycle - Complete Workflow

## Overview

The Model Risk Management (MRM) lifecycle in MRM Core follows a **dbt-inspired workflow** that covers the full model validation and governance process from development to production monitoring.

## The Five-Stage Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    MRM LIFECYCLE                             │
│                                                              │
│  1. INIT → 2. DEVELOP → 3. TEST → 4. PUBLISH → 5. MONITOR  │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. INIT - Project Initialization

**Initialize a new MRM project with scaffolding and templates.**

### Commands
```bash
# Create new project
mrm init my-mrm-project

# Navigate into project
cd my-mrm-project

# Explore structure
mrm list models
mrm list tests
```

### What Gets Created
```
my-mrm-project/
├── mrm_project.yml          # Project configuration
├── profiles.yml             # Environment profiles
├── models/                  # Model configurations
│   └── example_model.yml
├── data/                    # Training/validation data
├── tests/                   # Custom test implementations
│   └── custom/
└── docs/                    # Documentation templates
```

### Key Concepts
- **Project**: A collection of models to be validated
- **Models**: YAML configs defining model location, data, and tests
- **Profiles**: Environment-specific settings (dev, staging, prod)

---

## 2. DEVELOP - Model Configuration

**Configure models with metadata, dependencies, and validation requirements.**

### Model Configuration Example
```yaml
# models/credit_risk/scorecard.yml
model:
  name: credit_scorecard
  version: 1.0.0
  risk_tier: tier_1
  
  description: "Probability of Default model for consumer credit"
  owner: credit_risk_team
  use_case: consumer_lending
  methodology: logistic_regression
  
  # Model dependency (dbt-style)
  depends_on:
    - ref('macro_model')      # References another model
    - ref('feature_store')
  
  # Model location
  location:
    type: file
    path: models/credit_scorecard.pkl
    # Alternative sources:
    # type: mlflow
    # type: huggingface
    # type: python_class

datasets:
  training:
    type: csv
    path: data/training.csv
    
  validation:
    type: csv  
    path: data/validation.csv

tests:
  # Data quality tests
  - test: tabular_dataset.MissingValues
    config:
      dataset: validation
      threshold: 0.05
      
  # Model performance tests
  - test: model.Accuracy
    config:
      dataset: validation
      min_score: 0.70
      
  - test: model.ROCAUC
    config:
      dataset: validation
      min_score: 0.75
```

### Dependency Graph (dbt-style)
```yaml
# Define model dependencies
model:
  name: expected_loss
  depends_on:
    - ref('pd_model')
    - ref('lgd_model')
```

### Commands
```bash
# List all models
mrm list models

# Show dependency graph
mrm debug --show-dag

# View project config
mrm debug --show-config
```

---

## 3. TEST - Model Validation

**Run comprehensive validation tests to ensure model quality and compliance.**

### Test Execution
```bash
# Test all models
mrm test

# Test specific model
mrm test --model credit_scorecard

# Test with dependencies (dbt-style graph operators)
mrm test --select +credit_scorecard    # Model + upstream dependencies
mrm test --select credit_scorecard+    # Model + downstream dependents
mrm test --select +credit_scorecard+   # Model + all related

# Parallel execution
mrm test --threads 4

# Stop on first failure
mrm test --fail-fast

# Test by risk tier
mrm list models --tier tier_1
mrm test --models tier_1_models
```

### Built-in Test Categories

#### 1. Data Quality Tests
- `MissingValues` - Check for missing data
- `OutlierDetection` - Detect statistical outliers
- `ClassImbalance` - Validate class distribution
- `FeatureDistribution` - Check feature stats

#### 2. Model Performance Tests
- `Accuracy` - Classification accuracy
- `ROCAUC` - ROC-AUC score
- `Precision` / `Recall` / `F1Score`
- `Gini` - Gini coefficient (credit risk)

#### 3. Model Stability Tests
- `BacktestingPerformance` - Historical performance
- `SensitivityAnalysis` - Feature importance
- `PopulationStability` - PSI/CSI metrics

### Test Suites
```yaml
# Define reusable test suites
test_suites:
  credit_risk_validation:
    - model.Gini
    - model.ROCAUC
    - tabular_dataset.ClassImbalance
  
  tier_1_validation:
    - model.BacktestingPerformance
    - model.SensitivityAnalysis
    - model.PopulationStability
```

```bash
# Run test suite
mrm test --suite credit_risk_validation
```

### Custom Tests
```python
# tests/custom/my_test.py
from mrm.tests.base import BaseTest
from mrm.tests.library import register_test

@register_test
class CustomRegulatoryTest(BaseTest):
    """Custom regulatory compliance test"""
    
    category = "compliance"
    
    def run(self, model, datasets, config):
        # Your validation logic
        passed = self._check_compliance(model, datasets)
        
        return {
            'passed': passed,
            'score': 0.95,
            'details': {'regulation': 'SR 11-7'}
        }
```

---

## 4. PUBLISH - Model Registration

**Register validated models to enterprise catalogs (Databricks, MLflow, etc.).**

### Publishing Workflow

#### Step 1: Configure External Catalog
```yaml
# mrm_project.yml
catalogs:
  databricks:
    type: databricks_unity
    host: "{{ env_var('DATABRICKS_HOST') }}"  # dbt-style env vars
    token: "{{ env_var('DATABRICKS_TOKEN') }}"
    catalog: workspace  # Unity Catalog catalog name
    schema: default     # Unity Catalog schema name
    mlflow_registry: true  # Register to MLflow Registry with Unity Catalog
```

#### Step 2: Set Environment Variables
```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."  # Generate from User Settings > Developer
```

#### Step 3: Publish Model
```bash
# Publish model to Databricks Unity Catalog
mrm publish credit_scorecard

# Model will be registered as: <catalog>.<schema>.<model_name>
# Example: workspace.default.credit_scorecard
```

#### Alternative: Direct Registration
```bash
# Register using catalog add command
mrm catalog add \
  --name credit_scorecard \
  --from-file models/credit_scorecard.pkl \
  --catalog databricks
```

### What Happens During Publish

1. **Validation Data Loading**: Loads validation dataset from model config
2. **Signature Inference**: Automatically infers model signature from validation data (required for Unity Catalog)
3. **MLflow Experiment**: Creates or uses `/Shared/mrm-experiments` experiment
4. **Model Artifact Upload**: Model file uploaded to Databricks DBFS via MLflow
5. **Unity Catalog Registration**: Model registered in Unity Catalog with three-level namespace
6. **Versioning**: Automatic versioning (Version 1, 2, 3, etc.)
7. **Governance Metadata**: Risk tier, owner, use case attached as tags

**Requirements:**
- Model must have validation dataset configured in YAML
- Model must be sklearn-compatible (supports `predict()` method)
- Databricks workspace must have Unity Catalog enabled
- Token must have permissions to create experiments and register models

### Published Model Reference
```yaml
# Reference published model from catalog
model:
  name: ensemble_model
  location:
    type: catalog
    uri: databricks_uc://main.models/credit_scorecard
```

---

## 5. MONITOR - Ongoing Validation

**Continuous monitoring and periodic revalidation in production.**

### Monitoring Strategies

#### A. Scheduled Revalidation
```bash
# Run tests on production models (scheduled via cron/Airflow)
mrm test --profile prod --models credit_scorecard

# Generate validation reports
mrm test --models credit_scorecard --output-format html > report.html
```

#### B. Catalog Integration
```bash
# List models in catalog
mrm catalog resolve databricks_uc://main.models/credit_scorecard

# Refresh catalog cache
mrm catalog refresh

# Test catalog-referenced models
mrm test --select catalog_models
```

#### C. Drift Detection
```yaml
# Ongoing monitoring tests
tests:
  - test: model.PopulationStability
    config:
      dataset: production_data
      reference_dataset: validation
      psi_threshold: 0.1
      
  - test: model.PerformanceDrift
    config:
      dataset: production_data
      baseline_score: 0.75
      drift_threshold: 0.05
```

### Governance Compliance

#### Risk Tier Validation Frequency
```yaml
# mrm_project.yml
governance:
  risk_tiers:
    tier_1:
      validation_frequency: quarterly
      required_tests: [backtesting, sensitivity, stability]
    
    tier_2:
      validation_frequency: semi_annual
      required_tests: [backtesting, sensitivity]
```

#### Automated Compliance Checks
```bash
# Check if models meet governance requirements
mrm test --tier tier_1 --suite tier_1_validation

# Generate compliance report
mrm docs generate --template sr11-7
```

---

## Complete Workflow Example

### End-to-End Credit Risk Model Lifecycle

```bash
# 1. INIT
mrm init credit-risk-validation
cd credit-risk-validation

# 2. DEVELOP
# Create model config: models/pd_model.yml
cat > models/pd_model.yml << EOF
model:
  name: pd_model
  version: 1.0.0
  risk_tier: tier_1
  owner: credit_risk_team
  
  location:
    type: file
    path: models/pd_model.pkl
  
  datasets:
    validation:
      type: csv
      path: data/validation.csv
  
  tests:
    - test: model.ROCAUC
      config:
        dataset: validation
        min_score: 0.75
    
    - test: model.Gini
      config:
        dataset: validation
        min_score: 0.45
EOF

# 3. TEST
mrm test --model pd_model

# Test with dependencies
mrm test --select +pd_model+

# Parallel testing
mrm test --threads 4

# 4. PUBLISH
export DATABRICKS_HOST="https://workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."

mrm publish --model pd_model --version 1.0.0

# 5. MONITOR
# Schedule via cron/Airflow:
# 0 0 1 */3 * cd /path/to/project && mrm test --profile prod --model pd_model
```

---

## Best Practices

### 1. Version Control
```bash
# Track model configs in Git
git add models/ mrm_project.yml
git commit -m "Add pd_model config"
```

### 2. Environment Profiles
```yaml
# profiles.yml
dev:
  backend: local
  
staging:
  backend: mlflow
  mlflow_tracking_uri: http://staging-mlflow:5000
  
prod:
  backend: mlflow
  mlflow_tracking_uri: http://prod-mlflow:5000
  catalogs:
    databricks:
      host: https://prod-databricks.com
```

### 3. CI/CD Integration
```yaml
# .github/workflows/mrm-validation.yml
name: Model Validation
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install MRM
        run: pip install mrm-core
      - name: Run Tests
        run: mrm test --fail-fast
```

### 4. Model Governance
- **Tier-based validation**: Higher risk = more tests + higher frequency
- **Dependency tracking**: Test upstream models when downstream changes
- **Audit trail**: All test results logged and versioned
- **Continuous monitoring**: Scheduled revalidation in production

---

## CLI Quick Reference

```bash
# Project
mrm init <project>              # Create project
mrm debug --show-dag            # View dependency graph
mrm debug --show-catalog        # View model catalog

# Testing
mrm test                        # Test all
mrm test -m <model>             # Test specific model
mrm test --select +model+       # Test with dependencies
mrm test --threads 4            # Parallel
mrm test --fail-fast            # Stop on failure

# Publishing
mrm publish -m <model>          # Publish to catalog
mrm publish -m <model> --to <catalog>  # Specific catalog

# Catalog
mrm catalog resolve <uri>       # Resolve catalog URI
mrm catalog add -n <name> -f <file>    # Register model
mrm catalog refresh             # Refresh cache

# Discovery
mrm list models                 # List models
mrm list tests                  # List available tests
mrm list suites                 # List test suites
mrm list backends               # List backends
```

---

## Summary

The MRM lifecycle provides a **structured, repeatable process** for model validation and governance:

1. **INIT**: Set up project structure
2. **DEVELOP**: Configure models with dependencies and tests
3. **TEST**: Run comprehensive validation
4. **PUBLISH**: Register to enterprise catalogs
5. **MONITOR**: Continuous validation and drift detection

This workflow ensures:
- Models are thoroughly validated before production
- Dependencies are tracked and tested together
- Governance requirements are enforced
- Changes are auditable and reversible
- Production models are continuously monitored

**Everything you need to modernize model risk management workflows!**
