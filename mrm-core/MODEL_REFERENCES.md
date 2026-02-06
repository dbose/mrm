# Model References and External Catalogs

## Overview

MRM Core supports powerful model referencing similar to dbt's `ref()` function, plus direct integration with external model catalogs like HuggingFace Hub.

## Model Dependencies with ref()

### Basic Reference

Reference another model in your project:

```yaml
# models/credit_risk/lgd_model.yml
model:
  name: lgd_model
  version: 1.0.0
  risk_tier: tier_1
  
  location:
    type: file
    path: models/lgd_model.pkl

# models/credit_risk/expected_loss.yml
model:
  name: expected_loss
  version: 1.0.0
  risk_tier: tier_1
  
  # This model DEPENDS ON pd_model and lgd_model
  depends_on:
    - pd_model
    - lgd_model
  
  location:
    type: python_class
    path: src/models/expected_loss.py
    class: ExpectedLossModel
```

### Using ref() for Model Inputs

```yaml
# models/ensemble/credit_ensemble.yml
model:
  name: credit_ensemble
  version: 1.0.0
  risk_tier: tier_1
  
  # Reference base models
  depends_on:
    - pd_model
    - lgd_model
    - ead_model
  
  location:
    type: python_class
    path: src/models/ensemble.py
    class: CreditEnsemble
    
  # Model configuration references
  config:
    base_models:
      - ref: pd_model
      - ref: lgd_model
      - ref: ead_model
```

### DAG and Graph Operators

With dependencies defined, use graph operators to select models:

```bash
# Test pd_model and everything downstream
mrm test --select +pd_model+

# Test expected_loss and all its dependencies
mrm test --select +expected_loss

# Test pd_model and 1 level downstream
mrm test --select pd_model+1

# Test only models upstream of expected_loss
mrm test --select +expected_loss --exclude expected_loss
```

## HuggingFace Models

### Direct HuggingFace Reference

```yaml
# models/nlp/sentiment_analyzer.yml
model:
  name: sentiment_analyzer
  version: 1.0.0
  risk_tier: tier_2
  
  location:
    type: huggingface
    repo_id: ProsusAI/finbert
    revision: main
    task: sentiment-analysis
    use_pipeline: true

tests:
  - test: model.Accuracy
    config:
      min_score: 0.75
```

### Pre-configured Financial Models

```yaml
# models/nlp/esg_classifier.yml
model:
  name: esg_classifier
  version: 1.0.0
  risk_tier: tier_2
  
  location:
    type: huggingface
    repo_id: yiyanghkust/finbert-esg
    revision: main
    task: text-classification
    trust_remote_code: false

datasets:
  validation:
    type: csv
    path: data/esg_validation.csv

tests:
  - test_suite: classification_performance
```

### Using HuggingFace Models in Code

```python
from mrm.core.catalog import ModelCatalog, ModelRef, ModelSource

# Create catalog
catalog = ModelCatalog()

# Add HuggingFace model
catalog.add_huggingface_model(
    name='finbert_sentiment',
    repo_id='ProsusAI/finbert',
    task='sentiment-analysis',
    revision='main'
)

# Get reference
model_ref = catalog.ref('finbert_sentiment')
print(model_ref)
# ModelRef(source=huggingface, identifier=ProsusAI/finbert)
```

## MLflow Models

### Reference MLflow Registry

```yaml
# models/production/prod_scorecard.yml
model:
  name: prod_scorecard
  version: 2.0.0
  risk_tier: tier_1
  
  location:
    type: mlflow
    model_name: credit_scorecard
    stage: Production  # or version: "3"

tests:
  - test_suite: credit_risk
```

### MLflow with Versioning

```yaml
# models/staging/scorecard_candidate.yml
model:
  name: scorecard_candidate
  version: 2.1.0
  risk_tier: tier_1
  
  location:
    type: mlflow
    model_name: credit_scorecard
    version: "4"  # Specific version

tests:
  - test_suite: credit_risk
```

## S3/Cloud Storage Models

### S3 Reference

```yaml
# models/production/s3_model.yml
model:
  name: s3_production_model
  version: 1.0.0
  risk_tier: tier_1
  
  location:
    type: s3
    uri: s3/my-models-bucket/production/scorecard_v1.pkl
    region: us-east-1  # optional

tests:
  - test_suite: credit_risk
```

## Complex Example: Multi-Level Dependencies

```yaml
# Base models (no dependencies)

# models/base/macro_model.yml
model:
  name: macro_model
  location:
    type: file
    path: models/macro_model.pkl

# models/base/pd_model.yml
model:
  name: pd_model
  depends_on:
    - macro_model  # Depends on macro conditions
  location:
    type: file
    path: models/pd_model.pkl

# models/base/lgd_model.yml
model:
  name: lgd_model
  depends_on:
    - macro_model
  location:
    type: file
    path: models/lgd_model.pkl

# Composite model (depends on base models)

# models/composite/expected_loss.yml
model:
  name: expected_loss
  depends_on:
    - pd_model
    - lgd_model
  location:
    type: python_class
    path: src/models/expected_loss.py
    class: ExpectedLossModel

# Portfolio model (depends on expected_loss)

# models/portfolio/portfolio_var.yml
model:
  name: portfolio_var
  depends_on:
    - expected_loss
  location:
    type: python_class
    path: src/models/portfolio.py
    class: PortfolioVaR

# Stress testing (depends on everything)

# models/stress/stress_framework.yml
model:
  name: stress_framework
  depends_on:
    - macro_model
    - pd_model
    - lgd_model
    - expected_loss
    - portfolio_var
  location:
    type: python_class
    path: src/models/stress.py
    class: StressTestFramework
```

### Testing This Hierarchy

```bash
# Test everything in dependency order
mrm test

# Test macro_model and everything downstream
mrm test --select macro_model+

# Test portfolio_var and all its dependencies
mrm test --select +portfolio_var

# Test just the composite layer
mrm test --select expected_loss

# View the dependency graph
mrm debug --show-dag
```

## Using External Models in Custom Code

### Loading HuggingFace in Python

```python
from mrm.core.catalog import ModelRef, ModelSource

# Define HuggingFace model
hf_model_ref = ModelRef(
    source=ModelSource.HUGGINGFACE,
    identifier='ProsusAI/finbert',
    task='sentiment-analysis',
    revision='main'
)

# The TestRunner will automatically load this
# and wrap it with sklearn-like interface
```

### Custom Model Wrapper

```python
# src/models/ensemble.py
class CreditEnsemble:
    def __init__(self):
        from mrm.core.catalog import ModelCatalog
        
        # Get references to base models
        self.catalog = ModelCatalog()
        
        # These will be loaded by MRM
        self.pd_ref = self.catalog.ref('pd_model')
        self.lgd_ref = self.catalog.ref('lgd_model')
    
    def fit(self, X, y):
        # Ensemble training logic
        pass
    
    def predict(self, X):
        # Load referenced models
        # pd_model and lgd_model are loaded via ref()
        pass
```

## Model Catalog API

### Programmatic Catalog Management

```python
from mrm.core.catalog import ModelCatalog

# Create catalog
catalog = ModelCatalog()

# Register local model
catalog.register('my_model', ModelRef(
    source='local',
    identifier='models/my_model.pkl'
))

# Register HuggingFace model
catalog.add_huggingface_model(
    name='bert_base',
    repo_id='bert-base-uncased',
    task='fill-mask'
)

# Register MLflow model
catalog.add_mlflow_model(
    name='prod_model',
    model_name='credit_scorecard',
    stage='Production'
)

# Get reference
model_ref = catalog.ref('my_model')

# Resolve ref
resolved = catalog.resolve_ref(model_ref)

# List models
local_models = catalog.list_models(source_filter=ModelSource.LOCAL)
hf_models = catalog.list_models(source_filter=ModelSource.HUGGINGFACE)
```

## Execution Order

Models are automatically tested in dependency order:

```python
# Given this DAG:
# macro_model
#   ├─> pd_model
#   └─> lgd_model
#       └─> expected_loss

# Execution order:
# Level 0: [macro_model]
# Level 1: [pd_model, lgd_model]  # Can run in parallel
# Level 2: [expected_loss]

# mrm test --threads 4
# Will run Level 1 models in parallel
```

## Best Practices

### 1. Define Dependencies Explicitly

```yaml
model:
  name: composite_model
  depends_on:
    - base_model_1
    - base_model_2
```

### 2. Use ref() for Internal Models

```yaml
# Don't hardcode paths to other models
# Instead, use ref() to reference them by name
config:
  base_model:
    ref: pd_model  # Not: path: models/pd_model.pkl
```

### 3. Version External Models

```yaml
location:
  type: huggingface
  repo_id: ProsusAI/finbert
  revision: "v1.0.0"  # Pin version for reproducibility
```

### 4. Test Dependencies First

```bash
# Always test base models before composite models
mrm test --select +expected_loss

# This tests: macro_model, pd_model, lgd_model, then expected_loss
```

### 5. Use Appropriate Tiers

```yaml
# Base models: tier_1 (most critical)
# Composite models: tier_2 (depends on tier_1)
# Application models: tier_3
```

## Troubleshooting

### Circular Dependencies

```bash
# Error: Cycle detected in model dependencies
# Fix: Review depends_on and remove circular references
```

### Missing Reference

```bash
# Error: Model reference 'base_model' not found
# Fix: Ensure base_model is defined in models/
```

### HuggingFace Load Errors

```bash
# Error: transformers not installed
# Fix: pip install transformers

# Error: Model not found
# Fix: Check repo_id is correct on HuggingFace Hub
```

## Summary

MRM Core provides:

 **ref()** - Reference internal models  
 **DAG** - Automatic dependency management  
 **Graph Operators** - `+model+` selection syntax  
 **HuggingFace** - Direct Hub integration  
 **MLflow** - Registry integration  
 **S3/Cloud** - Cloud storage support  
 **Topological Sort** - Automatic execution ordering  
 **Parallel Execution** - Run independent models concurrently  

Use these features to build sophisticated model validation workflows that mirror your actual model dependencies!
