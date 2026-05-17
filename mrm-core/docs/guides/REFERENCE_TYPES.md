"""
Complete Reference Types Example

Demonstrates all supported model reference types:
1. ref() - Reference to another model in the project
2. hf/ - HuggingFace Hub models  
3. file/ - Local files
4. mlflow/ - MLflow registry
5. catalog/ - Internal model catalogs
"""

# Example Model Configurations

## 1. Base Model (Local File)
```yaml
# models/base/pd_model.yml
model:
  name: pd_model
  version: 1.0.0
  risk_tier: tier_1
  
  location:
    type: file
    path: models/pd_model.pkl

tests:
  - test: model.ROCAUC
    config:
      min_score: 0.70
```

## 2. Model Using ref() to Base Model
```yaml
# models/composite/expected_loss.yml
model:
  name: expected_loss
  version: 1.0.0
  risk_tier: tier_1
  
  # Declare dependencies
  depends_on:
    - pd_model
    - lgd_model
  
  # This model can access pd_model via ref()
  location:
    type: python_class
    path: src/models/expected_loss.py
    class: ExpectedLossModel

tests:
  - test: model.ROCAUC
    config:
      min_score: 0.75
```

### Python Implementation with ref()
```python
# src/models/expected_loss.py
class ExpectedLossModel:
    """
    Composite model that uses base models
    
    The base models (pd_model, lgd_model) are loaded
    automatically by the MRM framework based on depends_on
    """
    
    def __init__(self):
        # Base models will be injected by framework
        self.pd_model = None
        self.lgd_model = None
    
    def set_base_models(self, pd_model, lgd_model):
        """Called by framework to inject dependencies"""
        self.pd_model = pd_model
        self.lgd_model = lgd_model
    
    def predict_proba(self, X):
        """Combine predictions from base models"""
        pd_probs = self.pd_model.predict_proba(X)
        lgd_probs = self.lgd_model.predict_proba(X)
        return (pd_probs + lgd_probs) / 2
```

## 3. HuggingFace Model (Financial BERT)
```yaml
# models/nlp/sentiment_analyzer.yml
model:
  name: sentiment_analyzer
  version: 1.0.0
  risk_tier: tier_2
  
  # Direct HuggingFace reference
  location:
    type: huggingface
    repo_id: ProsusAI/finbert
    revision: main
    task: sentiment-analysis
    trust_remote_code: false

datasets:
  validation:
    type: csv
    path: data/sentiment_validation.csv

tests:
  - test: model.Accuracy
    config:
      min_score: 0.75
```

### Alternative: String-based HuggingFace Reference
```yaml
# models/nlp/text_classifier.yml
model:
  name: text_classifier
  version: 1.0.0
  risk_tier: tier_2
  
  # Shorthand syntax
  location: "hf/distilbert-base-uncased:text-classification"

tests:
  - test: model.F1Score
    config:
      min_score: 0.70
```

## 4. MLflow Model (Production Scorecard)
```yaml
# models/production/prod_scorecard.yml
model:
  name: prod_scorecard
  version: 2.0.0
  risk_tier: tier_1
  
  location:
    type: mlflow
    model_name: credit_scorecard
    stage: Production  # Use production version

tests:
  - test_suite: credit_risk
```

### Alternative: Specific MLflow Version
```yaml
# models/staging/scorecard_v3.yml
model:
  name: scorecard_candidate
  version: 2.1.0
  risk_tier: tier_1
  
  location:
    type: mlflow
    model_name: credit_scorecard
    version: "3"  # Specific version

tests:
  - test_suite: credit_risk
```

## 5. Model Catalog Reference
```yaml
# models/external/base_llm.yml
model:
  name: base_llm
  version: 1.0.0
  risk_tier: tier_2
  
  # Reference from internal catalog
  location:
    type: catalog
    path: production/llms/gpt-3.5-turbo
    
tests:
  - test: custom.LLMHallucinationTest
```

## 6. Complex Hierarchy Example
```yaml
# Complete dependency chain with multiple reference types

# Base model (local file)
# models/base/macro_model.yml
model:
  name: macro_model
  location:
    type: file
    path: models/macro_model.pkl

---

# Statistical model (local file, depends on macro)
# models/base/pd_model.yml
model:
  name: pd_model
  depends_on:
    - macro_model
  location:
    type: file
    path: models/pd_model.pkl

---

# ML model (local file, depends on macro)
# models/base/lgd_model.yml
model:
  name: lgd_model
  depends_on:
    - macro_model
  location:
    type: file
    path: models/lgd_model.pkl

---

# Composite model (Python class, depends on base models)
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

---

# NLP model (HuggingFace, for text analysis)
# models/nlp/document_classifier.yml
model:
  name: document_classifier
  location: "hf/yiyanghkust/finbert-esg:text-classification"

---

# Final ensemble (depends on everything)
# models/ensemble/final_model.yml
model:
  name: final_ensemble
  depends_on:
    - expected_loss
    - document_classifier
  location:
    type: python_class
    path: src/models/ensemble.py
    class: FinalEnsemble
```

## Testing with Dependencies

```bash
# Test macro_model and everything downstream
mrm test --select macro_model+

# This will test in order:
# 1. macro_model
# 2. pd_model, lgd_model (parallel if --threads > 1)
# 3. expected_loss
# 4. final_ensemble

# Test final_ensemble and all dependencies
mrm test --select +final_ensemble

# This will test:
# 1. macro_model
# 2. pd_model, lgd_model, document_classifier (parallel)
# 3. expected_loss
# 4. final_ensemble

# Test just HuggingFace model
mrm test --models document_classifier

# View dependency graph
mrm debug --show-dag
```

## Python Usage

```python
from mrm.core.references import ModelReference, ModelLoader
from mrm.core.catalog import ModelCatalog
from pathlib import Path

# Create loader
loader = ModelLoader(
    project_root=Path.cwd(),
    backend=None,
    catalog=None
)

# Load different model types

# 1. Local file
model1 = loader.load('file/models/pd_model.pkl')

# 2. HuggingFace
model2 = loader.load('hf/ProsusAI/finbert:sentiment-analysis')

# 3. MLflow (requires backend)
# model3 = loader.load('mlflow/credit_scorecard')

# 4. Reference (requires catalog)
# model4 = loader.load("ref('pd_model')")
```

## Best Practices

### 1. Use ref() for Internal Dependencies
```yaml
# Good
depends_on:
  - pd_model

# Bad - hardcoding paths
config:
  pd_model_path: models/pd_model.pkl
```

### 2. Pin HuggingFace Versions
```yaml
# Good - reproducible
location:
  type: huggingface
  repo_id: ProsusAI/finbert
  revision: "v1.0.0"  # Specific version

# Bad - may change
location:
  type: huggingface
  repo_id: ProsusAI/finbert
  revision: main  # Could change
```

### 3. Organize by Reference Type
```
models/
├── base/           # Local file models
│   ├── pd_model.yml
│   └── lgd_model.yml
├── composite/      # Python class models with ref()
│   └── expected_loss.yml
├── external/       # HuggingFace, MLflow
│   ├── sentiment.yml
│   └── prod_scorecard.yml
└── ensemble/       # Final models
    └── final_model.yml
```

### 4. Document Dependencies
```yaml
model:
  name: expected_loss
  description: "Combines PD and LGD models to estimate expected loss"
  
  # Clear dependency documentation
  depends_on:
    - pd_model      # Probability of default
    - lgd_model     # Loss given default
  
  location:
    type: python_class
    path: src/models/expected_loss.py
    class: ExpectedLossModel
```

## Common Patterns

### Pattern 1: Base → Composite → Ensemble
```
Local Files (pd, lgd)
  → Python Class with ref() (expected_loss)
    → Python Class with ref() (portfolio_var)
```

### Pattern 2: External + Internal → Hybrid
```
HuggingFace (sentiment)
  + Local Files (pd, lgd)
    → Python Class combining both
```

### Pattern 3: MLflow Production Pipeline
```
MLflow Stage=Staging (candidate)
  → Tests pass
    → Promote to Production
      → Python Class references Production stage
```

## Summary

MRM Core supports:

 **ref('model')** - Reference internal models  
 **hf/org/model** - HuggingFace Hub  
 **mlflow/model** - MLflow registry  
 **catalog/path** - Internal catalogs  
 **file/path** - Local files  
 **Automatic resolution** - Dependencies loaded in order  
 **Mixed references** - Combine different types  
 **Type safety** - Validated at load time  

Use these reference types to build sophisticated model hierarchies!
