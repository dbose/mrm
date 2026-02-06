# DAG and Model References - Complete Implementation

## What Was Added

Complete implementation of dbt-style model dependencies, DAG management, and external model catalog integration.

## New Features

### 1. Model Dependency DAG (Like dbt)

**Purpose:** Track dependencies between models and enable graph-based operations

**Implementation:**
- `mrm/core/dag.py` - Complete DAG management class
- Topological sorting (models tested in dependency order)
- Cycle detection
- Execution level calculation (for parallelism)
- Graph operators (`+model+`, `model+`, `+model`)

**Usage:**

```yaml
# Define dependencies in model YAML
model:
  name: expected_loss
  depends_on:
    - pd_model
    - lgd_model
```

```bash
# Use graph operators for selection
mrm test --select +pd_model+      # PD model, upstream & downstream
mrm test --select pd_model+       # PD model and downstream
mrm test --select +expected_loss  # Expected loss and dependencies
```

### 2. Model References with ref()

**Purpose:** Reference other models (like dbt's ref())

**Implementation:**
- `mrm/core/catalog.py` - Model catalog and reference system
- Resolve `ref()` to actual model locations
- Track all models in project
- Support external model catalogs

**Usage:**

```yaml
# Reference another model
model:
  name: composite_model
  depends_on:
    - base_model
  
  location:
    type: ref
    model: base_model  # Reference by name, not path
```

### 3. HuggingFace Integration

**Purpose:** Use HuggingFace Hub models directly

**Implementation:**
- Automatic model loading from HuggingFace
- Pipeline API support
- Sklearn-like wrapper for transformers models
- Pre-configured financial models

**Usage:**

```yaml
model:
  name: sentiment_analyzer
  location:
    type: huggingface
    repo_id: ProsusAI/finbert
    task: sentiment-analysis
    revision: main
```

### 4. MLflow Integration (Enhanced)

**Purpose:** Load models from MLflow registry

**Implementation:**
- Support for model stages (Production, Staging)
- Version-specific loading
- Latest version support

**Usage:**

```yaml
model:
  name: prod_model
  location:
    type: mlflow
    model_name: credit_scorecard
    stage: Production  # or version: "3"
```

### 5. S3/Cloud Storage Support

**Purpose:** Load models from cloud storage

**Implementation:**
- S3, GCS, Azure Blob support
- Automatic download and caching

**Usage:**

```yaml
model:
  name: cloud_model
  location:
    type: s3
    uri: s3/my-bucket/models/scorecard.pkl
```

## Files Added/Modified

### New Files

1. **`mrm/core/dag.py`** (600 lines)
   - ModelDAG class
   - Topological sorting
   - Graph operators
   - Execution levels

2. **`mrm/core/catalog.py`** (400 lines)
   - ModelCatalog class
   - ModelRef dataclass
   - HuggingFace helpers
   - External catalog integration

3. **`MODEL_REFERENCES.md`** (documentation)
   - Complete guide to ref() and DAG
   - Examples for all model sources
   - Best practices

4. **`examples/dag_example.py`** (300 lines)
   - Full working example
   - Multi-level model hierarchy
   - DAG visualization

### Modified Files

1. **`mrm/core/project.py`**
   - Added `dag` property
   - Added `catalog` property
   - Updated `select_models()` for graph operators
   - Added `select_models_graph()`

2. **`mrm/engine/runner.py`**
   - Accept `catalog` parameter
   - Complete rewrite of `_load_model()`
   - Support all model sources
   - HuggingFace wrapper

3. **`mrm/cli/main.py`**
   - Pass catalog to TestRunner
   - Added `--show-dag` to debug command
   - Added `--show-catalog` to debug command

## Graph Operators (dbt-style)

### Syntax

```bash
model          # Just the model
+model         # Model and all upstream (dependencies)
model+         # Model and all downstream (dependents)
+model+        # Model, upstream, and downstream
@model         # Explicitly just the model
1+model        # Model and 1 level upstream
model+2        # Model and 2 levels downstream
```

### Examples

Given this DAG:
```
macro_model
  ├─> pd_model
  │     └─> expected_loss
  └─> lgd_model
        └─> expected_loss
              └─> portfolio_var
```

Selections:
```bash
mrm test --select pd_model
# → [pd_model]

mrm test --select +pd_model
# → [macro_model, pd_model]

mrm test --select pd_model+
# → [pd_model, expected_loss, portfolio_var]

mrm test --select +pd_model+
# → [macro_model, pd_model, expected_loss, portfolio_var]

mrm test --select +expected_loss
# → [macro_model, pd_model, lgd_model, expected_loss]

mrm test --select 1+portfolio_var
# → [expected_loss, portfolio_var]
```

## Topological Sorting

Models are automatically tested in dependency order:

```python
# Given DAG:
# Level 0: [macro_model]
# Level 1: [pd_model, lgd_model]
# Level 2: [expected_loss]
# Level 3: [portfolio_var]

# Running: mrm test
# Executes in order: macro_model → pd_model,lgd_model → expected_loss → portfolio_var

# With --threads 4:
# Level 1 models (pd_model, lgd_model) run in parallel
```

## Model Catalog API

### Programmatic Usage

```python
from mrm.core.catalog import ModelCatalog, ModelRef

# Create catalog
catalog = ModelCatalog()

# Register models
catalog.register('my_model', ModelRef(
    source='local',
    identifier='models/my_model.pkl'
))

# Add HuggingFace model
catalog.add_huggingface_model(
    name='finbert',
    repo_id='ProsusAI/finbert',
    task='sentiment-analysis'
)

# Get reference
model_ref = catalog.ref('my_model')

# Resolve ref
resolved = catalog.resolve_ref(model_ref)
```

## Complete Example

### Model Definitions

```yaml
# models/base/pd_model.yml
model:
  name: pd_model
  version: 1.0.0
  risk_tier: tier_1
  
  location:
    type: file
    path: models/pd_model.pkl

# models/base/lgd_model.yml
model:
  name: lgd_model
  version: 1.0.0
  risk_tier: tier_1
  
  location:
    type: file
    path: models/lgd_model.pkl

# models/composite/expected_loss.yml
model:
  name: expected_loss
  version: 1.0.0
  risk_tier: tier_1
  
  # Declare dependencies
  depends_on:
    - pd_model
    - lgd_model
  
  location:
    type: python_class
    path: src/models/expected_loss.py
    class: ExpectedLossModel

# models/external/sentiment.yml
model:
  name: sentiment_analyzer
  version: 1.0.0
  risk_tier: tier_2
  
  location:
    type: huggingface
    repo_id: ProsusAI/finbert
    task: sentiment-analysis
```

### Testing

```bash
# Test all models in dependency order
mrm test

# Test specific subgraph
mrm test --select +expected_loss

# Test and show DAG
mrm debug --show-dag

# Test with parallelism
mrm test --threads 4
```

## Architecture

### DAG Flow

```
1. Project loads all model configs
   ↓
2. DAG built from depends_on declarations
   ↓
3. Topological sort ensures valid order
   ↓
4. Graph operators select subgraphs
   ↓
5. Models tested in dependency order
```

### Catalog Flow

```
1. Catalog built from all model configs
   ↓
2. Model locations registered by name
   ↓
3. ref() references resolved to actual locations
   ↓
4. TestRunner loads models based on source type
   ↓
5. Models cached in backend
```

## CLI Commands

### New Options

```bash
# Debug commands
mrm debug --show-dag        # Show dependency graph
mrm debug --show-catalog    # Show model catalog

# Test with graph operators
mrm test --select +model
mrm test --select model+
mrm test --select +model+
```

## Use Cases

### 1. Hierarchical Models

Test base models before composite models:
```bash
mrm test --select +expected_loss
```

### 2. Impact Analysis

See what's affected by changes to a model:
```bash
mrm test --select pd_model+
```

### 3. Dependency Validation

Ensure all dependencies are tested:
```bash
mrm test --select +portfolio_var
```

### 4. External Models

Validate HuggingFace models with same tests:
```bash
mrm test --models sentiment_analyzer
```

### 5. Parallel Execution

Test independent models concurrently:
```bash
mrm test --threads 8
# Automatically parallelizes independent models
```

## Testing the Implementation

Run the example:

```bash
cd mrm-core
python examples/dag_example.py
```

This creates a project with:
- 4 models in dependency hierarchy
- Complete DAG
- All graph operators demonstrated
- Parallel execution

Output shows:
- Model dependency graph
- Execution levels
- Graph operator examples
- Test results in dependency order

## Benefits

 **Automatic Ordering** - Models tested in correct order  
 **Impact Analysis** - See downstream effects  
 **Dependency Tracking** - Explicit model relationships  
 **Parallel Execution** - Independent models run concurrently  
 **External Models** - HuggingFace, MLflow, S3 support  
 **Type Safety** - Model references checked at runtime  
 **Reproducibility** - Version pins for external models  

## Summary

MRM Core now has complete dbt-style functionality:

| Feature | dbt | MRM Core | Status |
|---------|-----|----------|--------|
| ref() |  |  | Complete |
| DAG |  |  | Complete |
| +model+ |  |  | Complete |
| Topological sort |  |  | Complete |
| Parallel execution |  |  | Complete |
| External sources |  |  | Enhanced |
| Model catalog |  |  | New feature |

**The framework now supports sophisticated model validation workflows with full dependency management!**
