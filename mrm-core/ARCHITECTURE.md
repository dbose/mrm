# MRM Core Architecture

## Overview

MRM Core is designed as a modular, extensible framework following the principle of separation of concerns. The architecture mirrors dbt's successful CLI-first approach while addressing the specific needs of model risk management in financial services.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        mrm CLI                               │
│  (Typer-based, user-facing commands)                        │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼─────┐
│ Config │      │ Plugins  │
│ Layer  │      │ System   │
└───┬────┘      └────┬─────┘
    │                │
    │     ┌──────────┴──────────────┐
    │     │                         │
┌───▼─────▼────┐              ┌────▼──────────┐
│  Execution   │              │   Backends    │
│   Engine     │◄─────────────┤  (Adapters)   │
└───┬──────────┘              └───────────────┘
    │                         - Local FS
    │                         - MLflow
┌───▼──────────┐              - Great Expectations
│   Storage    │              - W&B
│   Layer      │              - Custom
└──────────────┘
- Local filesystem
- S3/GCS/Azure
- Git LFS
```

## Core Components

### 1. CLI Layer (`mrm/cli/`)

**Purpose:** User-facing command-line interface

**Key Files:**
- `main.py` - Entry point, command definitions using Typer

**Commands:**
- `init` - Initialize new project
- `test` - Run validation tests
- `list` - List resources (models, tests, suites)
- `debug` - Debug configuration
- `docs` - Generate documentation
- `compile` - Compile project manifest

**Design Principles:**
- Rich terminal output with tables and colors
- Informative error messages
- dbt-style command syntax
- Progress indicators for long-running tasks

### 2. Configuration Layer (`mrm/core/`)

**Purpose:** Project configuration and management

**Key Files:**
- `project.py` - Project class, model selection
- `init.py` - Project initialization logic

**Key Classes:**

```python
class Project:
    """
    Represents an MRM project
    
    Responsibilities:
    - Load project configuration
    - Manage backend connections
    - Select models based on criteria
    - Provide access to project resources
    """
```

**Configuration Files:**
- `mrm_project.yml` - Project-level config (governance, test suites)
- `profiles.yml` - Environment-specific config (backends, credentials)

### 3. Test System (`mrm/tests/`)

**Purpose:** Test definition, registration, and execution

**Key Files:**
- `base.py` - Base test classes (MRMTest, DatasetTest, ModelTest)
- `library.py` - Test registry and discovery
- `builtin/tabular.py` - Dataset-level tests
- `builtin/model.py` - Model-level tests

**Test Registry Pattern:**

```python
# Global registry
registry = TestRegistry()

# Decorator for registration
@register_test
class MyTest(MRMTest):
    name = "category.MyTest"
    
    def run(self, model, dataset, **config):
        # Test logic
        return TestResult(passed=True, score=0.95)
```

**Test Categories:**
- Dataset tests - data quality, distributions, imbalance
- Model tests - performance metrics, discrimination
- Compliance tests - regulatory requirements, stress testing
- Custom tests - user-defined validation logic

### 4. Backend System (`mrm/backends/`)

**Purpose:** Storage abstraction for models, datasets, and test results

**Key Files:**
- `base.py` - Backend adapter interface
- `local.py` - Local filesystem implementation
- `mlflow.py` - MLflow integration

**Backend Interface:**

```python
class BackendAdapter(ABC):
    """Abstract base for all backends"""
    
    @abstractmethod
    def register_model(self, config, artifact) -> str:
        """Register model, return model_id"""
    
    @abstractmethod
    def get_model(self, model_id) -> Any:
        """Retrieve model artifact"""
    
    @abstractmethod
    def log_test_results(self, model_id, results):
        """Store test results"""
    
    @abstractmethod
    def get_test_history(self, model_id, limit) -> List[Dict]:
        """Get historical results"""
```

**Extensibility:**
New backends added by:
1. Implement `BackendAdapter` interface
2. Register in `pyproject.toml` plugins
3. Configure in `profiles.yml`

### 5. Execution Engine (`mrm/engine/`)

**Purpose:** Orchestrate test execution across models

**Key Files:**
- `runner.py` - TestRunner class

**Execution Flow:**

```
1. Load project configuration
2. Select models based on criteria
3. For each model:
   a. Load model artifact
   b. Load datasets
   c. Select tests to run
   d. Execute tests (sequential or parallel)
   e. Collect results
   f. Store to backend
4. Aggregate and display results
```

**Features:**
- Parallel execution (thread pool)
- Fail-fast mode
- Test filtering
- Progress tracking
- Error handling and recovery

### 6. Utilities (`mrm/utils/`)

**Purpose:** Common utility functions

**Key Files:**
- `yaml_utils.py` - YAML loading with env var substitution

**Functions:**
- `load_yaml()` - Load YAML with {{env_var('...')}} support
- `find_project_root()` - Locate project directory
- `validate_*_config()` - Configuration validation

## Data Flow

### Test Execution Flow

```
1. User: mrm test --models credit_scorecard
   │
   ▼
2. CLI: Parse arguments, load project
   │
   ▼
3. Project: Select models matching criteria
   │
   ▼
4. TestRunner: For each model
   │
   ├─► Load model from backend
   │
   ├─► Load datasets (CSV/parquet/pickle)
   │
   ├─► Get tests from config
   │
   ├─► For each test:
   │   ├─► Instantiate test class
   │   ├─► Run test.run(model, dataset, **config)
   │   └─► Collect TestResult
   │
   ├─► Store results to backend
   │
   └─► Return aggregated results
   │
   ▼
5. CLI: Display results table
```

### Model Registration Flow

```
1. User trains model
   │
   ▼
2. Model saved to disk/MLflow/etc
   │
   ▼
3. User creates model YAML config
   │
   ▼
4. mrm test reads config
   │
   ▼
5. Backend loads model based on location type
   │
   ▼
6. Tests run against model
   │
   ▼
7. Results stored to backend with model_id
```

## Extension Points

### 1. Custom Tests

```python
# tests/custom/my_test.py
from mrm.tests.base import MRMTest, TestResult
from mrm.tests.library import register_test

@register_test
class RegulatoryStressTest(MRMTest):
    name = "compliance.RegulatoryStress"
    category = "compliance"
    tags = ["regulatory", "stress_test"]
    
    def run(self, model, dataset, scenario='adverse', **config):
        # Apply stress scenario
        # Run predictions
        # Compare to thresholds
        return TestResult(passed=True, score=0.85)
```

### 2. Custom Backends

```python
# mrm/backends/my_backend.py
from mrm.backends.base import BackendAdapter

class MyBackend(BackendAdapter):
    def __init__(self, **config):
        # Initialize connection
        pass
    
    def register_model(self, model_config, model_artifact):
        # Store model
        return "model_id"
    
    # Implement other methods...
```

Register in `pyproject.toml`:
```toml
[tool.poetry.plugins."mrm.backends"]
mybackend = "mrm.backends.my_backend:MyBackend"
```

### 3. Test Suites

In `mrm_project.yml`:
```yaml
test_suites:
  my_custom_suite:
    - model.Accuracy
    - model.Precision
    - custom.MyTest
```

Use in model config:
```yaml
tests:
  - test_suite: my_custom_suite
```

## Design Patterns

### 1. Plugin Architecture

- Backends registered via entry points
- Tests registered via decorator pattern
- Dynamic loading at runtime
- Loose coupling between components

### 2. Registry Pattern

```python
# Central registry for tests
registry = TestRegistry()

# Registration
@register_test
class MyTest(MRMTest):
    ...

# Discovery
test_class = registry.get("my_test")
```

### 3. Adapter Pattern

```python
# Common interface
class BackendAdapter(ABC):
    @abstractmethod
    def register_model(...): ...

# Multiple implementations
class LocalBackend(BackendAdapter): ...
class MLflowBackend(BackendAdapter): ...
```

### 4. Command Pattern

```python
# Commands as functions
@app.command()
def test(models: str, ...):
    # Execute test command
    pass
```

## Configuration Strategy

### Project-Level (`mrm_project.yml`)

- Governance rules (risk tiers, frequencies)
- Test suites (reusable test collections)
- Backend definitions
- Project metadata

### Environment-Level (`profiles.yml`)

- Backend connections (dev, staging, prod)
- Credentials (via env vars)
- Target selection

### Model-Level (model YAML)

- Model metadata (name, version, tier)
- Model location (path, registry)
- Datasets
- Tests to run

## Testing Strategy

### Unit Tests

```python
# tests/test_tabular.py
def test_missing_values():
    test = MissingValues()
    dataset = pd.DataFrame({'a': [1, 2, np.nan]})
    result = test.run(dataset, threshold=0.5)
    assert not result.passed
```

### Integration Tests

```python
# tests/test_runner.py
def test_full_workflow():
    project = Project.load()
    runner = TestRunner(project.config, project.backend)
    results = runner.run_tests(models)
    assert all(r['all_passed'] for r in results.values())
```

## Performance Considerations

### Parallel Execution

```python
# Thread pool for model testing
with ThreadPoolExecutor(max_workers=threads) as executor:
    futures = {
        executor.submit(run_model_tests, config): config
        for config in model_configs
    }
```

### Caching

- Model artifacts cached in backend
- Test results stored with timestamps
- Configuration compiled once

### Resource Management

- Context managers for backend connections
- Lazy loading of models and datasets
- Memory-efficient data handling

## Security Considerations

- Environment variables for credentials
- No hardcoded secrets
- Backend authentication abstracted
- Audit trail via test result storage

## Future Enhancements

### State Management

- Track test results over time
- Compare current vs. baseline
- Detect drift and degradation

### Documentation Generation

- Auto-generate model cards
- Validation reports from test results
- Template-based rendering

### Lineage Tracking

- Model dependencies
- Dataset lineage
- Test result provenance

### CI/CD Integration

- Pre-commit hooks
- GitHub Actions workflows
- Automated testing pipelines

## Comparison to dbt

| Feature | dbt | MRM Core |
|---------|-----|----------|
| Domain | Data transformation | Model validation |
| Language | SQL/Python | Python |
| Artifacts | Tables/Views | Models/Tests |
| State | Manifest | Test results |
| Backends | Warehouses | Model registries |
| Selection | +model+ | tier:tier_1 |
| Tests | Data tests | Validation tests |

## Conclusion

MRM Core's architecture balances:
- **Simplicity** - Easy to understand and use
- **Extensibility** - Plugin architecture for customization
- **Performance** - Parallel execution, efficient caching
- **Compliance** - Audit trails, governance rules
- **Portability** - Multiple backends, local-first design

The design enables both solo practitioners and enterprise teams to standardize model risk management workflows while maintaining flexibility for their specific needs.
