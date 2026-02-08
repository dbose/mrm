# MRM Core

**Open Source Model Risk Management CLI Framework**

A dbt-inspired command-line tool for automating model validation, documentation, and risk management workflows in financial services.

---

## What's Included

**Core Framework** (`mrm-core/`)
- Complete MRM CLI with Python
- 18+ built-in validation tests (tabular dataset + CCR + compliance)
- dbt-style workflows (DAG, ref(), graph operators)
- Pluggable multi-standard compliance framework
- Validation trigger engine for ongoing monitoring
- Databricks Unity Catalog + MLflow integration
- HuggingFace Hub support

---

## Features

### dbt-Style Workflows

- **DAG Management** - Model dependencies with `depends_on`
- **ref()** - Reference models by name
- **Graph Operators** - `+model+`, `model+`, `+model`
- **Topological Sort** - Automatic dependency ordering
- **Parallel Execution** - Run independent models concurrently

### Model Sources

- **Local Files** - pickle, joblib
- **Python Classes** - Import and instantiate
- **MLflow** - Model registry integration
- **HuggingFace** - Direct Hub integration
- **S3/Cloud** - Cloud storage support
- **Model References** - ref() to other models
- **Catalogs** - Internal model registries

### Testing Framework

- **Dataset Tests** - MissingValues, ClassImbalance, OutlierDetection, FeatureDistribution
- **Model Tests** - Accuracy, ROCAUC, Gini, Precision, Recall, F1Score
- **CCR Validation Tests** - MCConvergence, EPEReasonableness, PFEBacktest, CVASensitivity, WrongWayRisk, ExposureProfileShape, CollateralEffectiveness
- **Compliance Tests** - GovernanceCheck (pluggable per-standard)
- **Custom Tests** - Easy plugin system with `@register_test` decorator
- **Test Suites** - Reusable test collections
- **Parallel Execution** - Multi-threaded test runner

### Pluggable Compliance Framework

Multi-standard regulatory compliance with three-tier plugin discovery:

| Tier | Mechanism | Example |
|------|-----------|---------|
| Bundled | Ships with MRM | CPS 230 (Australia) |
| External pip package | `mrm.compliance` entry point | `pip install mrm-sr117` |
| Custom local | `compliance_paths` in project YAML | `compliance/custom/my_std.py` |

- **ComplianceStandard ABC** - Abstract base for regulatory standards
- **ComplianceRegistry** - Decorator-based discovery (`@register_standard`)
- **Paragraph mapping** - Map tests to regulatory paragraphs with evidence
- **Report generation** - Per-standard compliance reports via `mrm docs generate`
- **Governance checks** - Automated checks loaded from each standard's definition
- **Backward compatibility** - Old configs and imports continue to work with deprecation warnings

**Bundled standard:** APRA CPS 230 (Operational Risk Management, Australia)

### Validation Trigger Engine

Automated triggers for re-validation based on regulatory and operational conditions:

| Trigger Type | Description |
|-------------|-------------|
| SCHEDULED | Calendar-based (quarterly, semi-annual, annual) |
| DRIFT | Data/model drift exceeds threshold |
| BREACH | Back-test breach rate exceeds limit |
| MATERIALITY | Portfolio notional or counterparty count changes |
| REGULATORY | Regulation or policy change mandates re-validation |
| MANUAL | Ad-hoc trigger by model owner |

- Trigger lifecycle: `active` -> `fired` -> `acknowledged` -> `resolved`
- JSON-persisted event log with evidence
- CLI management: `mrm triggers check`, `mrm triggers list`, `mrm triggers resolve`

### Databricks Unity Catalog & MLflow

Publish validated models directly to enterprise model catalogs:

- **`mrm publish`** - One-command publish to Databricks Unity Catalog with MLflow registration
- **`mrm catalog resolve`** - Fetch model metadata by Unity Catalog URI
- **`mrm catalog add`** - Register a model artifact into the catalog
- **`mrm catalog refresh`** - Clear and refresh catalog cache
- **Auto signature inference** - Infers MLflow model signature from validation data, feature names, or model coefficients
- **Three-level namespace** - `catalog.schema.model_name` (e.g. `workspace.default.ccr_monte_carlo`)
- **Environment variable config** - `{{ env_var('DATABRICKS_HOST') }}` substitution in YAML
- **Versioned registration** - Creates registered model versions in Unity Catalog

### CCR Monte Carlo Example

Full worked example of Counterparty Credit Risk validation (`ccr_example/`):

- **Monte Carlo engine** - Vasicek rate dynamics, IRS mark-to-market, netting, collateral
- **Risk metrics** - EPE, PFE, CVA, EAD computation across simulated paths
- **8 validation tests** - Convergence, reasonableness, backtesting, sensitivity, wrong-way risk, exposure shape, collateral effectiveness, governance
- **Compliance report** - CPS 230 regulatory report with paragraph-level evidence mapping
- **Trigger evaluation** - Scheduled and breach-driven re-validation triggers
- **Catalog publish** - Model registered to Databricks Unity Catalog with MLflow tracking

---

## Feature Execution Examples

### End-to-End Validation Run

```
$ cd ccr_example && python run_validation.py

========================================================================
  CCR MONTE CARLO MODEL -- COMPLIANCE VALIDATION
========================================================================

[1/5] Loading model and datasets...
  Model:      CCRMonteCarloModel
  Simulations: 5000
  Time steps:  60
  Dataset:     50 counterparties

[2/5] Running CCR validation tests...

  [PASS] ccr.MCConvergence (score: 0.9720)
  [FAIL] ccr.EPEReasonableness (score: 0.7800)
         Reason: 22% of EPE/notional ratios outside [0.001, 0.1]
  [FAIL] ccr.PFEBacktest (score: 0.8600)
         Reason: PFE breach rate 14.00% exceeds 10% threshold
  [PASS] ccr.CVASensitivity (score: 0.8926)
  [PASS] ccr.WrongWayRisk (score: 0.7752)
  [FAIL] ccr.ExposureProfileShape (score: 0.6000)
         Reason: Exposure profile shape anomaly detected
  [PASS] ccr.CollateralEffectiveness (score: 0.2836)
  [PASS] compliance.GovernanceCheck (score: 1.0000)

  Summary: 5/8 passed, 3 failed

[3/5] Evaluating validation triggers...

  [FIRED] scheduled: Scheduled re-validation: 90 days since last run
          Compliance: CPS 230 Para 34: Periodic review frequency
  [FIRED] breach: PFE breach rate 14.00% exceeds 10%
          Compliance: CPS 230 Para 36: Breach-driven re-validation

[4/5] Generating compliance regulatory report...

  Report written to: reports/ccr_monte_carlo_cps230_report.md
  Report size: 13,059 characters

[5/5] Saving test evidence...

  Evidence saved to: reports/validation_evidence.json

========================================================================
  VALIDATION FAILED -- 5/8 tests passed
========================================================================
```

### Compliance Report Generation via CLI

```
$ mrm docs generate ccr_monte_carlo --compliance standard:cps230

Generating compliance report (cps230) for: ccr_monte_carlo

Running tests for model: ccr_monte_carlo
  Running test: ccr.MCConvergence
     PASSED (score: 0.972)
  Running test: ccr.EPEReasonableness
     FAILED (score: 0.780)
  Running test: ccr.PFEBacktest
     FAILED (score: 0.860)
  Running test: ccr.CVASensitivity
     PASSED (score: 0.893)
  Running test: ccr.WrongWayRisk
     PASSED (score: 0.775)
  Running test: ccr.ExposureProfileShape
     FAILED (score: 0.600)
  Running test: ccr.CollateralEffectiveness
     PASSED (score: 0.284)
  Running test: compliance.GovernanceCheck
     PASSED (score: 1.000)

Report generated: reports/ccr_monte_carlo_cps230_report.md
Report size: 13059 characters
                     Test Results
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Model           ┃ Status  ┃ Tests ┃ Passed ┃ Failed ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ ccr_monte_carlo │  FAILED │     8 │      5 │      3 │
└─────────────────┴─────────┴───────┴────────┴────────┘

2 trigger(s) fired
  -  Scheduled re-validation: 90 days since last run
  -  PFE breach rate 14.00% exceeds 10%
```

### List Available Compliance Standards

```
$ mrm docs list-standards

                    Available Compliance Standards
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Name   ┃ Display Name                               ┃ Jurisdiction ┃ Version ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ cps230 │ APRA CPS 230 -- Operational Risk           │ AU           │ 2024    │
│        │ Management                                 │              │         │
└────────┴────────────────────────────────────────────┴──────────────┴─────────┘
```

### Publish to Databricks Unity Catalog

```
$ mrm publish ccr_monte_carlo

Publishing model: ccr_monte_carlo
Target catalog: databricks
Registering model artifact: models/ccr_monte_carlo.pkl
Loaded validation data for signature: data/validation.csv
MLflow tracking configured for Databricks Unity Catalog
Using MLflow experiment: /Shared/mrm-experiments
Signature inferred from validation data
Successfully registered model 'workspace.default.ccr_monte_carlo'.
Created version '1' of model 'workspace.default.ccr_monte_carlo'.

Model published successfully!

Registered as: ccr_monte_carlo
MLflow Model URI: dbfs:/databricks/mlflow-tracking/.../artifacts/model
Registry Version: 1

Next steps:
  1. View in Databricks MLflow: Models > Registered Models
  2. Reference in other projects using catalog URIs
  3. Run: mrm catalog resolve databricks_uc://ccr_monte_carlo
```

---

## CLI Commands

```bash
# Initialize new project
mrm init [project_name] --template=credit_risk

# List resources
mrm list models --tier=tier_1
mrm list tests
mrm list suites

# Run validation tests
mrm test --models ccr_monte_carlo
mrm test --select tier:tier_1
mrm test --select +pd_model  # With dependencies

# Generate compliance documentation (dbt-style)
mrm docs generate ccr_monte_carlo --compliance standard:cps230
mrm docs generate ccr_monte_carlo -c standard:cps230 -o report.md
mrm docs list-standards

# Manage validation triggers
mrm triggers check ccr_monte_carlo
mrm triggers list --model ccr_monte_carlo
mrm triggers resolve ccr_monte_carlo

# Publish to Databricks Unity Catalog
mrm publish ccr_monte_carlo
mrm publish ccr_monte_carlo --to databricks --version 1.0.0

# Catalog operations
mrm catalog resolve databricks_uc://workspace.default/ccr_monte_carlo
mrm catalog add --name ccr_monte_carlo --from-file models/ccr_monte_carlo.pkl
mrm catalog refresh

# Debug project
mrm debug --show-config --show-tests --show-dag

# Show version
mrm version
```

---

## Quick Start

```bash
# Install
cd mrm-core
pip install -e .

# Or with Poetry
poetry install

# Run CCR example end-to-end
cd ccr_example
python setup_ccr_example.py    # Generate synthetic data + pickle model
python run_validation.py        # Run 8 tests, evaluate triggers, generate report

# Via the CLI
mrm docs generate ccr_monte_carlo --compliance standard:cps230
mrm docs list-standards
mrm triggers check ccr_monte_carlo

# Publish to Databricks (requires credentials)
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
mrm publish ccr_monte_carlo
```

---

## Configuration

### Model YAML (`ccr_monte_carlo.yml`)

```yaml
model:
  name: ccr_monte_carlo
  version: 1.0.0
  risk_tier: tier_1
  owner: market_risk_team
  validation_frequency: quarterly

  location:
    type: file
    path: models/ccr_monte_carlo.pkl

tests:
  - test_suite: ccr_validation
  - test: ccr.MCConvergence
    config:
      max_relative_diff: 0.05

triggers:
  - type: scheduled
    schedule_days: 90
    compliance_reference: "CPS 230 Para 34: Periodic review frequency"
  - type: breach
    threshold: 0.10
    compliance_reference: "CPS 230 Para 36: Breach-driven re-validation"

compliance:
  standards:
    cps230:
      mapping:
        governance:
          - paragraph: "Para 8-10"
            requirement: "Risk identification and classification"
            evidence: "Model classified as Tier 1"
        risk_management:
          - paragraph: "Para 15-18"
            requirement: "Risk assessment methodology"
            evidence: "EPE reasonableness test validates exposure bounds"
```

### Project YAML (`mrm_project.yml`)

```yaml
name: ccr_example
version: 1.0.0

governance:
  risk_tiers:
    tier_1:
      validation_frequency: quarterly
      compliance_reference: "CPS 230 Para 8-14"
      required_tests:
        - ccr.MCConvergence
        - ccr.EPEReasonableness
        - ccr.PFEBacktest
        - ccr.CVASensitivity
        - ccr.WrongWayRisk
        - ccr.ExposureProfileShape
        - ccr.CollateralEffectiveness
        - compliance.GovernanceCheck

compliance:
  default_standard: cps230
  standards:
    cps230:
      enabled: true

compliance_paths:
  - compliance/custom

catalogs:
  databricks:
    type: databricks_unity
    host: "{{ env_var('DATABRICKS_HOST') }}"
    token: "{{ env_var('DATABRICKS_TOKEN') }}"
    catalog: workspace
    schema: default
    mlflow_registry: true
    cache_ttl_seconds: 300
```

---

## Build Custom Tests

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

## Build Custom Compliance Standards

```python
from mrm.compliance.base import ComplianceStandard
from mrm.compliance.registry import register_standard

@register_standard
class SR117Standard(ComplianceStandard):
    name = "sr117"
    display_name = "Fed SR 11-7 -- Model Risk Management Guidance"
    jurisdiction = "US"
    version = "2011"

    def get_paragraphs(self):
        return { ... }

    def get_test_mapping(self):
        return { ... }

    def get_governance_checks(self):
        return { ... }

    def generate_report(self, model_name, model_config, test_results,
                        trigger_events=None, output_path=None):
        # Build report string
        return report_text
```

---

## Project Structure

```
mrm/
├── mrm-core/
│   ├── mrm/
│   │   ├── cli/                        # CLI interface (Typer)
│   │   │   └── main.py                 #   All commands: test, docs, triggers, publish, catalog
│   │   ├── core/                       # Core functionality
│   │   │   ├── project.py              #   Project loader + model selection
│   │   │   ├── dag.py                  #   Dependency graph
│   │   │   ├── catalog.py              #   ModelCatalog, ModelRef, ModelSource
│   │   │   ├── triggers.py             #   Validation trigger engine
│   │   │   └── catalog_backends/
│   │   │       └── databricks_unity.py #   Databricks UC + MLflow backend
│   │   ├── compliance/                 # Pluggable compliance framework
│   │   │   ├── base.py                 #   ComplianceStandard ABC
│   │   │   ├── registry.py             #   ComplianceRegistry + @register_standard
│   │   │   ├── report_generator.py     #   Generic entry point
│   │   │   └── builtin/
│   │   │       └── cps230.py           #   APRA CPS 230 implementation
│   │   ├── tests/                      # Test framework
│   │   │   ├── base.py                 #   MRMTest, ComplianceTest, TestResult
│   │   │   ├── library.py              #   TestRegistry + @register_test
│   │   │   └── builtin/
│   │   │       ├── tabular.py          #   Dataset validation tests
│   │   │       └── ccr.py              #   CCR + governance tests (8 tests)
│   │   ├── engine/                     # Test runner
│   │   │   └── runner.py               #   TestRunner with model loading
│   │   ├── backends/                   # Storage backends (local, MLflow)
│   │   ├── reporting/                  # Legacy report module (deprecation shim)
│   │   └── utils/                      # YAML loading, helpers
│   ├── ccr_example/                    # CCR Monte Carlo worked example
│   │   ├── models/ccr/
│   │   │   ├── ccr_monte_carlo.py      #   Monte Carlo simulation engine
│   │   │   └── ccr_monte_carlo.yml     #   Model config + compliance mapping
│   │   ├── data/                       #   Synthetic counterparty data
│   │   ├── reports/                    #   Generated reports + JSON evidence
│   │   ├── setup_ccr_example.py        #   Bootstrap: generate data + pickle
│   │   ├── run_validation.py           #   End-to-end validation runner
│   │   └── mrm_project.yml             #   Project config + catalog config
│   ├── credit_risk_example/            # Credit risk example
│   ├── examples/                       # Additional examples
│   └── docs/                           # Documentation
```

---

## Requirements

- Python 3.9+
- Dependencies (auto-installed):
  - typer (>=0.12)
  - pydantic
  - pyyaml
  - rich
  - pandas, numpy, scikit-learn, scipy

Optional:
- mlflow (for MLflow backend + Databricks Unity Catalog)
- databricks-sdk (for Databricks table listing)
- transformers (for HuggingFace)
- great-expectations (for GE integration)

## Documentation

- [Getting Started](mrm-core/docs/GETTING_STARTED.md)
- [Architecture](mrm-core/docs/ARCHITECTURE.md)
- [Model References](mrm-core/docs/MODEL_REFERENCES.md)
- [DAG Features](mrm-core/docs/DAG_FEATURES.md)
- [Complete Features](mrm-core/docs/COMPLETE_FEATURES.md)
- [MRM Lifecycle](mrm-core/docs/MRM_LIFECYCLE.md)
- [Databricks Integration](mrm-core/designs/databricks_unity_catalog.md)

## Contributing

See [CONTRIBUTING.md](mrm-core/CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - See [LICENSE](mrm-core/LICENSE)

## Version

**MRM Core v0.1.0**

Built: February 2026
