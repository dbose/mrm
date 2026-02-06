"""Project initialization for MRM"""

from pathlib import Path
from typing import Optional
import shutil


def initialize_project(
    project_name: str,
    template: Optional[str] = None,
    backend: str = "local"
) -> Path:
    """
    Initialize a new MRM project
    
    Args:
        project_name: Name of the project
        template: Template to use (optional)
        backend: Default backend type
    
    Returns:
        Path to created project
    """
    project_path = Path.cwd() / project_name
    
    if project_path.exists():
        raise FileExistsError(f"Directory already exists: {project_path}")
    
    # Create directory structure
    project_path.mkdir(parents=True)
    (project_path / "models").mkdir()
    (project_path / "tests" / "custom").mkdir(parents=True)
    (project_path / "docs" / "templates").mkdir(parents=True)
    (project_path / "data").mkdir()
    (project_path / ".mrm").mkdir()
    
    # Create .gitignore
    gitignore_content = """# MRM
.mrm/
*.pkl
*.joblib

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv

# Data
*.csv
*.parquet
*.db
"""
    
    with open(project_path / ".gitignore", 'w') as f:
        f.write(gitignore_content)
    
    # Create mrm_project.yml
    project_config = f"""name: {project_name}
version: 1.0.0

# Model risk governance
governance:
  risk_tiers:
    tier_1: 
      description: "High materiality, high complexity"
      validation_frequency: quarterly
      required_tests: [backtesting, sensitivity, stability]
    tier_2:
      description: "Medium materiality"
      validation_frequency: semi_annual
      required_tests: [backtesting, sensitivity]
    tier_3:
      description: "Low materiality"
      validation_frequency: annual
      required_tests: [backtesting]

# Test suites (reusable)
test_suites:
  data_quality:
    - tabular_dataset.MissingValues
    - tabular_dataset.OutlierDetection
    - tabular_dataset.FeatureDistribution
  
  classification_performance:
    - model.Accuracy
    - model.ROCAUC
    - model.Precision
    - model.Recall
    - model.F1Score
  
  credit_risk:
    - model.Gini
    - model.ROCAUC
    - tabular_dataset.ClassImbalance

# Backend configuration
backends:
  default: local
  
  local:
    type: filesystem
    path: ~/.mrm/data

# Documentation templates
docs:
  templates:
    - model_description
    - validation_report

# Custom test paths
test_paths:
  - tests/custom
"""
    
    with open(project_path / "mrm_project.yml", 'w') as f:
        f.write(project_config)
    
    # Create profiles.yml
    profiles_config = f"""# MRM Profile Configuration
mrm:
  outputs:
    dev:
      backend: local
      
    prod:
      backend: local
      
  target: dev
"""
    
    with open(project_path / "profiles.yml", 'w') as f:
        f.write(profiles_config)
    
    # Create README
    readme_content = f"""# {project_name}

MRM project for model risk management.

## Getting Started

```bash
# List models
mrm list models

# Run validation tests
mrm test --models my_model

# Generate documentation
mrm docs generate
```

## Project Structure

- `models/` - Model definitions (YAML files)
- `tests/custom/` - Custom test implementations
- `data/` - Training and validation datasets
- `docs/` - Documentation templates

## Adding a Model

1. Create model definition in `models/`
2. Define tests in the YAML file
3. Run tests: `mrm test --models model_name`
"""
    
    with open(project_path / "README.md", 'w') as f:
        f.write(readme_content)
    
    # Create example model if template is specified
    if template == "credit_risk":
        _create_credit_risk_example(project_path)
    
    return project_path


def _create_credit_risk_example(project_path: Path):
    """Create credit risk model example"""
    
    models_dir = project_path / "models" / "credit_risk"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_config = """# Credit Risk Scorecard Model
model:
  name: credit_scorecard
  version: 1.0.0
  risk_tier: tier_1
  
  description: "Probability of Default model for consumer credit"
  owner: credit_risk_team
  use_case: consumer_lending
  methodology: logistic_regression
  
  # Model location (update with your model path)
  location:
    type: file
    path: models/credit_scorecard.pkl
  
# Datasets
datasets:
  training:
    type: parquet
    path: data/training.parquet
    
  validation:
    type: parquet  
    path: data/validation.parquet

# Validation tests
tests:
  # Use pre-defined test suite
  - test_suite: credit_risk
  
  # Additional specific tests
  - test: tabular_dataset.MissingValues
    config:
      dataset: validation
      threshold: 0.05
      
  - test: model.Gini
    config:
      dataset: validation
      min_score: 0.40

# Ongoing monitoring
monitoring:
  metrics:
    - gini_coefficient
    - population_stability_index
    - default_rate
"""
    
    with open(models_dir / "scorecard.yml", 'w') as f:
        f.write(model_config)
