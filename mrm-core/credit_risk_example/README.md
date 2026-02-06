# credit_risk_example

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

## Publishing to Databricks

Set your Databricks credentials as environment variables:

```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
```

The `mrm_project.yml` uses dbt-style environment variable interpolation:

```yaml
catalogs:
  databricks:
    host: "{{ env_var('DATABRICKS_HOST') }}"
    token: "{{ env_var('DATABRICKS_TOKEN') }}"
```

Then publish your model:

```bash
mrm publish credit_scorecard
```

The model will be registered in Databricks MLflow Registry.
