# credit_risk_example

MRM project for model risk management — demonstrates credit scoring model validation with scikit-learn.

## Getting Started

```bash
# Run validation tests
mrm test credit_scorecard

# Generate compliance report
mrm docs generate credit_scorecard --compliance standard:cps230
```

## Project Structure

- `models/` - Model definitions (YAML files) and model artifacts (pickle files)
- `tests/custom/` - Custom test implementations
- `data/` - Training and validation datasets
- `docs/` - Documentation templates
- `mlruns/` - Local MLflow tracking (used by MLflow integration)

## Training and Logging Models

This example shows two approaches: simple pickle files and MLflow tracking.

### Approach 1: Simple Pickle (Current)

```python
import pickle
from sklearn.linear_model import LogisticRegression

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save to pickle
with open('models/credit_scorecard.pkl', 'wb') as f:
    pickle.dump(model, f)
```

Then reference in `models/credit_risk/scorecard.yml`:

```yaml
model:
  location:
    type: file
    path: models/credit_scorecard.pkl
```

### Approach 2: MLflow Tracking (Recommended for Production)

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Enable local MLflow tracking
mlflow.set_tracking_uri("file://./mlruns")

# Enable autologging
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="credit_scorecard_v1"):
    # Train model
    model = LogisticRegression(C=1.0, max_iter=100)
    model.fit(X_train, y_train)
    
    # Log custom metrics
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log model (autolog does this automatically)
    mlflow.sklearn.log_model(model, "model")
    
    # Add MRM metadata
    mlflow.set_tags({
        "mrm.owner": "credit_risk_team",
        "mrm.risk_tier": "tier_1",
        "mrm.use_case": "consumer_lending"
    })
    
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
```

Then reference in `models/credit_risk/scorecard.yml`:

```yaml
model:
  location:
    type: mlflow
    model_uri: "runs:/<run_id>/model"
    # Or use registered model: "models:/credit_scorecard/1"
```

See [Framework Guides](../../docs/framework_guides/) for PyTorch, TensorFlow, and custom wrapper examples.

## Running Validation

```bash
# Run all tests for the credit scorecard
mrm test credit_scorecard

# Generate compliance report for specific standard
mrm docs generate credit_scorecard --compliance standard:cps230
mrm docs generate credit_scorecard --compliance standard:sr117
mrm docs generate credit_scorecard --compliance standard:euaiact
mrm docs generate credit_scorecard --compliance standard:osfie23
```

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
