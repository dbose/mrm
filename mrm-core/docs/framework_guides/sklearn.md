# Scikit-Learn Models in mrm

Scikit-learn models work out of the box with `mrm-core` using two approaches: local pickle/joblib files or MLflow tracking.

## Approach 1: Local Pickle/Joblib (Simplest)

### Training Script

```python
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Train model
X, y = make_classification(n_samples=1000, n_features=20)
model = LogisticRegression()
model.fit(X, y)

# Save to pickle
with open('models/my_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Or use joblib for better performance with large numpy arrays
import joblib
joblib.dump(model, 'models/my_model.pkl')
```

### Model Configuration (YAML)

```yaml
model:
  name: my_sklearn_model
  version: 1.0.0
  risk_tier: tier_2
  
  description: "Logistic regression for credit scoring"
  owner: data_science_team
  methodology: logistic_regression
  
  location:
    type: file
    path: models/my_model.pkl
  
datasets:
  training:
    type: csv
    path: data/training.csv
    
  validation:
    type: csv
    path: data/validation.csv

tests:
  - test: model.Accuracy
    config:
      dataset: validation
      min_score: 0.70
      
  - test: model.ROCAUC
    config:
      dataset: validation
      min_score: 0.75
```

### Run Validation

```bash
mrm test my_sklearn_model
mrm docs generate my_sklearn_model --compliance standard:cps230
```

---

## Approach 2: MLflow Tracking (Production Recommended)

MLflow provides version tracking, experiment management, and artifact storage. Works locally or with remote tracking servers.

### Setup Local MLflow Tracking

```bash
# In your project directory
export MLFLOW_TRACKING_URI="file://./mlruns"
```

### Training Script with MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Enable MLflow autologging (captures params, metrics, model)
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="credit_scorecard_v1"):
    # Train model
    X, y = make_classification(n_samples=1000, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LogisticRegression(C=1.0, max_iter=100)
    model.fit(X_train, y_train)
    
    # Log custom metrics
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))
    
    # Log model (autolog does this, but explicit example shown)
    mlflow.sklearn.log_model(model, "model")
    
    # Log additional metadata
    mlflow.set_tags({
        "mrm.owner": "data_science_team",
        "mrm.risk_tier": "tier_2",
        "mrm.use_case": "credit_scoring"
    })
    
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
```

### Model Configuration (YAML) — MLflow Source

```yaml
model:
  name: credit_scorecard
  version: 1.0.0
  risk_tier: tier_2
  
  description: "Logistic regression for credit scoring"
  owner: data_science_team
  methodology: logistic_regression
  
  location:
    type: mlflow
    model_uri: "runs:/<run_id>/model"  # Replace <run_id> with actual ID
    # Or use registered model:
    # model_uri: "models:/credit_scorecard/1"
  
datasets:
  validation:
    type: csv
    path: data/validation.csv

tests:
  - test: model.Accuracy
    config:
      dataset: validation
      min_score: 0.70
```

### Register Model in MLflow Model Registry

```python
import mlflow

client = mlflow.tracking.MlflowClient()

# Register model from a run
model_uri = f"runs:/{run_id}/model"
model_details = client.create_registered_model("credit_scorecard")

# Create a model version
model_version = client.create_model_version(
    name="credit_scorecard",
    source=model_uri,
    run_id=run_id
)

print(f"Model version: {model_version.version}")
```

Then use in YAML:

```yaml
location:
  type: mlflow
  model_uri: "models:/credit_scorecard/1"
```

---

## Best Practices

1. **Use MLflow for production models** — provides audit trail, versioning, lineage
2. **Local pickle is fine for dev/prototyping** — simple, no setup required
3. **Always include test datasets in your YAML** — mrm needs data to validate
4. **Save preprocessing pipelines with the model** — use `sklearn.pipeline.Pipeline` to bundle transformations

### Example: Pipeline with Preprocessing

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Save entire pipeline
import joblib
joblib.dump(pipeline, 'models/credit_scorecard.pkl')

# Or log to MLflow
mlflow.sklearn.log_model(pipeline, "model")
```

The pipeline's `predict()` and `predict_proba()` methods work with mrm tests automatically.

---

## Troubleshooting

### "Model file not found"
- Ensure `location.path` is relative to project root (where `mrm_project.yml` lives)
- Use `models/` subdirectory convention for clarity

### "AttributeError: 'NoneType' object has no attribute 'predict'"
- Check pickle file loads correctly: `pickle.load(open('models/model.pkl', 'rb'))`
- Verify file is not corrupted

### MLflow ImportError
- Install MLflow: `pip install mlflow`
- Or install mrm with MLflow extras: `pip install mrm-core[mlflow]`

### Tests fail with "could not extract target column"
- Ensure validation dataset has target variable as last column
- Or specify `target_column` in test config

---

## See Also

- [PyTorch Models](pytorch.md)
- [TensorFlow/Keras Models](tensorflow.md)
- [Custom Model Wrappers](custom_wrappers.md)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
