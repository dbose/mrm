# Framework Guides

Comprehensive guides for integrating different machine learning frameworks with mrm.

## Quick Links

- **[Scikit-Learn Models](sklearn.md)** — Pickle/joblib files and MLflow integration
- **[PyTorch Models](pytorch.md)** — MLflow pytorch flavor or custom wrappers
- **[TensorFlow/Keras Models](tensorflow.md)** — MLflow tensorflow flavor and SavedModel
- **[Custom Model Wrappers](custom_wrappers.md)** — Build your own interfaces for any model type

---

## Integration Strategy

mrm uses a **hybrid approach** for multi-framework support:

### Option A: MLflow (Recommended for Production)

MLflow provides version tracking, experiment management, and a unified interface across frameworks. Works with:
- Scikit-learn (`mlflow.sklearn`)
- PyTorch (`mlflow.pytorch`)
- TensorFlow/Keras (`mlflow.tensorflow`)
- XGBoost (`mlflow.xgboost`)
- LightGBM (`mlflow.lightgbm`)
- HuggingFace Transformers (`mlflow.transformers`)
- Custom models (`mlflow.pyfunc`)

**Benefits:**
- ✅ Unified `predict()` and `predict_proba()` interface
- ✅ Model versioning and lineage tracking
- ✅ Works locally (file-based) or with remote servers
- ✅ Air-gapped friendly (no external dependencies for local tracking)
- ✅ Audit trail for compliance (runs are immutable, timestamped)
- ✅ Device management (CPU/GPU) handled automatically

**Setup:**
```bash
# Install MLflow
pip install mlflow

# Use local tracking (no server required)
export MLFLOW_TRACKING_URI="file://./mlruns"
```

See individual framework guides for logging patterns.

---

### Option B: Direct Pickle/Joblib (Simple Use Cases)

For prototyping or simple sklearn models, save directly as pickle/joblib files.

```python
import pickle
model.fit(X_train, y_train)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

```yaml
model:
  location:
    type: file
    path: models/model.pkl
```

---

### Option C: Custom Wrappers (Advanced Use Cases)

For non-standard models or custom inference logic, implement a wrapper with `predict()` and `predict_proba()` methods.

```python
class MyModelWrapper:
    def predict(self, X):
        # Your inference logic
        return predictions
    
    def predict_proba(self, X):
        # Your probability logic
        return probabilities
```

See [Custom Model Wrappers](custom_wrappers.md) for details.

---

## Framework Compatibility Matrix

| Framework | Native Pickle | MLflow Flavor | Custom Wrapper | Notes |
|-----------|--------------|---------------|----------------|-------|
| Scikit-Learn | ✅ | ✅ | ✅ | Works out of box |
| XGBoost | ✅ | ✅ | ✅ | MLflow recommended |
| LightGBM | ✅ | ✅ | ✅ | MLflow recommended |
| PyTorch | ⚠️ | ✅ | ✅ | Requires wrapper for pickle |
| TensorFlow/Keras | ⚠️ | ✅ | ✅ | SavedModel or HDF5 |
| HuggingFace | ❌ | ✅ | ✅ | Use MLflow transformers flavor |
| ONNX | ❌ | ✅ | ✅ | Via onnxruntime wrapper |
| API/Endpoints | ❌ | ⚠️ | ✅ | Custom wrapper best |
| SAS/SPSS/R | ❌ | ❌ | ✅ | Subprocess wrapper |

**Legend:**
- ✅ Fully supported
- ⚠️ Requires additional work
- ❌ Not directly supported

---

## Best Practices

1. **Use MLflow for production models** — provides audit trail, versioning, lineage
2. **Local MLflow for air-gapped environments** — no external dependencies, full functionality
3. **Include preprocessing in model artifacts** — sklearn pipelines, PyTorch buffers, Keras preprocessing layers
4. **Log model signatures** — helps with deployment and input validation
5. **Add MRM metadata as tags** — owner, risk_tier, use_case for governance
6. **Test wrappers locally first** — validate `predict()` outputs before running mrm tests
7. **Handle device placement** — ensure CPU fallback for validation environments (especially PyTorch)

---

## Common Workflows

### Workflow 1: Local Development with Scikit-Learn

```python
# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Save
import pickle
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Validate
$ mrm test my_model
```

### Workflow 2: Production PyTorch with MLflow

```python
# Train
import mlflow.pytorch

with mlflow.start_run():
    model = train_pytorch_model()
    mlflow.pytorch.log_model(model, "model")
    run_id = mlflow.active_run().info.run_id

# Update YAML with run_id
# Run validation
$ mrm test pytorch_model --compliance standard:sr117
```

### Workflow 3: API-Based Model

```python
# Create wrapper
class APIWrapper:
    def __init__(self, endpoint):
        self.endpoint = endpoint
    
    def predict(self, X):
        response = requests.post(self.endpoint, json=X.tolist())
        return response.json()['predictions']

# Save wrapper instance
wrapper = APIWrapper('https://api.example.com/predict')
import pickle
with open('models/api_wrapper.pkl', 'wb') as f:
    pickle.dump(wrapper, f)

# Or reference class in YAML
```

---

## Troubleshooting

### "Model has no attribute 'predict'"
- **Solution:** Implement `predict()` method in wrapper, or ensure model was logged correctly to MLflow

### "Input dimension mismatch"
- **Solution:** Check feature count in validation data matches training

### "CUDA out of memory" (PyTorch/TensorFlow)
- **Solution:** Force CPU inference with `device='cpu'` (PyTorch) or `CUDA_VISIBLE_DEVICES="-1"` (TensorFlow)

### "MLflow model loading fails"
- **Solution:** Ensure MLflow version matches training environment, check tracking URI

### "Pickle deserialization fails"
- **Solution:** Add project root to `sys.path` (mrm runner does this automatically for custom classes)

---

## Getting Help

- [Main README](../../../README.md)
- [CCR Example](../../ccr_example/)
- [Credit Risk Example](../../credit_risk_example/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- GitHub Issues for bug reports

---

## Contributing

To add a new framework guide:
1. Create `framework_name.md` in this directory
2. Follow the template from existing guides (sklearn.md, pytorch.md, tensorflow.md)
3. Include working code examples
4. Show both MLflow and custom wrapper approaches
5. Add troubleshooting section
6. Update this README with link and compatibility matrix entry
