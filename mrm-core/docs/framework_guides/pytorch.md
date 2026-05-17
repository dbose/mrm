# PyTorch Models in mrm

PyTorch models integrate with `mrm-core` via MLflow's PyTorch flavor or custom wrappers. MLflow is the recommended approach for production.

## Approach 1: MLflow PyTorch Flavor (Recommended)

MLflow's PyTorch integration handles model serialization, device management (CPU/GPU), and provides a standard `predict()` interface.

### Training Script with MLflow

```python
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Define model
class CreditScorecardNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CreditScorecardNet(input_dim=20).to(device)

# ... training loop here ...

# Log model to MLflow
with mlflow.start_run(run_name="pytorch_credit_model_v1"):
    # Log training metrics
    mlflow.log_param("hidden_layers", "64,32")
    mlflow.log_param("activation", "relu")
    mlflow.log_metric("train_loss", final_train_loss)
    mlflow.log_metric("val_accuracy", val_accuracy)
    
    # Log PyTorch model with signature
    import pandas as pd
    from mlflow.models.signature import infer_signature
    
    # Create sample input for signature inference
    sample_input = X_val[:5]  # numpy array or pandas DataFrame
    sample_output = model(torch.tensor(sample_input, dtype=torch.float32)).detach().numpy()
    
    signature = infer_signature(sample_input, sample_output)
    
    mlflow.pytorch.log_model(
        model, 
        "model",
        signature=signature,
        input_example=sample_input
    )
    
    # Add MRM metadata
    mlflow.set_tags({
        "mrm.owner": "ml_team",
        "mrm.risk_tier": "tier_2",
        "mrm.framework": "pytorch",
        "mrm.use_case": "credit_scoring"
    })
    
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
```

### Model Configuration (YAML)

```yaml
model:
  name: pytorch_credit_model
  version: 1.0.0
  risk_tier: tier_2
  
  description: "PyTorch neural network for credit scoring"
  owner: ml_team
  methodology: neural_network
  
  location:
    type: mlflow
    model_uri: "runs:/<run_id>/model"
    # Or use registered model:
    # model_uri: "models:/pytorch_credit_model/1"
  
datasets:
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
mrm test pytorch_credit_model
mrm docs generate pytorch_credit_model --compliance standard:sr117
```

---

## Approach 2: Custom Wrapper (Without MLflow)

If you cannot use MLflow, create a wrapper class that implements the sklearn-like interface.

### Wrapper Class

```python
# models/pytorch_wrapper.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class CreditScorecardNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class PyTorchModelWrapper:
    """Wrapper to make PyTorch model compatible with mrm tests"""
    
    def __init__(self, model_path, input_dim=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CreditScorecardNet(input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def predict(self, X):
        """Predict class labels (0 or 1)"""
        with torch.no_grad():
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            probs = self.model(X_tensor).cpu().numpy().flatten()
            return (probs > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        with torch.no_grad():
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            probs = self.model(X_tensor).cpu().numpy().flatten()
            
            # Return shape (n_samples, 2) for binary classification
            return np.column_stack([1 - probs, probs])
```

### Save Model Weights

```python
# After training
torch.save(model.state_dict(), 'models/pytorch_model.pt')
```

### Model Configuration (YAML) — Custom Wrapper

```yaml
model:
  name: pytorch_credit_model
  version: 1.0.0
  risk_tier: tier_2
  
  description: "PyTorch neural network for credit scoring"
  owner: ml_team
  methodology: neural_network
  
  location:
    type: python_class
    path: models/pytorch_wrapper.py
    class: PyTorchModelWrapper
    kwargs:
      model_path: models/pytorch_model.pt
      input_dim: 20
  
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

---

## Best Practices

1. **Use MLflow for production PyTorch models** — handles device management, versioning, and artifact storage
2. **Include input signature in MLflow** — helps with model deployment and inference validation
3. **Log model in evaluation mode** — use `model.eval()` and `torch.no_grad()` during inference
4. **Handle device placement carefully** — wrap CPU/GPU logic in wrapper if not using MLflow
5. **Save preprocessing with the model** — include normalization constants, feature names

### Example: PyTorch Model with Preprocessing

```python
import torch
import mlflow.pytorch

class CreditModelWithPreprocessing(nn.Module):
    def __init__(self, input_dim, mean, std):
        super().__init__()
        # Register preprocessing as buffers (saved with model)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))
        
        # Model layers
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Normalize input
        x = (x - self.mean) / self.std
        
        # Forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Training
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
model = CreditModelWithPreprocessing(input_dim=20, mean=X_mean, std=X_std)

# ... train ...

# Log to MLflow
mlflow.pytorch.log_model(model, "model")
```

---

## GPU Support

PyTorch models may require GPU during training but should support CPU inference for validation environments.

### MLflow Approach

MLflow handles device placement automatically. Models logged with `mlflow.pytorch.log_model()` work on CPU by default when loaded.

### Custom Wrapper Approach

```python
class PyTorchModelWrapper:
    def __init__(self, model_path, input_dim=20, force_cpu=True):
        # Force CPU for validation environments
        if force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = CreditScorecardNet(input_dim).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
```

---

## Troubleshooting

### "RuntimeError: CUDA out of memory"
- Set `force_cpu=True` in wrapper
- Or ensure MLflow loads model with `map_location='cpu'`

### "Model has no attribute 'predict'"
- Verify wrapper implements `predict()` and `predict_proba()`
- Check model is instantiated correctly

### MLflow ImportError
- Install: `pip install mlflow torch`
- Or: `pip install mrm-core[mlflow]`

### "Input shape mismatch"
- Check `input_dim` matches training config
- Verify validation data has correct number of features
- Ensure no target column in input to predict()

---

## See Also

- [Scikit-Learn Models](sklearn.md)
- [TensorFlow/Keras Models](tensorflow.md)
- [Custom Model Wrappers](custom_wrappers.md)
- [MLflow PyTorch Documentation](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)
- [PyTorch Model Saving](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
