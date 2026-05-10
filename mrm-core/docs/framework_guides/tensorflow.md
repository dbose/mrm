# TensorFlow/Keras Models in mrm

TensorFlow and Keras models integrate with `mrm-core` via MLflow's TensorFlow flavor or custom wrappers. MLflow is the recommended approach for production.

## Approach 1: MLflow TensorFlow Flavor (Recommended)

MLflow supports both TensorFlow 2.x (with Keras integrated) and legacy TensorFlow 1.x models.

### Training Script with MLflow (TensorFlow 2.x / Keras)

```python
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd

# Define Keras model
def build_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_dim=input_dim),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

# Train model
model = build_model(input_dim=20)

# Enable MLflow autologging (captures params, metrics, model)
mlflow.tensorflow.autolog()

with mlflow.start_run(run_name="keras_credit_model_v1"):
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
    
    # Log custom metrics (autolog captures most, but you can add more)
    mlflow.log_metric("final_val_accuracy", val_acc)
    mlflow.log_metric("final_val_auc", val_auc)
    
    # Log model with signature
    from mlflow.models.signature import infer_signature
    
    sample_input = X_val[:5]
    sample_output = model.predict(sample_input)
    signature = infer_signature(sample_input, sample_output)
    
    mlflow.tensorflow.log_model(
        model,
        "model",
        signature=signature,
        input_example=sample_input
    )
    
    # Add MRM metadata
    mlflow.set_tags({
        "mrm.owner": "ml_team",
        "mrm.risk_tier": "tier_2",
        "mrm.framework": "tensorflow",
        "mrm.use_case": "credit_scoring"
    })
    
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
```

### Model Configuration (YAML)

```yaml
model:
  name: keras_credit_model
  version: 1.0.0
  risk_tier: tier_2
  
  description: "Keras neural network for credit scoring"
  owner: ml_team
  methodology: neural_network
  
  location:
    type: mlflow
    model_uri: "runs:/<run_id>/model"
    # Or use registered model:
    # model_uri: "models:/keras_credit_model/1"
  
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
mrm test keras_credit_model
mrm docs generate keras_credit_model --compliance standard:euaiact
```

---

## Approach 2: SavedModel Format via MLflow

TensorFlow's SavedModel format is production-grade and recommended for serving.

### Save and Log SavedModel

```python
import mlflow

with mlflow.start_run():
    # Train model
    model = build_model(input_dim=20)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    
    # Save as SavedModel
    model.save('models/saved_model/')
    
    # Log to MLflow
    mlflow.tensorflow.log_model(
        tf_saved_model_dir='models/saved_model/',
        tf_meta_graph_tags=None,
        tf_signature_def_key='serving_default',
        artifact_path='model'
    )
    
    run_id = mlflow.active_run().info.run_id
```

MLflow automatically creates a wrapper with `predict()` and `predict_proba()` methods.

---

## Approach 3: Custom Wrapper (Without MLflow)

For environments without MLflow, create a wrapper class.

### Wrapper Class

```python
# models/tensorflow_wrapper.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

class TensorFlowModelWrapper:
    """Wrapper to make TensorFlow/Keras model compatible with mrm tests"""
    
    def __init__(self, model_path):
        """
        Load TensorFlow model
        
        Args:
            model_path: Path to SavedModel directory or .h5 file
        """
        self.model = keras.models.load_model(model_path)
    
    def predict(self, X):
        """Predict class labels (0 or 1)"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        probs = self.model.predict(X, verbose=0).flatten()
        return (probs > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        probs = self.model.predict(X, verbose=0).flatten()
        
        # Return shape (n_samples, 2) for binary classification
        return np.column_stack([1 - probs, probs])
```

### Save Model

```python
# SavedModel format (recommended)
model.save('models/credit_model/')

# Or HDF5 format
model.save('models/credit_model.h5')
```

### Model Configuration (YAML) — Custom Wrapper

```yaml
model:
  name: keras_credit_model
  version: 1.0.0
  risk_tier: tier_2
  
  description: "Keras neural network for credit scoring"
  owner: ml_team
  methodology: neural_network
  
  location:
    type: python_class
    path: models/tensorflow_wrapper.py
    class: TensorFlowModelWrapper
    kwargs:
      model_path: models/credit_model/  # SavedModel directory
      # Or: model_path: models/credit_model.h5
  
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

1. **Use MLflow for production TensorFlow models** — handles versioning, serving, and deployment
2. **Prefer SavedModel over HDF5** — SavedModel is TensorFlow's native format, better for serving
3. **Include preprocessing layers in the model** — use `keras.layers.Normalization()` for scaling
4. **Log model signatures** — helps with model deployment and input validation
5. **Use autologging** — captures hyperparameters, metrics, and model automatically

### Example: Keras Model with Preprocessing Layers

```python
import tensorflow as tf
from tensorflow import keras

# Create preprocessing layer
normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)  # Compute mean/variance from training data

# Build model with preprocessing
model = keras.Sequential([
    normalizer,  # Normalization built into model
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)

# Log to MLflow
with mlflow.start_run():
    mlflow.tensorflow.log_model(model, "model")
```

Now the model handles normalization internally — no preprocessing needed at inference time.

---

## TensorFlow 1.x Models

For legacy TensorFlow 1.x models (session-based), use MLflow's `tf_saved_model_dir` parameter.

```python
import mlflow.tensorflow

with mlflow.start_run():
    # TensorFlow 1.x session-based model
    mlflow.tensorflow.log_model(
        tf_saved_model_dir='models/tf1_saved_model/',
        tf_meta_graph_tags=['serve'],
        tf_signature_def_key='serving_default',
        artifact_path='model'
    )
```

MLflow creates a wrapper that loads the SavedModel and exposes `predict()`.

---

## Multiclass Classification

For multiclass models (>2 classes), mrm tests automatically handle the output shape.

```python
# Multiclass model (3+ classes)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_dim=20),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')  # Multiclass output
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

The wrapper's `predict_proba()` returns shape `(n_samples, num_classes)` —mrm tests handle this automatically.

---

## GPU Support

TensorFlow automatically uses GPU if available. For CPU-only inference environments:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import tensorflow as tf
```

Or configure in wrapper:

```python
class TensorFlowModelWrapper:
    def __init__(self, model_path, force_cpu=True):
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        self.model = keras.models.load_model(model_path)
```

---

## Troubleshooting

### "No module named 'tensorflow'"
- Install: `pip install tensorflow mlflow`
- Or: `pip install mrm-core[mlflow] tensorflow`

### "Model has no attribute 'predict'"
- Verify model loaded correctly: `model = keras.models.load_model('path')`
- Check wrapper implements `predict()` and `predict_proba()`

### "Input shape mismatch"
- Check input features match model's `input_dim`
- Verify validation data has correct feature count
- Ensure no target column in input to predict()

### "CUDA error" / GPU issues
- Force CPU: `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`
- Or load model with: `tf.device('/cpu:0')`

### MLflow SavedModel loading fails
- Ensure TensorFlow version matches training environment
- Check `tf_signature_def_key` (default: `'serving_default'`)
- Verify SavedModel directory structure is intact

---

## See Also

- [Scikit-Learn Models](sklearn.md)
- [PyTorch Models](pytorch.md)
- [Custom Model Wrappers](custom_wrappers.md)
- [MLflow TensorFlow Documentation](https://mlflow.org/docs/latest/python_api/mlflow.tensorflow.html)
- [TensorFlow SavedModel Guide](https://www.tensorflow.org/guide/saved_model)
- [Keras Model Saving](https://www.tensorflow.org/guide/keras/save_and_serialize)
