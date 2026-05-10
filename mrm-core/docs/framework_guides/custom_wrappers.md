# Custom Model Wrappers in mrm

If your model doesn't fit standard patterns (sklearn, PyTorch, TensorFlow via MLflow), or you have custom inference logic, you can implement a custom wrapper.

## The Wrapper Interface

Any Python class with `predict()` and optionally `predict_proba()` methods works with mrm tests.

### Minimum Required Interface

```python
class MyModelWrapper:
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: Input features (pandas DataFrame or numpy array)
        
        Returns:
            numpy array of predictions (shape: n_samples,)
        """
        # Your inference logic here
        pass
```

### Full Interface (for Probabilistic Models)

```python
class MyModelWrapper:
    def predict(self, X):
        """Predict class labels (0/1 for binary, 0-k for multiclass)"""
        pass
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Returns:
            numpy array of shape (n_samples, n_classes)
            For binary: [[P(class=0), P(class=1)], ...]
        """
        pass
```

---

## Use Case 1: Custom Preprocessing Pipeline

Wrap a model with custom preprocessing that's not captured in the model artifact.

```python
# models/custom_wrapper.py
import numpy as np
import pandas as pd
import pickle

class CustomPreprocessingWrapper:
    def __init__(self, model_path, feature_config_path):
        """
        Load model and preprocessing config
        
        Args:
            model_path: Path to pickled model
            feature_config_path: Path to feature engineering config
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(feature_config_path, 'r') as f:
            import json
            self.feature_config = json.load(f)
    
    def _preprocess(self, X):
        """Custom feature engineering"""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        
        # Example: Log transform certain columns
        for col in self.feature_config.get('log_transform', []):
            df[col] = np.log1p(df[col])
        
        # Example: Interaction features
        for interaction in self.feature_config.get('interactions', []):
            col1, col2 = interaction
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        return df.values
    
    def predict(self, X):
        X_processed = self._preprocess(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        X_processed = self._preprocess(X)
        return self.model.predict_proba(X_processed)
```

### YAML Configuration

```yaml
model:
  name: custom_preprocessing_model
  location:
    type: python_class
    path: models/custom_wrapper.py
    class: CustomPreprocessingWrapper
    kwargs:
      model_path: models/base_model.pkl
      feature_config_path: config/features.json
```

---

## Use Case 2: Ensemble of Multiple Models

Wrap multiple models with custom ensemble logic.

```python
# models/ensemble_wrapper.py
import pickle
import numpy as np

class EnsembleWrapper:
    def __init__(self, model_paths, weights=None):
        """
        Load multiple models and combine predictions
        
        Args:
            model_paths: List of paths to model files
            weights: Optional weights for weighted average (else uniform)
        """
        self.models = []
        for path in model_paths:
            with open(path, 'rb') as f:
                self.models.append(pickle.load(f))
        
        if weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()
    
    def predict_proba(self, X):
        """Weighted average of probabilities"""
        all_probs = np.array([model.predict_proba(X) for model in self.models])
        weighted_probs = np.average(all_probs, axis=0, weights=self.weights)
        return weighted_probs
    
    def predict(self, X):
        """Class with highest weighted probability"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
```

### YAML Configuration

```yaml
model:
  name: ensemble_model
  location:
    type: python_class
    path: models/ensemble_wrapper.py
    class: EnsembleWrapper
    kwargs:
      model_paths:
        - models/model_a.pkl
        - models/model_b.pkl
        - models/model_c.pkl
      weights: [0.5, 0.3, 0.2]
```

---

## Use Case 3: API-Based Model (e.g., Hosted Endpoint)

Wrap a REST API or gRPC endpoint.

```python
# models/api_wrapper.py
import requests
import numpy as np
import pandas as pd
import os

class APIModelWrapper:
    def __init__(self, endpoint_url, api_key_env_var='MODEL_API_KEY'):
        """
        Wrap a REST API endpoint
        
        Args:
            endpoint_url: Full URL to prediction endpoint
            api_key_env_var: Environment variable containing API key
        """
        self.endpoint_url = endpoint_url
        self.api_key = os.environ.get(api_key_env_var)
        
        if not self.api_key:
            raise ValueError(f"API key not found in {api_key_env_var}")
    
    def predict(self, X):
        """Call API for predictions"""
        if isinstance(X, pd.DataFrame):
            payload = X.to_dict(orient='records')
        else:
            payload = X.tolist()
        
        response = requests.post(
            self.endpoint_url,
            json={'instances': payload},
            headers={'Authorization': f'Bearer {self.api_key}'},
            timeout=30
        )
        
        response.raise_for_status()
        predictions = response.json()['predictions']
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Call API for probabilities"""
        # Similar logic, but request probabilities
        if isinstance(X, pd.DataFrame):
            payload = X.to_dict(orient='records')
        else:
            payload = X.tolist()
        
        response = requests.post(
            f"{self.endpoint_url}/proba",
            json={'instances': payload},
            headers={'Authorization': f'Bearer {self.api_key}'},
            timeout=30
        )
        
        response.raise_for_status()
        probabilities = response.json()['probabilities']
        return np.array(probabilities)
```

### YAML Configuration

```yaml
model:
  name: api_hosted_model
  location:
    type: python_class
    path: models/api_wrapper.py
    class: APIModelWrapper
    kwargs:
      endpoint_url: "https://api.example.com/v1/predict"
      api_key_env_var: "CREDIT_MODEL_API_KEY"
```

**Security Note:** Store API keys in environment variables, never in YAML or code.

---

## Use Case 4: Legacy System Integration (SAS, SPSS, R)

Call external scripts/processes for prediction.

```python
# models/legacy_wrapper.py
import subprocess
import pandas as pd
import numpy as np
import tempfile
import os

class SASModelWrapper:
    def __init__(self, sas_script_path, sas_executable='/usr/bin/sas'):
        """
        Wrap a SAS scoring script
        
        Args:
            sas_script_path: Path to SAS script that reads input CSV and writes output CSV
            sas_executable: Path to SAS executable
        """
        self.sas_script = sas_script_path
        self.sas_executable = sas_executable
    
    def predict(self, X):
        """Run SAS script for predictions"""
        # Write input to temp CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_in:
            input_path = f_in.name
            if isinstance(X, pd.DataFrame):
                X.to_csv(f_in, index=False)
            else:
                pd.DataFrame(X).to_csv(f_in, index=False)
        
        output_path = input_path.replace('.csv', '_output.csv')
        
        try:
            # Run SAS script
            subprocess.run(
                [self.sas_executable, '-sysin', self.sas_script,
                 '-set', f'INPUT_FILE={input_path}',
                 '-set', f'OUTPUT_FILE={output_path}'],
                check=True,
                timeout=300
            )
            
            # Read predictions
            df_out = pd.read_csv(output_path)
            predictions = df_out['prediction'].values
            
            return predictions
        
        finally:
            # Cleanup temp files
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
```

---

## Use Case 5: Compound AI System (RAG, Agents, etc.)

Wrap multi-component AI systems with custom logic.

```python
# models/rag_wrapper.py
import numpy as np
from typing import Any

class RAGSystemWrapper:
    def __init__(self, retriever, generator, threshold=0.7):
        """
        Wrap a Retrieval-Augmented Generation system
        
        Args:
            retriever: Document retriever (e.g., vector DB client)
            generator: LLM generator (e.g., OpenAI client)
            threshold: Confidence threshold for classification
        """
        self.retriever = retriever
        self.generator = generator
        self.threshold = threshold
    
    def predict(self, X):
        """
        Predict for each query
        
        Args:
            X: pandas DataFrame with 'query' column
        
        Returns:
            Binary predictions (0/1)
        """
        predictions = []
        
        for query in X['query']:
            # Retrieve relevant docs
            docs = self.retriever.search(query, top_k=3)
            
            # Generate response + confidence score
            response, confidence = self.generator.generate(query, context=docs)
            
            # Classify based on threshold
            prediction = 1 if confidence >= self.threshold else 0
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Return confidence scores as probabilities"""
        confidences = []
        
        for query in X['query']:
            docs = self.retriever.search(query, top_k=3)
            response, confidence = self.generator.generate(query, context=docs)
            confidences.append(confidence)
        
        confidences = np.array(confidences)
        # Convert to (n_samples, 2) shape for binary classification
        return np.column_stack([1 - confidences, confidences])
```

---

## Best Practices

1. **Handle input types gracefully** — accept both pandas DataFrames and numpy arrays
2. **Return numpy arrays** — tests expect numpy array outputs
3. **Implement both methods** — `predict()` required, `predict_proba()` for probabilistic tests
4. **Error handling** — catch exceptions and provide clear error messages
5. **Logging** — use Python logging for debugging inference issues
6. **Thread safety** — ensure wrapper is safe for parallel test execution
7. **Resource cleanup** — close file handles, API connections in `__del__` or context manager

### Template with Best Practices

```python
# models/robust_wrapper.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class RobustModelWrapper:
    def __init__(self, **kwargs):
        """Initialize wrapper with configuration"""
        logger.info(f"Initializing wrapper with config: {kwargs}")
        # Your initialization here
    
    def _validate_input(self, X):
        """Validate and normalize input"""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
    
    def predict(self, X):
        try:
            X_validated = self._validate_input(X)
            logger.debug(f"Predicting for {len(X_validated)} samples")
            
            # Your prediction logic
            predictions = self._do_predict(X_validated)
            
            return np.array(predictions)
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_proba(self, X):
        try:
            X_validated = self._validate_input(X)
            
            # Your probability logic
            probas = self._do_predict_proba(X_validated)
            
            return np.array(probas)
        
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            raise
```

---

## Loading Your Wrapper

### Option 1: Python Class (Recommended)

```yaml
model:
  location:
    type: python_class
    path: models/my_wrapper.py
    class: MyModelWrapper
    kwargs:
      arg1: value1
      arg2: value2
```

### Option 2: Pickle the Wrapper Instance

```python
# Save wrapper instance
wrapper = MyModelWrapper(arg1=value1, arg2=value2)
with open('models/wrapper.pkl', 'wb') as f:
    pickle.dump(wrapper, f)
```

```yaml
model:
  location:
    type: file
    path: models/wrapper.pkl
```

---

## Testing Your Wrapper Locally

Before using with mrm, test the wrapper directly:

```python
import pandas as pd
from models.my_wrapper import MyModelWrapper

# Instantiate wrapper
wrapper = MyModelWrapper(arg1='value')

# Load test data
test_data = pd.read_csv('data/validation.csv')
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Test predict
predictions = wrapper.predict(X_test)
print(f"Predictions shape: {predictions.shape}")
print(f"Unique values: {np.unique(predictions)}")

# Test predict_proba
probabilities = wrapper.predict_proba(X_test)
print(f"Probabilities shape: {probabilities.shape}")
print(f"Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")

# Validate probabilities sum to 1
assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities don't sum to 1"
```

---

## Troubleshooting

### "Model has no attribute 'predict'"
- Verify method is implemented with correct signature
- Check class name matches YAML `class` field

### "Input dimension mismatch"
- Validate input shape in `_validate_input()` method
- Log input shape for debugging

### Wrapper imports fail
- Ensure `models/` is a Python package with `__init__.py`
- Or ensure project root is in `sys.path` (mrm runner does this automatically)

### Slow inference
- Profile with `cProfile` or `line_profiler`
- Cache expensive operations (e.g., model loading)
- Consider batch processing in `predict()`

---

## See Also

- [Scikit-Learn Models](sklearn.md)
- [PyTorch Models](pytorch.md)
- [TensorFlow/Keras Models](tensorflow.md)
- [MLflow Custom Flavors](https://mlflow.org/docs/latest/models.html#custom-flavors)
