"""MLflow backend for MRM"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from mrm.backends.base import BackendAdapter

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowBackend(BackendAdapter):
    """MLflow backend for model registry and experiment tracking"""
    
    def __init__(self, tracking_uri: str, experiment_name: str = "mrm-validation", **config):
        """
        Initialize MLflow backend
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of MLflow experiment
            **config: Additional configuration
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is not installed. Install with: pip install mrm-core[mlflow]"
            )
        
        super().__init__(**config)
        
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.client = MlflowClient(tracking_uri)
    
    def register_model(self, model_config: Dict, model_artifact: Any) -> str:
        """Register model in MLflow"""
        model_name = model_config['name']
        
        with mlflow.start_run(run_name=f"{model_name}_registration") as run:
            # Log model
            try:
                mlflow.sklearn.log_model(model_artifact, "model")
            except Exception:
                # Try generic python_function logging
                mlflow.pyfunc.log_model("model", python_model=model_artifact)
            
            # Log metadata as tags
            mlflow.set_tags({
                "mrm.name": model_config['name'],
                "mrm.version": model_config.get('version', '1.0.0'),
                "mrm.risk_tier": model_config.get('risk_tier', 'unspecified'),
                "mrm.owner": model_config.get('owner', 'unspecified'),
                "mrm.registered_at": datetime.now().isoformat()
            })
            
            # Log config as parameters
            for key, value in model_config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(f"mrm.{key}", value)
            
            # Register to model registry
            model_uri = f"runs{run.info.run_id}/model"
            
            try:
                self.client.create_registered_model(model_name)
            except Exception:
                # Model already exists
                pass
            
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run.info.run_id
            )
            
            return run.info.run_id
    
    def get_model(self, model_id: str) -> Any:
        """Retrieve model from MLflow"""
        try:
            # If model_id is a run_id
            model_uri = f"runs{model_id}/model"
            return mlflow.pyfunc.load_model(model_uri)
        except Exception:
            # If model_id is a model name, get latest version
            try:
                model_uri = f"models{model_id}/latest"
                return mlflow.pyfunc.load_model(model_uri)
            except Exception as e:
                raise ValueError(f"Could not load model '{model_id}': {e}")
    
    def log_test_results(self, model_id: str, test_results: Dict) -> None:
        """Store test results in MLflow"""
        with mlflow.start_run(run_id=model_id) as run:
            # Log test metrics
            for test_name, result in test_results.items():
                if hasattr(result, 'to_dict'):
                    result_dict = result.to_dict()
                else:
                    result_dict = result
                
                # Log score as metric
                if result_dict.get('score') is not None:
                    mlflow.log_metric(f"test.{test_name}.score", result_dict['score'])
                
                # Log pass/fail as tag
                mlflow.set_tag(f"test.{test_name}.passed", str(result_dict.get('passed', False)))
            
            # Log full results as artifact
            import json
            import tempfile
            
            serializable_results = {}
            for test_name, result in test_results.items():
                if hasattr(result, 'to_dict'):
                    serializable_results[test_name] = result.to_dict()
                else:
                    serializable_results[test_name] = result
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'test_results': serializable_results
                }, f, indent=2)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "test_results")
            
            import os
            os.unlink(temp_path)
    
    def get_test_history(self, model_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve historical test results from MLflow"""
        # This is a simplified implementation
        # In practice, you'd query MLflow for all runs associated with the model
        try:
            run = self.client.get_run(model_id)
            
            # Get test metrics
            metrics = run.data.metrics
            tags = run.data.tags
            
            test_results = {}
            for key, value in metrics.items():
                if key.startswith('test.'):
                    parts = key.split('.')
                    if len(parts) >= 3:
                        test_name = parts[1]
                        if test_name not in test_results:
                            test_results[test_name] = {}
                        test_results[test_name]['score'] = value
            
            for key, value in tags.items():
                if key.startswith('test.') and key.endswith('.passed'):
                    test_name = key.split('.')[1]
                    if test_name not in test_results:
                        test_results[test_name] = {}
                    test_results[test_name]['passed'] = value.lower() == 'true'
            
            return [{
                'timestamp': datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                'test_results': test_results
            }]
        
        except Exception:
            return []
    
    def list_models(self, filters: Optional[Dict] = None) -> List[Dict]:
        """List models from MLflow registry"""
        models = []
        
        try:
            registered_models = self.client.search_registered_models()
            
            for rm in registered_models:
                model_info = {
                    'name': rm.name,
                    'latest_versions': []
                }
                
                # Get latest version info
                if rm.latest_versions:
                    latest = rm.latest_versions[0]
                    run = self.client.get_run(latest.run_id)
                    
                    # Extract MRM tags
                    tags = run.data.tags
                    for key, value in tags.items():
                        if key.startswith('mrm.'):
                            field = key.replace('mrm.', '')
                            model_info[field] = value
                
                # Apply filters
                if filters:
                    match = True
                    for key, value in filters.items():
                        if model_info.get(key) != value:
                            match = False
                            break
                    
                    if not match:
                        continue
                
                models.append(model_info)
        
        except Exception:
            pass
        
        return models
    
    def close(self):
        """Close MLflow connection"""
        pass
