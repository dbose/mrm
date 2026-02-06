"""Local filesystem backend for MRM"""

import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from mrm.backends.base import BackendAdapter


class LocalBackend(BackendAdapter):
    """Local filesystem backend for model and test storage"""
    
    def __init__(self, path: str = None, **config):
        """
        Initialize local backend
        
        Args:
            path: Base path for storage (defaults to ~/.mrm/data)
            **config: Additional configuration
        """
        super().__init__(**config)
        
        if path is None:
            path = str(Path.home() / ".mrm" / "data")
        
        self.base_path = Path(path)
        self.models_path = self.base_path / "models"
        self.tests_path = self.base_path / "tests"
        self.datasets_path = self.base_path / "datasets"
        
        # Create directories
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.tests_path.mkdir(parents=True, exist_ok=True)
        self.datasets_path.mkdir(parents=True, exist_ok=True)
    
    def register_model(self, model_config: Dict, model_artifact: Any) -> str:
        """Register a model to local filesystem"""
        model_name = model_config['name']
        model_version = model_config.get('version', '1.0.0')
        model_id = f"{model_name}_v{model_version}"
        
        model_dir = self.models_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model artifact
        model_file = model_dir / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_artifact, f)
        
        # Save model config/metadata
        config_file = model_dir / "config.json"
        metadata = {
            **model_config,
            'model_id': model_id,
            'registered_at': datetime.now().isoformat(),
            'model_file': str(model_file.relative_to(self.base_path))
        }
        
        with open(config_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_id
    
    def get_model(self, model_id: str) -> Any:
        """Retrieve model from local filesystem"""
        model_dir = self.models_path / model_id
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model '{model_id}' not found in local backend")
        
        model_file = model_dir / "model.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model artifact not found for '{model_id}'")
        
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    
    def log_test_results(self, model_id: str, test_results: Dict) -> None:
        """Store test results to local filesystem"""
        test_dir = self.tests_path / model_id
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped test result file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = test_dir / f"test_results_{timestamp}.json"
        
        # Convert TestResult objects to dicts
        serializable_results = {}
        for test_name, result in test_results.items():
            if hasattr(result, 'to_dict'):
                serializable_results[test_name] = result.to_dict()
            else:
                serializable_results[test_name] = result
        
        # Add metadata
        output = {
            'model_id': model_id,
            'timestamp': datetime.now().isoformat(),
            'test_results': serializable_results
        }
        
        with open(result_file, 'w') as f:
            json.dump(output, f, indent=2)
    
    def get_test_history(self, model_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve historical test results"""
        test_dir = self.tests_path / model_id
        
        if not test_dir.exists():
            return []
        
        # Get all test result files sorted by timestamp (newest first)
        result_files = sorted(
            test_dir.glob("test_results_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        history = []
        for result_file in result_files[:limit]:
            with open(result_file, 'r') as f:
                history.append(json.load(f))
        
        return history
    
    def list_models(self, filters: Optional[Dict] = None) -> List[Dict]:
        """List models in local filesystem"""
        models = []
        
        for model_dir in self.models_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            config_file = model_dir / "config.json"
            if not config_file.exists():
                continue
            
            with open(config_file, 'r') as f:
                model_config = json.load(f)
            
            # Apply filters
            if filters:
                match = True
                for key, value in filters.items():
                    if model_config.get(key) != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            models.append(model_config)
        
        return models
    
    def save_dataset(self, dataset_id: str, dataset: Any) -> None:
        """Save dataset to local filesystem"""
        dataset_dir = self.datasets_path / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_file = dataset_dir / "dataset.pkl"
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)
    
    def get_dataset(self, dataset_id: str) -> Any:
        """Retrieve dataset from local filesystem"""
        dataset_file = self.datasets_path / dataset_id / "dataset.pkl"
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
        
        with open(dataset_file, 'rb') as f:
            return pickle.load(f)
    
    def close(self):
        """No cleanup needed for local backend"""
        pass
