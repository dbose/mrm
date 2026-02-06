"""Model reference resolution with ref() and catalog integration"""

from typing import Any, Dict, Optional, Union
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelReference:
    """
    Represents a reference to a model
    
    Supports multiple reference types:
    - ref('model_name') - Reference to another model in the project
    - hf/org/model-name - Hugging Face model
    - catalog/internal/model-name - Internal model catalog
    - file/path/to/model.pkl - Local file
    - mlflow/model-name - MLflow registry
    """
    
    def __init__(self, reference: Union[str, Dict]):
        """
        Initialize model reference
        
        Args:
            reference: String reference (e.g., "ref('pd_model')") or dict location config
        """
        if isinstance(reference, str):
            self.reference_string = reference
            self.type, self.path = self._parse_reference(reference)
        else:
            # Dict-based location config
            self.reference_string = None
            self.type = reference.get('type')
            self.path = reference.get('path') or reference.get('model_name') or reference.get('uri')
            self.config = reference
    
    def _parse_reference(self, ref: str) -> tuple[str, str]:
        """
        Parse reference string
        
        Args:
            ref: Reference string
        
        Returns:
            Tuple of (type, path)
        """
        # ref('model_name')
        if ref.startswith('ref('):
            match = re.match(r"ref\(['\"]([^'\"]+)['\"]\)", ref)
            if match:
                return ('ref', match.group(1))
        
        # Protocol-based references
        if '/' in ref:
            protocol, path = ref.split('/', 1)
            return (protocol, path)
        
        # Default to ref
        return ('ref', ref)
    
    def __repr__(self):
        return f"ModelReference({self.type}/{self.path})"


class ModelLoader:
    """
    Loads models from various sources
    
    Supports:
    - Local files (pickle, joblib)
    - Python classes
    - MLflow registry
    - Hugging Face Hub
    - Internal model catalogs
    - References to other models
    """
    
    def __init__(self, project_root: Path, backend=None, catalog=None):
        """
        Initialize model loader
        
        Args:
            project_root: Project root directory
            backend: Backend adapter (for MLflow etc.)
            catalog: Model catalog adapter (for internal registry)
        """
        self.project_root = project_root
        self.backend = backend
        self.catalog = catalog
        self._loaded_models: Dict[str, Any] = {}  # Cache
    
    def load(self, location: Union[Dict, str, ModelReference]) -> Any:
        """
        Load a model from specified location
        
        Args:
            location: Location specification
        
        Returns:
            Loaded model object
        """
        # Convert to ModelReference
        if not isinstance(location, ModelReference):
            ref = ModelReference(location)
        else:
            ref = location
        
        # Check cache
        cache_key = f"{ref.type}/{ref.path}"
        if cache_key in self._loaded_models:
            logger.debug(f"Loading from cache: {cache_key}")
            return self._loaded_models[cache_key]
        
        # Load based on type
        if ref.type == 'file':
            model = self._load_file(ref.path)
        
        elif ref.type == 'python_class':
            model = self._load_python_class(ref)
        
        elif ref.type == 'mlflow':
            model = self._load_mlflow(ref.path)
        
        elif ref.type == 'hf':
            model = self._load_huggingface(ref.path)
        
        elif ref.type == 'catalog':
            model = self._load_catalog(ref.path)
        
        elif ref.type == 'ref':
            model = self._load_ref(ref.path)
        
        else:
            raise ValueError(f"Unknown model reference type: {ref.type}")
        
        # Cache
        self._loaded_models[cache_key] = model
        
        return model
    
    def _load_file(self, path: str) -> Any:
        """Load model from file"""
        import pickle
        
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.project_root / file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        logger.info(f"Loading model from file: {file_path}")
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_python_class(self, ref: ModelReference) -> Any:
        """Load model from Python class"""
        import importlib
        import sys
        
        # Add project root to path
        sys.path.insert(0, str(self.project_root))
        
        config = getattr(ref, 'config', {})
        module_path = config.get('path', '').replace('.py', '').replace('/', '.')
        class_name = config.get('class')
        
        logger.info(f"Loading Python class: {module_path}.{class_name}")
        
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # Instantiate if callable
        if callable(model_class):
            return model_class()
        return model_class
    
    def _load_mlflow(self, model_id: str) -> Any:
        """Load model from MLflow"""
        if not self.backend:
            raise ValueError("MLflow backend not configured")
        
        logger.info(f"Loading model from MLflow: {model_id}")
        return self.backend.get_model(model_id)
    
    def _load_huggingface(self, model_path: str) -> Any:
        """
        Load model from Hugging Face Hub
        
        Args:
            model_path: HF model identifier (e.g., 'meta-llama/Llama-3-8B')
        
        Returns:
            Hugging Face model/pipeline
        """
        try:
            from transformers import AutoModel, AutoTokenizer, pipeline
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers"
            )
        
        logger.info(f"Loading model from Hugging Face: {model_path}")
        
        # Parse model path - may include task type
        # Format: org/model-name or org/model-name:task
        if ':' in model_path:
            model_name, task = model_path.split(':', 1)
        else:
            model_name = model_path
            task = None
        
        if task:
            # Load as pipeline
            return pipeline(task, model=model_name)
        else:
            # Load model and tokenizer
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return {'model': model, 'tokenizer': tokenizer}
    
    def _load_catalog(self, model_path: str) -> Any:
        """
        Load model from internal catalog
        
        Args:
            model_path: Catalog path (e.g., 'production/credit_scorecard')
        
        Returns:
            Model from catalog
        """
        if not self.catalog:
            raise ValueError("Model catalog not configured")
        
        logger.info(f"Loading model from catalog: {model_path}")
        return self.catalog.get_model(model_path)
    
    def _load_ref(self, model_name: str) -> Any:
        """
        Load referenced model from project
        
        This is a placeholder - actual implementation would need
        access to the full project to resolve refs
        
        Args:
            model_name: Name of referenced model
        
        Returns:
            Model object
        """
        logger.info(f"Resolving ref: {model_name}")
        
        # This will be called by the runner with proper context
        raise NotImplementedError(
            f"ref('{model_name}') cannot be resolved by ModelLoader directly. "
            f"Use TestRunner which has access to all models."
        )
    
    def clear_cache(self):
        """Clear model cache"""
        self._loaded_models.clear()


class ModelCatalog:
    """
    Abstract base for model catalogs
    
    Allows integration with internal model registries
    """
    
    def get_model(self, model_path: str) -> Any:
        """
        Get model from catalog
        
        Args:
            model_path: Catalog-specific path
        
        Returns:
            Model object
        """
        raise NotImplementedError
    
    def list_models(self, prefix: Optional[str] = None) -> list[str]:
        """
        List available models
        
        Args:
            prefix: Optional path prefix filter
        
        Returns:
            List of model paths
        """
        raise NotImplementedError


class SimpleFileCatalog(ModelCatalog):
    """
    Simple file-based model catalog
    
    Structure:
        catalog/
        ├── production/
        │   ├── credit_scorecard.pkl
        │   └── fraud_detector.pkl
        └── development/
            └── experimental.pkl
    """
    
    def __init__(self, catalog_root: Path):
        """
        Initialize file catalog
        
        Args:
            catalog_root: Root directory of catalog
        """
        self.catalog_root = Path(catalog_root)
    
    def get_model(self, model_path: str) -> Any:
        """Load model from catalog"""
        import pickle
        
        file_path = self.catalog_root / f"{model_path}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model not found in catalog: {model_path}")
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def list_models(self, prefix: Optional[str] = None) -> list[str]:
        """List models in catalog"""
        models = []
        
        search_path = self.catalog_root
        if prefix:
            search_path = search_path / prefix
        
        for pkl_file in search_path.rglob("*.pkl"):
            rel_path = pkl_file.relative_to(self.catalog_root)
            model_path = str(rel_path.with_suffix(''))
            models.append(model_path)
        
        return sorted(models)


def parse_depends_on(depends_on: list) -> list[str]:
    """
    Parse depends_on list to extract model names
    
    Handles:
    - Simple strings: ['pd_model', 'lgd_model']
    - ref() calls: ["ref('pd_model')", "ref('lgd_model')"]
    - Mixed: ['pd_model', "ref('lgd_model')"]
    
    Args:
        depends_on: List of dependencies
    
    Returns:
        List of model names
    """
    result = []
    
    for dep in depends_on:
        if isinstance(dep, str):
            if dep.startswith('ref('):
                # Extract from ref('name')
                match = re.match(r"ref\(['\"]([^'\"]+)['\"]\)", dep)
                if match:
                    result.append(match.group(1))
            else:
                # Plain string
                result.append(dep)
    
    return result
