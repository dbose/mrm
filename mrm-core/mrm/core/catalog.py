"""Model reference and catalog system for MRM"""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelSource(Enum):
    """Source types for models"""
    LOCAL = "local"              # Local file (pickle, joblib)
    PYTHON_CLASS = "python_class"  # Python class
    MLFLOW = "mlflow"            # MLflow registry
    HUGGINGFACE = "huggingface"  # HuggingFace hub
    REF = "ref"                  # Reference to another model
    S3 = "s3"                    # S3 bucket
    GCS = "gcs"                  # Google Cloud Storage
    AZURE = "azure"              # Azure Blob Storage


@dataclass
class ModelRef:
    """
    Reference to a model (similar to dbt's ref())
    
    Usage in YAML:
        location:
          type: ref
          model: base_pd_model
    
    Or for external models:
        location:
          type: huggingface
          repo_id: bert-base-uncased
          revision: main
    """
    
    source: ModelSource
    identifier: str
    metadata: Dict[str, Any]
    
    def __init__(
        self, 
        source: Union[str, ModelSource],
        identifier: str,
        **metadata
    ):
        """
        Initialize model reference
        
        Args:
            source: Source type (local, ref, huggingface, etc.)
            identifier: Model identifier (name, path, repo_id, etc.)
            **metadata: Additional metadata specific to source type
        """
        if isinstance(source, str):
            source = ModelSource(source)
        
        self.source = source
        self.identifier = identifier
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.source.value,
            'identifier': self.identifier,
            **self.metadata
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ModelRef':
        """
        Create ModelRef from configuration
        
        Args:
            config: Configuration dictionary
        
        Returns:
            ModelRef instance
        """
        source_type = config.get('type', 'local')
        # Normalize legacy alias 'file' to 'local' to match ModelSource values
        if source_type == 'file':
            source_type = 'local'
        
        # Map type to identifier key
        identifier_keys = {
            'local': 'path',
            'file': 'path',
            'python_class': 'class',
            'mlflow': 'model_name',
            'huggingface': 'repo_id',
            'ref': 'model',
            's3': 'uri',
            'gcs': 'uri',
            'azure': 'uri'
        }
        
        identifier_key = identifier_keys.get(source_type, 'identifier')
        identifier = config.get(identifier_key, '')
        
        # Extract metadata (everything except type and identifier)
        metadata = {k: v for k, v in config.items() if k not in ['type', identifier_key]}
        
        return cls(source=source_type, identifier=identifier, **metadata)
    
    def __repr__(self) -> str:
        return f"ModelRef(source={self.source.value}, identifier={self.identifier})"


class ModelCatalog:
    """
    Catalog of available models (internal and external)
    
    Similar to dbt's model registry, tracks all models
    in the project and provides ref() functionality
    """
    
    def __init__(self):
        """Initialize empty catalog"""
        self.models: Dict[str, ModelRef] = {}
        self.external_catalogs: Dict[str, Dict] = {
            'huggingface': {},
            'ollama': {},
            'openai': {}
        }
    
    def register(self, name: str, model_ref: ModelRef):
        """
        Register a model in the catalog
        
        Args:
            name: Model name
            model_ref: ModelRef instance
        """
        self.models[name] = model_ref
        logger.debug(f"Registered model: {name} ({model_ref.source.value})")
    
    def ref(self, name: str) -> Optional[ModelRef]:
        """
        Get reference to a model (like dbt's ref())
        
        Args:
            name: Model name
        
        Returns:
            ModelRef if found, None otherwise
        """
        return self.models.get(name)
    
    def resolve_ref(self, model_ref: ModelRef, catalog_context: Optional[Dict] = None) -> ModelRef:
        """
        Resolve a ref() to the actual model location
        
        Args:
            model_ref: ModelRef with source='ref'
            catalog_context: Optional catalog context (project models)
        
        Returns:
            Resolved ModelRef pointing to actual model
        """
        if model_ref.source != ModelSource.REF:
            return model_ref
        
        # Look up referenced model
        referenced_name = model_ref.identifier
        referenced_model = self.ref(referenced_name)
        
        if referenced_model is None:
            raise ValueError(
                f"Model reference '{referenced_name}' not found. "
                f"Available models: {list(self.models.keys())}"
            )
        
        # If the referenced model is itself a ref, resolve recursively
        if referenced_model.source == ModelSource.REF:
            return self.resolve_ref(referenced_model, catalog_context)
        
        return referenced_model
    
    def add_huggingface_model(
        self,
        name: str,
        repo_id: str,
        revision: str = "main",
        **kwargs
    ):
        """
        Add HuggingFace model to catalog
        
        Args:
            name: Local name for the model
            repo_id: HuggingFace repo ID (e.g., 'bert-base-uncased')
            revision: Model revision/branch (default: 'main')
            **kwargs: Additional parameters (task, trust_remote_code, etc.)
        """
        model_ref = ModelRef(
            source=ModelSource.HUGGINGFACE,
            identifier=repo_id,
            revision=revision,
            **kwargs
        )
        self.register(name, model_ref)
    
    def add_mlflow_model(
        self,
        name: str,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
        **kwargs
    ):
        """
        Add MLflow model to catalog
        
        Args:
            name: Local name for the model
            model_name: MLflow model name
            version: Model version (optional)
            stage: Model stage (Production, Staging, etc.)
            **kwargs: Additional parameters
        """
        model_ref = ModelRef(
            source=ModelSource.MLFLOW,
            identifier=model_name,
            version=version,
            stage=stage,
            **kwargs
        )
        self.register(name, model_ref)
    
    def list_models(self, source_filter: Optional[ModelSource] = None) -> Dict[str, ModelRef]:
        """
        List models in catalog
        
        Args:
            source_filter: Optional filter by source type
        
        Returns:
            Dictionary of model name -> ModelRef
        """
        if source_filter is None:
            return self.models.copy()
        
        return {
            name: ref 
            for name, ref in self.models.items() 
            if ref.source == source_filter
        }
    
    @classmethod
    def from_project(cls, model_configs: list) -> 'ModelCatalog':
        """
        Build catalog from project model configurations
        
        Args:
            model_configs: List of model configuration dictionaries
        
        Returns:
            ModelCatalog instance
        """
        catalog = cls()
        
        for config in model_configs:
            model_name = config['model']['name']
            location = config['model'].get('location', {})
            
            model_ref = ModelRef.from_config(location)
            catalog.register(model_name, model_ref)
        
        return catalog


def ref(model_name: str) -> Dict[str, Any]:
    """
    Helper function for YAML configuration (conceptual)
    
    In practice, YAML configs use:
        location:
          type: ref
          model: base_model_name
    
    This function documents the concept
    
    Args:
        model_name: Name of model to reference
    
    Returns:
        Configuration dictionary for model reference
    """
    return {
        'type': 'ref',
        'model': model_name
    }


# Pre-configured external model references
HUGGINGFACE_MODELS = {
    # Language Models
    'bert-base': {
        'repo_id': 'bert-base-uncased',
        'task': 'fill-mask'
    },
    'gpt2': {
        'repo_id': 'gpt2',
        'task': 'text-generation'
    },
    'roberta-base': {
        'repo_id': 'roberta-base',
        'task': 'fill-mask'
    },
    
    # Financial Models
    'finbert-sentiment': {
        'repo_id': 'ProsusAI/finbert',
        'task': 'sentiment-analysis'
    },
    'finbert-esg': {
        'repo_id': 'yiyanghkust/finbert-esg',
        'task': 'text-classification'
    },
    
    # Classification
    'distilbert-base': {
        'repo_id': 'distilbert-base-uncased',
        'task': 'text-classification'
    }
}


def get_hf_model_config(model_key: str) -> Dict[str, Any]:
    """
    Get pre-configured HuggingFace model
    
    Args:
        model_key: Key from HUGGINGFACE_MODELS
    
    Returns:
        Model configuration dictionary
    """
    if model_key not in HUGGINGFACE_MODELS:
        raise ValueError(
            f"Unknown HuggingFace model: {model_key}. "
            f"Available: {list(HUGGINGFACE_MODELS.keys())}"
        )
    
    config = HUGGINGFACE_MODELS[model_key].copy()
    config['type'] = 'huggingface'
    
    return config
