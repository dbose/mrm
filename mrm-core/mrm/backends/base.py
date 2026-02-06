"""Base backend adapter interface for MRM"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path


class BackendAdapter(ABC):
    """Base class for backend adapters"""
    
    def __init__(self, **config):
        """
        Initialize backend adapter
        
        Args:
            **config: Backend-specific configuration
        """
        self.config = config
    
    @abstractmethod
    def register_model(self, model_config: Dict, model_artifact: Any) -> str:
        """
        Register a model
        
        Args:
            model_config: Model configuration dictionary
            model_artifact: Model object to register
        
        Returns:
            model_id: Unique identifier for the registered model
        """
        pass
    
    @abstractmethod
    def get_model(self, model_id: str) -> Any:
        """
        Retrieve model artifact
        
        Args:
            model_id: Model identifier
        
        Returns:
            Model object
        """
        pass
    
    @abstractmethod
    def log_test_results(self, model_id: str, test_results: Dict) -> None:
        """
        Store test results
        
        Args:
            model_id: Model identifier
            test_results: Dictionary of test results
        """
        pass
    
    @abstractmethod
    def get_test_history(self, model_id: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve historical test results
        
        Args:
            model_id: Model identifier
            limit: Maximum number of results to return
        
        Returns:
            List of test result dictionaries
        """
        pass
    
    @abstractmethod
    def list_models(self, filters: Optional[Dict] = None) -> List[Dict]:
        """
        List models with optional filters
        
        Args:
            filters: Optional dictionary of filter criteria
        
        Returns:
            List of model metadata dictionaries
        """
        pass
    
    def save_dataset(self, dataset_id: str, dataset: Any) -> None:
        """
        Save dataset (optional)
        
        Args:
            dataset_id: Dataset identifier
            dataset: Dataset object
        """
        pass
    
    def get_dataset(self, dataset_id: str) -> Any:
        """
        Retrieve dataset (optional)
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            Dataset object
        """
        pass
    
    def close(self):
        """Close backend connection"""
        pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
