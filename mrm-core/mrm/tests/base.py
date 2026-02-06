"""Base classes for MRM validation tests"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


class TestResult:
    """Result from running a test"""
    
    def __init__(
        self,
        passed: bool,
        score: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.passed = passed
        self.score = score
        self.details = details or {}
        self.failure_reason = failure_reason
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'passed': bool(self.passed),
            'score': float(self.score) if self.score is not None else None,
            'details': self.details,
            'failure_reason': self.failure_reason,
            'metadata': self.metadata
        }


class MRMTest(ABC):
    """Base class for all MRM validation tests"""
    
    name: str = "base_test"
    description: str = "Base test class"
    category: str = "general"
    tags: List[str] = []
    
    def __init__(self):
        """Initialize test"""
        pass
    
    @abstractmethod
    def run(
        self, 
        model: Any = None, 
        dataset: Any = None,
        **config
    ) -> TestResult:
        """
        Run the test
        
        Args:
            model: The model to test (optional)
            dataset: The dataset to test (optional)
            **config: Additional configuration parameters
        
        Returns:
            TestResult object with test outcomes
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate test configuration
        
        Args:
            config: Configuration dictionary
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        return True
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name})>"


class DatasetTest(MRMTest):
    """Base class for dataset-level tests"""
    category = "dataset"
    
    @abstractmethod
    def run(self, dataset: Any, **config) -> TestResult:
        """Run dataset test"""
        pass


class ModelTest(MRMTest):
    """Base class for model-level tests"""
    category = "model"
    
    @abstractmethod
    def run(self, model: Any, dataset: Any, **config) -> TestResult:
        """Run model test"""
        pass


class ComplianceTest(MRMTest):
    """Base class for regulatory compliance tests"""
    category = "compliance"
    
    @abstractmethod
    def run(self, model: Any, dataset: Any, **config) -> TestResult:
        """Run compliance test"""
        pass
