"""
RAG Customer Service Assistant Model

Simple RAG system combining FAISS retrieval with LLM generation.
This module provides a loadable model interface for mrm-core.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class RAGAssistant:
    """RAG-based customer service assistant."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG assistant from configuration.
        
        Args:
            config: Model configuration from YAML
        """
        self.config = config
        self.name = config.get('name', 'rag_assistant')
        self.version = config.get('version', '1.0.0')
        
        # LLM endpoint will be initialized by test runner
        self.endpoint = None
        
    def generate(self, query: str, **kwargs) -> str:
        """
        Generate response for query.
        
        Args:
            query: User query
            **kwargs: Additional generation parameters
        
        Returns:
            Generated response
        """
        if not self.endpoint:
            from mrm.backends.llm_endpoints import get_llm_endpoint
            self.endpoint = get_llm_endpoint(self.config['location'])
        
        response, metadata = self.endpoint.generate_with_retrieval(query, **kwargs)
        return response
    
    def __repr__(self):
        return f"RAGAssistant(name={self.name}, version={self.version})"


def load_model(config: Dict[str, Any]) -> RAGAssistant:
    """
    Load RAG assistant model.
    
    Args:
        config: Model configuration from YAML
    
    Returns:
        Initialized RAGAssistant instance
    """
    return RAGAssistant(config)
