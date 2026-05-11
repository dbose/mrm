"""
Model backends for mrm-core.

Provides storage backends (local, MLflow) and model source adapters
(LLM endpoints, RAG retrievers).
"""

# Storage backends
from .base import BackendAdapter
from .local import LocalBackend
from .mlflow import MLflowBackend

# LLM endpoints (optional - requires genai dependencies)
try:
    from .llm_endpoints import (
        LLMEndpoint,
        OpenAIEndpoint,
        AnthropicEndpoint,
        BedrockEndpoint,
        DatabricksEndpoint,
        HuggingFaceEndpoint,
        get_llm_endpoint
    )
    LLM_ENDPOINTS_AVAILABLE = True
except ImportError:
    LLM_ENDPOINTS_AVAILABLE = False

# LiteLLM unified adapter (optional - requires litellm)
try:
    from .litellm_endpoint import LiteLLMEndpoint, get_litellm_endpoint
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# RAG retrievers (optional - requires genai dependencies)
try:
    from .rag_retriever import FAISSRetriever
    RAG_RETRIEVERS_AVAILABLE = True
except ImportError:
    RAG_RETRIEVERS_AVAILABLE = False

__all__ = [
    'BackendAdapter',
    'LocalBackend',
    'MLflowBackend',
]

# Add LLM exports if available
if LLM_ENDPOINTS_AVAILABLE:
    __all__.extend([
        'LLMEndpoint',
        'OpenAIEndpoint',
        'AnthropicEndpoint',
        'BedrockEndpoint',
        'DatabricksEndpoint',
        'HuggingFaceEndpoint',
        'get_llm_endpoint'
    ])

# Add LiteLLM exports if available
if LITELLM_AVAILABLE:
    __all__.extend([
        'LiteLLMEndpoint',
        'get_litellm_endpoint'
    ])

# Add RAG exports if available
if RAG_RETRIEVERS_AVAILABLE:
    __all__.append('FAISSRetriever')
