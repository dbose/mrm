"""
LiteLLM-based LLM Endpoint Adapter

Uses LiteLLM (https://github.com/BerriAI/litellm) to provide unified access
to 100+ LLM providers with a single consistent interface.

Supported providers include:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5)
- AWS Bedrock (Claude, Titan, etc.)
- Azure OpenAI
- Databricks
- HuggingFace
- Cohere, Replicate, Together AI, Perplexity, and 100+ more

Benefits over individual adapters:
- Single unified interface
- Automatic retries and fallbacks
- Built-in cost tracking
- Support for streaming
- Provider-agnostic code

Example configuration in model YAML:

```yaml
location:
  type: llm_endpoint
  provider: litellm  # Use LiteLLM unified interface
  model_name: gpt-4  # Or: claude-sonnet-4-6, bedrock/anthropic.claude-v2, etc.
  temperature: 0.3
  max_tokens: 500
  
  # Optional: specific provider settings
  api_base: https://api.openai.com/v1  # Provider-specific
  api_key_env: OPENAI_API_KEY  # Env var name for API key
  
  # RAG retriever (optional)
  retriever:
    type: faiss
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
    top_k: 3
    knowledge_base_path: data/knowledge_base.json
```

Environment variables:
- OPENAI_API_KEY for OpenAI
- ANTHROPIC_API_KEY for Anthropic
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY for Bedrock
- AZURE_API_KEY, AZURE_API_BASE for Azure OpenAI
- etc. (see LiteLLM docs)
"""

import os
import time
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LiteLLMEndpoint:
    """LiteLLM-based unified LLM endpoint adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LiteLLM endpoint.
        
        Args:
            config: Model location configuration from YAML
        """
        try:
            import litellm
            self.litellm = litellm
            
            # Suppress LiteLLM's verbose logging by default
            litellm.suppress_debug_info = True
            
        except ImportError:
            raise ImportError(
                "litellm package not installed. "
                "Install with: pip install litellm"
            )
        
        self.config = config
        self.model_name = config.get('model_name')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 1000)
        self.top_p = config.get('top_p', 1.0)
        
        # Normalize model name for LiteLLM provider detection
        self.model_name = self._normalize_model_name(self.model_name)
        
        # Optional provider-specific settings
        self.api_base = config.get('api_base')
        self.api_key_env = config.get('api_key_env')
        
        # Token counting for cost estimation
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # RAG retriever (optional)
        self.retriever = None
        if 'retriever' in config:
            self.retriever = self._init_retriever(config['retriever'])

        # Replay capture context (set externally). None = opt-out.
        self.replay_context: Optional[Any] = None
    
    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize model name to ensure LiteLLM can detect the provider.
        Adds provider prefix if not present and provider can be inferred.
        
        Args:
            model_name: Original model name from config
            
        Returns:
            Normalized model name with provider prefix if needed
        """
        if not model_name:
            return model_name
        
        # If already has provider prefix, return as-is
        if '/' in model_name:
            return model_name
        
        # Detect provider from model name patterns
        model_lower = model_name.lower()
        
        # Anthropic Claude models
        if 'claude' in model_lower:
            return f'anthropic/{model_name}'
        
        # OpenAI GPT models
        elif 'gpt-' in model_lower or model_lower.startswith('gpt'):
            return f'openai/{model_name}'
        
        # Cohere models
        elif 'command' in model_lower:
            return f'cohere/{model_name}'
        
        # Otherwise return as-is and let LiteLLM try to detect
        return model_name
    
    def _init_retriever(self, retriever_config: Dict[str, Any]):
        """Initialize RAG retriever if configured."""
        retriever_type = retriever_config.get('type', 'faiss')
        
        if retriever_type == 'faiss':
            try:
                from mrm.backends.rag_retriever import FAISSRetriever
                return FAISSRetriever(retriever_config)
            except ImportError as e:
                logger.warning(f"Could not initialize FAISS retriever: {e}")
                return None
        else:
            logger.warning(f"Unknown retriever type: {retriever_type}")
            return None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate completion from LLM via LiteLLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Provider-specific parameters
        
        Returns:
            Tuple of (response_text, metadata) where metadata includes:
                - prompt_tokens: int
                - completion_tokens: int
                - total_tokens: int
                - latency_ms: float
                - model: str
                - finish_reason: str
                - cost: float (if available)
        """
        start_time = time.time()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build completion kwargs
        temperature = kwargs.get('temperature', self.temperature)
        top_p = kwargs.get('top_p', self.top_p)

        completion_kwargs = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
        }

        # Anthropic forbids sending both temperature and top_p
        if 'anthropic' in self.model_name.lower() or 'claude' in self.model_name.lower():
            if 'top_p' in kwargs:
                completion_kwargs['top_p'] = top_p
            else:
                completion_kwargs['temperature'] = temperature
        else:
            completion_kwargs['temperature'] = temperature
            completion_kwargs['top_p'] = top_p
        
        # Add optional provider-specific settings
        if self.api_base:
            completion_kwargs['api_base'] = self.api_base
        
        # Set API key from environment if specified
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env)
            if api_key:
                completion_kwargs['api_key'] = api_key
        
        try:
            # Call LiteLLM completion
            response = self.litellm.completion(**completion_kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response
            response_text = response.choices[0].message.content
            
            # Extract token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            
            # Build metadata
            metadata = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'latency_ms': latency_ms,
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }
            
            # Add cost if available (LiteLLM computes this automatically)
            if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                metadata['cost'] = response._hidden_params['response_cost']
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"LiteLLM API error: {e}")
            raise
    
    def generate_with_retrieval(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate completion with RAG retrieval.

        Args:
            query: User query
            system_prompt: Optional system prompt
            **kwargs: Provider-specific parameters

        Returns:
            Tuple of (response_text, metadata)
        """
        retrieved_docs: Optional[list] = None
        if self.retriever:
            retrieved_docs = self.retriever.retrieve(query)

            # Build augmented prompt
            context = "\n\n".join([
                f"[Source {i+1}]: {doc['text']}"
                for i, doc in enumerate(retrieved_docs)
            ])

            augmented_prompt = f"""Use the following information to answer the question.

{context}

Question: {query}

Answer:"""

            # Add retrieval metadata
            response, metadata = self.generate(augmented_prompt, system_prompt, **kwargs)
            metadata['retrieval'] = {
                'num_docs': len(retrieved_docs),
                'doc_ids': [doc.get('id') for doc in retrieved_docs]
            }
            self._emit_replay(query, response, metadata, system_prompt, retrieved_docs, kwargs)
            return response, metadata
        else:
            # No retriever configured, just generate
            response, metadata = self.generate(query, system_prompt, **kwargs)
            self._emit_replay(query, response, metadata, system_prompt, None, kwargs)
            return response, metadata

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Replay-aware completion entry point (parallels legacy adapter)."""
        return self.generate_with_retrieval(prompt, system_prompt=system_prompt, **kwargs)

    def _emit_replay(
        self,
        prompt: str,
        response: str,
        metadata: Dict[str, Any],
        system_prompt: Optional[str],
        retrieved_docs: Optional[list],
        extra_kwargs: Dict[str, Any],
    ) -> None:
        if not getattr(self, 'replay_context', None):
            return
        try:
            from mrm.replay.instrument import record_llm_call
            from mrm.replay.record import InferenceParams
        except ImportError:
            return

        inference_params = InferenceParams(
            temperature=extra_kwargs.get('temperature', self.temperature),
            top_p=extra_kwargs.get('top_p', self.top_p),
            max_tokens=extra_kwargs.get('max_tokens', self.max_tokens),
            seed=extra_kwargs.get('seed'),
            retrieval_k=len(retrieved_docs) if retrieved_docs else None,
        )
        record_llm_call(
            replay_context=self.replay_context,
            prompt=prompt,
            response=response,
            metadata=metadata,
            system_prompt=system_prompt,
            retrieved_docs=retrieved_docs,
            inference_params=inference_params,
        )
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage."""
        return {
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_prompt_tokens + self.total_completion_tokens
        }
    
    def estimate_cost(self, pricing: Optional[Dict[str, float]] = None) -> float:
        """
        Estimate cost based on token usage.
        
        Note: LiteLLM provides automatic cost calculation via 
        response._hidden_params['response_cost'], so explicit pricing
        is usually not needed.
        
        Args:
            pricing: Dict with 'prompt_token_cost' and 'completion_token_cost'
        
        Returns:
            Estimated cost in USD
        """
        if pricing:
            prompt_cost = self.total_prompt_tokens * pricing.get('prompt_token_cost', 0)
            completion_cost = self.total_completion_tokens * pricing.get('completion_token_cost', 0)
            return prompt_cost + completion_cost
        else:
            # Return 0 if no pricing provided - actual costs captured per-request
            return 0.0
    
    def health_check(self) -> bool:
        """
        Check if endpoint is reachable and credentials are valid.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple test generation with minimal tokens
            test_response = self.litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"LiteLLM health check failed: {e}")
            return False


def get_litellm_endpoint(config: Dict[str, Any]) -> LiteLLMEndpoint:
    """
    Factory function to create LiteLLM endpoint adapter.
    
    Args:
        config: Model location configuration from YAML
    
    Returns:
        LiteLLMEndpoint instance
    """
    return LiteLLMEndpoint(config)
