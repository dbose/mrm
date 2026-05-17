"""
LLM Endpoint Model Source

Provides adapters for multiple LLM providers (OpenAI, Anthropic, Bedrock,
Databricks, HuggingFace) as mrm model sources. Enables GenAI testing against
API-based language models.

Each adapter:
- Authenticates with provider-specific credentials (env vars)
- Sends prompts and retrieves completions
- Tracks token usage and cost
- Handles rate limiting and retries
- Supports RAG retrieval integration

Example configuration in model YAML:

```yaml
location:
  type: llm_endpoint
  provider: openai
  model_name: gpt-4
  temperature: 0.3
  max_tokens: 500
  
  # Optional RAG retriever
  retriever:
    type: faiss
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
    top_k: 3
    knowledge_base_path: data/knowledge_base.json
```
"""

import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LLMEndpoint(ABC):
    """Abstract base class for LLM endpoint adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM endpoint.

        Args:
            config: Model location configuration from YAML
        """
        self.config = config
        self.model_name = config.get('model_name')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 1000)
        self.top_p = config.get('top_p', 1.0)

        # Token counting for cost estimation
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # RAG retriever (optional)
        self.retriever = None
        if 'retriever' in config:
            self.retriever = self._init_retriever(config['retriever'])

        # Replay capture context (set externally by the runner / CLI).
        # When None, replay is opt-out and zero-overhead.
        self.replay_context: Optional[Any] = None
    
    def _init_retriever(self, retriever_config: Dict[str, Any]):
        """Initialize RAG retriever if configured."""
        retriever_type = retriever_config.get('type', 'faiss')
        
        if retriever_type == 'faiss':
            try:
                from .rag_retriever import FAISSRetriever
                return FAISSRetriever(retriever_config)
            except ImportError as e:
                logger.warning(f"Could not initialize FAISS retriever: {e}")
                return None
        else:
            logger.warning(f"Unknown retriever type: {retriever_type}")
            return None
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate completion from LLM.
        
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
        """
        pass
    
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
        # Retrieve relevant context
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
            self._emit_replay(
                prompt=query,
                response=response,
                metadata=metadata,
                system_prompt=system_prompt,
                retrieved_docs=retrieved_docs,
                extra_kwargs=kwargs,
            )
            return response, metadata
        else:
            # No retriever configured, just generate
            response, metadata = self.generate(query, system_prompt, **kwargs)
            self._emit_replay(
                prompt=query,
                response=response,
                metadata=metadata,
                system_prompt=system_prompt,
                retrieved_docs=None,
                extra_kwargs=kwargs,
            )
            return response, metadata

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Replay-aware completion entry point.

        Always emits a DecisionRecord when ``self.replay_context`` is
        set. Delegates to ``generate_with_retrieval`` so the RAG path
        is exercised when a retriever is configured.
        """
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
        """Emit a DecisionRecord if a replay context is attached."""
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
        
        Args:
            pricing: Dict with 'prompt_token_cost' and 'completion_token_cost'
                    If not provided, uses default pricing for the model
        
        Returns:
            Estimated cost in USD
        """
        if pricing is None:
            pricing = self._get_default_pricing()
        
        prompt_cost = self.total_prompt_tokens * pricing.get('prompt_token_cost', 0)
        completion_cost = self.total_completion_tokens * pricing.get('completion_token_cost', 0)
        
        return prompt_cost + completion_cost
    
    def _get_default_pricing(self) -> Dict[str, float]:
        """Get default pricing for model (override in subclasses)."""
        return {
            'prompt_token_cost': 0.0,
            'completion_token_cost': 0.0
        }
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if endpoint is reachable and credentials are valid.
        
        Returns:
            True if healthy, False otherwise
        """
        pass


class OpenAIEndpoint(LLMEndpoint):
    """OpenAI API endpoint adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=config.get('api_base', 'https://api.openai.com/v1')
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate completion via OpenAI API."""
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                top_p=kwargs.get('top_p', self.top_p)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response
            response_text = response.choices[0].message.content
            
            # Track tokens
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            
            metadata = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'latency_ms': latency_ms,
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _get_default_pricing(self) -> Dict[str, float]:
        """OpenAI pricing (as of May 2026 - approximate)."""
        # GPT-4 pricing
        if 'gpt-4' in self.model_name.lower():
            return {
                'prompt_token_cost': 0.00003,  # $0.03 per 1K tokens
                'completion_token_cost': 0.00006  # $0.06 per 1K tokens
            }
        # GPT-3.5 pricing
        elif 'gpt-3.5' in self.model_name.lower():
            return {
                'prompt_token_cost': 0.0000015,
                'completion_token_cost': 0.000002
            }
        else:
            return super()._get_default_pricing()
    
    def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            # Try listing models as health check
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False


class AnthropicEndpoint(LLMEndpoint):
    """Anthropic Claude API endpoint adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate completion via Anthropic API."""
        start_time = time.time()
        
        try:
            # Anthropic uses system parameter, not messages
            create_kwargs = {
                'model': self.model_name,
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature),
                'messages': [{"role": "user", "content": prompt}]
            }
            
            if system_prompt:
                create_kwargs['system'] = system_prompt
            
            response = self.client.messages.create(**create_kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response text
            response_text = response.content[0].text
            
            # Track tokens
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            
            metadata = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'latency_ms': latency_ms,
                'model': response.model,
                'finish_reason': response.stop_reason
            }
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def _get_default_pricing(self) -> Dict[str, float]:
        """Anthropic Claude pricing (approximate)."""
        if 'opus' in self.model_name.lower():
            return {
                'prompt_token_cost': 0.000015,
                'completion_token_cost': 0.000075
            }
        elif 'sonnet' in self.model_name.lower():
            return {
                'prompt_token_cost': 0.000003,
                'completion_token_cost': 0.000015
            }
        else:
            return super()._get_default_pricing()
    
    def health_check(self) -> bool:
        """Check Anthropic API health."""
        try:
            # Simple test generation
            self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False


class BedrockEndpoint(LLMEndpoint):
    """AWS Bedrock endpoint adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 package not installed. "
                "Install with: pip install boto3"
            )
        
        region = config.get('region', 'us-east-1')
        self.client = boto3.client('bedrock-runtime', region_name=region)
        self.model_id = self.model_name
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate completion via AWS Bedrock."""
        start_time = time.time()
        
        try:
            # Build request body (format depends on model)
            if 'anthropic.claude' in self.model_id:
                # Anthropic Claude on Bedrock
                body = {
                    'prompt': f"\n\nHuman: {prompt}\n\nAssistant:",
                    'max_tokens_to_sample': kwargs.get('max_tokens', self.max_tokens),
                    'temperature': kwargs.get('temperature', self.temperature),
                    'top_p': kwargs.get('top_p', self.top_p)
                }
            elif 'amazon.titan' in self.model_id:
                # Amazon Titan
                body = {
                    'inputText': prompt,
                    'textGenerationConfig': {
                        'maxTokenCount': kwargs.get('max_tokens', self.max_tokens),
                        'temperature': kwargs.get('temperature', self.temperature),
                        'topP': kwargs.get('top_p', self.top_p)
                    }
                }
            else:
                raise ValueError(f"Unsupported Bedrock model: {self.model_id}")
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            if 'anthropic.claude' in self.model_id:
                response_text = response_body.get('completion', '')
                # Bedrock doesn't always return token counts
                prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
                completion_tokens = len(response_text.split()) * 1.3
            elif 'amazon.titan' in self.model_id:
                response_text = response_body['results'][0]['outputText']
                prompt_tokens = response_body.get('inputTextTokenCount', 0)
                completion_tokens = response_body['results'][0].get('tokenCount', 0)
            
            self.total_prompt_tokens += int(prompt_tokens)
            self.total_completion_tokens += int(completion_tokens)
            
            metadata = {
                'prompt_tokens': int(prompt_tokens),
                'completion_tokens': int(completion_tokens),
                'total_tokens': int(prompt_tokens + completion_tokens),
                'latency_ms': latency_ms,
                'model': self.model_id,
                'finish_reason': response_body.get('stop_reason', response_body.get('results', [{}])[0].get('completionReason', 'unknown'))
            }
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check Bedrock endpoint health."""
        try:
            # List available models as health check
            import boto3
            bedrock_client = boto3.client('bedrock', region_name=self.client.meta.region_name)
            bedrock_client.list_foundation_models()
            return True
        except Exception as e:
            logger.error(f"Bedrock health check failed: {e}")
            return False


class DatabricksEndpoint(LLMEndpoint):
    """Databricks Model Serving endpoint adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Databricks connection
        self.host = os.getenv('DATABRICKS_HOST')
        self.token = os.getenv('DATABRICKS_TOKEN')
        
        if not self.host or not self.token:
            raise ValueError(
                "DATABRICKS_HOST and DATABRICKS_TOKEN environment variables required"
            )
        
        self.endpoint_name = config.get('endpoint_name', self.model_name)
        self.api_url = f"{self.host.rstrip('/')}/serving-endpoints/{self.endpoint_name}/invocations"
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate completion via Databricks Model Serving."""
        import requests
        
        start_time = time.time()
        
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        # Build request (format may vary by deployed model)
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        payload = {
            'inputs': [full_prompt],
            'params': {
                'temperature': kwargs.get('temperature', self.temperature),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'top_p': kwargs.get('top_p', self.top_p)
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=kwargs.get('timeout', 60)
            )
            response.raise_for_status()
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = response.json()
            
            # Extract response (format depends on deployed model)
            if 'predictions' in result:
                response_text = result['predictions'][0]
            elif 'choices' in result:
                response_text = result['choices'][0].get('text', '')
            else:
                response_text = str(result)
            
            # Estimate tokens (Databricks may not always return counts)
            prompt_tokens = len(full_prompt.split()) * 1.3
            completion_tokens = len(response_text.split()) * 1.3
            
            self.total_prompt_tokens += int(prompt_tokens)
            self.total_completion_tokens += int(completion_tokens)
            
            metadata = {
                'prompt_tokens': int(prompt_tokens),
                'completion_tokens': int(completion_tokens),
                'total_tokens': int(prompt_tokens + completion_tokens),
                'latency_ms': latency_ms,
                'model': self.endpoint_name,
                'finish_reason': 'stop'
            }
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"Databricks API error: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check Databricks endpoint health."""
        import requests
        
        try:
            headers = {'Authorization': f'Bearer {self.token}'}
            health_url = f"{self.host.rstrip('/')}/api/2.0/serving-endpoints/{self.endpoint_name}"
            
            response = requests.get(health_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            endpoint_info = response.json()
            state = endpoint_info.get('state', {}).get('ready', '')
            
            return state == 'READY'
            
        except Exception as e:
            logger.error(f"Databricks health check failed: {e}")
            return False


class HuggingFaceEndpoint(LLMEndpoint):
    """HuggingFace Inference API endpoint adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate completion via HuggingFace Inference API."""
        import requests
        
        start_time = time.time()
        
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        # Combine system prompt if provided
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        payload = {
            'inputs': full_prompt,
            'parameters': {
                'temperature': kwargs.get('temperature', self.temperature),
                'max_new_tokens': kwargs.get('max_tokens', self.max_tokens),
                'top_p': kwargs.get('top_p', self.top_p),
                'return_full_text': False
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=kwargs.get('timeout', 60)
            )
            response.raise_for_status()
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = response.json()
            
            # Extract response
            if isinstance(result, list) and len(result) > 0:
                response_text = result[0].get('generated_text', '')
            else:
                response_text = result.get('generated_text', str(result))
            
            # Estimate tokens (HF doesn't always return counts)
            prompt_tokens = len(full_prompt.split()) * 1.3
            completion_tokens = len(response_text.split()) * 1.3
            
            self.total_prompt_tokens += int(prompt_tokens)
            self.total_completion_tokens += int(completion_tokens)
            
            metadata = {
                'prompt_tokens': int(prompt_tokens),
                'completion_tokens': int(completion_tokens),
                'total_tokens': int(prompt_tokens + completion_tokens),
                'latency_ms': latency_ms,
                'model': self.model_name,
                'finish_reason': 'stop'
            }
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check HuggingFace API health."""
        import requests
        
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            # Simple test query
            response = requests.post(
                self.api_url,
                headers=headers,
                json={'inputs': 'Hi', 'parameters': {'max_new_tokens': 5}},
                timeout=30
            )
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"HuggingFace health check failed: {e}")
            return False


def get_llm_endpoint(config: Dict[str, Any]) -> LLMEndpoint:
    """
    Factory function to create appropriate LLM endpoint adapter.
    
    Args:
        config: Model location configuration from YAML
    
    Returns:
        LLMEndpoint instance for the specified provider
    
    Raises:
        ValueError: If provider is unknown or not supported
    
    Note:
        As of mrm-core v0.2+, using provider='litellm' is recommended.
        LiteLLM provides unified access to 100+ providers with a single
        interface, automatic cost tracking, and built-in retries.
        
        Legacy provider-specific adapters (openai, anthropic, etc.) are
        still supported for backward compatibility.
    """
    provider = config.get('provider', '').lower()
    
    # Recommended: Use LiteLLM for unified interface
    if provider == 'litellm':
        try:
            from .litellm_endpoint import LiteLLMEndpoint
            return LiteLLMEndpoint(config)
        except ImportError:
            raise ImportError(
                "litellm not installed. "
                "Install with: pip install litellm"
            )
    
    # Legacy provider-specific adapters (backward compatibility)
    elif provider == 'openai':
        return OpenAIEndpoint(config)
    elif provider == 'anthropic':
        return AnthropicEndpoint(config)
    elif provider == 'bedrock':
        return BedrockEndpoint(config)
    elif provider == 'databricks':
        return DatabricksEndpoint(config)
    elif provider == 'huggingface':
        return HuggingFaceEndpoint(config)
    else:
        # If no provider specified or unknown, suggest LiteLLM
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported: litellm (recommended), openai, anthropic, bedrock, databricks, huggingface. "
            f"For best compatibility, use provider='litellm' which supports 100+ models."
        )
