"""OpenAI LLM provider implementation"""
import os
from typing import Dict, List, Tuple, Any
from .base import LLMProvider
from ..config.models import LLMConfig

class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            import openai
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in environment variable: {self.config.api_key_env}")
            
            client = openai.OpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            return client
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def generate_response(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """Generate response using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            
            content = response.choices[0].message.content
            metadata = {
                "model": self.config.model,
                "tokens_used": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason
            }
            
            return content, metadata
            
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
