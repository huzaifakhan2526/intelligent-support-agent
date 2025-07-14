"""Base LLM provider interface"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """Generate response from LLM"""
        pass
