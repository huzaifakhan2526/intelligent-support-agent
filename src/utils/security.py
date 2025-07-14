"""Security utilities"""
from typing import Tuple
from ..config.models import SecurityConfig

class SecurityManager:
    """Handles security and content filtering"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def validate_input(self, user_input: str) -> Tuple[bool, str]:
        """Validate user input for security"""
        if len(user_input) > self.config.max_input_length:
            return False, f"Input too long (max {self.config.max_input_length} characters)"
        
        if self.config.enable_content_filter:
            for pattern in self.config.blocked_patterns:
                if pattern.lower() in user_input.lower():
                    return False, "Input contains blocked content"
        
        return True, ""
