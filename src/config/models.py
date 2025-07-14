"""Configuration data models"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None

@dataclass
class ConversationConfig:
    """Conversation flow configuration"""
    max_context_length: int = 10
    context_window_tokens: int = 4000
    enable_memory: bool = True
    memory_decay_turns: int = 20
    greeting_enabled: bool = True
    fallback_enabled: bool = True
    max_retries: int = 3

@dataclass
class PromptConfig:
    """Prompt engineering configuration"""
    system_prompt_template: str = "default_system_prompt"
    user_prompt_template: str = "default_user_prompt"
    chain_of_thought_enabled: bool = True
    reasoning_prefix: str = "Let me think about this step by step:"
    confidence_threshold: float = 0.7
    include_examples: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    log_conversations: bool = True

@dataclass
class SecurityConfig:
    """Security and safety configuration"""
    enable_content_filter: bool = True
    max_input_length: int = 2000
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    blocked_patterns: List[str] = None
    
    def __post_init__(self):
        if self.blocked_patterns is None:
            self.blocked_patterns = []

@dataclass
class BotConfig:
    """Main bot configuration container"""
    bot_name: str = "SupportBot"
    version: str = "1.0.0"
    environment: str = "development"
    llm: LLMConfig = None
    conversation: ConversationConfig = None
    prompts: PromptConfig = None
    logging: LoggingConfig = None
    security: SecurityConfig = None
    
    def __post_init__(self):
        if self.llm is None:
            self.llm = LLMConfig()
        if self.conversation is None:
            self.conversation = ConversationConfig()
        if self.prompts is None:
            self.prompts = PromptConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.security is None:
            self.security = SecurityConfig()
