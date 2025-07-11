"""
Intelligent Customer Support Agent - Phase 1: Simple Chain-of-Thought Chatbot
A fully configurable foundation that will scale through all 10 phases.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path
import yaml
from enum import Enum

# ============================================================================
# CONFIGURATION SYSTEM
# ============================================================================

class ConfigLoader:
    """Central configuration loader supporting multiple formats"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                return yaml.safe_load(file)
            elif config_path.suffix.lower() == '.json':
                return json.load(file)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

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

# ============================================================================
# CORE COMPONENTS
# ============================================================================

class ConversationState(Enum):
    """Conversation states"""
    GREETING = "greeting"
    LISTENING = "listening"
    THINKING = "thinking"
    RESPONDING = "responding"
    WAITING = "waiting"
    CLOSING = "closing"

@dataclass
class Message:
    """Individual message structure"""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ConversationContext:
    """Conversation context and memory"""
    messages: List[Message]
    state: ConversationState
    user_id: str
    session_id: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class PromptManager:
    """Manages prompt templates and generation"""
    
    def __init__(self, config: PromptConfig):
        self.config = config
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates"""
        return {
            "default_system_prompt": """You are {bot_name}, an intelligent customer support assistant.
You are helpful, professional, and empathetic. You think step by step to provide the best possible assistance.

Key principles:
- Always be polite and professional
- Think through problems systematically
- Ask clarifying questions when needed
- Provide clear, actionable solutions
- Show empathy for user concerns

{reasoning_instruction}

Current conversation context: {context_summary}""",
            
            "default_user_prompt": """User message: {user_input}

{chain_of_thought_instruction}

Please provide a helpful response.""",
            
            "chain_of_thought_instruction": """Before responding, please think through:
1. What is the user asking or concerned about?
2. What information do I need to provide a complete answer?
3. What would be the most helpful response?

Your reasoning: """,
            
            "greeting_prompt": """Welcome! I'm {bot_name}, your customer support assistant. 
How can I help you today?""",
            
            "fallback_prompt": """I apologize, but I'm having trouble understanding your request. 
Could you please rephrase or provide more details about what you need help with?"""
        }
    
    def get_system_prompt(self, context: ConversationContext, bot_name: str) -> str:
        """Generate system prompt"""
        template = self.templates.get(self.config.system_prompt_template, 
                                    self.templates["default_system_prompt"])
        
        reasoning_instruction = ""
        if self.config.chain_of_thought_enabled:
            reasoning_instruction = f"Always start your reasoning with: '{self.config.reasoning_prefix}'"
        
        context_summary = self._generate_context_summary(context)
        
        return template.format(
            bot_name=bot_name,
            reasoning_instruction=reasoning_instruction,
            context_summary=context_summary
        )
    
    def get_user_prompt(self, user_input: str) -> str:
        """Generate user prompt"""
        template = self.templates.get(self.config.user_prompt_template,
                                    self.templates["default_user_prompt"])
        
        chain_instruction = ""
        if self.config.chain_of_thought_enabled:
            chain_instruction = self.templates["chain_of_thought_instruction"]
        
        return template.format(
            user_input=user_input,
            chain_of_thought_instruction=chain_instruction
        )
    
    def _generate_context_summary(self, context: ConversationContext) -> str:
        """Generate a summary of conversation context"""
        if not context.messages:
            return "This is the start of a new conversation."
        
        recent_messages = context.messages[-3:]  # Last 3 messages
        summary_parts = []
        
        for msg in recent_messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            summary_parts.append(f"{role_label}: {msg.content[:100]}...")
        
        return "\n".join(summary_parts)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """Generate response from LLM"""
        pass

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

class ConversationManager:
    """Manages conversation state and context"""
    
    def __init__(self, config: ConversationConfig):
        self.config = config
        self.conversations: Dict[str, ConversationContext] = {}
    
    def get_or_create_conversation(self, user_id: str, session_id: str) -> ConversationContext:
        """Get existing conversation or create new one"""
        conv_key = f"{user_id}:{session_id}"
        
        if conv_key not in self.conversations:
            self.conversations[conv_key] = ConversationContext(
                messages=[],
                state=ConversationState.GREETING,
                user_id=user_id,
                session_id=session_id,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        return self.conversations[conv_key]
    
    def add_message(self, conversation: ConversationContext, message: Message):
        """Add message to conversation"""
        conversation.messages.append(message)
        conversation.updated_at = datetime.now()
        
        # Trim context if too long
        if len(conversation.messages) > self.config.max_context_length:
            # Keep system message + recent messages
            system_msgs = [m for m in conversation.messages if m.role == "system"]
            recent_msgs = conversation.messages[-(self.config.max_context_length-len(system_msgs)):]
            conversation.messages = system_msgs + recent_msgs
    
    def get_context_messages(self, conversation: ConversationContext) -> List[Dict[str, str]]:
        """Get messages formatted for LLM"""
        return [
            {"role": msg.role, "content": msg.content} 
            for msg in conversation.messages
        ]

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

# ============================================================================
# MAIN BOT CLASS
# ============================================================================

class ChainOfThoughtChatbot:
    """Main chatbot class with full configurability"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load configuration
        config_data = ConfigLoader.load_config(config_path)
        self.config = BotConfig(**config_data)
        
        # Initialize components
        self.prompt_manager = PromptManager(self.config.prompts)
        self.llm_provider = self._create_llm_provider()
        self.conversation_manager = ConversationManager(self.config.conversation)
        self.security_manager = SecurityManager(self.config.security)
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Initialized {self.config.bot_name} v{self.config.version}")
    
    def _create_llm_provider(self) -> LLMProvider:
        """Factory method for LLM provider"""
        if self.config.llm.provider.lower() == "openai":
            return OpenAIProvider(self.config.llm)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level.upper()),
            format=self.config.logging.format
        )
        self.logger = logging.getLogger(self.config.bot_name)
        
        if self.config.logging.file_path:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.config.logging.file_path,
                maxBytes=self.config.logging.max_file_size,
                backupCount=self.config.logging.backup_count
            )
            file_handler.setFormatter(logging.Formatter(self.config.logging.format))
            self.logger.addHandler(file_handler)
    
    def chat(self, user_input: str, user_id: str = "default", session_id: str = "default") -> str:
        """Main chat interface"""
        try:
            # Security validation
            is_valid, error_msg = self.security_manager.validate_input(user_input)
            if not is_valid:
                self.logger.warning(f"Security validation failed: {error_msg}")
                return "I'm sorry, but I can't process that request."
            
            # Get conversation context
            conversation = self.conversation_manager.get_or_create_conversation(user_id, session_id)
            
            # Handle greeting state
            if conversation.state == ConversationState.GREETING and self.config.conversation.greeting_enabled:
                greeting = self.prompt_manager.templates["greeting_prompt"].format(
                    bot_name=self.config.bot_name
                )
                conversation.state = ConversationState.LISTENING
                return greeting
            
            # Add user message
            user_message = Message(
                role="user",
                content=user_input,
                timestamp=datetime.now()
            )
            self.conversation_manager.add_message(conversation, user_message)
            
            # Generate response
            response = self._generate_response(conversation)
            
            # Add assistant message
            assistant_message = Message(
                role="assistant",
                content=response,
                timestamp=datetime.now()
            )
            self.conversation_manager.add_message(conversation, assistant_message)
            
            # Log conversation if enabled
            if self.config.logging.log_conversations:
                self.logger.info(f"User [{user_id}]: {user_input}")
                self.logger.info(f"Bot: {response}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Chat error: {str(e)}")
            return "I apologize, but I'm having technical difficulties. Please try again."
    
    def _generate_response(self, conversation: ConversationContext) -> str:
        """Generate response using LLM"""
        conversation.state = ConversationState.THINKING
        
        # Prepare system prompt
        system_prompt = self.prompt_manager.get_system_prompt(conversation, self.config.bot_name)
        
        # Prepare messages for LLM
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        context_messages = self.conversation_manager.get_context_messages(conversation)
        messages.extend(context_messages)
        
        # Generate response
        conversation.state = ConversationState.RESPONDING
        response_content, metadata = self.llm_provider.generate_response(messages)
        
        conversation.state = ConversationState.WAITING
        
        return response_content
    
    def get_conversation_history(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        conv_key = f"{user_id}:{session_id}"
        if conv_key not in self.conversation_manager.conversations:
            return []
        
        conversation = self.conversation_manager.conversations[conv_key]
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in conversation.messages
        ]
    
    def clear_conversation(self, user_id: str, session_id: str):
        """Clear conversation history"""
        conv_key = f"{user_id}:{session_id}"
        if conv_key in self.conversation_manager.conversations:
            del self.conversation_manager.conversations[conv_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics"""
        total_conversations = len(self.conversation_manager.conversations)
        total_messages = sum(
            len(conv.messages) 
            for conv in self.conversation_manager.conversations.values()
        )
        
        return {
            "bot_name": self.config.bot_name,
            "version": self.config.version,
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "environment": self.config.environment
        }

# ============================================================================
# EXAMPLE USAGE AND CONFIGURATION
# ============================================================================

def create_default_config():
    """Create default configuration file"""
    config = {
        "bot_name": "SupportBot",
        "version": "1.0.0",
        "environment": "development",
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key_env": "OPENAI_API_KEY"
        },
        "conversation": {
            "max_context_length": 10,
            "enable_memory": True,
            "greeting_enabled": True,
            "fallback_enabled": True
        },
        "prompts": {
            "chain_of_thought_enabled": True,
            "reasoning_prefix": "Let me think about this step by step:",
            "include_examples": True
        },
        "logging": {
            "level": "INFO",
            "log_conversations": True,
            "file_path": "logs/chatbot.log"
        },
        "security": {
            "enable_content_filter": True,
            "max_input_length": 2000,
            "blocked_patterns": ["spam", "abuse"]
        }
    }
    
    with open("configs/config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config

def main():
    """Example usage"""
    # Create default config if it doesn't exist
    if not Path("configs/config.yaml").exists():
        create_default_config()
        print("Created default configs/config.yaml - please set your OpenAI API key in environment")
        return
    
    # Initialize bot
    bot = ChainOfThoughtChatbot("configs/config.yaml")
    
    print(f"=== {bot.config.bot_name} v{bot.config.version} ===")
    print("Type 'quit' to exit, 'stats' for statistics, 'clear' to clear conversation")
    print()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'stats':
            stats = bot.get_stats()
            print(f"Stats: {json.dumps(stats, indent=2)}")
            continue
        elif user_input.lower() == 'clear':
            bot.clear_conversation("default", "default")
            print("Conversation cleared!")
            continue
        
        if user_input:
            response = bot.chat(user_input)
            print(f"Bot: {response}")
            print()

if __name__ == "__main__":
    main()
