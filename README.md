# Intelligent Customer Support Agent - Phase 1

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone/download and setup**
```bash
git clone <repository-url>
cd intelligent-support-agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Configure API key

bash# Edit .env file and add your OpenAI API key
OPENAI_API_KEY=your_actual_api_key_here

Run the chatbot

bashpython src/chatbot.py
📋 Features
✅ Chain-of-Thought Reasoning - Step-by-step problem solving
✅ Fully Configurable - Everything configurable via YAML
✅ Conversation Memory - Context-aware responses
✅ Security Layer - Input validation and filtering
✅ Multi-Environment - Dev, test, production configs
✅ Production Ready - Logging, error handling, monitoring
⚙️ Configuration
All settings in configs/config.yaml:

LLM provider and model settings
Conversation behavior
Prompt templates
Security settings
Logging configuration

🔧 Usage
Basic Chat
pythonfrom src.chatbot import ChainOfThoughtChatbot

bot = ChainOfThoughtChatbot("configs/config.yaml")
response = bot.chat("Hello, I need help", "user123", "session456")
Commands

quit - Exit the chatbot
stats - Show statistics
clear - Clear conversation history

🧪 Testing
bashmake test        # Run tests
make test-cov    # Run with coverage
make format      # Format code
make lint        # Lint code
🚀 Ready for All 10 Phases!
This foundation scales through:

Phase 2: Intent Classification
Phase 3: State Machine
Phase 4: RAG/FAQ System
Phase 5: Multi-Tool Agent
Phases 6-10: Advanced features


Get started: Edit .env, add your OpenAI API key, then run python src/chatbot.py
