"""Test configuration loading"""
import pytest
import tempfile
import os
import yaml
import json
from src.config.loader import ConfigLoader
from src.config.models import BotConfig

def test_load_yaml_config():
    """Test loading YAML configuration"""
    config_data = {
        "bot_name": "TestBot",
        "version": "1.0.0",
        "llm": {"provider": "openai", "model": "gpt-3.5-turbo"}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        loaded_config = ConfigLoader.load_config(temp_path)
        assert loaded_config["bot_name"] == "TestBot"
        assert loaded_config["llm"]["provider"] == "openai"
    finally:
        os.unlink(temp_path)

def test_bot_config_creation():
    """Test creating bot config with defaults"""
    config = BotConfig()
    assert config.bot_name == "SupportBot"
    assert config.version == "1.0.0"
