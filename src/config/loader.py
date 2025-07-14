"""Configuration loading utilities"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Central configuration loader supporting multiple formats"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(file)
            elif config_path.suffix.lower() == '.json':
                return json.load(file)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
