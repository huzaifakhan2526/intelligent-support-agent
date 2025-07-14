"""Logging utilities"""
import logging
from logging.handlers import RotatingFileHandler
from ..config.models import LoggingConfig

def setup_logging(config: LoggingConfig, logger_name: str):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format
    )
    logger = logging.getLogger(logger_name)
    
    if config.file_path:
        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(logging.Formatter(config.format))
        logger.addHandler(file_handler)
    
    return logger
