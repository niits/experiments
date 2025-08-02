"""Logging configuration for the writing tool package."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
) -> logging.Logger:
    """Setup a logger with the specified configuration.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
        handler: Custom handler (default: StreamHandler to stdout)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Default handler
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with default configuration.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return setup_logger(name)
