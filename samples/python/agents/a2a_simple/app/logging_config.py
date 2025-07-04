"""Colored logging configuration for the butler agent"""

import logging
import sys
from typing import Optional

# ANSI color codes
COLORS = {
    'DEBUG': '\033[36m',      # Cyan
    'INFO': '\033[32m',       # Green
    'WARNING': '\033[33m',    # Yellow
    'ERROR': '\033[31m',      # Red
    'CRITICAL': '\033[35m',   # Magenta
    'RESET': '\033[0m',       # Reset
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels"""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
        
        # Create format strings for each log level with colors
        self.formatters = {}
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            if self.use_colors and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
                color = COLORS.get(level, COLORS['RESET'])
                reset = COLORS['RESET']
                # Color the level name
                level_format = fmt.replace('%(levelname)s', f'{color}%(levelname)s{reset}')
                # Also make the whole line colored for ERROR and CRITICAL
                if level in ['ERROR', 'CRITICAL']:
                    level_format = f'{color}{level_format}{reset}'
            else:
                level_format = fmt
            self.formatters[level] = logging.Formatter(level_format, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        # Get the appropriate formatter for this level
        formatter = self.formatters.get(record.levelname, self.formatters['INFO'])
        
        # Add file location info
        record.location = f"{record.filename}:{record.lineno}"
        
        return formatter.format(record)


def setup_colored_logging(level: str = "INFO"):
    """Setup colored logging for the application"""
    
    # Create a colored formatter with file location
    formatter = ColoredFormatter(
        fmt='%(asctime)s - %(location)-20s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create a stream handler with the colored formatter
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add our colored handler
    root_logger.addHandler(handler)
    
    # Also configure specific loggers
    loggers_to_configure = [
        'app.butler_agent_enhanced',
        'app.butler_executor_enhanced',
        '__main__',
        'httpx',
        'a2a.client.client',
        'uvicorn',
        'uvicorn.error',
        'uvicorn.access'
    ]
    
    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = True  # Use root logger's handlers


def get_colored_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a logger with colored output"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Only add handler if logger doesn't already have one
        formatter = ColoredFormatter(
            fmt='%(asctime)s - %(location)-20s - %(levelname)-8s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = False  # Don't propagate to root logger
    
    return logger