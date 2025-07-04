"""Enhanced rich logging configuration for better colored output"""

import logging
import sys
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Create a custom theme for different log levels
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "debug": "dim cyan",
    "success": "bold green",
})

# Create console with custom theme
console = Console(theme=custom_theme)


class RichColoredHandler(RichHandler):
    """Enhanced RichHandler with better formatting"""
    
    def __init__(self, **kwargs):
        # Extract our custom parameters before passing to parent
        show_path = kwargs.pop('show_path', True)
        show_time = kwargs.pop('show_time', True)
        log_time_format = kwargs.pop('log_time_format', "[%Y-%m-%d %H:%M:%S]")
        omit_repeated_times = kwargs.pop('omit_repeated_times', False)
        
        super().__init__(
            console=console,
            show_time=show_time,
            show_level=True,
            show_path=show_path,
            enable_link_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            log_time_format=log_time_format,
            omit_repeated_times=omit_repeated_times,
            **kwargs
        )


def setup_rich_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    show_path: bool = True,
    show_time: bool = True
):
    """
    Setup rich colored logging for the entire application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path for file logging
        show_path: Show file path and line number
        show_time: Show timestamp
    """
    
    # Create the rich handler with custom formatting
    rich_handler = RichColoredHandler(
        show_path=show_path,
        show_time=show_time,
        omit_repeated_times=False,
        log_time_format="[%Y-%m-%d %H:%M:%S]"
    )
    
    # Configure the formatting
    FORMAT = "%(message)s"
    rich_handler.setFormatter(logging.Formatter(FORMAT))
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add the rich handler
    root_logger.addHandler(rich_handler)
    
    # Optionally add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers to avoid duplicate messages
    loggers_to_configure = [
        'app.butler_agent',
        'app.butler_agent_enhanced',
        'app.butler_executor',
        'app.butler_executor_enhanced',
        'app.agent',
        'app.agent_executor',
        'app.agent_call',
        'app.butler_tools',
        '__main__',
        'httpx',
        'httpcore',
        'a2a.client.client',
        'a2a.client',
        'uvicorn',
        'uvicorn.error',
        'uvicorn.access'
    ]
    
    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = True  # Use root logger's handlers
        
    # Set httpx to WARNING to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Customize uvicorn logging
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_rich_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a logger with rich colored output
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger with rich output
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Only add handler if logger doesn't already have one
        handler = RichColoredHandler(
            show_path=True,
            show_time=True,
            log_time_format="[%Y-%m-%d %H:%M:%S]"
        )
        
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = False  # Don't propagate to root logger
    
    return logger


# Convenience function to log with custom styles
def log_success(logger: logging.Logger, message: str):
    """Log a success message with green color"""
    logger.info(f"[success]✅ {message}[/success]")


def log_error(logger: logging.Logger, message: str):
    """Log an error message with red color"""
    logger.error(f"[error]❌ {message}[/error]")


def log_warning(logger: logging.Logger, message: str):
    """Log a warning message with yellow color"""
    logger.warning(f"[warning]⚠️  {message}[/warning]")


def log_info(logger: logging.Logger, message: str):
    """Log an info message with cyan color"""
    logger.info(f"[info]ℹ️  {message}[/info]")


def log_debug(logger: logging.Logger, message: str):
    """Log a debug message with dim cyan color"""
    logger.debug(f"[debug]🔍 {message}[/debug]")


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_rich_logging(level="DEBUG")
    
    # Get a logger
    logger = get_rich_logger(__name__)
    
    # Test different log levels
    log_debug(logger, "This is a debug message")
    log_info(logger, "This is an info message")
    log_success(logger, "This is a success message")
    log_warning(logger, "This is a warning message")
    log_error(logger, "This is an error message")
    logger.critical("This is a critical message")