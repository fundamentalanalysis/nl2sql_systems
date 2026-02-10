"""Logging configuration using loguru."""
import sys
from pathlib import Path
from loguru import logger
from app.config import settings


def setup_logging():
    """Configure logging for the application."""
    # Remove default handler
    logger.remove()
    
    # Add console handler with formatting (stderr is more reliable with uvicorn/streamlit)
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        colorize=False,  # Disable colors for better PowerShell compatibility
        enqueue=False,   # Avoid queue buffering so logs show immediately in the terminal
    )
    
    # Add file handler
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        rotation="10 MB",
        retention="1 week",
        compression="zip",
    )
    
    logger.info("Logging configured successfully")
    return logger


# Initialize logger
app_logger = setup_logging()
