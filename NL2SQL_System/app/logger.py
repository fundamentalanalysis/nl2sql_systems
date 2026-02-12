"""Logging configuration using loguru."""
import sys
from pathlib import Path
from loguru import logger
from app.config import settings


def setup_logging():
    """Configure logging for the application."""
    # Remove default handler
    logger.remove()

    def _is_trace_record(record):
        return bool(record["extra"].get("trace_event"))

    def _is_app_record(record):
        return not _is_trace_record(record)
    
    # Add console handler with formatting (stderr is more reliable with uvicorn/streamlit)
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        colorize=False,  # Disable colors for better PowerShell compatibility
        enqueue=False,   # Avoid queue buffering so logs show immediately in the terminal
        filter=_is_app_record,
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
        filter=_is_app_record,
    )

    # Structured JSON trace sink for machine/audit consumption.
    trace_log_file = "logs/ai_trace.log"
    Path(trace_log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        trace_log_file,
        format="{message}",
        level="INFO",
        rotation="10 MB",
        retention="1 week",
        compression="zip",
        filter=_is_trace_record,
    )
    
    logger.info("Logging configured successfully")
    return logger


# Initialize logger
app_logger = setup_logging()
