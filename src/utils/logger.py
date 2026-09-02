"""Centralized Logging Infrastructure Module.

Provides standard Python logging setup with configurable log levels, colorized console
output, and optional file logging for quantitative research pipeline modules.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "QuantPipeline", level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """Configures and returns a standard Python logger instance.

    Args:
        name: Name of the logger symbol. Defaults to "QuantPipeline".
        level: Logging severity level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional filepath to write log output.

    Returns:
        logging.Logger: Configured logger instance ready for logging calls.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default global logger
logger = setup_logger()
