"""
Logging utilities with consistent formatting and file output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

_logger_instance: logging.Logger | None = None


def get_logger(name: str = "quant_bench", log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Get or create a configured logger."""
    global _logger_instance
    if _logger_instance:
        return _logger_instance

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)

    # File handler (optional)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_path / f"run_{timestamp}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)

    _logger_instance = logger
    return logger
