"""Logging utilities for FertiScope research workflows."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "fertiscope", level: int = logging.INFO) -> logging.Logger:
    """Return a configured structured logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

