"""Loguru logging configuration for the forecasting pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_LOG_DIR: Path = _PROJECT_ROOT / "logs"

_CONSOLE_FORMAT: str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)

_FILE_FORMAT: str = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{module}:{function}:{line} — "
    "{message}"
)


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru with console and rotating file sinks.

    Parameters
    ----------
    level:
        Minimum log level for **both** sinks (e.g. ``"DEBUG"``, ``"INFO"``).
    """
    # Remove default handler to avoid duplicate output
    logger.remove()

    # Console sink — colorful
    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File sink — rotating at 10 MB
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        _LOG_DIR / "pipeline_{time:YYYY-MM-DD}.log",
        format=_FILE_FORMAT,
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,  # thread-safe
    )

    logger.info("Logging initialised — level={}", level)
