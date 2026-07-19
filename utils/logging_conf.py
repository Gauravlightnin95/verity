"""Structured per-agent logging: console + rotating JSON file handler on a
named "verity" logger tree - never touches the root logger, so importing
this module can't clobber handlers any other library or the host app set up.

Usage: `from utils.logging_conf import get_logger; log = get_logger(__name__)`
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Final

from pythonjsonlogger import jsonlogger

from app.config import settings

_CONSOLE_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES: Final[int] = 5 * 1024 * 1024
_BACKUP_COUNT: Final[int] = 3
_NAMESPACE: Final[str] = "verity"

_configured = False


def _build_console_handler() -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    if settings.log_format == "json":
        formatter: logging.Formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s", datefmt=_DATE_FORMAT
        )
    else:
        formatter = logging.Formatter(fmt=_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)
    handler.setFormatter(formatter)
    return handler


def _build_file_handler() -> logging.Handler:
    settings.log_file_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        filename=str(settings.log_file_path),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(
        jsonlogger.JsonFormatter(fmt="%(asctime)s %(levelname)s %(name)s %(message)s", datefmt=_DATE_FORMAT)
    )
    return handler


def configure_logging() -> None:
    """Configure the "verity" logger tree exactly once per process. Safe to
    call repeatedly - later calls are no-ops."""
    global _configured
    if _configured:
        return

    verity_logger = logging.getLogger(_NAMESPACE)
    verity_logger.setLevel(settings.log_level)
    verity_logger.propagate = False
    verity_logger.addHandler(_build_console_handler())
    verity_logger.addHandler(_build_file_handler())

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the "verity" namespace. Pass `__name__` from
    the calling module."""
    configure_logging()
    return logging.getLogger(f"{_NAMESPACE}.{name}")
