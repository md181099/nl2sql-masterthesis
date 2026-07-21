#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "transformers",
)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False)


def _resolve_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    value = logging.getLevelName(str(level).upper())
    if isinstance(value, int):
        return value
    raise ValueError(f"Invalid log level: {level}")


def _resolve_log_format(log_format: str) -> str:
    fmt = str(log_format).strip().lower()
    if fmt not in {"text", "json"}:
        raise ValueError("Invalid log format. Use 'text' or 'json'.")
    return fmt


def setup_logging(level: str | int = "INFO", log_format: str = "text") -> logging.Logger:
    resolved_level = _resolve_log_level(level)
    resolved_format = _resolve_log_format(log_format)
    if resolved_format == "json":
        formatter: logging.Formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)

    root_logger.setLevel(resolved_level)

    # Keep project logs at configured level, but quiet noisy third-party libraries.
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    return root_logger
