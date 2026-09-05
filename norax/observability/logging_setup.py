"""Structured JSON logging setup for Norax.

When NORAX_LOG_JSON=1 is set, logs are emitted as JSON lines with:
  timestamp, level, logger, message, and any extra fields.

This makes logs machine-parseable for log aggregation systems (Loki,
Elasticsearch, Datadog) and enables structured queries on log data.

Usage:
  from norax.observability.logging_setup import setup_logging
  setup_logging()  # call at startup, before any other code logs
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Include exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Include any extra attributes passed to the log call
        for key, val in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "taskName",
            ):
                try:
                    json.dumps(val)  # test serializability
                    log_entry[key] = val
                except (TypeError, ValueError):
                    log_entry[key] = str(val)
        return json.dumps(log_entry, default=str)


def setup_logging() -> None:
    """Configure logging — JSON if NORAX_LOG_JSON=1, plain text otherwise."""
    level_name = os.environ.get("NORAX_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    use_json = os.environ.get("NORAX_LOG_JSON", "0") == "1"

    handler = logging.StreamHandler(sys.stderr)
    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Reduce noise from chatty libraries
    for noisy in ("httpx", "httpcore", "uvicorn.access", "discord.http", "discord.gateway"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
