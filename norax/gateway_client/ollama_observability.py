"""Ollama observability — metrics + daily JSONL usage logs."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("norax.gateway_client.ollama_observability")

_DEFAULT_LOG_DIR = Path.home() / ".local" / "state" / "norax" / "logs" / "gateway"


def _default_log_dir() -> Path:
    return Path(os.environ.get("NORAX_LOG_DIR", _DEFAULT_LOG_DIR.parent)) / "gateway"


class OllamaObservability:
    """Record Ollama gateway calls to Prometheus + daily JSONL."""

    def __init__(
        self,
        metrics: Any | None = None,
        *,
        log_dir: Path | str | None = None,
    ) -> None:
        self.metrics = metrics
        self.log_dir = Path(log_dir) if log_dir is not None else _default_log_dir()

    def record_call(
        self,
        *,
        model: str,
        latency_ms: float,
        tool_count: int = 0,
        profile_role: str = "general",
        status: str = "ok",
        routed_from: str | None = None,
    ) -> None:
        if self.metrics is not None:
            try:
                self.metrics.ollama_requests.labels(
                    model=model,
                    status=status,
                ).inc()
                self.metrics.ollama_latency.labels(model=model).observe(
                    latency_ms / 1000.0,
                )
                if tool_count:
                    self.metrics.ollama_tools.labels(model=model).inc(tool_count)
            except Exception:  # noqa: BLE001
                log.debug("ollama_obs.metrics_failed", exc_info=True)

        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "tool_count": tool_count,
            "profile_role": profile_role,
            "status": status,
        }
        if routed_from:
            entry["routed_from"] = routed_from
        self._append_daily_log(entry)

    def _append_daily_log(self, entry: dict[str, Any]) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            day = datetime.now(UTC).strftime("%Y-%m-%d")
            path = self.log_dir / f"ollama-{day}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:  # noqa: BLE001
            log.debug("ollama_obs.log_failed", exc_info=True)
