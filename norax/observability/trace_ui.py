"""Tracing & Observability UI — web-based trace viewer for agent behavior.

Provides a lightweight HTTP server that serves:
  - Timeline view of tool calls, memory retrieval, model calls
  - Token/cost tracking per turn
  - Memory retrieval visualization (which signals fired, what was returned)
  - Causal graph visualization

Architecture:
  - TraceCollector: collects trace events from the runtime
  - TraceServer: HTTP server serving the trace UI
  - TraceViewer: HTML/JS frontend (embedded, no external deps)

The trace UI is served at http://localhost:8895 when enabled.

Usage:
    collector = TraceCollector()
    collector.record_turn(...)
    server = TraceServer(collector, port=8895)
    server.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger("norax.observability.trace_ui")

# ── Trace data structures ────────────────────────────────────────────────


@dataclass
class TraceEvent:
    """A single trace event."""

    timestamp: float
    type: str  # turn_start, tool_call, memory_retrieval, model_call, turn_end
    data: dict = field(default_factory=dict)
    turn_id: str = ""
    duration_ms: float = 0.0


@dataclass
class TurnTrace:
    """All events for a single turn."""

    turn_id: str
    start_time: float
    end_time: float = 0.0
    events: list[TraceEvent] = field(default_factory=list)
    tool_calls: int = 0
    memory_hits: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    rounds: int = 0

    def total_duration_ms(self) -> float:
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return 0.0


# ── Trace Collector ─────────────────────────────────────────────────────


class TraceCollector:
    """Collects trace events for the observability UI.

    Bounded ring buffer — keeps last N turns.
    """

    def __init__(self, max_turns: int = 100) -> None:
        self.max_turns = max_turns
        self._turns: deque[TurnTrace] = deque(maxlen=max_turns)
        self._current: dict[str, TurnTrace] = {}
        self._lock = asyncio.Lock()

    async def start_turn(self, turn_id: str, model: str = "") -> None:
        async with self._lock:
            trace = TurnTrace(turn_id=turn_id, start_time=time.time(), model=model)
            self._current[turn_id] = trace

    async def record_event(
        self, turn_id: str, event_type: str, data: dict, duration_ms: float = 0.0
    ) -> None:
        async with self._lock:
            trace = self._current.get(turn_id)
            if trace is None:
                return
            event = TraceEvent(
                timestamp=time.time(),
                type=event_type,
                data=data,
                turn_id=turn_id,
                duration_ms=duration_ms,
            )
            trace.events.append(event)
            if event_type == "tool_call":
                trace.tool_calls += 1
            elif event_type == "memory_retrieval":
                trace.memory_hits += 1
            elif event_type == "model_call":
                trace.model_calls += 1
                usage = data.get("usage", {})
                trace.input_tokens += int(usage.get("input_tokens", 0))
                trace.output_tokens += int(usage.get("output_tokens", 0))

    async def end_turn(self, turn_id: str, rounds: int = 0) -> None:
        async with self._lock:
            trace = self._current.pop(turn_id, None)
            if trace is None:
                return
            trace.end_time = time.time()
            trace.rounds = rounds
            self._turns.append(trace)

    def get_recent_turns(self, limit: int = 20) -> list[dict]:
        """Get recent turns as dicts for the UI."""
        turns = list(self._turns)[-limit:]
        return [self._turn_to_dict(t) for t in turns]

    def get_turn(self, turn_id: str) -> dict | None:
        """Get a specific turn's trace."""
        for t in self._turns:
            if t.turn_id == turn_id:
                return self._turn_to_dict(t)
        return None

    def get_stats(self) -> dict:
        """Get aggregate stats."""
        turns = list(self._turns)
        if not turns:
            return {"total_turns": 0}
        total_tokens_in = sum(t.input_tokens for t in turns)
        total_tokens_out = sum(t.output_tokens for t in turns)
        total_tool_calls = sum(t.tool_calls for t in turns)
        total_memory_hits = sum(t.memory_hits for t in turns)
        avg_duration = sum(t.total_duration_ms() for t in turns) / len(turns)
        return {
            "total_turns": len(turns),
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_tool_calls": total_tool_calls,
            "total_memory_hits": total_memory_hits,
            "avg_duration_ms": round(avg_duration, 1),
        }

    def _turn_to_dict(self, t: TurnTrace) -> dict:
        return {
            "turn_id": t.turn_id,
            "start_time": t.start_time,
            "end_time": t.end_time,
            "duration_ms": round(t.total_duration_ms(), 1),
            "tool_calls": t.tool_calls,
            "memory_hits": t.memory_hits,
            "model_calls": t.model_calls,
            "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens,
            "model": t.model,
            "rounds": t.rounds,
            "events": [
                {
                    "timestamp": e.timestamp,
                    "type": e.type,
                    "data": e.data,
                    "duration_ms": e.duration_ms,
                }
                for e in t.events
            ],
        }


# ── Trace HTTP Server ───────────────────────────────────────────────────


class TraceServer:
    """Lightweight HTTP server for the trace UI.

    Serves a single-page app with JSON API endpoints.
    """

    def __init__(self, collector: TraceCollector, port: int = 8895) -> None:
        self.collector = collector
        self.port = port
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection,
            host="127.0.0.1",
            port=self.port,
        )
        log.info("trace_ui: serving on http://127.0.0.1:%d", self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            # Read HTTP request
            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            request_line = data.decode("utf-8", errors="replace").split("\r\n")[0]
            parts = request_line.split()
            if len(parts) < 2:
                writer.close()
                return
            _method, path = parts[0], parts[1]

            if path == "/" or path == "/index.html":
                body = self._render_html().encode("utf-8")
                self._send_response(writer, 200, "text/html", body)
            elif path == "/api/turns":
                limit = 20
                if "?" in path:
                    # Parse query string
                    pass
                turns = self.collector.get_recent_turns(limit)
                body = json.dumps({"turns": turns, "stats": self.collector.get_stats()}).encode(
                    "utf-8"
                )
                self._send_response(writer, 200, "application/json", body)
            elif path.startswith("/api/turn/"):
                turn_id = path.split("/")[-1]
                turn = self.collector.get_turn(turn_id)
                if turn:
                    body = json.dumps(turn).encode("utf-8")
                    self._send_response(writer, 200, "application/json", body)
                else:
                    self._send_response(writer, 404, "application/json", b'{"error":"not found"}')
            elif path == "/api/stats":
                body = json.dumps(self.collector.get_stats()).encode("utf-8")
                self._send_response(writer, 200, "application/json", body)
            else:
                self._send_response(writer, 404, "text/plain", b"Not found")
        except Exception as exc:
            log.debug("trace_ui.connection error: %s", exc)
        finally:
            writer.close()

    def _send_response(
        self, writer: asyncio.StreamWriter, code: int, content_type: str, body: bytes
    ) -> None:
        writer.write(
            f"HTTP/1.1 {code} OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode("utf-8") if False else None
        # Write headers
        header = (
            f"HTTP/1.1 {code} OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        writer.write(header.encode("utf-8"))
        writer.write(body)

    def _render_html(self) -> str:
        """Render the trace UI as a single-page HTML app."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Norax Trace UI</title>
<style>
body { font-family: monospace; margin: 20px; background: #1a1a2e; color: #e0e0e0; }
h1 { color: #00ff88; }
.stats { display: flex; gap: 20px; margin: 20px 0; }
.stat { background: #16213e; padding: 15px; border-radius: 8px; min-width: 120px; }
.stat-value { font-size: 24px; color: #00ff88; }
.stat-label { font-size: 12px; color: #888; }
table { width: 100%; border-collapse: collapse; margin-top: 20px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }
th { color: #00ff88; }
td { font-size: 13px; }
.turn-row { cursor: pointer; }
.turn-row:hover { background: #16213e; }
.events { margin-top: 20px; }
.event { padding: 8px; margin: 4px 0; background: #16213e; border-radius: 4px; font-size: 12px; }
.event-type { color: #00ff88; font-weight: bold; }
</style>
</head>
<body>
<h1>Norax Trace UI</h1>
<div class="stats" id="stats"></div>
<table>
<thead><tr>
<th>Turn ID</th><th>Model</th><th>Rounds</th><th>Tools</th><th>Memory</th>
<th>Tokens In</th><th>Tokens Out</th><th>Duration</th>
</tr></thead>
<tbody id="turns"></tbody>
</table>
<div class="events" id="events"></div>
<script>
async function load() {
  const resp = await fetch('/api/turns');
  const data = await resp.json();
  // Stats
  const statsEl = document.getElementById('stats');
  const s = data.stats;
  const items = [
    ['Turns', s.total_turns], ['Tokens In', s.total_tokens_in],
    ['Tokens Out', s.total_tokens_out], ['Tool Calls', s.total_tool_calls],
    ['Memory Hits', s.total_memory_hits], ['Avg ms', s.avg_duration_ms],
  ];
  statsEl.innerHTML = items.map(([l,v]) =>
    `<div class="stat"><div class="stat-value">${v||0}</div><div class="stat-label">${l}</div></div>`
  ).join('');
  // Turns table
  const tbody = document.getElementById('turns');
  tbody.innerHTML = (data.turns||[]).map(t => `
    <tr class="turn-row" onclick="loadTurn('${t.turn_id}')">
      <td>${t.turn_id.slice(0,12)}</td>
      <td>${t.model||''}</td>
      <td>${t.rounds}</td>
      <td>${t.tool_calls}</td>
      <td>${t.memory_hits}</td>
      <td>${t.input_tokens}</td>
      <td>${t.output_tokens}</td>
      <td>${t.duration_ms}ms</td>
    </tr>
  `).join('');
}
async function loadTurn(id) {
  const resp = await fetch('/api/turn/' + id);
  const t = await resp.json();
  const el = document.getElementById('events');
  el.innerHTML = '<h2>Turn ' + id.slice(0,12) + '</h2>' +
    (t.events||[]).map(e => `<div class="event"><span class="event-type">${e.type}</span> ${JSON.stringify(e.data).slice(0,200)} <span style="color:#888">${e.duration_ms}ms</span></div>`).join('');
}
load();
setInterval(load, 5000);
</script>
</body>
</html>"""


# ── Singleton ────────────────────────────────────────────────────────────

_collector: TraceCollector | None = None
_server: TraceServer | None = None


def get_trace_collector() -> TraceCollector:
    global _collector
    if _collector is None:
        _collector = TraceCollector()
    return _collector


async def start_trace_server(port: int = 8895) -> TraceServer | None:
    global _server
    if _server is not None:
        return _server
    _server = TraceServer(get_trace_collector(), port=port)
    try:
        await _server.start()
        return _server
    except Exception as exc:
        log.warning("trace_ui.start failed: %s", exc)
        _server = None
        return None
