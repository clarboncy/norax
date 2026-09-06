"""Operational capability, cache-warmup, and serving probes."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..dispatch import tools as dispatch_tools
from ._mixin import RuntimeAccessMixin
from .health import _record_operational_probe, _remote_relay_probe_result
from .validation import _explicit_result_ok

log = logging.getLogger("norax.runtime.core")


class OperationalMixin(RuntimeAccessMixin):
    _last_probe_result: dict[str, Any] | None

    async def _capability_probe_loop(self) -> None:
        """Continuously refresh operational capability evidence."""
        include_interactive = True
        while True:
            try:
                await self._probe_operational_tools(include_interactive=include_interactive)
                self.capabilities.mark_ok("operational_probe_loop")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.capabilities.mark_degraded("operational_probe_loop", str(exc)[:500])
                log.warning("operational_capability_probe.failed: %r", exc)
            # Browser launch and desktop capture are genuine but heavyweight
            # and potentially disruptive. Prove them once at startup; normal
            # tool calls are the next source of live evidence. Their registry
            # entries become explicitly stale rather than being faked fresh.
            include_interactive = False
            await asyncio.sleep(300)

    async def _probe_operational_tools(self, *, include_interactive: bool = True) -> None:
        """Verify advertised owner capabilities against the deployed host.

        These are lightweight, non-destructive startup probes. A registered
        tool with a missing browser binary, dead display, or unavailable
        execution backend must be visible in /readyz instead of failing only
        after the model tries to use it.
        """

        async def probe(name: str, operation) -> None:
            try:
                result = await operation()
                _record_operational_probe(self.capabilities, name, result)
            except Exception as exc:  # noqa: BLE001
                self.capabilities.mark_failed(name, exc)

        async def tool_manifest() -> dict[str, Any]:
            from ..brain.policy import ALL_OWNER_TOOLS

            missing = sorted(
                set(dispatch_tools.REGISTRY) - {"memory_search"} - set(ALL_OWNER_TOOLS)
            )
            stale = sorted(set(ALL_OWNER_TOOLS) - set(dispatch_tools.REGISTRY))
            return {"ok": not missing and not stale, "missing": missing, "stale": stale}

        async def browser() -> dict[str, Any]:
            from ..dispatch.browser import probe_browser_backend

            return await probe_browser_backend(timeout=15)

        async def computer() -> dict[str, Any]:
            descriptor, filename = tempfile.mkstemp(prefix="norax-display-probe-", suffix=".png")
            os.close(descriptor)
            path = Path(filename)
            try:
                return await dispatch_tools.t_computer_use(action="screenshot", path=str(path))
            finally:
                path.unlink(missing_ok=True)

        async def sandbox() -> dict[str, Any]:
            from ..dispatch.sandbox import get_sandbox_manager

            runtime = get_sandbox_manager().runtime
            if not runtime:
                return {"ok": False, "error": "no docker or podman executable found"}
            proc = await asyncio.create_subprocess_exec(
                runtime,
                "info",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=8.0)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return {"ok": False, "error": "docker info timed out"}
            return {
                "ok": proc.returncode == 0,
                "runtime": runtime,
                "response_observed": bool(out.strip()),
                "error": err.decode(errors="replace")[:300],
            }

        async def web_search() -> dict[str, Any]:
            import httpx

            url = os.environ.get("NORAX_SEARXNG_URL", "http://127.0.0.1:8889").rstrip("/")
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"{url}/healthz")
                if 200 <= response.status_code < 300:
                    return {
                        "ok": True,
                        "backend": "searxng",
                        "status_code": response.status_code,
                    }
            except Exception:  # noqa: BLE001
                pass
            configured = bool(
                os.environ.get("NORAX_SERPER_API_KEY") or os.environ.get("NORAX_TAVILY_API_KEY")
            )
            return {
                "ok": configured,
                "backend": "api" if configured else "none",
                "degraded_reason": (
                    "local search is unavailable; configured API fallback was not live-probed"
                    if configured
                    else ""
                ),
            }

        async def remote_relay() -> dict[str, Any]:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://127.0.0.1:8765/health")
                response.raise_for_status()
                payload = response.json()
            return _remote_relay_probe_result(payload)

        operations = [
            ("tool_manifest", tool_manifest),
            ("sandbox_tool", sandbox),
            ("web_search_tool", web_search),
            ("remote_relay", remote_relay),
        ]
        if include_interactive:
            operations.extend(
                [
                    ("browser_tool", browser),
                    ("computer_tool", computer),
                ]
            )
        await asyncio.gather(*(probe(name, operation) for name, operation in operations))

    async def _warmup_caches(self) -> None:
        """Pre-load embedding caches from disk so first query is fast.

        Loads .npz caches for both LocalRetriever (canonical) and
        ExternalRetriever (sleep+intel) in parallel. Only embeds the delta
        (new/changed neurons since cache was written). If no cache exists,
        triggers a full build so subsequent queries avoid the cold-build cost.
        """
        try:
            tasks: list[Any] = []
            if self._hybrid is not None and self._hybrid.embedding is not None:
                # Pre-load the embed model into GPU memory so the first real
                # embed doesn't time out during cold load.
                embedder = self._hybrid.embedding.embedder
                if hasattr(embedder, "warmup"):
                    tasks.append(embedder.warmup())
                tasks.append(self._hybrid.ensure_index())
            if self._context_injector is not None and self._context_injector.external is not None:
                tasks.append(self._context_injector.external.refresh_if_stale())
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                failures = [result for result in results if isinstance(result, BaseException)]
                if failures:
                    log.warning(
                        "warmup_caches.partial failures=%s",
                        [type(failure).__name__ for failure in failures],
                    )
                else:
                    log.info("embedding caches warmed up")
        except Exception as e:
            log.warning("warmup_caches.failed: %r (first query will build lazily)", e)

    async def _run_completion_probe(self) -> dict[str, Any]:
        """Probe the primary→failover serving chain with explicit evidence.

        The default transport mode performs an authenticated model-catalog
        request and generates no tokens.  Operators who need a synthetic
        inference can opt into ``NORAX_COMPLETION_PROBE_MODE=completion``.
        Results never conflate transport reachability with a completed model
        response.
        """
        from ..gateway_client import GatewayRequest

        primary = getattr(self, "_effective_model", None)
        probe_mode = getattr(self, "_completion_probe_mode", "transport")
        candidates = list(
            dict.fromkeys(
                [model for model in [primary, *getattr(self, "failover_models", [])] if model]
            )
        )
        failures: list[str] = []
        probed_routes: set[tuple[str, str]] = set()
        for model in candidates:
            t0 = time.monotonic()
            try:
                provider = "?"
                route_url = "?"
                route_for = getattr(self.gateway, "route_for", None)
                if callable(route_for):
                    provider, route_url = route_for(model)

                if probe_mode == "transport":
                    route_key = (str(provider), str(route_url))
                    if route_key in probed_routes:
                        continue
                    probed_routes.add(route_key)
                    transport_probe = getattr(self.gateway, "transport_probe", None)
                    if not callable(transport_probe):
                        raise RuntimeError("gateway does not implement a transport probe")
                    evidence = await transport_probe(model, timeout=5.0)
                    latency_ms = (time.monotonic() - t0) * 1000
                    if not _explicit_result_ok(evidence):
                        failures.append(f"{model}: {str(evidence)[:240]}")
                        continue
                    return {
                        "ok": True,
                        "enabled": True,
                        "mode": "transport",
                        "transport_verified": True,
                        "completion_verified": False,
                        "model": model,
                        "provider": evidence.get("provider", provider),
                        "fallback": model != primary,
                        "latency_ms": round(latency_ms, 1),
                        "evidence": {
                            key: value
                            for key, value in evidence.items()
                            if key in {"endpoint", "kind", "status_code"}
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                    }

                req = GatewayRequest(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Reply with exactly OK and nothing else.",
                        }
                    ],
                    # Disable private reasoning for this binary liveness task.
                    # A 32-token ceiling accommodates provider wrappers while
                    # the expected visible completion remains only two tokens.
                    max_tokens=32,
                    temperature=0.0,
                    metadata={"reasoning_effort": "none", "synthetic_probe": True},
                )
                resp = await self.gateway.chat(req)
                latency_ms = (time.monotonic() - t0) * 1000
                observed = resp.content.strip()
                if observed != "OK":
                    digest = hashlib.sha256(observed.encode("utf-8")).hexdigest()[:12]
                    failures.append(
                        f"{model}: unexpected response chars={len(observed)} sha256={digest}"
                    )
                    continue
                return {
                    "ok": True,
                    "enabled": True,
                    "mode": "completion",
                    "transport_verified": True,
                    "completion_verified": True,
                    "model": model,
                    "provider": provider,
                    "fallback": model != primary,
                    "latency_ms": round(latency_ms, 1),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{model}: {type(exc).__name__}: {exc}"[:300])
        return {
            "ok": False,
            "enabled": True,
            "mode": probe_mode,
            "transport_verified": False,
            "completion_verified": False,
            "model": primary or "?",
            "failures": failures,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _completion_probe_defer_reason(self, *, now: float | None = None) -> str | None:
        """Return why a synthetic inference should yield to user traffic."""
        if getattr(self, "_completion_probe_mode", "transport") != "completion":
            return None
        if getattr(self, "_turn_work_count", 0) > 0:
            return "queued_or_active_turn"
        active = getattr(self, "_active_turn_tasks", {})
        if any(not task.done() for task in active.values()):
            return "active_turn"
        last_user_turn = getattr(self, "_last_user_turn_time", None)
        if last_user_turn is None:
            return None
        idle_seconds = (time.time() if now is None else now) - last_user_turn
        if idle_seconds < getattr(self, "_completion_probe_idle_seconds", 30):
            return "recent_turn"
        return None

    @staticmethod
    def _completion_probe_age(result: object, *, now: datetime | None = None) -> float:
        if not isinstance(result, dict):
            return float("inf")
        try:
            timestamp = datetime.fromisoformat(str(result["timestamp"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            return float("inf")
        current = now or datetime.now(UTC)
        try:
            return max(0.0, (current - timestamp).total_seconds())
        except TypeError:
            return float("inf")

    async def _completion_probe_loop(self) -> None:
        """Periodically verify serving without needlessly competing with turns."""
        probe_interval = getattr(self, "_completion_probe_interval_seconds", 600)
        busy_retry = 15
        max_deferral = 840
        while True:
            try:
                defer_reason = self._completion_probe_defer_reason()
                probe_age = self._completion_probe_age(getattr(self, "_last_probe_result", None))
                if defer_reason is not None and probe_age < max_deferral:
                    log.debug(
                        "completion_probe.deferred reason=%s evidence_age=%.1fs",
                        defer_reason,
                        probe_age,
                    )
                    await asyncio.sleep(min(busy_retry, max(1.0, max_deferral - probe_age)))
                    continue
                self._last_probe_result = await self._run_completion_probe()
                if self._last_probe_result.get("ok") is True:
                    self.capabilities.mark_ok("serving_probe_loop")
                    log.debug(
                        "completion_probe.ok model=%s latency=%sms fallback=%s",
                        self._last_probe_result["model"],
                        self._last_probe_result["latency_ms"],
                        self._last_probe_result["fallback"],
                    )
                else:
                    failures = "; ".join(self._last_probe_result.get("failures", []))
                    self.capabilities.mark_failed(
                        "serving_probe_loop",
                        RuntimeError(failures or "serving probe failed"),
                    )
                    log.warning(
                        "completion_probe.failed failures=%s",
                        self._last_probe_result.get("failures", []),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._last_probe_result = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                self.capabilities.mark_failed("serving_probe_loop", exc)
                log.warning("completion_probe.failed error=%r", exc)
            await asyncio.sleep(probe_interval)
